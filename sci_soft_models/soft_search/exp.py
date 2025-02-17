#!/usr/bin/env python

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin
from distributed import as_completed
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Pipeline,
    Trainer,
    TrainingArguments,
    pipeline,
)

from ..utils import find_device
from .constants import MODEL_STR_INPUT_TEMPLATE
from .data import EXP_FILES_DIR, load_soft_search_2025_training_dataset

###############################################################################

# Models used for testing, both fine-tune and semantic logit
BASE_MODELS = {
    "bert": "google-bert/bert-base-uncased",
    "deberta": "microsoft/deberta-v3-base",
    "modern-bert": "answerdotai/ModernBERT-base",
}

# Fine-tune default settings
_CURRENT_DIR = Path(__file__).parent
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
EPOCH_VALUES = [1, 2, 3, 4]

# Evaluation storage path
EVAL_STORAGE_PATH = _CURRENT_DIR / "exp-model-eval-results"
TRAINING_RESULTS_STORAGE_PATH = EXP_FILES_DIR / "exp-training-results.csv"

###############################################################################


@dataclass
class EvaluationResults(DataClassJsonMixin):
    model: str
    epoch_val: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate(
    model: Pipeline,
    test_df: pd.DataFrame,
    model_name: str,
    epoch_val: int,
    eval_storage_path: Path,
) -> EvaluationResults:
    # Evaluate the model
    print("Evaluating model")

    # Unpack test set
    x_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()

    # Make prediction
    y_pred = model.predict(x_test)

    # Get the actual predictions from Pipeline
    if isinstance(model, Pipeline):
        y_pred = [pred["label"] for pred in y_pred]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
    )

    # Print results
    print(
        f"Accuracy: {accuracy}, "
        f"Precision: {precision}, "
        f"Recall: {recall}, "
        f"F1: {f1}, "
    )

    # Model short name
    this_model_eval_storage = eval_storage_path / model_name
    this_model_eval_storage.mkdir(exist_ok=True, parents=True)

    # Epoch value
    this_model_eval_storage = this_model_eval_storage / f"epochs-{epoch_val}"
    this_model_eval_storage.mkdir(exist_ok=True, parents=True)

    # Create confusion matrix display
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
    )

    # Save confusion matrix
    cm.figure_.savefig(this_model_eval_storage / "confusion.png")

    # Add a "predicted" column
    test_df["predicted"] = y_pred

    # Find rows of misclassifications
    misclassifications = test_df[test_df["label"] != test_df["predicted"]]

    # Save misclassifications
    misclassifications.to_csv(
        this_model_eval_storage / "misclassifications.csv",
        index=False,
    )

    return EvaluationResults(
        model=model_name,
        epoch_val=epoch_val,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _ft_eval(
    model_short_name: str,
    hf_model_path: str,
    features: datasets.Features,
    num_classes: int,
    label2id: dict[str, str],
    id2label: dict[str, str],
    epoch_val: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> EvaluationResults:
    # Set seed
    np.random.seed(12)
    random.seed(12)

    print(f"Fine-tuning model: {model_short_name}")
    # Delete existing temp storage if exists
    if DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH.exists():
        shutil.rmtree(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    def tokenize_function(
        examples: dict[str, list[str]],
        tokenizer: AutoTokenizer = tokenizer,
    ) -> dict[str, list[int]]:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Convert to datasets
    train_dataset = datasets.Dataset.from_pandas(
        train_df.copy(),
        features=features,
        preserve_index=False,
    )
    test_dataset = datasets.Dataset.from_pandas(
        test_df.copy(),
        features=features,
        preserve_index=False,
    )

    # Store to dataset dict
    ds_dict = datasets.DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    tokenized_ds_dict = ds_dict.map(tokenize_function, batched=True)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_path,
        num_labels=num_classes,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    # Create Training Args and Trainer
    training_args = TrainingArguments(
        output_dir=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
        overwrite_output_dir=True,
        num_train_epochs=epoch_val,
        learning_rate=1e-5,
        logging_steps=10,
        auto_find_batch_size=True,
        seed=12,
        save_strategy="no",
    )
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_ds_dict["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Training model...")

    # Train
    trainer.train()
    trainer.save_model(str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH))

    # Find device
    device = find_device()

    # Evaluate the model
    ft_transformer_pipe = pipeline(
        task="text-classification",
        model=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
        tokenizer=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
        padding=True,
        truncation=True,
        device=device,
        trust_remote_code=True,
    )

    return evaluate(
        model=ft_transformer_pipe,
        test_df=test_df.copy(),
        model_name=model_short_name,
        epoch_val=epoch_val,
        eval_storage_path=EVAL_STORAGE_PATH,
    ).to_dict()


def run(  # noqa: C901
    results_output_path: Path = TRAINING_RESULTS_STORAGE_PATH,
    use_coiled: bool = False,
    coiled_vm_type: str = "g5.xlarge",
    coiled_min_workers: int = 1,
    coiled_max_workers: int = 4,
    coiled_keepalive: str = "3 minutes",
) -> None:
    print("Starting experimental run for SoftSearch...")

    # Delete prior results and then remake
    shutil.rmtree(EVAL_STORAGE_PATH, ignore_errors=True)
    EVAL_STORAGE_PATH.mkdir(exist_ok=True)

    # Delete prior results and then remake
    shutil.rmtree(EXP_FILES_DIR, ignore_errors=True)
    EXP_FILES_DIR.mkdir(exist_ok=True)

    # Set seed
    np.random.seed(12)
    random.seed(12)

    ###############################################################################

    # Load data
    full_set = load_soft_search_2025_training_dataset()

    # Rename column from "software_produced" to "label"
    full_set = full_set.rename(columns={"software_produced": "label"})

    # Map values in label column from True/False
    # to "software-produced"/"software-not-produced"
    full_set["label"] = full_set["label"].apply(
        lambda x: "software-produced" if x else "software-not-produced"
    )

    # Create the "text" columns
    full_set["text"] = full_set.apply(
        lambda x: MODEL_STR_INPUT_TEMPLATE.format(
            award_title=x["title"],
            award_abstract=x["abstractText"],
            award_outcomes=x["projectOutComesReport"],
        ),
        axis=1,
    )

    # Subset to only include the "grant_id", "text", and "label" columns
    full_set = full_set[
        [
            "grant_id",
            "directorate",
            "reduced_directorate",
            "text",
            "label",
            "stratify_group",
        ]
    ]

    # Store class details required for feature construction
    num_classes = full_set["label"].nunique()
    class_labels = list(full_set["label"].unique())

    # Construct label to id and vice-versa LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(class_labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Construct features for the dataset
    features = datasets.Features(
        grant_id=datasets.Value("string"),
        directorate=datasets.Value("string"),
        reduced_directorate=datasets.Value("string"),
        text=datasets.Value("string"),
        label=datasets.ClassLabel(
            num_classes=num_classes,
            names=class_labels,
        ),
        stratify_group=datasets.Value("string"),
    )

    # Split once
    train_df, test_df = train_test_split(
        full_set,
        test_size=0.2,
        random_state=12,
        stratify=full_set["stratify_group"],
    )

    # Create a dataframe where the rows are the different splits
    # and there are three columns one column is the split name,
    # the other columns are the counts of match
    split_counts = []
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df),
    ]:
        split_counts.append(
            {
                "split": split_name,
                **split_df["label"].value_counts().to_dict(),
                **{
                    f"{k}%": v
                    for k, v in split_df["label"]
                    .value_counts(normalize=True)
                    .to_dict()
                    .items()
                },
            }
        )
    split_counts_df = pd.DataFrame(split_counts)
    print("Split counts:")
    print(split_counts_df)
    print()

    # Print example input
    print("Example input:")
    print("-" * 20)
    print()
    print(train_df.sample(1).iloc[0].text)
    print()
    print("-" * 20)
    print()

    # Handle coiled
    if use_coiled:
        import coiled

        # Load env
        load_dotenv()

        # Get AWS_APPLICATION_TAG_KEY_VALUE variable
        tags = {}
        if "AWS_APPLICATION_TAG_KEY_VALUE" in os.environ:
            key_and_value = os.environ["AWS_APPLICATION_TAG_KEY_VALUE"]
            key, value = key_and_value.split("=")
            tags[key] = value
        else:
            print("WARNING: AWS_APPLICATION_TAG_KEY_VALUE not found in env")

        # Check that AWS_PROFILE is set
        if "AWS_PROFILE" in os.environ:
            aws_profile = os.environ["AWS_PROFILE"]
            print(f"Using AWS_PROFILE: {aws_profile}")
        else:
            print("WARNING: AWS_PROFILE not found in env, using default")

        # Print coiled settings
        print("Coiled settings:")
        print(f"VM type: {coiled_vm_type}")
        print(f"Min workers: {coiled_min_workers}")
        print(f"Max workers: {coiled_max_workers}")
        print(f"Keepalive: {coiled_keepalive}")
        print()

        # Get coiled cluster
        with coiled.Cluster(
            name="fine-tune-eval-soft-search-exp",
            scheduler_vm_types=["g4dn.xlarge"],
            worker_vm_types=[coiled_vm_type],
            n_workers=[coiled_min_workers, coiled_max_workers],
            worker_disk_size=48,
            worker_options={"nthreads": 1},
            idle_timeout=coiled_keepalive,
            tags=tags,
            spot_policy="on-demand",
        ) as cluster:
            # Get client
            client = cluster.get_client()

            # Iter over epochs and models
            futures = []
            for epoch_val in EPOCH_VALUES:
                for model_short_name, hf_model_path in BASE_MODELS.items():
                    futures.append(
                        client.submit(
                            _ft_eval,
                            key=f"ft-eval-{model_short_name}-{epoch_val}",
                            model_short_name=model_short_name,
                            hf_model_path=hf_model_path,
                            features=features,
                            num_classes=num_classes,
                            label2id=label2id,
                            id2label=id2label,
                            epoch_val=epoch_val,
                            train_df=train_df,
                            test_df=test_df,
                        )
                    )

            # Get results
            results = []
            for _, result in tqdm(
                as_completed(futures, with_results=True),
                desc="Fine-tuning and evaluation futures",
                leave=False,
                total=len(BASE_MODELS) * len(EPOCH_VALUES),
            ):
                results.append(result)

                # Print results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(
                    by="f1", ascending=False
                ).reset_index(drop=True)
                results_df.to_csv(results_output_path, index=False)
                print("Current standings")
                print(
                    tabulate(
                        results_df.head(10),
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                    )
                )

                print()

    # Otherwise use normal function
    else:
        results = []
        for epoch_val in tqdm(
            EPOCH_VALUES,
            desc="Multiple Epochs Testing",
            leave=False,
        ):
            # Fine-tune from each base
            for model_short_name, hf_model_path in tqdm(
                BASE_MODELS.items(),
                desc="Fine-tune models",
                leave=False,
            ):
                eval_res = _ft_eval(
                    model_short_name=model_short_name,
                    hf_model_path=hf_model_path,
                    features=features,
                    num_classes=num_classes,
                    label2id=label2id,
                    id2label=id2label,
                    epoch_val=epoch_val,
                    train_df=train_df,
                    test_df=test_df,
                )

                # Append to results
                results.append(eval_res)

                # Print results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(
                    by="f1", ascending=False
                ).reset_index(drop=True)
                results_df.to_csv(results_output_path, index=False)
                print("Current standings")
                print(
                    tabulate(
                        results_df.head(10),
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                    )
                )

                print()

    print()
    print("-" * 80)
    print()

    # Print results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(results_output_path, index=False)
    print("Final standings")
    print(
        tabulate(
            results_df.head(10),
            headers="keys",
            tablefmt="psql",
            showindex=False,
        )
    )

    # Save results
    results_df.to_csv(results_output_path, index=False)
