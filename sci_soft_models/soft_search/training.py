#!/usr/bin/env python

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from ..utils import find_device
from .constants import MODEL_STR_INPUT_TEMPLATE, TRAINED_UPLOADED_MODEL_NAME
from .data import FINAL_MODEL_TRAINING_DATA_DIR, load_soft_search_2025_training_dataset

###############################################################################

# Fine-tune default settings
_CURRENT_DIR = Path(__file__).parent
TRAINED_MODEL_DIR = _CURRENT_DIR / "nsf-soft-search-v2"

###############################################################################


def _ft_pred_test(
    hf_model_path: str,
    trained_model_name: str,
    features: datasets.Features,
    num_classes: int,
    label2id: dict[str, str],
    id2label: dict[str, str],
    epoch_val: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hf_token: str,
) -> pd.DataFrame:
    # Set seed
    np.random.seed(12)
    random.seed(12)

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
        output_dir=str(TRAINED_MODEL_DIR),
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
    trainer.save_model(str(TRAINED_MODEL_DIR))

    # Upload model to Hugging Face
    print("Uploading model to Hugging Face...")
    trainer.push_to_hub(
        trained_model_name,
        token=hf_token,
    )

    # Find device
    device = find_device()

    # Load pipeline
    ft_transformer_pipe = pipeline(
        task="text-classification",
        model=str(TRAINED_MODEL_DIR),
        tokenizer=str(TRAINED_MODEL_DIR),
        padding=True,
        truncation=True,
        device=device,
        trust_remote_code=True,
    )

    # Make predictions
    y_pred = [pred["label"] for pred in ft_transformer_pipe(test_df["text"].tolist())]
    test_df["predicted_label"] = y_pred

    return test_df


def run(
    base_model_name: str = "answerdotai/ModernBERT-base",
    num_training_epochs: int = 2,
    test_size: float = 0.2,
    trained_model_name: str = TRAINED_UPLOADED_MODEL_NAME,
    model_eval_outputs_dir: Path = FINAL_MODEL_TRAINING_DATA_DIR,
    confusion_matrix_save_name: str = "nsf-soft-search-v2-confusion-matrix.png",
    misclassifications_save_name: str = "nsf-soft-search-v2-misclassifications.csv",
    use_coiled: bool = False,
    coiled_vm_type: str = "g5.xlarge",
    coiled_keepalive: str = "3 minutes",
    **kwargs: Any,
) -> None:
    # Delete prior results and then remake
    shutil.rmtree(str(TRAINED_MODEL_DIR), ignore_errors=True)
    shutil.rmtree(model_eval_outputs_dir, ignore_errors=True)
    model_eval_outputs_dir.mkdir(exist_ok=True, parents=True)

    # Set seed
    np.random.seed(12)
    random.seed(12)

    # Load env
    load_dotenv()

    # Check that HF_AUTH_TOKEN is set
    if "HF_AUTH_TOKEN" not in os.environ:
        raise OSError("HF_AUTH_TOKEN is not set in the environment")

    # Get HF token
    hf_token = os.environ["HF_AUTH_TOKEN"]

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
        test_size=test_size,
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
        print(f"Keepalive: {coiled_keepalive}")
        print()

        # Wrap function
        @coiled.function(
            name="ft-pred-nsf-soft-search-v2",
            vm_type=coiled_vm_type,
            idle_timeout=coiled_keepalive,
            disk_size=48,
            tags=tags,
            spot_policy="on-demand",
        )
        def _ft_pred_test_coiled(
            hf_model_path: str,
            trained_model_name: str,
            features: datasets.Features,
            num_classes: int,
            label2id: dict[str, str],
            id2label: dict[str, str],
            epoch_val: int,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            hf_token: str,
        ) -> pd.DataFrame:
            return _ft_pred_test(
                hf_model_path=hf_model_path,
                trained_model_name=trained_model_name,
                features=features,
                num_classes=num_classes,
                label2id=label2id,
                id2label=id2label,
                epoch_val=epoch_val,
                train_df=train_df,
                test_df=test_df,
                hf_token=hf_token,
            )

        selected_ft_pred_func = _ft_pred_test_coiled

    # Run locally
    else:
        selected_ft_pred_func = _ft_pred_test

    # Run fine-tune and prediction
    predicted_values_after_ft = selected_ft_pred_func(
        hf_model_path=base_model_name,
        trained_model_name=trained_model_name,
        features=features,
        num_classes=num_classes,
        label2id=label2id,
        id2label=id2label,
        epoch_val=num_training_epochs,
        train_df=train_df,
        test_df=test_df,
        hf_token=hf_token,
    )

    accuracy = accuracy_score(
        predicted_values_after_ft["label"].tolist(),
        predicted_values_after_ft["predicted_label"].tolist(),
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        predicted_values_after_ft["label"].tolist(),
        predicted_values_after_ft["predicted_label"].tolist(),
        average="macro",
    )

    # Print results
    print(
        f"Evaluation results -- "
        f"Accuracy: {accuracy:.3f}, "
        f"Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, "
        f"F1: {f1:.3f}"
    )

    # Store results to JSON
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    results_save_path = model_eval_outputs_dir / "overall-results.json"
    with open(results_save_path, "w") as open_f:
        json.dump(results, open_f)

    # Iter over reduced_directorate and get accuracy, precision, recall, f1
    directorate_results = []
    for reduced_directorate in predicted_values_after_ft[
        "reduced_directorate"
    ].unique():
        reduced_directorate_df = predicted_values_after_ft[
            predicted_values_after_ft["reduced_directorate"] == reduced_directorate
        ]
        dir_accuracy = accuracy_score(
            reduced_directorate_df["label"].tolist(),
            reduced_directorate_df["predicted_label"].tolist(),
        )
        dir_precision, dir_recall, dir_f1, _ = precision_recall_fscore_support(
            reduced_directorate_df["label"].tolist(),
            reduced_directorate_df["predicted_label"].tolist(),
            average="macro",
        )
        directorate_results.append(
            {
                "reduced_directorate": reduced_directorate,
                "accuracy": dir_accuracy,
                "precision": dir_precision,
                "recall": dir_recall,
                "f1": dir_f1,
                "count": len(reduced_directorate_df),
            }
        )

    # Store directorate results to CSV
    directorate_results_df = pd.DataFrame(directorate_results).sort_values(
        by="f1", ascending=False
    )
    directorate_results_save_path = model_eval_outputs_dir / "directorate-results.csv"
    directorate_results_df.to_csv(directorate_results_save_path, index=False)
    print("Directorate results:")
    print(directorate_results_df)
    print()

    # Create confusion matrix and ROC curve
    confusion_matrix = ConfusionMatrixDisplay.from_predictions(
        predicted_values_after_ft["label"].tolist(),
        predicted_values_after_ft["predicted_label"].tolist(),
    )
    confusion_matrix_save_path = model_eval_outputs_dir / confusion_matrix_save_name
    confusion_matrix.figure_.savefig(confusion_matrix_save_path)

    # Store misclassifications
    misclassifications = predicted_values_after_ft.loc[
        predicted_values_after_ft["label"]
        != predicted_values_after_ft["predicted_label"]
    ]
    misclassifications_save_path = model_eval_outputs_dir / misclassifications_save_name
    misclassifications.to_csv(misclassifications_save_path, index=False)

    print("Done!")
