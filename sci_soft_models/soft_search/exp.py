#!/usr/bin/env python

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from autotrain.trainers.text_classification.__main__ import train as ft_train
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm
from transformers import Pipeline, pipeline

from ..utils import find_device
from .constants import MODEL_STR_INPUT_TEMPLATE
from .data import EXP_FILES_DIR, load_soft_search_2025_training_dataset

###############################################################################

# Models used for testing, both fine-tune and semantic logit
BASE_MODELS = {
    "bert": "google-bert/bert-base-uncased",
    "deberta": "microsoft/deberta-v3-base",
    "modern-bert": "answerdotai/ModernBERT-base",
    "nomic-bert": "nomic-ai/nomic-bert-2048",
    "gte-mlm-base": "Alibaba-NLP/gte-en-mlm-base",
}

# Fine-tune default settings
DEFAULT_HF_DATASET_PATH = "evamxb/soft-search-2025-training-dataset"
_CURRENT_DIR = Path(__file__).parent
DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH = Path("autotrain-text-classification-temp/")
DEFAULT_MODEL_MAX_SEQ_LENGTH = 2048
# EPOCH_VALUES = [1, 2, 3, 4, 5]
EPOCH_VALUES = [1]
FINE_TUNE_COMMAND_DICT = {
    "data_path": DEFAULT_HF_DATASET_PATH,
    "project_name": str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
    "text_column": "text",
    "target_column": "label",
    "train_split": "train",
    "lr": 1e-5,
    "auto_find_batch_size": True,
    "seed": 12,
    "max_seq_length": DEFAULT_MODEL_MAX_SEQ_LENGTH,
    "logging_steps": 10,
}

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
    this_model_eval_storage.mkdir(exist_ok=True)

    # Epoch value
    this_model_eval_storage = this_model_eval_storage / f"epochs-{epoch_val}"
    this_model_eval_storage.mkdir(exist_ok=True)

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


def run(
    results_output_path: Path = TRAINING_RESULTS_STORAGE_PATH,
) -> None:
    # Load environment variables
    load_dotenv()
    FINE_TUNE_COMMAND_DICT["token"] = os.environ["HF_AUTH_TOKEN"]

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

    full_set = full_set.sample(500)

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

    # Convert to datasets
    train_dataset = datasets.Dataset.from_pandas(
        train_df,
        features=features,
        preserve_index=False,
    )
    test_dataset = datasets.Dataset.from_pandas(
        test_df,
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

    # Push to hub
    print("Pushing dataset to hub")
    ds_dict.push_to_hub(
        DEFAULT_HF_DATASET_PATH,
        private=True,
        token=os.environ["HF_AUTH_TOKEN"],
    )
    print()
    print()

    results = []
    # Iter through epochs
    for epoch_val in tqdm(
        EPOCH_VALUES,
        desc="Multiple Epochs Testing",
        leave=False,
    ):
        # Set seed
        np.random.seed(12)
        random.seed(12)

        # Fine-tune from each base
        for model_short_name, hf_model_path in tqdm(
            BASE_MODELS.items(),
            desc="Fine-tune models",
            leave=False,
        ):
            # Delete existing temp storage if exists
            if DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH.exists():
                shutil.rmtree(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH)

            # Update the fine-tune command dict
            this_iter_command_dict = FINE_TUNE_COMMAND_DICT.copy()
            this_iter_command_dict["model"] = hf_model_path
            this_iter_command_dict["epochs"] = epoch_val

            # Train the model
            ft_train(
                this_iter_command_dict,
            )

            # Find device
            device = find_device()

            # Evaluate the model
            ft_transformer_pipe = pipeline(
                task="text-classification",
                model=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                tokenizer=str(DEFAULT_FINE_TUNE_TEMP_STORAGE_PATH),
                padding=True,
                truncation=True,
                max_length=DEFAULT_MODEL_MAX_SEQ_LENGTH,
                device=device,
            )

            results.append(
                evaluate(
                    model=ft_transformer_pipe,
                    test_df=test_df.copy(),
                    model_name=model_short_name,
                    epoch_val=epoch_val,
                    eval_storage_path=EVAL_STORAGE_PATH,
                ).to_dict(),
            )

        print()

        # Print results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="f1", ascending=False).reset_index(
            drop=True
        )
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
