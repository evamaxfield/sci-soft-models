#!/usr/bin/env python

import os
import random
import datasets
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from dotenv import load_dotenv

from .data import (
    load_annotated_dev_author_em_dataset,
    load_author_contributors_dataset,
    load_developer_contributors_dataset,
)

import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

from .constants import MODEL_STR_INPUT_TEMPLATE, TRAINED_UPLOADED_MODEL_NAME

###############################################################################

_CURRENT_DIR = Path(__file__).parent
DEFAULT_MODEL_EVAL_OUTPUTS_DIR = _CURRENT_DIR / "official-model-eval-results"

###############################################################################


def run(
    base_model_name: str = "microsoft/deberta-v3-base",
    test_size: float = 0.1,
    model_name: str = TRAINED_UPLOADED_MODEL_NAME,
    model_eval_outputs_dir: Path = DEFAULT_MODEL_EVAL_OUTPUTS_DIR,
    confusion_matrix_save_name: str = "dev-author-em-confusion-matrix.png",
    misclassifications_save_name: str = "dev-author-em-misclassifications.csv",
) -> None:
    # Load env
    load_dotenv()

    # Check that HF_AUTH_TOKEN is set
    if "HF_AUTH_TOKEN" not in os.environ:
        raise EnvironmentError("HF_AUTH_TOKEN is not set in the environment")

    # Set seed
    random.seed(12)
    np.random.seed(12)

    # Load the datasets
    dev_author_full_details = load_annotated_dev_author_em_dataset()

    # Cast semantic_scholar_id to string
    dev_author_full_details["semantic_scholar_id"] = dev_author_full_details[
        "semantic_scholar_id"
    ].astype(str)

    # Load the authors dataset
    authors = load_author_contributors_dataset()

    # Drop everything but the author_id and the name
    authors = authors[["author_id", "name"]].dropna()

    # Load the repos dataset to get to devs
    devs = load_developer_contributors_dataset()

    # Get unique devs by grouping by username
    # and then taking the first email and first "name"
    devs = devs.groupby("username").first().reset_index()

    # For each row in terra, get the matching author and dev
    matched_details = []
    for _, row in dev_author_full_details.iterrows():
        try:
            # Get the author details
            author_details = authors[
                authors["author_id"] == row["semantic_scholar_id"]
            ].iloc[0]

            # Get the dev details
            dev_details = devs[devs["username"] == row["github_id"]].iloc[0]

            # Add to the dataset
            matched_details.append(
                {
                    "dev_username": row["github_id"],
                    "dev_name": dev_details["name"],
                    "dev_email": dev_details["email"],
                    "author_id": row["semantic_scholar_id"],
                    "author_name": author_details["name"],
                    "label": "match" if row["match"] else "no-match",
                }
            )
        except Exception:
            pass

    # Convert to dataframe
    matched_details_df = pd.DataFrame(matched_details)

    # Create splits
    # Create test set holdout by selecting 10% random unique devs and authors
    # and then adding all of their comparisons to the test set
    unique_devs = pd.Series(matched_details_df["dev_username"].unique())
    unique_authors = pd.Series(matched_details_df["author_id"].unique())
    test_devs = unique_devs.sample(
        frac=test_size,
        random_state=12,
        replace=False,
    )
    test_authors = unique_authors.sample(
        frac=test_size,
        random_state=12,
        replace=False,
    )
    train_rows = []
    test_rows = []
    for _, row in matched_details_df.iterrows():
        # Clean input strings by removing whitespace
        cleaned_row = row.copy().str.strip()

        # Format the data to training-ready strings
        model_input_str = MODEL_STR_INPUT_TEMPLATE.format(
            dev_username=cleaned_row["dev_username"],
            dev_name=cleaned_row["dev_name"],
            dev_email=cleaned_row["dev_email"],
            author_name=cleaned_row["author_name"],
        )

        if (
            row["dev_username"] in test_devs.values
            or row["author_id"] in test_authors.values
        ):
            test_rows.append(
                {
                    "text": model_input_str,
                    "label": row["label"],
                }
            )
        else:
            train_rows.append(
                {
                    "text": model_input_str,
                    "label": row["label"],
                }
            )

    # Create train and test sets
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    # Print input example
    print("Example input:")
    print(train_df.iloc[0]["text"])
    print()
    print()

    # Log split counts
    split_counts = []
    for split_name, split_df in [
        ("train", train_df),
        ("test", test_df),
    ]:
        split_counts.append(
            {
                "split": split_name,
                **split_df["label"].value_counts().to_dict(),
            }
        )
    split_counts_df = pd.DataFrame(split_counts)
    print("Split counts:")
    print(split_counts_df)
    print()

    # Get n classes and labels
    num_classes = matched_details_df["label"].nunique()
    class_labels = matched_details_df["label"].unique().tolist()

    # Construct features for the dataset
    features = datasets.Features(
        text=datasets.Value("string"),
        label=datasets.ClassLabel(
            num_classes=num_classes,
            names=class_labels,
        ),
    )

    # Create dataset dict
    ds_dict = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_pandas(
                train_df,
                features=features,
                preserve_index=False,
            ),
            "test": datasets.Dataset.from_pandas(
                test_df,
                features=features,
                preserve_index=False,
            ),
        }
    )

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_ds_dict = ds_dict.map(tokenize_function, batched=True)

    # Construct label to id and vice-versa LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(class_labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Create Training Args and Trainer
    training_args = TrainingArguments(
        output_dir="dev-author-em-clf",
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=1e-5,
        logging_steps=10,
        auto_find_batch_size=True,
        seed=12,
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
    trainer.save_model("dev-author-em-clf")

    # Load model from dir as pipeline
    trained_clf = pipeline(
        "text-classification",
        model="dev-author-em-clf/",
    )

    # Evaluate
    print("Evaluating model...")
    y_pred = [pred["label"] for pred in trained_clf(test_df["text"].tolist())]
    accuracy = accuracy_score(test_df["label"].tolist(), y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"].tolist(),
        y_pred,
        average="binary",
        pos_label="match",
    )

    # Print results
    print(
        f"Evaluation results -- "
        f"Accuracy: {accuracy:.3f}, "
        f"Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, "
        f"F1: {f1:.3f}"
    )

    # Make the outputs dir
    model_eval_outputs_dir.mkdir(exist_ok=True, parents=True)

    # Create confusion matrix and ROC curve
    confusion_matrix = ConfusionMatrixDisplay.from_predictions(
        test_df["label"].tolist(),
        y_pred,
    )
    confusion_matrix_save_path = model_eval_outputs_dir / confusion_matrix_save_name
    confusion_matrix.figure_.savefig(confusion_matrix_save_path)

    # Add predicted values to test_df to find misclassifications
    test_df["predicted"] = y_pred
    misclassifications = test_df.loc[test_df["label"] != test_df["predicted"]]

    # Store misclassifications
    misclassifications_save_path = model_eval_outputs_dir / misclassifications_save_name
    misclassifications.to_csv(misclassifications_save_path, index=False)

    # Upload model to Hugging Face
    print("Uploading model to Hugging Face...")
    trainer.push_to_hub(
        model_name,
        token=os.getenv("HF_AUTH_TOKEN"),
    )

    print("Done!")