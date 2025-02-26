"""Stored dataset loaders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dataclasses_json import DataClassJsonMixin
from dotenv import load_dotenv

###############################################################################
# Dataverse info

DATAVERSE_HOST = "https://dataverse.harvard.edu/"
DATAVERSE_RS_GRAPH_V1_DATASET_DOI = "10.7910/DVN/KPYVI1"

###############################################################################


def _download_harvard_dataverse_data() -> None:
    # Check if any of the dataverse data is missing
    misclassifications_path = (
        FINAL_MODEL_TRAINING_DATA_DIR / "dev-author-em-misclassifications.csv"
    )
    test_set_path = FINAL_MODEL_TRAINING_DATA_DIR / "test-set.parquet"
    train_set_path = FINAL_MODEL_TRAINING_DATA_DIR / "train-set.parquet"
    if any(
        not path.exists()
        for path in [
            ANNOTATED_DEV_AUTHOR_EM_PATH,
            misclassifications_path,
            test_set_path,
            train_set_path,
        ]
    ):
        try:
            from easyDataverse import Dataverse

        except ImportError as e:
            raise ImportError(
                f"Certain datasets contain linked PII and "
                f"as such are not available in the public repo. "
                f"It is available via request from Harvard Dataverse: "
                f"https://doi.org/{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}. "
                f"Once you have access, please add your Harvard Dataverse API token "
                f"to your environment as 'DATAVERSE_TOKEN'.\n\n"
                f"In addition, please install data download requirements via: "
                f"pip install sci-soft-models[data]"
            ) from e

        # Log that we are downloading the data
        print("Downloading data from Harvard Dataverse...")

        # Load env and see if dataverse token is present
        load_dotenv()
        if "DATAVERSE_TOKEN" not in os.environ:
            raise ValueError(
                f"Certain datasets contain linked PII and "
                f"as such are not available in the public repo. "
                f"It is available via request from Harvard Dataverse: "
                f"https://doi.org/{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}. "
                f"Once you have access, please add your Harvard Dataverse API token "
                f"to your environment as 'DATAVERSE_TOKEN'."
            )

        # Otherwise, get the token and download the file
        dataverse_token = os.environ["DATAVERSE_TOKEN"]

        # Init Dataverse
        dv = Dataverse(DATAVERSE_HOST, api_token=dataverse_token)

        # Download all related files
        dv.load_dataset(
            pid=f"doi:{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}",
            filedir=FINAL_MODEL_TRAINING_DATA_DIR,
            filenames=[
                "train-set.parquet",
                "test-set.parquet",
                "dev-author-em-misclassifications.tab",
            ],
        )

        # Download annotated dev files too
        dv.load_dataset(
            pid=f"doi:{DATAVERSE_RS_GRAPH_V1_DATASET_DOI}",
            filedir=DATA_FILES_DIR,
            filenames=[
                "annotated-dev-author-em-resolved.tab",
            ],
        )

        # Convert .tab files to .csv
        pd.read_csv(
            FINAL_MODEL_TRAINING_DATA_DIR / "dev-author-em-misclassifications.tab",
            sep="\t",
        ).to_csv(
            FINAL_MODEL_TRAINING_DATA_DIR / "dev-author-em-misclassifications.csv",
            index=False,
        )
        pd.read_csv(
            DATA_FILES_DIR / "annotated-dev-author-em-resolved.tab",
            sep="\t",
        ).to_csv(
            ANNOTATED_DEV_AUTHOR_EM_PATH,
            index=False,
        )

        # Unlink the .tab files
        (
            FINAL_MODEL_TRAINING_DATA_DIR / "dev-author-em-misclassifications.tab"
        ).unlink()
        (DATA_FILES_DIR / "annotated-dev-author-em-resolved.tab").unlink()


###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

# Dev Author EM datasets
ANNOTATED_DEV_AUTHOR_EM_PATH = DATA_FILES_DIR / "annotated-dev-author-em-resolved.csv"
EXP_FILES_DIR = DATA_FILES_DIR / "experiments"
FINAL_MODEL_TRAINING_DATA_DIR = DATA_FILES_DIR / "final-model-training-data"

###############################################################################


def load_annotated_dev_author_em_dataset() -> pd.DataFrame:
    """Load the annotated dev author em dataset."""
    _download_harvard_dataverse_data()
    return pd.read_csv(ANNOTATED_DEV_AUTHOR_EM_PATH)


###############################################################################

# Dataset sources are found via path globbing
DATASET_SOURCE_FILE_PATTERN = "-short-paper-details.parquet"

# Other datasets are formed from enrichment and have hardcoded paths
EXTENDED_PAPER_DETAILS_PATH = DATA_FILES_DIR / "extended-paper-details.parquet"
REPO_CONTRIBUTORS_PATH = DATA_FILES_DIR / "repo-contributors.parquet"

###############################################################################


def load_basic_repos_dataset() -> pd.DataFrame:
    """Load the base dataset (all dataset sources)."""
    # Find all dataset files
    dataset_files = list(DATA_FILES_DIR.glob(f"*{DATASET_SOURCE_FILE_PATTERN}"))

    # Load all datasets
    datasets = []
    for dataset_file in dataset_files:
        datasets.append(pd.read_parquet(dataset_file))

    # Concatenate
    rs_graph = pd.concat(datasets)

    # Drop duplicates and keep first
    rs_graph = rs_graph.drop_duplicates(subset=["repo"], keep="first")

    return rs_graph


def load_extended_paper_details_dataset() -> pd.DataFrame:
    """Load the extended paper details dataset."""
    return pd.read_parquet(EXTENDED_PAPER_DETAILS_PATH)


@dataclass
class AuthorContribution(DataClassJsonMixin):
    author_id: str
    name: str
    doi: str
    repo: str


def load_author_contributors_dataset() -> pd.DataFrame:
    # Load extended paper details dataset
    paper_details_df = load_extended_paper_details_dataset()
    repos_df = load_basic_repos_dataset()

    # Create a look up table for each author
    author_contributions = []
    for _, paper_details in paper_details_df.iterrows():
        # Get DOI so we don't have to do a lot of getitems
        doi = paper_details["doi"]

        # Get matching row in the repos dataset
        repo_row = repos_df.loc[repos_df.doi == doi]

        # Skip if no matching row
        if len(repo_row) == 0:
            continue
        else:
            repo_row = repo_row.iloc[0]

        # Iter each author
        for author_details in paper_details["authors"]:
            a_id = author_details["author_id"]

            # Add new author
            author_contributions.append(
                AuthorContribution(
                    author_id=a_id,
                    name=author_details["name"],
                    doi=doi,
                    repo=repo_row["repo"],
                )
            )

    # Convert to dataframe
    all_author_details_df = pd.DataFrame(
        [author_contrib.to_dict() for author_contrib in author_contributions]
    )
    return all_author_details_df


def load_developer_contributors_dataset() -> pd.DataFrame:
    """Load the repo contributors dataset."""
    return pd.read_parquet(REPO_CONTRIBUTORS_PATH)


def load_experimental_training_results() -> pd.DataFrame:
    """Load the experimental training results."""
    return pd.read_csv(EXP_FILES_DIR / "exp-training-results.csv")


def load_final_model_training_results() -> dict:
    """Load the final model training results."""
    with open(FINAL_MODEL_TRAINING_DATA_DIR / "results.json") as f:
        return json.load(f)


def load_final_model_training_split_details() -> pd.DataFrame:
    """Load the final model training split details."""
    return pd.read_parquet(FINAL_MODEL_TRAINING_DATA_DIR / "split-counts.parquet")


def load_final_model_training_train_set() -> pd.DataFrame:
    """Load the final model training train set."""
    _download_harvard_dataverse_data()
    return pd.read_parquet(FINAL_MODEL_TRAINING_DATA_DIR / "train-set.parquet")


def load_final_model_training_test_set() -> pd.DataFrame:
    """Load the final model training test set."""
    _download_harvard_dataverse_data()
    return pd.read_parquet(FINAL_MODEL_TRAINING_DATA_DIR / "test-set.parquet")
