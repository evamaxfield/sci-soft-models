"""Stored dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dataclasses_json import DataClassJsonMixin

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

# Annotated datasets

# Dev Author EM datasets
ANNOTATED_DEV_AUTHOR_EM_PATH = DATA_FILES_DIR / "annotated-dev-author-em-resolved.csv"

###############################################################################


def load_annotated_dev_author_em_dataset() -> pd.DataFrame:
    """Load the annotated dev author em dataset."""
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
    all_author_details_df = pd.DataFrame([
        author_contrib.to_dict() for author_contrib in author_contributions
    ])
    return all_author_details_df


def load_developer_contributors_dataset() -> pd.DataFrame:
    """Load the repo contributors dataset."""
    return pd.read_parquet(REPO_CONTRIBUTORS_PATH)
