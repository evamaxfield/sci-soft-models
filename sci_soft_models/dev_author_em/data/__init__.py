"""Stored dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from dataclasses_json import DataClassJsonMixin

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

# Dev Author EM datasets
ANNOTATED_DEV_AUTHOR_EM_PATH = DATA_FILES_DIR / "annotated-dev-author-em-resolved.csv"

# Data fetching URLs
SCI_SOFT_MODELS_DATA_URL_TEMPLATE = "https://raw.githubusercontent.com/evamaxfield/sci-soft-models/refs/tags/{version}/sci_soft_models/dev_author_em/data/files/{filename}"
SCI_SOFT_MODELS_DATA_FETCH_DEFAULT_VERSION = "v0.2.3"
SCI_SOFT_MODELS_DATA_FILES = [
    "annotated-dev-author-em-resolved.csv",
    "extended-paper-details.parquet",
    "joss-short-paper-details.parquet",
    "repo-contributors.parquet",
    "softwarex-short-paper-details.parquet",
]

###############################################################################


def _fetch_data() -> None:
    print("Fetching dev-author-em model data...")
    try:
        # Iter over data files and request and store them in the data files dir
        for filename in SCI_SOFT_MODELS_DATA_FILES:
            # Get storage path
            storage_path = DATA_FILES_DIR / filename

            # Store if not already stored
            if not storage_path.exists():
                # Fetch data
                url = SCI_SOFT_MODELS_DATA_URL_TEMPLATE.format(
                    version=SCI_SOFT_MODELS_DATA_FETCH_DEFAULT_VERSION,
                    filename=filename,
                )

                # Request as stream
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()

                    # Write to file
                    with open(storage_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

    except Exception as e:
        print(f"Failed to fetch dev-author-em model data. Error: {e}")


def _check_local_data_and_fetch() -> None:
    # Check if the data files dir exists
    if not DATA_FILES_DIR.exists():
        DATA_FILES_DIR.mkdir(parents=True)

    # Fetch data
    _fetch_data()


def load_annotated_dev_author_em_dataset() -> pd.DataFrame:
    """Load the annotated dev author em dataset."""
    _check_local_data_and_fetch()
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
    _check_local_data_and_fetch()

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
    _check_local_data_and_fetch()
    return pd.read_parquet(EXTENDED_PAPER_DETAILS_PATH)


@dataclass
class AuthorContribution(DataClassJsonMixin):
    author_id: str
    name: str
    doi: str
    repo: str


def load_author_contributors_dataset() -> pd.DataFrame:
    _check_local_data_and_fetch()

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
    _check_local_data_and_fetch()

    return pd.read_parquet(REPO_CONTRIBUTORS_PATH)
