"""Stored dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

EXP_FILES_DIR = DATA_FILES_DIR / "experiments"

SOFT_SEARCH_TRAINING_2022_PATH = (
    DATA_FILES_DIR / "soft-search-training-2022.parquet"
)
EAGER_SOFT_SUST_SURVEY_DATASET_PATH = (
    DATA_FILES_DIR / "eager-soft-sust-survey.parquet"
)
EAGER_SOFT_MINING_DATASET_PATH = DATA_FILES_DIR / "eager-soft-mining.parquet"

###############################################################################

def load_soft_search_2022_dataset() -> pd.DataFrame:
    """Load the original soft search training 2022 dataset."""
    df = pd.read_parquet(SOFT_SEARCH_TRAINING_2022_PATH)
    df = df.rename(columns={"label": "software_produced", "nsf_award_id": "grant_id"})
    df["software_produced"] = df["software_produced"] == "software-predicted"
    df["software_produced_label_source"] = "soft-search-training-2022"
    return df[[
        "software_produced_label_source",
        "grant_id",
        "software_produced",
    ]]

def load_eager_soft_sust_survey_grant_details_dataset() -> pd.DataFrame:
    """Load the EAGER software sustainability survey dataset."""
    return pd.read_parquet(EAGER_SOFT_SUST_SURVEY_DATASET_PATH)

def load_eager_soft_mining_dataset() -> pd.DataFrame:
    """Load the soft search dataset from the RS graph and EAGER software mining."""
    return pd.read_parquet(EAGER_SOFT_MINING_DATASET_PATH)

def load_soft_search_2025_dataset() -> pd.DataFrame:
    """Load the combined software dataset."""
    # Load the datasets
    soft_search_2022_df = load_soft_search_2022_dataset()
    eager_soft_sust_survey_df = load_eager_soft_sust_survey_grant_details_dataset()
    soft_search_from_rs_graph_df = load_eager_soft_mining_dataset()

    # Combine the datasets
    combined = pd.concat([
        soft_search_from_rs_graph_df,
        eager_soft_sust_survey_df,
        soft_search_2022_df,
    ]).reset_index(drop=True)

    # Drop duplicates on grant id
    combined = combined.drop_duplicates(subset=["grant_id"])

    return combined