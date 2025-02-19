"""Stored dataset loaders."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

EXP_FILES_DIR = DATA_FILES_DIR / "experiments"
FINAL_MODEL_TRAINING_DATA_DIR = DATA_FILES_DIR / "final-model-training-data"

SOFT_SEARCH_TRAINING_2022_PATH = DATA_FILES_DIR / "soft-search-training-2022.parquet"
EAGER_SOFT_SUST_SURVEY_DATASET_PATH = DATA_FILES_DIR / "eager-soft-sust-survey.parquet"
EAGER_SOFT_MINING_DATASET_PATH = DATA_FILES_DIR / "eager-soft-mining.parquet"
SOFT_SEARCH_TRAINING_2025_PATH = DATA_FILES_DIR / "soft-search-training-2025.parquet"
SOFT_SEARCH_TRAINING_2025_MANUAL_HOLDOUT_PATH = (
    DATA_FILES_DIR / "soft-search-training-2025-manual-holdout.parquet"
)

###############################################################################

# List of NSF award detail fields to retrieve
NSF_AWARD_FIELDS = [
    # NSF Details
    "agency",
    "fundAgencyCode",
    "awardAgencyCode",
    "cfdaNumber",
    "ueiNumber",
    "parentUeiNumber",
    "poName",
    "poEmail",
    "primaryProgram",
    "transType",
    # Dates and Amounts
    "date",
    "startDate",
    "expDate",
    "estimatedTotalAmt",
    "fundsObligatedAmt",
    "fundProgramName",
    # Awardee and PI
    "awardee",
    "awardeeName",
    "awardeeStateCode",
    "pdPIName",
    "piFirstName",
    "piMiddeInitial",
    "piLastName",
    "piEmail",
    "coPDPI",
    "perfLocation",
    "perfDistrictCode",
    "perfStateCode",
    # Grant Details and Outcomes
    "title",
    "abstractText",
    "projectOutComesReport",
    "publicationResearch",
    "publicationConference",
]

NSF_AWARD_FIELDS_STR = ",".join(NSF_AWARD_FIELDS)


class NSFDirectorates:
    Biological_Sciences = "BIO"
    Computer_and_Information_Science_and_Engineering = "CISE"
    Education_and_Human_Resources = "EHR"
    Engineering = "ENG"
    Geosciences = "GEO"
    Integrative_Activities = "OIA"
    International_Science_and_Engineering = "OISE"
    Mathematical_and_Physical_Sciences = "MPS"
    Social_Behavioral_and_Economic_Sciences = "SBE"
    Technology_Innovation_and_Partnerships = "TIP"


CFDA_NUMBER_TO_DIRECTORATE_LUT = {
    "47.041": NSFDirectorates.Engineering,
    "47.049": NSFDirectorates.Mathematical_and_Physical_Sciences,
    "47.050": NSFDirectorates.Geosciences,
    "47.070": NSFDirectorates.Computer_and_Information_Science_and_Engineering,
    "47.074": NSFDirectorates.Biological_Sciences,
    "47.075": NSFDirectorates.Social_Behavioral_and_Economic_Sciences,
    "47.076": NSFDirectorates.Education_and_Human_Resources,
    "47.079": NSFDirectorates.International_Science_and_Engineering,
    "47.083": NSFDirectorates.Integrative_Activities,
    "47.084": NSFDirectorates.Technology_Innovation_and_Partnerships,
}


NSF_DIRECTORATE_TO_CFDA_NUMBER_LUT = {
    code: number for number, code in CFDA_NUMBER_TO_DIRECTORATE_LUT.items()
}

###############################################################################


def _load_soft_search_2022_dataset() -> pd.DataFrame:
    """Load the original soft search training 2022 dataset."""
    df = pd.read_parquet(SOFT_SEARCH_TRAINING_2022_PATH)
    df = df.rename(columns={"label": "software_produced", "nsf_award_id": "grant_id"})
    df["software_produced"] = df["software_produced"] == "software-predicted"
    df["software_produced_label_source"] = "soft-search-training-2022"
    return df[
        [
            "software_produced_label_source",
            "grant_id",
            "software_produced",
        ]
    ]


def _load_eager_soft_sust_survey_grant_details_dataset() -> pd.DataFrame:
    """Load the EAGER software sustainability survey dataset."""
    return pd.read_parquet(EAGER_SOFT_SUST_SURVEY_DATASET_PATH)


def _load_eager_soft_mining_dataset() -> pd.DataFrame:
    """Load the soft search dataset from the RS graph and EAGER software mining."""
    return pd.read_parquet(EAGER_SOFT_MINING_DATASET_PATH)


def _get_award_details_from_nsf(grant_id: str) -> dict:
    # Create request URL
    request_url = (
        f"https://www.research.gov/awardapi-service/v1/awards/"
        f"{grant_id}.json"
        f"?printFields={NSF_AWARD_FIELDS_STR}"
    )

    # Sleep to avoid rate limit
    time.sleep(0.05)

    # Make request
    response = requests.get(request_url)
    response.raise_for_status()

    # Parse response
    response_data = response.json()

    # Get awards
    awards = response_data["response"]["award"]

    # Handle not found
    if len(awards) == 0:
        return {
            "grant_id": grant_id,
            **{col: None for col in NSF_AWARD_FIELDS},
        }

    # Add grant id to award details
    award_details = awards[0]
    award_details["grant_id"] = grant_id

    return award_details


def _create_soft_search_2025_training_dataset(
    output_path: Path = SOFT_SEARCH_TRAINING_2025_PATH,
) -> pd.DataFrame:
    """Create the soft search 2025 training dataset."""
    # Load the datasets
    soft_search_2022_df = _load_soft_search_2022_dataset()
    eager_soft_sust_survey_df = _load_eager_soft_sust_survey_grant_details_dataset()
    soft_search_from_rs_graph_df = _load_eager_soft_mining_dataset()

    # Combine the datasets
    combined = pd.concat(
        [
            soft_search_from_rs_graph_df,
            eager_soft_sust_survey_df,
            soft_search_2022_df,
        ]
    ).reset_index(drop=True)

    # Drop duplicates on grant id
    combined = combined.drop_duplicates(subset=["grant_id"])

    # Read in existing output
    if output_path.exists():
        existing_output = pd.read_parquet(output_path)
        nsf_award_details = existing_output[
            [
                "grant_id",
                *NSF_AWARD_FIELDS,
            ]
        ].to_dict(orient="records")
        to_process = combined[~combined["grant_id"].isin(existing_output["grant_id"])]
    else:
        to_process = combined
        nsf_award_details = []

    # Get the NSF award details
    for i, grant_id in tqdm(
        enumerate(to_process["grant_id"]),
        desc="Fetching NSF Award Details",
        total=len(to_process),
    ):
        award_details = _get_award_details_from_nsf(grant_id)
        nsf_award_details.append(award_details)

        # Cache
        if i % 25 == 0:
            nsf_award_details_df = pd.DataFrame(nsf_award_details)

            # Always add the full set of columns
            for col in NSF_AWARD_FIELDS:
                if col not in nsf_award_details_df.columns:
                    nsf_award_details_df[col] = None

            # Merge the NSF award details
            annotated_awards = nsf_award_details_df.merge(
                combined, on="grant_id", how="left"
            )
            annotated_awards.to_parquet(output_path)

    # Create the NSF award details dataframe
    nsf_award_details_df = pd.DataFrame(nsf_award_details)

    # Always add the full set of columns
    for col in NSF_AWARD_FIELDS:
        if col not in nsf_award_details_df.columns:
            nsf_award_details_df[col] = None

    # Merge the NSF award details
    annotated_awards = nsf_award_details_df.merge(combined, on="grant_id", how="left")

    # Save the output
    annotated_awards.to_parquet(output_path)

    return annotated_awards


def _create_manual_holdout_for_soft_search_2025_training_dataset(
    manual_holdout_path: Path = SOFT_SEARCH_TRAINING_2025_MANUAL_HOLDOUT_PATH,
) -> None:
    # Load
    df = pd.read_parquet(SOFT_SEARCH_TRAINING_2025_PATH)

    # Drop NaNs on subset "title" and "abstractText"
    df = df.dropna(subset=["title", "abstractText"])

    # Add column for "directorate" based on "cfdaNumber"
    df["directorate"] = df["cfdaNumber"].map(CFDA_NUMBER_TO_DIRECTORATE_LUT)

    # Drop NaNs in "software_produced" and "directorate"
    df = df.dropna(subset=["software_produced", "directorate"])

    # Get value counts of directorate
    directorate_value_counts = df["directorate"].value_counts()

    # Create a "reduced_directorate" column that sets directorate to "other"
    # if the value count is less than 500
    df["reduced_directorate"] = df["directorate"].where(
        df["directorate"].isin(
            directorate_value_counts[directorate_value_counts >= 500].index
        ),
        "other",
    )

    # Create column for "stratify_group
    df["stratify_group"] = (
        df["reduced_directorate"] + "-" + df["software_produced"].astype(str)
    )

    # Set seed for random
    random.seed(42)
    np.random.seed(42)

    # For each stratify group, sample 5 examples to store into a manual holdout set
    sorted_stratifed_groups = sorted(df["stratify_group"].unique())
    heldout_groups = []
    for stratify_group in sorted_stratifed_groups:
        stratify_group_df = df.loc[df["stratify_group"] == stratify_group]
        manual_holdout_subset = stratify_group_df.sample(n=5, random_state=42)
        heldout_groups.append(manual_holdout_subset)

    # Concatenate the manual holdout sets
    manual_holdout = pd.concat(heldout_groups).reset_index(drop=True)

    # Write manual holdout
    manual_holdout.to_parquet(manual_holdout_path)


def load_soft_search_2025_training_dataset() -> pd.DataFrame:
    """Load the soft search 2025 training dataset."""
    # Load
    df = pd.read_parquet(SOFT_SEARCH_TRAINING_2025_PATH)

    # Drop NaNs on subset "title" and "abstractText"
    df = df.dropna(subset=["title", "abstractText"])

    # Add column for "directorate" based on "cfdaNumber"
    df["directorate"] = df["cfdaNumber"].map(CFDA_NUMBER_TO_DIRECTORATE_LUT)

    # Drop NaNs in "software_produced" and "directorate"
    df = df.dropna(subset=["software_produced", "directorate"])

    # Get value counts of directorate
    directorate_value_counts = df["directorate"].value_counts()

    # Create a "reduced_directorate" column that sets directorate to "other"
    # if the value count is less than 500
    df["reduced_directorate"] = df["directorate"].where(
        df["directorate"].isin(
            directorate_value_counts[directorate_value_counts >= 500].index
        ),
        "other",
    )

    # Create column for "stratify_group
    df["stratify_group"] = (
        df["reduced_directorate"] + "-" + df["software_produced"].astype(str)
    )

    # Read in manual hold out dataset
    manual_holdout = pd.read_parquet(SOFT_SEARCH_TRAINING_2025_MANUAL_HOLDOUT_PATH)

    # Remove the manual holdout from the training dataset
    df = df.loc[~df["grant_id"].isin(manual_holdout["grant_id"])].reset_index(drop=True)

    return df
