"""Stored dataset loaders."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

###############################################################################
# Local storage paths

DATA_FILES_DIR = Path(__file__).parent / "files"

EXP_FILES_DIR = DATA_FILES_DIR / "experiments"

SOFT_SEARCH_TRAINING_2022_PATH = DATA_FILES_DIR / "soft-search-training-2022.parquet"
EAGER_SOFT_SUST_SURVEY_DATASET_PATH = DATA_FILES_DIR / "eager-soft-sust-survey.parquet"
EAGER_SOFT_MINING_DATASET_PATH = DATA_FILES_DIR / "eager-soft-mining.parquet"
SOFT_SEARCH_TRAINING_2025_PATH = DATA_FILES_DIR / "soft-search-training-2025.parquet"

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
