#!/usr/bin/env python

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from transformers import Pipeline, pipeline

from .. import __version__
from ..types import ModelDetails
from ..utils import find_device
from .constants import MODEL_STR_INPUT_TEMPLATE, TRAINED_UPLOADED_MODEL_NAME

###############################################################################


@dataclass
class AwardDetails(DataClassJsonMixin):
    id_: str
    title: str
    abstract: str
    outcomes_report: str | None


@dataclass
class AwardDetailsWithSoftwareProductionPrediction(DataClassJsonMixin):
    award_details: AwardDetails
    software_production_prediction: str
    confidence: float


###############################################################################


def get_model_details() -> ModelDetails:
    """
    Get the name and version of the model.

    Returns
    -------
    ModelDetails
        The name and version of the model.
    """
    return ModelDetails(name=__name__, version=__version__)


def load_nsf_soft_search_v2(use_available_device: bool | str = True) -> Pipeline:
    """
    Load the NSF software production prediction model.

    Parameters
    ----------
    use_available_device: bool | str | int
        Whether to use the available device, by default True
        Can pass a string for a specific device to use.

    Returns
    -------
    Pipeline
        The loaded model.
    """
    device = find_device(use_available_device=use_available_device)

    return pipeline(
        "text-classification",
        model=TRAINED_UPLOADED_MODEL_NAME,
        device=device,
    )


def predict_software_production_from_awards(
    awards: list[AwardDetails],
    loaded_nsf_software_production_model: Pipeline | None = None,
    use_available_device: bool | str = True,
) -> list[AwardDetailsWithSoftwareProductionPrediction]:
    """
    Embed award details and predict software production.

    Parameters
    ----------
    awards: list[AwardDetails]
        The award details to predict software production for.
    loaded_nsf_software_production_model: Pipeline | None
        The loaded model, by default None
    use_available_device: bool | str
        Whether to use the available device, by default True
        Can pass a string for a specific device to use.

    Returns
    -------
    list[AwardDetailsWithSoftwareProductionPrediction]
        The predictions attached to each award
    """
    # If no loaded classifer, load the model
    if loaded_nsf_software_production_model is None:
        clf = load_nsf_soft_search_v2(use_available_device=use_available_device)
    else:
        clf = loaded_nsf_software_production_model

    # Prepare the inputs
    inputs: list[dict[str, str | AwardDetails]] = []
    for award in awards:
        input_: dict[str, str | AwardDetails] = {
            "text": MODEL_STR_INPUT_TEMPLATE.format(
                award_title=award.title,
                award_abstract=award.abstract,
                award_outcomes=str(award.outcomes_report),
            ),
            "award_details": award,
        }
        inputs.append(input_)

    # Predict software production
    outputs: list[dict[str, str | float]] = clf([input_["text"] for input_ in inputs])

    # Combine outputs with inputs
    resolved_outputs: list[AwardDetailsWithSoftwareProductionPrediction] = []
    for input_, output_ in zip(inputs, outputs, strict=True):
        # Unpack output
        output_label = output_["label"]
        output_score = output_["score"]

        # Assert to make types happy
        assert isinstance(output_label, str)
        assert isinstance(output_score, float)
        assert isinstance(input_["award_details"], AwardDetails)

        resolved_outputs.append(
            AwardDetailsWithSoftwareProductionPrediction(
                award_details=input_["award_details"],
                software_production_prediction=output_label,
                confidence=output_score,
            )
        )

    return resolved_outputs
