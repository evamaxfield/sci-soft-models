#!/usr/bin/env python

import logging
from dataclasses import dataclass

import torch
from dataclasses_json import DataClassJsonMixin
from transformers import Pipeline, pipeline

from .. import __version__
from ..types import ModelDetails
from .constants import MODEL_STR_INPUT_TEMPLATE, TRAINED_UPLOADED_MODEL_NAME

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


@dataclass
class DeveloperDetails(DataClassJsonMixin):
    username: str
    name: str | None = None
    email: str | None = None


@dataclass
class MatchedDevAuthor(DataClassJsonMixin):
    dev: DeveloperDetails
    author: str
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


def load_dev_author_em_model(use_available_device: bool | str = True) -> Pipeline:
    """
    Load the author-dev EM model.

    Parameters
    ----------
    use_available_device: bool | str | int
        Whether to use the available device, by default True
        Can pass a string for a specific device to use.

    Returns
    -------
    Pipeline
        The author-dev EM model.
    """
    # Check for device
    if isinstance(use_available_device, str):
        device = use_available_device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return pipeline(
        "text-classification",
        model=TRAINED_UPLOADED_MODEL_NAME,
        device=device,
    )


def match_devs_and_authors(
    devs: list[DeveloperDetails],
    authors: list[str],
    loaded_dev_author_em_model: Pipeline | None = None,
    use_available_device: bool | str = True,
) -> list[MatchedDevAuthor]:
    """
    Embed developers and authors and predict matches.

    Only use this on a set of authors and developers
    from the same repo-document pair. This will not work globally.

    Parameters
    ----------
    devs: list[DeveloperDetails]
        The developers to embed.
    authors: list[str]
        The authors to embed.
    loaded_dev_author_em_model: Pipeline, optional
        The loaded author EM model, by default None
    use_available_device: bool | str
        Whether to use the available device, by default True
        Can pass a string for a specific device to use.

    Returns
    -------
    list[MatchedDevAuthor]
        The predicted matches
    """
    # If no loaded classifer, load the model
    if loaded_dev_author_em_model is None:
        clf = load_dev_author_em_model(use_available_device=use_available_device)
    else:
        clf = loaded_dev_author_em_model

    # Create xarray for filled templates
    inputs: list[dict[str, str | DeveloperDetails]] = []
    for dev in devs:
        for author in authors:
            inputs.append(
                {
                    "dev_username": dev.username,
                    "dev_details": dev,
                    "author": author,
                    "text": MODEL_STR_INPUT_TEMPLATE.format(
                        dev_username=dev.username,
                        dev_name=dev.name,
                        dev_email=dev.email,
                        author_name=author,
                    ),
                }
            )

    # Predict the matches
    log.debug("Predicting matches")
    outputs: list[dict[str, str | float]] = clf([input_["text"] for input_ in inputs])

    # Extract the matches
    matches = []
    for input_, output_ in zip(inputs, outputs, strict=True):
        # Unpack output
        output_label = output_["label"]
        output_score = output_["score"]

        # Assert to make types happy
        assert isinstance(output_label, str)
        assert isinstance(output_score, float)
        assert isinstance(input_["dev_details"], DeveloperDetails)

        if output_label == "match":
            matches.append(
                MatchedDevAuthor(
                    dev=input_["dev_details"],
                    author=input_["author"],
                    confidence=output_score,
                )
            )

    return matches
