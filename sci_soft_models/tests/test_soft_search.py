#!/usr/bin/env python

from sci_soft_models import soft_search

###############################################################################


EAGER_AWARD_ID = "2211275"
EAGER_AWARD_TITLE = "Collaborative Research: EAGER: Characterizing Research Software from NSF Awards"  # noqa: E501
EAGER_AWARD_ABSTRACT = """
Software underlies many of the national economic advances of the last 60 years, from better weather models that enable more productivity to better-designed products that lower material usage and costs. Most of these advances originated in research software, initially via algorithms and licensing, and more recently via open-source code. Despite these numerous successes, much of the software produced in science and engineering research is not sustainable - it is not shared, maintained, or developed in ways that enable meaningful reuse. This lack of sustainable development hampers both research and economic progress. Using a combination of self-reported and administrative data this project is systematically investigating the development and maintenance of software produced in research projects funded by the National Science Foundation. The data collected for this study is being used to develop a set of models for research software sustainability planning. These models support scientists and engineers planning for software sustainability; help research funding agencies be better prepared to evaluate research software in grant applications, and enable institutions that support software development and maintenance to be equipped to support impactful research that in turn, produces impactful research software.

The goals of this project are: 1) to understand what factors influence software sustainability by gathering data from grant-funded research projects; 2) to describe current models of sustainability planning and suggest potential new models that could increase the likelihood of achieving long-term software sustainability; and 3) to develop emergent methods to evaluate research software sustainability. Data collection to meet these goals includes a survey and interviews with researchers that have produced software as part of an NSF-funded award. This project also uses emerging methods in analyzing research software code repositories in order to understand what activities in software development correlate with sustainability. This research has three intended impacts: 1) to provide a proof of concept for large-scale analysis of research software sustainability using a mixed-methods approach; 2) to modify existing standardized metrics of software health so that they can be used to evaluate the sustainability of research software; and 3) to create an initial set of sustainability models that researchers can use to better plan research software projects, funders can use to make better award decisions, and institutions can use to better allocate internal resources.
""".strip()  # noqa: E501

EXAMPLE_AWARDS = [
    soft_search.AwardDetails(
        id_=EAGER_AWARD_ID,
        title=EAGER_AWARD_TITLE,
        abstract=EAGER_AWARD_ABSTRACT,
        outcomes_report=None,
    ),
    soft_search.AwardDetails(
        id_=EAGER_AWARD_ID,
        title=EAGER_AWARD_TITLE,
        abstract=EAGER_AWARD_ABSTRACT,
        outcomes_report=(
            "We did not produce any software. "
            "This is meerly a test to see how the prediction changes. "
            "It shouldn't change. "
            "The prediction should be in favor of producing software."
        ),
    ),
    soft_search.AwardDetails(
        id_=EAGER_AWARD_ID,
        title=EAGER_AWARD_TITLE,
        abstract=EAGER_AWARD_ABSTRACT,
        outcomes_report=(
            "Ultimately we didn't produce any software or code "
            "as a part of our research as a majority of the work "
            "was spent on surveys and interviews with research software developers."
        ),
    ),
]

EXAMPLE_EXPECTED_OUTPUTS = [
    soft_search.AwardDetailsWithSoftwareProductionPrediction(
        award_details=EXAMPLE_AWARDS[0],
        software_production_prediction="software-produced",
        confidence=0.0,
    ),
    soft_search.AwardDetailsWithSoftwareProductionPrediction(
        award_details=EXAMPLE_AWARDS[1],
        software_production_prediction="software-produced",
        confidence=0.0,
    ),
    soft_search.AwardDetailsWithSoftwareProductionPrediction(
        award_details=EXAMPLE_AWARDS[2],
        software_production_prediction="software-not-produced",
        confidence=0.0,
    ),
]


def test_nsf_soft_search_v2() -> None:
    """Test the software prediction from awards."""
    software_predictions = soft_search.predict_software_production_from_awards(
        EXAMPLE_AWARDS
    )

    # Create tuple of predicted and expected outputs
    predicted = tuple([x.software_production_prediction for x in software_predictions])
    expected = tuple(
        [x.software_production_prediction for x in EXAMPLE_EXPECTED_OUTPUTS]
    )

    assert predicted == expected
