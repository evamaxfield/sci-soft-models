#!/usr/bin/env python

from sci_soft_models import dev_author_em

###############################################################################

EXAMPLE_DEVS = [
    dev_author_em.DeveloperDetails(
        username="evamaxfield", name="Eva Maxfield Brown",
    ),
    dev_author_em.DeveloperDetails(
        username="nniiicc", name="Nic Weber",
    ),
]

EXAMPLE_PARTIAL_DEVS = [
    dev_author_em.DeveloperDetails(
        username="evamaxfield", name="Eva Maxfield Brown",
    ),
    dev_author_em.DeveloperDetails(
        username="nniiicc",
    ),
]

EXAMPLE_MATCHING_AUTHORS = [
    "Eva Brown",
    "Nicholas Weber",
]

EXAMPLE_NO_MATCHING_AUTHORS = [
    "John Doe",
    "Jane Doe",
]

def test_dev_author_em_matching_authors() -> None:
    """Test the matching of authors and developers."""
    matched_authors = dev_author_em.match_devs_and_authors(
        devs=EXAMPLE_DEVS,
        authors=EXAMPLE_MATCHING_AUTHORS,
    )
    assert len(matched_authors) == 2
    assert matched_authors[0].author == EXAMPLE_MATCHING_AUTHORS[0]
    assert matched_authors[1].author == EXAMPLE_MATCHING_AUTHORS[1]
    assert matched_authors[0].dev.username == EXAMPLE_DEVS[0].username
    assert matched_authors[1].dev.username == EXAMPLE_DEVS[1].username

def test_dev_author_em_partial_devs() -> None:
    """Test the matching of authors and developers with partial devs."""
    matched_authors = dev_author_em.match_devs_and_authors(
        devs=EXAMPLE_PARTIAL_DEVS,
        authors=EXAMPLE_MATCHING_AUTHORS,
    )
    assert len(matched_authors) == 1
    assert matched_authors[0].author == EXAMPLE_MATCHING_AUTHORS[0]
    assert matched_authors[0].dev.username == EXAMPLE_PARTIAL_DEVS[0].username

def test_dev_author_em_no_matching_authors() -> None:
    """Test the matching of authors and developers with no matches."""
    matched_authors = dev_author_em.match_devs_and_authors(
        devs=EXAMPLE_DEVS,
        authors=EXAMPLE_NO_MATCHING_AUTHORS,
    )
    assert len(matched_authors) == 0