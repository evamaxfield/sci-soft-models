"""Training and application of the developer-author entity matching model."""

from .main import DeveloperDetails, MatchedDevAuthor, load_dev_author_em_model, match_devs_and_authors

__all__ = [
    "DeveloperDetails",
    "MatchedDevAuthor",
    "load_dev_author_em_model",
    "match_devs_and_authors",
]