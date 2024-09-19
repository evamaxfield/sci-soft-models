"""Top-level package for sci-soft-models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sci-soft-models")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamaxfieldbrown@gmail.com"
