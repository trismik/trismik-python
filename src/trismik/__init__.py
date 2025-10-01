"""
Trismik Python Client.

A Python client for the Trismik API.
"""

import importlib.metadata

from trismik._async.client import TrismikAsyncClient
from trismik._sync.client import TrismikClient
from trismik.types import (
    AdaptiveTestScore,
    TrismikDataset,
    TrismikItem,
    TrismikMeResponse,
    TrismikProject,
    TrismikRunMetadata,
    TrismikRunResults,
)

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    # Clients
    "TrismikAsyncClient",
    "TrismikClient",
    # Common types
    "AdaptiveTestScore",
    "TrismikDataset",
    "TrismikItem",
    "TrismikMeResponse",
    "TrismikProject",
    "TrismikRunMetadata",
    "TrismikRunResults",
    # Version
    "__version__",
]
