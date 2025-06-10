"""
Trismik Python Client.

A Python client for the Trismik API.
"""

import importlib.metadata

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)
