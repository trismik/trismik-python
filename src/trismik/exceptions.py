"""
Exception classes for the Trismik client.

This module defines custom exceptions used throughout the Trismik client
library.
"""


class TrismikError(Exception):
    """Base class for all exceptions raised by the Trismik package."""


class TrismikApiError(TrismikError):
    """
    Exception raised when an error occurs during API interaction.

    This exception is raised when there is an error during API communication.
    """
