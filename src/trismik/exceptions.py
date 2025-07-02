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


class TrismikPayloadTooLargeError(TrismikApiError):
    """
    Exception raised when the request payload exceeds the server's size limit.

    This exception is raised when a 413 "Content Too Large" error is received
    from the API, indicating that the request payload (typically metadata)
    exceeds the server's size limit.
    """

    def __init__(self, message: str):
        """
        Initialize the TrismikPayloadTooLargeError.

        Args:
            message (str): The error message from the server.
        """
        super().__init__(message)

    def __str__(self) -> str:
        """Return a human-readable string representation of the exception."""
        return f"Payload too large: {self.args[0]}"


class TrismikValidationError(TrismikApiError):
    """
    Exception raised when the request fails validation.

    This exception is raised when a 422 "Unprocessable Entity" error is received
    from the API, indicating that the request failed validation (e.g., duplicate
    item IDs, unknown item IDs in replay requests).
    """

    def __init__(self, message: str):
        """
        Initialize the TrismikValidationError.

        Args:
            message (str): The error message from the server.
        """
        super().__init__(message)

    def __str__(self) -> str:
        """Return a human-readable string representation of the exception."""
        return f"Validation error: {self.args[0]}"
