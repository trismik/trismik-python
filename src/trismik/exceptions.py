"""
Exception classes for the Trismik client.

This module defines custom exceptions used throughout the Trismik client
library.
"""

from typing import Optional


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


class TrismikNotFoundError(TrismikApiError):
    """
    Exception raised when the requested resource is not found.

    This exception is raised when a 404 "Not Found" error is received from the
    API. The Trismik backend masks cross-account access as 404 to prevent
    probing of resource IDs.
    """

    def __init__(self, message: str):
        """
        Initialize the TrismikNotFoundError.

        Args:
            message (str): The error message from the server.
        """
        super().__init__(message)

    def __str__(self) -> str:
        """Return a human-readable string representation of the exception."""
        return f"Not found: {self.args[0]}"


class TrismikRateLimitedError(TrismikApiError):
    """
    Exception raised when the API rate limit has been exceeded.

    This exception is raised when a 429 "Too Many Requests" error is received
    from the API. If the server provided a Retry-After header, its value is
    available via the ``retry_after_seconds`` attribute.
    """

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        """
        Initialize the TrismikRateLimitedError.

        Args:
            message (str): The error message from the server.
            retry_after_seconds (Optional[float]): Seconds to wait before
                retrying, parsed from the Retry-After header when present.
        """
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

    def __str__(self) -> str:
        """Return a human-readable string representation of the exception."""
        if self.retry_after_seconds is not None:
            return f"Rate limited: {self.args[0]} (retry after {self.retry_after_seconds}s)"
        return f"Rate limited: {self.args[0]}"
