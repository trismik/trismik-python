import os
from typing import Optional, Union

import httpx

from trismik.exceptions import TrismikError


class TrismikUtils:
    """
    Utility functions for the Trismik client.

    This class provides helper methods for error handling and configuration
    management.
    """

    @staticmethod
    def get_error_message(response: httpx.Response) -> str:
        """
        Extract error message from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response containing the error.

        Returns:
            str: The error message from the response JSON or content.
        """
        try:
            json_data = response.json()
            title = str(json_data.get("title", "Unknown error"))
            message = str(json_data.get("detail", ""))
            if len(message):
                return title + ": " + message
            return title
        except (httpx.RequestError, ValueError):
            error_message: str = response.content.decode("utf-8", errors="ignore")
            return error_message

    @staticmethod
    def required_option(value: Optional[str], name: str, env: str) -> str:
        """
        Get a required configuration option.

        Get a required configuration option from either the provided value or
        environment variable.

        Args:
            value (Optional[str]): The provided value for the option.
            name (str): The name of the option for error messages.
            env (str): The environment variable name to check if value is None.

        Returns:
            str: The option value.

        Raises:
            TrismikError: If neither value nor environment variable is set.
        """
        if value is None:
            value = os.environ.get(env)
        if value is None:
            raise TrismikError(
                f"The {name} client option must be set either by passing "
                f"{env} to the client or by setting the {env} "
                "environment variable"
            )
        return value

    @staticmethod
    def option(
        value: Optional[str],
        default: str,
        env: str,
    ) -> str:
        """
        Get an optional configuration value.

        Get an optional configuration value from either the provided value,
        environment variable, or default value.

        Args:
            value (Optional[str]): The provided value for the option.
            default (str): The default value to use if neither value nor env var
                is set.
            env (str): The environment variable name to check if value is None.

        Returns:
            str: The option value, falling back to default if not set.
        """
        if value is None:
            value = os.environ.get(env)
        if value is None:
            return default
        return value

    @staticmethod
    def metric_value_to_type(value: Union[str, float, int, bool]) -> str:
        """
        Automatically determine valueType from Python value type.

        Args:
            value (Union[str, float, int, bool]): The metric value to analyze.

        Returns:
            str: The corresponding valueType string for the API.

        Raises:
            TypeError: If the value type is not supported.
        """
        if isinstance(value, bool):
            # Handle bool first since bool is a subclass of int in Python
            return "Boolean"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, float):
            return "Float"
        elif isinstance(value, int):
            return "Integer"
        else:
            raise TypeError(
                f"Unsupported metric value type: {type(value).__name__}. "
                f"Supported types: str, float, int, bool"
            )
