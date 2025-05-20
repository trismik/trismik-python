"""
Trismik client for interacting with the Trismik API.

This module provides a synchronous client for interacting with the Trismik API.
It wraps the async client to provide a synchronous interface.

.. deprecated:: 0.9.2
    This module is deprecated and will be removed in a future version.
"""

import asyncio
import warnings
from typing import Any, List, Optional

import nest_asyncio

from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikResponse,
    TrismikResult,
    TrismikSession,
    TrismikSessionMetadata,
    TrismikTest,
)


class TrismikClient:
    """
    Synchronous client for the Trismik API.

    This class provides a synchronous interface to interact with the Trismik
    API, handling authentication, test sessions, and responses.

    .. deprecated:: 0.9.2
        This class is deprecated and will be removed in a future version.
    """

    _serviceUrl: str = "https://zoo-dashboard.trismik.com/api"

    def __init__(
        self,
        service_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Trismik client.

        Args:
            service_url (Optional[str]): URL of the Trismik service.
            api_key (Optional[str]): API key for the Trismik service.
            http_client (Optional[Any]): HTTP client to use for requests
                (ignored, kept for backward compatibility).

        Raises:
            TrismikError: If service_url or api_key are not provided and not
                found in environment.
            TrismikApiError: If API request fails.
        """
        warnings.warn(
            "TrismikClient is deprecated since version 0.9.2 and will be "
            "removed in a future version. Please use "
            "trismik.client_async.TrismikAsyncClient directly instead. ",
            DeprecationWarning,
            stacklevel=2,
        )
        # Create and store a single event loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Allow nested event loops (needed for Jupyter, etc)
        nest_asyncio.apply(self._loop)

        # Create async client with our loop
        self._async_client = TrismikAsyncClient(
            service_url=service_url, api_key=api_key
        )

    def authenticate(self) -> TrismikAuth:
        """
        Authenticate with the Trismik service.

        Returns:
            TrismikAuth: Authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(self._async_client.authenticate())

    def refresh_token(self, token: str) -> TrismikAuth:
        """
        Refresh the authentication token.

        Args:
            token (str): Current authentication token.

        Returns:
            TrismikAuth: New authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.refresh_token(token)
        )

    def available_tests(self, token: str) -> List[TrismikTest]:
        """
        Get a list of available tests.

        Args:
            token (str): Authentication token.

        Returns:
            List[TrismikTest]: List of available tests.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.available_tests(token)
        )

    def create_session(
        self, test_id: str, metadata: TrismikSessionMetadata, token: str
    ) -> TrismikSession:
        """
        Create a new session for a test.

        Args:
            test_id (str): ID of the test.
            metadata (TrismikSessionMetadata): Metadata for the session.
            token (str): Authentication token.

        Returns:
            TrismikSession: New session.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.create_session(test_id, metadata, token)
        )

    def create_replay_session(
        self,
        previous_session_id: str,
        metadata: TrismikSessionMetadata,
        token: str,
    ) -> TrismikSession:
        """
        Create a new session that replays a previous session.

        Create a new session that replays exactly the question sequence of a
        previous session.

        Args:
            previous_session_id (str): Session id of the session to replay.
            metadata (TrismikSessionMetadata): Metadata for the session.
            token (str): Authentication token.

        Returns:
            TrismikSession: New session.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.create_replay_session(
                previous_session_id, metadata, token
            )
        )

    def add_metadata(
        self, session_id: str, metadata: TrismikSessionMetadata, token: str
    ) -> None:
        """
        Add metadata to the session, merging it with any already stored.

        Args:
            session_id (str): ID of the session object.
            metadata (TrismikSessionMetadata): Object containing the metadata
                to add.
            token (str): Authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.add_metadata(session_id, metadata, token)
        )

    def current_item(self, session_url: str, token: str) -> TrismikItem:
        """
        Get the current test item.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            TrismikItem: Current test item.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.current_item(session_url, token)
        )

    def respond_to_current_item(
        self, session_url: str, value: Any, token: str
    ) -> Optional[TrismikItem]:
        """
        Respond to the current test item.

        Args:
            session_url (str): URL of the session.
            value (Any): Response value.
            token (str): Authentication token.

        Returns:
            Optional[TrismikItem]: Next test item or None if session is
            finished.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.respond_to_current_item(
                session_url, value, token
            )
        )

    def results(self, session_url: str, token: str) -> List[TrismikResult]:
        """
        Get the results of a session.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            List[TrismikResult]: Results of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.results(session_url, token)
        )

    def responses(self, session_url: str, token: str) -> List[TrismikResponse]:
        """
        Get responses to session items.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            List[TrismikResponse]: Responses of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        return self._loop.run_until_complete(
            self._async_client.responses(session_url, token)
        )
