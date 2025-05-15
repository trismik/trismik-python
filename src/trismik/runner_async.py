"""
Trismik async runner for running tests.

This module provides an asynchronous runner for running Trismik tests.
"""

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikRunResults,
    TrismikSessionMetadata,
)


class TrismikAsyncRunner:
    """
    Asynchronous runner for Trismik tests.

    This class provides an asynchronous interface for running Trismik tests.
    It handles authentication, session management, and test execution in an
    asynchronous manner.
    """

    def __init__(
        self,
        item_processor: Callable[[TrismikItem], Awaitable[Any]],
        client: Optional[TrismikAsyncClient] = None,
        auth: Optional[TrismikAuth] = None,
    ) -> None:
        """
        Initialize a new Trismik async runner.

        Args:
            item_processor (Callable[[TrismikItem], Awaitable[Any]]): Async
                function to process test items.
            client (Optional[TrismikAsyncClient]): Trismik client to use for
                requests.
            auth (Optional[TrismikAuth]): Authentication token to use for
                requests.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._item_processor = item_processor
        self._client = client
        self._auth = auth

    async def run(
        self,
        test_id: str,
        session_metadata: TrismikSessionMetadata,
        with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Run a test.

        Args:
            test_id (str): ID of the test to run.
            session_metadata (TrismikSessionMetadata): Metadata for the
                session.
            with_responses (bool): If True, responses will be included with
                the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        await self._init()
        await self._refresh_token_if_needed()
        session = await self._client.create_session(
            test_id, session_metadata, self._auth.token
        )

        await self._run_session(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(
                session.url, self._auth.token
            )
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    async def run_replay(
        self,
        previous_session_id: str,
        session_metadata: TrismikSessionMetadata,
        with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Replay the exact sequence of questions from a previous session.

        Args:
            previous_session_id (str): ID of a previous session to replay.
            session_metadata (TrismikSessionMetadata): Metadata for the
                session.
            with_responses (bool): If True, responses will be included with
                the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        await self._init()
        await self._refresh_token_if_needed()
        session = await self._client.create_replay_session(
            previous_session_id, session_metadata, self._auth.token
        )

        await self._run_session(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(
                session.url, self._auth.token
            )
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    async def _run_session(self, session_url: str) -> None:
        """
        Run a test session.

        Args:
            session_url (str): URL of the session to run.

        Raises:
            TrismikApiError: If API request fails.
        """
        await self._init()
        await self._refresh_token_if_needed()
        item = await self._client.current_item(session_url, self._auth.token)
        while item is not None:
            await self._refresh_token_if_needed()
            response = await self._item_processor(item)
            item = await self._client.respond_to_current_item(
                session_url, response, self._auth.token
            )

    async def _init(self) -> None:
        """
        Initialize the client and authenticate if needed.

        Raises:
            TrismikApiError: If API request fails.
        """
        if self._client is None:
            self._client = TrismikAsyncClient()

        if self._auth is None:
            self._auth = await self._client.authenticate()

    async def _refresh_token_if_needed(self) -> None:
        """
        Refresh the authentication token if it's about to expire.

        Raises:
            TrismikApiError: If API request fails.
        """
        if self._token_needs_refresh():
            self._auth = await self._client.refresh_token(self._auth.token)

    def _token_needs_refresh(self) -> bool:
        """
        Check if the authentication token needs to be refreshed.

        Returns:
            bool: True if the token needs to be refreshed, False otherwise.
        """
        return self._auth.expires < (datetime.now() + timedelta(minutes=5))
