"""
Trismik adaptive test runner.

This module provides both synchronous and asynchronous interfaces for running
Trismik tests. The async implementation is the core, with sync methods wrapping
the async ones.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import nest_asyncio
from tqdm.auto import tqdm

from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikRunResults,
    TrismikSessionMetadata,
)


class AdaptiveTest:
    """
    Trismik test runner with both sync and async interfaces.

    This class provides both synchronous and asynchronous interfaces for
    running Trismik tests. The async implementation is the core, with sync
    methods wrapping the async ones.
    """

    def __init__(
        self,
        item_processor: Callable[[TrismikItem], Any],
        client: Optional[TrismikAsyncClient] = None,
        auth: Optional[TrismikAuth] = None,
        max_items: int = 60,
    ) -> None:
        """
        Initialize a new Trismik runner.

        Args:
            item_processor (Callable[[TrismikItem], Any]): Function to process
              test items. For async usage, this should be an async function.
            client (Optional[TrismikAsyncClient]): Trismik async client to use
              for requests.
            auth (Optional[TrismikAuth]): Authentication token to use for
              requests.
            max_items (int): Maximum number of items to process. Default is 60.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._item_processor = item_processor
        self._client = client
        self._auth = auth
        self._max_items = max_items
        self._loop = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get or create an event loop, handling nested loops if needed.

        Returns:
            asyncio.AbstractEventLoop: The event loop to use.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in this thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Allow nested event loops (needed for Jupyter, etc)
        nest_asyncio.apply(loop)
        return loop

    def run(
        self,
        test_id: str,
        session_metadata: TrismikSessionMetadata,
        with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Run a test synchronously.

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
        loop = self._get_loop()
        return loop.run_until_complete(
            self.run_async(test_id, session_metadata, with_responses)
        )

    async def run_async(
        self,
        test_id: str,
        session_metadata: TrismikSessionMetadata,
        with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Run a test asynchronously.

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
        await self._init_async()
        assert (
            self._client is not None
        ), "Client should be initialized by _init_async()"
        assert (
            self._auth is not None
        ), "Auth should be initialized by _init_async()"
        await self._refresh_token_if_needed_async()
        session = await self._client.create_session(
            test_id, session_metadata, self._auth.token
        )

        await self._run_session_async(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(
                session.url, self._auth.token
            )
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    def run_replay(
        self,
        previous_session_id: str,
        session_metadata: TrismikSessionMetadata,
        with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Replay the exact sequence of questions from a previous session.

        Wraps the run_replay_async method.

        Args:
            previous_session_id (str): ID of a previous session to replay.
            session_metadata (TrismikSessionMetadata): Metadata for the
             session.
            with_responses (bool): If True, responses will be included
             with the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        return loop.run_until_complete(
            self.run_replay_async(
                previous_session_id, session_metadata, with_responses
            )
        )

    async def run_replay_async(
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
            with_responses (bool): If True, responses will be included
              with the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        await self._init_async()
        assert (
            self._client is not None
        ), "Client should be initialized by _init_async()"
        assert (
            self._auth is not None
        ), "Auth should be initialized by _init_async()"
        await self._refresh_token_if_needed_async()
        session = await self._client.create_replay_session(
            previous_session_id, session_metadata, self._auth.token
        )

        await self._run_session_async(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(
                session.url, self._auth.token
            )
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    async def _run_session_async(self, session_url: str) -> None:
        """
        Run a test session asynchronously.

        Args:
            session_url (str): URL of the session to run.

        Raises:
            TrismikApiError: If API request fails.
        """
        await self._init_async()
        assert (
            self._client is not None
        ), "Client should be initialized by _init_async()"
        assert (
            self._auth is not None
        ), "Auth should be initialized by _init_async()"
        await self._refresh_token_if_needed_async()
        item = await self._client.current_item(session_url, self._auth.token)
        with tqdm(total=self._max_items, desc="Running test") as pbar:
            while item is not None:
                await self._refresh_token_if_needed_async()
                # Handle both sync and async item processors
                if asyncio.iscoroutinefunction(self._item_processor):
                    response = await self._item_processor(item)
                else:
                    response = self._item_processor(item)
                next_item = await self._client.respond_to_current_item(
                    session_url, response, self._auth.token
                )
                pbar.update(1)
                if next_item is None:
                    break
                item = next_item

    async def _init_async(self) -> None:
        """
        Initialize the client and authenticate if needed asynchronously.

        Raises:
            TrismikApiError: If API request fails.
        """
        if self._client is None:
            self._client = TrismikAsyncClient()

        if self._auth is None:
            self._auth = await self._client.authenticate()

    async def _refresh_token_if_needed_async(self) -> None:
        """
        Refresh the authentication token if it's about to expire asynchronously.

        Raises:
            TrismikApiError: If API request fails.
        """
        assert (
            self._client is not None
        ), "Client should be initialized by _init_async()"
        assert (
            self._auth is not None
        ), "Auth should be initialized by _init_async()"
        if self._token_needs_refresh():
            self._auth = await self._client.refresh_token(self._auth.token)

    def _token_needs_refresh(self) -> bool:
        """
        Check if the authentication token needs to be refreshed.

        Returns:
            bool: True if the token needs to be refreshed, False otherwise.
        """
        assert (
            self._auth is not None
        ), "Auth should be initialized by _init_async()"
        return self._auth.expires < (datetime.now() + timedelta(minutes=5))
