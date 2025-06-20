"""
Trismik adaptive test runner.

This module provides both synchronous and asynchronous interfaces for running
Trismik tests. The async implementation is the core, with sync methods wrapping
the async ones.
"""

import asyncio
from typing import Any, Callable, Optional

import nest_asyncio
from tqdm.auto import tqdm

from trismik.client_async import TrismikAsyncClient
from trismik.types import TrismikItem, TrismikRunResults, TrismikSessionMetadata


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
        api_key: Optional[str] = None,
        max_items: int = 60,
    ) -> None:
        """
        Initialize a new Trismik runner.

        Args:
            item_processor (Callable[[TrismikItem], Any]): Function to process
              test items. For async usage, this should be an async function.
            client (Optional[TrismikAsyncClient]): Trismik async client to use
              for requests. If not provided, a new one will be created.
            api_key (Optional[str]): API key to use if a new client is created.
            max_items (int): Maximum number of items to process. Default is 60.

        Raises:
            ValueError: If both client and api_key are provided.
            TrismikApiError: If API request fails.
        """
        if client and api_key:
            raise ValueError(
                "Either 'client' or 'api_key' should be provided, not both."
            )
        self._item_processor = item_processor
        if client:
            self._client = client
        else:
            self._client = TrismikAsyncClient(api_key=api_key)
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
        session = await self._client.create_session(test_id, session_metadata)

        await self._run_session_async(session.url)
        results = await self._client.results(session.url)

        if with_responses:
            responses = await self._client.responses(session.url)
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
        session = await self._client.create_replay_session(
            previous_session_id, session_metadata
        )

        await self._run_session_async(session.url)
        results = await self._client.results(session.url)

        if with_responses:
            responses = await self._client.responses(session.url)
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
        item = await self._client.current_item(session_url)
        with tqdm(total=self._max_items, desc="Running test") as pbar:
            while item is not None:
                # Handle both sync and async item processors
                if asyncio.iscoroutinefunction(self._item_processor):
                    response = await self._item_processor(item)
                else:
                    response = self._item_processor(item)
                next_item = await self._client.respond_to_current_item(
                    session_url, response
                )
                pbar.update(1)
                if next_item is None:
                    break
                item = next_item
