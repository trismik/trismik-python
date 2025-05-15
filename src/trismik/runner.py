"""
Trismik runner for running tests.

This module provides a synchronous runner for running Trismik tests. It wraps
the async runner to provide a synchronous interface.
"""

import asyncio
from typing import Any, Callable, Optional

import nest_asyncio

from trismik.client import TrismikClient
from trismik.client_async import TrismikAsyncClient
from trismik.runner_async import TrismikAsyncRunner
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikRunResults,
    TrismikSessionMetadata,
)


class TrismikRunner:
    """
    Synchronous runner for Trismik tests.

    This class provides a synchronous interface for running Trismik tests by
    wrapping the asynchronous runner. It handles event loop management and
    provides a simple interface for running tests and replaying sessions.
    """

    def __init__(
        self,
        item_processor: Callable[[TrismikItem], Any],
        client: Optional[TrismikClient] = None,
        auth: Optional[TrismikAuth] = None,
    ) -> None:
        """
        Initialize a new Trismik runner.

        Args:
            item_processor (Callable[[TrismikItem], Any]): Function to process
                test items.
            client (Optional[TrismikClient]): Trismik client to use for
                requests.
            auth (Optional[TrismikAuth]): Authentication token to use for
                requests.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._item_processor = item_processor
        self._client = client
        self._auth = auth
        self._loop = None
        self._async_runner = None

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

    def _get_async_runner(self) -> TrismikAsyncRunner:
        """
        Get or create the async runner instance.

        Returns:
            TrismikAsyncRunner: The async runner instance.
        """
        if self._async_runner is None:
            # Create a wrapper for the sync item processor to make it async
            async def async_item_processor(item: TrismikItem) -> Any:
                return self._item_processor(item)

            # Create a new async client with the same configuration
            # as the sync client
            async_client = None
            if self._client:
                async_client = TrismikAsyncClient(
                    service_url=self._client.service_url,
                    api_key=self._client.api_key,
                )

            self._async_runner = TrismikAsyncRunner(
                item_processor=async_item_processor,
                client=async_client,
                auth=self._auth,
            )
        return self._async_runner

    def run(
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
        loop = self._get_loop()
        async_runner = self._get_async_runner()
        return loop.run_until_complete(
            async_runner.run(test_id, session_metadata, with_responses)
        )

    def run_replay(
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
        loop = self._get_loop()
        async_runner = self._get_async_runner()
        return loop.run_until_complete(
            async_runner.run_replay(
                previous_session_id, session_metadata, with_responses
            )
        )
