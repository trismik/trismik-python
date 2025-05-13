from datetime import datetime, timedelta
from typing import List, Callable, Any, Optional
import asyncio
import nest_asyncio

from trismik.client import TrismikClient
from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikTest,
    TrismikAuth,
    TrismikSession,
    TrismikItem,
    TrismikResult,
    TrismikResponse,
    TrismikSessionMetadata,
    TrismikRunResults,
)
from trismik.runner_async import TrismikAsyncRunner


class TrismikRunner:
    def __init__(
            self,
            item_processor: Callable[[TrismikItem], Any],
            client: Optional[TrismikClient] = None,
            auth: Optional[TrismikAuth] = None,
    ) -> None:
        """
        Initializes a new Trismik runner.

        Args:
            item_processor (Callable[[TrismikItem], Any]): Function to process test items.
            client (Optional[TrismikClient]): Trismik client to use for requests.
            auth (Optional[TrismikAuth]): Authentication token to use for requests

        Raises:
            TrismikApiError: If API request fails.
        """
        self._item_processor = item_processor
        self._client = client
        self._auth = auth
        self._loop = None
        self._async_runner = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop, handling nested loops if needed"""
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
        """Get or create the async runner instance"""
        if self._async_runner is None:
            # Create a wrapper for the sync item processor to make it async
            async def async_item_processor(item: TrismikItem) -> Any:
                return self._item_processor(item)

            # Create a new async client with the same configuration as the sync client
            async_client = None
            if self._client:
                async_client = TrismikAsyncClient(
                    service_url=self._client.service_url,
                    api_key=self._client.api_key
                )

            self._async_runner = TrismikAsyncRunner(
                item_processor=async_item_processor,
                client=async_client,
                auth=self._auth
            )
        return self._async_runner

    def run(self,
            test_id: str,
            session_metadata: TrismikSessionMetadata,
            with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Runs a test.

        Args:
            test_id (str): ID of the test to run.
            with_responses (bool): If True, responses will be included with the results.

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

    def run_replay(self,
            previous_session_id: str,
            session_metadata: TrismikSessionMetadata,
            with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Replay the exact sequence of questions from a previous session

        Args:
            previous_session_id (str): ID of a previous session to replay
            with_responses (bool): If True, responses will be included with the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        async_runner = self._get_async_runner()
        return loop.run_until_complete(
            async_runner.run_replay(previous_session_id, session_metadata, with_responses)
        )
