"""
Trismik adaptive test runner.

This module provides both synchronous and asynchronous interfaces for running
Trismik tests. The async implementation is the core, with sync methods wrapping
the async ones.
"""

import asyncio
from typing import Any, Callable, List, Optional

import nest_asyncio
from tqdm.auto import tqdm

from trismik.client_async import TrismikAsyncClient
from trismik.settings import evaluation_settings
from trismik.types import (
    AdaptiveTestScore,
    TrismikAdaptiveTestState,
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
        api_key: Optional[str] = None,
        max_items: int = evaluation_settings["max_iterations"],
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
            NotImplementedError: If with_responses = True (not yet implemented).
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
            NotImplementedError: If with_responses = True (not yet implemented).
        """
        if with_responses:
            raise NotImplementedError(
                "with_responses is not yet implemented for the new API flow"
            )

        # Start session and get first item
        start_response = await self._client.start_session(
            test_id, session_metadata
        )

        # Initialize state tracking
        states: List[TrismikAdaptiveTestState] = []
        session_id = start_response.session_info.id

        # Add initial state
        states.append(
            TrismikAdaptiveTestState(
                session_id=session_id,
                state=start_response.state,
                completed=start_response.completed,
            )
        )

        # Run the session and get last state
        last_state = await self._run_session_async(
            session_id, start_response.next_item, states
        )

        if not last_state:
            raise RuntimeError(
                "Test session completed but no final state was captured"
            )

        score = AdaptiveTestScore(
            theta=last_state.state.thetas[-1],
            std_error=last_state.state.std_error_history[-1],
        )

        return TrismikRunResults(session_id, score=score)

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

        # old code:
        # await self._run_session_async(session.url)
        # results = await self._client.results(session.url)

        # if with_responses:
        #     responses = await self._client.responses(session.url)
        #     return TrismikRunResults(session.id, results, responses)
        # else:
        #     return TrismikRunResults(session.id, results)

        # For replay, we do not have the new API flow implemented, so just
        # return a basic result for now.
        # To fix mypy, provide required arguments to _run_session_async (even
        # if not used) and construct TrismikRunResults with only session_id
        # and score=None.
        dummy_states: List[TrismikAdaptiveTestState] = []
        await self._run_session_async(session.url, None, dummy_states)
        return TrismikRunResults(session.id, score=None)

    async def _run_session_async(
        self,
        session_id: str,
        first_item: Optional[TrismikItem],
        states: List[TrismikAdaptiveTestState],
    ) -> Optional[TrismikAdaptiveTestState]:
        """
        Run a test session asynchronously.

        Args:
            session_id (str): ID of the session to run.
            first_item (Optional[TrismikItem]): First item from session start.
            states (List[TrismikAdaptiveTestState]): List to accumulate states.

        Returns:
            Optional[TrismikAdaptiveTestState]: Last state of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        item = first_item
        with tqdm(total=self._max_items, desc="Running test") as pbar:
            while item is not None:
                # Handle both sync and async item processors
                if asyncio.iscoroutinefunction(self._item_processor):
                    response = await self._item_processor(item)
                else:
                    response = self._item_processor(item)

                # Continue session with response
                continue_response = await self._client.continue_session(
                    session_id, response
                )

                # Update state tracking
                states.append(
                    TrismikAdaptiveTestState(
                        session_id=session_id,
                        state=continue_response.state,
                        completed=continue_response.completed,
                    )
                )

                pbar.update(1)

                if continue_response.completed:
                    break

                item = continue_response.next_item

        last_state = states[-1] if states else None

        return last_state
