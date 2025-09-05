"""
Trismik adaptive test runner.

This module provides both synchronous and asynchronous interfaces for running
Trismik tests. The async implementation is the core, with sync methods wrapping
the async ones.
"""

import asyncio
from typing import Any, Callable, Dict, List, Literal, Optional, Union, overload

import nest_asyncio
from tqdm.auto import tqdm

from trismik.client_async import TrismikAsyncClient
from trismik.settings import evaluation_settings
from trismik.types import (
    AdaptiveTestScore,
    TrismikAdaptiveTestState,
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
    TrismikDataset,
    TrismikItem,
    TrismikMeResponse,
    TrismikReplayRequest,
    TrismikReplayRequestItem,
    TrismikRunMetadata,
    TrismikRunResults,
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

    def list_datasets(self) -> List[TrismikDataset]:
        """
        Get a list of available datasets synchronously.

        Returns:
            List[TrismikDataset]: List of available datasets.

        Raises:
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        return loop.run_until_complete(self.list_datasets_async())

    async def list_datasets_async(self) -> List[TrismikDataset]:
        """
        Get a list of available datasets asynchronously.

        Returns:
            List[TrismikDataset]: List of available datasets.

        Raises:
            TrismikApiError: If API request fails.
        """
        return await self._client.list_datasets()

    def me(self) -> TrismikMeResponse:
        """
        Get current user information synchronously.

        Returns:
            TrismikMeResponse: User information including validity and payload.

        Raises:
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        return loop.run_until_complete(self.me_async())

    async def me_async(self) -> TrismikMeResponse:
        """
        Get current user information asynchronously.

        Returns:
            TrismikMeResponse: User information including validity and payload.

        Raises:
            TrismikApiError: If API request fails.
        """
        return await self._client.me()

    @overload
    def run(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[True],
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    def run(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[False],
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    def run(
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Run a test synchronously.

        Args:
            test_id (str): ID of the test to run.
            project_id (str): ID of the project.
            experiment (str): Name of the experiment.
            run_metadata (TrismikRunMetadata): Metadata for the
              run.
            return_dict (bool): If True, return results as a dictionary instead
              of TrismikRunResults object. Defaults to True.
            with_responses (bool): If True, responses will be included with
              the results.

        Returns:
            Union[TrismikRunResults, Dict[str, Any]]: Either TrismikRunResults
              object or dictionary representation based on return_dict
              parameter.

        Raises:
            TrismikApiError: If API request fails.
            NotImplementedError: If with_responses = True (not yet implemented).
        """
        loop = self._get_loop()
        if return_dict:
            return loop.run_until_complete(
                self.run_async(
                    test_id,
                    project_id,
                    experiment,
                    run_metadata,
                    True,
                    with_responses,
                )
            )
        else:
            return loop.run_until_complete(
                self.run_async(
                    test_id,
                    project_id,
                    experiment,
                    run_metadata,
                    False,
                    with_responses,
                )
            )

    @overload
    async def run_async(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[True],
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    async def run_async(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[False],
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    async def run_async(
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Run a test asynchronously.

        Args:
            test_id: ID of the test to run.
            project_id: ID of the project.
            experiment: Name of the experiment.
            run_metadata: Metadata for the run.
            return_dict: If True, return results as a dictionary instead
              of TrismikRunResults object. Defaults to True.
            with_responses: If True, responses will be included with
              the results.

        Returns:
            Union[TrismikRunResults, Dict[str, Any]]: Either TrismikRunResults
              object or dictionary representation based on return_dict
              parameter.

        Raises:
            TrismikApiError: If API request fails.
            NotImplementedError: If with_responses = True (not yet implemented).
        """
        if with_responses:
            raise NotImplementedError(
                "with_responses is not yet implemented for the new API flow"
            )

        # Start run and get first item
        start_response = await self._client.start_run(
            test_id, project_id, experiment, run_metadata
        )

        # Initialize state tracking
        states: List[TrismikAdaptiveTestState] = []
        run_id = start_response.run_info.id

        # Add initial state
        states.append(
            TrismikAdaptiveTestState(
                run_id=run_id,
                state=start_response.state,
                completed=start_response.completed,
            )
        )

        # Run the test and get last state
        last_state = await self._run_async(
            run_id, start_response.next_item, states
        )

        if not last_state:
            raise RuntimeError(
                "Test run completed but no final state was captured"
            )

        score = AdaptiveTestScore(
            theta=last_state.state.thetas[-1],
            std_error=last_state.state.std_error_history[-1],
        )

        results = TrismikRunResults(run_id, score=score)

        if return_dict:
            return {
                "run_id": results.run_id,
                "score": (
                    {
                        "theta": results.score.theta,
                        "std_error": results.score.std_error,
                    }
                    if results.score
                    else None
                ),
                "responses": results.responses,
            }
        else:
            return results

    @overload
    def run_replay(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[True],
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    def run_replay(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[False],
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    def run_replay(
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Replay the exact sequence of questions from a previous run.

        Wraps the run_replay_async method.

        Args:
            previous_run_id: ID of a previous run to replay.
            run_metadata: Metadata for the replay run.
            return_dict: If True, return results as a dictionary instead
              of TrismikRunResults object. Defaults to True.
            with_responses: If True, responses will be included
             with the results.

        Returns:
            Union[TrismikRunResults, Dict[str, Any]]: Either TrismikRunResults
              object or dictionary representation based on return_dict
              parameter.

        Raises:
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        if return_dict:
            return loop.run_until_complete(
                self.run_replay_async(
                    previous_run_id,
                    run_metadata,
                    True,
                    with_responses,
                )
            )
        else:
            return loop.run_until_complete(
                self.run_replay_async(
                    previous_run_id,
                    run_metadata,
                    False,
                    with_responses,
                )
            )

    @overload
    async def run_replay_async(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[True],
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    async def run_replay_async(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: Literal[False],
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    async def run_replay_async(
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Replay the exact sequence of questions from a previous run.

        Args:
            previous_run_id: ID of a previous run to replay.
            run_metadata: Metadata for the run.
            return_dict: If True, return results as a dictionary instead
              of TrismikRunResults object. Defaults to True.
            with_responses: If True, responses will be included
              with the results.

        Returns:
            Union[TrismikRunResults, Dict[str, Any]]: Either TrismikRunResults
              object or dictionary representation based on return_dict
              parameter.

        Raises:
            TrismikApiError: If API request fails.
        """
        # Get the original run summary to access dataset and responses
        original_summary = await self._client.run_summary(previous_run_id)

        # Build replay request by processing each item in the original order
        replay_items = []
        with tqdm(
            total=len(original_summary.dataset), desc="Running replay..."
        ) as pbar:
            for item in original_summary.dataset:
                # Handle both sync and async item processors
                if asyncio.iscoroutinefunction(self._item_processor):
                    response = await self._item_processor(item)
                else:
                    response = self._item_processor(item)

                # Create replay request item
                replay_item = TrismikReplayRequestItem(
                    itemId=item.id, itemChoiceId=response
                )
                replay_items.append(replay_item)
                pbar.update(1)

        # Create replay request
        replay_request = TrismikReplayRequest(responses=replay_items)

        # Submit replay with metadata
        replay_response = await self._client.submit_replay(
            previous_run_id, replay_request, run_metadata
        )

        # Create score from replay response
        score = AdaptiveTestScore(
            theta=replay_response.state.thetas[-1],
            std_error=replay_response.state.std_error_history[-1],
        )

        # Return results with optional responses
        if with_responses:
            results = TrismikRunResults(
                run_id=replay_response.id,
                score=score,
                responses=replay_response.responses,
            )
        else:
            results = TrismikRunResults(run_id=replay_response.id, score=score)

        if return_dict:
            return {
                "run_id": results.run_id,
                "score": (
                    {
                        "theta": results.score.theta,
                        "std_error": results.score.std_error,
                    }
                    if results.score
                    else None
                ),
                "responses": (
                    [
                        {
                            "dataset_item_id": resp.dataset_item_id,
                            "value": resp.value,
                            "correct": resp.correct,
                        }
                        for resp in results.responses
                    ]
                    if results.responses
                    else None
                ),
            }
        else:
            return results

    async def _run_async(
        self,
        run_id: str,
        first_item: Optional[TrismikItem],
        states: List[TrismikAdaptiveTestState],
    ) -> Optional[TrismikAdaptiveTestState]:
        """
        Run a test asynchronously.

        Args:
            run_id (str): ID of the run to execute.
            first_item (Optional[TrismikItem]): First item from run start.
            states (List[TrismikAdaptiveTestState]): List to accumulate states.

        Returns:
            Optional[TrismikAdaptiveTestState]: Last state of the run.

        Raises:
            TrismikApiError: If API request fails.
        """
        item = first_item
        with tqdm(total=self._max_items, desc="Evaluating") as pbar:
            while item is not None:
                # Handle both sync and async item processors
                if asyncio.iscoroutinefunction(self._item_processor):
                    response = await self._item_processor(item)
                else:
                    response = self._item_processor(item)

                # Continue run with response
                continue_response = await self._client.continue_run(
                    run_id, response
                )

                # Update state tracking
                states.append(
                    TrismikAdaptiveTestState(
                        run_id=run_id,
                        state=continue_response.state,
                        completed=continue_response.completed,
                    )
                )

                pbar.update(1)

                if continue_response.completed:
                    pbar.total = pbar.n  # Update total to current position
                    pbar.refresh()
                    break

                item = continue_response.next_item

        last_state = states[-1] if states else None

        return last_state

    def submit_classic_eval(
        self, classic_eval_request: TrismikClassicEvalRequest
    ) -> TrismikClassicEvalResponse:
        """
        Submit a classic evaluation run with pre-computed results synchronously.

        Args:
            classic_eval_request (TrismikClassicEvalRequest): Request containing
                project info, dataset, model outputs, and metrics.

        Returns:
            TrismikClassicEvalResponse: Response from the classic evaluation
                endpoint.

        Raises:
            TrismikPayloadTooLargeError: If the request payload exceeds the
                server's size limit.
            TrismikValidationError: If the request fails validation.
            TrismikApiError: If API request fails.
        """
        loop = self._get_loop()
        return loop.run_until_complete(
            self.submit_classic_eval_async(classic_eval_request)
        )

    async def submit_classic_eval_async(
        self, classic_eval_request: TrismikClassicEvalRequest
    ) -> TrismikClassicEvalResponse:
        """
        Submit a classic evaluation run with pre-computed results async.

        This method allows you to submit pre-computed model outputs and metrics
        for evaluation without running an interactive test.

        Args:
            classic_eval_request (TrismikClassicEvalRequest): Request containing
                project info, dataset, model outputs, and metrics.

        Returns:
            TrismikClassicEvalResponse: Response from the classic evaluation
                endpoint.

        Raises:
            TrismikPayloadTooLargeError: If the request payload exceeds the
                server's size limit.
            TrismikValidationError: If the request fails validation.
            TrismikApiError: If API request fails.
        """
        return await self._client.submit_classic_eval(classic_eval_request)
