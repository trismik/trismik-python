"""
Trismik async client for interacting with the Trismik API.

This module provides an asynchronous client for interacting with the Trismik
API. It uses httpx for making HTTP requests.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Union, overload

import httpx

from trismik._async.helpers import process_item
from trismik._mapper import TrismikResponseMapper
from trismik._utils import TrismikUtils
from trismik.exceptions import TrismikApiError, TrismikPayloadTooLargeError, TrismikValidationError
from trismik.settings import client_settings, environment_settings, evaluation_settings
from trismik.types import (
    AdaptiveTestScore,
    TrismikAdaptiveTestState,
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
    TrismikDataset,
    TrismikItem,
    TrismikMeResponse,
    TrismikProject,
    TrismikReplayRequest,
    TrismikReplayRequestItem,
    TrismikReplayResponse,
    TrismikRunMetadata,
    TrismikRunResponse,
    TrismikRunResults,
    TrismikRunSummary,
)


class TrismikAsyncClient:
    """
    Client for the Trismik API.

    Provides methods to interact with the Trismik API, including
    dataset management, test runs, and response handling.

    Supports context manager protocol for automatic resource cleanup.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        max_items: int = evaluation_settings["max_iterations"],
    ) -> None:
        """
        Initialize the Trismik client.

        Args:
            service_url: URL of the Trismik service. If not provided, uses
                the default endpoint or TRISMIK_SERVICE_URL environment
                 variable.
            api_key: API key for authentication. If not provided, reads from
                the TRISMIK_API_KEY environment variable.
            http_client: Custom HTTP client to use for requests. If not provided,
                a new client will be created automatically and managed by this
                instance.
            max_items: Maximum number of items to process in adaptive tests.
                Defaults to evaluation_settings["max_iterations"] (150).

        Raises:
            TrismikError: If api_key is not provided and not found in the
            environment.
            TrismikApiError: If API request fails.
        """
        self._service_url = TrismikUtils.option(
            service_url,
            client_settings["endpoint"],
            environment_settings["trismik_service_url"],
        )
        self._api_key = TrismikUtils.required_option(
            api_key, "api_key", environment_settings["trismik_api_key"]
        )

        # Set default headers with API key
        default_headers = {"x-api-key": self._api_key}

        # Track whether we own the client (created it vs user-provided)
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=self._service_url, headers=default_headers, timeout=30.0
        )
        self._max_items = max_items

    async def __aenter__(self) -> "TrismikAsyncClient":
        """
        Enter context manager.

        Returns the client instance for use in with-statement.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """
        Exit context manager and close client if owned.

        Automatically closes the HTTP client if it was created by this
        instance (not user-provided). Ensures proper resource cleanup.
        """
        await self.aclose()

    async def aclose(self) -> None:
        """
        Explicitly close the HTTP client if owned.

        Call this method when you're done with the client to ensure
        proper cleanup of resources. Only closes the client if it was
        created by this instance (not user-provided).

        If you use the client as a context manager, this is called
        automatically on exit.
        """
        if self._owns_client:
            await self._http_client.aclose()

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> Exception:
        """
        Handle HTTP errors and return appropriate Trismik exceptions.

        Args:
            e (httpx.HTTPStatusError): The HTTP status error to handle.

        Returns:
            Exception: The appropriate Trismik exception to raise.
        """
        if e.response.status_code == 413:
            # Handle payload too large error specifically
            try:
                backend_message = e.response.json().get("detail", "Payload too large.")
            except Exception:
                backend_message = "Payload too large."
            return TrismikPayloadTooLargeError(backend_message)
        elif e.response.status_code == 422:
            # Handle validation error specifically
            try:
                backend_message = e.response.json().get("detail", "Validation failed.")
            except Exception:
                backend_message = "Validation failed."
            return TrismikValidationError(backend_message)
        else:
            return TrismikApiError(TrismikUtils.get_error_message(e.response))

    async def list_datasets(self) -> List[TrismikDataset]:
        """
        Get a list of available datasets.

        Returns:
            List[TrismikDataset]: List of available datasets.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/datasets"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_datasets(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def start_run(
        self,
        dataset_id: str,
        project_id: str,
        experiment: str,
        metadata: Optional[TrismikRunMetadata] = None,
    ) -> TrismikRunResponse:
        """
        Start a new run for a dataset and get the first item.

        Args:
            dataset_id (str): ID of the dataset.
            project_id (str): ID of the project.
            experiment (str): Name of the experiment.
            metadata (Optional[TrismikRunMetadata]): Run metadata.

        Returns:
            TrismikRunResponse: Run response.

        Raises:
            TrismikPayloadTooLargeError: If the request payload exceeds the
            server's size limit.
            TrismikApiError: If API request fails.
        """
        try:
            url = "/runs/start"
            body = {
                "datasetId": dataset_id,
                "projectId": project_id,
                "experiment": experiment,
                "metadata": metadata.toDict() if metadata else {},
            }
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_run_response(json)
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def continue_run(self, run_id: str, item_choice_id: str) -> TrismikRunResponse:
        """
        Continue a run: respond to the current item and get the next one.

        Args:
            run_id (str): ID of the run.
            item_choice_id (str): ID of the chosen item response.

        Returns:
            TrismikRunResponse: Run response.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/runs/continue"
            body = {"itemChoiceId": item_choice_id, "runId": run_id}
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_run_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def run_summary(self, run_id: str) -> TrismikRunSummary:
        """
        Get run summary including responses, dataset, and state.

        Args:
            run_id (str): ID of the run.

        Returns:
            TrismikRunSummary: Complete run summary with responses,
                dataset, state, and metadata.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/runs/adaptive/{run_id}"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_run_summary(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def submit_replay(
        self,
        run_id: str,
        replay_request: TrismikReplayRequest,
        metadata: Optional[TrismikRunMetadata] = None,
    ) -> TrismikReplayResponse:
        """
        Submit a replay of a run with specific responses.

        Args:
            run_id (str): ID of the run to replay.
            replay_request (TrismikReplayRequest): Request containing responses
                to submit.
            metadata (Optional[TrismikRunMetadata]): Run metadata.

        Returns:
            TrismikReplayResponse: Response from the replay endpoint.

        Raises:
            TrismikPayloadTooLargeError: If the request payload exceeds the
                server's size limit.
            TrismikValidationError: If the request fails validation (e.g.,
                duplicate item IDs, unknown item IDs).
            TrismikApiError: If API request fails.
        """
        try:
            url = f"runs/{run_id}/replay"

            # Convert TrismikReplayRequestItem objects to dictionaries
            responses_dict = [
                {"itemId": item.itemId, "itemChoiceId": item.itemChoiceId}
                for item in replay_request.responses
            ]

            body = {
                "responses": responses_dict,
                "metadata": metadata.toDict() if metadata else {},
            }
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_replay_response(json)
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def me(self) -> TrismikMeResponse:
        """
        Get current user information.

        Returns:
            TrismikMeResponse: User information including validity and payload.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "../admin/api-keys/me"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_me_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def submit_classic_eval(
        self, classic_eval_request: TrismikClassicEvalRequest
    ) -> TrismikClassicEvalResponse:
        """
        Submit a classic evaluation run with pre-computed results.

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
        try:
            url = "/runs/classic"

            # Convert request object to dictionary
            items_dict = [
                {
                    "datasetItemId": item.datasetItemId,
                    "modelInput": item.modelInput,
                    "modelOutput": item.modelOutput,
                    "goldOutput": item.goldOutput,
                    "metrics": item.metrics,
                }
                for item in classic_eval_request.items
            ]

            metrics_dict = [
                {
                    "metricId": metric.metricId,
                    "valueType": TrismikUtils.metric_value_to_type(metric.value),
                    "value": metric.value,
                }
                for metric in classic_eval_request.metrics
            ]

            body = {
                "projectId": classic_eval_request.projectId,
                "experimentName": classic_eval_request.experimentName,
                "datasetId": classic_eval_request.datasetId,
                "modelName": classic_eval_request.modelName,
                "hyperparameters": classic_eval_request.hyperparameters,
                "items": items_dict,
                "metrics": metrics_dict,
            }

            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_classic_eval_response(json)
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def create_project(
        self,
        name: str,
        team_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> TrismikProject:
        """
        Create a new project.

        Args:
            name (str): Name of the project.
            team_id (Optional[str]): ID of the team to create the
                project in.
            description (Optional[str]): Optional description of the project.

        Returns:
            TrismikProject: Created project information.

        Raises:
            TrismikValidationError: If the request fails validation.
            TrismikApiError: If API request fails.
        """
        try:
            url = "../admin/public/projects"

            body = {
                "name": name,
            }
            if team_id is not None:
                body["teamId"] = team_id
            if description is not None:
                body["description"] = description

            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_project(json)
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    # ===== Test Orchestration Methods =====

    @overload
    async def run(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: Literal[True] = True,
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    async def run(  # noqa: E704
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: Literal[False] = False,
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    async def run(
        self,
        test_id: str,
        project_id: str,
        experiment: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Run an adaptive test.

        Args:
            test_id: ID of the test to run.
            project_id: ID of the project.
            experiment: Name of the experiment.
            run_metadata: Metadata for the run.
            item_processor: Function to process test items (can be sync or async).
            on_progress: Optional callback for progress updates (current, total).
            return_dict: If True, return dict instead of TrismikRunResults.
                Defaults to True.
            with_responses: If True, include responses in results.

        Returns:
            Test results as TrismikRunResults or dict.

        Raises:
            TrismikApiError: If API request fails.
            NotImplementedError: If with_responses=True (not yet implemented).
        """
        if with_responses:
            raise NotImplementedError("with_responses is not yet implemented for the new API flow")

        # Start run and get first item
        start_response = await self.start_run(test_id, project_id, experiment, run_metadata)

        # Initialize state tracking
        states: List[TrismikAdaptiveTestState] = []
        run_id = start_response.run_info.id
        states.append(
            TrismikAdaptiveTestState(
                run_id=run_id,
                state=start_response.state,
                completed=start_response.completed,
            )
        )

        # Run the test and get last state
        last_state = await self.run_test_loop(
            run_id,
            start_response.next_item,
            states,
            item_processor,
            on_progress,
        )

        if not last_state:
            raise RuntimeError("Test run completed but no final state was captured")

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

    async def run_test_loop(
        self,
        run_id: str,
        first_item: Optional[TrismikItem],
        states: List[TrismikAdaptiveTestState],
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Optional[TrismikAdaptiveTestState]:
        """
        Core test execution loop.

        This method contains the main test orchestration logic.
        The sync version is auto-generated by unasync.

        Args:
            run_id: ID of the run to execute.
            first_item: First item from run start.
            states: List to accumulate states.
            item_processor: Function to process test items.
            on_progress: Optional callback for progress updates.

        Returns:
            Last state of the run.

        Raises:
            TrismikApiError: If API request fails.
        """
        item = first_item
        current = 0

        while item is not None:
            # Report progress
            if on_progress:
                on_progress(current, self._max_items)

            # Process item with helper (handles both sync and async processors)
            response = await process_item(item_processor, item)

            # Continue run with response
            continue_response = await self.continue_run(run_id, response)

            # Update state tracking
            states.append(
                TrismikAdaptiveTestState(
                    run_id=run_id,
                    state=continue_response.state,
                    completed=continue_response.completed,
                )
            )

            current += 1

            if continue_response.completed:
                # Final progress update
                if on_progress:
                    on_progress(current, current)
                break

            item = continue_response.next_item

        return states[-1] if states else None

    @overload
    async def run_replay(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: Literal[True] = True,
        with_responses: bool = False,
    ) -> Dict[str, Any]: ...

    @overload
    async def run_replay(  # noqa: E704
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: Literal[False] = False,
        with_responses: bool = False,
    ) -> TrismikRunResults: ...

    async def run_replay(
        self,
        previous_run_id: str,
        run_metadata: TrismikRunMetadata,
        item_processor: Callable[[TrismikItem], Any],
        on_progress: Optional[Callable[[int, int], None]] = None,
        return_dict: bool = True,
        with_responses: bool = False,
    ) -> Union[TrismikRunResults, Dict[str, Any]]:
        """
        Replay the exact sequence of questions from a previous run.

        Args:
            previous_run_id: ID of a previous run to replay.
            run_metadata: Metadata for the replay run.
            item_processor: Function to process test items (can be sync or async).
            on_progress: Optional callback for progress updates (current, total).
            return_dict: If True, return dict instead of TrismikRunResults.
            with_responses: If True, include responses in results.

        Returns:
            Test results as TrismikRunResults or dict.

        Raises:
            TrismikApiError: If API request fails.
        """
        # Get the original run summary
        original_summary = await self.run_summary(previous_run_id)

        # Build replay request by processing each item
        replay_items = []
        total = len(original_summary.dataset)

        for idx, item in enumerate(original_summary.dataset):
            # Report progress
            if on_progress:
                on_progress(idx, total)

            # Process item with helper (handles both sync and async processors)
            response = await process_item(item_processor, item)

            # Create replay request item
            replay_item = TrismikReplayRequestItem(itemId=item.id, itemChoiceId=response)
            replay_items.append(replay_item)

        # Final progress update
        if on_progress:
            on_progress(total, total)

        # Create and submit replay request
        replay_request = TrismikReplayRequest(responses=replay_items)
        replay_response = await self.submit_replay(previous_run_id, replay_request, run_metadata)

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
