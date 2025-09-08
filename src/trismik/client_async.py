"""
Trismik async client for interacting with the Trismik API.

This module provides an asynchronous client for interacting with the Trismik
API. It uses httpx for making HTTP requests.
"""

from typing import List, Optional

import httpx

from trismik._mapper import TrismikResponseMapper
from trismik._utils import TrismikUtils
from trismik.exceptions import (
    TrismikApiError,
    TrismikPayloadTooLargeError,
    TrismikValidationError,
)
from trismik.settings import client_settings, environment_settings
from trismik.types import (
    TrismikClassicEvalRequest,
    TrismikClassicEvalResponse,
    TrismikDataset,
    TrismikMeResponse,
    TrismikReplayRequest,
    TrismikReplayResponse,
    TrismikRunMetadata,
    TrismikRunResponse,
    TrismikRunSummary,
)


class TrismikAsyncClient:
    """
    Asynchronous client for the Trismik API.

    This class provides an asynchronous interface to interact with the Trismik
    API, handling authentication, dataset runs, and responses.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Initialize the Trismik async client.

        Args:
            service_url (Optional[str]): URL of the Trismik service.
            api_key (Optional[str]): API key for the Trismik service.
            http_client (Optional[httpx.AsyncClient]): HTTP client to use for
                requests.

        Raises:
            TrismikError: If service_url or api_key are not provided and not
                found in environment.
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

        self._http_client = http_client or httpx.AsyncClient(
            base_url=self._service_url, headers=default_headers
        )

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
                backend_message = e.response.json().get(
                    "detail", "Payload too large."
                )
            except Exception:
                backend_message = "Payload too large."
            return TrismikPayloadTooLargeError(backend_message)
        elif e.response.status_code == 422:
            # Handle validation error specifically
            try:
                backend_message = e.response.json().get(
                    "detail", "Validation failed."
                )
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
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
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

    async def continue_run(
        self, run_id: str, item_choice_id: str
    ) -> TrismikRunResponse:
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
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
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
            url = f"/runs/{run_id}"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_run_summary(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
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
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
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
                    "valueType": TrismikUtils.metric_value_to_type(
                        metric.value
                    ),
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
