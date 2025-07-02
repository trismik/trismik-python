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
    TrismikReplayRequest,
    TrismikReplayResponse,
    TrismikSessionMetadata,
    TrismikSessionResponse,
    TrismikSessionSummary,
    TrismikTest,
)


class TrismikAsyncClient:
    """
    Asynchronous client for the Trismik API.

    This class provides an asynchronous interface to interact with the Trismik
    API, handling authentication, test sessions, and responses.
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

    async def list_tests(self) -> List[TrismikTest]:
        """
        Get a list of available tests.

        Returns:
            List[TrismikTest]: List of available tests.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/test/summary"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_tests(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def start_session(
        self, test_id: str, metadata: Optional[TrismikSessionMetadata] = None
    ) -> TrismikSessionResponse:
        """
        Start a new session for a test and get the first item.

        Args:
            test_id (str): ID of the test.
            metadata (Optional[TrismikSessionMetadata]): Session metadata.

        Returns:
            TrismikSessionResponse: Session response.

        Raises:
            TrismikPayloadTooLargeError: If the request payload exceeds the
            server's size limit.
            TrismikApiError: If API request fails.
        """
        try:
            url = "/sessions/start"
            body = {
                "testId": test_id,
                "metadata": metadata.toDict() if metadata else {},
            }
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session_response(json)
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def continue_session(
        self, session_id: str, item_choice_id: str
    ) -> TrismikSessionResponse:
        """
        Continue a session: respond to the current item and get the next one.

        Args:
            session_id (str): ID of the session.
            item_choice_id (str): ID of the chosen item response.

        Returns:
            TrismikSessionResponse: Session response.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/sessions/{session_id}/continue"
            body = {"itemChoiceId": item_choice_id}
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def session_summary(self, session_id: str) -> TrismikSessionSummary:
        """
        Get session summary including responses, dataset, and state.

        Args:
            session_id (str): ID of the session.

        Returns:
            TrismikSessionSummary: Complete session summary with responses,
                dataset, state, and metadata.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/sessions/{session_id}/summary"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session_summary(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def submit_replay(
        self,
        session_id: str,
        replay_request: TrismikReplayRequest,
        metadata: Optional[TrismikSessionMetadata] = None,
    ) -> TrismikReplayResponse:
        """
        Submit a replay of a session with specific responses.

        Args:
            session_id (str): ID of the session to replay.
            replay_request (TrismikReplayRequest): Request containing responses
                to submit.
            metadata (Optional[TrismikSessionMetadata]): Session metadata.

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
            url = f"sessions/{session_id}/replay"

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
