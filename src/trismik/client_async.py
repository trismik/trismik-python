"""
Trismik async client for interacting with the Trismik API.

This module provides an asynchronous client for interacting with the Trismik
API. It uses httpx for making HTTP requests.
"""

from typing import List, Optional

import httpx

from trismik._mapper import TrismikResponseMapper
from trismik._utils import TrismikUtils
from trismik.exceptions import TrismikApiError
from trismik.settings import client_settings, environment_settings
from trismik.types import (
    TrismikReplayRequest,
    TrismikReplayResponse,
    TrismikResult,
    TrismikSession,
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
        self, test_id: str, metadata: TrismikSessionMetadata
    ) -> TrismikSessionResponse:
        """
        Start a new session for a test and get the first item.

        Args:
            test_id (str): ID of the test.
            metadata (TrismikSessionMetadata): Metadata for the session.

        Returns:
            TrismikSessionResponse: Session response.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/sessions/start"
            body = {"testId": test_id}
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

    async def create_replay_session(
        self,
        previous_session_id: str,
        metadata: TrismikSessionMetadata,
    ) -> TrismikSession:
        """
        Create a new session that replays a previous session.

        Create a new session that replays exactly the question sequence of a
        previous session.

        Args:
            previous_session_id (str): Session id of the session to replay.
            metadata (TrismikSessionMetadata): Metadata for the session.

        Returns:
            TrismikSession: New session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/sessions/replay"
            body = {
                "previousSessionToken": previous_session_id,
                "metadata": metadata.toDict(),
            }
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def add_metadata(
        self, session_id: str, metadata: TrismikSessionMetadata
    ) -> None:
        """
        Add metadata to the session, merging it with any already stored.

        Args:
            session_id (str): ID of the session object.
            metadata (TrismikSessionMetadata): Object containing the metadata
                to add.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/client/sessions/{session_id}/metadata"
            body = metadata.toDict()
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def results(self, session_url: str) -> List[TrismikResult]:
        """
        Get the results of a session.

        Args:
            session_url (str): URL of the session.

        Returns:
            List[TrismikResult]: Results of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"{session_url}/results"
            response = await self._http_client.get(url)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_results(json)
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
        self, session_id: str, replay_request: TrismikReplayRequest
    ) -> TrismikReplayResponse:
        """
        Submit a replay of a session with specific responses.

        Args:
            session_id (str): ID of the session to replay.
            replay_request (TrismikReplayRequest): Request containing responses
                to submit.

        Returns:
            TrismikReplayResponse: Response from the replay endpoint.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/api/v1/sessions/{session_id}/replay"

            # Convert TrismikReplayRequestItem objects to dictionaries
            responses_dict = [
                {"itemId": item.itemId, "itemChoiceId": item.itemChoiceId}
                for item in replay_request.responses
            ]

            body = {"responses": responses_dict}
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_replay_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e
