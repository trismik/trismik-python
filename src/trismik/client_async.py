"""
Trismik async client for interacting with the Trismik API.

This module provides an asynchronous client for interacting with the Trismik
API. It uses httpx for making HTTP requests.
"""

from typing import Any, List, Optional

import httpx

from trismik._mapper import TrismikResponseMapper
from trismik._utils import TrismikUtils
from trismik.exceptions import TrismikApiError
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikResponse,
    TrismikResult,
    TrismikSession,
    TrismikSessionMetadata,
    TrismikTest,
)


class TrismikAsyncClient:
    """
    Asynchronous client for the Trismik API.

    This class provides an asynchronous interface to interact with the Trismik
    API, handling authentication, test sessions, and responses.
    """

    _serviceUrl: str = "https://zoo-dashboard.trismik.com/api"

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
            http_client (Optional[httpx.Client]): HTTP client to use for
                requests.

        Raises:
            TrismikError: If service_url or api_key are not provided and not
                found in environment.
            TrismikApiError: If API request fails.
        """
        self._service_url = TrismikUtils.option(
            service_url, self._serviceUrl, "TRISMIK_SERVICE_URL"
        )
        self._api_key = TrismikUtils.required_option(
            api_key, "api_key", "TRISMIK_API_KEY"
        )
        self._http_client = http_client or httpx.AsyncClient(
            base_url=self._service_url
        )

    async def authenticate(self) -> TrismikAuth:
        """
        Authenticate with the Trismik service.

        Returns:
            TrismikAuth: Authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/auth"
            body = {"apiKey": self._api_key}
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def refresh_token(self, token: str) -> TrismikAuth:
        """
        Refresh the authentication token.

        Args:
            token (str): Current authentication token.

        Returns:
            TrismikAuth: New authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/token"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def available_tests(self, token: str) -> List[TrismikTest]:
        """
        Get a list of available tests.

        Args:
            token (str): Authentication token.

        Returns:
            List[TrismikTest]: List of available tests.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/tests"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_tests(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def create_session(
        self, test_id: str, metadata: TrismikSessionMetadata, token: str
    ) -> TrismikSession:
        """
        Create a new session for a test.

        Args:
            test_id (str): ID of the test.
            metadata (TrismikSessionMetadata): Metadata for the session.
            token (str): Authentication token.

        Returns:
            TrismikSession: New session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/sessions"
            headers = {"Authorization": f"Bearer {token}"}
            body = {"testId": test_id, "metadata": metadata.toDict()}
            response = await self._http_client.post(
                url, headers=headers, json=body
            )
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session(json)
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
        token: str,
    ) -> TrismikSession:
        """
        Create a new session that replays a previous session.

        Create a new session that replays exactly the question sequence of a
        previous session.

        Args:
            previous_session_id (str): Session id of the session to replay.
            metadata (TrismikSessionMetadata): Metadata for the session.
            token (str): Authentication token.

        Returns:
            TrismikSession: New session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/sessions/replay"
            headers = {"Authorization": f"Bearer {token}"}
            body = {
                "previousSessionToken": previous_session_id,
                "metadata": metadata.toDict(),
            }
            response = await self._http_client.post(
                url, headers=headers, json=body
            )
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
        self, session_id: str, metadata: TrismikSessionMetadata, token: str
    ) -> None:
        """
        Add metadata to the session, merging it with any already stored.

        Args:
            session_id (str): ID of the session object.
            metadata (TrismikSessionMetadata): Object containing the metadata
                to add.
            token (str): Authentication token.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"/client/sessions/{session_id}/metadata"
            headers = {"Authorization": f"Bearer {token}"}
            body = metadata.toDict()
            response = await self._http_client.post(
                url, headers=headers, json=body
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def current_item(self, session_url: str, token: str) -> TrismikItem:
        """
        Get the current test item.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            TrismikItem: Current test item.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"{session_url}/item"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_item(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def respond_to_current_item(
        self, session_url: str, value: Any, token: str
    ) -> Optional[TrismikItem]:
        """
        Respond to the current test item.

        Args:
            session_url (str): URL of the session.
            value (Any): Response value.
            token (str): Authentication token.

        Returns:
            Optional[TrismikItem]: Next test item or None if session is
            finished.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"{session_url}/item"
            body = {"value": value}
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.post(
                url, headers=headers, json=body
            )
            response.raise_for_status()
            if response.status_code == 204:
                return None
            else:
                json = response.json()
                return TrismikResponseMapper.to_item(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def results(
        self, session_url: str, token: str
    ) -> List[TrismikResult]:
        """
        Get the results of a session.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            List[TrismikResult]: Results of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"{session_url}/results"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_results(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def responses(
        self, session_url: str, token: str
    ) -> List[TrismikResponse]:
        """
        Get responses to session items.

        Args:
            session_url (str): URL of the session.
            token (str): Authentication token.

        Returns:
            List[TrismikResponse]: Responses of the session.

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = f"{session_url}/responses"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_responses(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                TrismikUtils.get_error_message(e.response)
            ) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e
