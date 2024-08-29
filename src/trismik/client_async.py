from typing import List, Any, Optional

import httpx

from ._mapper import TrismikResponseMapper
from ._utils import TrismikUtils
from .exceptions import TrismikApiError
from .types import (
    TrismikTest,
    TrismikAuth,
    TrismikSession,
    TrismikItem,
    TrismikResult
)


class TrismikAsyncClient:
    def __init__(
            self,
            service_url: Optional[str] = None,
            api_key: Optional[str] = None,
            http_client: Optional[httpx.AsyncClient] | None = None,
    ) -> None:
        """
        Initializes a new Trismik client (async version).

        Args:
            service_url (Optional[str]): URL of the Trismik service.
            api_key (Optional[str]): API key for the Trismik service.
            http_client (Optional[httpx.Client]): HTTP client to use for requests.

        Raises:
            TrismikError: If service_url or api_key are not provided and not found in environment.
            TrismikApiError: If API request fails.
        """
        self._service_url = TrismikUtils.required_option(
                service_url, "service_url", "TRISMIK_SERVICE_URL"
        )
        self._api_key = TrismikUtils.required_option(
                api_key, "api_key", "TRISMIK_API_KEY"
        )
        self._http_client = http_client or httpx.AsyncClient(
                base_url=self._service_url)

    async def authenticate(self) -> TrismikAuth:
        """
        Authenticates with the Trismik service.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def refresh_token(self, token: str) -> TrismikAuth:
        """
        Refreshes the authentication token.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def available_tests(self, token: str) -> List[TrismikTest]:
        """
        Retrieves a list of available tests.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def create_session(self, test_id: str, token: str) -> TrismikSession:
        """
        Creates a new session for a test.

        Args:
            test_id (str): ID of the test.
            token (str): Authentication token.

        Returns:
            TrismikSession: New session

        Raises:
            TrismikApiError: If API request fails.
        """
        try:
            url = "/client/sessions"
            headers = {"Authorization": f"Bearer {token}"}
            body = {"testId": test_id}
            response = await self._http_client.post(url, headers=headers,
                                                    json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def current_item(
            self,
            session_url: str,
            token: str
    ) -> TrismikItem:
        """
        Retrieves the current test item.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def respond_to_current_item(
            self,
            session_url: str,
            value: Any,
            token: str
    ) -> TrismikItem | None:
        """
        Responds to the current test item.

        Args:
            session_url (str): URL of the session.
            value (Any): Response value.
            token (str): Authentication token.

        Returns:
            TrismikItem | None: Next test item or None if session is finished.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def results(self,
            session_url: str,
            token: str
    ) -> List[TrismikResult]:
        """
        Retrieves the results of a session.

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
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e
