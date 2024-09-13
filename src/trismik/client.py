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
    TrismikResult,
    TrismikResponse,
)


class TrismikClient:
    _serviceUrl: str = "https://trismik.e-psychometrics.com/api"

    def __init__(
            self,
            service_url: Optional[str] = None,
            api_key: Optional[str] = None,
            http_client: Optional[httpx.Client] | None = None,
    ) -> None:
        """
        Initializes a new Trismik client.

        Args:
            service_url (Optional[str]): URL of the Trismik service.
            api_key (Optional[str]): API key for the Trismik service.
            http_client (Optional[httpx.Client]): HTTP client to use for requests.

        Raises:
            TrismikError: If service_url or api_key are not provided and not found in environment.
            TrismikApiError: If API request fails.
        """
        self._service_url = TrismikUtils.option(
                service_url, self._serviceUrl, "TRISMIK_SERVICE_URL"
        )
        self._api_key = TrismikUtils.required_option(
                api_key, "api_key", "TRISMIK_API_KEY"
        )
        self._http_client = http_client or httpx.Client(
                base_url=self._service_url)

    def authenticate(self) -> TrismikAuth:
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
            response = self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def refresh_token(self, token: str) -> TrismikAuth:
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
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def available_tests(self, token: str) -> List[TrismikTest]:
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
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_tests(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def create_session(self, test_id: str, token: str) -> TrismikSession:
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
            body = {"testId": test_id, }
            response = self._http_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_session(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def current_item(
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
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_item(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def respond_to_current_item(
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
            response = self._http_client.post(url, headers=headers, json=body)
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

    def results(self, session_url: str, token: str) -> List[TrismikResult]:
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
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_results(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def responses(self,
            session_url: str,
            token: str
    ) -> List[TrismikResponse]:
        """
        Retrieves responses to session items.

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
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_responses(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e
