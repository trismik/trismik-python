from typing import List, Any, Optional

import httpx

from .exceptions import TrismikApiError
from .mapper import TrismikResponseMapper
from .types import (
    TrismikTest,
    TrismikAuthResponse,
    TrismikSession,
    TrismikItem,
    TrismikResult
)
from .utils import TrismikUtils


class Trismik:
    def __init__(
            self,
            service_url: Optional[str] = None,
            api_key: Optional[str] = None,
            http_client: Optional[httpx.Client] | None = None,
    ) -> None:
        self._service_url = TrismikUtils.required_option(
                service_url, "service_url", "TRISMIK_SERVICE_URL"
        )
        self._api_key = TrismikUtils.required_option(
                api_key, "api_key", "TRISMIK_API_KEY"
        )
        self._http_client = http_client or httpx.Client(
                base_url=self._service_url)

    def authenticate(self) -> TrismikAuthResponse:
        try:
            url = "/client/auth"
            body = {"apiKey": self._api_key}
            response = self._http_client.post(url, json=body)
            response.raise_for_status()
            return TrismikResponseMapper.to_auth_response(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def refresh_token(self, token: str) -> TrismikAuthResponse:
        try:
            url = "/client/token"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            return TrismikResponseMapper.to_auth_response(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def available_tests(self, token: str) -> List[TrismikTest]:
        try:
            url = "/client/tests"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            return TrismikResponseMapper.to_tests(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def create_session(self, test_id: str, token: str) -> TrismikSession:
        try:
            url = "/client/sessions"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            body = {
                "testId": test_id,
            }
            response = self._http_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return TrismikResponseMapper.to_session(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def current_item(
            self,
            session_url: str,
            token: str
    ) -> TrismikItem:
        try:
            url = f"{session_url}/item"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            return TrismikResponseMapper.to_item(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def respond_to_current_item(
            self,
            session_url: str,
            value: Any,
            token: str
    ) -> TrismikItem | None:
        try:
            url = f"{session_url}/item"
            body = {
                "value": value
            }
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self._http_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            else:
                return TrismikResponseMapper.to_item(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def results(self, session_url: str, token: str) -> List[TrismikResult]:
        try:
            url = f"{session_url}/results"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            return TrismikResponseMapper.to_results(response.json())
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

