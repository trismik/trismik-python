from typing import List, Any, Optional

import httpx

from .exceptions import TrismikApiError
from ._mapper import TrismikResponseMapper
from .types import (
    TrismikTest,
    TrismikAuthResponse,
    TrismikSession,
    TrismikItem,
    TrismikResult
)
from ._utils import TrismikUtils


class TrismikAsync:
    def __init__(
            self,
            service_url: Optional[str] = None,
            api_key: Optional[str] = None,
            http_client: Optional[httpx.AsyncClient] | None = None,
    ) -> None:
        self._service_url = TrismikUtils.required_option(
                service_url, "service_url", "TRISMIK_SERVICE_URL"
        )
        self._api_key = TrismikUtils.required_option(
                api_key, "api_key", "TRISMIK_API_KEY"
        )
        self._http_client = http_client or httpx.Client(
                base_url=self._service_url)

    async def authenticate(self) -> TrismikAuthResponse:
        try:
            url = "/client/auth"
            body = {"apiKey": self._api_key}
            response = await self._http_client.post(url, json=body)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def refresh_token(self, token: str) -> TrismikAuthResponse:
        try:
            url = "/client/token"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_auth_response(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    async def available_tests(self, token: str) -> List[TrismikTest]:
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

    async def results(self, session_url: str, token: str) -> List[
        TrismikResult]:
        try:
            url = f"{session_url}/results"
            headers = { "Authorization": f"Bearer {token}" }
            response = await self._http_client.get(url, headers=headers)
            response.raise_for_status()
            json = response.json()
            return TrismikResponseMapper.to_results(json)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(
                    TrismikUtils.get_error_message(e.response)) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e