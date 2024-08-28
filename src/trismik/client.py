import os
from typing import List, Any, Optional

import httpx
from dateutil.parser import parse as parse_date

from .exceptions import TrismikError, TrismikApiError
from .types import (
    TrismikTest,
    TrismikAuthResponse,
    TrismikSession,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikTextChoice, TrismikResult
)


class Trismik:
    def __init__(
            self,
            service_url: Optional[str] = None,
            api_key: Optional[str] = None,
            http_client: Optional[httpx.Client] | None = None,
    ) -> None:
        self.service_url = self._validate_option(
                service_url, "service_url", "TRISMIK_SERVICE_URL"
        )
        self.api_key = self._validate_option(
                api_key, "api_key", "TRISMIK_API_KEY"
        )
        self.http_client = http_client or httpx.Client(
                base_url=self.service_url)

    def authenticate(self) -> TrismikAuthResponse:
        try:
            url = "/client/auth"
            body = {"apiKey": self.api_key}
            response = self.http_client.post(url, json=body)
            response.raise_for_status()
            return self._to_auth_response(response)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def refresh_token(self, token: str) -> TrismikAuthResponse:
        try:
            url = "/client/token"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self.http_client.get(url, headers=headers)
            response.raise_for_status()
            return self._to_auth_response(response)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def available_tests(self, token: str) -> List[TrismikTest]:
        try:
            url = "/client/tests"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self.http_client.get(url, headers=headers)
            response.raise_for_status()
            return self._to_tests(response)
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
            response = self.http_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._to_session(response)
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
            response = self.http_client.get(url, headers=headers)
            response.raise_for_status()
            return self._to_item(response)
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
            response = self.http_client.post(url, headers=headers, json=body)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            else:
                return self._to_item(response)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    def results(self, session_url: str, token: str) -> List[TrismikResult]:
        try:
            url = f"{session_url}/results"
            headers = {
                "Authorization": f"Bearer {token}"
            }
            response = self.http_client.get(url, headers=headers)
            response.raise_for_status()
            return self._to_results(response)
        except httpx.HTTPStatusError as e:
            raise TrismikApiError(response.json()["message"]) from e

    @staticmethod
    def _validate_option(
            value: str,
            name: str,
            env: str
    ) -> str:
        if value is None:
            value = os.environ.get(env)
        if value is None:
            raise TrismikError(
                    f"The {name} client option must be set either by passing "
                    f"{env} to the client or by setting the {env} "
                    "environment variable"
            )
        return value

    @staticmethod
    def _to_auth_response(response: httpx.Response) -> TrismikAuthResponse:
        data = response.json()
        return TrismikAuthResponse(
                token=data["token"],
                expires=parse_date(data["expires"]),
        )

    @staticmethod
    def _to_tests(response: httpx.Response) -> List[TrismikTest]:
        data = response.json()
        return [
            TrismikTest(
                    id=item["id"],
                    name=item["name"],
            ) for item in data
        ]

    @staticmethod
    def _to_session(response: httpx.Response) -> TrismikSession:
        data = response.json()
        return TrismikSession(
                id=data["id"],
                url=data["url"],
                status=data["status"],
        )

    @staticmethod
    def _to_item(response: httpx.Response) -> TrismikItem:
        data = response.json()
        if data["type"] == "multiple_choice_text":
            return TrismikMultipleChoiceTextItem(
                    question=data["question"],
                    choices=[
                        TrismikTextChoice(
                                id=choice["id"],
                                text=choice["text"],
                        ) for choice in data["choices"]
                    ]
            )
        else:
            raise TrismikApiError(
                    f"API has returned unrecognized item type: {data['type']}")

    @staticmethod
    def _to_results(response: httpx.Response) -> List[TrismikResult]:
        data = response.json()
        return [
            TrismikResult(
                    trait=item["trait"],
                    name=item["name"],
                    value=item["value"],
            ) for item in data
        ]
