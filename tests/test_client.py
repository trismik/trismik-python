from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

from trismik import (
    Trismik,
    TrismikError,
    TrismikApiError,
    TrismikMultipleChoiceTextItem,
)
from ._mocker import TrismikResponseMocker


class TestTrismik:

    def test_initialization_with_explicit_values(self) -> None:
        Trismik(
                service_url="service_url",
                api_key="api_key",
        )

    def test_initialization_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv('TRISMIK_SERVICE_URL', 'service_url')
        monkeypatch.setenv('TRISMIK_API_KEY', 'api_key')
        Trismik(
                service_url=None,
                api_key=None,
        )

    def test_initialization_when_service_url_not_provided(
            self,
            monkeypatch
    ) -> None:
        monkeypatch.delenv('TRISMIK_SERVICE_URL', raising=False)
        with pytest.raises(TrismikError,
                           match="service_url client option must be set"):
            Trismik(
                    service_url=None,
                    api_key="api_key"
            )

    def test_initialization_when_api_key_not_provided(
            self,
            monkeypatch
    ) -> None:
        monkeypatch.delenv('TRISMIK_API_KEY', raising=False)
        with pytest.raises(TrismikError,
                           match="api_key client option must be set"):
            Trismik(
                    service_url="service_url",
                    api_key=None,
            )

    def test_authenticate(self) -> None:
        client = Trismik(http_client=self._mock_auth_response())
        response = client.authenticate()
        assert response.token == "token"
        assert response.expires == datetime(2024, 8, 28, 14, 18, 10, 92400)

    def test_authenticate_when_failed(self) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = Trismik(http_client=self._mock_error_response(401))
            client.authenticate()

    def test_refresh_token(self) -> None:
        client = Trismik(http_client=self._mock_auth_response())
        response = client.refresh_token("token")
        assert response.token == "token"
        assert response.expires == datetime(2024, 8, 28, 14, 18, 10, 92400)

    def test_refresh_token_when_failed(self) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = Trismik(http_client=self._mock_error_response(401))
            client.refresh_token("token")

    def test_available_tests(self) -> None:
        client = Trismik(http_client=self._mock_tests_response())
        tests = client.available_tests("token")
        assert len(tests) == 5
        assert tests[0].id == "fluency"
        assert tests[0].name == "Fluency"

    def test_available_tests_when_failed(self) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = Trismik(http_client=self._mock_error_response(401))
            client.available_tests("token")

    def test_create_session(self) -> None:
        client = Trismik(http_client=self._mock_session_response())
        session = client.create_session("fluency", "token")
        assert session.id == "id"
        assert session.url == "url"
        assert session.status == "status"

    def test_create_session_when_failed(self) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = Trismik(http_client=self._mock_error_response(401))
            client.create_session("fluency", "token")

    def test_current_item(self) -> None:
        client = Trismik(http_client=self._mock_item_response())
        item = client.current_item("url", "token")
        assert item.question == "question"
        assert isinstance(item, TrismikMultipleChoiceTextItem)
        assert len(item.choices) == 3
        assert item.choices[0].id == "choice_id_1"
        assert item.choices[0].text == "choice_text_1"

    def test_respond_to_current_item(self) -> None:
        client = Trismik(http_client=self._mock_item_response())
        item = client.respond_to_current_item("url", "choice_id_1", "token")
        assert item.question == "question"
        assert isinstance(item, TrismikMultipleChoiceTextItem)
        assert len(item.choices) == 3
        assert item.choices[0].id == "choice_id_1"
        assert item.choices[0].text == "choice_text_1"

    def test_respond_to_current_item_when_finished(self) -> None:
        client = Trismik(http_client=self._mock_no_content_response())
        item = client.respond_to_current_item("url", "choice_id_1", "token")
        assert item is None

    @pytest.fixture(scope='function', autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv('TRISMIK_SERVICE_URL', 'service_url')
        monkeypatch.setenv('TRISMIK_API_KEY', 'api_key')

    @staticmethod
    def _mock_auth_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.auth_response()
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_tests_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.tests_response()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_session_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.session_response()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_item_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.item_response()
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_error_response(status) -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.error_response(status)
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_no_content_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.no_content_response()
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client
