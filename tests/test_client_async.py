from unittest.mock import MagicMock

import httpx
import pytest

from trismik.client_async import TrismikAsyncClient
from trismik.exceptions import TrismikApiError, TrismikError
from trismik.types import TrismikSessionMetadata

from ._mocker import TrismikResponseMocker


class TestTrismikAsyncClient:

    def test_should_initialize_with_explicit_values(self) -> None:
        TrismikAsyncClient(
            service_url="service_url",
            api_key="api_key",
        )

    def test_should_initialize_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TRISMIK_SERVICE_URL", "service_url")
        monkeypatch.setenv("TRISMIK_API_KEY", "api_key")
        TrismikAsyncClient(
            service_url=None,
            api_key=None,
        )

    def test_should_initialize_with_default_service_url(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("TRISMIK_SERVICE_URL", raising=False)
        TrismikAsyncClient(service_url=None, api_key="api_key")

    def test_should_fail_initialize_when_api_key_not_provided(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("TRISMIK_API_KEY", raising=False)
        with pytest.raises(
            TrismikError, match="api_key client option must be set"
        ):
            TrismikAsyncClient(
                service_url="service_url",
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_should_list_tests(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_tests_response())
        tests = await client.list_tests()
        assert len(tests) == 5
        assert tests[0].id == "fluency"
        assert tests[0].name == "Fluency"

    @pytest.mark.asyncio
    async def test_should_fail_list_tests_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.list_tests()

    @pytest.mark.asyncio
    async def test_should_start_session(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_session_start_response()
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        response = await client.start_session("test_id", metadata)
        assert response.session_info.id == "session_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_1"
        assert len(response.state.thetas) == 1

    @pytest.mark.asyncio
    async def test_should_fail_start_session_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            metadata = TrismikSessionMetadata(
                model_metadata=TrismikSessionMetadata.ModelMetadata(
                    name="test_model"
                ),
                test_configuration={},
                inference_setup={},
            )
            await client.start_session("test_id", metadata)

    @pytest.mark.asyncio
    async def test_should_continue_session(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_session_continue_response()
        )
        response = await client.continue_session("session_id", "choice_1")
        assert response.session_info.id == "session_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_2"
        assert len(response.state.thetas) == 2

    @pytest.mark.asyncio
    async def test_should_end_session_on_continue(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_session_end_response()
        )
        response = await client.continue_session("session_id", "choice_2")
        assert response.session_info.id == "session_id"
        assert response.completed is True
        assert response.next_item is None
        assert len(response.state.thetas) == 3

    @pytest.mark.asyncio
    async def test_should_fail_continue_session_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.continue_session("session_id", "choice_1")

    @pytest.mark.asyncio
    async def test_should_get_results(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_results_response())
        results = await client.results("url")
        assert len(results) == 1
        assert results[0].trait == "trait"
        assert results[0].name == "name"
        assert results[0].value == "value"

    @pytest.mark.asyncio
    async def test_should_fail_get_results_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.results("url")

    @pytest.mark.asyncio
    async def test_should_get_session_summary(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_session_summary_response()
        )
        summary = await client.session_summary("session_id")

        # Check id and test_id
        assert summary.id == "session_id"
        assert summary.test_id == "test_id"

        # Check state
        assert len(summary.state.thetas) == 1
        assert summary.state.thetas[0] == 1.0

        # Check responses
        assert len(summary.responses) == 1
        assert summary.responses[0].dataset_item_id == "item_id"
        assert summary.responses[0].value == "value"
        assert summary.responses[0].correct is True

        # Check dataset
        assert len(summary.dataset) == 1
        assert summary.dataset[0].id == "item_id"
        assert summary.dataset[0].question == "Test question"
        assert len(summary.dataset[0].choices) == 2

        # Check completion status
        assert summary.completed is True
        # Check metadata
        assert summary.metadata == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_should_fail_get_session_summary_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.session_summary("session_id")

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TRISMIK_SERVICE_URL", "service_url")
        monkeypatch.setenv("TRISMIK_API_KEY", "api_key")

    @staticmethod
    def _mock_tests_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.tests()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_session_start_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.session_start()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_session_continue_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.session_continue()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_session_end_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.session_end()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_error_response(status: int) -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.error(status)
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_results_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.results()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_session_summary_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.session_summary()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_no_content_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.no_content()
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client
