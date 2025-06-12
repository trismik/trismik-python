from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trismik.client import TrismikClient
from trismik.runner import TrismikRunner
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResponse,
    TrismikResult,
    TrismikRunResults,
    TrismikSession,
    TrismikSessionMetadata,
    TrismikTextChoice,
)


class TestTrismikRunner:
    def test_run_delegates_and_returns_results(self, runner):
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = runner.run("test_id", metadata)
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    def test_run_with_responses_delegates_and_returns_results(self, runner):
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = runner.run("test_id", metadata, with_responses=True)
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    @pytest.fixture
    def item(self) -> TrismikItem:
        return TrismikMultipleChoiceTextItem(
            id="id",
            question="question",
            choices=[TrismikTextChoice(id="id", text="text")],
        )

    @pytest.fixture
    def mock_client(self, auth, item) -> TrismikClient:
        client = MagicMock(spec=TrismikClient)
        client.service_url = "http://test.service.url"
        client.api_key = "test_api_key"
        client.authenticate.return_value = auth
        client.refresh_token.return_value = auth
        client.create_session.return_value = TrismikSession(
            id="id", url="url", status="status"
        )
        client.current_item.return_value = item
        client.respond_to_current_item.side_effect = [item, None]
        client.results.return_value = [
            TrismikResult(trait="example", name="test", value="value")
        ]
        client.responses.return_value = [
            TrismikResponse(item_id="id", value="value", score=1.0)
        ]
        async_client_mock = MagicMock()
        async_client_mock._service_url = "http://test.service.url"
        async_client_mock._api_key = "test_api_key"
        client._async_client = async_client_mock
        return client

    @pytest.fixture
    def auth(self) -> TrismikAuth:
        return TrismikAuth(
            token="token", expires=datetime.now() + timedelta(hours=1)
        )

    @pytest.fixture
    def runner(self, mock_client, auth) -> TrismikRunner:
        return TrismikRunner(
            item_processor=lambda _: "processed_response",
            client=mock_client,
            auth=auth,
        )

    @pytest.fixture(autouse=True)
    def patch_async_client(self, auth, item):
        with (
            patch("trismik.runner.TrismikAsyncClient") as AsyncClientMock,
            patch("trismik.runner_async.TrismikAsyncRunner") as AsyncRunnerMock,
        ):
            async_client_instance = MagicMock()
            async_client_instance.authenticate = AsyncMock(return_value=auth)
            async_client_instance.refresh_token = AsyncMock(return_value=auth)
            async_client_instance.create_session = AsyncMock(
                return_value=TrismikSession(id="id", url="url", status="status")
            )
            async_client_instance.current_item = AsyncMock(return_value=item)
            async_client_instance.respond_to_current_item = AsyncMock(
                side_effect=[item, None]
            )
            async_client_instance.results = AsyncMock(
                return_value=[
                    TrismikResult(trait="example", name="test", value="value")
                ]
            )
            async_client_instance.responses = AsyncMock(
                return_value=[
                    TrismikResponse(item_id="id", value="value", score=1.0)
                ]
            )
            AsyncClientMock.return_value = async_client_instance

            async_runner_instance = MagicMock()
            async_runner_instance.run = AsyncMock(
                return_value=TrismikRunResults(
                    session_id="id",
                    results=[
                        TrismikResult(
                            trait="example", name="test", value="value"
                        )
                    ],
                )
            )
            async_runner_instance.run_replay = AsyncMock(
                return_value=TrismikRunResults(
                    session_id="id",
                    results=[
                        TrismikResult(
                            trait="example", name="test", value="value"
                        )
                    ],
                )
            )
            AsyncRunnerMock.return_value = async_runner_instance
            yield
