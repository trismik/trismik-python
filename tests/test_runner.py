from datetime import datetime, timedelta
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from trismik.client import TrismikClient
from trismik.runner import TrismikRunner
from trismik.types import (
    TrismikAuth, TrismikItem, TrismikMultipleChoiceTextItem, TrismikResult, TrismikResponse, TrismikRunResults, TrismikSession, TrismikTextChoice, TrismikSessionMetadata 
)


class TestTrismikRunner:

    # noinspection PyUnresolvedReferences
    def test_should_run_test(self, runner, mock_client):
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={}
        )
        results = runner.run("test_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with("test_id", metadata, "token")
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
                "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    # noinspection PyUnresolvedReferences
    def test_should_run_test_with_responses(self, runner, mock_client):
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={}
        )
        results = runner.run("test_id", metadata, with_responses=True)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with("test_id", metadata, "token")
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
                "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        mock_client.responses.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1
        assert len(results.responses) == 1

    # noinspection PyUnresolvedReferences
    def test_should_authenticate_itself_when_auth_was_not_provided(
            self,
            item_processor,
            mock_client
    ) -> None:
        runner = TrismikRunner(
                item_processor=item_processor,
                client=mock_client,
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={}
        )
        runner.run("test_id", metadata)

        mock_client.authenticate.assert_called_once()

    # noinspection PyUnresolvedReferences
    def test_should_refresh_token_when_close_to_expiration(
            self,
            item_processor,
            mock_client
    ) -> None:
        runner = TrismikRunner(
                item_processor=item_processor,
                client=mock_client,
                auth=TrismikAuth(
                        token="token",
                        expires=datetime.now() + timedelta(minutes=1)
                )
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={}
        )
        runner.run("test_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_called_once_with("token")

    @pytest.fixture
    def item(self) -> TrismikItem:
        return TrismikMultipleChoiceTextItem(
                id="id",
                question="question",
                choices=[
                    TrismikTextChoice(id="id", text="text")
                ]
        )

    @pytest.fixture
    def mock_client(self, auth, item) -> TrismikClient:
        client = MagicMock(spec=TrismikClient)
        client.authenticate.return_value = auth
        client.refresh_token.return_value = auth
        client.create_session.return_value = TrismikSession(
                id="id",
                url="url",
                status="status"
        )
        client.current_item.return_value = item
        client.respond_to_current_item.side_effect = [item, None]
        client.results.return_value = [
            TrismikResult(trait="example", name="test", value="value")
        ]
        client.responses.return_value = [
            TrismikResponse(item_id="id", value="value", score=1.0)
        ]
        return client

    @pytest.fixture
    def item_processor(self) -> Callable[[TrismikItem], Any]:
        def processor(_: TrismikItem) -> Any:
            return "processed_response"

        return processor

    @pytest.fixture
    def auth(self) -> TrismikAuth:
        return TrismikAuth(
                token="token",
                expires=datetime.now() + timedelta(hours=1)
        )

    @pytest.fixture
    def runner(self, mock_client, auth, item_processor) -> TrismikRunner:
        return TrismikRunner(
                item_processor=item_processor,
                client=mock_client,
                auth=auth
        )
