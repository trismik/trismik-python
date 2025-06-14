"""
Tests for the AdaptiveTest class.

This module tests both synchronous and asynchronous functionality of the
AdaptiveTest class.
"""

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable
from unittest.mock import MagicMock

import pytest

from trismik.adaptive_test import AdaptiveTest
from trismik.client_async import TrismikAsyncClient
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


class TestAdaptiveTest:
    """Test suite for the AdaptiveTest class."""

    @pytest.fixture
    def item(self) -> TrismikItem:
        """Create a test item."""
        return TrismikMultipleChoiceTextItem(
            id="id",
            question="question",
            choices=[TrismikTextChoice(id="id", text="text")],
        )

    @pytest.fixture
    def mock_client(self, auth, item) -> TrismikAsyncClient:
        """Create a mock async client."""
        client = MagicMock(spec=TrismikAsyncClient)
        client.authenticate.return_value = auth
        client.refresh_token.return_value = auth

        # Create a session with a real string URL
        session = TrismikSession(id="id", url="url", status="status")
        client.create_session.return_value = session
        client.create_replay_session.return_value = session

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
    def auth(self) -> TrismikAuth:
        """Create a test auth token."""
        return TrismikAuth(
            token="token", expires=datetime.now() + timedelta(hours=1)
        )

    @pytest.fixture
    def sync_item_processor(self) -> Callable[[TrismikItem], Any]:
        """Create a synchronous item processor."""

        def processor(_: TrismikItem) -> Any:
            return "processed_response"

        return processor

    @pytest.fixture
    def async_item_processor(self) -> Callable[[TrismikItem], Awaitable[Any]]:
        """Create an asynchronous item processor."""

        async def processor(_: TrismikItem) -> Any:
            return "processed_response"

        return processor

    @pytest.fixture
    def sync_runner(
        self, sync_item_processor, mock_client, auth
    ) -> AdaptiveTest:
        """Create a test instance with sync item processor."""
        return AdaptiveTest(
            item_processor=sync_item_processor,
            client=mock_client,
            auth=auth,
        )

    @pytest.fixture
    def async_runner(
        self, async_item_processor, mock_client, auth
    ) -> AdaptiveTest:
        """Create a test instance with async item processor."""
        return AdaptiveTest(
            item_processor=async_item_processor,
            client=mock_client,
            auth=auth,
        )

    def test_run_sync(self, sync_runner, mock_client):
        """Test running a test synchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run("test_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with(
            "test_id", metadata, "token"
        )
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
            "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    @pytest.mark.asyncio
    async def test_run_async(self, async_runner, mock_client):
        """Test running a test asynchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_async("test_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with(
            "test_id", metadata, "token"
        )
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
            "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    def test_run_with_responses_sync(self, sync_runner, mock_client):
        """Test running a test with responses synchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run("test_id", metadata, with_responses=True)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with(
            "test_id", metadata, "token"
        )
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

    @pytest.mark.asyncio
    async def test_run_with_responses_async(self, async_runner, mock_client):
        """Test running a test with responses asynchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_async(
            "test_id", metadata, with_responses=True
        )

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_session.assert_called_once_with(
            "test_id", metadata, "token"
        )
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

    def test_run_replay_sync(self, sync_runner, mock_client):
        """Test replaying a test synchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay("previous_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_replay_session.assert_called_once_with(
            "previous_id", metadata, "token"
        )
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
            "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    @pytest.mark.asyncio
    async def test_run_replay_async(self, async_runner, mock_client):
        """Test replaying a test asynchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_replay_async("previous_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_not_called()
        mock_client.create_replay_session.assert_called_once_with(
            "previous_id", metadata, "token"
        )
        mock_client.current_item.assert_called_once_with("url", "token")
        mock_client.respond_to_current_item.assert_called_with(
            "url", "processed_response", "token"
        )
        assert mock_client.respond_to_current_item.call_count == 2
        mock_client.results.assert_called_once_with("url", "token")
        assert isinstance(results, TrismikRunResults)
        assert len(results.results) == 1

    def test_should_authenticate_itself_when_auth_was_not_provided(
        self, sync_item_processor, mock_client
    ):
        """Test that the runner authenticates itself when no auth provided."""
        runner = AdaptiveTest(
            item_processor=sync_item_processor,
            client=mock_client,
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        runner.run("test_id", metadata)

        mock_client.authenticate.assert_called_once()

    def test_should_refresh_token_when_close_to_expiration(
        self, sync_item_processor, mock_client
    ):
        """Test that the runner refreshes the token when close to expiration."""
        runner = AdaptiveTest(
            item_processor=sync_item_processor,
            client=mock_client,
            auth=TrismikAuth(
                token="token", expires=datetime.now() + timedelta(minutes=1)
            ),
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        runner.run("test_id", metadata)

        mock_client.authenticate.assert_not_called()
        mock_client.refresh_token.assert_called_once_with("token")

    def test_max_items_parameter(self, sync_runner, mock_client):
        """Test that the max_items parameter is respected."""
        runner = AdaptiveTest(
            item_processor=sync_runner._item_processor,
            client=mock_client,
            auth=sync_runner._auth,
            max_items=2,
        )
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        runner.run("test_id", metadata)

        assert mock_client.respond_to_current_item.call_count == 2
