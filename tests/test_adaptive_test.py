"""
Tests for the AdaptiveTest class.

This module tests both synchronous and asynchronous functionality of the
AdaptiveTest class.
"""

from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from trismik.adaptive_test import AdaptiveTest
from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    AdaptiveTestScore,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikRunResults,
    TrismikSessionInfo,
    TrismikSessionMetadata,
    TrismikSessionResponse,
    TrismikSessionState,
    TrismikTextChoice,
)


class TestAdaptiveTest:
    """Test suite for the AdaptiveTest class."""

    @pytest.fixture
    def mock_client(self) -> TrismikAsyncClient:
        """Create a mock async client."""
        client = MagicMock(spec=TrismikAsyncClient)

        start_response = TrismikSessionResponse(
            session_info=TrismikSessionInfo(id="session_id"),
            state=TrismikSessionState(
                responses=["item_1"],
                thetas=[1.0],
                std_error_history=[0.5],
                kl_info_history=[0.1],
                effective_difficulties=[0.2],
            ),
            next_item=TrismikMultipleChoiceTextItem(
                id="item_1",
                question="q1",
                choices=[TrismikTextChoice(id="c1", text="t1")],
            ),
            completed=False,
        )

        continue_response = TrismikSessionResponse(
            session_info=TrismikSessionInfo(id="session_id"),
            state=TrismikSessionState(
                responses=["item_1", "item_2"],
                thetas=[1.0, 1.2],
                std_error_history=[0.5, 0.4],
                kl_info_history=[0.1, 0.12],
                effective_difficulties=[0.2, 0.25],
            ),
            next_item=TrismikMultipleChoiceTextItem(
                id="item_2",
                question="q2",
                choices=[TrismikTextChoice(id="c2", text="t2")],
            ),
            completed=False,
        )

        end_response = TrismikSessionResponse(
            session_info=TrismikSessionInfo(id="session_id"),
            state=TrismikSessionState(
                responses=["item_1", "item_2", "item_3"],
                thetas=[1.0, 1.2, 1.3],
                std_error_history=[0.5, 0.4, 0.3],
                kl_info_history=[0.1, 0.12, 0.13],
                effective_difficulties=[0.2, 0.25, 0.3],
            ),
            next_item=None,
            completed=True,
        )

        client.start_session = AsyncMock(return_value=start_response)
        client.continue_session = AsyncMock(
            side_effect=[continue_response, end_response]
        )
        return client

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
    def sync_runner(self, sync_item_processor, mock_client) -> AdaptiveTest:
        """Create a test instance with sync item processor."""
        return AdaptiveTest(
            item_processor=sync_item_processor,
            client=mock_client,
        )

    @pytest.fixture
    def async_runner(self, async_item_processor, mock_client) -> AdaptiveTest:
        """Create a test instance with async item processor."""
        return AdaptiveTest(
            item_processor=async_item_processor,
            client=mock_client,
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

        mock_client.start_session.assert_called_once_with("test_id", metadata)
        assert mock_client.continue_session.call_count == 2
        mock_client.continue_session.assert_called_with(
            "session_id", "processed_response"
        )
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta value from mock

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

        mock_client.start_session.assert_called_once_with("test_id", metadata)
        assert mock_client.continue_session.call_count == 2
        mock_client.continue_session.assert_called_with(
            "session_id", "processed_response"
        )
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta value from mock

    def test_run_with_responses_sync(self, sync_runner):
        """Test running a test with responses synchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        with pytest.raises(NotImplementedError):
            sync_runner.run("test_id", metadata, with_responses=True)

    @pytest.mark.asyncio
    async def test_run_with_responses_async(self, async_runner):
        """Test running a test with responses asynchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        with pytest.raises(NotImplementedError):
            await async_runner.run_async(
                "test_id", metadata, with_responses=True
            )

    @pytest.mark.skip(
        reason="Replay functionality not updated for new API flow yet."
    )
    def test_run_replay_sync(self, sync_runner):
        """Test replaying a test synchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        sync_runner.run_replay("previous_id", metadata)

    @pytest.mark.skip(
        reason="Replay functionality not updated for new API flow yet."
    )
    @pytest.mark.asyncio
    async def test_run_replay_async(self, async_runner):
        """Test replaying a test asynchronously."""
        metadata = TrismikSessionMetadata(
            model_metadata=TrismikSessionMetadata.ModelMetadata(
                name="test_model"
            ),
            test_configuration={},
            inference_setup={},
        )
        await async_runner.run_replay_async("previous_id", metadata)

    def test_should_create_client_when_api_key_provided(
        self, sync_item_processor
    ):
        """Test that the runner creates a client when api_key is provided."""
        runner = AdaptiveTest(
            item_processor=sync_item_processor,
            api_key="test_api_key",
        )
        assert isinstance(runner._client, TrismikAsyncClient)

    def test_should_raise_error_when_both_client_and_api_key_provided(
        self, sync_item_processor, mock_client
    ):
        """Test that the runner raises an error with both client and api_key."""
        with pytest.raises(
            ValueError,
            match="Either 'client' or 'api_key' should be provided, not both.",
        ):
            AdaptiveTest(
                item_processor=sync_item_processor,
                client=mock_client,
                api_key="test_api_key",
            )
