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
    TrismikDataset,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikReplayResponse,
    TrismikResponse,
    TrismikRunInfo,
    TrismikRunMetadata,
    TrismikRunResponse,
    TrismikRunResults,
    TrismikRunState,
    TrismikRunSummary,
    TrismikTextChoice,
)


class TestAdaptiveTest:
    """Test suite for the AdaptiveTest class."""

    @pytest.fixture
    def mock_client(self) -> TrismikAsyncClient:
        """Create a mock async client."""
        client = MagicMock(spec=TrismikAsyncClient)

        start_response = TrismikRunResponse(
            run_info=TrismikRunInfo(id="session_id"),
            state=TrismikRunState(
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

        continue_response = TrismikRunResponse(
            run_info=TrismikRunInfo(id="session_id"),
            state=TrismikRunState(
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

        end_response = TrismikRunResponse(
            run_info=TrismikRunInfo(id="session_id"),
            state=TrismikRunState(
                responses=["item_1", "item_2", "item_3"],
                thetas=[1.0, 1.2, 1.3],
                std_error_history=[0.5, 0.4, 0.3],
                kl_info_history=[0.1, 0.12, 0.13],
                effective_difficulties=[0.2, 0.25, 0.3],
            ),
            next_item=None,
            completed=True,
        )

        # Mock responses for replay functionality
        session_summary_response = TrismikRunSummary(
            id="previous_session_id",
            dataset_id="test_id",
            state=TrismikRunState(
                responses=["item_1", "item_2"],
                thetas=[1.0, 1.2],
                std_error_history=[0.5, 0.4],
                kl_info_history=[0.1, 0.12],
                effective_difficulties=[0.2, 0.25],
            ),
            dataset=[
                TrismikMultipleChoiceTextItem(
                    id="item_1",
                    question="q1",
                    choices=[TrismikTextChoice(id="c1", text="t1")],
                ),
                TrismikMultipleChoiceTextItem(
                    id="item_2",
                    question="q2",
                    choices=[TrismikTextChoice(id="c2", text="t2")],
                ),
            ],
            responses=[
                TrismikResponse(
                    dataset_item_id="item_1",
                    value="c1",
                    correct=True,
                ),
                TrismikResponse(
                    dataset_item_id="item_2",
                    value="c2",
                    correct=False,
                ),
            ],
            metadata={"original": "metadata"},
        )

        replay_response = TrismikReplayResponse(
            id="replay_session_id",
            datasetId="test_id",
            state=TrismikRunState(
                responses=["item_1", "item_2"],
                thetas=[1.1, 1.3],
                std_error_history=[0.45, 0.35],
                kl_info_history=[0.11, 0.13],
                effective_difficulties=[0.21, 0.26],
            ),
            replay_of_run="previous_session_id",
            completedAt=None,
            createdAt=None,
            metadata={"replay": "metadata"},
            dataset=[
                TrismikMultipleChoiceTextItem(
                    id="item_1",
                    question="q1",
                    choices=[TrismikTextChoice(id="c1", text="t1")],
                ),
                TrismikMultipleChoiceTextItem(
                    id="item_2",
                    question="q2",
                    choices=[TrismikTextChoice(id="c2", text="t2")],
                ),
            ],
            responses=[
                TrismikResponse(
                    dataset_item_id="item_1",
                    value="c1",
                    correct=True,
                ),
                TrismikResponse(
                    dataset_item_id="item_2",
                    value="c2",
                    correct=False,
                ),
            ],
        )

        # Mock list_datasets response
        list_datasets_response = [
            TrismikDataset(id="test_1", name="Test 1"),
            TrismikDataset(id="test_2", name="Test 2"),
        ]

        client.list_datasets = AsyncMock(return_value=list_datasets_response)
        client.start_run = AsyncMock(return_value=start_response)
        client.continue_run = AsyncMock(
            side_effect=[continue_response, end_response]
        )
        client.run_summary = AsyncMock(return_value=session_summary_response)
        client.submit_replay = AsyncMock(return_value=replay_response)
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

    def test_list_datasets_sync(self, sync_runner, mock_client):
        """Test listing datasets synchronously."""
        datasets = sync_runner.list_datasets()

        mock_client.list_datasets.assert_called_once()
        assert isinstance(datasets, list)
        assert len(datasets) == 2
        assert datasets[0].id == "test_1"
        assert datasets[0].name == "Test 1"
        assert datasets[1].id == "test_2"
        assert datasets[1].name == "Test 2"

    @pytest.mark.asyncio
    async def test_list_datasets_async(self, async_runner, mock_client):
        """Test listing datasets asynchronously."""
        datasets = await async_runner.list_datasets_async()

        mock_client.list_datasets.assert_called_once()
        assert isinstance(datasets, list)
        assert len(datasets) == 2
        assert datasets[0].id == "test_1"
        assert datasets[0].name == "Test 1"
        assert datasets[1].id == "test_2"
        assert datasets[1].name == "Test 2"

    def test_run_sync(self, sync_runner, mock_client):
        """Test running a test synchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run(
            "test_id", "project_id", "experiment", metadata, return_dict=False
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        mock_client.continue_run.assert_called_with(
            "session_id", "processed_response"
        )
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta value from mock

    @pytest.mark.asyncio
    async def test_run_async(self, async_runner, mock_client):
        """Test running a test asynchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_async(
            "test_id", "project_id", "experiment", metadata, return_dict=False
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        mock_client.continue_run.assert_called_with(
            "session_id", "processed_response"
        )
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta value from mock

    def test_run_with_responses_sync(self, sync_runner):
        """Test running a test with responses synchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        with pytest.raises(NotImplementedError):
            sync_runner.run(
                "test_id",
                "project_id",
                "experiment",
                metadata,
                with_responses=True,
                return_dict=False,
            )

    @pytest.mark.asyncio
    async def test_run_with_responses_async(self, async_runner):
        """Test running a test with responses asynchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        with pytest.raises(NotImplementedError):
            await async_runner.run_async(
                "test_id",
                "project_id",
                "experiment",
                metadata,
                with_responses=True,
                return_dict=False,
            )

    def test_run_replay_sync(self, sync_runner, mock_client):
        """Test replaying a test synchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay(
            "previous_session_id", metadata, return_dict=False
        )

        # Verify session_summary was called
        mock_client.run_summary.assert_called_once_with("previous_session_id")

        # Verify submit_replay was called with correct parameters
        mock_client.submit_replay.assert_called_once()
        call_args = mock_client.submit_replay.call_args
        assert call_args[0][0] == "previous_session_id"  # session_id
        assert len(call_args[0][1].responses) == 2  # replay_request

        # Verify results
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta from replay response
        assert (
            results.score.std_error == 0.35
        )  # Final std_error from replay response

    @pytest.mark.asyncio
    async def test_run_replay_async(self, async_runner, mock_client):
        """Test replaying a test asynchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_replay_async(
            "previous_session_id", metadata, return_dict=False
        )

        # Verify session_summary was called
        mock_client.run_summary.assert_called_once_with("previous_session_id")

        # Verify submit_replay was called with correct parameters
        mock_client.submit_replay.assert_called_once()
        call_args = mock_client.submit_replay.call_args
        assert call_args[0][0] == "previous_session_id"  # session_id
        assert len(call_args[0][1].responses) == 2  # replay_request

        # Verify results
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3  # Final theta from replay response
        assert (
            results.score.std_error == 0.35
        )  # Final std_error from replay response

    def test_run_replay_with_responses_sync(self, sync_runner, mock_client):
        """Test replaying a test with responses synchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay(
            "previous_session_id",
            metadata,
            with_responses=True,
            return_dict=False,
        )

        # Verify session_summary was called
        mock_client.run_summary.assert_called_once_with("previous_session_id")

        # Verify submit_replay was called
        mock_client.submit_replay.assert_called_once()

        # Verify results include responses
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.responses is not None
        assert len(results.responses) == 2
        assert results.responses[0].dataset_item_id == "item_1"
        assert results.responses[1].dataset_item_id == "item_2"

    @pytest.mark.asyncio
    async def test_run_replay_with_responses_async(
        self, async_runner, mock_client
    ):
        """Test replaying a test with responses asynchronously."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_replay_async(
            "previous_session_id",
            metadata,
            with_responses=True,
            return_dict=False,
        )

        # Verify session_summary was called
        mock_client.run_summary.assert_called_once_with("previous_session_id")

        # Verify submit_replay was called
        mock_client.submit_replay.assert_called_once()

        # Verify results include responses
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.responses is not None
        assert len(results.responses) == 2
        assert results.responses[0].dataset_item_id == "item_1"
        assert results.responses[1].dataset_item_id == "item_2"

    def test_run_replay_with_async_processor(self, async_runner, mock_client):
        """Test replaying a test with async item processor."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = async_runner.run_replay(
            "previous_session_id", metadata, return_dict=False
        )

        # Verify session_summary was called
        mock_client.run_summary.assert_called_once_with("previous_session_id")

        # Verify submit_replay was called
        mock_client.submit_replay.assert_called_once()

        # Verify results
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)

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

    def test_run_sync_return_dict_true(self, sync_runner, mock_client):
        """Test running a test synchronously with return_dict=True."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run(
            "test_id", "project_id", "experiment", metadata, return_dict=True
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        mock_client.continue_run.assert_called_with(
            "session_id", "processed_response"
        )
        assert isinstance(results, dict)
        assert "session_id" in results
        assert "score" in results
        assert "responses" in results
        assert results["session_id"] == "session_id"
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.3
        assert results["responses"] is None

    def test_run_sync_return_dict_false(self, sync_runner, mock_client):
        """Test running a test synchronously with return_dict=False."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run(
            "test_id", "project_id", "experiment", metadata, return_dict=False
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3

    @pytest.mark.asyncio
    async def test_run_async_return_dict_true(self, async_runner, mock_client):
        """Test running a test asynchronously with return_dict=True."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_async(
            "test_id", "project_id", "experiment", metadata, return_dict=True
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        assert isinstance(results, dict)
        assert "session_id" in results
        assert "score" in results
        assert "responses" in results
        assert results["session_id"] == "session_id"
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.3
        assert results["responses"] is None

    @pytest.mark.asyncio
    async def test_run_async_return_dict_false(self, async_runner, mock_client):
        """Test running a test asynchronously with return_dict=False."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_async(
            "test_id", "project_id", "experiment", metadata, return_dict=False
        )

        mock_client.start_run.assert_called_once_with(
            "test_id", "project_id", "experiment", metadata
        )
        assert mock_client.continue_run.call_count == 2
        assert isinstance(results, TrismikRunResults)
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3

    def test_run_replay_sync_return_dict_true(self, sync_runner, mock_client):
        """Test replaying a test synchronously with return_dict=True."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay(
            "previous_session_id", metadata, return_dict=True
        )

        mock_client.run_summary.assert_called_once_with("previous_session_id")
        mock_client.submit_replay.assert_called_once()
        assert isinstance(results, dict)
        assert "session_id" in results
        assert "score" in results
        assert "responses" in results
        assert results["session_id"] == "replay_session_id"
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.35
        assert results["responses"] is None

    def test_run_replay_sync_return_dict_false(self, sync_runner, mock_client):
        """Test replaying a test synchronously with return_dict=False."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay(
            "previous_session_id", metadata, return_dict=False
        )

        mock_client.run_summary.assert_called_once_with("previous_session_id")
        mock_client.submit_replay.assert_called_once()
        assert isinstance(results, TrismikRunResults)
        assert results.run_id == "replay_session_id"
        assert isinstance(results.score, AdaptiveTestScore)
        assert results.score.theta == 1.3

    def test_run_replay_with_responses_return_dict_true(
        self, sync_runner, mock_client
    ):
        """Test replaying a test with responses and return_dict=True."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = sync_runner.run_replay(
            "previous_session_id",
            metadata,
            with_responses=True,
            return_dict=True,
        )

        mock_client.run_summary.assert_called_once_with("previous_session_id")
        mock_client.submit_replay.assert_called_once()
        assert isinstance(results, dict)
        assert "session_id" in results
        assert "score" in results
        assert "responses" in results
        assert results["session_id"] == "replay_session_id"
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.35
        assert results["responses"] is not None
        assert len(results["responses"]) == 2
        assert results["responses"][0]["dataset_item_id"] == "item_1"
        assert results["responses"][0]["value"] == "c1"
        assert results["responses"][0]["correct"] is True
        assert results["responses"][1]["dataset_item_id"] == "item_2"
        assert results["responses"][1]["value"] == "c2"
        assert results["responses"][1]["correct"] is False

    @pytest.mark.asyncio
    async def test_run_replay_async_return_dict_true(
        self, async_runner, mock_client
    ):
        """Test replaying a test asynchronously with return_dict=True."""
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        results = await async_runner.run_replay_async(
            "previous_session_id", metadata, return_dict=True
        )

        mock_client.run_summary.assert_called_once_with("previous_session_id")
        mock_client.submit_replay.assert_called_once()
        assert isinstance(results, dict)
        assert "session_id" in results
        assert "score" in results
        assert "responses" in results
        assert results["session_id"] == "replay_session_id"
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.35
        assert results["responses"] is None
