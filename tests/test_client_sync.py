"""Tests for sync client functionality.

Following the CONTRIBUTING.md guidelines, this module contains:
- Strategic tests to verify unasync transformation works correctly
- Sync-specific behavior tests (e.g., rejecting async processors)
- Context manager tests
"""

from unittest.mock import MagicMock

import httpx
import pytest

from trismik import TrismikClient
from trismik.exceptions import TrismikApiError
from trismik.settings import environment_settings
from trismik.types import TrismikRunMetadata

from ._mocker import TrismikResponseMocker


class TestTrismikClient:
    """Test suite for TrismikClient (sync variant)."""

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        """Set environment variables for testing."""
        monkeypatch.setenv(environment_settings["trismik_service_url"], "service_url")
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    # ===== Strategic HTTP Method Tests (verify transformation) =====

    def test_should_list_datasets(self) -> None:
        """Test list_datasets (verify sync transformation works)."""
        client = TrismikClient(http_client=self._mock_tests_response())
        datasets = client.list_datasets()
        assert len(datasets) == 5
        assert datasets[0].id == "fluency"
        assert datasets[0].name == "Fluency"

    def test_should_fail_list_datasets_when_api_returned_error(self) -> None:
        """Test error handling in sync variant."""
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikClient(http_client=self._mock_error_response(401))
            client.list_datasets()

    def test_should_start_run(self) -> None:
        """Test start_run (verify sync transformation of complex types)."""
        client = TrismikClient(http_client=self._mock_run_start_response())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        response = client.start_run("test_id", "project_id", "experiment", metadata)
        assert response.run_info.id == "run_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_1"

    def test_should_continue_run(self) -> None:
        """Test continue_run (verify sync loop logic)."""
        client = TrismikClient(http_client=self._mock_run_continue_response())
        response = client.continue_run("run_id", "choice_1")
        assert response.run_info.id == "run_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_2"

    # ===== Context Manager Tests =====

    def test_context_manager_should_enter_and_exit(self) -> None:
        """Test sync context manager enters and exits correctly."""
        mock_client = MagicMock(spec=httpx.Client)

        with TrismikClient(api_key="test_key", http_client=mock_client) as client:
            assert client is not None
            assert isinstance(client, TrismikClient)

        # Client should not be closed since it was user-provided
        mock_client.close.assert_not_called()

    def test_context_manager_should_close_owned_client(self) -> None:
        """Test context manager closes client when owned."""
        client = TrismikClient(api_key="test_key")

        # Mock the close method on the internal client
        client._http_client.close = MagicMock()

        with client:
            # Should be inside context
            assert client._owns_client is True

        # Client should be closed after context exit
        client._http_client.close.assert_called_once()

    def test_context_manager_should_not_close_user_provided_client(
        self,
    ) -> None:
        """Test context manager doesn't close user-provided client."""
        mock_client = MagicMock(spec=httpx.Client)

        client = TrismikClient(api_key="test_key", http_client=mock_client)
        assert client._owns_client is False

        with client:
            pass

        # User-provided client should not be closed
        mock_client.close.assert_not_called()

    def test_explicit_close_should_close_owned_client(self) -> None:
        """Test explicit close() closes client when owned."""
        client = TrismikClient(api_key="test_key")

        # Mock the close method on the internal client
        client._http_client.close = MagicMock()

        client.close()

        # Client should be closed
        client._http_client.close.assert_called_once()

    def test_explicit_close_should_not_close_user_provided_client(
        self,
    ) -> None:
        """Test explicit close() doesn't close user-provided client."""
        mock_client = MagicMock(spec=httpx.Client)

        client = TrismikClient(api_key="test_key", http_client=mock_client)

        client.close()

        # User-provided client should not be closed
        mock_client.close.assert_not_called()

    def test_owns_client_flag_set_correctly_when_client_created(self) -> None:
        """Test _owns_client flag is True when we create the client."""
        client = TrismikClient(api_key="test_key")
        assert client._owns_client is True

    def test_owns_client_flag_set_correctly_when_client_provided(self) -> None:
        """Test _owns_client flag is False when client is user-provided."""
        mock_client = MagicMock(spec=httpx.Client)
        client = TrismikClient(api_key="test_key", http_client=mock_client)
        assert client._owns_client is False

    # ===== Orchestration Tests =====

    def test_run_should_execute_complete_test_flow(self) -> None:
        """Test that run() executes full adaptive test flow (sync)."""
        client = TrismikClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        results = client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=processor,
        )

        assert results["run_id"] == "run_id"
        assert results["score"] is not None
        assert results["score"]["theta"] == 1.3
        assert results["score"]["std_error"] == 0.3

    def test_run_should_call_progress_callback(self) -> None:
        """Test that run() calls on_progress callback (sync)."""
        client = TrismikClient(
            http_client=self._mock_complete_run_flow(),
            max_items=60,
        )
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        progress_calls = []

        def progress_callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        def processor(item):
            return item.choices[0].id

        client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=progress_callback,
        )

        # Should be called multiple times
        assert len(progress_calls) >= 3
        # First call should be (0, max_items=60)
        assert progress_calls[0] == (0, 60)
        # Final call should be (current, current) when complete
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_run_should_work_without_progress_callback(self) -> None:
        """Test that run() works when on_progress is None (sync)."""
        client = TrismikClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        # Should not raise when on_progress is None
        results = client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=None,
        )

        assert results["run_id"] == "run_id"

    def test_run_replay_should_execute_replay_flow(self) -> None:
        """Test that run_replay() executes replay flow (sync)."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.Client)
        mock_client.get.return_value = TrismikResponseMocker.run_summary()
        mock_client.post.return_value = TrismikResponseMocker.run_replay()

        client = TrismikClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        results = client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
        )

        assert results["run_id"] == "replay_run_id"
        assert results["score"] is not None

    def test_run_replay_should_call_progress_callback(self) -> None:
        """Test that run_replay() calls on_progress callback (sync)."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.Client)
        mock_client.get.return_value = TrismikResponseMocker.run_summary()
        mock_client.post.return_value = TrismikResponseMocker.run_replay()

        client = TrismikClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        progress_calls = []

        def progress_callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        def processor(item):
            return item.choices[0].id

        client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=progress_callback,
        )

        # Should be called for each item in the dataset plus final call
        assert len(progress_calls) >= 1
        # Final call should be (total, total)
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_run_replay_should_work_without_progress_callback(self) -> None:
        """Test that run_replay() works when on_progress is None (sync)."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.Client)
        mock_client.get.return_value = TrismikResponseMocker.run_summary()
        mock_client.post.return_value = TrismikResponseMocker.run_replay()

        client = TrismikClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        # Should not raise when on_progress is None
        results = client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=None,
        )

        assert results["run_id"] == "replay_run_id"

    # ===== Sync-Specific Behavior Tests =====

    def test_run_should_reject_async_processor(self) -> None:
        """Test that sync client rejects async processors."""
        client = TrismikClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        async def async_processor(item):
            return item.choices[0].id

        # Should raise TypeError for async processor
        with pytest.raises(TypeError) as exc_info:
            client.run(
                test_id="test_123",
                project_id="proj_456",
                experiment="exp_1",
                run_metadata=metadata,
                item_processor=async_processor,
            )

        assert "Sync client cannot use async item_processor" in str(exc_info.value)
        assert "Use TrismikAsyncClient instead" in str(exc_info.value)

    def test_run_replay_should_reject_async_processor(self) -> None:
        """Test that sync run_replay rejects async processors."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.Client)
        mock_client.get.return_value = TrismikResponseMocker.run_summary()
        mock_client.post.return_value = TrismikResponseMocker.run_replay()

        client = TrismikClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        async def async_processor(item):
            return item.choices[0].id

        # Should raise TypeError for async processor
        with pytest.raises(TypeError) as exc_info:
            client.run_replay(
                previous_run_id="run_123",
                run_metadata=metadata,
                item_processor=async_processor,
            )

        assert "Sync client cannot use async item_processor" in str(exc_info.value)
        assert "Use TrismikAsyncClient instead" in str(exc_info.value)

    # ===== Mock Helper Methods =====

    @staticmethod
    def _mock_tests_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.tests()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_error_response(status: int) -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.error(status)
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_run_start_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.run_start()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_run_continue_response() -> httpx.Client:
        http_client = MagicMock(httpx.Client)
        response = TrismikResponseMocker.run_continue()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_complete_run_flow() -> httpx.Client:
        """Mock HTTP client for complete run flow."""
        mock_client = MagicMock(spec=httpx.Client)

        # Set up post responses in order: start, continue (not complete), end
        mock_client.post.side_effect = [
            TrismikResponseMocker.run_start(),
            TrismikResponseMocker.run_continue(),
            TrismikResponseMocker.run_end(),
        ]

        return mock_client
