from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from trismik.client_async import TrismikAsyncClient
from trismik.exceptions import (
    TrismikApiError,
    TrismikError,
    TrismikPayloadTooLargeError,
    TrismikValidationError,
)
from trismik.settings import environment_settings
from trismik.types import (
    TrismikReplayRequest,
    TrismikReplayRequestItem,
    TrismikRunMetadata,
)

from ._mocker import TrismikResponseMocker


class TestTrismikAsyncClient:

    def test_should_initialize_with_explicit_values(self) -> None:
        TrismikAsyncClient(
            service_url="service_url",
            api_key="api_key",
        )

    def test_should_initialize_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv(
            environment_settings["trismik_service_url"], "service_url"
        )
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")
        TrismikAsyncClient(
            service_url=None,
            api_key=None,
        )

    def test_should_initialize_with_default_service_url(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv(
            environment_settings["trismik_service_url"], raising=False
        )
        TrismikAsyncClient(service_url=None, api_key="api_key")

    def test_should_fail_initialize_when_api_key_not_provided(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv(
            environment_settings["trismik_api_key"], raising=False
        )
        with pytest.raises(
            TrismikError, match="api_key client option must be set"
        ):
            TrismikAsyncClient(
                service_url="service_url",
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_should_list_datasets(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_tests_response())
        datasets = await client.list_datasets()
        assert len(datasets) == 5
        assert datasets[0].id == "fluency"
        assert datasets[0].name == "Fluency"

    @pytest.mark.asyncio
    async def test_should_fail_list_datasets_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.list_datasets()

    @pytest.mark.asyncio
    async def test_should_start_run(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_start_response())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        response = await client.start_run(
            "test_id", "project_id", "experiment", metadata
        )
        assert response.run_info.id == "run_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_1"
        assert len(response.state.thetas) == 1

    @pytest.mark.asyncio
    async def test_should_fail_start_run_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            metadata = TrismikRunMetadata(
                model_metadata=TrismikRunMetadata.ModelMetadata(
                    name="test_model"
                ),
                test_configuration={},
                inference_setup={},
            )
            await client.start_run(
                "test_id", "project_id", "experiment", metadata
            )

    @pytest.mark.asyncio
    async def test_should_fail_start_run_when_payload_too_large(
        self,
    ) -> None:
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(413)
            )
            metadata = TrismikRunMetadata(
                model_metadata=TrismikRunMetadata.ModelMetadata(
                    name="test_model"
                ),
                test_configuration={},
                inference_setup={},
            )
            await client.start_run(
                "test_id", "project_id", "experiment", metadata
            )

    @pytest.mark.asyncio
    async def test_should_continue_run(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_run_continue_response()
        )
        response = await client.continue_run("run_id", "choice_1")
        assert response.run_info.id == "run_id"
        assert response.completed is False
        assert response.next_item is not None
        assert response.next_item.id == "item_2"
        assert len(response.state.thetas) == 2

    @pytest.mark.asyncio
    async def test_should_end_run_on_continue(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_end_response())
        response = await client.continue_run("run_id", "choice_2")
        assert response.run_info.id == "run_id"
        assert response.completed is True
        assert response.next_item is None
        assert len(response.state.thetas) == 3

    @pytest.mark.asyncio
    async def test_should_fail_continue_run_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.continue_run("run_id", "choice_1")

    @pytest.mark.asyncio
    async def test_should_get_run_summary(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_run_summary_response()
        )
        summary = await client.run_summary("run_id")

        # Check id and test_id
        assert summary.id == "run_id"
        assert summary.dataset_id == "test_id"

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

        # Check metadata
        assert summary.metadata == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_should_fail_get_run_summary_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            await client.run_summary("run_id")

    @pytest.mark.asyncio
    async def test_should_submit_replay(self) -> None:
        client = TrismikAsyncClient(
            http_client=self._mock_run_replay_response()
        )
        replay_request = TrismikReplayRequest(
            responses=[
                TrismikReplayRequestItem(
                    itemId="item_id", itemChoiceId="choice_id"
                )
            ]
        )
        response = await client.submit_replay("run_id", replay_request)

        # Check basic properties
        assert response.id == "replay_run_id"
        assert response.datasetId == "test_id"
        assert response.replay_of_run == "original_run_id"

        # Check state
        assert len(response.state.thetas) == 1
        assert response.state.thetas[0] == 1.0

        # Check dates
        assert response.completedAt is not None
        assert response.createdAt is not None

        # Check dataset and responses
        assert len(response.dataset) == 1
        assert response.dataset[0].id == "item_id"
        assert len(response.responses) == 1
        assert response.responses[0].dataset_item_id == "item_id"

        # Check metadata
        assert response.metadata == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_should_fail_submit_replay_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(401)
            )
            replay_request = TrismikReplayRequest(
                responses=[
                    TrismikReplayRequestItem(
                        itemId="item_id", itemChoiceId="choice_id"
                    )
                ]
            )
            await client.submit_replay("run_id", replay_request)

    @pytest.mark.asyncio
    async def test_should_fail_submit_replay_when_payload_too_large(
        self,
    ) -> None:
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(413)
            )
            replay_request = TrismikReplayRequest(
                responses=[
                    TrismikReplayRequestItem(
                        itemId="item_id", itemChoiceId="choice_id"
                    )
                ]
            )
            await client.submit_replay("run_id", replay_request)

    @pytest.mark.asyncio
    async def test_should_send_metadata_in_request_body(self) -> None:
        """Test that metadata is sent in the request body."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "runInfo": {"id": "run_id"},
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "nextItem": None,
            "completed": True,
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata("test_model"),
            test_configuration={"max_items": 20},
            inference_setup={"temperature": 0.7},
        )

        await client.start_run("test_id", "project_id", "experiment", metadata)

        # Verify the request was made with the correct body
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {
            "datasetId": "test_id",
            "projectId": "project_id",
            "experiment": "experiment",
            "metadata": metadata.toDict(),
        }

    @pytest.mark.asyncio
    async def test_should_send_empty_metadata_when_none_provided(self) -> None:
        """Test that empty metadata is sent when no metadata is provided."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "runInfo": {"id": "run_id"},
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "nextItem": None,
            "completed": True,
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        await client.start_run("test_id", "project_id", "experiment")

        # Verify the request was made with empty metadata
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {
            "datasetId": "test_id",
            "projectId": "project_id",
            "experiment": "experiment",
            "metadata": {},
        }

    @pytest.mark.asyncio
    async def test_should_send_metadata_in_replay_request_body(self) -> None:
        """Test that metadata is sent in the replay request body."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "replay_run_id",
            "datasetId": "test_id",
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "replayOfRun": "original_run_id",
            "metadata": {},
            "dataset": [],
            "responses": [],
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata("test_model"),
            test_configuration={"max_items": 20},
            inference_setup={"temperature": 0.7},
        )

        replay_request = TrismikReplayRequest(
            responses=[
                TrismikReplayRequestItem(
                    itemId="item_id", itemChoiceId="choice_id"
                )
            ]
        )

        await client.submit_replay("run_id", replay_request, metadata)

        # Verify the request was made with the correct body
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {
            "responses": [{"itemId": "item_id", "itemChoiceId": "choice_id"}],
            "metadata": metadata.toDict() if metadata else {},
        }

    @pytest.mark.asyncio
    async def test_should_send_empty_metadata_in_replay_when_none_provided(
        self,
    ) -> None:
        """Test that empty metadata is sent when no metadata is provided."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "replay_run_id",
            "datasetId": "test_id",
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "replayOfRun": "original_run_id",
            "metadata": {},
            "dataset": [],
            "responses": [],
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        replay_request = TrismikReplayRequest(
            responses=[
                TrismikReplayRequestItem(
                    itemId="item_id", itemChoiceId="choice_id"
                )
            ]
        )

        await client.submit_replay("run_id", replay_request)

        # Verify the request was made with empty metadata
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["json"] == {
            "responses": [{"itemId": "item_id", "itemChoiceId": "choice_id"}],
            "metadata": {},
        }

    @pytest.mark.asyncio
    async def test_should_fail_submit_replay_when_validation_error(
        self,
    ) -> None:
        """Test that a 422 validation error raises TrismikValidationError."""
        with pytest.raises(TrismikValidationError):
            client = TrismikAsyncClient(
                http_client=self._mock_error_response(422)
            )
            replay_request = TrismikReplayRequest(
                responses=[
                    TrismikReplayRequestItem(
                        itemId="item_id", itemChoiceId="choice_id"
                    )
                ]
            )
            await client.submit_replay("run_id", replay_request)

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(
            environment_settings["trismik_service_url"], "service_url"
        )
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @staticmethod
    def _mock_tests_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.tests()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_run_start_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.run_start()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_run_continue_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.run_continue()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_run_end_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.run_end()
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
    def _mock_run_summary_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.run_summary()
        http_client.get.return_value = response
        return http_client

    @staticmethod
    def _mock_run_replay_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.run_replay()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_no_content_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.no_content()
        http_client.get.return_value = response
        http_client.post.return_value = response
        return http_client
