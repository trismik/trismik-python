from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from trismik import TrismikAsyncClient
from trismik.exceptions import (
    TrismikApiError,
    TrismikError,
    TrismikPayloadTooLargeError,
    TrismikValidationError,
)
from trismik.settings import environment_settings
from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
    TrismikProject,
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
        monkeypatch.setenv(environment_settings["trismik_service_url"], "service_url")
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")
        TrismikAsyncClient(
            service_url=None,
            api_key=None,
        )

    def test_should_initialize_with_default_service_url(self, monkeypatch) -> None:
        monkeypatch.delenv(environment_settings["trismik_service_url"], raising=False)
        TrismikAsyncClient(service_url=None, api_key="api_key")

    def test_should_fail_initialize_when_api_key_not_provided(self, monkeypatch) -> None:
        monkeypatch.delenv(environment_settings["trismik_api_key"], raising=False)
        with pytest.raises(TrismikError, match="api_key client option must be set"):
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            await client.list_datasets()

    @pytest.mark.asyncio
    async def test_should_start_run(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_start_response())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )
        response = await client.start_run("test_id", "project_id", "experiment", metadata)
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            metadata = TrismikRunMetadata(
                model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
                test_configuration={},
                inference_setup={},
            )
            await client.start_run("test_id", "project_id", "experiment", metadata)

    @pytest.mark.asyncio
    async def test_should_fail_start_run_when_payload_too_large(
        self,
    ) -> None:
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(413))
            metadata = TrismikRunMetadata(
                model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
                test_configuration={},
                inference_setup={},
            )
            await client.start_run("test_id", "project_id", "experiment", metadata)

    @pytest.mark.asyncio
    async def test_should_continue_run(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_continue_response())
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            await client.continue_run("run_id", "choice_1")

    @pytest.mark.asyncio
    async def test_should_get_run_summary(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_summary_response())
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            await client.run_summary("run_id")

    @pytest.mark.asyncio
    async def test_should_submit_replay(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_run_replay_response())
        replay_request = TrismikReplayRequest(
            responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            replay_request = TrismikReplayRequest(
                responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
            )
            await client.submit_replay("run_id", replay_request)

    @pytest.mark.asyncio
    async def test_should_fail_submit_replay_when_payload_too_large(
        self,
    ) -> None:
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(413))
            replay_request = TrismikReplayRequest(
                responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
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
            responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
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
            responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
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
            client = TrismikAsyncClient(http_client=self._mock_error_response(422))
            replay_request = TrismikReplayRequest(
                responses=[TrismikReplayRequestItem(itemId="item_id", itemChoiceId="choice_id")]
            )
            await client.submit_replay("run_id", replay_request)

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_service_url"], "service_url")
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

    @pytest.mark.asyncio
    async def test_should_get_me_response(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_me_response())
        me_response = await client.me()

        assert me_response.user.id == "user123"
        assert me_response.user.email == "test@example.com"
        assert me_response.user.firstname == "Test"
        assert me_response.user.lastname == "User"
        assert me_response.user.createdAt == "2025-09-01T11:54:00.261Z"
        assert me_response.user.account_id == "acc123"
        assert len(me_response.teams) == 1
        assert me_response.teams[0].id == "team123"
        assert me_response.teams[0].name == "Test Team"
        assert me_response.teams[0].role == "Owner"
        assert me_response.teams[0].account_id == "acc123"

    @pytest.mark.asyncio
    async def test_should_fail_me_when_api_returned_error(self) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            await client.me()

    @staticmethod
    def _mock_me_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.me()
        http_client.get.return_value = response
        return http_client

    @pytest.mark.asyncio
    async def test_should_submit_classic_eval(self) -> None:
        client = TrismikAsyncClient(http_client=self._mock_classic_eval_response())

        # Create test data
        items = [
            TrismikClassicEvalItem(
                datasetItemId="test-item-id",
                modelInput="Test input",
                modelOutput="Test output",
                goldOutput="Gold output",
                metrics={"accuracy": 0.95},
            )
        ]

        metrics = [TrismikClassicEvalMetric(metricId="overall_score", value=0.85)]

        request = TrismikClassicEvalRequest(
            projectId="proj123",
            experimentName="test_experiment",
            datasetId="dataset123",
            modelName="gpt-4",
            hyperparameters={"temperature": 0.1},
            items=items,
            metrics=metrics,
        )

        response = await client.submit_classic_eval(request)

        # Check basic properties
        assert response.id == "classic_run_id"
        assert response.projectId == "proj123"
        assert response.experimentName == "test_experiment"
        assert response.datasetId == "dataset123"
        assert response.modelName == "gpt-4"
        assert response.type == "Classic"
        assert response.responseCount == 3

        # Check user info
        assert response.user.id == "user123"
        assert response.user.email == "test@example.com"
        assert response.user.firstname == "Test"
        assert response.user.lastname == "User"

        # Check hyperparameters
        assert response.hyperparameters["temperature"] == 0.1
        assert response.hyperparameters["max_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_should_fail_submit_classic_eval_when_api_returned_error(
        self,
    ) -> None:
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))

            request = TrismikClassicEvalRequest(
                projectId="proj123",
                experimentName="test_experiment",
                datasetId="dataset123",
                modelName="gpt-4",
                hyperparameters={},
                items=[],
                metrics=[],
            )

            await client.submit_classic_eval(request)

    @pytest.mark.asyncio
    async def test_should_fail_submit_classic_eval_when_payload_too_large(
        self,
    ) -> None:
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(413))

            request = TrismikClassicEvalRequest(
                projectId="proj123",
                experimentName="test_experiment",
                datasetId="dataset123",
                modelName="gpt-4",
                hyperparameters={},
                items=[],
                metrics=[],
            )

            await client.submit_classic_eval(request)

    @pytest.mark.asyncio
    async def test_should_fail_submit_classic_eval_when_validation_error(
        self,
    ) -> None:
        with pytest.raises(TrismikValidationError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(422))

            request = TrismikClassicEvalRequest(
                projectId="proj123",
                experimentName="test_experiment",
                datasetId="dataset123",
                modelName="gpt-4",
                hyperparameters={},
                items=[],
                metrics=[],
            )

            await client.submit_classic_eval(request)

    @staticmethod
    def _mock_classic_eval_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.submit_classic_eval()
        http_client.post.return_value = response
        return http_client

    @pytest.mark.asyncio
    async def test_should_create_project_with_description(self) -> None:
        """Test successful project creation with name and description."""
        client = TrismikAsyncClient(http_client=self._mock_create_project_response())

        project = await client.create_project(
            name="Test Project",
            team_id="org123",
            description="A test project",
        )

        # Verify response mapping
        assert isinstance(project, TrismikProject)
        assert project.id == "project123"
        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert project.accountId == "org123"
        assert project.createdAt == "2025-09-12T10:00:00.000Z"
        assert project.updatedAt == "2025-09-12T10:00:00.000Z"

    @pytest.mark.asyncio
    async def test_should_create_project_without_description(self) -> None:
        """Test successful project creation with name only."""
        client = TrismikAsyncClient(http_client=self._mock_create_project_no_description_response())

        project = await client.create_project(name="Test Project No Desc", team_id="org123")

        # Verify response mapping
        assert isinstance(project, TrismikProject)
        assert project.id == "project456"
        assert project.name == "Test Project No Desc"
        assert project.description is None
        assert project.accountId == "org123"
        assert project.createdAt == "2025-09-12T10:00:00.000Z"
        assert project.updatedAt == "2025-09-12T10:00:00.000Z"

    @pytest.mark.asyncio
    async def test_should_create_project_with_proper_request_body_and_headers(
        self,
    ) -> None:
        """Test that create_project sends correct request body and headers."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "project789",
            "name": "Mock Project",
            "description": "Mock description",
            "accountId": "org456",
            "createdAt": "2025-09-12T11:00:00.000Z",
            "updatedAt": "2025-09-12T11:00:00.000Z",
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        await client.create_project(
            name="Mock Project",
            team_id="org456",
            description="Mock description",
        )

        # Verify the request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check URL
        assert call_args[0][0] == "../admin/public/projects"

        # Check request body
        assert call_args[1]["json"] == {
            "name": "Mock Project",
            "description": "Mock description",
            "teamId": "org456",
        }

    @pytest.mark.asyncio
    async def test_should_create_project_without_description_in_body(
        self,
    ) -> None:
        """Test that description is omitted from body when None."""
        # Mock the HTTP client to capture the request
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "project999",
            "name": "No Desc Project",
            "description": None,
            "accountId": "org789",
            "createdAt": "2025-09-12T12:00:00.000Z",
            "updatedAt": "2025-09-12T12:00:00.000Z",
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        client = TrismikAsyncClient(http_client=mock_client)

        await client.create_project(name="No Desc Project", team_id="org789")

        # Verify the request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check request body - should contain teamId but not description
        assert call_args[1]["json"] == {
            "name": "No Desc Project",
            "teamId": "org789",
        }

    @pytest.mark.asyncio
    async def test_should_fail_create_project_when_api_returned_error(
        self,
    ) -> None:
        """Test that general API errors raise TrismikApiError."""
        with pytest.raises(TrismikApiError, match="message"):
            client = TrismikAsyncClient(http_client=self._mock_error_response(401))
            await client.create_project("Test Project", "org123")

    @pytest.mark.asyncio
    async def test_should_fail_create_project_when_validation_error(
        self,
    ) -> None:
        """Test that 422 validation error raises TrismikValidationError."""
        with pytest.raises(TrismikValidationError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(422))
            await client.create_project("", "org123")  # Empty name for validation error

    @pytest.mark.asyncio
    async def test_should_fail_create_project_when_payload_too_large(
        self,
    ) -> None:
        """Test that 413 error raises TrismikPayloadTooLargeError."""
        with pytest.raises(TrismikPayloadTooLargeError):
            client = TrismikAsyncClient(http_client=self._mock_error_response(413))
            await client.create_project("Test Project", "org123")

    @pytest.mark.asyncio
    async def test_context_manager_should_enter_and_exit(self) -> None:
        """Test async context manager enters and exits correctly."""
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        async with TrismikAsyncClient(api_key="test_key", http_client=mock_client) as client:
            assert client is not None
            assert isinstance(client, TrismikAsyncClient)

        # Client should not be closed since it was user-provided
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_should_close_owned_client(self) -> None:
        """Test context manager closes client when owned."""
        client = TrismikAsyncClient(api_key="test_key")

        # Mock the close method on the internal client
        client._http_client.aclose = AsyncMock()

        async with client:
            # Should be inside context
            assert client._owns_client is True

        # Client should be closed after context exit
        client._http_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_should_not_close_user_provided_client(
        self,
    ) -> None:
        """Test context manager doesn't close user-provided client."""
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        client = TrismikAsyncClient(api_key="test_key", http_client=mock_client)
        assert client._owns_client is False

        async with client:
            pass

        # User-provided client should not be closed
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_aclose_should_close_owned_client(self) -> None:
        """Test explicit aclose() closes client when owned."""
        client = TrismikAsyncClient(api_key="test_key")

        # Mock the close method on the internal client
        client._http_client.aclose = AsyncMock()

        await client.aclose()

        # Client should be closed
        client._http_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_aclose_should_not_close_user_provided_client(
        self,
    ) -> None:
        """Test explicit aclose() doesn't close user-provided client."""
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.aclose = AsyncMock()

        client = TrismikAsyncClient(api_key="test_key", http_client=mock_client)

        await client.aclose()

        # User-provided client should not be closed
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_should_execute_complete_test_flow(self) -> None:
        """Test that run() executes full adaptive test flow."""
        client = TrismikAsyncClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        results = await client.run(
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

    @pytest.mark.asyncio
    async def test_run_should_accept_async_processor(self) -> None:
        """Test that run() works with async item processor."""
        client = TrismikAsyncClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        async def async_processor(item):
            return item.choices[0].id

        results = await client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=async_processor,
        )

        assert results["run_id"] == "run_id"

    @pytest.mark.asyncio
    async def test_run_replay_should_execute_replay_flow(self) -> None:
        """Test that run_replay() executes replay flow."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=TrismikResponseMocker.run_summary())
        mock_client.post = AsyncMock(return_value=TrismikResponseMocker.run_replay())

        client = TrismikAsyncClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        results = await client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
        )

        assert results["run_id"] == "replay_run_id"
        assert results["score"] is not None

    @staticmethod
    def _mock_complete_run_flow() -> httpx.AsyncClient:
        """Mock HTTP client for complete run flow."""
        mock_client = MagicMock(spec=httpx.AsyncClient)

        # Set up post responses in order: start, continue (not complete), end
        mock_client.post = AsyncMock(
            side_effect=[
                TrismikResponseMocker.run_start(),
                TrismikResponseMocker.run_continue(),
                TrismikResponseMocker.run_end(),
            ]
        )

        return mock_client

    @staticmethod
    def _mock_create_project_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.create_project()
        http_client.post.return_value = response
        return http_client

    @staticmethod
    def _mock_create_project_no_description_response() -> httpx.AsyncClient:
        http_client = MagicMock(httpx.AsyncClient)
        response = TrismikResponseMocker.create_project_no_description()
        http_client.post.return_value = response
        return http_client

    @pytest.mark.asyncio
    async def test_run_should_call_progress_callback(self) -> None:
        """Test that run() calls on_progress callback with correct arguments."""
        client = TrismikAsyncClient(
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

        await client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=progress_callback,
        )

        # Should be called: (0, 60), (1, 60), (2, 60), (3, 3) final
        assert len(progress_calls) >= 3
        # First call should be (0, max_items=60)
        assert progress_calls[0] == (0, 60)
        # Final call should be (current, current) when complete
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @pytest.mark.asyncio
    async def test_run_should_work_without_progress_callback(self) -> None:
        """Test that run() works when on_progress is None."""
        client = TrismikAsyncClient(http_client=self._mock_complete_run_flow())
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        # Should not raise when on_progress is None
        results = await client.run(
            test_id="test_123",
            project_id="proj_456",
            experiment="exp_1",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=None,
        )

        assert results["run_id"] == "run_id"

    @pytest.mark.asyncio
    async def test_run_replay_should_call_progress_callback(self) -> None:
        """Test that run_replay() calls on_progress callback."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=TrismikResponseMocker.run_summary())
        mock_client.post = AsyncMock(return_value=TrismikResponseMocker.run_replay())

        client = TrismikAsyncClient(http_client=mock_client)
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

        await client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=progress_callback,
        )

        # Should be called for each item in the dataset plus final call
        assert len(progress_calls) >= 1
        # Final call should be (total, total)
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @pytest.mark.asyncio
    async def test_run_replay_should_work_without_progress_callback(self) -> None:
        """Test that run_replay() works when on_progress is None."""
        # Create mock client that returns summary then accepts replay
        mock_client = MagicMock(httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=TrismikResponseMocker.run_summary())
        mock_client.post = AsyncMock(return_value=TrismikResponseMocker.run_replay())

        client = TrismikAsyncClient(http_client=mock_client)
        metadata = TrismikRunMetadata(
            model_metadata=TrismikRunMetadata.ModelMetadata(name="test_model"),
            test_configuration={},
            inference_setup={},
        )

        def processor(item):
            return item.choices[0].id

        # Should not raise when on_progress is None
        results = await client.run_replay(
            previous_run_id="run_123",
            run_metadata=metadata,
            item_processor=processor,
            on_progress=None,
        )

        assert results["run_id"] == "replay_run_id"
