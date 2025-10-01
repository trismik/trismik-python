"""Tests for sync client context manager functionality."""

from unittest.mock import MagicMock

import httpx
import pytest

from trismik import TrismikClient
from trismik.settings import environment_settings


class TestTrismikClientContextManager:
    """Test suite for TrismikClient context manager functionality."""

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        """Set environment variables for testing."""
        monkeypatch.setenv(environment_settings["trismik_service_url"], "service_url")
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

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
