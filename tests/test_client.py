from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from trismik.client import TrismikClient
from trismik.exceptions import TrismikApiError
from trismik.types import TrismikAuth, TrismikTest, TrismikSession, TrismikItem, TrismikResult, TrismikResponse, TrismikSessionMetadata

@pytest.fixture
def auth():
    return TrismikAuth(token="token", expires=datetime.now() + timedelta(hours=1))

@pytest.fixture
def item():
    return MagicMock(spec=TrismikItem)

@pytest.fixture(autouse=True)
def mock_async_client(auth, item):
    with patch("trismik.client.TrismikAsyncClient") as AsyncClientMock:
        async_client_instance = MagicMock()
        async_client_instance.authenticate = AsyncMock(return_value=auth)
        async_client_instance.refresh_token = AsyncMock(return_value=auth)
        async_client_instance.available_tests = AsyncMock(return_value=[TrismikTest(id="test", name="Test")])
        async_client_instance.create_session = AsyncMock(return_value=MagicMock(spec=TrismikSession))
        async_client_instance.create_replay_session = AsyncMock(return_value=MagicMock(spec=TrismikSession))
        async_client_instance.current_item = AsyncMock(return_value=item)
        async_client_instance.respond_to_current_item = AsyncMock(return_value=item)
        async_client_instance.results = AsyncMock(return_value=[MagicMock(spec=TrismikResult)])
        async_client_instance.responses = AsyncMock(return_value=[MagicMock(spec=TrismikResponse)])
        async_client_instance.add_metadata = AsyncMock(return_value=None)
        AsyncClientMock.return_value = async_client_instance
        yield async_client_instance

def test_authenticate_delegates(mock_async_client):
    client = TrismikClient()
    result = client.authenticate()
    assert result == mock_async_client.authenticate.return_value
    mock_async_client.authenticate.assert_called_once()

def test_error_propagation(mock_async_client):
    mock_async_client.authenticate.side_effect = TrismikApiError("fail")
    client = TrismikClient()
    with pytest.raises(TrismikApiError, match="fail"):
        client.authenticate()

def test_available_tests_delegates(mock_async_client):
    client = TrismikClient()
    token = "token"
    result = client.available_tests(token)
    assert result == mock_async_client.available_tests.return_value
    mock_async_client.available_tests.assert_called_once_with(token)

def test_create_session_delegates(mock_async_client):
    client = TrismikClient()
    metadata = MagicMock(spec=TrismikSessionMetadata)
    token = "token"
    test_id = "test_id"
    result = client.create_session(test_id, metadata, token)
    assert result == mock_async_client.create_session.return_value
    mock_async_client.create_session.assert_called_once_with(test_id, metadata, token)

def test_create_replay_session_delegates(mock_async_client):
    client = TrismikClient()
    metadata = MagicMock(spec=TrismikSessionMetadata)
    token = "token"
    prev_id = "prev_id"
    result = client.create_replay_session(prev_id, metadata, token)
    assert result == mock_async_client.create_replay_session.return_value
    mock_async_client.create_replay_session.assert_called_once_with(prev_id, metadata, token)

def test_add_metadata_delegates(mock_async_client):
    client = TrismikClient()
    session_id = "session_id"
    metadata = MagicMock(spec=TrismikSessionMetadata)
    token = "token"
    result = client.add_metadata(session_id, metadata, token)
    assert result == mock_async_client.add_metadata.return_value
    mock_async_client.add_metadata.assert_called_once_with(session_id, metadata, token)

def test_current_item_delegates(mock_async_client):
    client = TrismikClient()
    session_url = "url"
    token = "token"
    result = client.current_item(session_url, token)
    assert result == mock_async_client.current_item.return_value
    mock_async_client.current_item.assert_called_once_with(session_url, token)

def test_respond_to_current_item_delegates(mock_async_client):
    client = TrismikClient()
    session_url = "url"
    value = "value"
    token = "token"
    result = client.respond_to_current_item(session_url, value, token)
    assert result == mock_async_client.respond_to_current_item.return_value
    mock_async_client.respond_to_current_item.assert_called_once_with(session_url, value, token)

def test_results_delegates(mock_async_client):
    client = TrismikClient()
    session_url = "url"
    token = "token"
    result = client.results(session_url, token)
    assert result == mock_async_client.results.return_value
    mock_async_client.results.assert_called_once_with(session_url, token)

def test_responses_delegates(mock_async_client):
    client = TrismikClient()
    session_url = "url"
    token = "token"
    result = client.responses(session_url, token)
    assert result == mock_async_client.responses.return_value
    mock_async_client.responses.assert_called_once_with(session_url, token)
