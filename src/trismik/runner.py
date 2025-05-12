from datetime import datetime, timedelta
from typing import List, Callable, Any, Optional

from trismik.client import TrismikClient
from trismik.types import (
    TrismikTest,
    TrismikAuth,
    TrismikSession,
    TrismikItem,
    TrismikResult,
    TrismikResponse,
    TrismikSessionMetadata,
    TrismikRunResults,
)


class TrismikRunner:
    def __init__(
            self,
            item_processor: Callable[[TrismikItem], Any],
            client: Optional[TrismikClient] = None,
            auth: Optional[TrismikAuth] = None,
    ) -> None:
        """
        Initializes a new Trismik runner.

        Args:
            item_processor (Callable[[TrismikItem], Any]): Function to process test items.
            client (Optional[TrismikClient]): Trismik client to use for requests.
            auth (Optional[TrismikAuth]): Authentication token to use for requests

        Raises:
            TrismikApiError: If API request fails.
        """
        self._item_processor = item_processor
        self._client = client
        self._auth = auth

    def run(self,
            test_id: str,
            session_metadata: TrismikSessionMetadata,
            with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Runs a test.

        Args:
            test_id (str): ID of the test to run.
            with_responses (bool): If True, responses will be included with the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._init()
        self._refresh_token_if_needed()
        session = self._client.create_session(test_id, session_metadata, self._auth.token)

        self._run_session(session.url)
        results = self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = self._client.responses(session.url, self._auth.token)
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    def _run_session(self, session_url: str) -> None:
        item = self._client.current_item(session_url, self._auth.token)
        while item is not None:
            self._refresh_token_if_needed()
            response = self._item_processor(item)
            item = self._client.respond_to_current_item(
                    session_url, response, self._auth.token
            )

    def run_replay(self,
            previous_session_id: str,
            session_metadata: TrismikSessionMetadata,
            with_responses: bool = False,
    ) -> TrismikRunResults:
        """
        Replay the exact sequence of questions from a previous session

        Args:
            previous_session_id (str): ID of a previous session to replay
            with_responses (bool): If True, responses will be included with the results.

        Returns:
            TrismikRunResults: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._init()
        self._refresh_token_if_needed()
        session = self._client.create_replay_session(previous_session_id, session_metadata, self._auth.token)

        self._run_session(session.url)
        results = self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = self._client.responses(session.url, self._auth.token)
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)

    def _init(self) -> None:
        if self._client is None:
            self._client = TrismikClient()

        if self._auth is None:
            self._auth = self._client.authenticate()

    def _refresh_token_if_needed(self) -> None:
        if self._token_needs_refresh():
            self._auth = self._client.refresh_token(self._auth.token)

    def _token_needs_refresh(self) -> bool:
        return self._auth.expires < (datetime.now() + timedelta(minutes=5))
