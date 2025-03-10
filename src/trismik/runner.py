from datetime import datetime, timedelta
from typing import List, Callable, Any, Optional

from .client import TrismikClient
from .types import (
    TrismikAuth,
    TrismikItem,
    TrismikResult,
    TrismikResultsAndResponses,
    TrismikSessionMetadata,
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
            with_responses: bool = False,
            session_metadata: Optional[TrismikSessionMetadata] = None
    ) -> List[TrismikResult] | TrismikResultsAndResponses:
        """
        Runs a test.

        Args:
            test_id (str): ID of the test to run.
            with_responses (bool): If True, responses will be included with the results.

        Returns:
            List[TrismikResult] | TrismikResultsAndResponses: Either just test results, or with responses.

        Raises:
            TrismikApiError: If API request fails.
        """
        self._init()
        self._refresh_token_if_needed()
        session = self._client.create_session(test_id, self._auth.token)

        if session_metadata is not None:
            self._client.add_metadata(session.id, session_metadata, self._auth.token)

        self._run_session(session.url)
        results = self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = self._client.responses(session.url, self._auth.token)
            return TrismikResultsAndResponses(results, responses)
        else:
            return results

    def _run_session(self, session_url: str) -> None:
        item = self._client.current_item(session_url, self._auth.token)
        while item is not None:
            self._refresh_token_if_needed()
            response = self._item_processor(item)
            item = self._client.respond_to_current_item(
                    session_url, response, self._auth.token
            )

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
