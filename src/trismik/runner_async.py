from datetime import datetime, timedelta
from typing import List, Callable, Any, Awaitable, Optional

from .client_async import TrismikAsyncClient
from .types import (
    TrismikAuth,
    TrismikItem,
    TrismikResult,
    TrismikRunResults,
    TrismikSessionMetadata,
)


class TrismikAsyncRunner:
    def __init__(
            self,
            item_processor: Callable[[TrismikItem], Awaitable[Any]],
            client: Optional[TrismikAsyncClient] = None,
            auth: Optional[TrismikAuth] = None,
    ) -> None:
        """
        Initializes a new Trismik runner (async version).

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

    async def run(self,
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
        await self._init()
        await self._refresh_token_if_needed()
        session = await self._client.create_session(test_id, session_metadata, self._auth.token)

        await self._run_session(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(session.url,
                                                     self._auth.token)
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)
        
    async def run_replay(self,
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
        await self._init()
        await self._refresh_token_if_needed()
        session = await self._client.create_replay_session(previous_session_id, session_metadata, self._auth.token)

        await self._run_session(session.url)
        results = await self._client.results(session.url, self._auth.token)

        if with_responses:
            responses = await self._client.responses(session.url, self._auth.token)
            return TrismikRunResults(session.id, results, responses)
        else:
            return TrismikRunResults(session.id, results)
        
    async def _run_session(self, session_url: str) -> None:
        await self._init()
        await self._refresh_token_if_needed()
        item = await self._client.current_item(session_url, self._auth.token)
        while item is not None:
            await self._refresh_token_if_needed()
            response = await self._item_processor(item)
            item = await self._client.respond_to_current_item(
                    session_url, response, self._auth.token)

    async def _init(self) -> None:
        if self._client is None:
            self._client = TrismikAsyncClient()

        if self._auth is None:
            self._auth = await self._client.authenticate()

    async def _refresh_token_if_needed(self) -> None:
        if self._token_needs_refresh():
            self._auth = await self._client.refresh_token(self._auth.token)

    def _token_needs_refresh(self) -> bool:
        return self._auth.expires < (datetime.now() + timedelta(minutes=5))
