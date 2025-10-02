"""
Minimal test file to validate unasync transformation pipeline.

This file tests that unasync correctly transforms:
- async def -> def
- await -> (removed)
- httpx.AsyncClient -> httpx.Client
- TrismikAsyncClient -> TrismikClient
- __aenter__/__aexit__ -> __enter__/__exit__
- _async -> _sync in imports
"""

from typing import Optional

import httpx


class TrismikAsyncClient:
    """Test async client to validate transformation."""

    def __init__(
        self,
        api_key: str,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize test client."""
        self._api_key = api_key
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(headers={"x-api-key": api_key})

    async def __aenter__(self) -> "TrismikAsyncClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        if self._owns_client:
            await self._http_client.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._owns_client:
            await self._http_client.aclose()

    async def get_data(self) -> str:
        """Test async method with await."""
        response = await self._http_client.get("/test")
        response.raise_for_status()
        return str(response.text)

    async def process_items(self) -> int:
        """Test async method with multiple awaits."""
        count = 0
        async with self._http_client as client:
            response = await client.get("/items")
            data = response.json()
            count = len(data)
        return count
