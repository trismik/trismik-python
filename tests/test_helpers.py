"""
Tests for the helper modules.

This module tests both async and sync helper functions.
"""

import asyncio
from typing import Any

import pytest

from trismik._async.helpers import process_item as async_process_item
from trismik._sync.helpers import process_item as sync_process_item
from trismik.types import TrismikItem


class TestAsyncHelpers:
    """Test suite for async helper functions."""

    @pytest.fixture
    def sample_item(self) -> TrismikItem:
        """Create a sample TrismikItem for testing."""
        return TrismikItem(id="test_item_1")

    @pytest.mark.asyncio
    async def test_process_item_with_sync_processor(self, sample_item: TrismikItem) -> None:
        """Test async process_item with sync processor."""

        def sync_processor(item: TrismikItem) -> str:
            return f"processed_{item.id}"

        result = await async_process_item(sync_processor, sample_item)
        assert result == "processed_test_item_1"

    @pytest.mark.asyncio
    async def test_process_item_with_async_processor(self, sample_item: TrismikItem) -> None:
        """Test async process_item with async processor."""

        async def async_processor(item: TrismikItem) -> str:
            await asyncio.sleep(0)  # Simulate async work
            return f"async_processed_{item.id}"

        result = await async_process_item(async_processor, sample_item)
        assert result == "async_processed_test_item_1"

    @pytest.mark.asyncio
    async def test_process_item_returns_any_type(self, sample_item: TrismikItem) -> None:
        """Test that process_item can return any type."""

        def processor_dict(item: TrismikItem) -> dict[str, Any]:
            return {"id": item.id, "processed": True}

        result = await async_process_item(processor_dict, sample_item)
        assert result == {"id": "test_item_1", "processed": True}


class TestSyncHelpers:
    """Test suite for sync helper functions."""

    @pytest.fixture
    def sample_item(self) -> TrismikItem:
        """Create a sample TrismikItem for testing."""
        return TrismikItem(id="test_item_1")

    def test_process_item_with_sync_processor(self, sample_item: TrismikItem) -> None:
        """Test sync process_item with sync processor."""

        def sync_processor(item: TrismikItem) -> str:
            return f"processed_{item.id}"

        result = sync_process_item(sync_processor, sample_item)
        assert result == "processed_test_item_1"

    def test_process_item_rejects_async_processor(self, sample_item: TrismikItem) -> None:
        """Test that sync process_item rejects async processors."""

        async def async_processor(item: TrismikItem) -> str:
            return f"async_processed_{item.id}"

        with pytest.raises(TypeError) as exc_info:
            sync_process_item(async_processor, sample_item)

        assert "Sync client cannot use async item_processor" in str(exc_info.value)
        assert "Use TrismikAsyncClient instead" in str(exc_info.value)

    def test_process_item_returns_any_type(self, sample_item: TrismikItem) -> None:
        """Test that process_item can return any type."""

        def processor_dict(item: TrismikItem) -> dict[str, Any]:
            return {"id": item.id, "processed": True}

        result = sync_process_item(processor_dict, sample_item)
        assert result == {"id": "test_item_1", "processed": True}
