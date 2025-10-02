"""Async helper functions for Trismik client."""

import asyncio
from typing import Any, Callable

from trismik.types import TrismikItem


async def process_item(item_processor: Callable[[TrismikItem], Any], item: TrismikItem) -> Any:
    """
    Process a test item with either sync or async processor.

    Args:
        item_processor: Function to process the item (can be sync or async)
        item: The test item to process

    Returns:
        The processor's response
    """
    if asyncio.iscoroutinefunction(item_processor):
        return await item_processor(item)
    # Run sync processor in thread pool to avoid blocking
    return await asyncio.to_thread(item_processor, item)
