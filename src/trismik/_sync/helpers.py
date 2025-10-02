"""Sync helper functions for Trismik client."""

import asyncio
from typing import Any, Callable

from trismik.types import TrismikItem


def process_item(item_processor: Callable[[TrismikItem], Any], item: TrismikItem) -> Any:
    """
    Process a test item with sync processor only.

    Args:
        item_processor: Function to process the item (must be sync)
        item: The test item to process

    Returns:
        The processor's response

    Raises:
        TypeError: If processor is async
    """
    if asyncio.iscoroutinefunction(item_processor):
        raise TypeError(
            "Sync client cannot use async item_processor. Use TrismikAsyncClient instead."
        )
    return item_processor(item)
