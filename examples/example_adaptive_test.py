"""
Example usage adaptive testing through the Trismik API.

This file provides a skeleton for how to use the AdaptiveTest class to run
tests. In this class, we mock the item processing by picking the first choice.
In a real application, you would implement your own model inference in
either process_item_sync or process_item_async.
"""

import asyncio
from typing import Any, List, Optional

from _sample_metadata import sample_metadata
from dotenv import load_dotenv

from trismik.adaptive_test import AdaptiveTest
from trismik.types import (
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResponse,
    TrismikResult,
)


def mock_inference(item: TrismikItem) -> Any:
    """
    Process a test item synchronously and return a response.

    This is where you would call your model to run inference and return its
    response.

    Args:
        item (TrismikItem): Test item to process.

    Returns:
        Any: Response to the test item (depends on item type).
    """
    # Concrete type is determined by checking against its class.
    if isinstance(item, TrismikMultipleChoiceTextItem):
        # For TrismikMultipleChoiceTextItem, the expected response is a
        # choice id.

        # Here, item.question is the string containing the question, and
        # item.choices is a list containing the possible choices.
        # You should call your model with item.question (plus any other
        # information you need to put in the prompt), and post-process the
        # model's output to make sure it's a valid choice id.
        # If not, you can just retry until you get a valid response.

        # Here we just pick the first choice to demonstrate the usage of
        # the API.

        return item.choices[0].id
    else:
        raise RuntimeError("Encountered unknown item type")


async def mock_inference_async(item: TrismikItem) -> Any:
    """
    Process a test item asynchronously and return a response.

    Args:
        item (TrismikItem): Test item to process.

    Returns:
        Any: Response to the test item (depends on item type).
    """
    # Same implementation as sync version, but async.
    # See mock_inference for more details.
    if isinstance(item, TrismikMultipleChoiceTextItem):
        return item.choices[0].id
    else:
        raise RuntimeError("Encountered unknown item type")


def print_results(results: List[TrismikResult]) -> None:
    """Print test results with trait, name, and value."""
    print("\nResults...")
    for result in results:
        print(f"{result.trait} ({result.name}): {result.value}")


def print_responses(responses: Optional[List[TrismikResponse]]) -> None:
    """Print test responses with item ID and correctness."""
    if responses is None:
        return

    print("\nResponses...")
    for response in responses:
        correct = "correct" if response.score > 0 else "incorrect"
        print(f"{response.item_id}: {correct}")


def run_sync_example() -> None:
    """Run an adaptive test synchronously using the AdaptiveTest class."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(mock_inference)

    print("\nStarting test...")
    results = runner.run(
        "MMLUPro2024",
        with_responses=True,
        session_metadata=sample_metadata,
    )
    print_results(results.results)
    print_responses(results.responses)

    print("\nReplay run")
    results = runner.run_replay(
        results.session_id, sample_metadata, with_responses=True
    )
    print_results(results.results)
    print_responses(results.responses)


async def run_async_example() -> None:
    """Run an adaptive test asynchronously using the AdaptiveTest class."""
    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(mock_inference_async)

    print("\nStarting test...")
    results = await runner.run_async(
        "MMLUPro2024",
        with_responses=True,
        session_metadata=sample_metadata,
    )
    print_results(results.results)
    print_responses(results.responses)

    print("\nReplay run")
    results = await runner.run_replay_async(
        results.session_id, sample_metadata, with_responses=True
    )
    print_results(results.results)
    print_responses(results.responses)


async def main() -> None:
    """
    Run both synchronous and asynchronous examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()

    # Run sync example
    run_sync_example()

    # Run async example
    await run_async_example()


if __name__ == "__main__":
    asyncio.run(main())
