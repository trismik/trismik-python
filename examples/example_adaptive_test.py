"""
Example usage adaptive testing through the Trismik API.

This file provides a skeleton for how to use the AdaptiveTest class to run
tests. In this class, we mock the item processing by picking the first choice.
In a real application, you would implement your own model inference in
either process_item_sync or process_item_async.

This example also demonstrates replay functionality with custom metadata.
The replay sessions use different metadata than the original sessions to
show how you can track different model configurations, hardware setups,
or test parameters when replaying sessions.
"""

import asyncio
from typing import Any

from _sample_metadata import replay_metadata, sample_metadata
from dotenv import load_dotenv

from trismik.adaptive_test import AdaptiveTest
from trismik.types import (
    AdaptiveTestScore,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
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


def print_score(score: AdaptiveTestScore) -> None:
    """Print adaptive test score with thetas, standard errors, and KL info."""
    print("\nAdaptive Test Score...")
    print(f"Final theta: {score.theta}")
    print(f"Final standard error: {score.std_error}")


def run_sync_example() -> None:
    """Run an adaptive test synchronously using the AdaptiveTest class."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(mock_inference)

    print("\nStarting test...")
    results = runner.run(
        "MMLUPro2025",
        session_metadata=sample_metadata,
    )

    print(f"Session {results.session_id} completed.")

    if results.score is not None:
        print_score(results.score)
    else:
        print("No score available.")

    print("\nReplay run")
    # Update replay metadata with the original session ID
    # This demonstrates how you can customize metadata for replay sessions
    # to track different model configurations, hardware, or test parameters
    replay_metadata.test_configuration["original_session_id"] = (
        results.session_id
    )

    replay_results = runner.run_replay(
        results.session_id, replay_metadata, with_responses=True
    )
    print(f"Replay session {replay_results.session_id} completed.")
    if replay_results.score is not None:
        print_score(replay_results.score)
    if replay_results.responses is not None:
        print(f"Number of responses: {len(replay_results.responses)}")


async def run_async_example() -> None:
    """Run an adaptive test asynchronously using the AdaptiveTest class."""
    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(mock_inference_async)

    print("\nStarting test...")
    results = await runner.run_async(
        "MMLUPro2025",
        session_metadata=sample_metadata,
    )

    print(f"Session {results.session_id} completed.")

    if results.score is not None:
        print_score(results.score)
    else:
        print("No score available.")

    print("\nReplay run")
    # Update replay metadata with the original session ID
    # This demonstrates how you can customize metadata for replay sessions
    # to track different model configurations, hardware, or test parameters
    replay_metadata.test_configuration["original_session_id"] = (
        results.session_id
    )

    replay_results = await runner.run_replay_async(
        results.session_id, replay_metadata, with_responses=True
    )
    print(f"Replay session {replay_results.session_id} completed.")
    if replay_results.score is not None:
        print_score(replay_results.score)
    if replay_results.responses is not None:
        print(f"Number of responses: {len(replay_results.responses)}")


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
