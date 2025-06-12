"""
Example usage of the TrismikRunner class.

.. deprecated:: 0.9.2
    This example is deprecated and will be removed in a future version.
    Please use examples/example_adaptive_test.py instead, which demonstrates
    both synchronous and asynchronous usage of the new AdaptiveTest class.
"""

from typing import Any, List, Optional

from _sample_metadata import sample_metadata
from dotenv import load_dotenv

from trismik.runner import TrismikRunner
from trismik.types import (
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResponse,
    TrismikResult,
)


def process_item(item: TrismikItem) -> Any:
    """
    Process a test item and return a response.

    Args:
        item (TrismikItem): Test item to process.

    Returns:
        Any: Response to the test item (depends on item type).
    """
    # Concrete type is determined by checking against its class.
    if isinstance(item, TrismikMultipleChoiceTextItem):
        # For TrismikMultipleChoiceTextItem, the expected response is a choice
        # id. Here we just pick the first choice. In a real application, here
        # is where you would call a model to get the response.
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


def main() -> None:
    """
    Run a test using the TrismikRunner class and then replay it.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()
    runner = TrismikRunner(process_item)

    print("\nStarting test...")
    results = runner.run(
        "MMLUPro2024",  # Assuming it is available
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


if __name__ == "__main__":
    main()
