"""
Example usage of the TrismikAsyncClient class.

This module demonstrates how to use the TrismikAsyncClient to run tests and
replay sessions asynchronously.
"""

import asyncio
from typing import Any, List, Optional

from _sample_metadata import sample_metadata
from dotenv import load_dotenv

from trismik.client_async import TrismikAsyncClient
from trismik.types import (
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResponse,
    TrismikResult,
    TrismikTest,
)


def print_tests(tests: List[TrismikTest]) -> None:
    """Print available tests with their IDs and names."""
    print("Available tests:")
    for test in tests:
        print(f"{test.id} ({test.name})")


async def run_test(
    client: TrismikAsyncClient, session_url: str, token: str
) -> None:
    """Run a test session by processing items until completion."""
    print("\nStarting test...")
    item: Optional[TrismikItem] = await client.current_item(session_url, token)
    while item:
        response = await process_item(item)
        item = await client.respond_to_current_item(
            session_url, response, token
        )


async def process_item(item: TrismikItem) -> Any:
    """
    Process a test item and return a response.

    Args:
        item (TrismikItem): Test item to process.

    Returns:
        Any: Response to the test item (depends on item type).
    """
    # Concrete type is determined by checking against its class.
    if isinstance(item, TrismikMultipleChoiceTextItem):
        # For TrismikMultipleChoiceTextItem, expected response is a choice id.
        # In reality, you would probably want to process the item in a more
        # sophisticated way than just always answering with the first choice.
        print(f"Processing item: {item.id}...")
        return item.choices[0].id
    else:
        raise RuntimeError("Encountered unknown item type")


def print_results(results: List[TrismikResult]) -> None:
    """Print test results with trait, name, and value."""
    print("\nResults...")
    for result in results:
        print(f"{result.trait} ({result.name}): {result.value}")


def print_responses(responses: List[TrismikResponse]) -> None:
    """Print test responses with item ID and correctness."""
    print("\nResponses...")
    for response in responses:
        correct = "correct" if response.score > 0 else "incorrect"
        print(f"{response.item_id}: {correct}")


async def main() -> None:
    """
    Run a test using the TrismikAsyncClient class and then replay it.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()
    client = TrismikAsyncClient()
    token = (await client.authenticate()).token
    tests = await client.available_tests(token)

    if not tests:
        raise RuntimeError("No tests available")

    print_tests(tests)
    test_id = "MMLUPro2024"  # Assuming it is available
    session = await client.create_session(test_id, sample_metadata, token)

    await client.add_metadata(session.id, sample_metadata, token)

    await run_test(client, session.url, token)
    results = await client.results(session.url, token)
    print_results(results)
    responses = await client.responses(session.url, token)
    print_responses(responses)

    print("\nReplay run")

    replay_session = await client.create_replay_session(
        session.id, sample_metadata, token
    )
    await run_test(client, replay_session.url, token)
    results = await client.results(replay_session.url, token)
    print_results(results)
    responses = await client.responses(replay_session.url, token)
    print_responses(responses)


if __name__ == "__main__":
    asyncio.run(main())
