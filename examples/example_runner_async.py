import asyncio
from typing import Any, List

from dotenv import load_dotenv

from trismik import (
    TrismikAsyncRunner,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResult,
    TrismikResponse,
)


async def process_item(item: TrismikItem) -> Any:
    """
    Processes returned test item.

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
    print("\nResults...")
    for result in results:
        print(f"{result.trait} ({result.name}): {result.value}")


def print_responses(responses: List[TrismikResponse]) -> None:
    print("\nResponses...")
    for response in responses:
        correct = "correct" if response.score > 0 else "incorrect"
        print(f"{response.item_id}: {correct}")


async def main():
    """
    Runs a test using the TrismikRunner class.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()
    runner = TrismikAsyncRunner(process_item)

    print("\nStarting test...")
    results_and_responses = await runner.run("Tox2024",
                                             with_responses=True)  # Assuming it is available
    print_results(results_and_responses.results)
    print_responses(results_and_responses.responses)


if __name__ == "__main__":
    asyncio.run(main())
