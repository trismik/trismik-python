from typing import Any, List

from dotenv import load_dotenv

from trismik import (
    TrismikRunner,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResult,
    TrismikResponse,
)

from _sample_metadata import ( sample_metadata )

def process_item(item: TrismikItem) -> Any:
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


def print_responses(responses: List[TrismikResponse] | None) -> None:

    if responses is None:
        return

    print("\nResponses...")
    for response in responses:
        correct = "correct" if response.score > 0 else "incorrect"
        print(f"{response.item_id}: {correct}")


def main():
    """
    Runs a test using the TrismikRunner class and then replays it.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()
    runner = TrismikRunner(process_item)

    print("\nStarting test...")
    results = runner.run("Tox2024", # Assuming it is available
                                       with_responses=True, 
                                       session_metadata=sample_metadata)  
    print_results(results.results)
    print_responses(results.responses)

    print("\nReplay run")

    results = runner.run_replay(results.session_id, sample_metadata, with_responses=True)
    print_results(results.results)
    print_responses(results.responses)

if __name__ == "__main__":
    main()
