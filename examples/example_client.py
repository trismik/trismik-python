from typing import Any, List

from dotenv import load_dotenv

from trismik import (
    TrismikClient,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResult,
    TrismikResponse,
    TrismikSessionMetadata
)


def print_tests(tests) -> None:
    print("Available tests:")
    for test in tests:
        print(f"{test.id} ({test.name})")


def run_test(
        client: TrismikClient,
        session_url: str,
        token: str
) -> None:
    print("\nStarting test...")
    item = client.current_item(session_url, token)
    while item:
        response = process_item(item)
        item = client.respond_to_current_item(session_url, response, token)


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


def print_responses(responses: List[TrismikResponse]) -> None:
    print("\nResponses...")
    for response in responses:
        correct = "correct" if response.score > 0 else "incorrect"
        print(f"{response.item_id}: {correct}")


def main():
    """
    Runs a test using the TrismikClient class.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()
    client = TrismikClient()
    token = client.authenticate().token
    tests = client.available_tests(token)

    if not tests:
        raise RuntimeError("No tests available")

    print_tests(tests)
    test_id = "Tox2024"  # Assuming it is available
    session = client.create_session(test_id, token)

    client.add_metadata(session.id, TrismikSessionMetadata(
        model_metadata={
            "name": "Give first response"
        },
        test_configuration={
            "task_name": "Tox2024",
        },
        inference_setup={
            "type": "None",
            "network_size": 0
        }
        )
        , token)

    run_test(client, session.url, token)
    results = client.results(session.url, token)
    print_results(results)
    responses = client.responses(session.url, token)
    print_responses(responses)


if __name__ == "__main__":
    main()
