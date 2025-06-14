"""
Example usage adaptive testing through the Trismik API and the OpenAI API.

This example uses gpt-4.1-nano-2025-04-14 as an example. You can use any
other model that you have access to. Remember to provide your own Trismik
API key in the .env file. The OpenAI API key can be provided in the .env file
as well, or you can provide it as an environment variable.

The content of the .env file should look like this:

```
TRISMIK_API_KEY=your-trismik-api-key
OPENAI_API_KEY=your-openai-api-key
```
"""

import asyncio
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from trismik.adaptive_test import AdaptiveTest
from trismik.types import (
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResult,
    TrismikSessionMetadata,
)

model_name = "gpt-4.1-nano-2025-04-14"

session_metadata = TrismikSessionMetadata(
    model_metadata=TrismikSessionMetadata.ModelMetadata(
        name=model_name,
        provider="OpenAI",
    ),
    test_configuration={
        "task_name": "MMLUPro2024",
        "response_format": "Multiple-choice",
    },
    inference_setup={},
)


def inference(client: OpenAI, item: TrismikItem, max_retries: int = 5) -> str:
    """
    Run inference on an item using the OpenAI API.

    Args:
        client (OpenAI): OpenAI client.
        item (TrismikItem): Item to run inference on.
        max_retries (int): Maximum number of retries.
    """
    assert isinstance(item, TrismikMultipleChoiceTextItem)

    # We construct the prompt from the question and the possible choices.
    # We transformed MMLUPro2024 to be in the form of a multiple-choice
    # question, so the prompt reflects that.
    prompt = f"{item.question}\nOptions:\n" + "\n".join(
        [f"- {choice.id}: {choice.text}" for choice in item.choices]
    )

    # The system message contains the instructions for the model. We ask the
    # model to adhere strictly to the instructions; the ability of a model to
    # do that is based on the quality and size of the model. We suggest to
    # always do a post-processing step to ensure the model adheres to the
    # instructions.
    messages = [
        {
            "role": "developer",
            "content": """
Answer the question you are given using only a single letter \
(for example, 'A'). \
Do not use punctuation. \
Do not show your reasoning. \
Do not provide any explanation. \
Follow the instructions exactly and \
always answer using a single uppercase letter.

For example, if the question is "What is the capital of France?" and the \
choices are "A. Paris", "B. London", "C. Rome", "D. Madrid",
- the answer should be "A"
- the answer should NOT be "Paris" or "A. Paris" or "A: Paris"

Please adhere strictly to the instructions.
""".strip(),
        },
        {"role": "user", "content": prompt},
    ]

    final_answer = None
    tries = 0
    valid_ids = [choice.id for choice in item.choices]

    while final_answer is None and tries < max_retries:

        response = client.responses.create(
            model=model_name,
            input=messages,
        )
        answer = response.output_text.strip()

        if answer in valid_ids:
            final_answer = answer
        else:
            tries += 1

    if final_answer is None:
        raise RuntimeError(
            f"Failed to run inference on question {item.question}, "
            f"{item.choices}; the last model response was {answer}."
        )

    assert type(final_answer) is str
    return final_answer


def print_results(results: List[TrismikResult]) -> None:
    """Print test results with trait, name, and value."""
    print("\nResults...")
    for result in results:
        print(f"{result.trait} ({result.name}): {result.value}")


def run_sync_example(client: OpenAI) -> None:
    """Run an adaptive test synchronously using the AdaptiveTest class."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(lambda item: inference(client, item))

    print("\nStarting test...")
    results = runner.run(
        "MMLUPro2024",
        with_responses=True,
        session_metadata=session_metadata,
    )
    print_results(results.results)

    # Uncomment to replay the exact same questions from the previous run.
    # This is useful to test the stability of the model - note that this
    # works best with temperature > 0.

    # print("\nReplay run")
    # results = runner.run_replay(
    #     results.session_id, session_metadata, with_responses=True
    # )
    # print_results(results.results)


async def run_async_example(client: OpenAI) -> None:
    """Run an adaptive test asynchronously using the AdaptiveTest class."""
    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(lambda item: inference(client, item))

    print("\nStarting test...")
    results = await runner.run_async(
        "MMLUPro2024",
        with_responses=True,
        session_metadata=session_metadata,
    )
    print_results(results.results)

    # Uncomment to replay the exact same questions from the previous run.
    # This is useful to test the stability of the model - note that this
    # works best with temperature > 0.

    # print("\nReplay run")
    # results = await runner.run_replay_async(
    #     results.session_id, session_metadata, with_responses=True
    # )
    # print_results(results.results)


async def main() -> None:
    """
    Run both synchronous and asynchronous examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    load_dotenv()

    client = OpenAI()

    # Run sync example
    run_sync_example(client)

    # Run async example
    await run_async_example(client)


if __name__ == "__main__":
    asyncio.run(main())
