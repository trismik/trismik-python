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

This example demonstrates asynchronous usage with AsyncOpenAI, which is
appropriate for API-based inference to maximize I/O concurrency.
"""

import asyncio

from _cli_helpers import create_base_parser, create_progress_callback, generate_random_hash
from dotenv import load_dotenv
from openai import AsyncOpenAI

from trismik import TrismikAsyncClient
from trismik.types import (
    AdaptiveTestScore,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikRunMetadata,
)

model_name = "gpt-4.1-nano-2025-04-14"


def create_run_metadata(dataset_name: str) -> TrismikRunMetadata:
    """Create run metadata for the given dataset."""
    return TrismikRunMetadata(
        model_metadata=TrismikRunMetadata.ModelMetadata(
            name=model_name,
            provider="OpenAI",
        ),
        test_configuration={
            "task_name": dataset_name,
            "response_format": "Multiple-choice",
        },
        inference_setup={},
    )


async def inference(client: AsyncOpenAI, item: TrismikItem, max_retries: int = 5) -> str:
    """
    Run inference on an item using the OpenAI API asynchronously.

    Args:
        client (AsyncOpenAI): AsyncOpenAI client.
        item (TrismikItem): Item to run inference on.
        max_retries (int): Maximum number of retries.
    """
    assert isinstance(item, TrismikMultipleChoiceTextItem)

    # We construct the prompt from the question and the possible choices.
    # We transformed MMLUPro2025 to be in the form of a multiple-choice
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

        response = await client.responses.create(
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


def print_score(score: AdaptiveTestScore) -> None:
    """Print adaptive test score with thetas, standard errors, and KL info."""
    print("\nAdaptive Test Score...")
    print(f"Final theta: {score.theta}")
    print(f"Final standard error: {score.std_error}")


async def run_example(
    client: AsyncOpenAI, dataset_name: str, project_id: str, experiment: str
) -> None:
    """Run an adaptive test asynchronously using the TrismikAsyncClient."""
    print("\n=== Running Asynchronous Example ===")

    async def process_item(item: TrismikItem) -> str:
        """Async wrapper for inference."""
        return await inference(client, item)

    async with TrismikAsyncClient() as trismik_client:
        # Get user information
        me_response = await trismik_client.me()
        print(
            f"User: {me_response.user.firstname} {me_response.user.lastname} "
            f"({me_response.user.email})"
        )
        team_names = [team.name for team in me_response.teams]
        print(f"Teams: {', '.join(team_names)}")

        print(f"\nStarting run with dataset name: {dataset_name}")
        results = await trismik_client.run(
            dataset_name,
            project_id,
            experiment,
            run_metadata=create_run_metadata(dataset_name),
            item_processor=process_item,
            on_progress=create_progress_callback("Running test"),
            return_dict=False,
        )

        print(f"Run {results.run_id} completed.")

        if results.score is not None:
            print_score(results.score)
        else:
            print("No score available.")

        # Uncomment to replay the exact same questions from the previous run.
        # This is useful to test the stability of the model - note that this
        # works best with temperature > 0.

        # print("\nReplay run")
        # replay_results = await trismik_client.run_replay(
        #     results.run_id,
        #     create_run_metadata(dataset_name),
        #     item_processor=process_item,
        #     on_progress=create_progress_callback("Replaying test"),
        #     with_responses=True,
        #     return_dict=False,
        # )
        # print(f"Replay run {replay_results.run_id} completed.")
        # if replay_results.score is not None:
        #     print_score(replay_results.score)


async def main() -> None:
    """
    Run asynchronous adaptive testing example with OpenAI API.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.

    Project and experiment names are auto-generated if not specified via CLI.
    """
    # Parse command line arguments using base parser
    parser = create_base_parser("Run adaptive testing example with Trismik API")
    args = parser.parse_args()

    load_dotenv()

    # Handle project creation or use provided ID
    if args.project_id is None:
        project_name = f"example_project_{generate_random_hash()}"
        print(f"Creating new project: {project_name}")
        async with TrismikAsyncClient() as temp_client:
            project = await temp_client.create_project(name=project_name)
            project_id = project.id
            print(f"Created project: {project.name} (ID: {project.id})")
    else:
        project_id = args.project_id
        print(f"Using existing project ID: {project_id}")

    # Handle experiment name
    if args.experiment is None:
        experiment = f"example_experiment_{generate_random_hash()}"
        print(f"Generated experiment name: {experiment}")
    else:
        experiment = args.experiment

    # Initialize AsyncOpenAI client
    client = AsyncOpenAI()

    # Run async example
    await run_example(client, args.dataset_name, project_id, experiment)


if __name__ == "__main__":
    asyncio.run(main())
