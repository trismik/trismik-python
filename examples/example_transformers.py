"""
Example of adaptive testing of a Hugging Face model through the Trismik API.

This example uses Phi-3-small-8k-instruct as an example. You can use any
other model that you have access to. Remember to provide your own Trismik
API key in the .env file.
"""

import asyncio
import re

import transformers
from _cli_helpers import create_base_parser, create_progress_callback, generate_random_hash
from dotenv import load_dotenv

from trismik import TrismikAsyncClient, TrismikClient
from trismik.types import (
    AdaptiveTestScore,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikRunMetadata,
)


def create_run_metadata(dataset_name: str) -> TrismikRunMetadata:
    """Create run metadata for the given dataset."""
    return TrismikRunMetadata(
        model_metadata=TrismikRunMetadata.ModelMetadata(
            name="microsoft/Phi-3-small-8k-instruct",
            parameters="3.84B",
            provider="Microsoft",
        ),
        test_configuration={
            "task_name": dataset_name,
            "response_format": "Multiple-choice",
        },
        inference_setup={
            "max_tokens": 1024,
        },
    )


generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


def inference(pipeline: transformers.pipeline, item: TrismikItem, max_retries: int = 5) -> str:
    """
    Run inference on an item using a Hugging Face model.

    Args:
        pipeline (transformers.pipeline): Hugging Face pipeline.
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
            "role": "system",
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

        outputs = pipeline(messages, **generation_args)
        answer = outputs[0]["generated_text"].strip()

        # No matter how hard you ask, Phi-4-mini-instruct might still
        # return answer in the form of "A: paris". We add a simple
        # post-processing step to extract the letter.
        if len(answer) != 1:
            match = re.match(r"^([A-Z]): .+", answer)
            if match:
                answer = match.group(1)

        if answer in valid_ids:
            final_answer = answer
        else:
            tries += 1

    if final_answer is None:
        raise RuntimeError(
            f"Failed to run inference on question {item.question}, "
            f"{item.choices}; the last model response was {answer}."
        )

    assert isinstance(final_answer, str)
    return final_answer


def print_score(score: AdaptiveTestScore) -> None:
    """Print adaptive test score with thetas, standard errors, and KL info."""
    print("\nAdaptive Test Score...")
    print(f"Final theta: {score.theta}")
    print(f"Final standard error: {score.std_error}")


def run_sync_example(
    pipeline: transformers.pipeline,
    dataset_name: str,
    project_id: str,
    experiment: str,
) -> None:
    """Run an adaptive test synchronously using the TrismikClient."""
    print("\n=== Running Synchronous Example ===")

    with TrismikClient() as trismik_client:
        # Get user information
        me_response = trismik_client.me()
        print(
            f"User: {me_response.user.firstname} {me_response.user.lastname} "
            f"({me_response.user.email})"
        )
        team_names = [team.name for team in me_response.teams]
        print(f"Teams: {', '.join(team_names)}")

        print(f"\nStarting run with dataset name: {dataset_name}")
        results = trismik_client.run(
            dataset_name,
            project_id,
            experiment,
            run_metadata=create_run_metadata(dataset_name),
            item_processor=lambda item: inference(pipeline, item),
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
        # replay_results = trismik_client.run_replay(
        #     results.run_id,
        #     create_run_metadata(dataset_name),
        #     item_processor=lambda item: inference(pipeline, item),
        #     on_progress=create_progress_callback("Replaying test"),
        #     with_responses=True,
        #     return_dict=False,
        # )
        # print(f"Replay run {replay_results.run_id} completed.")
        # if replay_results.score is not None:
        #     print_score(replay_results.score)


async def run_async_example(
    pipeline: transformers.pipeline,
    dataset_name: str,
    project_id: str,
    experiment: str,
) -> None:
    """Run an adaptive test asynchronously using the TrismikAsyncClient."""

    print("\n=== Running Asynchronous Example ===")

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
            item_processor=lambda item: inference(pipeline, item),
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
        #     item_processor=lambda item: inference(pipeline, item),
        #     on_progress=create_progress_callback("Replaying test"),
        #     with_responses=True,
        #     return_dict=False,
        # )
        # print(f"Replay run {replay_results.run_id} completed.")
        # if replay_results.score is not None:
        #     print_score(replay_results.score)


async def main() -> None:
    """
    Run both synchronous and asynchronous examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.

    Project and experiment names are auto-generated if not specified via CLI.
    """
    # Parse command line arguments using base parser
    parser = create_base_parser("Run adaptive testing examples with Trismik API")
    args = parser.parse_args()

    load_dotenv()

    # Handle project creation or use provided ID
    if args.project_id is None:
        project_name = f"example_project_{generate_random_hash()}"
        print(f"Creating new project: {project_name}")
        with TrismikClient() as temp_client:
            project = temp_client.create_project(name=project_name)
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

    # We choose Phi-4-mini-instruct as an example as it requires
    # relatively low memory to run (< 8 GB). You can use any other
    # model that you have access to.

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Run sync example
    run_sync_example(pipeline, args.dataset_name, project_id, experiment)

    # Run async example
    await run_async_example(pipeline, args.dataset_name, project_id, experiment)


if __name__ == "__main__":
    asyncio.run(main())
