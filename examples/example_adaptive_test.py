"""
Example usage adaptive testing through the Trismik API.

This file provides a skeleton for how to use the TrismikClient and
TrismikAsyncClient classes to run tests. In this example, we mock the item
processing by picking the first choice. In a real application, you would
implement your own model inference in either a sync or async function.

This example also demonstrates replay functionality with custom metadata.
The replay runs use different metadata than the original runs to
show how you can track different model configurations, hardware setups,
or test parameters when replaying runs.
"""

import asyncio
import time
from typing import Any

from _cli_helpers import create_base_parser, create_progress_callback, generate_random_hash
from _sample_metadata import replay_metadata, sample_metadata
from dotenv import load_dotenv

from trismik import TrismikAsyncClient, TrismikClient
from trismik.types import AdaptiveTestScore, TrismikItem, TrismikMultipleChoiceTextItem


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


def run_sync_example(dataset_name: str, project_id: str, experiment: str) -> None:
    """Run an adaptive test synchronously using the TrismikClient."""
    print("\n=== Running Synchronous Example ===")

    with TrismikClient() as client:
        print(f"\nStarting run with dataset name: {dataset_name}")
        results = client.run(
            dataset_name,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference,
            on_progress=create_progress_callback("Running test"),
            return_dict=False,
        )

        print(f"Run {results.run_id} completed.")

        if results.score is not None:
            print_score(results.score)
        else:
            print("No score available.")

        print("\nReplay run")
        # Update replay metadata with the original run ID
        # Note that we use different metadata for the replay run, for example
        # to track that we're using a different model.
        replay_metadata.test_configuration["original_run_id"] = results.run_id

        # Wait a few seconds while the run is saved by the backend
        time.sleep(5)

        replay_results = client.run_replay(
            results.run_id,
            replay_metadata,
            item_processor=mock_inference,
            on_progress=create_progress_callback("Replaying test"),
            with_responses=True,
            return_dict=False,
        )
        print(f"Replay run {replay_results.run_id} completed.")
        if replay_results.score is not None:
            print_score(replay_results.score)
        if replay_results.responses is not None:
            print(f"Number of responses: {len(replay_results.responses)}")


async def run_async_example(dataset_name: str, project_id: str, experiment: str) -> None:
    """Run an adaptive test asynchronously using the TrismikAsyncClient."""
    print("\n=== Running Asynchronous Example ===")

    async with TrismikAsyncClient() as client:
        # Get user information
        me_response = await client.me()
        print(
            f"User: {me_response.user.firstname} {me_response.user.lastname} "
            f"({me_response.user.email})"
        )
        team_names = [org.name for org in me_response.teams]
        print(f"Teams: {', '.join(team_names)}")

        # List available datasets
        available_datasets = await client.list_datasets()
        print("\nAvailable datasets:")
        for dataset in available_datasets:
            print(f"- {dataset.id}")

        print(f"\nStarting run with dataset name: {dataset_name}")
        results = await client.run(
            dataset_name,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference_async,
            on_progress=create_progress_callback("Running test"),
            return_dict=False,
        )

        print(f"Run {results.run_id} completed.")

        if results.score is not None:
            print_score(results.score)
        else:
            print("No score available.")

        print("\nReplay run")
        # Update replay metadata with the original run ID
        # This demonstrates how you can customize metadata for replay runs
        # to track different model configurations, hardware, or test parameters

        replay_metadata.test_configuration["original_run_id"] = results.run_id

        await asyncio.sleep(10)  # Wait 10 seconds before replay
        replay_results = await client.run_replay(
            results.run_id,
            replay_metadata,
            item_processor=mock_inference_async,
            on_progress=create_progress_callback("Replaying test"),
            with_responses=True,
            return_dict=False,
        )
        print(f"Replay run {replay_results.run_id} completed.")
        if replay_results.score is not None:
            print_score(replay_results.score)
        if replay_results.responses is not None:
            print(f"Number of responses: {len(replay_results.responses)}")


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

    # Run sync example
    run_sync_example(args.dataset_name, project_id, experiment)

    # Run async example
    await run_async_example(args.dataset_name, project_id, experiment)


if __name__ == "__main__":
    asyncio.run(main())
