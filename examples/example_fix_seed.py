"""
Example demonstrating the seed parameter for reproducible adaptive tests.

When a seed is provided, the same seed with the same response history will
produce the same item sequence. This is useful for debugging, benchmarking,
and ensuring consistent test conditions across runs.

This example runs two adaptive tests with the same seed and shows that they
receive the same sequence of items.
"""

import asyncio
from typing import Any

from _cli_helpers import create_base_parser, create_progress_callback, generate_random_hash
from _sample_metadata import sample_metadata
from dotenv import load_dotenv

from trismik import TrismikAsyncClient, TrismikClient
from trismik.types import AdaptiveTestScore, TrismikItem, TrismikMultipleChoiceTextItem

SEED = 42


def mock_inference(item: TrismikItem) -> Any:
    """
    Process a test item synchronously and return a response.

    Args:
        item (TrismikItem): Test item to process.

    Returns:
        Any: Response to the test item (depends on item type).
    """
    if isinstance(item, TrismikMultipleChoiceTextItem):
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
    if isinstance(item, TrismikMultipleChoiceTextItem):
        return item.choices[0].id
    else:
        raise RuntimeError("Encountered unknown item type")


def print_score(score: AdaptiveTestScore) -> None:
    """Print adaptive test score with thetas, standard errors, and KL info."""
    print(f"  Final theta: {score.theta}")
    print(f"  Final standard error: {score.std_error}")


def run_sync_example(dataset_name: str, project_id: str, experiment: str) -> None:
    """Run two seeded adaptive tests synchronously and compare results."""
    print("\n=== Running Synchronous Seeded Example ===")

    with TrismikClient() as client:
        dataset_info = client.get_dataset_info(dataset_name)
        split = dataset_info.splits[0]

        print(f"\nRun 1 (seed={SEED})")
        results_1 = client.run(
            dataset_name,
            split,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference,
            on_progress=create_progress_callback("Run 1"),
            return_dict=False,
            seed=SEED,
        )
        print(f"Run {results_1.run_id} completed.")
        if results_1.score is not None:
            print_score(results_1.score)

        print(f"\nRun 2 (seed={SEED})")
        results_2 = client.run(
            dataset_name,
            split,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference,
            on_progress=create_progress_callback("Run 2"),
            return_dict=False,
            seed=SEED,
        )
        print(f"Run {results_2.run_id} completed.")
        if results_2.score is not None:
            print_score(results_2.score)

        if results_1.score is not None and results_2.score is not None:
            if results_1.score.theta == results_2.score.theta:
                print("\nScores match: both runs produced the same result.")
            else:
                print("\nScores differ: unexpected for identical seeds and responses.")


async def run_async_example(dataset_name: str, project_id: str, experiment: str) -> None:
    """Run two seeded adaptive tests asynchronously and compare results."""
    print("\n=== Running Asynchronous Seeded Example ===")

    async with TrismikAsyncClient() as client:
        dataset_info = await client.get_dataset_info(dataset_name)
        split = dataset_info.splits[0]

        print(f"\nRun 1 (seed={SEED})")
        results_1 = await client.run(
            dataset_name,
            split,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference_async,
            on_progress=create_progress_callback("Run 1"),
            return_dict=False,
            seed=SEED,
        )
        print(f"Run {results_1.run_id} completed.")
        if results_1.score is not None:
            print_score(results_1.score)

        print(f"\nRun 2 (seed={SEED})")
        results_2 = await client.run(
            dataset_name,
            split,
            project_id,
            experiment,
            run_metadata=sample_metadata,
            item_processor=mock_inference_async,
            on_progress=create_progress_callback("Run 2"),
            return_dict=False,
            seed=SEED,
        )
        print(f"Run {results_2.run_id} completed.")
        if results_2.score is not None:
            print_score(results_2.score)

        if results_1.score is not None and results_2.score is not None:
            if results_1.score.theta == results_2.score.theta:
                print("\nScores match: both runs produced the same result.")
            else:
                print("\nScores differ: unexpected for identical seeds and responses.")


async def main() -> None:
    """
    Run both synchronous and asynchronous seeded examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.

    Project and experiment names are auto-generated if not specified via CLI.
    """
    parser = create_base_parser("Run seeded adaptive testing examples with Trismik API")
    args = parser.parse_args()

    load_dotenv()

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

    if args.experiment is None:
        experiment = f"example_experiment_{generate_random_hash()}"
        print(f"Generated experiment name: {experiment}")
    else:
        experiment = args.experiment

    run_sync_example(args.dataset_name, project_id, experiment)
    await run_async_example(args.dataset_name, project_id, experiment)


if __name__ == "__main__":
    asyncio.run(main())
