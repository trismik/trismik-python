"""
Example usage of project creation and classic evaluation through Trismik API.

This file demonstrates how to use the Trismik client to:
- Create a new project with auto-generated names
- Submit a classic evaluation run to that newly created project

This example shows both sync and async usage patterns for creating
projects and submitting evaluations. It loads mock data from mock_data.json
and submits it to the Trismik API for evaluation recording and analysis.

The example auto-generates project and experiment names using random hashes,
making it easy to run without specifying arguments.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from _cli_helpers import create_base_parser, generate_random_hash
from dotenv import load_dotenv

from trismik import TrismikAsyncClient, TrismikClient
from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
)


def load_mock_data() -> Dict[str, Any]:
    """
    Load mock data from the mock_data.json file.

    Returns:
        Dict[str, Any]: Dictionary containing the mock evaluation data.
    """
    mock_data_path = Path(__file__).parent / "mock_data.json"

    try:
        with open(mock_data_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mock data file not found at {mock_data_path}. "
            "Make sure mock_data.json exists in the examples directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in mock_data.json: {e}")


def create_classic_eval_request(
    mock_data: Dict[str, Any], project_id: str, experiment: str
) -> TrismikClassicEvalRequest:
    """
    Create a TrismikClassicEvalRequest from the mock data.

    Args:
        mock_data (Dict[str, Any]): Mock data loaded from JSON.
        project_id (str): Project ID for the evaluation.
        experiment (str): Experiment name for the evaluation.

    Returns:
        TrismikClassicEvalRequest: Request object ready for submission.
    """
    # Create items list
    items = []
    for item_data in mock_data["items"]:
        item = TrismikClassicEvalItem(
            datasetItemId=item_data["datasetItemId"],
            modelInput=item_data["modelInput"],
            modelOutput=item_data["modelOutput"],
            goldOutput=item_data["goldOutput"],
            metrics=item_data["metrics"],
        )
        items.append(item)

    # Create metrics list
    metrics = []
    for metric_data in mock_data["metrics"]:
        metric = TrismikClassicEvalMetric(
            metricId=metric_data["metricId"],
            value=metric_data["value"],
        )
        metrics.append(metric)

    # Create the request
    request = TrismikClassicEvalRequest(
        projectId=project_id,
        experimentName=experiment,
        datasetId=mock_data["datasetId"],
        modelName=mock_data["modelName"],
        hyperparameters=mock_data["hyperparameters"],
        items=items,
        metrics=metrics,
    )

    return request


def run_sync_example(project_id: str, experiment: str) -> None:
    """Submit a classic evaluation synchronously using provided project."""
    print("\n=== Running Synchronous Example ===")

    with TrismikClient() as client:
        # Get user information
        me_response = client.me()
        print(
            f"User: {me_response.user.firstname} {me_response.user.lastname} "
            f"({me_response.user.email})"
        )
        team_names = [team.name for team in me_response.teams]
        print(f"Teams: {', '.join(team_names)}")

        # Load mock data and create request
        mock_data = load_mock_data()
        classic_eval_request = create_classic_eval_request(mock_data, project_id, experiment)

        # Submit the evaluation
        print("Submitting mock output of classic eval run...")
        response = client.submit_classic_eval(classic_eval_request)
        print(f"Run {response.id} submitted to project {project_id}.")


async def run_async_example(project_id: str, experiment: str) -> None:
    """Submit a classic evaluation asynchronously using provided project."""
    print("\n=== Running Asynchronous Example ===")

    async with TrismikAsyncClient() as client:
        # Get user information
        me_response = await client.me()
        print(
            f"User: {me_response.user.firstname} {me_response.user.lastname} "
            f"({me_response.user.email})"
        )
        team_names = [team.name for team in me_response.teams]
        print(f"Teams: {', '.join(team_names)}")

        # Load mock data and create request
        mock_data = load_mock_data()
        classic_eval_request = create_classic_eval_request(mock_data, project_id, experiment)

        # Submit the evaluation
        print("Submitting mock output of classic eval run...")
        response = await client.submit_classic_eval(classic_eval_request)
        print(f"Run {response.id} submitted to project {project_id}.")


async def main() -> None:
    """
    Run both synchronous and asynchronous project creation and evaluation.

    Generates random project and experiment names if not specified via CLI.
    Sync and async examples share the same project for comparison.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    # Parse command line arguments using base parser
    parser = create_base_parser("Create project and submit classic evaluation to Trismik")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Run only the synchronous example",
    )
    parser.add_argument(
        "--async-only",
        action="store_true",
        help="Run only the asynchronous example",
    )
    args = parser.parse_args()

    load_dotenv()

    print("=" * 60)
    print("Project Creation and Classic Evaluation Example")
    print("=" * 60)

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

    print("=" * 60)

    if args.sync_only:
        run_sync_example(project_id, experiment)
    elif args.async_only:
        await run_async_example(project_id, experiment)
    else:
        # Run both sync and async examples with shared project
        run_sync_example(project_id, experiment)
        await run_async_example(project_id, experiment)

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
