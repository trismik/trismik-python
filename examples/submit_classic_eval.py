"""
Example usage of classic evaluation through the Trismik API.

This file demonstrates how to use the AdaptiveTest class to submit a classic
evaluation run with pre-computed model outputs and metrics. Unlike adaptive
testing, classic evaluation allows you to submit all results at once rather
than answering questions iteratively.

This example loads mock data from mock_data.json and submits it to the Trismik
API for evaluation recording and analysis.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from trismik.adaptive_test import AdaptiveTest
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
    """Submit a classic evaluation synchronously."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(lambda x: None)

    # Get user information
    me_response = runner.me()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    print(f"Organization: {me_response.organization.name}")

    # Load mock data and create request
    mock_data = load_mock_data()
    classic_eval_request = create_classic_eval_request(
        mock_data, project_id, experiment
    )

    # Submit the evaluation
    print("Submitting mock output of classic eval run...")
    response = runner.submit_classic_eval(classic_eval_request)
    print(f"Run {response.id} submitted.")


async def run_async_example(project_id: str, experiment: str) -> None:
    """Submit a classic evaluation asynchronously."""
    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(lambda x: None)

    # Get user information
    me_response = await runner.me_async()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    print(f"Organization: {me_response.organization.name}")

    # Load mock data and create request
    mock_data = load_mock_data()
    classic_eval_request = create_classic_eval_request(
        mock_data, project_id, experiment
    )

    # Submit the evaluation
    print("Submitting mock output of classic eval run...")
    response = await runner.submit_classic_eval_async(classic_eval_request)
    print(f"Run {response.id} submitted.")


async def main() -> None:
    """
    Run both synchronous and asynchronous classic evaluation examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Submit classic evaluation examples to Trismik API"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="Project ID for the Trismik evaluation",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name for the Trismik evaluation",
    )
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

    if args.sync_only:
        run_sync_example(args.project_id, args.experiment)
    elif args.async_only:
        await run_async_example(args.project_id, args.experiment)
    else:
        # Run sync example
        run_sync_example(args.project_id, args.experiment)

        # Run async example
        await run_async_example(args.project_id, args.experiment)


if __name__ == "__main__":
    asyncio.run(main())
