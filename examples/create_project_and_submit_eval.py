"""
Example usage of project creation and classic evaluation through Trismik API.

This file demonstrates how to use the AdaptiveTest class to:
1. Auto-detect the user's default organization (where org.name == user.email)
2. Create a new project with auto-generated names
3. Submit a classic evaluation run to that newly created project

This example shows both sync and async usage patterns for creating
projects and submitting evaluations. It loads mock data from mock_data.json
and submits it to the Trismik API for evaluation recording and analysis.

The example auto-generates project and experiment names using random hashes,
making it easy to run without specifying arguments.
"""

import argparse
import asyncio
import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from trismik.adaptive_test import AdaptiveTest
from trismik.types import (
    TrismikClassicEvalItem,
    TrismikClassicEvalMetric,
    TrismikClassicEvalRequest,
    TrismikProject,
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


def generate_random_hash() -> str:
    """
    Generate a secure random 8-character hash for naming.

    Returns:
        str: Random 8-character string suitable for project/experiment names.
    """
    return secrets.token_hex(4)  # 4 bytes = 8 hex characters


def find_user_default_team_account_id(runner: AdaptiveTest) -> str:
    """
    Find the user's default team account ID.

    The default team is the one where team.name == user.email.

    Args:
        runner (AdaptiveTest): AdaptiveTest instance to use for API calls.

    Returns:
        str: Account ID of the user's default team.

    Raises:
        ValueError: If no default team is found.
    """
    me_response = runner.me()
    user_email = me_response.user.email

    for team in me_response.teams:
        if team.name == user_email:
            return team.account_id

    # If no match found, raise an error with helpful information
    team_names = [team.name for team in me_response.teams]
    raise ValueError(
        f"No default team found for user {user_email}. "
        f"Available teams: {', '.join(team_names)}. "
        "Expected to find a team where team.name == user.email."
    )


async def find_user_default_team_account_id_async(runner: AdaptiveTest) -> str:
    """
    Find the user's default team account ID asynchronously.

    The default team is the one where team.name == user.email.

    Args:
        runner (AdaptiveTest): AdaptiveTest instance to use for API calls.

    Returns:
        str: Account ID of the user's default team.

    Raises:
        ValueError: If no default team is found.
    """
    me_response = await runner.me_async()
    user_email = me_response.user.email

    for team in me_response.teams:
        if team.name == user_email:
            return team.account_id

    # If no match found, raise an error with helpful information
    team_names = [team.name for team in me_response.teams]
    raise ValueError(
        f"No default team found for user {user_email}. "
        f"Available teams: {', '.join(team_names)}. "
        "Expected to find a team where team.name == user.email."
    )


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


def run_sync_example(project_description: Optional[str] = None) -> None:
    """Create a project and submit a classic evaluation synchronously."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(lambda x: None)

    # Get user information and find default organization
    me_response = runner.me()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    team_names = [team.name for team in me_response.teams]
    print(f"Teams: {', '.join(team_names)}")

    # Find the user's default team (where team.name == user.email)
    account_id = find_user_default_team_account_id(runner)
    default_team_name = me_response.user.email
    print(f"Using default team: {default_team_name} ({account_id})")

    # Generate random names for project and experiment
    project_name = f"example_{generate_random_hash()}"
    experiment = f"example_{generate_random_hash()}"
    print(f"Generated project name: {project_name}")
    print(f"Generated experiment name: {experiment}")

    # Create a new project
    description = (
        project_description
        or f"Auto-generated project for {experiment} evaluation example"
    )
    print(f"Creating new project '{project_name}'...")
    project: TrismikProject = runner.create_project(
        name=project_name,
        organization_id=account_id,
        description=description,
    )
    print(f"Project created successfully: {project.name} (ID: {project.id})")

    # Load mock data and create request using the new project
    mock_data = load_mock_data()
    classic_eval_request = create_classic_eval_request(
        mock_data, project.id, experiment
    )

    # Submit the evaluation
    print("Submitting mock output of classic eval run...")
    response = runner.submit_classic_eval(classic_eval_request)
    print(f"Run {response.id} submitted to project {project.name}.")


async def run_async_example(project_description: Optional[str] = None) -> None:
    """Create a project and submit a classic evaluation asynchronously."""
    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(lambda x: None)

    # Get user information and find default organization
    me_response = await runner.me_async()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    team_names = [team.name for team in me_response.teams]
    print(f"Teams: {', '.join(team_names)}")

    # Find the user's default team (where team.name == user.email)
    account_id = await find_user_default_team_account_id_async(runner)
    default_team_name = me_response.user.email
    print(f"Using default team: {default_team_name} ({account_id})")

    # Generate random names for project and experiment
    project_name = f"example_{generate_random_hash()}"
    experiment = f"example_{generate_random_hash()}"
    print(f"Generated project name: {project_name}")
    print(f"Generated experiment name: {experiment}")

    # Create a new project
    description = (
        project_description
        or f"Auto-generated project for {experiment} evaluation example"
    )
    print(f"Creating new project '{project_name}'...")
    project: TrismikProject = await runner.create_project_async(
        name=project_name,
        organization_id=account_id,
        description=description,
    )
    print(f"Project created successfully: {project.name} (ID: {project.id})")

    # Load mock data and create request using the new project
    mock_data = load_mock_data()
    classic_eval_request = create_classic_eval_request(
        mock_data, project.id, experiment
    )

    # Submit the evaluation
    print("Submitting mock output of classic eval run...")
    response = await runner.submit_classic_eval_async(classic_eval_request)
    print(f"Run {response.id} submitted to project {project.name}.")


async def main() -> None:
    """
    Run both synchronous and asynchronous project creation and evaluation.

    This example auto-detects the user's default organization and generates
    random project and experiment names, making it easy to run without
    specifying arguments.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Create project and submit classic evaluation to Trismik. "
            "Auto-generates project names and detects default organization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both sync and async examples with auto-generated names
  %(prog)s

  # Run only sync example
  %(prog)s --sync-only

  # Run only async example
  %(prog)s --async-only

  # Add optional description
  %(prog)s --project-description "My test project"
        """,
    )
    parser.add_argument(
        "--project-description",
        type=str,
        help="Optional description for the created projects",
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

    print("=" * 60)
    print("Project Creation and Classic Evaluation Example")
    print("=" * 60)
    print("This example will:")
    print("1. Auto-detect your default organization")
    print("2. Generate random project and experiment names")
    print("3. Create projects and submit evaluations")
    print("=" * 60)

    if args.sync_only:
        run_sync_example(args.project_description)
    elif args.async_only:
        await run_async_example(args.project_description)
    else:
        # Run both sync and async examples
        run_sync_example(args.project_description)
        await run_async_example(args.project_description)

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
