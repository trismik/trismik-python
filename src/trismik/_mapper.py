from datetime import datetime
from typing import Any, Dict, List

from trismik.exceptions import TrismikApiError
from trismik.types import (
    TrismikClassicEvalResponse,
    TrismikDataset,
    TrismikItem,
    TrismikMeResponse,
    TrismikMultipleChoiceTextItem,
    TrismikProject,
    TrismikReplayResponse,
    TrismikResponse,
    TrismikResult,
    TrismikRun,
    TrismikRunInfo,
    TrismikRunResponse,
    TrismikRunState,
    TrismikRunSummary,
    TrismikTeam,
    TrismikTextChoice,
    TrismikUserInfo,
)


class TrismikResponseMapper:
    """
    Maps JSON responses from the Trismik API to Python objects.

    This class provides static methods to convert JSON responses from various
    API endpoints into their corresponding Python object representations.
    """

    @staticmethod
    def to_datasets(json: Dict[str, Any]) -> List[TrismikDataset]:
        """
        Convert JSON response to a list of TrismikDataset objects.

        Args:
            json (Dict[str, Any]): JSON response containing dataset data.

        Returns:
            List[TrismikDataset]: List of
            dataset objects with IDs and names.
        """
        return [
            TrismikDataset(
                id=item["id"],
                name=item["name"],
            )
            for item in json["data"]
        ]

    @staticmethod
    def to_run(json: Dict[str, Any]) -> TrismikRun:
        """
        Convert JSON response to a TrismikRun object.

        Args:
            json (Dict[str, Any]): JSON response containing run data.

        Returns:
            TrismikRun: Run object with ID, URL, and status.
        """
        return TrismikRun(
            id=json["id"],
            url=json["url"],
            status=json["status"],
        )

    @staticmethod
    def to_run_info(json: Dict[str, Any]) -> TrismikRunInfo:
        """
        Convert JSON response to a TrismikRunInfo object.

        Args:
            json (Dict[str, Any]): JSON response containing run info.

        Returns:
            TrismikRunInfo: Run info object with ID.
        """
        return TrismikRunInfo(id=json["id"])

    @staticmethod
    def to_run_state(json: Dict[str, Any]) -> TrismikRunState:
        """
        Convert JSON response to a TrismikRunState object.

        Args:
            json (Dict[str, Any]): JSON response containing run state.

        Returns:
            TrismikRunState: Run state object.
        """
        return TrismikRunState(
            responses=json.get("responses", []),
            thetas=json.get("thetas", []),
            std_error_history=json.get("std_error_history", []),
            kl_info_history=json.get("kl_info_history", []),
            effective_difficulties=json.get("effective_difficulties", []),
        )

    @staticmethod
    def to_run_response(json: Dict[str, Any]) -> TrismikRunResponse:
        """
        Convert JSON response to a TrismikRunResponse object.

        Args:
            json (Dict[str, Any]): JSON response from run endpoints.

        Returns:
            TrismikRunResponse: Run response.
        """
        return TrismikRunResponse(
            run_info=TrismikResponseMapper.to_run_info(json["runInfo"]),
            state=TrismikResponseMapper.to_run_state(json["state"]),
            next_item=(
                TrismikResponseMapper.to_item(json["nextItem"]) if json.get("nextItem") else None
            ),
            completed=json.get("completed", False),
        )

    @staticmethod
    def to_run_summary(json: Dict[str, Any]) -> TrismikRunSummary:
        """
        Convert JSON response to a TrismikRunSummary object.

        Args:
            json (Dict[str, Any]): JSON response from run summary endpoint.

        Returns:
            TrismikRunSummary: Complete run summary.
        """
        return TrismikRunSummary(
            id=json["id"],
            dataset_id=json["datasetId"],
            state=TrismikResponseMapper.to_run_state(json["state"]),
            dataset=[TrismikResponseMapper.to_item(item) for item in json.get("dataset", [])],
            responses=TrismikResponseMapper.to_responses(json.get("responses", [])),
            metadata=json.get("metadata", {}),
        )

    @staticmethod
    def to_item(json: Dict[str, Any]) -> TrismikItem:
        """
        Convert JSON response to a TrismikItem object.

        Args:
            json (Dict[str, Any]): JSON response containing item data.

        Returns:
            TrismikItem: Item object with question and choices.

        Raises:
            TrismikApiError: If the item type is not recognized.
        """
        if "question" in json and "choices" in json:
            return TrismikMultipleChoiceTextItem(
                id=json["id"],
                question=json["question"],
                choices=[
                    TrismikTextChoice(
                        id=choice["id"],
                        text=choice["value"],
                    )
                    for choice in json["choices"]
                ],
            )
        else:
            item_type = json.get("type", "unknown")
            raise TrismikApiError(f"API has returned unrecognized item type: {item_type}")

    @staticmethod
    def to_results(json: List[Dict[str, Any]]) -> List[TrismikResult]:
        """
        Convert JSON response to a list of TrismikResult objects.

        Args:
            json (List[Dict[str, Any]]): JSON response containing result data.

        Returns:
            List[TrismikResult]: List of result objects with trait, name, and
            value.
        """
        return [
            TrismikResult(
                trait=item["trait"],
                name=item["name"],
                value=item["value"],
            )
            for item in json
        ]

    @staticmethod
    def to_responses(json: List[Dict[str, Any]]) -> List[TrismikResponse]:
        """
        Convert JSON response to a list of TrismikResponse objects.

        Args:
            json (List[Dict[str, Any]]): JSON response containing response data.

        Returns:
            List[TrismikResponse]: List of response objects with dataset item
                ID, value, and correctness.
        """
        return [
            TrismikResponse(
                dataset_item_id=response["datasetItemId"],
                value=response["value"],
                correct=response["correct"],
            )
            for response in json
        ]

    @staticmethod
    def to_replay_response(json: Dict[str, Any]) -> TrismikReplayResponse:
        """
        Convert JSON response to a TrismikReplayResponse object.

        Args:
            json (Dict[str, Any]): JSON response from replay endpoint.

        Returns:
            TrismikReplayResponse: Replay response object.
        """
        # Parse datetime strings if they exist
        completed_at = None
        if "completedAt" in json and json["completedAt"]:
            completed_at = datetime.fromisoformat(json["completedAt"].replace("Z", "+00:00"))

        created_at = None
        if "createdAt" in json and json["createdAt"]:
            created_at = datetime.fromisoformat(json["createdAt"].replace("Z", "+00:00"))

        return TrismikReplayResponse(
            id=json["id"],
            datasetId=json["datasetId"],
            state=TrismikResponseMapper.to_run_state(json["state"]),
            replay_of_run=json["replayOfRun"],
            completedAt=completed_at,
            createdAt=created_at,
            metadata=json.get("metadata", {}),
            dataset=[TrismikResponseMapper.to_item(item) for item in json.get("dataset", [])],
            responses=TrismikResponseMapper.to_responses(json.get("responses", [])),
        )

    @staticmethod
    def to_me_response(json: Dict[str, Any]) -> TrismikMeResponse:
        """
        Convert JSON response to a TrismikMeResponse object.

        Args:
            json (Dict[str, Any]): JSON response from /admin/api-keys/me
                endpoint.

        Returns:
            TrismikMeResponse: Me response object.
        """
        user_data = json["user"]
        teams_data = json["teams"]

        user_info = TrismikUserInfo(
            id=user_data["id"],
            email=user_data["email"],
            firstname=user_data["firstname"],
            lastname=user_data["lastname"],
            createdAt=user_data.get("createdAt"),
            account_id=user_data.get("accountId"),
        )

        teams = [
            TrismikTeam(
                id=team_data["id"],
                name=team_data["name"],
                role=team_data["role"],
                account_id=team_data["accountId"],
            )
            for team_data in teams_data
        ]

        return TrismikMeResponse(user=user_info, teams=teams)

    @staticmethod
    def to_classic_eval_response(json: Dict[str, Any]) -> TrismikClassicEvalResponse:
        """
        Convert JSON response to a TrismikClassicEvalResponse object.

        Args:
            json (Dict[str, Any]): JSON response from classic evaluation
                endpoint.

        Returns:
            TrismikClassicEvalResponse: Classic evaluation response object.
        """
        user_data = json["user"]
        user_info = TrismikUserInfo(
            id=user_data["id"],
            email=user_data["email"],
            firstname=user_data["firstname"],
            lastname=user_data["lastname"],
            account_id=user_data.get("accountId"),
        )

        return TrismikClassicEvalResponse(
            id=json["id"],
            accountId=json["accountId"],
            projectId=json["projectId"],
            experimentId=json["experimentId"],
            experimentName=json["experimentName"],
            datasetId=json["datasetId"],
            userId=json["userId"],
            type=json["type"],
            modelName=json["modelName"],
            hyperparameters=json.get("hyperparameters", {}),
            createdAt=json["createdAt"],
            user=user_info,
            responseCount=json["responseCount"],
        )

    @staticmethod
    def to_project(json: Dict[str, Any]) -> TrismikProject:
        """
        Convert JSON response to a TrismikProject object.

        Args:
            json (Dict[str, Any]): JSON response from project creation endpoint.

        Returns:
            TrismikProject: Project object.
        """
        return TrismikProject(
            id=json["id"],
            name=json["name"],
            description=json.get("description"),
            accountId=json["accountId"],
            createdAt=json["createdAt"],
            updatedAt=json["updatedAt"],
        )
