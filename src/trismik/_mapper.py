from datetime import datetime
from typing import Any, Dict, List

from trismik.exceptions import TrismikApiError
from trismik.types import (
    TrismikDataset,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikReplayResponse,
    TrismikResponse,
    TrismikResult,
    TrismikSession,
    TrismikSessionInfo,
    TrismikSessionResponse,
    TrismikSessionState,
    TrismikSessionSummary,
    TrismikTextChoice,
)


class TrismikResponseMapper:
    """
    Maps JSON responses from the Trismik API to Python objects.

    This class provides static methods to convert JSON responses from various
    API endpoints into their corresponding Python object representations.
    """

    @staticmethod
    def to_tests(json: List[Dict[str, Any]]) -> List[TrismikDataset]:
        """
        Convert JSON response to a list of TrismikTest objects.

        Args:
            json (List[Dict[str, Any]]): JSON response containing test data.

        Returns:
            List[TrismikDataset]: List of
            dataset objects with IDs and names.
        """
        return [
            TrismikDataset(
                id=item["id"],
                name=item["name"],
            )
            for item in json
        ]

    @staticmethod
    def to_session(json: Dict[str, Any]) -> TrismikSession:
        """
        Convert JSON response to a TrismikSession object.

        Args:
            json (Dict[str, Any]): JSON response containing session data.

        Returns:
            TrismikSession: Session object with ID, URL, and status.
        """
        return TrismikSession(
            id=json["id"],
            url=json["url"],
            status=json["status"],
        )

    @staticmethod
    def to_session_info(json: Dict[str, Any]) -> TrismikSessionInfo:
        """
        Convert JSON response to a TrismikSessionInfo object.

        Args:
            json (Dict[str, Any]): JSON response containing session info.

        Returns:
            TrismikSessionInfo: Session info object with ID.
        """
        return TrismikSessionInfo(id=json["id"])

    @staticmethod
    def to_session_state(json: Dict[str, Any]) -> TrismikSessionState:
        """
        Convert JSON response to a TrismikSessionState object.

        Args:
            json (Dict[str, Any]): JSON response containing session state.

        Returns:
            TrismikSessionState: Session state object.
        """
        return TrismikSessionState(
            responses=json.get("responses", []),
            thetas=json.get("thetas", []),
            std_error_history=json.get("std_error_history", []),
            kl_info_history=json.get("kl_info_history", []),
            effective_difficulties=json.get("effective_difficulties", []),
        )

    @staticmethod
    def to_session_response(json: Dict[str, Any]) -> TrismikSessionResponse:
        """
        Convert JSON response to a TrismikSessionResponse object.

        Args:
            json (Dict[str, Any]): JSON response from session endpoints.

        Returns:
            TrismikSessionResponse: Session response.
        """
        return TrismikSessionResponse(
            session_info=TrismikResponseMapper.to_session_info(
                json["sessionInfo"]
            ),
            state=TrismikResponseMapper.to_session_state(json["state"]),
            next_item=(
                TrismikResponseMapper.to_item(json["nextItem"])
                if json.get("nextItem")
                else None
            ),
            completed=json.get("completed", False),
        )

    @staticmethod
    def to_session_summary(json: Dict[str, Any]) -> TrismikSessionSummary:
        """
        Convert JSON response to a TrismikSessionSummary object.

        Args:
            json (Dict[str, Any]): JSON response from session summary endpoint.

        Returns:
            TrismikSessionSummary: Complete session summary.
        """
        return TrismikSessionSummary(
            id=json["id"],
            test_id=json["testId"],
            state=TrismikResponseMapper.to_session_state(json["state"]),
            dataset=[
                TrismikResponseMapper.to_item(item)
                for item in json.get("dataset", [])
            ],
            responses=TrismikResponseMapper.to_responses(
                json.get("responses", [])
            ),
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
            raise TrismikApiError(
                f"API has returned unrecognized item type: {item_type}"
            )

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
            completed_at = datetime.fromisoformat(
                json["completedAt"].replace("Z", "+00:00")
            )

        created_at = None
        if "createdAt" in json and json["createdAt"]:
            created_at = datetime.fromisoformat(
                json["createdAt"].replace("Z", "+00:00")
            )

        return TrismikReplayResponse(
            id=json["id"],
            testId=json["testId"],
            state=TrismikResponseMapper.to_session_state(json["state"]),
            replay_of_session=json["replay_of_session"],
            completedAt=completed_at,
            createdAt=created_at,
            metadata=json.get("metadata", {}),
            dataset=[
                TrismikResponseMapper.to_item(item)
                for item in json.get("dataset", [])
            ],
            responses=TrismikResponseMapper.to_responses(
                json.get("responses", [])
            ),
        )
