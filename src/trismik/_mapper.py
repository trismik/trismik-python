from typing import Any, Dict, List

from dateutil.parser import parse as parse_date

from trismik.exceptions import TrismikApiError
from trismik.types import (
    TrismikAuth,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikResponse,
    TrismikResult,
    TrismikSession,
    TrismikTest,
    TrismikTextChoice,
)


class TrismikResponseMapper:
    """
    Maps JSON responses from the Trismik API to Python objects.

    This class provides static methods to convert JSON responses from various
    API endpoints into their corresponding Python object representations.
    """

    @staticmethod
    def to_auth(json: Dict[str, Any]) -> TrismikAuth:
        """
        Convert JSON response to a TrismikAuth object.

        Args:
            json (Dict[str, Any]): JSON response containing auth data.

        Returns:
            TrismikAuth: Authentication object with token and expiration.
        """
        return TrismikAuth(
            token=json["token"],
            expires=parse_date(json["expires"]),
        )

    @staticmethod
    def to_tests(json: List[Dict[str, Any]]) -> List[TrismikTest]:
        """
        Convert JSON response to a list of TrismikTest objects.

        Args:
            json (List[Dict[str, Any]]): JSON response containing test data.

        Returns:
            List[TrismikTest]: List of test objects with IDs and names.
        """
        return [
            TrismikTest(
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
        if json["type"] == "multiple_choice_text":
            return TrismikMultipleChoiceTextItem(
                id=json["id"],
                question=json["question"],
                choices=[
                    TrismikTextChoice(
                        id=choice["id"],
                        text=choice["text"],
                    )
                    for choice in json["choices"]
                ],
            )
        else:
            raise TrismikApiError(
                f"API has returned unrecognized item type: {json['type']}"
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
            List[TrismikResponse]: List of response objects with item ID, value,
                and score.
        """
        return [
            TrismikResponse(
                item_id=response["itemId"],
                value=response["value"],
                score=response["score"],
            )
            for response in json
        ]
