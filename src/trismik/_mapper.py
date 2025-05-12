from typing import List, Any, Dict

from dateutil.parser import parse as parse_date

from trismik.exceptions import TrismikApiError
from trismik.types import (
    TrismikItem,
    TrismikResult,
    TrismikMultipleChoiceTextItem,
    TrismikTextChoice,
    TrismikAuth,
    TrismikTest,
    TrismikSession,
    TrismikResponse,
    TrismikSessionMetadata
)


class TrismikResponseMapper:

    @staticmethod
    def to_auth(json: dict[str, Any]) -> TrismikAuth:
        return TrismikAuth(
                token=json["token"],
                expires=parse_date(json["expires"]),
        )

    @staticmethod
    def to_tests(json: List[dict[str, Any]]) -> List[TrismikTest]:
        return [
            TrismikTest(
                    id=item["id"],
                    name=item["name"],
            ) for item in json
        ]

    @staticmethod
    def to_session(json: dict[str, Any]) -> TrismikSession:
        return TrismikSession(
                id=json["id"],
                url=json["url"],
                status=json["status"],
        )

    @staticmethod
    def to_item(json: dict[str, Any]) -> TrismikItem:
        if json["type"] == "multiple_choice_text":
            return TrismikMultipleChoiceTextItem(
                    id=json["id"],
                    question=json["question"],
                    choices=[
                        TrismikTextChoice(
                                id=choice["id"],
                                text=choice["text"],
                        ) for choice in json["choices"]
                    ]
            )
        else:
            raise TrismikApiError(
                    f"API has returned unrecognized item type: {json['type']}")

    @staticmethod
    def to_results(json: List[dict[str, Any]]) -> List[TrismikResult]:
        return [
            TrismikResult(
                    trait=item["trait"],
                    name=item["name"],
                    value=item["value"],
            ) for item in json
        ]

    @staticmethod
    def to_responses(json: List[dict[str, Any]]) -> List[TrismikResponse]:
        return [
            TrismikResponse(
                    item_id=response["itemId"],
                    value=response["value"],
                    score=response["score"],
            ) for response in json
        ]
