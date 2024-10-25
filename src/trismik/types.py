from dataclasses import dataclass
from datetime import datetime
from typing import List, Any


@dataclass
class TrismikAuth:
    """
    Authentication token.

    Attributes:
        token (str): Authentication token value.
        expires (datetime): Expiration date.
    """
    token: str
    expires: datetime


@dataclass
class TrismikTest:
    """
    Available test.

    Attributes:
        id (str): Test ID.
        name (str): Test name.
    """
    id: str
    name: str


@dataclass
class TrismikSession:
    """
    Test session.

    Attributes:
        id (str): Session ID.
        url (str): Session URL.
        status (str): Session status
    """
    id: str
    url: str
    status: str


@dataclass
class TrismikItem:
    """
    Base class for test items.

    Attributes:
        id (str): Item ID.
    """
    id: str


@dataclass
class TrismikChoice:
    """
    Base class for choices in items that use them.

    Attributes:
        id (str): Choice ID.
    """
    id: str


@dataclass
class TrismikTextChoice(TrismikChoice):
    """
    Text choice.

    Attributes:
        text (str): Choice text.
    """
    text: str


@dataclass
class TrismikMultipleChoiceTextItem(TrismikItem):
    """
    Multiple choice text item.

    Attributes:
        question (str): Question text.
        choices (List[TrismikTextChoice]): List of choices.
    """
    question: str
    choices: List[TrismikTextChoice]


@dataclass
class TrismikResult:
    """
    Test result.

    Attributes:
        trait (str): Trait name.
        name (str): Result name/type.
        value (Any): Result value.
    """
    trait: str
    name: str
    value: Any


@dataclass
class TrismikResponse:
    """
    Test result.

    Attributes:
        item_id (str): Item ID.
        value (Any): Result value.
        score (float): Score.
    """
    item_id: str
    value: Any
    score: float


@dataclass
class TrismikResultsAndResponses:
    """
    Test results and responses.

    Attributes:
        results (List[TrismikResult]): Results.
        responses (List[TrismikResponse]): Responses.
    """
    results: List[TrismikResult]
    responses: List[TrismikResponse]
