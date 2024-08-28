from dataclasses import dataclass
from datetime import datetime
from typing import List, Any


@dataclass
class TrismikAuthResponse:
    token: str
    expires: datetime


@dataclass
class TrismikTest:
    id: str
    name: str


@dataclass
class TrismikSession:
    id: str
    url: str
    status: str


@dataclass
class TrismikResult:
    trait: str
    name: str
    value: Any


@dataclass
class TrismikItem:
    question: str


@dataclass
class TrismikChoice:
    id: str


@dataclass
class TrismikTextChoice(TrismikChoice):
    text: str


@dataclass
class TrismikMultipleChoiceTextItem(TrismikItem):
    choices: List[TrismikTextChoice]
