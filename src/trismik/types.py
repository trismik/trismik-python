"""
Type definitions for the Trismik client.

This module defines the data structures used throughout the Trismik client
library.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TrismikAuth:
    """Authentication token and expiration time."""

    token: str
    expires: datetime


@dataclass
class TrismikTest:
    """Test metadata including ID and name."""

    id: str
    name: str


@dataclass
class TrismikSession:
    """Session metadata including ID, URL, and status."""

    id: str
    url: str
    status: str


@dataclass
class TrismikItem:
    """Base class for test items."""

    id: str


@dataclass
class TrismikChoice:
    """Base class for choices in items that use them."""

    id: str


@dataclass
class TrismikTextChoice(TrismikChoice):
    """Text choice for multiple choice questions."""

    text: str


@dataclass
class TrismikMultipleChoiceTextItem(TrismikItem):
    """Multiple choice text question."""

    question: str
    choices: List[TrismikTextChoice]


@dataclass
class TrismikResult:
    """Test result for a specific trait."""

    trait: str
    name: str
    value: Any


@dataclass
class TrismikResponse:
    """Response to a test item."""

    item_id: str
    value: Any
    score: float


@dataclass
class TrismikRunResults:
    """Test results and responses."""

    session_id: str
    results: List[TrismikResult]
    responses: Optional[List[TrismikResponse]] = None


@dataclass
class TrismikSessionMetadata:
    """Metadata for a test session."""

    class ModelMetadata:
        """Model metadata for a test session."""

        def __init__(self, name: str, **kwargs: Any):
            """Initialize ModelMetadata with a name and optional attributes."""
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

    model_metadata: ModelMetadata
    test_configuration: dict[str, Any]
    inference_setup: dict[str, Any]

    def toDict(self) -> Dict[str, Any]:
        """Convert session metadata to a dictionary."""
        return {
            "model_metadata": vars(self.model_metadata),
            "test_configuration": self.test_configuration,
            "inference_setup": self.inference_setup,
        }
