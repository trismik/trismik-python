"""
Type definitions for the Trismik client.

This module defines the data structures used throughout the Trismik client
library.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
class TrismikSessionInfo:
    """Session info from new API endpoints."""

    id: str


@dataclass
class TrismikSessionState:
    """Session state including responses, thetas, and other metrics."""

    responses: List[str]
    thetas: List[float]
    std_error_history: List[float]
    kl_info_history: List[float]
    effective_difficulties: List[float]


@dataclass
class TrismikSessionResponse:
    """Response from session endpoints (start and continue)."""

    session_info: TrismikSessionInfo
    state: TrismikSessionState
    next_item: Optional["TrismikItem"]
    completed: bool


@dataclass
class TrismikAdaptiveTestState:
    """State tracking for adaptive tests."""

    session_id: str
    state: TrismikSessionState
    completed: bool


@dataclass
class AdaptiveTestScore:
    """Final scores of an adaptive test run."""

    thetas: List[float]
    std_error_history: List[float]
    kl_info_history: List[float]


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
    responses: Optional[List[TrismikResponse]] = None
    score: Optional[AdaptiveTestScore] = None


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
    test_configuration: Dict[str, Any]
    inference_setup: Dict[str, Any]

    def toDict(self) -> Dict[str, Any]:
        """Convert session metadata to a dictionary."""
        return {
            "model_metadata": vars(self.model_metadata),
            "test_configuration": self.test_configuration,
            "inference_setup": self.inference_setup,
        }
