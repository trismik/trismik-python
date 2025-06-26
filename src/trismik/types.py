"""
Type definitions for the Trismik client.

This module defines the data structures used throughout the Trismik client
library.
"""

from dataclasses import dataclass, field
from datetime import datetime
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

    theta: float
    std_error: float


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

    dataset_item_id: str
    value: Any
    correct: bool


@dataclass
class TrismikRunResults:
    """Test results and responses."""

    session_id: str
    score: Optional[AdaptiveTestScore] = None
    responses: Optional[List[TrismikResponse]] = None


@dataclass
class TrismikSessionSummary:
    """Complete session summary."""

    id: str
    test_id: str
    state: TrismikSessionState
    dataset: List[TrismikItem]
    responses: List[TrismikResponse]
    metadata: dict

    @property
    def theta(self) -> float:
        """Get the theta of the session."""
        return self.state.thetas[-1]

    @property
    def std_error(self) -> float:
        """Get the standard error of the session."""
        return self.state.std_error_history[-1]

    @property
    def total_responses(self) -> int:
        """Get the total number of responses in the session."""
        return len(self.responses)

    @property
    def correct_responses(self) -> int:
        """Get the number of correct responses in the session."""
        return sum(response.correct for response in self.responses)

    @property
    def wrong_responses(self) -> int:
        """Get the number of wrong responses in the session."""
        return self.total_responses - self.correct_responses


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


@dataclass
class TrismikReplayRequestItem:
    """Item in a replay request."""

    itemId: str
    itemChoiceId: str


@dataclass
class TrismikReplayRequest:
    """Request to replay a session with specific responses."""

    responses: List[TrismikReplayRequestItem]


@dataclass
class TrismikReplayResponse:
    """Response from a replay session request."""

    id: str
    testId: str
    state: TrismikSessionState
    replay_of_session: str
    completedAt: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset: List[TrismikItem] = field(default_factory=list)
    responses: List[TrismikResponse] = field(default_factory=list)
