"""
Type definitions for the Trismik client.

This module defines the data structures used throughout the Trismik client
library.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class TrismikDataset:
    """Test metadata including ID and name."""

    id: str
    name: str


@dataclass
class TrismikRun:
    """Run metadata including ID, URL, and status."""

    id: str
    url: str
    status: str


@dataclass
class TrismikRunInfo:
    """Run info from new API endpoints."""

    id: str


@dataclass
class TrismikRunState:
    """Run state including responses, thetas, and other metrics."""

    responses: List[str]
    thetas: List[float]
    std_error_history: List[float]
    kl_info_history: List[float]
    effective_difficulties: List[float]


@dataclass
class TrismikRunResponse:
    """Response from run endpoints (start and continue)."""

    run_info: TrismikRunInfo
    state: TrismikRunState
    next_item: Optional["TrismikItem"]
    completed: bool


@dataclass
class TrismikAdaptiveTestState:
    """State tracking for adaptive tests."""

    run_id: str
    state: TrismikRunState
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

    run_id: str
    score: Optional[AdaptiveTestScore] = None
    responses: Optional[List[TrismikResponse]] = None


@dataclass
class TrismikRunSummary:
    """Complete run summary."""

    id: str
    dataset_id: str
    state: TrismikRunState
    dataset: List[TrismikItem]
    responses: List[TrismikResponse]
    metadata: dict

    @property
    def theta(self) -> float:
        """Get the theta of the run."""
        return self.state.thetas[-1]

    @property
    def std_error(self) -> float:
        """Get the standard error of the run."""
        return self.state.std_error_history[-1]

    @property
    def total_responses(self) -> int:
        """Get the total number of responses in the run."""
        return len(self.responses)

    @property
    def correct_responses(self) -> int:
        """Get the number of correct responses in the run."""
        return sum(response.correct for response in self.responses)

    @property
    def wrong_responses(self) -> int:
        """Get the number of wrong responses in the run."""
        return self.total_responses - self.correct_responses


@dataclass
class TrismikRunMetadata:
    """Metadata for a test run."""

    class ModelMetadata:
        """Model metadata for a test run."""

        def __init__(self, name: str, **kwargs: Any):
            """Initialize ModelMetadata with a name and optional attributes."""
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

    model_metadata: ModelMetadata
    test_configuration: Dict[str, Any]
    inference_setup: Dict[str, Any]

    def toDict(self) -> Dict[str, Any]:
        """Convert run metadata to a dictionary."""
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
    """Request to replay a run with specific responses."""

    responses: List[TrismikReplayRequestItem]


@dataclass
class TrismikReplayResponse:
    """Response from a replay run request."""

    id: str
    datasetId: str
    state: TrismikRunState
    replay_of_run: str
    completedAt: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset: List[TrismikItem] = field(default_factory=list)
    responses: List[TrismikResponse] = field(default_factory=list)


@dataclass
class TrismikTeam:
    """Team information."""

    id: str
    name: str
    role: str
    account_id: str


@dataclass
class TrismikUserInfo:
    """User information."""

    id: str
    email: str
    firstname: str
    lastname: str
    account_id: str
    createdAt: Optional[str] = None


@dataclass
class TrismikMeResponse:
    """Response from the /admin/api-keys/me endpoint."""

    user: TrismikUserInfo
    teams: List[TrismikTeam]


@dataclass
class TrismikClassicEvalItem:
    """Item in a classic evaluation request."""

    datasetItemId: str
    modelInput: str
    modelOutput: str
    goldOutput: str
    metrics: Dict[str, Any]


@dataclass
class TrismikClassicEvalMetric:
    """Metric in a classic evaluation request."""

    metricId: str
    value: Union[str, float, int, bool]


@dataclass
class TrismikClassicEvalRequest:
    """Request to submit a classic evaluation."""

    projectId: str
    experimentName: str
    datasetId: str
    modelName: str
    hyperparameters: Dict[str, Any]
    items: List[TrismikClassicEvalItem]
    metrics: List[TrismikClassicEvalMetric]


@dataclass
class TrismikClassicEvalResponse:
    """Response from a classic evaluation submission."""

    id: str
    accountId: str
    projectId: str
    experimentId: str
    experimentName: str
    datasetId: str
    userId: str
    type: str
    modelName: str
    hyperparameters: Dict[str, Any]
    createdAt: str
    user: TrismikUserInfo
    responseCount: int


@dataclass
class TrismikProjectRequest:
    """Request to create a new project."""

    name: str
    description: Optional[str] = None


@dataclass
class TrismikProject:
    """Project information."""

    id: str
    name: str
    description: Optional[str]
    accountId: str
    createdAt: str
    updatedAt: str
