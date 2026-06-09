"""
Type definitions for the Trismik client.

This module defines the data structures used throughout the Trismik client
library.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union


@dataclass
class TrismikDataset:
    """Test metadata including ID and name."""

    id: str
    name: str
    isAdaptive: bool
    splits: List[str]


@dataclass
class TrismikDatasetVersion:
    """Version information for a dataset."""

    year: int
    month: int
    revision: int


@dataclass
class TrismikDatacard:
    """Detailed metadata card for a dataset."""

    task: str
    license: str
    version: "TrismikDatasetVersion"
    datasetHf: str
    languages: List[str]
    datasetUrl: str
    itemsCount: int
    splitsUsed: List[str]
    datasetName: str
    description: str
    exampleItem: str
    originalReadmeUrl: str
    originalReadmeDownloadedAt: str
    originalItemCount: int
    difficultyEstimators: List[str]


@dataclass
class TrismikDatasetInfo:
    """Complete dataset information including datacard."""

    id: str
    name: str
    isAdaptive: bool
    splits: List[str]
    datacard: TrismikDatacard


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
class TrismikOpenEndedTextItem(TrismikItem):
    """Open-ended text question (no choices)."""

    question: str
    reference: Optional[str] = None
    response_text: Optional[str] = None


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
    dataset_item_type: Optional[str] = None

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
    itemChoiceId: Optional[str] = None
    textResponse: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Quick-Compare feature: user datasets, evaluations, export, account balance.
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class TrismikPageMeta:
    """Pagination metadata returned by paginated list endpoints."""

    page: int
    limit: int
    total_items: int
    total_pages: int
    has_next_page: bool
    has_previous_page: bool


@dataclass
class TrismikPage(Generic[T]):
    """Generic paginated page of items."""

    data: List[T]
    meta: TrismikPageMeta


# --- User datasets ---------------------------------------------------------


FileType = Literal["csv", "tsv", "json", "jsonl", "parquet"]


@dataclass
class TrismikUserDatasetPresignedPost:
    """Presigned S3 endpoint for uploading a user dataset payload."""

    url: str
    fields: Dict[str, str]
    method: Literal["POST", "PUT"] = "POST"
    expires_at: Optional[datetime] = None


@dataclass
class TrismikUserDatasetInitResponse:
    """Response from initializing a user dataset upload."""

    dataset_id: str
    name: str
    status: str
    upload: TrismikUserDatasetPresignedPost


@dataclass
class TrismikUserDatasetColumn:
    """Column metadata for a user dataset (name and inferred type)."""

    name: str
    type: str = ""


@dataclass
class TrismikUserDataset:
    """User dataset summary."""

    id: str
    name: str
    created_at: datetime
    status: Optional[str] = None
    s3_path: Optional[str] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    uploaded_by: Optional[str] = None
    updated_at: Optional[datetime] = None


@dataclass
class TrismikUserDatasetDetail:
    """User dataset with full column metadata."""

    id: str
    name: str
    created_at: datetime
    status: Optional[str] = None
    s3_path: Optional[str] = None
    file_type: Optional[str] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    uploaded_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    finalized_at: Optional[datetime] = None
    columns: List[TrismikUserDatasetColumn] = field(default_factory=list)
    preview: List[Dict[str, Any]] = field(default_factory=list)


# --- Evaluation dataset variants (polymorphic) -----------------------------


@dataclass
class TrismikEvaluationDatasetHF:
    """HuggingFace-hosted dataset reference for an evaluation."""

    repo: str
    sub_dataset: Optional[str] = None
    splits: Optional[List[str]] = None


@dataclass
class TrismikEvaluationDatasetUser:
    """User-uploaded dataset reference for an evaluation."""

    dataset_id: str


TrismikEvaluationDataset = Union[TrismikEvaluationDatasetHF, TrismikEvaluationDatasetUser]


# --- Evaluation model input variants (polymorphic) ------------------------


@dataclass
class TrismikEvaluationModelInputColumn:
    """Use a single dataset column as the model input."""

    column: str


@dataclass
class TrismikEvaluationModelInputTemplate:
    """Use a Mustache-style template to compose the model input."""

    template: str


TrismikEvaluationModelInput = Union[
    TrismikEvaluationModelInputColumn,
    TrismikEvaluationModelInputTemplate,
]


# --- Evaluation metric variants (polymorphic) ------------------------------


@dataclass
class TrismikEvaluationMetricAccuracy:
    """Accuracy metric against a gold column."""

    gold_column: str


@dataclass
class TrismikEvaluationMetricExactMatch:
    """Exact-match metric against a gold column."""

    gold_column: str
    normalize: bool = True


@dataclass
class TrismikEvaluationMetricBleu:
    """BLEU metric against a gold column."""

    gold_column: str


@dataclass
class TrismikEvaluationMetricRougeL:
    """ROUGE-L metric against a gold column."""

    gold_column: str


@dataclass
class TrismikJudge:
    """A single LLM-as-judge configuration.

    Each judge evaluates every model output and produces its own, independent
    score. ``name`` is a user-facing label and must be unique among the judges
    of an evaluation.
    """

    name: str
    judge_model: str
    rubric: str
    scale_min: int = 1
    scale_max: int = 5
    temperature: Optional[float] = None
    thinking: Optional[bool] = None


@dataclass
class TrismikEvaluationMetricLLMAJ:
    """LLM-as-judge metric.

    Carries one or more judges; each judge evaluates every model output and
    produces a separate score/difficulty. Must be the first metric in an
    evaluation when present (server constraint).

    Two ways to construct it:

    - Multi-judge: pass ``judges=[TrismikJudge(...), ...]``.
    - Single-judge (legacy, back-compatible): pass ``judge_model`` + ``rubric``
      (and optionally ``scale_min``/``scale_max``); a single judge is
      synthesised, named after the judge model.
    """

    judge_model: Optional[str] = None
    rubric: Optional[str] = None
    gold_column: Optional[str] = None
    scale_min: int = 1
    scale_max: int = 5
    judges: List[TrismikJudge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.judges:
            if self.judge_model is None or self.rubric is None:
                raise ValueError(
                    "LLMAJ metric requires either judges=[TrismikJudge(...)] "
                    "or judge_model + rubric."
                )
            self.judges = [
                TrismikJudge(
                    name=self.judge_model,
                    judge_model=self.judge_model,
                    rubric=self.rubric,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                )
            ]


TrismikEvaluationMetric = Union[
    TrismikEvaluationMetricAccuracy,
    TrismikEvaluationMetricExactMatch,
    TrismikEvaluationMetricBleu,
    TrismikEvaluationMetricRougeL,
    TrismikEvaluationMetricLLMAJ,
]


# --- Evaluation model configuration ---------------------------------------


@dataclass
class TrismikEvaluationModelConfig:
    """Configuration for a single model in an evaluation."""

    id: str
    thinking: bool = False
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# --- Evaluation core types -------------------------------------------------


@dataclass
class TrismikEvaluationBatch:
    """One model/split batch inside an evaluation."""

    id: str
    model: str
    status: str
    split: Optional[str] = None
    items_processed: Optional[int] = None
    items_total: Optional[int] = None
    failed_count: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class TrismikEvaluationScore:
    """Aggregated score for a single (model, metric) pair."""

    model: str
    metric: str
    value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class TrismikEvaluationProgress:
    """Progress snapshot for a running evaluation."""

    percentage: float
    cost_usd_used: float
    cost_usd_budget: float
    items_processed: Optional[int] = None
    items_total: Optional[int] = None


@dataclass
class TrismikEvaluationSummary:
    """Slim evaluation view used by list endpoints."""

    id: str
    name: str
    status: str
    created_at: datetime
    models: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None
    progress_percentage: Optional[float] = None
    budget_limit_usd: Optional[float] = None
    is_demo: bool = False
    updated_at: Optional[datetime] = None


@dataclass
class TrismikEvaluation:
    """Detailed evaluation view used by create / get / wait endpoints."""

    id: str
    name: str
    status: str
    created_at: datetime
    models: List[str] = field(default_factory=list)
    dataset_name: Optional[str] = None
    progress: Optional[TrismikEvaluationProgress] = None
    batches: List[TrismikEvaluationBatch] = field(default_factory=list)
    scores: List[TrismikEvaluationScore] = field(default_factory=list)
    error_message: Optional[str] = None
    budget_limit_usd: Optional[float] = None
    sample_limit: Optional[Dict[str, int]] = None
    total_cost_usd: Optional[float] = None
    is_demo: bool = False
    updated_at: Optional[datetime] = None


# --- Evaluation export -----------------------------------------------------


@dataclass
class TrismikEvaluationExportUrl:
    """Presigned download URL for evaluation results."""

    status: str
    format: Literal["parquet", "csv", "jsonl"]
    url: Optional[str] = None
    expires_at: Optional[datetime] = None


# --- Account balance -------------------------------------------------------


@dataclass
class TrismikAccountBalance:
    """Account balance including in-flight holds."""

    balance_usd: float
    held_usd: float
    available_usd: float
