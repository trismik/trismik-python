"""
Trismik Python Client.

A Python client for the Trismik API.
"""

import importlib.metadata

from trismik._async.client import TrismikAsyncClient
from trismik._async.quick_compare import TrismikQuickCompareAsyncClient
from trismik._sync.client import TrismikClient
from trismik._sync.quick_compare import TrismikQuickCompareClient
from trismik.exceptions import (
    TrismikApiError,
    TrismikError,
    TrismikNotFoundError,
    TrismikPayloadTooLargeError,
    TrismikRateLimitedError,
    TrismikValidationError,
)
from trismik.types import (
    AdaptiveTestScore,
    TrismikAccountBalance,
    TrismikDataset,
    TrismikEvaluation,
    TrismikEvaluationBatch,
    TrismikEvaluationDataset,
    TrismikEvaluationDatasetHF,
    TrismikEvaluationDatasetUser,
    TrismikEvaluationExportUrl,
    TrismikEvaluationMetric,
    TrismikEvaluationMetricAccuracy,
    TrismikEvaluationMetricBleu,
    TrismikEvaluationMetricExactMatch,
    TrismikEvaluationMetricLLMAJ,
    TrismikEvaluationMetricRougeL,
    TrismikEvaluationModelConfig,
    TrismikEvaluationModelInput,
    TrismikEvaluationModelInputColumn,
    TrismikEvaluationModelInputTemplate,
    TrismikEvaluationProgress,
    TrismikEvaluationScore,
    TrismikEvaluationSummary,
    TrismikItem,
    TrismikMeResponse,
    TrismikMultipleChoiceTextItem,
    TrismikOpenEndedTextItem,
    TrismikPage,
    TrismikPageMeta,
    TrismikProject,
    TrismikRunMetadata,
    TrismikRunResults,
    TrismikUserDataset,
    TrismikUserDatasetColumn,
    TrismikUserDatasetDetail,
    TrismikUserDatasetInitResponse,
    TrismikUserDatasetPresignedPost,
)

# get version from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [
    # Clients
    "TrismikAsyncClient",
    "TrismikClient",
    "TrismikQuickCompareAsyncClient",
    "TrismikQuickCompareClient",
    # Exceptions
    "TrismikError",
    "TrismikApiError",
    "TrismikNotFoundError",
    "TrismikPayloadTooLargeError",
    "TrismikRateLimitedError",
    "TrismikValidationError",
    # Adaptive testing types
    "AdaptiveTestScore",
    "TrismikDataset",
    "TrismikItem",
    "TrismikMeResponse",
    "TrismikMultipleChoiceTextItem",
    "TrismikOpenEndedTextItem",
    "TrismikProject",
    "TrismikRunMetadata",
    "TrismikRunResults",
    # Pagination
    "TrismikPage",
    "TrismikPageMeta",
    # Quick Compare - user datasets
    "TrismikUserDataset",
    "TrismikUserDatasetColumn",
    "TrismikUserDatasetDetail",
    "TrismikUserDatasetInitResponse",
    "TrismikUserDatasetPresignedPost",
    # Quick Compare - evaluation polymorphic variants
    "TrismikEvaluationDataset",
    "TrismikEvaluationDatasetHF",
    "TrismikEvaluationDatasetUser",
    "TrismikEvaluationModelInput",
    "TrismikEvaluationModelInputColumn",
    "TrismikEvaluationModelInputTemplate",
    "TrismikEvaluationMetric",
    "TrismikEvaluationMetricAccuracy",
    "TrismikEvaluationMetricExactMatch",
    "TrismikEvaluationMetricBleu",
    "TrismikEvaluationMetricRougeL",
    "TrismikEvaluationMetricLLMAJ",
    "TrismikEvaluationModelConfig",
    # Quick Compare - evaluation core
    "TrismikEvaluation",
    "TrismikEvaluationSummary",
    "TrismikEvaluationProgress",
    "TrismikEvaluationBatch",
    "TrismikEvaluationScore",
    "TrismikEvaluationExportUrl",
    # Quick Compare - account
    "TrismikAccountBalance",
    # Version
    "__version__",
]
