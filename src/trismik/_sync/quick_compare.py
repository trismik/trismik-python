"""Client for the Trismik Quick Compare feature (async source of truth)."""

import time
from time import monotonic as _now
from typing import Any, Dict, List, Literal, Optional, Union

import httpx

from trismik._quick_compare_mapper import TrismikQuickCompareResponseMapper, evaluation_request_body
from trismik._sync._upload import _default_content_type, _read_source_to_bytes, upload_to_presigned
from trismik._utils import TrismikUtils, handle_http_error
from trismik.exceptions import TrismikApiError, TrismikRateLimitedError, TrismikValidationError
from trismik.settings import client_settings, environment_settings
from trismik.types import (
    TrismikAccountBalance,
    TrismikEvaluation,
    TrismikEvaluationDataset,
    TrismikEvaluationExportUrl,
    TrismikEvaluationMetric,
    TrismikEvaluationMetricLLMAJ,
    TrismikEvaluationModelConfig,
    TrismikEvaluationModelInput,
    TrismikEvaluationSummary,
    TrismikPage,
    TrismikUserDataset,
    TrismikUserDatasetDetail,
    TrismikUserDatasetInitResponse,
)

ProgressCallback = Optional[Any]


class TrismikQuickCompareClient:
    """Client for the Trismik Quick Compare API.

    Exposes the bank-maker-api endpoints (user datasets, evaluations, export)
    and the admin-api account balance endpoint. Can be instantiated standalone
    or obtained as a lazy accessor via :attr:`TrismikAsyncClient.quick_compare`,
    in which case it reuses the parent's HTTP client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        service_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize the Quick Compare client.

        Args:
            api_key: API key. Falls back to ``TRISMIK_API_KEY`` env var.
            service_url: Base service URL. Falls back to ``TRISMIK_SERVICE_URL``
                env var or the default endpoint.
            http_client: Optional pre-configured ``httpx.AsyncClient``. When
                provided, this instance does not own it and will not close it
                on ``aclose()``.

        Raises:
            TrismikError: If ``api_key`` is missing in both arguments and env.
        """
        self._service_url = TrismikUtils.option(
            service_url,
            client_settings["endpoint"],
            environment_settings["trismik_service_url"],
        )
        self._api_key = TrismikUtils.required_option(
            api_key, "api_key", environment_settings["trismik_api_key"]
        )
        default_headers = {"x-api-key": self._api_key}
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.Client(
            base_url=self._service_url, headers=default_headers, timeout=30.0
        )

    def __enter__(self) -> "TrismikQuickCompareClient":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Exit context manager and close client if owned."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._owns_client:
            self._http_client.close()

    # ------------------------------------------------------------------
    # User datasets — read
    # ------------------------------------------------------------------

    def list_user_datasets(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        sort_by: Optional[Literal["id", "name", "createdAt"]] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        search: Optional[str] = None,
    ) -> TrismikPage[TrismikUserDataset]:
        """List user datasets with pagination.

        Args:
            page: 1-based page number.
            limit: Page size (max 100 server-side).
            sort_by: Field to sort by.
            sort_order: Sort direction.
            search: Free-text search.

        Returns:
            Paginated list of user datasets.

        Raises:
            TrismikApiError: If the API request fails.
        """
        params = _build_list_params(page, limit, sort_by, sort_order, search)
        try:
            response = self._http_client.get("/bank-maker-api/public/datasets", params=params)
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_user_dataset_page(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def get_user_dataset(self, dataset_id: str) -> TrismikUserDatasetDetail:
        """Fetch a single user dataset with column metadata.

        Args:
            dataset_id: UUID of the dataset.

        Returns:
            Dataset detail including columns.

        Raises:
            TrismikNotFoundError: If the dataset does not exist (or belongs to
                another account).
            TrismikApiError: For other API failures.
        """
        try:
            response = self._http_client.get(f"/bank-maker-api/public/datasets/{dataset_id}")
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_user_dataset_detail(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def list_user_dataset_rows(
        self,
        dataset_id: str,
        *,
        page: int = 1,
        limit: int = 50,
    ) -> TrismikPage[Dict[str, Any]]:
        """List rows of a user dataset with pagination.

        Args:
            dataset_id: UUID of the dataset.
            page: 1-based page number.
            limit: Page size.

        Returns:
            Paginated rows (each row is a dict).

        Raises:
            TrismikNotFoundError: If the dataset does not exist.
            TrismikApiError: For other API failures.
        """
        params: Dict[str, Any] = {"page": page, "limit": limit}
        try:
            response = self._http_client.get(
                f"/bank-maker-api/public/datasets/{dataset_id}/rows",
                params=params,
            )
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_user_dataset_rows_page(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    # ------------------------------------------------------------------
    # User datasets — write & upload
    # ------------------------------------------------------------------

    def init_user_dataset(
        self,
        *,
        name: str,
        file_type: str,
        size_bytes: int,
    ) -> TrismikUserDatasetInitResponse:
        """Initialize an upload, receiving an S3 presigned POST.

        Args:
            name: Dataset name (1-255 chars).
            file_type: One of ``csv``, ``tsv``, ``json``, ``jsonl``, ``parquet``.
            size_bytes: Declared payload size in bytes; enforced by the S3
                policy on the subsequent upload.

        Returns:
            Dataset id, name, status, and presigned upload target.

        Raises:
            TrismikApiError: If the API request fails.
            TrismikValidationError: If the server rejects the payload (422).
        """
        body: Dict[str, Any] = {
            "name": name,
            "fileType": file_type,
            "sizeBytes": size_bytes,
        }
        try:
            response = self._http_client.post("/bank-maker-api/user-datasets", json=body)
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_user_dataset_init(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def finalize_user_dataset(self, dataset_id: str) -> TrismikUserDatasetDetail:
        """Finalize a dataset upload after the S3 POST completed.

        Args:
            dataset_id: UUID of the dataset returned by :meth:`init_user_dataset`.

        Returns:
            The dataset detail in its post-finalize ``ready`` state, including
            inferred columns, row count, and a preview.

        Raises:
            TrismikApiError: If the API request fails.
        """
        try:
            response = self._http_client.post(
                f"/bank-maker-api/user-datasets/{dataset_id}/finalize"
            )
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_user_dataset_detail(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def upload_user_dataset(
        self,
        source: Any,
        *,
        name: str,
        file_type: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> TrismikUserDatasetDetail:
        """Upload a dataset end-to-end (init → S3 → finalize).

        Args:
            source: File path (``str`` or ``Path``), raw ``bytes``, or a
                binary file-like object opened for reading.
            name: Dataset name.
            file_type: One of ``csv``, ``tsv``, ``json``, ``jsonl``, ``parquet``.
                When ``source`` is a path, inferred from the extension if
                omitted. Required for bytes/file-like sources.
            content_type: MIME type used on the S3 POST. Defaults derived from
                ``file_type`` (e.g. ``text/csv``, ``application/x-parquet``).

        Returns:
            The finalized dataset detail.

        Raises:
            TrismikValidationError: If ``file_type`` cannot be inferred.
            TrismikApiError: If init, S3 upload, or finalize fail. If S3 fails,
                ``finalize`` is **not** invoked.
        """
        data, inferred_file_type = _read_source_to_bytes(source)
        effective_file_type = file_type or inferred_file_type
        if effective_file_type is None:
            raise TrismikValidationError(
                "file_type is required for byte/file-like sources (cannot be inferred)."
            )
        effective_content_type = content_type or _default_content_type(effective_file_type)
        init = self.init_user_dataset(
            name=name, file_type=effective_file_type, size_bytes=len(data)
        )
        upload_to_presigned(init.upload, data, content_type=effective_content_type)
        return self.finalize_user_dataset(init.dataset_id)

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def create_evaluation(
        self,
        *,
        name: str,
        dataset: TrismikEvaluationDataset,
        model_input: TrismikEvaluationModelInput,
        metrics: List[TrismikEvaluationMetric],
        models: List[Union[str, TrismikEvaluationModelConfig]],
        budget_limit_usd: float,
        sample_limit: Optional[Dict[str, int]] = None,
        dataset_name: Optional[str] = None,
    ) -> TrismikEvaluation:
        """Create a new evaluation.

        Args:
            name: Evaluation name.
            dataset: Source dataset (HuggingFace or user dataset variant).
            model_input: How the model input is composed (column or template).
            metrics: Metrics to compute. If a LLM-as-judge metric is present
                it must be the only one of its kind and the first in the list
                (server constraint, validated client-side eagerly).
            models: Models to run. Can be plain strings (defaults
                ``thinking=False``) or :class:`TrismikEvaluationModelConfig`
                for fine-grained control.
            budget_limit_usd: Maximum spend in USD.
            sample_limit: Optional mapping from split name to sample cap.
            dataset_name: Optional human-readable dataset label. Defaults to
                the value of ``name``.

        Returns:
            The created evaluation record.

        Raises:
            TrismikValidationError: For client-side LLMaJ ordering violations or
                422 responses from the server.
            TrismikApiError: For other API failures.
        """
        _validate_metrics(metrics)
        body = evaluation_request_body(
            name=name,
            dataset_name=dataset_name or name,
            dataset=dataset,
            model_input=model_input,
            metrics=metrics,
            models=models,
            budget_limit_usd=budget_limit_usd,
            sample_limit=sample_limit,
        )
        try:
            response = self._http_client.post("/bank-maker-api/evaluations", json=body)
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_evaluation(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def get_evaluation(self, evaluation_id: str) -> TrismikEvaluation:
        """Fetch an evaluation by id.

        Args:
            evaluation_id: UUID of the evaluation.

        Returns:
            Detailed evaluation including progress, batches, and scores.

        Raises:
            TrismikNotFoundError: If the evaluation does not exist.
            TrismikRateLimitedError: If the per-resource rate limit was hit.
            TrismikApiError: For other API failures.
        """
        try:
            response = self._http_client.get(f"/bank-maker-api/evaluations/{evaluation_id}")
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_evaluation(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def list_evaluations(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        sort_by: Optional[Literal["id", "name", "status", "createdAt"]] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
        search: Optional[str] = None,
    ) -> TrismikPage[TrismikEvaluationSummary]:
        """List evaluations with pagination.

        Args:
            page: 1-based page number.
            limit: Page size.
            sort_by: Field to sort by.
            sort_order: Sort direction.
            search: Free-text search.

        Returns:
            Paginated slim view of evaluations.

        Raises:
            TrismikApiError: If the API request fails.
        """
        params = _build_list_params(page, limit, sort_by, sort_order, search)
        try:
            response = self._http_client.get("/bank-maker-api/evaluations", params=params)
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_evaluation_page(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def get_evaluation_export_url(
        self,
        evaluation_id: str,
        *,
        format: Literal["parquet", "csv", "jsonl"] = "parquet",
    ) -> TrismikEvaluationExportUrl:
        """Request a presigned download URL for evaluation results.

        Args:
            evaluation_id: UUID of the evaluation.
            format: Output format.

        Returns:
            Presigned URL and status. If status is ``"generating"`` the URL is
            not yet available and the call should be retried.

        Raises:
            TrismikNotFoundError: If the evaluation does not exist.
            TrismikApiError: For other API failures.
        """
        try:
            response = self._http_client.get(
                f"/bank-maker-api/evaluations/{evaluation_id}/export-url",
                params={"format": format},
            )
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_evaluation_export_url(
                response.json(), format=format
            )
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e

    def wait_for_evaluation(
        self,
        evaluation_id: str,
        *,
        poll_interval: float = 10.0,
        timeout: Optional[float] = 3600.0,
        on_progress: Optional[Any] = None,
    ) -> TrismikEvaluation:
        """Poll an evaluation until it reaches a terminal status.

        Args:
            evaluation_id: UUID of the evaluation to watch.
            poll_interval: Seconds between polls. Must be >= 6.0 to respect the
                server-side limit of 10 reads / 60 s per resource.
            timeout: Hard timeout in seconds. ``None`` disables it.
            on_progress: Optional callable invoked with the latest
                :class:`TrismikEvaluation` after each successful poll.

        Returns:
            The completed evaluation.

        Raises:
            ValueError: If ``poll_interval`` is below the safe lower bound.
            TrismikApiError: If the evaluation finishes in a failed state.
            TimeoutError: If ``timeout`` elapses before completion.
        """
        if poll_interval < 6.0:
            raise ValueError(
                "poll_interval must be >= 6.0s (server rate-limits: 10/60s per resource)."
            )
        deadline = (_now() + timeout) if timeout else None
        while True:
            try:
                evaluation = self.get_evaluation(evaluation_id)
            except TrismikRateLimitedError as e:
                time.sleep(e.retry_after_seconds or poll_interval * 2)
                continue
            if on_progress is not None:
                on_progress(evaluation)
            if evaluation.status == "completed":
                return evaluation
            if evaluation.status == "failed":
                message = evaluation.error_message or "no error message provided"
                raise TrismikApiError(f"Evaluation {evaluation_id} failed: {message}")
            if deadline is not None and _now() >= deadline:
                raise TimeoutError(f"Timed out waiting for evaluation {evaluation_id} to complete.")
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_balance(self) -> TrismikAccountBalance:
        """Fetch the account balance, including in-flight holds.

        Returns:
            Balance breakdown: total, held, and available.

        Raises:
            TrismikApiError: If the API request fails.
        """
        try:
            response = self._http_client.get("/admin-api/account/balance")
            response.raise_for_status()
            return TrismikQuickCompareResponseMapper.to_account_balance(response.json())
        except httpx.HTTPStatusError as e:
            raise handle_http_error(e) from e
        except httpx.HTTPError as e:
            raise TrismikApiError(str(e)) from e


def _build_list_params(
    page: int,
    limit: int,
    sort_by: Optional[str],
    sort_order: Optional[str],
    search: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": page, "limit": limit}
    if sort_by is not None:
        params["sortBy"] = sort_by
    if sort_order is not None:
        params["sortOrder"] = sort_order
    if search is not None:
        params["search"] = search
    return params


def _validate_metrics(metrics: List[TrismikEvaluationMetric]) -> None:
    llmaj_indices = [
        i for i, m in enumerate(metrics) if isinstance(m, TrismikEvaluationMetricLLMAJ)
    ]
    if len(llmaj_indices) > 1:
        raise TrismikValidationError("At most one LLM-as-judge metric is allowed per evaluation.")
    if llmaj_indices and llmaj_indices[0] != 0:
        raise TrismikValidationError("LLM-as-judge metric must be the first metric in the list.")
    for m in metrics:
        if not isinstance(m, TrismikEvaluationMetricLLMAJ):
            continue
        if len(m.judges) < 1:
            raise TrismikValidationError("LLM-as-judge metric requires at least one judge.")
        names = [j.name for j in m.judges]
        if len(names) != len(set(names)):
            raise TrismikValidationError("Judge names must be unique within an evaluation.")
