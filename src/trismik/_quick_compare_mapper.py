"""Response/request mappers for the Quick Compare feature."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from trismik.types import (
    TrismikAccountBalance,
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
    TrismikPage,
    TrismikPageMeta,
    TrismikUserDataset,
    TrismikUserDatasetColumn,
    TrismikUserDatasetDetail,
    TrismikUserDatasetInitResponse,
    TrismikUserDatasetPresignedPost,
)


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --- Request serializers (module-private) ----------------------------------


def _dataset_to_dict(dataset: TrismikEvaluationDataset) -> Dict[str, Any]:
    if isinstance(dataset, TrismikEvaluationDatasetHF):
        body: Dict[str, Any] = {"type": "huggingface", "link": dataset.repo}
        if dataset.sub_dataset is not None:
            body["subDataset"] = dataset.sub_dataset
        if dataset.splits is not None:
            body["splits"] = dataset.splits
        return body
    if isinstance(dataset, TrismikEvaluationDatasetUser):
        return {"type": "userDataset", "id": dataset.dataset_id}
    raise TypeError(f"Unsupported dataset variant: {type(dataset).__name__}")


def _model_input_to_dict(model_input: TrismikEvaluationModelInput) -> Dict[str, Any]:
    if isinstance(model_input, TrismikEvaluationModelInputColumn):
        return {"type": "column", "column": model_input.column}
    if isinstance(model_input, TrismikEvaluationModelInputTemplate):
        return {"type": "template", "template": model_input.template}
    raise TypeError(f"Unsupported model input variant: {type(model_input).__name__}")


def _metric_to_dict(metric: TrismikEvaluationMetric) -> Dict[str, Any]:
    if isinstance(metric, TrismikEvaluationMetricAccuracy):
        return {"type": "accuracy", "truthColumn": metric.gold_column}
    if isinstance(metric, TrismikEvaluationMetricExactMatch):
        return {
            "type": "exact_match",
            "truthColumn": metric.gold_column,
            "normalize": metric.normalize,
        }
    if isinstance(metric, TrismikEvaluationMetricBleu):
        return {"type": "bleu", "truthColumn": metric.gold_column}
    if isinstance(metric, TrismikEvaluationMetricRougeL):
        return {"type": "rougel", "truthColumn": metric.gold_column}
    if isinstance(metric, TrismikEvaluationMetricLLMAJ):
        judges: List[Dict[str, Any]] = []
        for judge in metric.judges:
            judge_body: Dict[str, Any] = {
                "name": judge.name,
                "judgeModel": judge.judge_model,
                "judgePromptTemplate": judge.rubric,
                "judgeScaleMin": judge.scale_min,
                "judgeScaleMax": judge.scale_max,
            }
            if judge.temperature is not None:
                judge_body["temperature"] = judge.temperature
            if judge.thinking is not None:
                judge_body["thinking"] = judge.thinking
            judges.append(judge_body)
        return {"type": "llmaj", "judges": judges}
    raise TypeError(f"Unsupported metric variant: {type(metric).__name__}")


def _model_config_to_dict(
    model: Union[str, TrismikEvaluationModelConfig],
) -> Dict[str, Any]:
    if isinstance(model, str):
        return {"id": model, "thinking": False}
    body: Dict[str, Any] = {"id": model.id, "thinking": model.thinking}
    if model.reasoning_effort is not None:
        body["reasoningEffort"] = model.reasoning_effort
    if model.max_tokens is not None:
        body["maxTokens"] = model.max_tokens
    if model.temperature is not None:
        body["temperature"] = model.temperature
    return body


def evaluation_request_body(
    *,
    name: str,
    dataset_name: str,
    dataset: TrismikEvaluationDataset,
    model_input: TrismikEvaluationModelInput,
    metrics: List[TrismikEvaluationMetric],
    models: List[Union[str, TrismikEvaluationModelConfig]],
    budget_limit_usd: float,
    sample_limit: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build the JSON body for ``POST /bank-maker-api/evaluations``."""
    body: Dict[str, Any] = {
        "name": name,
        "datasetName": dataset_name,
        "dataset": _dataset_to_dict(dataset),
        "modelInput": _model_input_to_dict(model_input),
        "metrics": [_metric_to_dict(m) for m in metrics],
        "modelSelection": {"models": [_model_config_to_dict(m) for m in models]},
        "budgetLimit": budget_limit_usd,
    }
    if sample_limit is not None:
        body["sampleLimit"] = sample_limit
    return body


# --- Response mappers ------------------------------------------------------


class TrismikQuickCompareResponseMapper:
    """Maps Quick Compare JSON responses to typed Python objects."""

    @staticmethod
    def to_page_meta(json: Dict[str, Any]) -> TrismikPageMeta:
        """Convert ``meta`` JSON to TrismikPageMeta."""
        return TrismikPageMeta(
            page=json["page"],
            limit=json["limit"],
            total_items=json["totalItems"],
            total_pages=json["totalPages"],
            has_next_page=json.get("hasNextPage", False),
            has_previous_page=json.get("hasPreviousPage", False),
        )

    @staticmethod
    def to_user_dataset_init(json: Dict[str, Any]) -> TrismikUserDatasetInitResponse:
        """Convert init-upload response to TrismikUserDatasetInitResponse."""
        upload_json = json["upload"]
        upload = TrismikUserDatasetPresignedPost(
            url=upload_json["url"],
            fields=upload_json.get("fields", {}),
            method=cast(Any, upload_json.get("method", "POST")),
            expires_at=_parse_datetime(upload_json.get("expiresAt")),
        )
        return TrismikUserDatasetInitResponse(
            dataset_id=str(json["id"]),
            name=json["name"],
            status=json.get("status", "pending"),
            upload=upload,
        )

    @staticmethod
    def _dataset_name(json: Dict[str, Any]) -> str:
        return json.get("datasetName") or json.get("name") or ""

    @staticmethod
    def to_user_dataset(json: Dict[str, Any]) -> TrismikUserDataset:
        """Convert JSON to TrismikUserDataset (slim view)."""
        return TrismikUserDataset(
            id=str(json["id"]),
            name=TrismikQuickCompareResponseMapper._dataset_name(json),
            created_at=_require_datetime(json["createdAt"]),
            status=json.get("status"),
            s3_path=json.get("s3Path"),
            row_count=json.get("rowCount"),
            size_bytes=json.get("sizeBytes"),
            uploaded_by=json.get("uploadedBy"),
            updated_at=_parse_datetime(json.get("updatedAt")),
        )

    @staticmethod
    def _parse_columns(columns_json: Any) -> List[TrismikUserDatasetColumn]:
        if not columns_json:
            return []
        out: List[TrismikUserDatasetColumn] = []
        for col in columns_json:
            if isinstance(col, str):
                out.append(TrismikUserDatasetColumn(name=col))
            elif isinstance(col, dict):
                out.append(TrismikUserDatasetColumn(name=col["name"], type=col.get("type", "")))
        return out

    @staticmethod
    def to_user_dataset_detail(json: Dict[str, Any]) -> TrismikUserDatasetDetail:
        """Convert JSON to TrismikUserDatasetDetail (with columns)."""
        finalized_at = _parse_datetime(json.get("finalizedAt"))
        return TrismikUserDatasetDetail(
            id=str(json["id"]),
            name=TrismikQuickCompareResponseMapper._dataset_name(json),
            created_at=_require_datetime(json["createdAt"]),
            status=json.get("status"),
            s3_path=json.get("s3Path"),
            file_type=json.get("fileType"),
            row_count=json.get("rowCount"),
            size_bytes=json.get("sizeBytes"),
            uploaded_by=json.get("uploadedBy"),
            updated_at=_parse_datetime(json.get("updatedAt")),
            finalized_at=finalized_at,
            columns=TrismikQuickCompareResponseMapper._parse_columns(json.get("columns")),
            preview=list(json.get("preview", [])),
        )

    @staticmethod
    def to_user_dataset_page(json: Dict[str, Any]) -> TrismikPage[TrismikUserDataset]:
        """Convert paginated list of user datasets."""
        data = [TrismikQuickCompareResponseMapper.to_user_dataset(item) for item in json["data"]]
        meta = TrismikQuickCompareResponseMapper.to_page_meta(json["meta"])
        return TrismikPage(data=data, meta=meta)

    @staticmethod
    def to_user_dataset_rows_page(json: Dict[str, Any]) -> TrismikPage[Dict[str, Any]]:
        """Convert paginated user dataset rows (rows are arbitrary dicts)."""
        rows = list(json.get("data", []))
        meta = TrismikQuickCompareResponseMapper.to_page_meta(json["meta"])
        return TrismikPage(data=rows, meta=meta)

    @staticmethod
    def _to_progress(
        json: Dict[str, Any],
        budget: Optional[float],
    ) -> Optional[TrismikEvaluationProgress]:
        if json.get("progressPercentage") is None:
            return None
        inference = json.get("inferenceProgressPercentage")
        judge = json.get("judgeProgressPercentage")
        return TrismikEvaluationProgress(
            percentage=float(json["progressPercentage"]),
            cost_usd_used=float(json.get("totalCost") or 0.0),
            cost_usd_budget=float(budget or 0.0),
            items_processed=json.get("itemsProcessed"),
            items_total=json.get("itemsTotal"),
            inference_percentage=float(inference) if inference is not None else None,
            judge_percentage=float(judge) if judge is not None else None,
        )

    @staticmethod
    def _models_from_detail(json: Dict[str, Any]) -> List[str]:
        configs = json.get("modelConfigs") or json.get("modelSelection", {}).get("models", [])
        return [m["id"] for m in configs if "id" in m]

    @staticmethod
    def _batches(json: Dict[str, Any]) -> List[TrismikEvaluationBatch]:
        return [
            TrismikEvaluationBatch(
                id=b["id"],
                model=b.get("modelName") or b.get("model", ""),
                status=b.get("status", "unknown"),
                split=b.get("split"),
                items_processed=b.get("completedCount"),
                items_total=b.get("totalRequests"),
                failed_count=b.get("failedCount"),
                cost_usd=b.get("totalCost"),
            )
            for b in json.get("batches", [])
        ]

    @staticmethod
    def _scores(json: Dict[str, Any]) -> List[TrismikEvaluationScore]:
        scores: List[TrismikEvaluationScore] = []
        for batch in json.get("batches", []):
            model_name = batch.get("modelName") or batch.get("model", "")
            for s in batch.get("scores", []):
                scores.append(
                    TrismikEvaluationScore(
                        model=model_name,
                        metric=s.get("metric") or s.get("metricId", ""),
                        value=float(s.get("avgScore", s.get("value", 0.0))),
                        min_value=s.get("minScore"),
                        max_value=s.get("maxScore"),
                    )
                )
        return scores

    @staticmethod
    def to_evaluation(json: Dict[str, Any]) -> TrismikEvaluation:
        """Convert detail JSON to TrismikEvaluation."""
        budget = json.get("budgetLimit")
        sample_limit = json.get("sampleLimit")
        return TrismikEvaluation(
            id=json["id"],
            name=json["name"],
            status=json["status"],
            created_at=_require_datetime(json["createdAt"]),
            models=TrismikQuickCompareResponseMapper._models_from_detail(json),
            dataset_name=json.get("datasetName"),
            progress=TrismikQuickCompareResponseMapper._to_progress(json, budget),
            batches=TrismikQuickCompareResponseMapper._batches(json),
            scores=TrismikQuickCompareResponseMapper._scores(json),
            error_message=json.get("errorMessage"),
            budget_limit_usd=budget,
            sample_limit=sample_limit if isinstance(sample_limit, dict) else None,
            total_cost_usd=json.get("totalCost"),
            is_demo=bool(json.get("isDemo", False)),
            updated_at=_parse_datetime(json.get("updatedAt")),
        )

    @staticmethod
    def to_evaluation_summary(json: Dict[str, Any]) -> TrismikEvaluationSummary:
        """Convert slim list-item JSON to TrismikEvaluationSummary."""
        return TrismikEvaluationSummary(
            id=json["id"],
            name=json["name"],
            status=json["status"],
            created_at=_require_datetime(json["createdAt"]),
            models=list(json.get("models", [])),
            dataset_name=json.get("datasetName"),
            progress_percentage=json.get("progressPercentage"),
            budget_limit_usd=json.get("budgetLimit"),
            is_demo=bool(json.get("isDemo", False)),
            updated_at=_parse_datetime(json.get("updatedAt")),
        )

    @staticmethod
    def to_evaluation_page(json: Dict[str, Any]) -> TrismikPage[TrismikEvaluationSummary]:
        """Convert paginated evaluations list."""
        data = [
            TrismikQuickCompareResponseMapper.to_evaluation_summary(item) for item in json["data"]
        ]
        meta = TrismikQuickCompareResponseMapper.to_page_meta(json["meta"])
        return TrismikPage(data=data, meta=meta)

    @staticmethod
    def to_evaluation_export_url(
        json: Dict[str, Any],
        format: str = "parquet",
    ) -> TrismikEvaluationExportUrl:
        """Convert export-url JSON to TrismikEvaluationExportUrl."""
        return TrismikEvaluationExportUrl(
            status=json.get("status", "ready"),
            format=cast(Any, json.get("format", format)),
            url=json.get("url"),
            expires_at=_parse_datetime(json.get("expiresAt") or json.get("expires_at")),
        )

    @staticmethod
    def to_account_balance(json: Dict[str, Any]) -> TrismikAccountBalance:
        """Convert /admin-api/account/balance JSON to TrismikAccountBalance."""
        return TrismikAccountBalance(
            balance_usd=float(json["balanceUsd"]),
            held_usd=float(json["heldUsd"]),
            available_usd=float(json["availableUsd"]),
        )
