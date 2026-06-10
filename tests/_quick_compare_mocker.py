"""Mock httpx responses for quick-compare tests."""

from typing import Any, Dict, List, Optional

import httpx


def _req() -> httpx.Request:
    return httpx.Request("GET", "https://api.test/x")


class TrismikQuickCompareResponseMocker:
    """Factory of canned ``httpx.Response`` objects for quick-compare tests."""

    @staticmethod
    def user_dataset_init() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "id": "ds-1",
                "name": "example",
                "status": "pending",
                "upload": {
                    "url": "https://s3.example.com/bucket",
                    "method": "POST",
                    "fields": {"key": "uploads/ds-1", "policy": "..."},
                    "expiresAt": "2030-01-01T00:00:00Z",
                },
            },
        )

    @staticmethod
    def user_dataset(status: str = "ready") -> httpx.Response:
        """Shape returned by the finalize endpoint (FinalizeUserDatasetResponse)."""
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "id": "ds-1",
                "name": "example",
                "status": status,
                "fileType": "csv",
                "sizeBytes": 1024,
                "rowCount": 10,
                "columns": [
                    {"name": "question", "type": "VARCHAR"},
                    {"name": "answer", "type": "VARCHAR"},
                ],
                "preview": [{"question": "Q1", "answer": "A1"}],
                "createdAt": "2026-05-01T00:00:00Z",
                "finalizedAt": "2026-05-01T00:01:00Z",
            },
        )

    @staticmethod
    def user_dataset_detail() -> httpx.Response:
        """Shape returned by GET /public/datasets/{id} (UserDatasetDetailResponse)."""
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "id": "ds-1",
                "datasetName": "example",
                "s3Path": "s3://bucket/ds-1",
                "fileType": "csv",
                "columns": ["question", "answer"],
                "rowCount": 10,
                "sizeBytes": 1024,
                "status": "READY",
                "uploadedBy": "user-1",
                "createdAt": "2026-05-01T00:00:00Z",
                "updatedAt": "2026-05-01T00:01:00Z",
                "finalizedAt": "2026-05-01T00:01:00Z",
            },
        )

    @staticmethod
    def user_datasets_page() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "data": [
                    {
                        "id": "ds-1",
                        "datasetName": "alpha",
                        "s3Path": "s3://bucket/ds-1",
                        "rowCount": 10,
                        "sizeBytes": 1024,
                        "uploadedBy": "user-1",
                        "createdAt": "2026-05-01T00:00:00Z",
                        "updatedAt": "2026-05-01T00:01:00Z",
                    },
                    {
                        "id": "ds-2",
                        "datasetName": "beta",
                        "s3Path": "s3://bucket/ds-2",
                        "createdAt": "2026-05-02T00:00:00Z",
                        "updatedAt": "2026-05-02T00:01:00Z",
                    },
                ],
                "meta": {
                    "page": 1,
                    "limit": 20,
                    "totalItems": 2,
                    "totalPages": 1,
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                },
            },
        )

    @staticmethod
    def user_dataset_rows_page() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "data": [
                    {"question": "What is 2+2?", "answer": "4"},
                    {"question": "Capital of France?", "answer": "Paris"},
                ],
                "meta": {
                    "page": 1,
                    "limit": 50,
                    "totalItems": 2,
                    "totalPages": 1,
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                },
            },
        )

    @staticmethod
    def evaluation(
        status: str = "processing",
        progress_percentage: Optional[float] = 50.0,
        error_message: Optional[str] = None,
        scores: Optional[List[Dict[str, Any]]] = None,
        inference_progress_percentage: Optional[float] = None,
        judge_progress_percentage: Optional[float] = None,
    ) -> httpx.Response:
        body: Dict[str, Any] = {
            "id": "ev-1",
            "name": "demo",
            "status": status,
            "datasetName": "example",
            "createdAt": "2026-05-21T10:00:00Z",
            "updatedAt": "2026-05-21T10:01:00Z",
            "modelConfigs": [
                {"id": "gpt-4o-mini", "thinking": False},
                {"id": "claude-haiku-4-5", "thinking": False},
            ],
            "progressPercentage": progress_percentage,
            "inferenceProgressPercentage": inference_progress_percentage,
            "judgeProgressPercentage": judge_progress_percentage,
            "budgetLimit": 2.0,
            "sampleLimit": {"test": 50},
            "totalCost": 0.5,
            "batches": [
                {
                    "id": "b-1",
                    "modelName": "gpt-4o-mini",
                    "status": status,
                    "totalRequests": 50,
                    "completedCount": 25,
                    "failedCount": 0,
                    "scores": scores or [],
                }
            ],
        }
        if error_message is not None:
            body["errorMessage"] = error_message
        return httpx.Response(request=_req(), status_code=200, json=body)

    @staticmethod
    def evaluations_page() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "data": [
                    {
                        "id": "ev-1",
                        "name": "first",
                        "status": "completed",
                        "createdAt": "2026-05-21T10:00:00Z",
                        "models": ["gpt-4o-mini"],
                        "datasetName": "example",
                        "budgetLimit": 1.0,
                        "isDemo": False,
                    },
                    {
                        "id": "ev-2",
                        "name": "second",
                        "status": "processing",
                        "createdAt": "2026-05-21T11:00:00Z",
                        "models": ["claude-haiku-4-5"],
                        "datasetName": "example",
                        "progressPercentage": 25.0,
                        "budgetLimit": 1.5,
                        "isDemo": False,
                    },
                ],
                "meta": {
                    "page": 1,
                    "limit": 20,
                    "totalItems": 2,
                    "totalPages": 1,
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                },
            },
        )

    @staticmethod
    def evaluation_export_url(format: str = "parquet") -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={
                "status": "ready",
                "format": format,
                "url": "https://s3.example.com/exports/ev-1",
                "expiresAt": "2026-05-21T12:00:00Z",
            },
        )

    @staticmethod
    def account_balance() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=200,
            json={"balanceUsd": "10.5", "heldUsd": "2.0", "availableUsd": "8.5"},
        )

    @staticmethod
    def rate_limited(retry_after: Optional[float] = None) -> httpx.Response:
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return httpx.Response(
            request=_req(),
            status_code=429,
            headers=headers,
            json={"title": "Too many requests"},
        )

    @staticmethod
    def not_found() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=404,
            json={"title": "Not found"},
        )

    @staticmethod
    def s3_upload_success() -> httpx.Response:
        return httpx.Response(request=_req(), status_code=204)

    @staticmethod
    def s3_upload_failure() -> httpx.Response:
        return httpx.Response(
            request=_req(),
            status_code=403,
            content=b"AccessDenied",
        )
