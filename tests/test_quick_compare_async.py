"""Tests for the async Quick Compare client."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from trismik import TrismikAsyncClient
from trismik._async.quick_compare import TrismikQuickCompareAsyncClient
from trismik.exceptions import (
    TrismikApiError,
    TrismikNotFoundError,
    TrismikRateLimitedError,
    TrismikValidationError,
)
from trismik.settings import environment_settings
from trismik.types import (
    TrismikEvaluationDatasetHF,
    TrismikEvaluationDatasetUser,
    TrismikEvaluationMetricAccuracy,
    TrismikEvaluationMetricBleu,
    TrismikEvaluationMetricExactMatch,
    TrismikEvaluationMetricLLMAJ,
    TrismikEvaluationMetricRougeL,
    TrismikEvaluationModelConfig,
    TrismikEvaluationModelInputColumn,
    TrismikEvaluationModelInputTemplate,
    TrismikJudge,
)

from ._quick_compare_mocker import TrismikQuickCompareResponseMocker as M


class TestQuickCompareInit:
    """Initialization paths for the standalone client."""

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_service_url"], "service_url")
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    def test_standalone_with_explicit_values(self) -> None:
        TrismikQuickCompareAsyncClient(api_key="k", service_url="u")

    def test_accessor_shares_http_client(self) -> None:
        parent = TrismikAsyncClient(api_key="k", service_url="u")
        qc = parent.quick_compare
        assert qc._http_client is parent._http_client
        assert qc._owns_client is False
        # Cached: subsequent access returns same instance
        assert parent.quick_compare is qc

    @pytest.mark.asyncio
    async def test_accessor_does_not_close_parent_http_client(self) -> None:
        parent_client = MagicMock(httpx.AsyncClient)
        parent_client.aclose = AsyncMock()
        parent = TrismikAsyncClient(api_key="k", http_client=parent_client)
        qc = parent.quick_compare
        await qc.aclose()
        parent_client.aclose.assert_not_called()


class TestUserDatasets:

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @pytest.mark.asyncio
    async def test_list_user_datasets(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.user_datasets_page()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        page = await client.list_user_datasets(page=1, limit=20, search="alpha")
        assert page.meta.total_items == 2
        assert page.data[0].id == "ds-1"
        args, kwargs = http.get.call_args
        assert args[0] == "/bank-maker-api/public/datasets"
        assert kwargs["params"]["page"] == 1
        assert kwargs["params"]["search"] == "alpha"

    @pytest.mark.asyncio
    async def test_get_user_dataset_happy(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.user_dataset_detail()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        ds = await client.get_user_dataset("ds-1")
        assert ds.id == "ds-1"
        assert len(ds.columns) == 2
        assert ds.columns[0].name == "question"

    @pytest.mark.asyncio
    async def test_get_user_dataset_not_found(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        not_found = M.not_found()
        not_found_error = httpx.HTTPStatusError(
            "not found", request=not_found.request, response=not_found
        )
        http.get.return_value = MagicMock(raise_for_status=MagicMock(side_effect=not_found_error))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        with pytest.raises(TrismikNotFoundError):
            await client.get_user_dataset("missing")

    @pytest.mark.asyncio
    async def test_list_user_dataset_rows(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.user_dataset_rows_page()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        rows = await client.list_user_dataset_rows("ds-1", page=1, limit=50)
        assert len(rows.data) == 2
        assert rows.data[0]["answer"] == "4"

    @pytest.mark.asyncio
    async def test_init_finalize_dataset(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(side_effect=[M.user_dataset_init(), M.user_dataset()])
        client = TrismikQuickCompareAsyncClient(http_client=http)
        init = await client.init_user_dataset(name="x", file_type="csv", size_bytes=100)
        assert init.dataset_id == "ds-1"
        assert init.upload.method == "POST"
        # Verify request body shape
        init_body = http.post.call_args_list[0].kwargs["json"]
        assert init_body == {"name": "x", "fileType": "csv", "sizeBytes": 100}
        ds = await client.finalize_user_dataset(init.dataset_id)
        assert ds.status == "ready"
        assert ds.file_type == "csv"
        assert len(ds.columns) == 2
        assert ds.columns[0].name == "question"


class TestUploadWrapper:

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @pytest.mark.asyncio
    async def test_upload_with_bytes_calls_init_s3_finalize(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(side_effect=[M.user_dataset_init(), M.user_dataset()])
        client = TrismikQuickCompareAsyncClient(http_client=http)

        captured = {}

        async def fake_upload(presigned, data, *, content_type):
            captured["presigned"] = presigned
            captured["data_len"] = len(data)
            captured["content_type"] = content_type

        monkeypatch.setattr("trismik._async.quick_compare.upload_to_presigned", fake_upload)
        payload = b"hello,world\n1,2\n"
        ds = await client.upload_user_dataset(payload, name="x", file_type="csv")
        assert ds.id == "ds-1"
        assert captured["data_len"] == len(payload)
        assert captured["content_type"] == "text/csv"
        # init + finalize, no other calls
        assert http.post.await_count == 2
        init_body = http.post.call_args_list[0].kwargs["json"]
        assert init_body == {
            "name": "x",
            "fileType": "csv",
            "sizeBytes": len(payload),
        }

    @pytest.mark.asyncio
    async def test_upload_requires_file_type_for_bytes(self) -> None:
        client = TrismikQuickCompareAsyncClient(http_client=MagicMock(httpx.AsyncClient))
        with pytest.raises(TrismikValidationError, match="file_type is required"):
            await client.upload_user_dataset(b"x", name="x")

    @pytest.mark.asyncio
    async def test_upload_skips_finalize_when_s3_fails(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(side_effect=[M.user_dataset_init(), M.user_dataset()])
        client = TrismikQuickCompareAsyncClient(http_client=http)

        async def fake_upload_fail(presigned, data, *, content_type):
            raise TrismikApiError("S3 boom")

        monkeypatch.setattr("trismik._async.quick_compare.upload_to_presigned", fake_upload_fail)
        with pytest.raises(TrismikApiError, match="S3 boom"):
            await client.upload_user_dataset(b"x", name="x", file_type="csv")
        assert http.post.await_count == 1  # only init, finalize NOT called


class TestEvaluations:

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @pytest.mark.asyncio
    async def test_create_evaluation_with_user_dataset_and_template(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(return_value=M.evaluation(status="processing"))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        ev = await client.create_evaluation(
            name="demo",
            dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
            model_input=TrismikEvaluationModelInputTemplate(template="Q: {q}"),
            metrics=[TrismikEvaluationMetricAccuracy(gold_column="answer")],
            models=["gpt-4o-mini"],
            budget_limit_usd=1.0,
        )
        assert ev.id == "ev-1"
        body = http.post.call_args.kwargs["json"]
        assert body["dataset"] == {"type": "userDataset", "id": "ds-1"}
        assert body["modelInput"] == {"type": "template", "template": "Q: {q}"}
        assert body["metrics"][0]["type"] == "accuracy"
        assert body["metrics"][0]["truthColumn"] == "answer"

    @pytest.mark.asyncio
    async def test_create_evaluation_with_hf_and_column_input(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(return_value=M.evaluation(status="processing"))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        await client.create_evaluation(
            name="demo",
            dataset=TrismikEvaluationDatasetHF(
                repo="trismik/test", sub_dataset="config1", splits=["test"]
            ),
            model_input=TrismikEvaluationModelInputColumn(column="question"),
            metrics=[TrismikEvaluationMetricBleu(gold_column="answer")],
            models=[TrismikEvaluationModelConfig(id="m", thinking=True, temperature=0.2)],
            budget_limit_usd=1.0,
        )
        body = http.post.call_args.kwargs["json"]
        assert body["dataset"] == {
            "type": "huggingface",
            "link": "trismik/test",
            "subDataset": "config1",
            "splits": ["test"],
        }
        assert body["modelInput"] == {"type": "column", "column": "question"}
        assert body["metrics"][0]["type"] == "bleu"
        assert body["modelSelection"]["models"][0] == {
            "id": "m",
            "thinking": True,
            "temperature": 0.2,
        }

    @pytest.mark.asyncio
    async def test_create_evaluation_all_metric_variants(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(return_value=M.evaluation(status="processing"))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        await client.create_evaluation(
            name="demo",
            dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
            model_input=TrismikEvaluationModelInputColumn(column="q"),
            metrics=[
                TrismikEvaluationMetricLLMAJ(judge_model="gpt-4o-mini", rubric="r"),
                TrismikEvaluationMetricAccuracy(gold_column="a"),
                TrismikEvaluationMetricExactMatch(gold_column="a", normalize=False),
                TrismikEvaluationMetricRougeL(gold_column="a"),
            ],
            models=["m1"],
            budget_limit_usd=1.0,
        )
        body = http.post.call_args.kwargs["json"]
        types = [m["type"] for m in body["metrics"]]
        assert types == ["llmaj", "accuracy", "exact_match", "rougel"]
        assert body["metrics"][2]["normalize"] is False
        # Legacy single-judge construction is serialised as a one-element
        # judges[] list (back-compat).
        judges = body["metrics"][0]["judges"]
        assert len(judges) == 1
        assert judges[0]["judgeModel"] == "gpt-4o-mini"
        assert judges[0]["judgePromptTemplate"] == "r"
        assert judges[0]["name"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_create_evaluation_multi_judge_body(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock(return_value=M.evaluation(status="processing"))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        await client.create_evaluation(
            name="demo",
            dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
            model_input=TrismikEvaluationModelInputColumn(column="q"),
            metrics=[
                TrismikEvaluationMetricLLMAJ(
                    judges=[
                        TrismikJudge(
                            name="strict",
                            judge_model="gpt-4o",
                            rubric="be strict",
                            scale_min=1,
                            scale_max=5,
                            temperature=0.0,
                        ),
                        TrismikJudge(
                            name="lenient",
                            judge_model="claude",
                            rubric="be lenient",
                            scale_min=0,
                            scale_max=10,
                            thinking=True,
                        ),
                    ]
                ),
                TrismikEvaluationMetricAccuracy(gold_column="a"),
            ],
            models=["m1"],
            budget_limit_usd=1.0,
        )
        judges = http.post.call_args.kwargs["json"]["metrics"][0]["judges"]
        assert [j["name"] for j in judges] == ["strict", "lenient"]
        assert judges[0]["judgeModel"] == "gpt-4o"
        assert judges[0]["temperature"] == 0.0
        assert "thinking" not in judges[0]
        assert judges[1]["judgeScaleMax"] == 10
        assert judges[1]["thinking"] is True

    @pytest.mark.asyncio
    async def test_create_evaluation_duplicate_judge_names_raises(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        with pytest.raises(TrismikValidationError, match="unique"):
            await client.create_evaluation(
                name="demo",
                dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
                model_input=TrismikEvaluationModelInputColumn(column="q"),
                metrics=[
                    TrismikEvaluationMetricLLMAJ(
                        judges=[
                            TrismikJudge(name="dup", judge_model="a", rubric="r"),
                            TrismikJudge(name="dup", judge_model="b", rubric="r"),
                        ]
                    ),
                ],
                models=["m1"],
                budget_limit_usd=1.0,
            )
        http.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_evaluation_llmaj_not_first_raises(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        with pytest.raises(TrismikValidationError, match="must be the first"):
            await client.create_evaluation(
                name="demo",
                dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
                model_input=TrismikEvaluationModelInputColumn(column="q"),
                metrics=[
                    TrismikEvaluationMetricAccuracy(gold_column="a"),
                    TrismikEvaluationMetricLLMAJ(judge_model="j", rubric="r"),
                ],
                models=["m1"],
                budget_limit_usd=1.0,
            )
        http.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_evaluation_two_llmaj_raises(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.post = AsyncMock()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        with pytest.raises(TrismikValidationError, match="At most one"):
            await client.create_evaluation(
                name="demo",
                dataset=TrismikEvaluationDatasetUser(dataset_id="ds-1"),
                model_input=TrismikEvaluationModelInputColumn(column="q"),
                metrics=[
                    TrismikEvaluationMetricLLMAJ(judge_model="j1", rubric="r"),
                    TrismikEvaluationMetricLLMAJ(judge_model="j2", rubric="r"),
                ],
                models=["m"],
                budget_limit_usd=1.0,
            )
        http.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_evaluation_happy(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.evaluation(status="processing", progress_percentage=42.0)
        client = TrismikQuickCompareAsyncClient(http_client=http)
        ev = await client.get_evaluation("ev-1")
        assert ev.id == "ev-1"
        assert ev.progress is not None
        assert ev.progress.percentage == 42.0
        # No per-phase figures in the payload → both default to None.
        assert ev.progress.inference_percentage is None
        assert ev.progress.judge_percentage is None

    @pytest.mark.asyncio
    async def test_get_evaluation_exposes_per_phase_progress(self) -> None:
        """LLMAJ runs surface overall + inference + judge progress."""
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.evaluation(
            status="processing",
            progress_percentage=75.0,
            inference_progress_percentage=100.0,
            judge_progress_percentage=50.0,
        )
        client = TrismikQuickCompareAsyncClient(http_client=http)
        ev = await client.get_evaluation("ev-1")
        assert ev.progress is not None
        assert ev.progress.percentage == 75.0
        assert ev.progress.inference_percentage == 100.0
        assert ev.progress.judge_percentage == 50.0

    @pytest.mark.asyncio
    async def test_get_evaluation_rate_limited(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        rl = M.rate_limited(retry_after=15)
        rl_error = httpx.HTTPStatusError("rate", request=rl.request, response=rl)
        http.get.return_value = MagicMock(raise_for_status=MagicMock(side_effect=rl_error))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        with pytest.raises(TrismikRateLimitedError) as exc_info:
            await client.get_evaluation("ev-1")
        assert exc_info.value.retry_after_seconds == 15.0

    @pytest.mark.asyncio
    async def test_list_evaluations(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.evaluations_page()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        page = await client.list_evaluations(
            page=1, limit=20, sort_by="createdAt", sort_order="desc", search="demo"
        )
        assert page.meta.total_items == 2
        assert page.data[0].id == "ev-1"
        assert page.data[1].progress_percentage == 25.0
        params = http.get.call_args.kwargs["params"]
        assert params["sortBy"] == "createdAt"
        assert params["sortOrder"] == "desc"
        assert params["search"] == "demo"

    @pytest.mark.asyncio
    async def test_export_url(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.evaluation_export_url(format="csv")
        client = TrismikQuickCompareAsyncClient(http_client=http)
        export = await client.get_evaluation_export_url("ev-1", format="csv")
        assert export.format == "csv"
        assert export.url is not None
        assert http.get.call_args.kwargs["params"]["format"] == "csv"


class TestWaitForEvaluation:

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @pytest.mark.asyncio
    async def test_completes_after_processing(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get = AsyncMock(
            side_effect=[
                M.evaluation(status="processing"),
                M.evaluation(
                    status="completed",
                    progress_percentage=100.0,
                    scores=[
                        {"metric": "accuracy", "avgScore": 0.85},
                    ],
                ),
            ]
        )
        client = TrismikQuickCompareAsyncClient(http_client=http)
        sleeps = []
        monkeypatch.setattr(
            "trismik._async.quick_compare.asyncio.sleep",
            AsyncMock(side_effect=lambda s: sleeps.append(s)),
        )
        progress = []
        done = await client.wait_for_evaluation(
            "ev-1", poll_interval=6.0, on_progress=progress.append
        )
        assert done.status == "completed"
        assert len(done.scores) == 1
        assert done.scores[0].value == 0.85
        # Slept once between processing and completed
        assert sleeps == [6.0]
        assert len(progress) == 2

    @pytest.mark.asyncio
    async def test_failed_raises(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get = AsyncMock(
            return_value=M.evaluation(
                status="failed", progress_percentage=10.0, error_message="boom"
            )
        )
        client = TrismikQuickCompareAsyncClient(http_client=http)
        monkeypatch.setattr("trismik._async.quick_compare.asyncio.sleep", AsyncMock())
        with pytest.raises(TrismikApiError, match="boom"):
            await client.wait_for_evaluation("ev-1", poll_interval=6.0)

    @pytest.mark.asyncio
    async def test_timeout(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get = AsyncMock(return_value=M.evaluation(status="processing"))
        client = TrismikQuickCompareAsyncClient(http_client=http)
        monkeypatch.setattr("trismik._async.quick_compare.asyncio.sleep", AsyncMock())
        # Fake monotonic to force timeout immediately after first iteration
        times = iter([0.0, 0.0, 100.0])
        monkeypatch.setattr("trismik._async.quick_compare._now", lambda: next(times))
        with pytest.raises(TimeoutError):
            await client.wait_for_evaluation("ev-1", poll_interval=6.0, timeout=10.0)

    @pytest.mark.asyncio
    async def test_poll_interval_too_small(self) -> None:
        client = TrismikQuickCompareAsyncClient(http_client=MagicMock(httpx.AsyncClient))
        with pytest.raises(ValueError, match=">= 6.0"):
            await client.wait_for_evaluation("ev-1", poll_interval=3.0)

    @pytest.mark.asyncio
    async def test_handles_rate_limit_with_retry_after(self, monkeypatch) -> None:
        http = MagicMock(httpx.AsyncClient)
        rl = M.rate_limited(retry_after=8)
        rl_error = httpx.HTTPStatusError("rate", request=rl.request, response=rl)
        rl_response = MagicMock(raise_for_status=MagicMock(side_effect=rl_error))
        http.get = AsyncMock(side_effect=[rl_response, M.evaluation(status="completed")])
        client = TrismikQuickCompareAsyncClient(http_client=http)
        sleeps = []
        monkeypatch.setattr(
            "trismik._async.quick_compare.asyncio.sleep",
            AsyncMock(side_effect=lambda s: sleeps.append(s)),
        )
        done = await client.wait_for_evaluation("ev-1", poll_interval=6.0)
        assert done.status == "completed"
        # First sleep was the rate-limit backoff (Retry-After: 8.0s)
        assert sleeps[0] == 8.0


class TestAccountBalance:

    @pytest.fixture(scope="function", autouse=True)
    def set_env(self, monkeypatch) -> None:
        monkeypatch.setenv(environment_settings["trismik_api_key"], "api_key")

    @pytest.mark.asyncio
    async def test_balance(self) -> None:
        http = MagicMock(httpx.AsyncClient)
        http.get.return_value = M.account_balance()
        client = TrismikQuickCompareAsyncClient(http_client=http)
        b = await client.get_account_balance()
        assert b.balance_usd == 10.5
        assert b.held_usd == 2.0
        assert b.available_usd == 8.5
        assert http.get.call_args.args[0] == "/admin-api/account/balance"
