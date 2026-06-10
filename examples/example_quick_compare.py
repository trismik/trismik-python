"""End-to-end example for the Trismik Quick Compare feature.

Demonstrates the full life-cycle exposed by ``TrismikQuickCompareAsyncClient``:

1. Check account balance.
2. Upload a local CSV as a user dataset (init + S3 + finalize).
3. Create an evaluation across multiple models with mixed metrics
   (LLM-as-judge + accuracy).
4. Poll for completion.
5. Request a presigned export URL for the results.

Requires ``TRISMIK_API_KEY`` in the environment (or a ``.env`` file).
"""

import asyncio
import secrets
from pathlib import Path

from dotenv import load_dotenv

from trismik import (
    TrismikAsyncClient,
    TrismikEvaluation,
    TrismikEvaluationDatasetUser,
    TrismikEvaluationMetricAccuracy,
    TrismikEvaluationMetricLLMAJ,
    TrismikEvaluationModelInputColumn,
)

JUDGE_TEMPLATE = (
    "You are an expert evaluator assessing the accuracy of model "
    "predictions. Compare the prediction to the reference answer and rate "
    "based on semantic equivalence and factual correctness.\n\n"
    "Use the following scale:\n"
    "1 - Completely incorrect: Factually wrong, irrelevant, or does not "
    "address the question\n"
    "2 - Mostly incorrect: Addresses the topic but contains major errors or "
    "misses most key points\n"
    "3 - Partially correct: Contains some correct information but has "
    "significant gaps or errors in key points\n"
    "4 - Mostly correct: All key points present and accurate, but minor "
    "details missing or slightly imprecise\n"
    "5 - Completely correct: Fully accurate, semantically equivalent, all "
    "key points and details correct\n\n"
    "Focus on semantic meaning rather than exact wording. Minor phrasing "
    "differences are acceptable if the meaning is preserved.\n\n"
    "Respond with ONLY the rating number (1-5).\n\n"
    "Question: {{row.answer}}\n\n"
    "Prediction: {{response}}\n\n"
    "Rate the accuracy of the prediction:"
)


def _log_progress(evaluation: TrismikEvaluation) -> None:
    progress = evaluation.progress
    pct = progress.percentage if progress is not None else 0.0
    inf_pct = progress.inference_percentage if progress is not None else 0.0
    judge_pct = progress.judge_percentage if progress is not None else None
    line = f"  [{evaluation.status}] {pct:.1f}% - [Inference Progress] {inf_pct:.1f}%"
    if judge_pct is not None:
        line += f" - [Judge Progress] {judge_pct:.1f}%"
    print(line)


async def main() -> None:
    """Run the quick-compare end-to-end flow."""
    load_dotenv()
    async with TrismikAsyncClient() as client:
        quick_compare = client.quick_compare

        # 1) Balance
        balance = await quick_compare.get_account_balance()
        print(
            f"Balance: ${balance.balance_usd:.2f} "
            f"(held ${balance.held_usd:.2f}, available ${balance.available_usd:.2f})"
        )

        # Suffix used to keep dataset and evaluation names unique per run.
        run_suffix = secrets.token_hex(4)

        # 2) Upload dataset
        dataset_csv = Path(__file__).parent / "data" / "qa.csv"
        dataset = await quick_compare.upload_user_dataset(
            dataset_csv,
            name=f"example-qa-{run_suffix}",
        )
        print(f"Uploaded dataset {dataset.id} " f"({dataset.status}, {dataset.row_count} rows)")

        # 3) Create evaluation
        evaluation = await quick_compare.create_evaluation(
            name=f"quick-compare-demo-{run_suffix}",
            dataset=TrismikEvaluationDatasetUser(dataset_id=dataset.id),
            model_input=TrismikEvaluationModelInputColumn(column="question"),
            metrics=[
                TrismikEvaluationMetricLLMAJ(
                    judge_model="openai/openai/gpt-5_nano",
                    rubric=JUDGE_TEMPLATE,
                ),
                TrismikEvaluationMetricAccuracy(gold_column="answer"),
            ],
            models=["openai/openai/gpt-4.1_mini", "aws/amazon/nova_2_lite"],
            budget_limit_usd=2.0,
            sample_limit={"test": 50},
        )
        print(f"Created evaluation {evaluation.id}")

        # 4) Wait for completion
        completed = await quick_compare.wait_for_evaluation(
            evaluation.id,
            on_progress=_log_progress,
        )
        for score in completed.scores:
            print(f"  {score.model} | {score.metric}: {score.value:.3f}")

        # 5) Export URL
        export = await quick_compare.get_evaluation_export_url(completed.id, format="parquet")
        print(f"Export ({export.status}): {export.url}")


if __name__ == "__main__":
    asyncio.run(main())
