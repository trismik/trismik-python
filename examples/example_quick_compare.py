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
    TrismikEvaluationModelInputTemplate,
)


def _log_progress(evaluation: TrismikEvaluation) -> None:
    pct = evaluation.progress.percentage if evaluation.progress is not None else 0.0
    print(f"  [{evaluation.status}] {pct:.1f}%")


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
            model_input=TrismikEvaluationModelInputTemplate(
                template="Q: {question}\nA:",
            ),
            metrics=[
                TrismikEvaluationMetricLLMAJ(
                    judge_model="openai/openai/gpt-5_nano",
                    rubric="Rate factual accuracy from 1 to 5.",
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
