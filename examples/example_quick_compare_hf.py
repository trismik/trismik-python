"""Quick Compare example using a HuggingFace dataset (Winogrande).

Differs from ``example_quick_compare.py`` only in the dataset source:
instead of uploading a local CSV, this script targets the public HF dataset
``allenai/winogrande`` (sub-dataset ``winogrande_xs``, split ``train``).

Requires ``TRISMIK_API_KEY`` in the environment (or a ``.env`` file).
"""

import asyncio
import secrets

from dotenv import load_dotenv

from trismik import (
    TrismikAsyncClient,
    TrismikEvaluation,
    TrismikEvaluationDatasetHF,
    TrismikEvaluationMetricExactMatch,
    TrismikEvaluationModelInputTemplate,
)

PROMPT_TEMPLATE = """Sentence: {{row.sentence}}

Options:
1. {{row.option1}}
2. {{row.option2}}

Rules:
- Answer with just 1 or 2
- Do not add any extra text or explanation"""


def _log_progress(evaluation: TrismikEvaluation) -> None:
    pct = evaluation.progress.percentage if evaluation.progress is not None else 0.0
    inf_pct = evaluation.progress.inference_percentage if evaluation.progress is not None else 0.0
    judge_pct = evaluation.progress.judge_percentage
    if judge_pct is not None:
        print(f"  [{evaluation.status}] {pct:.1f}% - [Inference Progress] {inf_pct:.1f}% - [Judge Progress] {judge_pct:.1f}%")
    else:
        print(f"  [{evaluation.status}] {pct:.1f}% - [Inference Progress] {inf_pct:.1f}%")

async def main() -> None:
    """Run the quick-compare flow against allenai/winogrande on HF."""
    load_dotenv()
    async with TrismikAsyncClient() as client:
        quick_compare = client.quick_compare

        run_suffix = secrets.token_hex(4)

        # Balance check
        balance = await quick_compare.get_account_balance()
        print(
            f"Balance: ${balance.balance_usd:.2f} "
            f"(held ${balance.held_usd:.2f}, available ${balance.available_usd:.2f})"
        )

        # Create evaluation on Winogrande (xs subset, train split)
        evaluation = await quick_compare.create_evaluation(
            name=f"winogrande-exact-match-{run_suffix}",
            dataset=TrismikEvaluationDatasetHF(
                repo="allenai/winogrande",
                sub_dataset="winogrande_xs",
                splits=["train"],
            ),
            model_input=TrismikEvaluationModelInputTemplate(template=PROMPT_TEMPLATE),
            metrics=[TrismikEvaluationMetricExactMatch(gold_column="answer")],
            models=["openai/openai/gpt-4.1_mini", "aws/amazon/nova_2_lite"],
            budget_limit_usd=2.0,
            sample_limit={"train": 50},
        )
        print(f"Created evaluation {evaluation.id}")

        # Wait for completion
        completed = await quick_compare.wait_for_evaluation(
            evaluation.id,
            on_progress=_log_progress,
        )
        for score in completed.scores:
            print(f"  {score.model} | {score.metric}: {score.value:.3f}")

        # Export results
        export = await quick_compare.get_evaluation_export_url(completed.id, format="parquet")
        print(f"Export ({export.status}): {export.url}")


if __name__ == "__main__":
    asyncio.run(main())
