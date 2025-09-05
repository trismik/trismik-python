"""
Example of adaptive testing of a Hugging Face model through the Trismik API.

This example uses Phi-3-small-8k-instruct as an example. You can use any
other model that you have access to. Remember to provide your own Trismik
API key in the .env file.
"""

import argparse
import asyncio
import re

import transformers
from dotenv import load_dotenv

from trismik.adaptive_test import AdaptiveTest
from trismik.types import (
    AdaptiveTestScore,
    TrismikItem,
    TrismikMultipleChoiceTextItem,
    TrismikRunMetadata,
)


def create_run_metadata(dataset_name: str) -> TrismikRunMetadata:
    """Create run metadata for the given dataset."""
    return TrismikRunMetadata(
        model_metadata=TrismikRunMetadata.ModelMetadata(
            name="microsoft/Phi-3-small-8k-instruct",
            parameters="3.84B",
            provider="Microsoft",
        ),
        test_configuration={
            "task_name": dataset_name,
            "response_format": "Multiple-choice",
        },
        inference_setup={
            "max_tokens": 1024,
        },
    )


generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


def inference(
    pipeline: transformers.pipeline, item: TrismikItem, max_retries: int = 5
) -> str:
    """
    Run inference on an item using a Hugging Face model.

    Args:
        pipeline (transformers.pipeline): Hugging Face pipeline.
        item (TrismikItem): Item to run inference on.
        max_retries (int): Maximum number of retries.
    """
    assert isinstance(item, TrismikMultipleChoiceTextItem)

    # We construct the prompt from the question and the possible choices.
    # We transformed MMLUPro2025 to be in the form of a multiple-choice
    # question, so the prompt reflects that.
    prompt = f"{item.question}\nOptions:\n" + "\n".join(
        [f"- {choice.id}: {choice.text}" for choice in item.choices]
    )

    # The system message contains the instructions for the model. We ask the
    # model to adhere strictly to the instructions; the ability of a model to
    # do that is based on the quality and size of the model. We suggest to
    # always do a post-processing step to ensure the model adheres to the
    # instructions.
    messages = [
        {
            "role": "system",
            "content": """
Answer the question you are given using only a single letter \
(for example, 'A'). \
Do not use punctuation. \
Do not show your reasoning. \
Do not provide any explanation. \
Follow the instructions exactly and \
always answer using a single uppercase letter.

For example, if the question is "What is the capital of France?" and the \
choices are "A. Paris", "B. London", "C. Rome", "D. Madrid",
- the answer should be "A"
- the answer should NOT be "Paris" or "A. Paris" or "A: Paris"

Please adhere strictly to the instructions.
""".strip(),
        },
        {"role": "user", "content": prompt},
    ]

    final_answer = None
    tries = 0
    valid_ids = [choice.id for choice in item.choices]

    while final_answer is None and tries < max_retries:

        outputs = pipeline(messages, **generation_args)
        answer = outputs[0]["generated_text"].strip()

        # No matter how hard you ask, Phi-4-mini-instruct might still
        # return answer in the form of "A: paris". We add a simple
        # post-processing step to extract the letter.
        if len(answer) != 1:
            match = re.match(r"^([A-Z]): .+", answer)
            if match:
                answer = match.group(1)

        if answer in valid_ids:
            final_answer = answer
        else:
            tries += 1

    if final_answer is None:
        raise RuntimeError(
            f"Failed to run inference on question {item.question}, "
            f"{item.choices}; the last model response was {answer}."
        )

    assert isinstance(final_answer, str)
    return final_answer


def print_score(score: AdaptiveTestScore) -> None:
    """Print adaptive test score with thetas, standard errors, and KL info."""
    print("\nAdaptive Test Score...")
    print(f"Final theta: {score.theta}")
    print(f"Final standard error: {score.std_error}")


def run_sync_example(
    pipeline: transformers.pipeline,
    dataset_name: str,
    project_id: str,
    experiment: str,
) -> None:
    """Run an adaptive test synchronously using the AdaptiveTest class."""
    print("\n=== Running Synchronous Example ===")
    runner = AdaptiveTest(lambda item: inference(pipeline, item))

    # Get user information
    me_response = runner.me()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    print(f"Organization: {me_response.organization.name}")

    print(f"\nStarting run with dataset name: {dataset_name}")
    results = runner.run(
        dataset_name,
        project_id,
        experiment,
        run_metadata=create_run_metadata(dataset_name),
        return_dict=False,
    )

    print(f"Run {results.run_id} completed.")

    if results.score is not None:
        print_score(results.score)
    else:
        print("No score available.")

    # Uncomment to replay the exact same questions from the previous run.
    # This is useful to test the stability of the model - note that this
    # works best with temperature > 0.

    # print("\nReplay run")
    # results = runner.run_replay(
    #     results.run_id, run_metadata, with_responses=True
    # )
    # print_results(results.results)


async def run_async_example(
    pipeline: transformers.pipeline,
    dataset_name: str,
    project_id: str,
    experiment: str,
) -> None:
    """Run an adaptive test asynchronously using the AdaptiveTest class."""

    print("\n=== Running Asynchronous Example ===")
    runner = AdaptiveTest(lambda item: inference(pipeline, item))

    # Get user information
    me_response = await runner.me_async()
    print(
        f"User: {me_response.user.firstname} {me_response.user.lastname} "
        f"({me_response.user.email})"
    )
    print(f"Organization: {me_response.organization.name}")

    print(f"\nStarting run with dataset name: {dataset_name}")
    results = await runner.run_async(
        dataset_name,
        project_id,
        experiment,
        run_metadata=create_run_metadata(dataset_name),
        return_dict=False,
    )

    print(f"Run {results.run_id} completed.")

    if results.score is not None:
        print_score(results.score)
    else:
        print("No score available.")

    # Uncomment to replay the exact same questions from the previous run.
    # This is useful to test the stability of the model - note that this
    # works best with temperature > 0.

    # print("\nReplay run")
    # results = await runner.run_replay_async(
    #     results.run_id, run_metadata, with_responses=True
    # )
    # print_results(results.results)


async def main() -> None:
    """
    Run both synchronous and asynchronous examples.

    Assumes TRISMIK_SERVICE_URL and TRISMIK_API_KEY are set either in
    environment or in .env file.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run adaptive testing examples with Trismik API"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MMLUPro2024",
        help="Name of the dataset to run (default: FinRAG2025)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="Project ID for the Trismik run",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name for the Trismik run",
    )
    args = parser.parse_args()

    load_dotenv()

    # We choose Phi-4-mini-instruct as an example as it requires
    # relatively low memory to run (< 8 GB). You can use any other
    # model that you have access to.

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/Phi-4-mini-instruct",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Run sync example
    run_sync_example(
        pipeline, args.dataset_name, args.project_id, args.experiment
    )

    # Run async example
    await run_async_example(
        pipeline, args.dataset_name, args.project_id, args.experiment
    )


if __name__ == "__main__":
    asyncio.run(main())
