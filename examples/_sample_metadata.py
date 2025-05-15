from typing import Final

from trismik.types import TrismikSessionMetadata

sample_metadata: Final[TrismikSessionMetadata] = TrismikSessionMetadata(
    model_metadata=TrismikSessionMetadata.ModelMetadata(
        name="custom-llm-v1.2",
        version="1.2.0",
        architecture="Transformer",
        parameters="13B",
        provider="Hugging Face",
        training_dataset="Synthetic Financial + PubMed2023",
        training_cutoff_date="2023-12-31",
        price_per_input_token=0.00001,
        price_per_output_token=0.00002,
        temperature=0.7,
        top_p=0.95,
        stop_sequences=["\n"],
        max_tokens=500,
        prompt_strategy="Standard instruction + context prepending",
        prompt="Provide a concise summary of the following text: ...",
    ),
    test_configuration={
        "task_name": "LayMed2024",
        "response_format": "Multiple-choice",
        "confidence_score": 0.85,
        "test_stage": "Validation",
    },
    inference_setup={
        "hardware": "NVIDIA A100, 80GB",
        "framework": "PyTorch",
        "latency_ms": 450,
        "batch_size": 8,
        "max_tokens": 500,
    },
)
