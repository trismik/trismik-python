from trismik.types import TrismikSessionMetadata

sample_metadata: TrismikSessionMetadata = TrismikSessionMetadata(
    model_metadata=TrismikSessionMetadata.ModelMetadata(
        name="example-model",
        version="0.0.1",
        architecture="ExampleNet",
        parameters="N/A",
        provider="ExampleProvider",
        training_dataset="ExampleDataset",
        training_cutoff_date="2024-01-01",
        price_per_input_token=0.0,
        price_per_output_token=0.0,
        temperature=1.0,
        top_p=1.0,
        stop_sequences=["<END>"],
        max_tokens=100,
        prompt_strategy="Example strategy",
        prompt="This is an example prompt.",
    ),
    test_configuration={
        "task_name": "ExampleTask",
        "response_format": "ExampleFormat",
        "confidence_score": 1.0,
        "test_stage": "ExampleStage",
    },
    inference_setup={
        "hardware": "ExampleHardware",
        "framework": "ExampleFramework",
        "latency_ms": 0,
        "batch_size": 1,
        "max_tokens": 100,
    },
)

# Replay session metadata with different configuration
replay_metadata: TrismikSessionMetadata = TrismikSessionMetadata(
    model_metadata=TrismikSessionMetadata.ModelMetadata(
        name="replay-model",
        version="1.0.0",
        architecture="ReplayNet",
        parameters="7B",
        provider="ReplayProvider",
        training_dataset="ReplayDataset",
        training_cutoff_date="2024-06-01",
        price_per_input_token=0.001,
        price_per_output_token=0.002,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["<END>", "<STOP>"],
        max_tokens=200,
        prompt_strategy="Replay strategy with different parameters",
        prompt="This is a replay prompt with different configuration.",
    ),
    test_configuration={
        "task_name": "ReplayTask",
        "response_format": "ReplayFormat",
        "confidence_score": 0.95,
        "test_stage": "ReplayStage",
        "replay_mode": True,
        "original_session_id": "placeholder",  # Will be set dynamically
    },
    inference_setup={
        "hardware": "ReplayHardware",
        "framework": "ReplayFramework",
        "latency_ms": 50,
        "batch_size": 4,
        "max_tokens": 200,
        "replay_timestamp": "2024-12-19T10:00:00Z",
    },
)
