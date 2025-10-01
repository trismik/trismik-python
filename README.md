<h1 align="center"> Trismik SDK</h1>

<p align="center">
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/trismik">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [API Key Setup](#api-key-setup)
  - [Basic Usage](#basic-usage)
- [Features](#features)
  - [Progress Reporting](#progress-reporting)
  - [Replay Functionality](#replay-functionality)
- [Examples](#examples)
- [Interpreting Results](#interpreting-results)
  - [Theta (θ)](#theta-θ)
  - [Other Metrics](#other-metrics)
- [Contributing](#contributing)
- [License](#license)

## Overview

[**Trismik**](https://trismik.com) is a Cambridge, UK based startup offering adversarial testing for LLMs. The APIs we provide through this library allow you to call our adaptive test engine and evaluate LLMs up to 95% faster (and cheaper!) than traditional evaluation techniques.

Our **adaptive testing** algorithm allows to estimate the precision of the model by looking only at a small portion of a dataset. Through this library, we provide access to a number of open source datasets over several dimensions (reasoning, toxicity, tool use...) to speed up model testing in several scenarios, like foundation model training, supervised fine tuning, prompt engineering, and so on.

## Quick Start

### Installation

To use our API, you need to get an API key first. Please register on [dashboard.trismik.com](https://dashboard.trismik.com) and obtain an API key.

Trismik is available via [pypi](https://pypi.org/project/trismik/). To install Trismik, run the following in your terminal (in a virtualenv, if you use one):

```bash
pip install trismik
```

### API Key Setup

You can provide your API key in one of the following ways:

1. **Environment Variable**:
   ```bash
   export TRISMIK_API_KEY="your-api-key"
   ```

2. **`.env` File**:
   ```bash
   # .env
   TRISMIK_API_KEY=your-api-key
   ```
   Then load it with `python-dotenv`:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

3. **Direct Initialization**:
   ```python
   client = TrismikClient(api_key="YOUR_API_KEY")
   ```

### Basic Usage

Here's the simplest way to run an adaptive test:

```python
from trismik import TrismikClient, TrismikRunMetadata
from trismik.types import TrismikItem

# Define your item processor
def model_inference(item: TrismikItem) -> str:
    # Your model inference logic here
    # See examples/ folder for real-world implementations
    return item.choices[0].id  # Example: pick first choice

# Run the test
with TrismikClient() as client:
    results = client.run(
        test_id="MMLUPro2024",
        project_id="your-project-id",  # Get from dashboard or create with client.create_project()
        experiment="my-experiment",
        run_metadata=TrismikRunMetadata(
            model_metadata={"name": "my-model", "provider": "local"},
            test_configuration={"task_name": "MMLUPro2024"},
            inference_setup={},
        ),
        item_processor=model_inference,
    )

    print(f"Theta: {results.score.theta}")
    print(f"Standard Error: {results.score.std_error}")
```

**For async usage:**

```python
from trismik import TrismikAsyncClient

async with TrismikAsyncClient() as client:
    results = await client.run(
        test_id="MMLUPro2024",
        project_id="your-project-id",
        experiment="my-experiment",
        run_metadata=TrismikRunMetadata(...),
        item_processor=model_inference,  # Can be sync or async
    )
```

## Features

### Progress Reporting

Add optional progress tracking with a callback:

```python
from tqdm.auto import tqdm
from trismik.settings import evaluation_settings

def create_progress_callback():
    pbar = tqdm(total=evaluation_settings["max_iterations"], desc="Running test")

    def callback(current: int, total: int):
        pbar.total = total
        pbar.n = current
        pbar.refresh()
        if current >= total:
            pbar.close()

    return callback

# Use it in your run
with TrismikClient() as client:
    results = client.run(
        # ... other parameters ...
        on_progress=create_progress_callback(),
    )
```

The library is silent by default - progress reporting is entirely optional.

### Replay Functionality

Replay the exact sequence of questions from a previous run to test model stability:

```python
with TrismikClient() as client:
    # Run initial test
    results = client.run(
        test_id="MMLUPro2024",
        project_id="your-project-id",
        experiment="experiment-1",
        run_metadata=metadata,
        item_processor=model_inference,
    )

    # Replay with same questions
    replay_results = client.run_replay(
        previous_run_id=results.run_id,
        run_metadata=new_metadata,
        item_processor=model_inference,
        with_responses=True,  # Include individual responses
    )
```

## Examples

Complete working examples are available in the `examples/` folder:

- **[`example_adaptive_test.py`](examples/example_adaptive_test.py)** - Basic adaptive testing with both sync and async patterns, including replay functionality
- **[`example_openai.py`](examples/example_openai.py)** - Integration with OpenAI API models
- **[`example_transformers.py`](examples/example_transformers.py)** - Integration with Hugging Face Transformers models

To run the examples:

```bash
# Clone the repository and install with examples dependencies
git clone https://github.com/trismik/trismik-python
cd trismik-python
poetry install --with examples

# Run an example
poetry run python examples/example_adaptive_test.py --dataset-name MMLUPro2024
```

## Interpreting Results

### Theta (θ)

Our adaptive test returns several values; however, you will be interested mainly in `theta`. Theta ($\theta$) is our metric; it measures the ability of the model on a certain dataset, and it can be used as a proxy to approximate the original metric used on that dataset. For example, on an accuracy-based dataset, a high theta correlates with a high accuracy, and low theta correlates with low accuracy.

$\theta$ is intrinsically linked to the difficulty of the items a model can answer correctly. On a datasets where the item difficulties are balanced and evenly distributed, $\theta=0$ corresponds to a 50% chance for a model to get an answer right - in other words, to an accuracy of 50%.
A negative theta means that the model will give more bad answers than good ones, while a positive theta means that the model will give more good answers than bad answers.
While theta is unbounded in our implementation (i.e. $-\infty < \theta < \infty$), in practice we have that for most cases $\theta$ will take values between -3 and 3.

Compared to classical benchmark testing, $\theta$ from adaptive testing uses fewer but more informative items while avoiding noise from overly easy or difficult questions. This makes it a more efficient and stable measure, especially on very large datasets.

### Other Metrics

- **Standard Deviation (`std`)**:
  - A measure of the uncertainty or error in the theta estimate
  - A smaller `std` indicates a more precise estimate
  - You should see a `std` around or below 0.25

- **Correct Responses (`responsesCorrect`)**:
  - The number of correct answers delivered by the model

  - **Important note**: A higher number of correct answers does not necessarily
  correlate with a high theta. Our algorithm navigates the dataset to find a
   balance of "hard" and "easy" items for your model, so by the end of the test,
  it encounters a representative mix of inputs it can and cannot handle. In
   practice, expect responsesCorrect to be roughly half of responsesTotal.

- **Total Responses (`responsesTotal`)**:
  - The number of items processed before reaching a stable theta.
  - Expected range: 60 ≤ responses_total ≤ 150

## Contributing

See `CONTRIBUTING.md`.

## License

This library is licensed under the MIT license. See `LICENSE` file.
