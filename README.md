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
   client = TrismikAsyncClient(api_key="YOUR_API_KEY")
   ```

### Basic Usage

Running a test is straightforward:

1. Implement a method that wraps model inference over a dataset item
2. Create an `AdaptiveTest` instance
3. Run the test!

Here's a basic example:

```python
def model_inference(item: TrismikItem) -> Any:
    model_output = ...  # call your model here
    return model_output


# Initialize the test runner
runner = AdaptiveTest(model_inference)

# Run the test
results = await runner.run_async(
    "MMLUPro2025",  # or any dataset we support
    with_responses=True,
    run_metadata=sample_metadata,
)

# Print the test output
for result in results:
    print(f"{result.trait} ({result.name}): {result.value}")
```

### Examples

You can find more examples in the `examples` folder:
- [`example_transformers.py`](examples/example_transformers.py) - Example using Hugging Face Transformers models
- [`example_openai.py`](examples/example_openai.py) - Example using OpenAI models
- [`example_adaptive_test.py`](examples/example_adaptive_test.py) - Example of adaptive testing configuration

To run the examples, you will need to clone this repo, navigate to the
source folder, and then run:

```bash
poetry install --with examples
poetry run python examples/example_adaptive_test.py # or any other example
```

## Interpreting Results

### Theta (θ)

Our adversarial test returns several values; however, you will be interested mainly in `theta`. Theta ($\theta$) is our metric; it measures the ability of the model on a certain dataset, and it can be used as a proxy to approximate the original metric used on that dataset. For example, on an accuracy-based dataset, a high theta correlates with a high accuracy, and low theta correlates with low accuracy.

To interpret a theta score, consider that $\theta=0$ corresponds to a 50% chance for a model to get an answer right - in other words, to an accuracy of 50%.
A negative theta means that the model will give more bad answers then good ones, while a positive theta means that the model will give more good answers then bad answers.
While theta is unbounded in our implementation (i.e. $-\infty < \theta < \infty$), in practice we have that for most cases $\theta$ will take values between -3 and 3.

Compared to classical benchmark testing, the estimated accuracy from adaptive testing uses fewer but more informative items while avoiding noise from overly easy or difficult questions. This makes it a more efficient and stable measure, especially on very large datasets.

### Other Metrics

- **Standard Deviation (`std`)**:
  - A measure of the uncertainty or error in the theta estimate
  - A smaller `std` indicates a more precise estimate
  - You should see a `std` around or below 0.25

- **Correct Responses (`responsesCorrect`)**:
  - The number of correct answers delivered by the model

  - **Important note**: A higher number of correct answers does not necessarily
  correlate with a high theta. Our algorithm navigates the dataset to find a
   balance of “hard” and “easy” items for your model, so by the end of the test,
  it encounters a representative mix of inputs it can and cannot handle. In
   practice, expect responsesCorrect to be roughly half of responsesTotal.

- **Total Responses (`responsesTotal`)**:
  - The number of items processed before reaching a stable theta.
  - Expected range: 60 ≤ responses_total ≤ 80

## Contributing

See `CONTRIBUTING.md`.

## License

This library is licensed under the MIT license. See `LICENSE` file.
