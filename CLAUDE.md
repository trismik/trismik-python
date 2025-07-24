# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Trismik Python SDK, a client library for accessing Trismik's adversarial testing API for LLMs. The project is structured as a Poetry-managed Python package that provides both synchronous and asynchronous interfaces for running adaptive tests on language models.

### Core Architecture

- **`src/trismik/adaptive_test.py`**: Main entry point containing the `AdaptiveTest` class that provides both sync and async interfaces for running tests
- **`src/trismik/client_async.py`**: Core async HTTP client (`TrismikAsyncClient`) that handles API communication
- **`src/trismik/types.py`**: Type definitions for all API request/response objects, test items, and results
- **`src/trismik/_mapper.py`**: Response mapping utilities for converting API responses to internal types
- **`src/trismik/settings.py`**: Configuration settings for client behavior and evaluation parameters
- **`src/trismik/exceptions.py`**: Custom exception classes for API errors

The async client is the foundation - sync methods in `AdaptiveTest` wrap the async implementations using `nest_asyncio` to handle event loop management.

## Development Commands

### Environment Setup
```bash
# Install dependencies (development)
poetry install

# Install with examples dependencies
poetry install --with examples

# Use Python 3.9 environment
poetry env use 3.9
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_adaptive_test.py

# Run with coverage
poetry run pytest --cov=src/trismik
```

### Code Quality
**Important**: Always use pre-commit hooks instead of running tools directly:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run all pre-commit hooks manually
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run black
poetry run pre-commit run flake8
poetry run pre-commit run mypy
poetry run pre-commit run isort
```

Pre-commit automatically runs: black (formatting), autoflake (unused imports), isort (import sorting), flake8 (linting), mypy (type checking), and pytest (tests).

### Running Examples
```bash
# Basic adaptive test example
poetry run python examples/example_adaptive_test.py

# OpenAI integration example
poetry run python examples/example_openai.py

# Transformers integration example
poetry run python examples/example_transformers.py
```

## Key Development Patterns

### API Client Pattern
The codebase follows an async-first pattern where `TrismikAsyncClient` handles all HTTP communication. Sync methods in `AdaptiveTest` wrap async calls using `nest_asyncio.apply()` for compatibility.

### Type Safety
All API interactions are strongly typed using Pydantic-style dataclasses in `types.py`. The `_mapper.py` module handles conversion between API JSON responses and typed objects.

### Error Handling
Custom exceptions in `exceptions.py` map to specific HTTP status codes:
- `TrismikValidationError` (422)
- `TrismikPayloadTooLargeError` (413)
- `TrismikApiError` (general API errors)

### Configuration
Settings are centralized in `settings.py` with environment variable support. API keys can be provided via `TRISMIK_API_KEY` environment variable or direct parameter.

## Code Style Requirements

- Line length: 80 characters (enforced by black)
- Python 3.9+ compatibility required
- Type hints mandatory for all public APIs (`disallow_untyped_defs = true`)
- Import sorting via isort with black profile
- Docstrings required for public functions (enforced by flake8-docstrings)
- All async functions must be properly typed with `Awaitable` return types

## Testing Architecture

Tests use pytest with async support (`pytest-asyncio`). The `tests/_mocker.py` module provides mock utilities for HTTP responses. Test files follow the pattern `test_[module_name].py` and include both sync and async test variants where applicable.
