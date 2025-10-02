# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Trismik Python SDK, a client library for accessing Trismik's adversarial testing API for LLMs. The project is structured as a Poetry-managed Python package that provides both synchronous and asynchronous interfaces for running adaptive tests on language models.

### Core Architecture

The codebase follows an **async-first, sync-generated** pattern using the `unasync` library:

- **`src/trismik/_async/`**: Source of truth for all client code (async implementations)
  - **`client.py`**: Core async HTTP client (`TrismikAsyncClient`) that handles all API communication
  - **`helpers.py`**: Async helper functions for test item processing
  - **`_test_transform.py`**: Test transformation utilities
- **`src/trismik/_sync/`**: Auto-generated sync code (DO NOT EDIT DIRECTLY)
  - Generated via `unasync` from `_async/` on every commit via pre-commit hook
  - Contains sync versions: `TrismikClient`, sync helpers, etc.
- **`src/trismik/types.py`**: Type definitions for all API request/response objects, test items, and results
- **`src/trismik/_mapper.py`**: Response mapping utilities for converting API responses to internal types
- **`src/trismik/settings.py`**: Configuration settings (client, environment, evaluation parameters)
- **`src/trismik/exceptions.py`**: Custom exception classes for API errors
- **`src/trismik/_utils.py`**: Utility functions (headers, error handling, etc.)

**Critical**: All development happens in `_async/`. The sync client is automatically generated via `unasync` transformations defined in `pyproject.toml`. The pre-commit hook runs `run_unasync.py` to generate sync code before each commit.

## Development Commands

### Environment Setup
```bash
# Install dependencies (development)
poetry install

# Install with examples dependencies
poetry install --with examples

# Use Python 3.9 environment
poetry env use 3.9

# Install pre-commit hooks (required for development)
poetry run pre-commit install
```

**API Key Setup**: Create a `.env` file with `TRISMIK_API_KEY=your-key` for local development. The SDK will automatically load it via `python-dotenv`.

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

- Line length: 100 characters (enforced by black)
- Python 3.9+ compatibility required
- Type hints mandatory for all public APIs (`disallow_untyped_defs = true`)
- Import sorting via isort with black profile
- Docstrings required for public functions (enforced by flake8-docstrings)
- All async functions must be properly typed with return type annotations

## Testing Philosophy

**Async tests are the source of truth.** Write comprehensive async tests for all functionality. Sync tests should be strategic:
- Test the async-to-sync transformation mechanism
- Test sync-specific behaviors (e.g., event loop handling)
- Don't duplicate every async test case in sync

The `tests/_mocker.py` module provides mock utilities for HTTP responses. Pre-commit hooks automatically run the full test suite before each commit.

## Project Guidelines

- **SOLID Principles**: Follow single responsibility, open/closed, and dependency inversion principles
- **KISS (Keep It Simple)**: Favor simplicity over cleverness
- **DRY (Don't Repeat Yourself)**: Share code via the `_async/` → `_sync/` generation pattern
