# Contributing to Trismik

Thank you for your interest in contributing to Trismik! This document provides guidelines and instructions for setting up your development environment using Poetry.

## Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation) installed on your system

## Setting Up Your Development Environment

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd trismik
   ```

2. **Create a Poetry Environment with Python 3.9**

   Poetry will automatically create a virtual environment for you. To ensure it uses Python 3.9, run:

   ```bash
   poetry env use 3.9
   ```

   This command will create a new Poetry environment using Python 3.9. If you have multiple Python versions installed, make sure Python 3.9 is available on your system.

3. **Install Dependencies**

   Install all project dependencies using Poetry:

   ```bash
   poetry install
   ```

   This will install all dependencies specified in `pyproject.toml`, including development dependencies.

   Note that if you want to run the examples, you will need to run

   ```bash
   poetry install --with examples
   ```

   This will include additional dependences we need for running examples (`openai`, `transformers`, and so on).

4. **Set Up API Key**

   Create a `.env` file in the project root with your Trismik API key:

   ```bash
   # .env
   TRISMIK_API_KEY=your-api-key-here
   ```

   The SDK will automatically load this via `python-dotenv` when running tests and examples.



## Running Tests

To run the tests, ensure your Poetry environment is activated, then run:

```bash
poetry run pytest
```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and automate code generation. To set up pre-commit hooks, run:

```bash
poetry run pre-commit install
```

This will install the pre-commit hooks, which will run automatically on every git commit.

**What runs on each commit:**
- **black**: Code formatting (100 character line length)
- **autoflake**: Remove unused imports
- **isort**: Sort imports (black profile)
- **flake8**: Linting and docstring checks
- **mypy**: Type checking
- **unasync**: Generate sync code from async source (critical!)
- **pytest**: Full test suite

You can run all hooks manually:
```bash
poetry run pre-commit run --all-files
```

**Important**: Always use `poetry run` for all commands to ensure you're using the Poetry-managed environment.

## Submitting Changes

We follow [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow), a lightweight, branch-based workflow that supports teams and projects where deployments are made regularly. The main branch should always be deployable, and new features are developed in feature branches.

Before submitting any changes, please ensure:
1. All tests pass (`poetry run pytest`)
2. You have installed and run the pre-commit hooks (`poetry run pre-commit install`)
3. Your code follows our style guidelines (enforced by pre-commit hooks)

### For External Contributors
1. Fork the repository to your GitHub account
2. Clone your fork locally
3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/trismik/trismik-python.git
   ```
4. Create a new branch for your changes
5. Make your changes and commit them
6. Push your branch to your fork
7. Create a pull request from your fork to the main repository

### For Direct Contributors
1. Create a new branch for your changes
2. Make your changes and commit them
3. Push your branch to the repository
4. Create a pull request

**Important**: Pull requests will only be merged if all tests pass and the code meets our quality standards. Make sure to run the test suite locally before submitting your changes.

## Development Guidelines

### Code Architecture

This project uses an **async-first, sync-generated** pattern:
- All client code lives in `src/trismik/_async/`
- **NEVER edit `src/trismik/_sync/` directly** - it's auto-generated via `unasync`
- The pre-commit hook automatically generates sync code from async source
- Transformation rules are defined in `pyproject.toml` under `[tool.unasync]`

### Testing Strategy

**Async tests are the source of truth.** Write comprehensive async tests for all functionality.

Sync tests should be **strategic, not comprehensive**:
- Test the async-to-sync transformation mechanism
- Test sync-specific behaviors (e.g., event loop handling)
- Don't duplicate every async test case in sync

This approach avoids redundant test maintenance while ensuring both interfaces work correctly.

### Project Principles

Follow these software engineering principles:

- **SOLID**: Single responsibility, open/closed, and dependency inversion
- **KISS (Keep It Simple)**: Favor simplicity over cleverness
- **DRY (Don't Repeat Yourself)**: Share code via the `_async/` → `_sync/` generation pattern

Thank you for contributing to Trismik!
