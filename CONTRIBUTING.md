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


## Running Tests

To run the tests, ensure your Poetry environment is activated, then run:

```bash
poetry run pytest
```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. To set up pre-commit hooks, run:

```bash
poetry run pre-commit install
```

This will install the pre-commit hooks, which will run automatically on every git commit.

## Code Style

This project uses Black for code formatting and isort for import sorting. To format your code, run:

```bash
poetry run black .
poetry run isort .
```

## Submitting Changes

1. Create a new branch for your changes.
2. Make your changes and commit them.
3. Push your branch to the repository.
4. Create a pull request.

Thank you for contributing to Trismik!
