"""
Shared utilities for Trismik examples.

This module provides common functionality used across example scripts:
- Random hash generation for auto-naming
- Progress callbacks with tqdm
- Argument parser setup with standard options
"""

import argparse
import secrets
from typing import Callable, Optional

from tqdm.auto import tqdm


def generate_random_hash() -> str:
    """
    Generate a secure random 8-character hash for naming.

    Returns:
        str: 8-character hexadecimal string.
    """
    return secrets.token_hex(4)


def create_progress_callback(desc: str = "Progress") -> Callable[[int, int], None]:
    """
    Create a progress callback that uses tqdm.

    Args:
        desc: Description for the progress bar.

    Returns:
        Callback function compatible with client.run() on_progress parameter.
    """
    pbar: Optional[tqdm] = None

    def callback(current: int, total: int) -> None:
        nonlocal pbar

        if pbar is None:
            pbar = tqdm(total=total, desc=desc)

        pbar.total = total
        pbar.n = current
        pbar.refresh()

        if current >= total and pbar is not None:
            pbar.close()
            pbar = None

    return callback


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """
    Create argument parser with common optional arguments.

    Returns an ArgumentParser (not parsed args) so examples can add
    custom arguments before calling parse_args().

    Args:
        description: Description for the argument parser.

    Returns:
        ArgumentParser with common optional arguments added.

    Example:
        >>> parser = create_base_parser("My example")
        >>> parser.add_argument("--model-name", type=str, default="gpt-4")
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MMLUPro2024",
        help="Name of the dataset to run (default: MMLUPro2024)",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help=("Project ID for the Trismik run " "(optional - creates new project if not provided)"),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help=(
            "Experiment name for the Trismik run "
            "(optional - generates random name if not provided)"
        ),
    )
    return parser
