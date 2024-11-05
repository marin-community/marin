"""
executor_utils.py

Helpful functions for the executor
"""

import logging
from typing import Any

from deepdiff import DeepDiff

logger = logging.getLogger("ray")  # Initialize logger


def compare_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> bool:
    """Given 2 dictionaries, compare them and print the differences."""

    # DeepDiff is slow, so we only use it if the dictionaries are different
    if dict1 == dict2:
        return True

    # Use DeepDiff to compare the two dictionaries
    diff = DeepDiff(dict1, dict2, ignore_order=True, verbose_level=2)

    # If there's no difference, return True
    if not diff:
        return True
    else:
        logger.warning(diff.pretty())  # Log the differences
        return False
