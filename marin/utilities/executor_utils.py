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

    # Use DeepDiff to compare the two dictionaries
    diff = DeepDiff(dict1, dict2, ignore_order=True, verbose_level=2)

    # If there's no difference, return True
    if not diff:
        return True
    else:
        # Process additions
        for path, value in diff.get("dictionary_item_added", {}).items():
            logger.warning(f"+ {path}: {value}")

        # Process deletions
        for path, value in diff.get("dictionary_item_removed", {}).items():
            logger.warning(f"- {path}: {value}")

        # Process changed values
        for path, change in diff.get("values_changed", {}).items():
            logger.warning(f'- {path}: {change["old_value"]}')
            logger.warning(f'+ {path}: {change["new_value"]}')

        return False
