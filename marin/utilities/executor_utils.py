"""
executor_utils.py

Helpful functions for the executor
"""

import logging

from deepdiff import DeepDiff

logger = logging.getLogger("ray")  # Initialize logger


def compare_dicts(dict1, dict2):
    """Given 2 dictionaries, compare them and print the differences."""

    # Use DeepDiff to compare the two dictionaries
    diff = DeepDiff(dict1, dict2, ignore_order=True)

    # If there's no difference, print that they are equal
    if not diff:
        return False
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

        return True
