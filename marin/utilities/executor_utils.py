"""
executor_utils.py

Helpful functions for the executor
"""

import logging
from typing import Any

import ray
import toml
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


def get_pip_dependencies(
    pip_dep: list[str], pyproject_path: str = "pyproject.toml", include_parents_pip: bool = True
) -> list[str]:
    """Given a list of pip dependencies, this function searches keys of project.optional-dependencies in pyproject.toml,
    If key is not found it is considered a pacakge name. include_parents_pip: include the parent pip packages.
    Finally return a list of all dependencies."""
    dependencies = []
    try:
        with open(pyproject_path, "r") as f:
            pyproject = toml.load(f)
            optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})
            for dep in pip_dep:
                if dep in optional_dependencies:
                    dependencies.extend(optional_dependencies[dep])
                else:
                    dependencies.append(dep)
    except FileNotFoundError:
        logger.error(f"File {pyproject_path} not found.")
    except toml.TomlDecodeError:
        logger.error(f"Failed to parse {pyproject_path}.")

    if include_parents_pip:
        dependencies.extend(ray.get_runtime_context().runtime_env.get("pip", {}).get("packages", []))

    return dependencies
