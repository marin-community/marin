"""
executor_utils.py

Helpful functions for the executor
"""

import logging
import re
from typing import TYPE_CHECKING, Any

import ray
import toml
from deepdiff import DeepDiff

if TYPE_CHECKING:
    from marin.execution.executor import InputName
else:
    InputName = Any


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


def ckpt_path_to_step_name(path: str | InputName) -> str:
    """
    Converts a path pointing to a levanter or huggingface checkpoint into a name we can use as an id for an analysis
    step or similar. For instance, if the path is "checkpoints/{run_name}/checkpoints/step-{train_step_number}",
    we would get "run_name-train_step_number"

    If it's "meta-llama/Meta-Llama-3.1-8B" it should just be "Meta-Llama-3.1-8B"

    This method works with both strings and InputNames.

    If an input name, it expect the InputName's name to be something like "checkpoints/step-{train_step_number}"
    """
    from marin.execution.executor import InputName

    def _get_step(path: str) -> bool:
        # make sure it looks like step-{train_step_number}
        g = re.match(r"step-(\d+)/?$", path)
        if g is None:
            raise ValueError(f"Invalid path: {path}")

        return g.group(1)

    if isinstance(path, str):
        # see if it looks like an hf hub path: "org/model". If so, just return the last component
        if re.match("^[^/]+/[^/]+$", path):  # exactly 1 slash
            return path.split("/")[-1]

        # we want llama-8b-tootsie-phase2-730000
        if path.endswith("/"):
            path = path[:-1]

        components = path.split("/")
        name = components[-3].split("/")[-1]
        step = _get_step(components[-1])
    elif isinstance(path, InputName):
        name = path.step.name
        name = name.split("/")[-1]
        components = path.name.split("/")
        if not components[-1]:
            components = components[:-1]
        step = _get_step(components[-1])
    else:
        raise ValueError(f"Unknown type for path: {path}")

    return f"{name}-{step}"
