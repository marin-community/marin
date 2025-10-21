# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
executor_utils.py

Helpful functions for the executor
"""

import logging
import re
from typing import TYPE_CHECKING, Any

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
        if (
            re.match("^[^/]+/[^/]+$", path) or "gcsfuse_mount/models" in path
        ):  # exactly 1 slash or in the pretrained gcsfuse dir
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
        if path.name:
            components = path.name.split("/")
            if not components[-1]:
                components = components[:-1]
            step = _get_step(components[-1])
        else:
            return name

    else:
        raise ValueError(f"Unknown type for path: {path}")

    return f"{name}-{step}"
