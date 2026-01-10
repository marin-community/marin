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

from marin.execution.executor import InputName

logger = logging.getLogger("ray")  # Initialize logger


def ckpt_path_to_step_name(path: str | InputName) -> str:
    """
    Converts a path pointing to a levanter or huggingface checkpoint into a name we can use as an id for an analysis
    step or similar. For instance, if the path is "checkpoints/{run_name}/checkpoints/step-{train_step_number}",
    we would get "run_name-train_step_number"

    If it's "meta-llama/Meta-Llama-3.1-8B" it should just be "Meta-Llama-3.1-8B"

    This method works with both strings and InputNames.

    If an input name, it expect the InputName's name to be something like "checkpoints/step-{train_step_number}"
    """

    def _get_step(path: str) -> str:
        # make sure it looks like step-{train_step_number}
        g = re.match(r"step-(\d+)/?$", path)
        if g is None:
            raise ValueError(f"Invalid path: {path}")

        return g.group(1)

    if isinstance(path, str):
        # If this looks like an HF hub path ("org/model"), return the last component.
        if re.match("^[^/]+/[^/]+$", path):
            return path.split("/")[-1]

        if path.endswith("/"):
            path = path[:-1]

        last_component = path.split("/")[-1]
        # If this looks like a Levanter checkpoint directory but doesn't end in a step,
        # treat it as invalid rather than falling back to the basename.
        if re.search(r"(^|/)checkpoints/[^/]+/checkpoints$", path) or re.search(
            r"(^|/)checkpoints/[^/]+/checkpoints/", path
        ):
            if not last_component.startswith("step-"):
                raise ValueError(f"Invalid path: {path}")

        # Otherwise, treat it as a generic path and use its basename.
        if not last_component.startswith("step-"):
            return last_component

        # we want llama-8b-tootsie-phase2-730000
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
            if components[-1].startswith("step-"):
                step = _get_step(components[-1])
            elif len(components) >= 2 and components[-2] == "checkpoints":
                raise ValueError(f"Invalid path: {path.name}")
            else:
                return components[-1]
        else:
            return name

    else:
        raise ValueError(f"Unknown type for path: {path}")

    return f"{name}-{step}"
