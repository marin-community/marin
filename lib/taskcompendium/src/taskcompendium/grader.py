# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package a standard VerifyIT grader with its private input files."""

import json
from typing import Any

from verifyit.spec import ScriptSpec, Spec, mode_of, spec_to_table

from taskcompendium.environment import EnvironmentFile
from taskcompendium.models import TaskSpec, VerifierSpec
from taskcompendium.runtime.resources import inline_resource


def grader_package(spec: Spec, resources: tuple[EnvironmentFile, ...] = ()) -> VerifierSpec:
    """Package a VerifyIT specification with its private grading resources."""
    parameters = spec_to_table(spec)
    parameters.pop("mode")
    return VerifierSpec(
        kind=mode_of(spec).value,
        parameters_json=json.dumps(parameters, allow_nan=False),
        files=tuple(file.model_copy(update={"path": "/tests" + file.path}) for file in resources),
    )


def script_package(script: bytes, config: dict[str, Any], *, timeout: float = 60) -> VerifierSpec:
    """Bundle trusted recipe code and its private JSON configuration."""
    return grader_package(
        ScriptSpec(path="grader.py", verdict_file="verdict.json", timeout=timeout),
        (
            inline_resource("grader.py", script),
            inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
        ),
    )


def grader_config(task: TaskSpec) -> dict[str, Any]:
    """Read the bundled private configuration of a script grader."""
    resource = next(resource for resource in task.verifier.files if resource.path == "/tests/config.json")
    value = json.loads(resource.content)
    if not isinstance(value, dict):
        raise ValueError("Grader configuration must be a JSON object")
    return value
