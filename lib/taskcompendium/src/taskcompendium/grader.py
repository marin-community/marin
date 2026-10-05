# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package a standard VerifyIT grader with its private input files."""

import json
from dataclasses import dataclass
from typing import Any

from verifyit.spec import ScriptSpec, Spec, mode_of, spec_to_table

from taskcompendium.models import TaskResource, TaskSpec, VerifierSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes


@dataclass(frozen=True)
class GraderPackage:
    """A verifier descriptor and files rooted at the grader's private tests directory."""

    verifier: VerifierSpec
    resources: tuple[TaskResource, ...] = ()


def grader_package(spec: Spec, resources: tuple[TaskResource, ...] = ()) -> GraderPackage:
    """Package a VerifyIT specification with its private grading resources."""
    parameters = spec_to_table(spec)
    parameters.pop("mode")
    verifier = VerifierSpec(kind=mode_of(spec).value, parameters_json=json.dumps(parameters, allow_nan=False))
    return GraderPackage(verifier, resources)


def script_package(script: bytes, config: dict[str, Any], *, timeout: float = 60) -> GraderPackage:
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
    resource = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    value = json.loads(resource_bytes(resource))
    if not isinstance(value, dict):
        raise ValueError("Grader configuration must be a JSON object")
    return value
