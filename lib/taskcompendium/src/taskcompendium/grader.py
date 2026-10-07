# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package a standard VerifyIT grader with its private input files."""

import json
from dataclasses import dataclass
from typing import Any

from verifyit.spec import Spec, mode_of, spec_to_table

from taskcompendium.models import TaskResource, TaskSpec, VerifierSpec
from taskcompendium.native_grader import NATIVE_COMMAND_KIND, NativeCommandSpec
from taskcompendium.runtime.resources import resource_bytes

SOURCE_UNAVAILABLE_KIND = "source_unavailable"


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


def native_command_package(spec: NativeCommandSpec, resources: tuple[TaskResource, ...] = ()) -> GraderPackage:
    """Package an unchanged source command with its private files."""
    return GraderPackage(VerifierSpec(kind=NATIVE_COMMAND_KIND, parameters_json=spec.model_dump_json()), resources)


def grader_config(task: TaskSpec) -> dict[str, Any]:
    """Read a declared source contract or an executable grader's private configuration."""
    if task.verifier.kind == SOURCE_UNAVAILABLE_KIND:
        value = json.loads(task.verifier.parameters_json)
    else:
        resource = next(resource for resource in task.resources.verifier if resource.path == "config.json")
        value = json.loads(resource_bytes(resource))
    if not isinstance(value, dict):
        raise ValueError("Grader configuration must be a JSON object")
    return value
