# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package a grader with the verifier resources installed under its ``/tests`` directory."""

import copy
import json
from dataclasses import dataclass
from typing import Any

from verifyit.spec import Spec, spec_to_table

from taskcompendium.models import EnvironmentRequirements, Grader, NoGrader, TaskResource, TaskSpec, VerifyitGrader
from taskcompendium.runtime.resources import resource_bytes


@dataclass(frozen=True)
class GraderPackage:
    """A grader and the verifier resources it reads, with paths relative to ``/tests``."""

    grader: Grader
    resources: tuple[TaskResource, ...] = ()


def verifyit_package(
    spec: Spec, resources: tuple[TaskResource, ...] = (), environment: EnvironmentRequirements | None = None
) -> GraderPackage:
    """Package a verifyit specification; ``environment`` selects grading in a fresh machine."""
    parameters = spec_to_table(spec)
    mode = parameters.pop("mode")
    return GraderPackage(VerifyitGrader(mode=mode, parameters=parameters, environment=environment), resources)


def grader_config(task: TaskSpec) -> dict[str, Any]:
    """Read a NoGrader's source contract or the ``config.json`` verifier resource."""
    if isinstance(task.grader, NoGrader):
        return copy.deepcopy(task.grader.contract)
    resource = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    value = json.loads(resource_bytes(resource))
    if not isinstance(value, dict):
        raise ValueError("Grader configuration must be a JSON object")
    return value
