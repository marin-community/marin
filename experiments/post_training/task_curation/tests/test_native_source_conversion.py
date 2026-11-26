# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The native source conversion pins FROM bases and refuses static blockers."""

import pytest
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    DockerBuildExecution,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.harbor_cohort_conversion import CohortConversionRejected
from experiments.post_training.task_curation.native_source_conversion import (
    BasePin,
    BasePinRejected,
    convert_native_source_row,
    pin_package_dockerfile,
    pinned_dockerfile,
)

PINS = {
    "python:3.9-bookworm": "python:3.9-bookworm@sha256:" + "a" * 64,
    "ubuntu:22.04": "ubuntu:22.04@sha256:" + "b" * 64,
    "namanjain12/orange3_final:deadbeef": "namanjain12/orange3_final@sha256:" + "c" * 64,
}


def _resolve(reference: str) -> str:
    if reference not in PINS:
        raise ValueError(f"unresolvable in the test: {reference}")
    return PINS[reference]


DOCKERFILE = """\
FROM --platform=linux/amd64 python:3.9-bookworm AS base
RUN true
FROM base AS final
FROM --platform=linux/amd64 namanjain12/orange3_final:deadbeef
RUN true
"""

EXPECTED = """\
FROM --platform=linux/amd64 python:3.9-bookworm@sha256:{digest_a} AS base
RUN true
FROM base AS final
FROM --platform=linux/amd64 namanjain12/orange3_final@sha256:{digest_c}
RUN true
"""


def test_pinned_dockerfile_rewrites_external_bases_only():
    text, pins = pinned_dockerfile(DOCKERFILE, _resolve)
    assert text == EXPECTED.format(digest_a="a" * 64, digest_c="c" * 64)
    assert pins == (
        BasePin(base="python:3.9-bookworm", pinned=PINS["python:3.9-bookworm"]),
        BasePin(base="namanjain12/orange3_final:deadbeef", pinned=PINS["namanjain12/orange3_final:deadbeef"]),
    )


def test_pinned_dockerfile_passes_pinned_and_scratch_references():
    text = f"FROM scratch\nFROM ubuntu:22.04@sha256:{'b' * 64}\n"
    pinned, pins = pinned_dockerfile(text, _resolve)
    assert pinned == text
    assert pins == ()


def test_pinned_dockerfile_refuses_unresolvable_base():
    with pytest.raises(BasePinRejected):
        pinned_dockerfile("FROM ghcr.io/org/image:latest\n", _resolve)


def _package(dockerfile: str) -> TaskSpec:
    build = DockerBuildContext(
        files=(inline_resource("Dockerfile", dockerfile.encode()),),
        empty_directories=(),
    )
    return TaskSpec(
        id="TaskTrove:native-fixture/task-1",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the repository."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=build,
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The native Harbor Trial executes this package's declared verifier"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("environment/Dockerfile", dockerfile.encode()),
                inline_resource("instruction.md", b"Fix the repository.\n"),
                inline_resource("task.toml", b"[verifier]\ntimeout_sec = 900.0\n"),
                inline_resource(
                    "tests/test.sh",
                    b"#!/bin/bash\nexit 0\n",
                ),
            ),
            oracle=(inline_resource("solution/solve.sh", b"#!/bin/bash\ntrue\n"),),
        ),
        tags=("harbor",),
    )


def test_pin_package_dockerfile_rewrites_only_the_dockerfile_resource():
    task = _package(DOCKERFILE)
    pinned, pins = pin_package_dockerfile(task, _resolve)
    assert len(pins) == 2
    assert {resource.path for resource in pinned.resources.verifier} == {
        resource.path for resource in task.resources.verifier
    }
    dockerfile = next(r for r in pinned.resources.verifier if r.path == "environment/Dockerfile")
    assert resource_bytes(dockerfile).decode() == EXPECTED.format(digest_a="a" * 64, digest_c="c" * 64)


def test_convert_native_source_row_yields_executable_grader_without_blockers():
    task = _package(DOCKERFILE)
    outcome = convert_native_source_row(task, input_task_sha256=task_spec_sha256(task), resolve=_resolve)
    assert isinstance(outcome.task.grader, ScriptGrader)
    assert outcome.base_pins[0].base == "python:3.9-bookworm"
    assert task_blockers(outcome.task, docker_build=DockerBuildExecution.DECLARATIVE) == set()
    assert outcome.input_task_sha256 == task_spec_sha256(task)
    assert outcome.output_task_sha256 != outcome.input_task_sha256


def test_convert_native_source_row_refuses_when_blockers_remain():
    # An oracle-less package still carries a static blocker after pinning; the
    # conversion must reject it instead of silently shrinking the population.
    task = _package(DOCKERFILE)
    stripped = task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": ()})})
    with pytest.raises(CohortConversionRejected):
        convert_native_source_row(stripped, input_task_sha256=task_spec_sha256(task), resolve=_resolve)
