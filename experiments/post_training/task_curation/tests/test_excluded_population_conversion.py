# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The excluded-population conversion pins bases, restores the oracle adapter and refuses drift."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
from experiments.post_training.task_curation.excluded_population_conversion import (
    ORACLE_ADAPTER,
    add_oracle_adapter,
    convert_population_rows,
    load_population,
)
from experiments.post_training.task_curation.harbor_cohort_conversion import CohortConversionRejected
from experiments.post_training.task_curation.native_source_conversion import BasePinMissing, pin_package_dockerfile

PINS = {
    "python:3.10-slim": "python@sha256:" + "a" * 64,
    "ubuntu:24.04": "ubuntu@sha256:" + "b" * 64,
}


def _resolve(reference: str) -> str:
    if reference not in PINS:
        raise ValueError(f"unresolvable in the test: {reference}")
    return PINS[reference]


V4_DOCKERFILE = """\
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest pytest-timeout
RUN mkdir -p /logs/verifier /context
COPY write_solution.sh /usr/local/bin/write_solution
RUN chmod +x /usr/local/bin/write_solution
"""

TEST_SCRIPT = b"""#!/bin/bash
# --- laion v4 patch: 4-tier forgiving verifier ---
mkdir -p /logs/verifier
echo "0" > /logs/verifier/reward.txt
[ -f /app/solution.txt ] || exit 1
[ -f /tests/solution.txt ] || exit 1
cmp -s /app/solution.txt /tests/solution.txt && echo "1" > /logs/verifier/reward.txt
exit 0
"""


def _v4_package(dockerfile: str, *, oracle_paths: tuple[str, ...] = ("solution/solution.txt",)) -> TaskSpec:
    """The archived laion-v4-patched package shape: gold file oracle, no executable solve.sh."""
    oracle = tuple(inline_resource(path, b"gold\n") for path in oracle_paths)
    return TaskSpec(
        id="TaskTrove:laion__fixture-v4/task-1",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="Complete the file."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", dockerfile.encode()),)),
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The native Harbor Trial executes this package's declared verifier"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("environment/Dockerfile", dockerfile.encode()),
                inline_resource("environment/write_solution.sh", b"#!/bin/bash\nmkdir -p /app\n"),
                inline_resource("instruction.md", b"Complete the file.\n"),
                inline_resource("task.toml", b"[verifier]\ntimeout_sec = 720.0\n"),
                inline_resource("tests/test.sh", TEST_SCRIPT),
                inline_resource("tests/solution.txt", b"gold\n"),
                inline_resource(".laion_v4_patched", b"laion v4 patch applied\n"),
            ),
            oracle=oracle,
        ),
        tags=("harbor",),
    )


def test_add_oracle_adapter_restores_the_executable_oracle():
    task = _v4_package(V4_DOCKERFILE)
    adapted = add_oracle_adapter(task)
    paths = {resource.path for resource in adapted.resources.oracle}
    assert paths == {"solution/solution.txt", "solution/solve.sh"}
    solve = next(resource for resource in adapted.resources.oracle if resource.path == "solution/solve.sh")
    assert resource_bytes(solve) == ORACLE_ADAPTER
    assert solve.mode == "0755"
    # every other resource is untouched
    assert {resource.path for resource in adapted.resources.verifier} == {
        resource.path for resource in task.resources.verifier
    }
    assert task_spec_sha256(adapted) != task_spec_sha256(task)


def test_add_oracle_adapter_is_idempotent():
    task = _v4_package(V4_DOCKERFILE, oracle_paths=("solution/solution.txt", "solution/solve.sh"))
    assert add_oracle_adapter(task) is task


def test_add_oracle_adapter_refuses_a_package_with_nothing_to_adapt():
    task = _v4_package(V4_DOCKERFILE, oracle_paths=())
    with pytest.raises(CohortConversionRejected):
        add_oracle_adapter(task)


def test_converted_row_pins_bases_and_clears_every_static_blocker():
    rows = [{"task_spec": _v4_package(V4_DOCKERFILE).model_dump_json()}]
    converted, receipts, counts = convert_population_rows(rows, _resolve)
    assert counts["converted_rows"] == 1 and counts["excluded_rows"] == 0
    outcome_task = TaskSpec.model_validate_json(converted[0]["task_spec"])
    assert isinstance(outcome_task.grader, ScriptGrader)
    assert task_blockers(outcome_task, docker_build=DockerBuildExecution.DECLARATIVE) == set()
    receipt = receipts[0]
    assert receipt["base_pins"] == [{"base": "python:3.10-slim", "pinned": PINS["python:3.10-slim"]}]
    assert receipt["input_task_sha256"] != receipt["output_task_sha256"]


def test_convert_population_rows_excludes_only_a_missing_base():
    missing_dockerfile = V4_DOCKERFILE.replace("python:3.10-slim", "eclipse-temurin:17-jdk-jammy")

    def resolving_missing(reference: str) -> str:
        if reference == "eclipse-temurin:17-jdk-jammy":
            raise BasePinMissing("eclipse-temurin:17-jdk-jammy: the tag no longer exists upstream")
        return _resolve(reference)

    rows = [
        {"task_spec": _v4_package(V4_DOCKERFILE).model_dump_json()},
        {"task_spec": _v4_package(missing_dockerfile).model_dump_json()},
    ]
    _converted, receipts, counts = convert_population_rows(rows, resolving_missing)
    assert counts["converted_rows"] == 1
    assert counts["excluded_rows"] == 1
    assert counts["exclusions"][0]["reason"] == "base_image_missing"
    assert len(receipts) == 1


def test_missing_base_survives_pinning_as_the_typed_rejection():
    """A registry-missing base is the typed per-task exclusion, never a generic wrap."""
    dockerfile = V4_DOCKERFILE.replace("python:3.10-slim", "eclipse-temurin:17-jdk-jammy")

    def missing(reference: str) -> str:
        raise BasePinMissing(f"The base image {reference} no longer exists upstream")

    with pytest.raises(BasePinMissing):
        pin_package_dockerfile(_v4_package(dockerfile), resolve=missing)


def test_load_population_refuses_a_changed_archive(tmp_path: Path):
    table = pa.table({"task_spec": [json.dumps({"id": "x"})]})
    path = tmp_path / "tasks.parquet"
    pq.write_table(table, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_population(str(path), digest) == [{"task_spec": json.dumps({"id": "x"})}]
    with pytest.raises(ValueError, match="changed"):
        load_population(str(path), "0" * 64)
