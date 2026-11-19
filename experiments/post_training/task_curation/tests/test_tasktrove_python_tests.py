# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hidden-pytest workspace tasks grade in the workspace image their archive specifies.

The behavior tests convert real DCAgent rows from the pinned upstream release and validate the
machines the production census produces, plus the staged verifyit entry on the grader interpreters.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, VerifyitGrader
from taskcompendium.pipeline.conversion import row_source, row_task_id
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.grading import SPEC_PATH, verifyit_spec
from taskcompendium.runtime.staging import resource_files
from taskcompendium.runtime.verifyit_runtime import verifyit_runtime_files
from verifyit.grade import VERDICT_JSON
from verifyit.spec import render_spec

from experiments.post_training.task_curation.datasets.tasktrove import python_tests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import unpack_task_binary
from experiments.post_training.task_curation.datasets.tasktrove.python_tests import (
    python_tests_grader_environment,
)
from experiments.post_training.task_curation.pipeline import source_recipe
from experiments.post_training.task_curation.tests.conversion import fixture_context

FIXTURE_ROW = Path(__file__).parent / "fixtures/dcagent_curriculum_easy_row.tar.gz"
ROW_ID = "curriculum-easy-0000"
"""One real row of DCAgent__exp_rpt_curriculum-easy @ 02923004846e4e73862c20962f823a6d05100e7a."""

STACK_FIXTURE_ROW = Path(__file__).parent / "fixtures/dcagent_stack_pytest_row.tar.gz"
STACK_ROW_ID = "stack-pytest-0005"
"""One real row of DCAgent__exp_rpt_stack-pytest-v2 at the same upstream revision."""


def stack_pytest_recipe():
    return next(
        source.config
        for source in python_tests.sources()
        if source.info.id == "TaskTrove:DCAgent__exp_rpt_stack-pytest-v2"
    )


def converted_real_row(pipeline, row_data: dict, locator: str):
    """Convert a real pinned row through the real locator so the per-config locks apply."""
    context = fixture_context(pipeline)
    decoded = unpack_task_binary(row_data, context)
    recipe = source_recipe(pipeline, {}, context.grader_environment)
    source = row_source(recipe, locator)
    row = RawRow(row_task_id(recipe, source), source, decoded)
    result = pipeline.convert(row, context)
    assert not hasattr(result, "task") or not hasattr(result, "detail"), f"rejected: {getattr(result, 'detail', result)}"
    return result.task if hasattr(result, "task") else result


def curriculum_easy_recipe():
    return next(
        source.config
        for source in python_tests.sources()
        if source.info.id == "TaskTrove:DCAgent__exp_rpt_curriculum-easy"
    )


@pytest.fixture(scope="module")
def dcagent_row() -> dict:
    return {"path": ROW_ID, "task_binary": FIXTURE_ROW.read_bytes()}


def test_real_dcagent_row_grades_in_its_workspace_image(dcagent_row):
    task = converted_real_row(curriculum_easy_recipe(), dcagent_row, "DCAgent__exp_rpt_curriculum-easy/tasks.parquet:0")
    actor = task.environment_requirements
    grader = task.grader.environment
    assert isinstance(task.grader, VerifyitGrader) and grader is not None
    # The grader machine is the actor's pinned workspace image; the source's packages lock,
    # which no Daytona grader machine can create, is gone.
    assert grader.docker_image == actor.docker_image
    assert grader.docker_image.startswith("python:3.10-slim@sha256:")
    assert grader.packages_lock is None and grader.docker_build is None
    assert grader.working_directory == actor.working_directory
    assert grader.environment_variables == actor.environment_variables
    # The archive's floating pip install is replaced by the hashed-lock install (the actor
    # lock rides as a worker resource), and the grader lock's install appends after it.
    assert actor.setup_commands == (
        "mkdir -p /app",
        python_tests.LOCK_INSTALL.format(lock=f"/{python_tests.ACTOR_LOCK_RESOURCE}"),
    )
    assert grader.setup_commands == (*actor.setup_commands, grader.setup_commands[-1])
    grader_path = python_tests.GRADER_LOCK_RESOURCE
    grader_lock = [resource for resource in task.resources.verifier if resource.path == grader_path]
    assert len(grader_lock) == 1
    worker_lock = [resource for resource in task.resources.worker if resource.path == python_tests.ACTOR_LOCK_RESOURCE]
    assert len(worker_lock) == 1
    assert "--require-hashes" in grader.setup_commands[-1] and "locks/grader.lock" in grader.setup_commands[-1]


def test_real_dcagent_row_actor_and_grader_validate_on_daytona(dcagent_row):
    """The production census validates both machines with no refused spec."""
    pytest.importorskip("daytona")  # the Daytona factory imports the provider SDK
    from experiments.post_training.task_curation.daytona_identity import count_lowered_spec_identities  # noqa: PLC0415

    task = converted_real_row(curriculum_easy_recipe(), dcagent_row, "DCAgent__exp_rpt_curriculum-easy/tasks.parquet:0")
    envelope = {
        "task": task.model_dump(mode="json"),
        "runtime": {"task_machine": {}, "verifier_machine": {}},
    }
    count = count_lowered_spec_identities([(ROW_ID, envelope)])
    assert count.failures == []
    assert count.execution_failures == []
    # Both machines run on the one shared pinned image identity.
    assert count.distinct == 1


def test_a_declared_grader_image_stays():
    context = ConversionContext(
        inputs={},
        grader_environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="grader@sha256:" + "0" * 64
        ),
    )
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="actor@sha256:" + "1" * 64
    )
    grader = python_tests_grader_environment(context, environment)
    assert grader == context.grader_environment


def test_real_stack_pytest_row_installs_its_declared_package_set():
    """The Stack Overflow grader installs the agent-image package set its tests import; its base runs 3.12."""
    locator = "DCAgent__exp_rpt_stack-pytest-v2/tasks.parquet:5"
    task = converted_real_row(
        stack_pytest_recipe(), {"path": STACK_ROW_ID, "task_binary": STACK_FIXTURE_ROW.read_bytes()}, locator
    )
    actor = task.environment_requirements
    grader = task.grader.environment
    assert grader.docker_image == actor.docker_image
    assert grader.docker_image.startswith("ubuntu:24.04@sha256:")
    assert grader.packages_lock is None
    # The stack-pytest base installs nothing through pip, so the actor setup keeps the
    # apt steps and only the grader lock applies.
    assert not any("pip install" in command for command in actor.setup_commands)
    grader_path = python_tests.GRADER_LOCK_RESOURCE
    grader_lock = [resource for resource in task.resources.verifier if resource.path == grader_path]
    assert len(grader_lock) == 1
    grader_install = [
        command for command in grader.setup_commands if "--require-hashes" in command and "locks/grader.lock" in command
    ]
    assert len(grader_install) == 1
    # The pinned install references the grader lock resource; the lock content carries the
    # Stack set (the grader lock file is staged beside the command, not echoed in it).
    assert len(grader_lock) == 1 and grader_lock[0].source.content_base64


@pytest.mark.parametrize("python", ["3.10", "3.11"])
def test_verifyit_entry_grades_a_real_row_on_python(python, dcagent_row, tmp_path):
    """The staged verifyit entry grades the real row's spec on a 3.10 and a 3.11 grader interpreter."""
    if shutil.which("uv") is None:
        pytest.skip("uv manages the grader interpreters")
    locator = "DCAgent__exp_rpt_curriculum-easy/tasks.parquet:0"
    task = converted_real_row(curriculum_easy_recipe(), dcagent_row, locator)
    grader = task.grader
    assert isinstance(grader, VerifyitGrader)
    tests = tmp_path / "tests"
    tests.mkdir()
    for staged in resource_files(task.resources.verifier, "/tests"):
        target = tests / staged.path.removeprefix("/tests/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(staged.data)
    files, runtime_sha256 = verifyit_runtime_files()
    for staged in files:
        target = tests / staged.path.removeprefix("/tests/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(staged.data)
        if staged.mode is not None:
            target.chmod(staged.mode)
    spec_path = tests / SPEC_PATH.removeprefix("/tests/")
    # The grader machine mounts the tests at /tests; the interpreter grades in this sandbox instead.
    rendered = render_spec(verifyit_spec(grader)).replace('"/tests/', f'"{tests}/')
    spec_path.write_text(rendered)
    logs = tmp_path / "logs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        python,
        *(f"--with={requirement}" for requirement in ("tomli==2.5.0", *python_tests.PYTHON_TEST_GRADERS)),
        "python",
        "-I",
        str(tests / ".taskcompendium-runtime/entry.py"),
        runtime_sha256,
        str(spec_path),
        "--logs-dir",
        str(logs),
        "--workspace",
        str(workspace),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=600)
    verdict = json.loads((logs / VERDICT_JSON).read_text())
    # An empty submission fails collection on the source's imports: a scored negative verdict.
    assert verdict["reward"] == 0.0 and verdict["status"] == "scored"
    assert verdict["detail"]["reason"] == "collection_error"
