# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove archive conversion types converter failures and keeps the archive's oracle."""

import base64
import json
import sys
from dataclasses import replace

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, PackageSpec
from taskcompendium.harbor.execution import native_execution_limits, task_with_declared_execution_requirements
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.conversion import convert_row as convert_mechanical_row
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, NormalizedTask, RawRow
from taskcompendium.pipeline.review import review_payload
from taskcompendium.runtime.shell import upload_resources
from verifyit.grade import Status
from verifyit.modes import grade_pytest
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.archives import TaskTroveConverter
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    INSTRUCTION,
    SOLVE_SH,
    archive_resource,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.code_contests import convert_code_contests
from experiments.post_training.task_curation.datasets.tasktrove.conversion.nl2bash import convert_nl2bash
from experiments.post_training.task_curation.datasets.tasktrove.conversion.python_unit_tests import (
    convert as convert_python,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    archive_conversion,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.taco import convert_taco
from experiments.post_training.task_curation.datasets.tasktrove.nl2bash import convert_nl2bash_task

DOCKERFILE = b"FROM python:3.12-slim\n"
SUM_PROMPT = b"Read two integers and print their sum to stdout in /app/solution.py."
ARCHIVE_ORACLE = b"#!/bin/bash\ncp /solution/solution.py /app/solution.py\n"
SUM_SOLUTION = b"a, b = map(int, input().split())\nprint(a + b)\n"


def archive_data(files: dict[str, bytes]) -> dict:
    """An archive row as ``unpack_task_binary`` decodes it."""
    files = {INSTRUCTION: SUM_PROMPT, "environment/Dockerfile": DOCKERFILE, **files}
    return {
        "instruction": files[INSTRUCTION].decode(),
        "files": {path: base64.b64encode(data).decode() for path, data in files.items()},
    }


def verifier_data(value: dict) -> dict[str, bytes]:
    return {"tests/verifier_data.json": json.dumps(value).encode()}


@pytest.mark.parametrize(
    "convert, files, kind, reason",
    [
        (convert_nl2bash, verifier_data({"expected_output": None}), ImportFailureKind.SOURCE_DEFECT, "null_grader"),
        (convert_nl2bash, {}, ImportFailureKind.CONVERTER_ERROR, "converter_error"),
        (
            convert_nemotron_structured_outputs,
            verifier_data({"schema": {"type": "object"}, "schema_type": "ini"}),
            ImportFailureKind.UNSUPPORTED,
            "unsupported_variant",
        ),
        (
            convert_code_contests,
            {
                INSTRUCTION: b"Read two integers, such as 3 4, and print their sum.",
                "tests/test_data.json": json.dumps({"inputs": ["3 4"], "outputs": ["7"]}).encode(),
            },
            ImportFailureKind.SOURCE_DEFECT,
            "gold_in_instruction",
        ),
    ],
    ids=["missing_expected_output", "missing_verifier_data", "unknown_schema_type", "samples_only"],
)
def test_converter_failures_become_typed_rejections(convert, files, kind, reason):
    rejection = archive_conversion(archive_data(files), convert)
    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (kind, reason)
    assert rejection.detail


def test_archive_solution_files_are_the_oracle_when_the_converter_supplies_none():
    files = {
        "tests/test_data.json": json.dumps({"inputs": ["10 20\n"], "outputs": ["30\n"]}).encode(),
        SOLVE_SH: ARCHIVE_ORACLE,
        "solution/solution.py": SUM_SOLUTION,
    }
    converted = archive_conversion(archive_data(files), convert_code_contests)
    assert isinstance(converted, ConvertedTask)
    assert converted.solution_files == {SOLVE_SH: ARCHIVE_ORACLE, "solution/solution.py": SUM_SOLUTION}


def test_converter_oracle_replaces_the_archive_oracle():
    files = {
        "tests/inputs/input_0.txt": b"10 20\n",
        "tests/outputs/output_0.txt": b"30\n",
        SOLVE_SH: b"#!/bin/bash\nexit 1\n",
        "solution/solution.py": SUM_SOLUTION,
    }
    converted = archive_conversion(archive_data(files), convert_taco)
    assert isinstance(converted, ConvertedTask)
    assert converted.solution_files[SOLVE_SH] != files[SOLVE_SH]
    assert converted.solution_files["solution/solution.py"] == SUM_SOLUTION


@pytest.mark.parametrize(
    "module, location",
    [
        ("calc", "calc.py"),
        ("arithmetic.calc", "/app/arithmetic/calc.py"),
        ("app.practical", "/app/practical.py"),
        ("calculator", "calculator.py"),
        ("app.calculator", "calculator.py"),
    ],
)
def test_python_archive_missing_public_module_scores_zero_then_implementation_passes(tmp_path, module, location):
    tests = f"from {module} import add\ndef test_add():\n    assert add(2, 3) == 5\n".encode()
    converted = archive_conversion(
        archive_data(
            {INSTRUCTION: f"Implement add(a, b) in {location} under /app.".encode(), "tests/test_solution.py": tests}
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    workspace = tmp_path / "app"
    workspace.mkdir()
    test = tmp_path / "test_solution.py"
    test.write_bytes(converted.data_files["tests/test_solution.py"])
    spec = replace(converted.spec, paths=(str(test),), python=sys.executable)
    empty = grade_pytest.grade(spec, tmp_path, workspace)
    assert (empty.status, empty.reward) == (Status.SCORED, 0.0)
    implementation = workspace.joinpath(*module.removeprefix("app.").split(".")).with_suffix(".py")
    implementation.parent.mkdir(parents=True, exist_ok=True)
    implementation.write_text("def add(a, b):\n    return a + b\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)


def test_python_archive_app_export_failure_scores_zero_then_package_passes(tmp_path):
    converted = archive_conversion(
        archive_data(
            {
                INSTRUCTION: b"Implement create_app in /app/app.py, or re-export it from /app/__init__.py.",
                "tests/test_solution.py": (
                    b"from app import create_app\ndef test_app():\n    assert create_app() == 'ready'\n"
                ),
            }
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    assert {"app", "app.app"}.issubset(converted.spec.candidate_modules)

    workspace = tmp_path / "app"
    workspace.mkdir()
    test = tmp_path / "test_solution.py"
    test.write_bytes(converted.data_files["tests/test_solution.py"])
    spec = replace(converted.spec, paths=(str(test),), python=sys.executable)
    missing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (missing.status, missing.reward) == (Status.SCORED, 0.0)

    (workspace / "__init__.py").write_text("def create_app():\n    return 'ready'\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)

    (workspace / "__init__.py").unlink()
    (workspace / "app.py").write_text("def create_app():\n    return 'ready'\n")
    passing = grade_pytest.grade(spec, tmp_path, workspace)
    assert (passing.status, passing.reward) == (Status.SCORED, 1.0)


def test_python_archive_does_not_infer_unstated_module_names():
    converted = archive_conversion(
        archive_data(
            {
                INSTRUCTION: b"Implement a Calculator in the /app directory.",
                "tests/test_solution.py": (
                    b"from calculator import Calculator\ndef test_add():\n    assert Calculator().add(2, 3) == 5\n"
                ),
            }
        ),
        convert_python,
    )
    assert isinstance(converted, ConvertedTask)
    assert isinstance(converted.spec, PytestSpec)
    assert "calculator" not in converted.spec.candidate_modules
    assert "app.calculator" not in converted.spec.candidate_modules


def test_a_conversion_lowers_the_archive_environment():
    """The task runs in its own archived recipe on a pinned base, not a shared image."""
    data = archive_data(verifier_data({"expected_output": "7\n"}))
    row = RawRow(id="row", source=Source(dataset="d", revision="r", row="row", importer_revision="1"), data=data)
    context = ConversionContext(
        inputs={},
        grader_environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="file:///lock"
        ),
    )
    converted = convert_nl2bash_task(row, context)
    assert isinstance(converted, NormalizedTask)
    requirements = converted.task.environment_requirements
    assert requirements.docker_image.startswith("python:3.12-slim@sha256:")
    assert requirements.working_directory == "/"  # the fixture recipe declares no WORKDIR
    # The fixture installs nothing; a recipe's RUN steps become setup commands, covered
    # by the lowering tests in test_environment_fidelity.
    assert requirements.setup_commands == ()


@pytest.mark.parametrize("entrypoint", ["archive", "mechanical"])
def test_archive_converter_rejects_an_agentic_task_without_a_declared_environment(entrypoint):
    source = Source(dataset="fixture", revision="1", row="missing", importer_revision="1")
    row = RawRow("missing", source, {"path": "missing", "files": {}})
    task = TaskSpec(
        id=row.id,
        source=source,
        context=ConversationInput(events=(TextMessage(role="user", content="Edit the workspace."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
    )

    def convert(_row, _context):
        return task

    context = ConversionContext({}, None)
    result = (
        TaskTroveConverter("fixture", convert)(row, context)
        if entrypoint == "archive"
        else convert_mechanical_row(row, convert, context)
    )
    assert isinstance(result, ImportRejection)
    assert result.reason == "missing_environment"


@pytest.mark.asyncio
async def test_archive_oracle_can_stage_existing_and_appended_resources_without_changing_bytes(tmp_path):
    data = {
        "path": "oracle-timestamps",
        "files": {
            SOLVE_SH: base64.b64encode(ARCHIVE_ORACLE).decode(),
            INSTRUCTION: base64.b64encode(SUM_PROMPT).decode(),
        },
        "file_metadata": {
            SOLVE_SH: {"mode": "0751", "mtime_ns": 123456789},
            INSTRUCTION: {"mode": "0640", "mtime_ns": 987654321},
        },
    }
    source = Source(dataset="fixture", revision="1", row="oracle-timestamps", importer_revision="1")
    row = RawRow("oracle-timestamps", source, data)
    task = TaskSpec(
        id=row.id,
        source=source,
        context=ConversationInput(events=(TextMessage(role="user", content="Reply with the result."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        resources=ResourceGroups(oracle=(archive_resource(data, SOLVE_SH),)),
    )

    def convert(_row, _context):
        return task

    converted = TaskTroveConverter("fixture", convert)(row, ConversionContext({}, None))
    assert isinstance(converted, TaskSpec)
    machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
    try:
        await upload_resources(machine, converted.resources.oracle, 10, user=None)
        for path, expected in ((SOLVE_SH, ARCHIVE_ORACLE), (INSTRUCTION, SUM_PROMPT)):
            target = tmp_path / path
            await machine.download("/" + path, target)
            assert target.read_bytes() == expected
    finally:
        await machine.close()
    assert {resource.path: resource.mode for resource in converted.resources.oracle} == {
        SOLVE_SH: "0751",
        INSTRUCTION: "0640",
    }


@pytest.mark.parametrize("complete", [False, True])
def test_archive_conversion_lifts_only_complete_original_limits(complete):
    native = b"[environment]\ncpus = 2\nmemory_mb = 4096\nstorage_mb = 8192\n[agent]\ntimeout_sec = 900\n"
    if complete:
        native += b"[verifier]\ntimeout_sec = 60\n"
    data = archive_data({"task.toml": native})
    data["path"] = "source-limits"
    data["file_metadata"] = {path: {"mode": "0644", "mtime_ns": 123} for path in data["files"]}
    source = Source(dataset="fixture", revision="pinned", row=data["path"], importer_revision="1")
    task = TaskSpec(
        id="source-limits",
        source=source,
        context=ConversationInput(events=(TextMessage(role="user", content="Reply."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
    )

    def convert(_row, _context):
        return task

    converted = TaskTroveConverter("fixture", convert)(RawRow(task.id, source, data), ConversionContext({}, None))
    assert isinstance(converted, TaskSpec)
    original = converted.model_copy(update={"execution_requirements": None})
    assert review_payload(converted) == review_payload(original)
    assert converted.source == task.source
    assert converted.id == task.id
    assert (converted.execution_requirements is not None) == complete
    if complete:
        assert converted.execution_requirements.model_dump() == {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 8192,
            "agent_timeout_sec": 900,
            "verifier_timeout_sec": 60,
        }
        with pytest.raises(ValueError, match="cpus conflicts"):
            task_with_declared_execution_requirements(converted, native_execution_limits({"environment": {"cpus": 3}}))
    assert (
        next(resource for resource in converted.resources.oracle if resource.path == "task.toml").source.content_base64
        == base64.b64encode(native).decode()
    )
