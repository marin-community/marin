# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell tasks graded on the files an agent leaves in a pinned image.

The agent works in a shell in its own image, with public setup files mounted at their archive
paths. A verifyit workspace mode (``stdio``, ``pytest``, ``script``) grades the captured output
files in a fresh machine of the grader image, with the hidden tests under ``/tests``. Oracle files
stay with the oracle role, which only grader controls mount.
"""

import ast
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.executable import workspace_task
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ProviderRequirement,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    OracleCommand,
    RawRow,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import BASH, INTERFACE
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    SOLVE_SH,
    archive_files,
    archive_resources,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import (
    ConvertedTask,
    ConvertFn,
    archive_conversion,
)

SOLUTION_PATHS = ("/app/solution.py", "/app/solution.cpp")
"""The program files a competitive-programming grader runs, whichever language the agent chose."""
ORACLE_COMMAND = f"bash /{SOLVE_SH}"

PYTHON_FILE = re.compile(r"(?<![\w/])(?:/app/|app/)?[A-Za-z_]\w*(?:/[A-Za-z_]\w*)*\.py\b")
PACKAGE = re.compile(r"(?:package (?:at|under)|package[^\n]{0,30} at) /app/([A-Za-z_]\w*)")

REPOSITORY_CAPABILITIES = ("shell", "filesystem", "git_repository")
REPOSITORY_FILES = ("tests/config.json", "tests/test.sh", "environment/Dockerfile")
CHECKOUT = re.compile(r"\bgit checkout\s+([^\s;&]+)")
REPOSITORY_GRADER = "source repository patch grader"
REPOSITORY_REQUIREMENTS = ("Isolated repository checkout and patch capture",)


def converted_workspace_task(
    row: RawRow,
    converted: ConvertedTask,
    *,
    instruction: str,
    environment: EnvironmentRequirements,
    grader_environment: EnvironmentRequirements,
    output_paths: tuple[str, ...],
) -> TaskSpec:
    """A workspace task from a TaskTrove converter's result.

    ``tests/`` files are hidden tests, except ``tests/setup_files/``, which seed the oracle; other
    data files are public. The converter's solution files are the oracle.
    """
    spec = converted.spec
    if isinstance(spec, PytestSpec):
        # The grader image owns its Python and report dependencies; the source Dockerfile's
        # virtualenv does not exist there.
        spec = replace(spec, python="python3")
    verifier, worker, oracle = [], [], []
    for path, data in converted.data_files.items():
        if path.startswith("tests/setup_files/"):
            oracle.append(inline_resource(path, data))
        elif path.startswith("tests/"):
            verifier.append(inline_resource(path.removeprefix("tests/"), data))
        else:
            worker.append(inline_resource(path, data))
    oracle.extend(inline_resource(path, data) for path, data in converted.solution_files.items())
    tags = (*converted.tags, f"language:{converted.language}") if converted.language else converted.tags
    return workspace_task(
        row,
        instruction=instruction,
        spec=spec,
        environment=environment,
        grader_environment=grader_environment,
        output_paths=output_paths,
        verifier=tuple(verifier),
        worker=tuple(worker),
        oracle=tuple(oracle),
        tags=tags,
    )


def _converter_changes(row: RawRow, converted: ConvertedTask) -> tuple[NormalizationChange, ...]:
    original = row.data["instruction"]
    if converted.instruction == original:
        return ()
    return (
        NormalizationChange(
            field="instruction",
            reason="The source converter corrected delivery boilerplate",
            original=original,
            replacement=converted.instruction,
        ),
    )


def tasktrove_archive_task(
    row: RawRow,
    *,
    convert: ConvertFn,
    environment: EnvironmentRequirements,
    grader_environment: EnvironmentRequirements,
    output_paths: tuple[str, ...],
) -> NormalizedTask | ImportRejection:
    """Convert an unpacked TaskTrove archive with ``convert`` into a workspace task."""
    converted = archive_conversion(row.data, convert)
    if isinstance(converted, ImportRejection):
        return converted
    task = converted_workspace_task(
        row,
        converted,
        instruction=converted.instruction,
        environment=environment,
        grader_environment=grader_environment,
        output_paths=output_paths,
    )
    return NormalizedTask(task, _converter_changes(row, converted))


def submission_paths(instruction: str) -> tuple[str, ...]:
    """The Python files the public instruction asks the agent to write, under ``/app``."""
    packages = set(PACKAGE.findall(instruction))
    paths = set()
    for match in PYTHON_FILE.finditer(instruction):
        filename = match.group()
        public_workspace_path = filename.startswith("/app/")
        filename = filename.removeprefix("/app/") if filename.startswith("/app/") else filename.removeprefix("app/")
        if not public_workspace_path and (filename.startswith("test_") or filename.startswith("tests/")):
            continue
        if "/" not in filename and len(packages) == 1 and filename.removesuffix(".py") not in packages:
            filename = f"{next(iter(packages))}/{filename}"
        paths.add(f"/app/{filename}")
    return tuple(sorted(paths))


@dataclass(frozen=True)
class Delivery:
    """The instruction and output files of a Python task, with the repairs that produced them."""

    instruction: str
    output_paths: tuple[str, ...]
    changes: tuple[NormalizationChange, ...]


def python_delivery(instruction: str, data_files: Mapping[str, bytes]) -> Delivery | ImportRejection:
    """Capture the Python files the instruction names.

    When it names none and every hidden test imports from one module whose imported names the
    instruction mentions, the instruction gains a delivery line naming that module's file.
    """
    paths = submission_paths(instruction)
    changes = []
    if not paths:
        tests = [
            data.decode() for path, data in data_files.items() if path.startswith("tests/") and path.endswith(".py")
        ]
        imports = [
            node
            for text in tests
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] not in sys.stdlib_module_names | {"pytest", "numpy"}
        ]
        modules = {node.module for node in imports}
        if len(modules) == 1 and all(alias.name in instruction for node in imports for alias in node.names):
            module = next(iter(modules))
            assert module is not None
            paths = (f"/app/{module.replace('.', '/')}.py",)
            replacement = instruction + f"\n\nDelivery: write the requested implementation to `{paths[0]}`.\n"
            changes.append(
                NormalizationChange(
                    field="instruction",
                    reason="Make the grader's module filename explicit without changing the requested API",
                    original=instruction,
                    replacement=replacement,
                )
            )
            instruction = replacement
    if not paths:
        return unsupported(
            "unsupported_public_output_contract",
            "No explicit public Python filename and hidden test imports require APIs absent from the request",
        )
    changes.append(
        NormalizationChange(
            field="output_paths",
            reason="Capture the Python filenames declared in the public instruction",
            original=json.dumps(SOLUTION_PATHS),
            replacement=json.dumps(paths),
        )
    )
    return Delivery(instruction, paths, tuple(changes))


def tasktrove_python_task(
    row: RawRow, *, convert: ConvertFn, environment: EnvironmentRequirements, grader_environment: EnvironmentRequirements
) -> NormalizedTask | ImportRejection:
    """Convert a TaskTrove archive into a workspace task capturing the Python files its instruction names."""
    converted = archive_conversion(row.data, convert)
    if isinstance(converted, ImportRejection):
        return converted
    delivery = python_delivery(converted.instruction, converted.data_files)
    if isinstance(delivery, ImportRejection):
        return delivery
    task = converted_workspace_task(
        row,
        converted,
        instruction=delivery.instruction,
        environment=environment,
        grader_environment=grader_environment,
        output_paths=delivery.output_paths,
    )
    return NormalizedTask(task, (*_converter_changes(row, converted), *delivery.changes))


def solve_script(task: TaskSpec) -> OracleCommand | None:
    """Run the source's ``solution/solve.sh`` oracle, when the task has one."""
    if not any(resource.path == SOLVE_SH for resource in task.resources.oracle):
        return None
    return OracleCommand(ORACLE_COMMAND)


def swe_task(row: RawRow, *, workspace: str) -> TaskSpec | ImportRejection:
    """A repository repair task whose source patch grader cannot run here.

    The archive's ``tests/config.json`` names the repository and its FAIL_TO_PASS and PASS_TO_PASS
    tests, and the instruction names the checkout. The task keeps the source grader's files and
    terms for review under a ``NoGrader``: grading needs the source's per-task repository image.
    """
    instruction = row.data.get("instruction")
    encoded = row.data.get("files")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(encoded, dict):
        return unsupported("missing_repository_input", "Public request and source files are required")
    if any(path not in encoded for path in REPOSITORY_FILES):
        return unsupported("missing_repository_contract", "config.json, test.sh, and Dockerfile required")
    files = archive_files(row.data).files
    config = json.loads(files["tests/config.json"])
    repository = config.get("repo")
    checkout = CHECKOUT.search(instruction)
    if not isinstance(repository, str) or not repository.strip() or checkout is None:
        return unsupported("missing_repository_ref", "Source repository and public checkout ref required")
    if not config.get("FAIL_TO_PASS") and not config.get("PASS_TO_PASS"):
        return unsupported("missing_repository_tests", "No source FAIL_TO_PASS or PASS_TO_PASS test IDs")
    grader = NoGrader(
        reason=f"Source evaluator {REPOSITORY_GRADER} requires: {'; '.join(REPOSITORY_REQUIREMENTS)}",
        contract={
            "evaluator": REPOSITORY_GRADER,
            "source_revision": row.source.revision,
            "contract": {
                "repository": repository,
                "source_ref": checkout[1],
                "workspace": workspace,
                "source_config": config,
                "source_grader_paths": sorted("/" + path for path in files if path.startswith("tests/")),
                "source_environment_sha256": hashlib.sha256(files["environment/Dockerfile"]).hexdigest(),
            },
            "runtime_requirements": list(REPOSITORY_REQUIREMENTS),
        },
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=REPOSITORY_CAPABILITIES,
            tool_providers={
                "shell": ProviderRequirement(
                    action_interface=INTERFACE,
                    initial_state={"repository": repository, "source_ref": checkout[1], "workspace": workspace},
                )
            },
        ),
        interaction_tools=(BASH,),
        resources=archive_resources(row.data),
        answer_type=AnswerType.STATE,
        answer_format=PlainText(),
        grader=grader,
    )
