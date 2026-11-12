# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell tasks graded on the files an agent leaves in a pinned image.

The agent works in a shell in its own image, with public setup files mounted at their archive
paths. A verifyit workspace mode (``stdio``, ``pytest``, ``script``) grades the captured output
files in a fresh machine of the grader image, with the hidden tests under ``/tests``. Oracle files
stay with the oracle role, which only grader controls mount.
"""

import hashlib
import json
import re
from dataclasses import replace

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.tasks import workspace_task
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
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
            command_semantics=CommandSemantics.LINUX_PROCESS,
            capabilities=REPOSITORY_CAPABILITIES,
            tool_providers={
                "shell": ProviderRequirement(
                    action_interface="shell:v1",
                    initial_state={"repository": repository, "source_ref": checkout[1], "workspace": workspace},
                )
            },
        ),
        resources=archive_resources(row.data),
        answer_type=AnswerType.STATE,
        answer_format=PlainText(),
        grader=grader,
    )
