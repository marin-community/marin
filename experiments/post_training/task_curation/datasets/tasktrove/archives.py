# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove sources: archived Harbor tasks from the original ``open-thoughts/TaskTrove`` release."""

import tomllib
from dataclasses import dataclass, replace

from taskcompendium.convert.tasks import shell_environment
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    PlainText,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Converter, ImportFailureKind, ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    TASK_TOML,
    TASKS_FILE,
    TASKTROVE_REPO,
    archive_file,
    archive_resource,
    archive_resources,
    archive_script_grader,
    unpack_task_binary,
)
from experiments.post_training.task_curation.datasets.tasktrove.source_defects import SOURCE_DEFECTS
from experiments.post_training.task_curation.environment_fidelity import (
    UnlowerableRecipe,
    archive_environment,
    environment_unused,
    missing_environment,
)
from experiments.post_training.task_curation.pipeline import HfSource

TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
ANSWER_FILE_DELIVERY = (
    ("write your final answer to `/app/answer.txt`", "return your final answer in the assistant response"),
    ("Write ONLY your final answer to **`/app/answer.txt`**", "Return ONLY your final answer in the assistant response"),
    ("The verifier reads that file", "The verifier reads the assistant response"),
)
"""The answer-file wording of the Nemotron Gym puzzle archives, and its reply-based replacement."""


@dataclass(frozen=True)
class TaskTroveConverter:
    """Retain reviewed source defects as explicit conversion outcomes."""

    config: str
    convert: Converter

    def __call__(self, row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
        reason = SOURCE_DEFECTS.get((self.config, row.data["path"]))
        if reason is not None:
            return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason="reviewed_defect", detail=reason)
        converted = self.convert(row, context)
        if isinstance(converted, ImportRejection):
            return converted
        task = converted.task if isinstance(converted, NormalizedTask) else converted
        # Reply-only conversions have no agent machine. Keep the archive as oracle
        # evidence, while enforcing environment fidelity for tasks that use a machine.
        reply_only = (
            task.answer_type in CONVERSATION_ANSWERS and task.environment_requirements == EnvironmentRequirements()
        )
        spec = task.model_dump(mode="json")
        missing = missing_environment(spec)
        if missing is not None:
            return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason="missing_environment", detail=missing)
        reused = None if reply_only else environment_unused(spec, row.data["files"])
        if reused is not None:
            return ImportRejection(
                kind=ImportFailureKind.CONVERTER_ERROR, reason="environment_substituted", detail=reused
            )
        existing = {resource.path for resource in (*task.resources.all, *task.resources.oracle)}
        # Oracle controls may upload these files through runtimes without timestamp support.
        oracle = task.resources.oracle + tuple(
            archive_resource(row.data, path).model_copy(update={"mtime_ns": None})
            for path in row.data["files"]
            if path not in existing and not path.startswith(("tests/", "setup_files/"))
        )
        task = task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": oracle})})
        return replace(converted, task=task) if isinstance(converted, NormalizedTask) else task


def native_file_task(
    row: RawRow, _context: ConversionContext, *, output_paths: tuple[str, ...]
) -> TaskSpec | ImportRejection:
    """Retain a file task's native instructions, verifier and environment recipe."""
    task_toml = archive_file(row.data, TASK_TOML)
    if task_toml is None:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="missing_task_config", detail="Archive declares no task.toml"
        )
    config = tomllib.loads(task_toml.decode())
    native_environment = config.get("environment") or {}
    agent = config.get("agent") or {}
    verifier = config.get("verifier") or {}
    if (
        native_environment.get("docker_image")
        or native_environment.get("setup")
        or agent.get("env")
        or agent.get("user")
        or verifier.get("environment")
    ):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="native_contract_requires_harbor",
            detail="Native bootstrap, user, agent variables or separate environments require Harbor import",
        )
    try:
        environment = archive_environment(row.data["files"])
    except UnlowerableRecipe:
        environment = EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=DockerBuildContext(
                files=tuple(
                    archive_resource(row.data, path).model_copy(update={"path": path.removeprefix("environment/")})
                    for path in row.data["files"]
                    if path.startswith("environment/")
                )
            ),
        )
    environment = shell_environment(environment)
    grader = archive_script_grader(
        row.data, required=(), environment=environment, answer_path=None, env=verifier.get("env") or {}
    )
    if isinstance(grader, ImportRejection):
        return grader
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=row.data["instruction"]),)),
        environment_requirements=environment,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        output_paths=output_paths,
        grader=grader,
        resources=archive_resources(row.data),
    )


def tasktrove_source(config: str) -> HfSource:
    """One TaskTrove config's task archives, unpacked into instruction, files and verifier data."""
    return HfSource(
        TASKTROVE_REPO, TASKTROVE_REVISION, (f"{config}/{TASKS_FILE}",), SourceFormat.PARQUET, decode=unpack_task_binary
    )
