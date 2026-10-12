# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a TaskCompendium ``TaskSpec`` from builder outputs.

These helpers own the serialized details a builder should not repeat: the grader model, resource
groups, and the capability requirements of a task machine. ``assemble`` adds the checks TaskSpec
itself does not make and returns a spec that survives a JSON round trip.

A ``TaskSpec`` holds the semantic task only. How it runs (machine backend, limits, deadlines) is
chosen when validation lowers it for RolloutEngine.
"""

from collections.abc import Sequence

from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerFormat,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskcompendium.submission import require_submission_compatibility
from verifyit.candidate import IN_PROCESS_MODES
from verifyit.spec import Spec, mode_of

SHELL_CAPABILITY = "shell"
FILESYSTEM_CAPABILITY = "filesystem"


def file(path: str, content: str, mode: int = 0o644) -> TaskResource:
    """A UTF-8 text file at ``path`` relative to the machine root.

    ``"workspace/numbers.txt"`` lands at ``/workspace/numbers.txt``; an absolute
    path fails TaskResource validation.
    """
    return TaskResource(path=path, source=inline_resource(path, content.encode()).source, mode=f"{mode:o}")


def requirements(*, workdir: str = "/workspace") -> EnvironmentRequirements:
    """A ShellSim task machine (no image) with a shell and a filesystem, working in ``workdir``.

    Agent-visible files go to ``assemble(files=)``.
    """
    return EnvironmentRequirements(
        capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY),
        command_semantics=CommandSemantics.SHELL_SIMULATOR,
        working_directory=workdir,
    )


def answer_grader(spec: Spec) -> GraderPackage:
    """A verifyit grader of the final answer, graded in process.

    Raises:
        ValueError: the mode does not grade in process (exact, numeric, mcq, math, ifeval,
            json_schema, xml_elements, csv_columns, structured_exact, predicted_action do).
    """
    mode = mode_of(spec)
    if mode not in IN_PROCESS_MODES:
        raise ValueError(f"verifyit mode {mode.value!r} does not grade in process")
    return verifyit_package(spec)


def assemble(
    task_id: str,
    instruction: str,
    answer_type: AnswerType,
    answer_format: AnswerFormat,
    grader: GraderPackage,
    source: Source,
    *,
    environment: EnvironmentRequirements | None,
    files: Sequence[TaskResource] = (),
    system: str | None = None,
    tags: Sequence[str] = (),
) -> TaskSpec:
    """Build a TaskSpec and reject one that RolloutEngine could not grade as intended.

    ``answer_format`` says how the model is asked for its final answer and how the answer is read
    from its conversation. ``environment`` is the task machine (``requirements(...)``) or ``None``
    for a task without one. ``files`` are the agent-visible files installed relative to the machine
    root; the grader's files stay private.

    Raises:
        ValueError: TaskSpec validation failed, files were given without a task machine, the task
            machine has no shell, the answer format cannot carry the answer for this grader, private
            grader content is agent-visible, or the spec does not survive a JSON round trip.
    """
    if environment is None and files:
        raise ValueError("Files require a task machine")
    if environment is not None and SHELL_CAPABILITY not in environment.capabilities:
        raise ValueError(f"A task machine needs the {SHELL_CAPABILITY!r} capability")
    system_events = () if system is None else (TextMessage(role="system", content=system),)
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(*system_events, TextMessage(role="user", content=instruction))),
        environment_requirements=environment or EnvironmentRequirements(),
        answer_type=answer_type,
        answer_format=answer_format,
        grader=grader.grader,
        source=source,
        resources=ResourceGroups(worker=tuple(files), verifier=grader.resources),
        tags=tuple(tags),
    )
    if spec.answer_type in CONVERSATION_ANSWERS:
        # As the Shellbox session does at start: only an answer read from the conversation has a format to check.
        require_submission_compatibility(spec)
    visible = {resource_bytes(item) for item in spec.resources.all + spec.resources.worker}
    leaked = sorted(item.path for item in spec.resources.verifier if resource_bytes(item) in visible)
    if leaked:
        raise ValueError(f"Private verifier file content is also agent-visible: {leaked}")
    restored = TaskSpec.model_validate_json(spec.model_dump_json())
    if restored != spec:
        raise ValueError(f"Task {task_id!r} does not survive a JSON round trip")
    return restored
