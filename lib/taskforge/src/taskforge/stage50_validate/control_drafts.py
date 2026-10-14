# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the fixed controls of a draft round from its task and its grader's stored output.

``write_controls`` asks the model for the controls the proposal's "Grader design and controls"
section calls for, so their ids and expectations follow the proposal. It sends a submission back
when ``validate_controls`` refuses the set, when a control's reward bounds cross the grader's
declared rejection ceiling (``check_expectations``: the most a submission wrong in substance can
earn under the grader's partial-credit scheme), or when a control repeats a candidate the grader
step graded while it prototyped the grader (``check_controls_not_graded``): controls are written,
not fitted to the grader. The reference answer and the empty answer are exempt, since the grader step
must grade both, and they are also the usual reference and malformed controls. The grader comes
from the draft's ``grader`` step output (``stage40_build_with_program.run.step_output``), so the reference
answer validation replays is the one the build checked.

The request lists the task's output paths, the only files a control may write, since the grader
reads nothing else from the candidate's machine. A control writes them at the absolute paths the
``TaskSpec`` lists, wherever they lie on the machine. ``check_control_files`` sends back a control
writing any other file, such as one of the task's own files or the build's reference files.

The accepted controls go to ``<directory>/controls.json`` (``spec.controls.controls_json``), and
``controls.replay`` replays them, each as a trial named by its id with a pass or fail verdict.
"""

import base64
import shlex
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator
from rigging.filesystem.atomic import atomic_rename
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskResource, TaskSpec, TextMessage, format_conversation, normalized_absolute_path
from taskcompendium.runtime.resources import resource_bytes

from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.spec.controls import (
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    Workspace,
    check_expectations,
    controls_json,
    parse_controls,
    reply,
    shell_turn,
    validate_controls,
)
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage40_build_with_program.context import BuildFailure
from taskforge.stage40_build_with_program.outputs import GradedCandidate, Grader
from taskforge.stage40_build_with_program.sdk import SUBMISSION, checked_structured, files_text, task_messages

CONTROLS_FILE = "controls.json"
CONTROLS_STEP = "controls"
"""The ``BuildFailure.step`` of a control set the model never got right."""
CONTROL_AUTHOR = "validate.write_controls"


class ControlFile(BaseModel):
    path: str = Field(
        description="Absolute path on the candidate's machine, spelled as in the task's output paths: one of "
        "those paths, or a file under an output path that is a directory."
    )
    content: str = Field(description="Complete UTF-8 file content.")

    @field_validator("path")
    @classmethod
    def absolute(cls, path: str) -> str:
        if not PurePosixPath(path).is_absolute():
            raise ValueError(
                f"the control file path {path!r} is relative; control files are the task's output paths, "
                "spelled as absolute paths exactly as the request lists them"
            )
        return normalized_absolute_path(path)


class ControlDraft(BaseModel):
    id: str = Field(description="Short id: letters, digits, '.', '_' or '-'.")
    kind: ControlKind = Field(
        description="positive: a correct candidate the grader must accept, at least reward_min; negative: a wrong, "
        "shortcut or reward-hacking candidate it must reject, at most reward_max; malformed: an empty or "
        "unparseable reply it must reject, at most reward_max."
    )
    category: ControlCategory = Field(
        description="positive: known_correct; negative: plausible_wrong, task_specific_shortcut or reward_hack; "
        "malformed: empty_or_malformed."
    )
    concern: ControlConcern = Field(
        description="The part of the grader the control exercises. known_correct: reference (the reference "
        "solution), acceptance (another answer that must count as right) or extraction; plausible_wrong: "
        "acceptance or extraction; task_specific_shortcut and reward_hack: shortcut or extraction; "
        "empty_or_malformed: extraction or acceptance."
    )
    final_reply: str = Field(description="The candidate's final reply.")
    files: list[ControlFile] = Field(
        description="Files the candidate writes before replying, at the task's output paths as listed. No other "
        "path may appear here, including the task's own files and the build's reference files; for a task "
        "without output paths, leave the list empty."
    )
    reward_min: float | None = Field(description="Positive controls: the minimum reward, e.g. 0.99. Otherwise null.")
    reward_max: float | None = Field(
        description="Negative and malformed controls: the maximum reward, at most the grader's rejection ceiling "
        "the request states. Otherwise null."
    )
    rationale: str = Field(description="Why this candidate must receive that grade.")


class ControlsDraft(BaseModel):
    """Submit the fixed controls for the task."""

    controls: list[ControlDraft]


def control_payload(draft: ControlDraft) -> Transcript:
    """The control as a transcript: one shell turn writing its files, if it has any, then its final reply."""
    if not draft.files:
        return Transcript(turns=(reply(draft.final_reply),))
    commands = tuple(
        (
            f"{draft.id}-write-{index}",
            f"mkdir -p {shlex.quote(str(PurePosixPath(f.path).parent))} && "
            f"printf %s {shlex.quote(base64.b64encode(f.content.encode()).decode())} "
            f"| base64 -d > {shlex.quote(f.path)}",
        )
        for index, f in enumerate(draft.files)
    )
    return Transcript(turns=(shell_turn(*commands), reply(draft.final_reply)))


def control_from_draft(draft: ControlDraft) -> Control:
    """The ``Control`` a submitted draft describes, expecting a graded attempt within its reward bounds."""
    return Control(
        id=draft.id,
        kind=draft.kind,
        category=draft.category,
        concern=draft.concern,
        author=CONTROL_AUTHOR,
        payload=control_payload(draft),
        expect=Expectation(status=Outcome.GRADED, reward_min=draft.reward_min, reward_max=draft.reward_max),
    )


def file_set(files: Sequence[TaskResource]) -> frozenset[tuple[str, bytes]]:
    """``files`` as (path, content) pairs, so two file lists compare by what they install."""
    return frozenset((resource.path, resource_bytes(resource)) for resource in files)


def _is_graded(control: Control, graded: Sequence[GradedCandidate], exempt_replies: frozenset[str]) -> bool:
    """Whether ``control`` is one of the ``graded`` candidates.

    A workspace control matches a candidate with the same files. A reply-only transcript matches a
    candidate with the same reply and no files. A transcript with shell calls writes files no
    candidate records, so it matches on its final reply alone, unless that reply is exempt.
    """
    if isinstance(control.payload, Workspace):
        files = file_set(control.payload.files)
        return bool(files) and any(file_set(c.files) == files for c in graded)
    final = control.payload.turns[-1]
    if not isinstance(final, TextMessage):
        return False
    if control.payload.calls():
        return final.content not in exempt_replies and any(c.reply == final.content for c in graded)
    return any(c.reply == final.content and not c.files for c in graded)


def check_control_files(drafts: Sequence[ControlDraft], output_paths: Sequence[str]) -> None:
    """Refuse a control that writes a file outside the task's ``output_paths``.

    The grader reads a candidate's files only at its output paths, so a control writing any other
    file tests nothing the grader sees. A control file counts as captured when its absolute path is
    an output path or lies under one.

    Raises:
        ValueError: naming each such control and its files.
    """
    roots = [PurePosixPath(path) for path in output_paths]

    def captured(file: ControlFile) -> bool:
        return any(PurePosixPath(file.path).is_relative_to(root) for root in roots)

    outside = {draft.id: paths for draft in drafts if (paths := [f.path for f in draft.files if not captured(f)])}
    if outside:
        allowed = ", ".join(output_paths) if output_paths else "none, so every control is a final reply alone"
        raise ValueError(
            f"controls {sorted(outside)} write files outside the task's output paths: {outside}. Control files are "
            f"only the task's output paths ({allowed})"
        )


def check_controls_not_graded(controls: Sequence[Control], grader: Grader) -> None:
    """Refuse a control that repeats a candidate in ``grader.prototyped``, other than the exempt two.

    Raises:
        ValueError: naming the controls that repeat a prototyped candidate.
    """
    exempt = {("", file_set(())), (grader.reference_reply, file_set(grader.reference_files))}
    candidates = [c for c in grader.prototyped if (c.reply, file_set(c.files)) not in exempt]
    exempt_replies = frozenset({"", grader.reference_reply})
    repeated = [control.id for control in controls if _is_graded(control, candidates, exempt_replies)]
    if repeated:
        raise ValueError(
            f"controls {repeated} are candidates the grader step graded while it prototyped the grader. Controls "
            "are written, not graded: write candidates the grader has not seen"
        )


def output_paths_text(output_paths: Sequence[str], shipped: Sequence[str]) -> str:
    """The request's statement of the files a control may write.

    ``shipped`` names the output paths the task's own files already occupy in the machine; a control
    edits those in place rather than writing them again from memory.
    """
    if not output_paths:
        return "The task has no output paths, so every control is a final reply alone with no files."
    text = (
        "A control writes files only at these output paths, the files the grader reads from the candidate's "
        f"machine, with each path absolute as listed: {', '.join(output_paths)}."
    )
    if shipped:
        text += (
            f" Of these, the task ships {', '.join(shipped)} in the machine already: a control changes such a "
            "file where the task's solution would, and leaves the rest of it as shipped."
        )
    return text


def ceiling_text(rejection_ceiling: float) -> str:
    """The request's statement of the grader's rejection ceiling."""
    if rejection_ceiling == 0:
        return (
            "The grader gives no partial credit: a submission wrong in substance scores 0, so negative and "
            "malformed controls demand reward_max 0."
        )
    return (
        f"The grader's partial-credit scheme lets a submission wrong in substance score up to {rejection_ceiling}, "
        f"so negative and malformed controls demand reward_max at most {rejection_ceiling}, and positive "
        f"controls demand more."
    )


def shipped_output_paths(task: TaskSpec) -> tuple[str, ...]:
    """The task's output paths that its agent-visible files already occupy."""
    shipped = {f"/{resource.path}" for resource in (*task.resources.all, *task.resources.worker)}
    return tuple(path for path in task.output_paths if path in shipped)


async def write_controls(
    task: TaskSpec,
    grader: Grader,
    proposal: TaskProposal,
    model: ModelClient,
    sampling: LLMPolicy,
    record: CallLedger,
    directory: Path,
) -> tuple[Control, ...]:
    """Write the fixed controls of ``task`` to ``<directory>/controls.json`` and return them.

    Args:
        task: The draft's task.
        grader: The output of the draft's grader step.
        proposal: The proposal the task was built from.
        model: The model that writes the controls.
        sampling: Its sampling policy.
        record: Where its calls are recorded.
        directory: The draft round's directory.

    Raises:
        BuildFailure: ``ATTEMPTS`` submissions all had a problem; ``step`` is ``CONTROLS_STEP``.
    """
    request = (
        f"# Task conversation\n\n{format_conversation(task.context.events)}\n\n"
        f"# Answer contract\n\n{grader.answer_contract}\n\n{SUBMISSION}\n\n"
        f"# Reference reply\n\n{grader.reference_reply}\n\n"
        f"# Reference files\n\n{files_text(grader.reference_files, '/')}\n\n"
        f"# Output paths\n\n{output_paths_text(task.output_paths, shipped_output_paths(task))}\n\n"
        f"# Rejection ceiling\n\n{ceiling_text(grader.rejection_ceiling)}\n\n"
        "Write the fixed controls from the proposal's 'Grader design and controls' section: at least one "
        "known_correct positive control with concern reference (reward_min 0.99), one plausible_wrong control "
        "with concern acceptance, one task_specific_shortcut or reward_hack control with concern shortcut, and "
        "one empty_or_malformed control with concern extraction. Keep extraction controls few: they pin only "
        "how the answer is parsed.\n"
        "- A positive control follows the answer contract exactly, in the reference reply's format, so that "
        "only its content is under test; a correct answer in a different format is not a positive control.\n"
        "- A negative control is wrong in substance: a plausible_wrong control gives a wrong answer in the "
        "contract's format, and a task_specific_shortcut or reward_hack control earns credit without doing the "
        "task (a copied reference, a hard-coded output, a grader-pleasing reply). A reply that is right but "
        "formatted differently is not a negative control; it belongs to extraction if anywhere.\n"
        "- Negative and malformed controls demand reward_max at most the rejection ceiling; a candidate the "
        "proposal's partial-credit scheme would score above it is not a negative control."
    )

    async def problem(submitted: ControlsDraft) -> str | None:
        try:
            check_control_files(submitted.controls, task.output_paths)
            controls = [control_from_draft(c) for c in submitted.controls]
            validate_controls(task, controls)
            check_expectations(controls, grader.rejection_ceiling)
            check_controls_not_graded(controls, grader)
        except ValueError as error:
            return str(error)
        return None

    try:
        submitted = await checked_structured(
            model, sampling, record, task_messages(proposal, request), ControlsDraft, problem
        )
    except BuildFailure as failure:
        raise BuildFailure(failure.message, CONTROLS_STEP) from failure
    controls = tuple(control_from_draft(c) for c in submitted.controls)
    directory.mkdir(parents=True, exist_ok=True)
    with atomic_rename(str(directory / CONTROLS_FILE)) as temp:
        Path(temp).write_bytes(controls_json(controls))
    return controls


def load_controls(directory: Path) -> tuple[Control, ...]:
    """Read the controls ``write_controls`` wrote to ``directory``."""
    return parse_controls((directory / CONTROLS_FILE).read_bytes())
