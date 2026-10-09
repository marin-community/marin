# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard builder template: research, fixtures, machine, grader, instructions, assembly, controls.

Steps, in order:

1. ``sources``: web research for the proposal's research items; a build without web tools fails a
   proposal that has any.
2. ``fixtures``: solver-visible files and private reference data.
3. ``requirements``: the task machine. Reasoning and shellsim proposals run in ShellSim (no image);
   a container proposal fails the build, because this host publishes no images.
4. ``grader`` (GRADER): a verifyit answer grader that runs in process (exact, numeric, math, mcq).
5. ``instructions``: the solver-facing prompt, checked to contain no graded answer.
6. ``assemble``: the TaskSpec, through ``b.spec.assemble``.
7. ``controls`` (CONTROLS): fixed controls written for the assembled task, a separate step from
   the grader. They are not replayed here; ``validate`` replays them.

``build`` lowers the task with ``b.lower`` outside any step: the lowered spec names this host's
machine backends, which no step key covers.

Every model-driven step (all but ``sources``, ``requirements`` and ``assemble``) takes a ``guidance``
string that a program uses to specialize the step's prompt without copying it. A step whose model
output fails a check gets the failure back and retries, up to ``ATTEMPTS`` requests, then fails the
build.
"""

import base64
import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskResource,
    TaskSpec,
    format_conversation,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import submission_instruction
from verifyit.spec import ExactSpec, MathSpec, McqSpec, NumericSpec

from taskforge.builder.sdk import NUMERIC_LITERALS, Build, BuildFailure, BuildOutput, Grader
from taskforge.builder.step import SDK_VERSION, StepRole, step
from taskforge.llm.policy import Message
from taskforge.proposal.model import Environment
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    reply,
    shell_turn,
    validate_controls,
)
from taskforge.spec.draft import file, grading_environment, machine, session

WORKDIR = "/workspace"
WORKSPACE = "workspace"
"""Solver-visible files live under this directory, relative to the machine root."""
GRADER_TIMEOUT = 300.0
VERIFIER_TIMEOUT = GRADER_TIMEOUT + 120.0
"""The session's grading deadline: the grader's own timeout plus a verifier machine's start."""
STARTUP_TIMEOUT = 600.0
CLEANUP_TIMEOUT = 120.0
MAX_TURNS = 64
ATTEMPTS = 3
TASK_MACHINE = machine(startup_timeout=STARTUP_TIMEOUT)
"""The template's task machine: no network, the factory's limits, the image's user."""
SESSION = session(
    max_turns=MAX_TURNS,
    model_turn_timeout=None,
    command_timeout=None,
    tool_turn_timeout=None,
    total_turn_timeout=None,
    attempt_timeout=None,
    verifier_timeout=VERIFIER_TIMEOUT,
    cleanup_timeout=CLEANUP_TIMEOUT,
)
"""Validation replaces the turn budget and deadlines with its own; the verifier timeout stands."""
ANSWER_FORMAT = PlainText()
"""Template tasks take a plain-text final reply: their answer type is text."""
SUBMISSION = (
    "RolloutEngine ends the solver's prompt with this submission instruction: "
    f"{submission_instruction(ANSWER_FORMAT)!r}. The whole final reply is the submission ({ANSWER_FORMAT.kind})."
)
SOURCE_DATASET = "taskforge"

TASK_CONTEXT = """\
You are building one reinforcement-learning task from an accepted task proposal. The task is \
solved by a model that sees only the instruction and the solver-visible files, and is graded by \
a private grader it never sees. Follow the proposal; where it is ambiguous, choose the option \
that keeps the grade exact and the task realistic.

# Proposal

{proposal}
"""

SHELLSIM_FACTS = """\
The task machine is ShellSim: a simulated shell with POSIX tools and a minimal `python3` shim, not \
CPython. No pip, no network. The solver works in {workdir} through a `shell(command)` tool and ends \
with a final text reply. Private grader files are never installed in the solver's machine.\
"""

CONTAINER_FACTS = """\
The task machine is a container started from the task image, without network. The solver works in \
{workdir} through a `shell(command)` tool and ends with a final text reply. Private grader files are \
never installed in the solver's machine.\
"""


@dataclass(frozen=True)
class Sources:
    notes: str
    turns: int


@dataclass(frozen=True)
class Fixtures:
    agent_files: tuple[TaskResource, ...]
    private_files: tuple[TaskResource, ...]
    facts: str


@dataclass(frozen=True)
class Instructions:
    system: str | None
    instruction: str


class FileDraft(BaseModel):
    path: str = Field(description="Path relative to its root, without a leading '/' or '..'.")
    content: str = Field(description="Complete UTF-8 file content.")
    executable: bool = Field(description="Whether the file is executable.")

    @field_validator("path")
    @classmethod
    def relative(cls, path: str) -> str:
        parts = path.split("/")
        if path.startswith("/") or "" in parts or ".." in parts:
            raise ValueError(f"file path {path!r} must be relative without '..'")
        return path


class FixturesDraft(BaseModel):
    """Submit the task fixtures: solver-visible files, private reference data, and the facts they encode."""

    agent_files: list[FileDraft] = Field(
        description=f"Solver-visible files, paths relative to the machine root under {WORKSPACE}/ "
        f"({WORKSPACE}/data.csv is {WORKDIR}/data.csv). May be empty."
    )
    private_files: list[FileDraft] = Field(
        description="Grader-only data files, paths relative to the grader's private directory. May be empty."
    )
    facts: str = Field(description="The ground truth these fixtures encode, with every value the grader checks.")

    @field_validator("agent_files")
    @classmethod
    def visible(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, WORKSPACE)


class GraderDraft(BaseModel):
    """Submit the private grader, the answer contract it enforces, and a reference answer."""

    kind: Literal["exact", "numeric", "math", "mcq"] = Field(
        description="exact, numeric, math or mcq: a generic check of the final reply, graded in process."
    )
    expected: str = Field(
        description="For exact, the expected final answer. For numeric, a numeric literal string such as 42, "
        "0.125, 1/8 or 1.5e3. For math, the expected expression, e.g. 3/4 or \\sqrt{2}. For mcq, the correct "
        "option letter. Otherwise empty."
    )
    tolerance: float = Field(description="For numeric, the absolute tolerance; otherwise 0.")
    answer_contract: str = Field(description="The exact output format the solver must follow, for the instruction.")
    reference_reply: str = Field(description="A complete correct final reply that follows the contract.")
    reference_files: list[FileDraft] = Field(
        description=f"Files a correct solver leaves under {WORKSPACE}/ (relative to the machine root), if the "
        "grader reads any; else empty."
    )
    secret_values: list[str] = Field(description="Graded answer strings that must not appear in the instruction.")

    @field_validator("reference_files")
    @classmethod
    def workspace(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, WORKSPACE)


class InstructionsDraft(BaseModel):
    """Submit the solver-facing instruction."""

    system: str = Field(description="An optional system message; empty for none.")
    instruction: str = Field(description="The complete user instruction, including the answer contract.")


class ControlDraft(BaseModel):
    id: str = Field(description="Short id: letters, digits, '.', '_' or '-'.")
    kind: ControlKind
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
    files: list[FileDraft] = Field(
        description=f"Files the candidate writes under {WORKSPACE}/ (relative to the machine root) before replying."
    )
    reward_min: float | None = Field(description="Positive controls: the minimum reward, e.g. 0.99. Otherwise null.")
    reward_max: float | None = Field(
        description=f"Negative and malformed controls: the maximum reward, at most {REJECTION_CEILING}."
    )
    rationale: str = Field(description="Why this candidate must receive that grade.")

    @field_validator("files")
    @classmethod
    def workspace(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, WORKSPACE)


class ControlsDraft(BaseModel):
    """Submit the fixed controls for the task."""

    controls: list[ControlDraft]


def _under(files: list[FileDraft], root: str) -> list[FileDraft]:
    outside = [f.path for f in files if not f.path.startswith(f"{root}/")]
    if outside:
        raise ValueError(f"these files must be under {root}/: {outside}")
    return files


def task_file(draft: FileDraft) -> TaskResource:
    return file(draft.path, draft.content, mode=0o755 if draft.executable else 0o644)


def files_text(files: Sequence[TaskResource], root: str) -> str:
    """Files as markdown sections titled by their path under ``root``, for prompts."""
    sections = (f"### {root}{f.path}\n```\n{resource_bytes(f).decode(errors='replace')}\n```" for f in files)
    return "\n\n".join(sections) or "(none)"


def machine_facts(environment: Environment) -> str:
    """What the solver's machine is, for prompts: a container, or ShellSim for every other environment."""
    facts = CONTAINER_FACTS if environment is Environment.CONTAINER else SHELLSIM_FACTS
    return facts.format(workdir=WORKDIR)


def task_messages(b: Build, request: str, guidance: str) -> list[Message]:
    """The shared system context (the proposal) plus one request and the program's guidance."""
    user = request if not guidance else f"{request}\n\n# Program guidance\n\n{guidance}"
    return [
        {"role": "system", "content": TASK_CONTEXT.format(proposal=b.proposal_text)},
        {"role": "user", "content": user},
    ]


async def accept(value: BaseModel) -> str | None:
    """Accept every submission that parses; no check beyond the output schema."""
    return None


async def structured_until[T: BaseModel](
    b: Build,
    messages: Sequence[Message],
    output_type: type[T],
    name: str,
    problem: Callable[[T], Awaitable[str | None]],
) -> T:
    """Ask for ``output_type`` until ``problem`` returns None, feeding each problem back; fail after ``ATTEMPTS``."""
    conversation: list[Message] = list(messages)
    issue = "no submission"
    for _ in range(ATTEMPTS):
        value = await b.llm.structured(conversation, output_type, name)
        found = await problem(value)
        if found is None:
            return value
        issue = found
        feedback: list[Message] = [
            {"role": "assistant", "content": f"[{name} arguments]\n{value.model_dump_json()}"},
            {"role": "user", "content": f"That submission failed a check:\n{issue}\nFix it and call `{name}` again."},
        ]
        conversation += feedback
    raise b.failure(f"{name}: no acceptable submission after {ATTEMPTS} requests; last problem: {issue}")


@step(StepRole.SOURCES)
async def sources(b: Build, guidance: str) -> Sources:
    """Research the proposal's research items on the web; the notes feed every later step."""
    items = b.proposal.header.research
    if not items:
        return Sources(notes="", turns=0)
    raise b.failure("web research is not configured for this build")


@step(StepRole.FIXTURES)
async def fixtures(b: Build, found: Sources, guidance: str) -> Fixtures:
    """Write the solver-visible files and the private reference data the proposal's build plan calls for."""
    request = (
        f"{machine_facts(b.proposal.header.environment)}\n\n"
        f"# Research notes\n\n{found.notes or '(none)'}\n\n"
        "Write the task fixtures from the proposal's Build plan. Solver-visible files go under "
        f"{WORKSPACE}/ (a reasoning task may put everything in the instruction and ship no files). "
        "Grader-only reference data goes in private_files. Compute every value exactly; the facts field "
        "must state every value the grader will check."
    )

    draft = await structured_until(b, task_messages(b, request, guidance), FixturesDraft, "submit_fixtures", accept)
    agent_files = tuple(task_file(f) for f in draft.agent_files)
    private_files = tuple(task_file(f) for f in draft.private_files)
    for resource in agent_files:
        b.emit(f"/{resource.path}", resource_bytes(resource))
    for resource in private_files:
        b.emit(f"private/{resource.path}", resource_bytes(resource))
    return Fixtures(agent_files=agent_files, private_files=private_files, facts=draft.facts)


@step(StepRole.ENVIRONMENT)
async def requirements(b: Build, made: Fixtures, guidance: str) -> EnvironmentRequirements:
    """The task machine: ShellSim for reasoning and shellsim proposals; no image for a container."""
    if b.proposal.header.environment is not Environment.CONTAINER:
        return b.spec.requirements(image=None, workdir=WORKDIR)
    raise b.failure("a container task needs a published image, and this host has no image builder")


def grader_package(b: Build, draft: GraderDraft, made: Fixtures, task_machine: EnvironmentRequirements) -> GraderPackage:
    """The in-process answer grader a draft describes."""
    if draft.kind == "exact":
        return b.spec.answer_grader(ExactSpec(expected=(draft.expected,), ignore_case=True, ignore_whitespace=True))
    if draft.kind == "numeric":
        return b.spec.answer_grader(
            NumericSpec(expected=draft.expected.strip(), tolerance_abs=draft.tolerance, tolerance_rel=0.0)
        )
    if draft.kind == "math":
        return b.spec.answer_grader(MathSpec(expected=draft.expected.strip()))
    return b.spec.answer_grader(McqSpec(expected=draft.expected.strip()))


@step(StepRole.GRADER)
async def grader(b: Build, made: Fixtures, task_machine: EnvironmentRequirements, guidance: str) -> Grader:
    """Write the in-process grader of the final reply, its answer contract and a reference answer."""
    request = (
        f"{machine_facts(b.proposal.header.environment)}\n\n"
        f"# Fixture facts\n\n{made.facts}\n\n# Solver-visible files\n\n{files_text(made.agent_files, '/')}\n\n"
        f"{SUBMISSION}\n\n{NUMERIC_LITERALS}\n\n"
        "Write the grader from the proposal's 'Grader design and controls' section, as one in-process kind "
        "(exact, numeric, math or mcq). The reference reply must earn full credit."
    )

    async def problem(draft: GraderDraft) -> str | None:
        try:
            grader_package(b, draft, made, task_machine)
        except (ValueError, BuildFailure) as error:
            return str(error)
        return None

    draft = await structured_until(b, task_messages(b, request, guidance), GraderDraft, "submit_grader", problem)
    return Grader(
        package=grader_package(b, draft, made, task_machine),
        answer_contract=draft.answer_contract,
        reference_reply=draft.reference_reply,
        reference_files=tuple(task_file(f) for f in draft.reference_files),
        secret_values=tuple(value for value in draft.secret_values if value.strip()),
    )


@step(StepRole.INSTRUCTIONS)
async def instructions(
    b: Build, made: Fixtures, task_machine: EnvironmentRequirements, graded: Grader, guidance: str
) -> Instructions:
    """Write the solver-facing instruction around the grader's answer contract."""
    request = (
        f"{machine_facts(b.proposal.header.environment)}\n\n"
        f"# Solver-visible files\n\n{files_text(made.agent_files, '/')}\n\n"
        f"# Answer contract the grader enforces\n\n{graded.answer_contract}\n\n{SUBMISSION}\n\n"
        "Write the solver-facing instruction from the proposal's Task section: the scenario, every input the "
        "solver needs that is not in a file, the deliverable, and the answer contract verbatim. Never include "
        "a graded answer, the grader's existence details, or hints that give the answer away. Do not describe "
        "the shell tool or the submission instruction: RolloutEngine adds both."
    )

    async def problem(draft: InstructionsDraft) -> str | None:
        text = f"{draft.system}\n{draft.instruction}"
        leaked = [value for value in graded.secret_values if value in text]
        return None if not leaked else f"the instruction contains graded answers: {leaked}"

    draft = await structured_until(
        b, task_messages(b, request, guidance), InstructionsDraft, "submit_instructions", problem
    )
    return Instructions(system=draft.system.strip() or None, instruction=draft.instruction)


@step(StepRole.ASSEMBLE)
async def assemble(
    b: Build, made: Fixtures, task_machine: EnvironmentRequirements, graded: Grader, text: Instructions
) -> TaskSpec:
    """The item's TaskSpec: the instructions, grader, agent files and task machine, sourced to the proposal."""
    header = b.proposal.header
    return b.spec.assemble(
        task_id=b.item_id,
        instruction=text.instruction,
        answer_type=AnswerType.TEXT,
        answer_format=ANSWER_FORMAT,
        grader=graded.package,
        source=Source(dataset=SOURCE_DATASET, revision=b.proposal.digest, row=header.id, importer_revision=SDK_VERSION),
        environment=task_machine,
        files=made.agent_files,
        system=text.system,
        tags=(header.environment, header.verification),
    )


def control_payload(draft: ControlDraft) -> Transcript:
    """The control as a transcript: one shell turn writing its files (if any), then its final reply."""
    if not draft.files:
        return Transcript(turns=(reply(draft.final_reply),))
    commands = tuple(
        (
            f"{draft.id}-write-{index}",
            f"mkdir -p {shlex.quote('/' + f.path.rsplit('/', 1)[0])} && "
            f"printf %s {shlex.quote(base64.b64encode(f.content.encode()).decode())} "
            f"| base64 -d > {shlex.quote(f'/{f.path}')}",
        )
        for index, f in enumerate(draft.files)
    )
    return Transcript(turns=(shell_turn(*commands), reply(draft.final_reply)))


def control_from_draft(draft: ControlDraft) -> Control:
    return Control(
        id=draft.id,
        kind=draft.kind,
        category=draft.category,
        concern=draft.concern,
        author="template.standard.controls",
        payload=control_payload(draft),
        expect=Expectation(status=Outcome.GRADED, reward_min=draft.reward_min, reward_max=draft.reward_max),
    )


@step(StepRole.CONTROLS)
async def controls(b: Build, task: TaskSpec, graded: Grader, guidance: str) -> tuple[Control, ...]:
    """Write the fixed controls for the assembled task."""
    request = (
        f"# Task conversation\n\n{format_conversation(task.context.events)}\n\n"
        f"# Answer contract\n\n{graded.answer_contract}\n\n{SUBMISSION}\n\n"
        f"# Reference reply\n\n{graded.reference_reply}\n\n"
        f"# Reference files\n\n{files_text(graded.reference_files, '/')}\n\n"
        "Write the fixed controls from the proposal's 'Grader design and controls' section: at least one "
        "known_correct positive control with concern reference (reward_min 0.99), one plausible_wrong control "
        "with concern acceptance, one task_specific_shortcut or reward_hack control with concern shortcut, and "
        "one empty_or_malformed control with concern extraction (negative and malformed controls: reward_max at "
        f"most {REJECTION_CEILING}). Keep extraction controls few: they pin only how the answer is parsed. "
        "A negative control must be a candidate the grader scores at most "
        f"{REJECTION_CEILING}; if the proposal's partial-credit controls would score higher, leave them out."
    )

    async def problem(draft: ControlsDraft) -> str | None:
        try:
            validate_controls(task, [control_from_draft(c) for c in draft.controls])
        except ValueError as error:
            return str(error)
        return None

    draft = await structured_until(b, task_messages(b, request, guidance), ControlsDraft, "submit_controls", problem)
    return tuple(control_from_draft(c) for c in draft.controls)


async def build(b: Build) -> BuildOutput:
    found = await sources(b, "")
    made = await fixtures(b, found, "")
    task_machine = await requirements(b, made, "")
    graded = await grader(b, made, task_machine, "")
    text = await instructions(b, made, task_machine, graded, "")
    task = await assemble(b, made, task_machine, graded, text)
    fixed = await controls(b, task, graded, "")
    lowered = b.lower(
        task,
        task_machine=TASK_MACHINE,
        verifier_machine=None if grading_environment(task) is None else TASK_MACHINE,
        session=SESSION,
    )
    return BuildOutput(task=task, lowered=lowered, controls=fixed)
