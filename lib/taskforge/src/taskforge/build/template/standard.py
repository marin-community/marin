# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard builder template: the session shape proven by the capability_env_gen builds.

Steps, in order:

1. ``sources``: web research for the proposal's research items (an agent with web tools).
2. ``fixtures``: solver-visible files and private reference data.
3. ``environment``: the task machine. Reasoning and shellsim proposals run in ShellSim, so the
   task can ship a grader script; container proposals get a ``DockerBuild``.
4. ``grader`` (GRADER): a task-specific ``ShellVerifierSpec`` script, or a verifyit-backed answer
   verifier, prototyped with ``b.try_grader`` on its reference answer and on an empty answer.
5. ``instructions``: the solver-facing prompt, checked to contain no graded answer.
6. ``assemble``: the TaskSpec, through ``b.spec.assemble``, with ``EXECUTION``.
7. ``controls`` (CONTROLS): fixed controls written for the assembled task, a separate step from
   the grader. They are not replayed here; ``validate`` replays them.

Every step takes a ``guidance`` string that a program uses to specialize the step's prompt
without copying it. A step whose model output fails a check gets the failure back and retries,
up to ``ATTEMPTS`` requests, then fails the build.
"""

import base64
import json
import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from taskcompendium.environment import DockerBuild, EnvironmentFile, EnvironmentKind, EnvironmentSpec, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import AnswerType, Source, TaskSpec, VerifierSpec, format_conversation
from verifyit.spec import ExactSpec, NumericSpec

from taskforge.build.sdk import Build, BuildOutput, Grader
from taskforge.build.step import SDK_VERSION, StepRole, step
from taskforge.llm.agent import AgentStop
from taskforge.llm.policy import Message
from taskforge.proposal.model import Environment
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlKind,
    Expectation,
    Transcript,
    reply,
    shell_turn,
    validate_controls,
)

WORKDIR = "/workspace"
GRADER_DIR = "/grader"
GRADER_SCRIPT = f"{GRADER_DIR}/grade.py"
GRADER_TIMEOUT = 300.0
FULL_CREDIT = 0.99
ATTEMPTS = 3
RESEARCH_TURNS = 32
EXECUTION = TaskExecution()
"""Template tasks have one stage and set no deadlines or agent user."""
SOURCE_DATASET = "taskforge"

TASK_CONTEXT = """\
You are building one reinforcement-learning task from an accepted task proposal. The task is \
solved by a model that sees only the instruction and the solver-visible files, and is graded by \
a private grader it never sees. Follow the proposal; where it is ambiguous, choose the option \
that keeps the grade exact and the task realistic.

# Proposal

{proposal}
"""

MACHINE_FACTS = """\
The task machine is ShellSim: a simulated shell with POSIX tools and a minimal `python3` shim, not \
CPython (json, re, fractions, math, sys work; traceback does not; a compiled regex's match and \
search take only the string). No pip, no network. The solver works in \
/workspace through a `shell(command)` tool and ends with a final text reply. Files under {grader} \
are private: they are installed only after the solver's final reply, then the grader runs.\
"""

GRADER_CONTRACT = """\
A script grader is `python3 {script}`. It runs in the task machine after the solver finishes, \
with the working directory {workdir}. Standard input is the whole conversation as a JSON list of \
OpenAI chat messages; the solver's final reply is the `content` string of the last message whose \
role is "assistant". The script prints exactly one number in [0, 1] (the reward) to stdout and \
exits 0, for every candidate: a malformed, empty, or adversarial answer scores low, it never \
crashes the script. Private data files it reads must live under {grader}/.\
"""


@dataclass(frozen=True)
class Sources:
    notes: str
    turns: int


@dataclass(frozen=True)
class Fixtures:
    agent_files: tuple[EnvironmentFile, ...]
    private_files: tuple[EnvironmentFile, ...]
    facts: str


@dataclass(frozen=True)
class Instructions:
    system: str | None
    instruction: str


class FileDraft(BaseModel):
    path: str = Field(description="Absolute path in the task machine.")
    content: str = Field(description="Complete UTF-8 file content.")
    executable: bool = Field(description="Whether the file is executable.")

    @field_validator("path")
    @classmethod
    def absolute(cls, path: str) -> str:
        if not path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"file path {path!r} must be absolute without '..'")
        return path


class FixturesDraft(BaseModel):
    """Submit the task fixtures: solver-visible files, private reference data, and the facts they encode."""

    agent_files: list[FileDraft] = Field(description=f"Solver-visible files, under {WORKDIR}/. May be empty.")
    private_files: list[FileDraft] = Field(description=f"Grader-only data files, under {GRADER_DIR}/. May be empty.")
    facts: str = Field(description="The ground truth these fixtures encode, with every value the grader checks.")

    @field_validator("agent_files")
    @classmethod
    def visible(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, WORKDIR)

    @field_validator("private_files")
    @classmethod
    def private(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, GRADER_DIR)


class DockerfileDraft(BaseModel):
    """Submit the task image: a Dockerfile and the build-context files it copies."""

    dockerfile: str = Field(description="Dockerfile content; WORKDIR /workspace.")
    context_files: list[FileDraft] = Field(
        description="Other build-context files, paths relative to the context root as /name."
    )


class GraderDraft(BaseModel):
    """Submit the private grader, the answer contract it enforces, and a reference answer."""

    kind: Literal["script", "exact", "numeric"] = Field(
        description="script: a Python grader script; exact or numeric: a generic answer check on the final reply."
    )
    script: str = Field(description="For kind=script, the grader script source; otherwise empty.")
    private_files: list[FileDraft] = Field(description=f"Data files the script reads, under {GRADER_DIR}/.")
    expected: str = Field(description="For exact or numeric, the expected final answer; otherwise empty.")
    tolerance: float = Field(description="For numeric, the absolute tolerance; otherwise 0.")
    answer_contract: str = Field(description="The exact output format the solver must follow, for the instruction.")
    reference_reply: str = Field(description="A complete correct final reply that follows the contract.")
    reference_files: list[FileDraft] = Field(
        description=f"Files a correct solver leaves under {WORKDIR}/, if the grader reads any; else empty."
    )
    secret_values: list[str] = Field(description="Graded answer strings that must not appear in the instruction.")

    @field_validator("private_files")
    @classmethod
    def private(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, GRADER_DIR)

    @field_validator("reference_files")
    @classmethod
    def workspace(cls, files: list[FileDraft]) -> list[FileDraft]:
        return _under(files, WORKDIR)


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
    final_reply: str = Field(description="The candidate's final reply.")
    files: list[FileDraft] = Field(description=f"Files the candidate writes under {WORKDIR}/ before replying.")
    reward_min: float | None = Field(description="Positive controls: the minimum reward, e.g. 0.99. Otherwise null.")
    reward_max: float | None = Field(
        description=f"Negative and malformed controls: the maximum reward, at most {REJECTION_CEILING}."
    )
    rationale: str = Field(description="Why this candidate must receive that grade.")


class ControlsDraft(BaseModel):
    """Submit the fixed controls for the task."""

    controls: list[ControlDraft]


def _under(files: list[FileDraft], root: str) -> list[FileDraft]:
    outside = [f.path for f in files if not f.path.startswith(f"{root}/")]
    if outside:
        raise ValueError(f"these files must be under {root}/: {outside}")
    return files


def environment_file(draft: FileDraft) -> EnvironmentFile:
    return EnvironmentFile(path=draft.path, content=draft.content.encode(), mode=0o755 if draft.executable else 0o644)


def files_text(files: Sequence[EnvironmentFile]) -> str:
    """Files as markdown sections, for prompts."""
    return "\n\n".join(f"### {f.path}\n```\n{f.content.decode(errors='replace')}\n```" for f in files) or "(none)"


def task_messages(b: Build, request: str, guidance: str) -> list[Message]:
    """The shared system context (the proposal) plus one request and the program's guidance."""
    user = request if not guidance else f"{request}\n\n# Program guidance\n\n{guidance}"
    return [
        {"role": "system", "content": TASK_CONTEXT.format(proposal=b.proposal_text)},
        {"role": "user", "content": user},
    ]


async def accept(value: BaseModel) -> str | None:
    """A ``structured_until`` check that accepts any valid submission."""
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
    request = (
        "Research these items with the web tools, then reply with concise notes: for each item, what you "
        "confirmed, the source URLs, and anything that contradicts the proposal.\n\n"
        + "\n".join(f"- ({item.kind}) {item.purpose}" for item in items)
    )
    run = await b.llm.agent(task_messages(b, request, guidance), b.research, RESEARCH_TURNS)
    b.check(run.stop is AgentStop.ANSWERED, f"research ended with {run.stop} after {len(run.turns)} turns")
    notes = run.turns[-1].completion.content
    b.emit("research/notes.md", notes.encode())
    return Sources(notes=notes, turns=len(run.turns))


@step(StepRole.FIXTURES)
async def fixtures(b: Build, found: Sources, guidance: str) -> Fixtures:
    """Write the solver-visible files and the private reference data the proposal's build plan calls for."""
    request = (
        f"{MACHINE_FACTS.format(grader=GRADER_DIR)}\n\n# Research notes\n\n{found.notes or '(none)'}\n\n"
        "Write the task fixtures from the proposal's Build plan. Solver-visible files go under "
        f"{WORKDIR}/ (a reasoning task may put everything in the instruction and ship no files). "
        f"Grader-only reference data goes under {GRADER_DIR}/. Compute every value exactly; the facts field "
        "must state every value the grader will check."
    )

    draft = await structured_until(b, task_messages(b, request, guidance), FixturesDraft, "submit_fixtures", accept)
    agent_files = tuple(environment_file(f) for f in draft.agent_files)
    private_files = tuple(environment_file(f) for f in draft.private_files)
    for f in (*agent_files, *private_files):
        b.emit(f.path, f.content)
    return Fixtures(agent_files=agent_files, private_files=private_files, facts=draft.facts)


@step(StepRole.ENVIRONMENT)
async def environment(b: Build, made: Fixtures, guidance: str) -> EnvironmentSpec:
    """The task machine: ShellSim for reasoning and shellsim proposals, a DockerBuild for containers."""
    if b.proposal.header.environment is not Environment.CONTAINER:
        return b.spec.environment(EnvironmentKind.SHELLSIM, files=made.agent_files, workdir=WORKDIR)
    request = (
        "Write the task image: a Dockerfile (WORKDIR /workspace, no network at run time) and any build-context "
        f"files it copies. The solver-visible files below are installed by the task, not the image.\n\n"
        f"{files_text(made.agent_files)}"
    )

    draft = await structured_until(b, task_messages(b, request, guidance), DockerfileDraft, "submit_image", accept)
    context = (EnvironmentFile(path="/Dockerfile", content=draft.dockerfile.encode()),)
    context += tuple(environment_file(f) for f in draft.context_files)
    return b.spec.environment(
        EnvironmentKind.DOCKER, image=DockerBuild(files=context), files=made.agent_files, workdir=WORKDIR
    )


def grader_verifier(b: Build, draft: GraderDraft, made: Fixtures) -> VerifierSpec:
    """The VerifierSpec a grader draft describes."""
    if draft.kind == "exact":
        return b.spec.answer_verifier(ExactSpec(expected=(draft.expected,), ignore_case=True, ignore_whitespace=True))
    if draft.kind == "numeric":
        return b.spec.answer_verifier(
            NumericSpec(expected=draft.expected.strip(), tolerance_abs=draft.tolerance, tolerance_rel=0.0)
        )
    files = (
        EnvironmentFile(path=GRADER_SCRIPT, content=draft.script.encode(), mode=0o755),
        *made.private_files,
        *(environment_file(f) for f in draft.private_files),
    )
    return b.spec.shell_verifier(
        argv=("python3", GRADER_SCRIPT), reward=StdoutReward(), timeout=GRADER_TIMEOUT, files=files
    )


@step(StepRole.GRADER)
async def grader(b: Build, made: Fixtures, machine: EnvironmentSpec, guidance: str) -> Grader:
    """Write the grader and prototype it: the reference answer gets full credit, an empty answer does not."""
    request = (
        f"{MACHINE_FACTS.format(grader=GRADER_DIR)}\n\n"
        f"{GRADER_CONTRACT.format(script=GRADER_SCRIPT, workdir=WORKDIR, grader=GRADER_DIR)}\n\n"
        f"# Fixture facts\n\n{made.facts}\n\n# Solver-visible files\n\n{files_text(made.agent_files)}\n\n"
        f"# Private files already under {GRADER_DIR}/\n\n{files_text(made.private_files)}\n\n"
        "Write the grader from the proposal's 'Grader design and controls' section. Prefer kind=script for "
        "anything beyond one exact or numeric answer. Give partial credit only where the proposal does. The "
        "reference reply and files must earn full credit."
    )

    async def problem(draft: GraderDraft) -> str | None:
        if draft.kind == "script" and not draft.script.strip():
            return "kind=script needs the script source"
        verifier = grader_verifier(b, draft, made)
        reference = await b.try_grader(
            machine,
            verifier,
            "(instruction)",
            draft.reference_reply,
            [environment_file(f) for f in draft.reference_files],
        )
        if reference.status != Outcome.GRADED or (reference.reward or 0.0) < FULL_CREDIT:
            return (
                f"the reference answer was graded {reference.status} reward={reference.reward}: "
                f"{_diagnostics(reference)}"
            )
        empty = await b.try_grader(machine, verifier, "(instruction)", "")
        if empty.status == Outcome.GRADED and (empty.reward or 0.0) > REJECTION_CEILING:
            return f"an empty answer got reward {empty.reward}; it must get at most {REJECTION_CEILING}"
        return None

    draft = await structured_until(b, task_messages(b, request, guidance), GraderDraft, "submit_grader", problem)
    verifier = grader_verifier(b, draft, made)
    if draft.kind == "script":
        b.emit(GRADER_SCRIPT, draft.script.encode())
    return Grader(
        verifier=verifier,
        answer_contract=draft.answer_contract,
        reference_reply=draft.reference_reply,
        reference_files=tuple(environment_file(f) for f in draft.reference_files),
        secret_values=tuple(value for value in draft.secret_values if value.strip()),
    )


def _diagnostics(grade: GradeResult) -> str:
    return json.dumps({"error": grade.error, **grade.diagnostics})[:4000]


@step(StepRole.INSTRUCTIONS)
async def instructions(b: Build, made: Fixtures, graded: Grader, guidance: str) -> Instructions:
    """Write the solver-facing instruction around the grader's answer contract."""
    request = (
        f"{MACHINE_FACTS.format(grader=GRADER_DIR)}\n\n# Solver-visible files\n\n{files_text(made.agent_files)}\n\n"
        f"# Answer contract the grader enforces\n\n{graded.answer_contract}\n\n"
        "Write the solver-facing instruction from the proposal's Task section: the scenario, every input the "
        "solver needs that is not in a file, the deliverable, and the answer contract verbatim. Never include "
        "a graded answer, the grader's existence details, or hints that give the answer away."
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
async def assemble(b: Build, machine: EnvironmentSpec, graded: Grader, text: Instructions) -> TaskSpec:
    """The TaskSpec, checked by ``b.spec.assemble``."""
    header = b.proposal.header
    return b.spec.assemble(
        task_id=b.item_id,
        instruction=text.instruction,
        answer_type=AnswerType.TEXT,
        environment=machine,
        verifier=graded.verifier,
        source=Source(dataset=SOURCE_DATASET, revision=b.proposal.digest, row=header.id, importer_revision=SDK_VERSION),
        execution=EXECUTION,
        system=text.system,
        metadata={"proposal_id": header.id, "environment": header.environment, "verification": header.verification},
        tags=(header.environment, header.verification),
    )


def control_payload(draft: ControlDraft) -> Transcript:
    """The control as a transcript: one shell turn writing its files (if any), then its final reply."""
    if not draft.files:
        return Transcript(turns=(reply(draft.final_reply),))
    commands = tuple(
        (
            f"{draft.id}-write-{index}",
            f"mkdir -p {shlex.quote(f.path.rsplit('/', 1)[0] or '/')} && "
            f"printf %s {shlex.quote(base64.b64encode(f.content.encode()).decode())} "
            f"| base64 -d > {shlex.quote(f.path)}",
        )
        for index, f in enumerate(draft.files)
    )
    return Transcript(turns=(shell_turn(*commands), reply(draft.final_reply)))


def control_from_draft(draft: ControlDraft) -> Control:
    return Control(
        id=draft.id,
        kind=draft.kind,
        category=draft.category,
        author="template.standard.controls",
        payload=control_payload(draft),
        expect=Expectation(status=Outcome.GRADED, reward_min=draft.reward_min, reward_max=draft.reward_max),
    )


@step(StepRole.CONTROLS)
async def controls(b: Build, task: TaskSpec, graded: Grader, guidance: str) -> tuple[Control, ...]:
    """Write fixed controls for the assembled task; ``validate`` replays them later."""
    request = (
        f"# Task conversation\n\n{format_conversation(task.context.events)}\n\n"
        f"# Answer contract\n\n{graded.answer_contract}\n\n"
        f"# Reference reply\n\n{graded.reference_reply}\n\n# Reference files\n\n{files_text(graded.reference_files)}\n\n"
        "Write the fixed controls from the proposal's 'Grader design and controls' section: at least one "
        "known_correct positive control (reward_min 0.99), one empty_or_malformed control, one plausible_wrong "
        "control, and one task_specific_shortcut or reward_hack control (reward_max at most "
        f"{REJECTION_CEILING}). A negative control must be a candidate the grader scores at most "
        f"{REJECTION_CEILING}; if the proposal's partial-credit controls would score higher, leave them out. "
        "Do not write partial controls: this template's grader prints one reward and reports no reward "
        "components, which a partial control needs."
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
    machine = await environment(b, made, "")
    graded = await grader(b, made, machine, "")
    text = await instructions(b, made, graded, "")
    task = await assemble(b, machine, graded, text)
    fixed = await controls(b, task, graded, "")
    return BuildOutput(task=task, execution=EXECUTION, controls=fixed)
