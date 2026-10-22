# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The functions, constants and structured-call schemas a build program's steps call.

An authored program defines only steps, output models and ``PROGRAM`` at module level, because a
step's memo key covers its own source alone. Everything else a step needs comes from here,
injected into the program's namespace by ``loader.compile_program`` (``SDK_EXPORTS``). A change to
this module that alters what an unchanged step returns comes with an ``SDK_VERSION`` bump.

Model calls go through ``structured``, ``structured_until`` and ``agent``, which record each call
under the running step (``context.call_ledger``). ``structured_until`` feeds a failed check back to
the model and asks again, up to ``ATTEMPTS`` requests, then raises ``BuildFailure``.
``research`` runs a web-research agent for one research item of the proposal.
"""

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import (
    ArtifactKind,
    CommandSemantics,
    EnvironmentRequirements,
    PlainText,
    ScriptGrader,
    TaskResource,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import submission_instruction
from verifyit.spec import ExactSpec, MathSpec, McqSpec, NumericSpec

from taskforge.llm.agent import AgentRun, AgentStop, AgentTool, AgentTurn, ToolOutcome, run_agent
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_structured
from taskforge.llm.structured import CutReplyError, StructuredOutputError
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec import draft
from taskforge.spec.decision import REJECTED_REWARD_MAX, DecisionGrader
from taskforge.spec.draft import ANSWER_PATH, GRADER_RESOURCE_ROOT
from taskforge.spec.rubric import Rubric
from taskforge.spec.verdict import GRADER_VERDICT_PATH, VERDICT_REWARD
from taskforge.stage10_proposal.model import ProposalHeader, SolverMachine, TaskProposal, render
from taskforge.stage40_build_with_program.context import BuildFailure, RLEnvBuildContext, call_ledger
from taskforge.stage40_build_with_program.outputs import Fixtures, GradedCandidate, Grader, ResearchFinding, WebCall

WORKDIR = "/workspace"
"""The solver's working directory in the task machine."""
WORKSPACE = "workspace"
"""Solver-visible files live under this directory, relative to the machine root."""
GRADER_PROGRAM = "grade.py"
GRADER_TIMEOUT = 300.0
"""Seconds a grader program may run on one candidate."""
FULL_CREDIT = 0.99
"""The least reward a reference answer must receive."""
RUBRIC_KINDS = frozenset({"decision_model", "llm_as_judge"})
"""``GraderDraft`` kinds a model applies through a rubric."""
EQUIVALENT_VARIANTS = 3
"""Equivalent variants of the reference a decision-model grader is calibrated on."""
ATTEMPTS = 3
"""Requests ``structured_until`` sends before it fails the build."""
CUT_REPLY_CHARS = 2000
"""How much of a reply the output limit cut off goes back to the model as its own turn."""
INSTRUCTION_WORD_LIMIT = 500
"""Words an instruction may hold outside the verbatim answer contract."""
INLINED_LINE_CHARS = 40
"""A solver-visible file line at least this long must not be copied into the instruction."""
HAND_COMPUTATION = re.compile(r"\bby hand\b|\bcompute (?:it )?manually\b", re.IGNORECASE)
GRADING_LANGUAGE = re.compile(
    r"\b(grader|graders|(?:un)?graded|grading|auto-?grad\w*|machine[- ]check\w*|checker|checkers|scorer|rubric)\b",
    re.IGNORECASE,
)
"""Words that describe the grading machinery to the solver. "parser" is left out: parsing tasks use it as
their subject, and a refusal the author cannot satisfy would fail the build."""
ANSWER_FORMAT = PlainText()
"""The answer format of template tasks: a plain-text final reply, read as a text answer."""
SUBMISSION = (
    "RolloutEngine ends the solver's prompt with this submission instruction: "
    f"{submission_instruction(ANSWER_FORMAT)!r}. The whole final reply is the submission ({ANSWER_FORMAT.kind})."
)
SOURCE_DATASET = "taskforge"
"""``Source.dataset`` of every task a build assembles."""
DOCKER_IMAGE_REQUIREMENTS = (
    "A task image (`publish_image(context, DockerBuild(...))`) must provide `sh` and `setsid` (util-linux, or "
    "busybox with its setsid applet): shellbox's Docker backend starts every command under setsid and refuses an "
    "image without it, so distroless and scratch images cannot run a task. On Iris the image runs under gVisor, "
    "which cannot switch users: leave the image's user as root."
)
NUMERIC_LITERALS = (
    "A numeric answer's expected value (verifyit `NumericSpec.expected`) is a literal string: an integer, "
    'decimal, scientific-notation number or integer fraction such as "42", "-0.125", "1.5e3" or "1/8". '
    'It is never a float or an expression (not 0.125, not "sqrt(2)").'
)
DIAGNOSTICS_CHARS = 4000
"""The longest ``grade_diagnostics`` text, the cut marker aside: a longer one keeps half at each end."""
WEB_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
WEB_ERROR_PREFIX = "error:"
"""How a web tool's output starts when the request failed (``taskforge.llm.web``)."""

TASK_CONTEXT = """\
You are building one reinforcement-learning task from an accepted task proposal. The task is \
solved by a model that sees only the instruction and the solver-visible files, and is graded by \
a private grader it never sees. Follow the proposal; where it is ambiguous, choose the option \
that keeps the grade exact and the task realistic.

# Proposal

{proposal}
"""

RESEARCH_PROMPT = """\
You are researching one item of an accepted task proposal before its task is built.

# Proposal

{proposal}

# Research item research[{index}] ({kind})

{purpose}

Look this item up with the web tools: search, then fetch the pages that settle it. Reply with notes \
for the builder: what you confirmed (exact values, versions, formats), the source URLs, the license of any \
data or code the task would ship, and anything that contradicts the proposal. Report only what the sources \
say; say plainly what you could not confirm.\
"""

SHELLSIM_FACTS = """\
The task machine is ShellSim: a simulated shell with POSIX tools, `awk`, `jq` and `python3`, which \
is CPython 3.13 with its standard library only. No pip, no network. The machine has a fixed compute \
budget, a few seconds of Python work over the whole attempt, and a virtual clock that advances only \
through `sleep`. The solver works in {workdir} through a `shell(command)` tool and ends with a final \
text reply. Private grader files are never installed in the solver's machine.\
"""

CPYTHON_FACTS = """\
The task machine is a container with CPython 3.12 and its standard library only (no third-party \
packages), `sh` and the usual Debian command-line tools, as root, without network. The solver works in \
{workdir} through a `shell(command)` tool, may write and run Python scripts there, and ends with a final \
text reply. Private grader files are never installed in the solver's machine.\
"""

PACKAGES_FACTS = """\
The task machine runs CPython 3.12 with its standard library and {packages}, `sh` and the usual \
command-line tools, without network; nothing else can be installed. The solver works in {workdir} \
through a `shell(command)` tool, may write and run Python scripts there, and ends with a final text \
reply. Private grader files are never installed in the solver's machine.\
"""

CONTAINER_FACTS = """\
The task machine is a container started from the task image, without network. The solver works in \
{workdir} through a `shell(command)` tool and ends with a final text reply. Private grader files are \
never installed in the solver's machine.\
"""

NO_MACHINE_FACTS = """\
The solver has no machine: it reads the instructions and replies in the conversation, with no tools and \
no files. Everything the solver needs is in the instructions, so agent_files is empty; grader-only \
reference data goes in private_files.\
"""

SCRIPT_CONTRACT = """\
The grader is `python3 {tests}/{script}`, run after the solver finishes in a fresh machine from \
Taskforge's grader-base image: CPython 3.12 with the standard library only, as root, working \
directory /app. It never runs in the solver's machine. `{answer}` holds the solver's final reply \
(the submission the answer format extracted); each reference file the grader reads, at `<path>` \
relative to the machine root, is at `/<path>` when the solver wrote it (absent otherwise); the \
private data files are under {tests}/, and {tests}/conversation.json is the whole conversation as a \
JSON list of chat messages. The script never reads standard input. It writes its verdict, one JSON object \
`{{"reward": r, "detail": {{"checks": [{{"name": ..., "passed": true|false, "weight": w}}, ...]}}}}`, to \
`{verdict}` and exits 0, for every candidate: one check per independently gradable property, and the reward \
is the weighted share of passed checks unless the proposal states another formula. A malformed, empty, or \
adversarial answer scores low (its checks fail); it never crashes the script.\
"""

CONTAINER_CONTRACT = """\
The grader is `python3 {tests}/{script}`, run after the solver finishes in a fresh container from the \
task image, as the image's user, with the working directory {workdir}. {workdir} there holds a copy of \
the solver's {workdir}, and `{answer}` holds the solver's final reply (the submission the answer \
format extracted). The private data files are under {tests}/, and {tests}/conversation.json is the \
whole conversation as a JSON list of chat messages. The script never reads standard input. It writes \
its verdict, one JSON object \
`{{"reward": r, "detail": {{"checks": [{{"name": ..., "passed": true|false, "weight": w}}, ...]}}}}`, to \
`{verdict}` and exits 0, for every candidate: one check per independently gradable property, and the reward \
is the weighted share of passed checks unless the proposal states another formula. A malformed, empty, or \
adversarial answer scores low (its checks fail); it never crashes the script.\
"""

FEEDBACK_PROMPT = "That submission failed a check:\n{problem}\nFix it and send the complete corrected submission."


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


def under_workspace(files: list[FileDraft]) -> list[FileDraft]:
    """``files``, after checking that each path lies under ``WORKSPACE/``.

    Raises:
        ValueError: a path lies outside ``WORKSPACE/``.
    """
    outside = [f.path for f in files if not f.path.startswith(f"{WORKSPACE}/")]
    if outside:
        raise ValueError(f"these files must be under {WORKSPACE}/: {outside}")
    return files


class FixturesDraft(BaseModel):
    """Submit the task fixtures: solver-visible files, private reference data, the facts they encode, and the
    packages of a package machine."""

    agent_files: list[FileDraft] = Field(
        description=f"Solver-visible files, paths relative to the machine root under {WORKSPACE}/ "
        f"({WORKSPACE}/data.csv is {WORKDIR}/data.csv). May be empty."
    )
    private_files: list[FileDraft] = Field(
        description="Grader-only data files, paths relative to the grader's private directory. May be empty."
    )
    facts: str = Field(description="The ground truth these fixtures encode, with every value the grader checks.")
    packages: list[str] = Field(
        description="When the proposal declares solver_machine: packages, the PyPI requirements the solver's "
        "machine installs, one per entry, such as numpy or pandas==2.2.3, each available as a wheel; otherwise empty."
    )

    @field_validator("agent_files")
    @classmethod
    def visible(cls, files: list[FileDraft]) -> list[FileDraft]:
        return under_workspace(files)

    @field_validator("packages")
    @classmethod
    def requirements(cls, packages: list[str]) -> list[str]:
        draft.check_packages(packages)
        return packages


class DockerfileDraft(BaseModel):
    """Submit the task image: a Dockerfile and the build-context files it copies."""

    dockerfile: str = Field(
        description=f"Dockerfile content; WORKDIR {WORKDIR}; the image provides sh, setsid and python3."
    )
    context_files: list[FileDraft] = Field(description="Other build-context files, paths relative to the context root.")


class GraderDraft(BaseModel):
    """Submit the private grader, the answer contract it enforces, and a reference answer."""

    kind: Literal["decision_model", "llm_as_judge", "exact", "numeric", "math", "mcq", "script"] = Field(
        description="decision_model: a decision model compares the whole reply with reference_reply against "
        "rubric (the usual choice for a reply a person reads; the reply needs no fixed format); "
        "llm_as_judge: a model judge applies rubric to a reply whose prose is the product; exact, "
        "numeric, math or mcq: a generic check of a final reply that is one value in an unambiguous format, "
        "graded in process; script: a Python grader program, for a deliverable with a strict structure the "
        "requester specified whose graded parts are all deterministic, or for what only execution can judge; "
        "never for prose."
    )
    rubric: Rubric | None = Field(
        description="For decision_model and llm_as_judge, the rubric: what a correct reply delivers, one item per "
        "property, each stated as its meaning with reference values, tolerances, units and accepted equivalences; "
        "otherwise null."
    )
    equivalent_variants: list[str] = Field(
        description="For decision_model, exactly three complete replies that are as correct as reference_reply "
        "and differ only in form: one paraphrased, one in different notation or units, one with its parts "
        "reordered. They calibrate the acceptance bar and must all be accepted. Otherwise empty."
    )
    script: str = Field(description="For kind=script, the grader program source; otherwise empty.")
    private_files: list[FileDraft] = Field(
        description="Further data files the script reads, paths relative to the grader's private directory."
    )
    expected: str = Field(
        description="For exact, the expected final answer. For numeric, a numeric literal string such as 42, "
        "0.125, 1/8 or 1.5e3. For math, the expected expression, e.g. 3/4 or \\sqrt{2}. For mcq, the correct "
        "option letter. Otherwise empty."
    )
    tolerance: float = Field(description="For numeric, the absolute tolerance; otherwise 0.")
    answer_contract: str = Field(
        description="For decision_model, the deliverable statement: what the solver hands back, in the "
        "requester's words, with no answer block, labelled fields or fixed layout unless the work itself has one "
        "or the requester would specify one. "
        "For the other kinds, the exact output format the solver must follow. Written as requirements for the "
        "instruction; never describe grading, parsing, credit or tolerances."
    )
    reference_reply: str = Field(description="A complete correct final reply that follows the contract.")
    reference_files: list[FileDraft] = Field(
        description=f"Files a correct solver leaves under {WORKSPACE}/ (relative to the machine root), if the "
        "grader reads any; else empty. Only these paths reach an image-less task's grader program."
    )
    secret_values: list[str] = Field(description="Graded answer strings that must not appear in the instruction.")
    rejection_ceiling: float = Field(
        description="The highest reward this grader gives a submission that is wrong in substance: 0 for an "
        "all-or-nothing grader, else the most partial credit a wrong answer can collect under the proposal's "
        "partial-credit scheme (for example 0.4 when format and a correct intermediate earn 0.4 together). The "
        "empty answer must score at most this, and validation's negative controls will demand at most this. "
        f"For decision_model, {REJECTED_REWARD_MAX}: below its bar a decision-model grader gives at most that."
    )

    @model_validator(mode="after")
    def rubric_kind(self) -> "GraderDraft":
        if (self.kind in RUBRIC_KINDS) != (self.rubric is not None):
            raise ValueError(f"give a rubric exactly when kind is one of {sorted(RUBRIC_KINDS)}")
        decides = self.kind == "decision_model"
        if decides and len(self.equivalent_variants) != EQUIVALENT_VARIANTS:
            raise ValueError(f"kind=decision_model needs exactly {EQUIVALENT_VARIANTS} equivalent_variants")
        if not decides and self.equivalent_variants:
            raise ValueError("equivalent_variants are for kind=decision_model only")
        if decides and self.rejection_ceiling != REJECTED_REWARD_MAX:
            raise ValueError(f"kind=decision_model has rejection_ceiling {REJECTED_REWARD_MAX}")
        return self

    @field_validator("reference_files")
    @classmethod
    def workspace(cls, files: list[FileDraft]) -> list[FileDraft]:
        return under_workspace(files)

    @field_validator("rejection_ceiling")
    @classmethod
    def below_full_credit(cls, ceiling: float) -> float:
        if not 0 <= ceiling < FULL_CREDIT:
            raise ValueError(f"rejection_ceiling is a reward from 0 up to, not including, {FULL_CREDIT}; got {ceiling}")
        return ceiling


class InstructionsDraft(BaseModel):
    """Submit the solver-facing instruction."""

    system: str = Field(description="An optional system message; empty for none.")
    instruction: str = Field(description="The complete user instruction, including the answer contract.")


def task_file(file: FileDraft) -> TaskResource:
    """The task resource ``file`` describes, with mode 755 when it is executable and 644 otherwise."""
    return draft.file(file.path, file.content, mode=0o755 if file.executable else 0o644)


def files_text(files: Sequence[TaskResource], root: str) -> str:
    """``files`` as markdown sections titled by their path under ``root``, for prompts."""
    sections = (f"### {root}{f.path}\n```\n{resource_bytes(f).decode(errors='replace')}\n```" for f in files)
    return "\n\n".join(sections) or "(none)"


def solver_machine(requirements: EnvironmentRequirements) -> SolverMachine | None:
    """The solver machine ``requirements`` (a ``draft.requirements`` result, or ``draft.NO_MACHINE``) describe,
    or ``None`` for the task's own image."""
    if draft.SHELL_CAPABILITY not in requirements.capabilities:
        return SolverMachine.NONE
    if requirements.docker_image == GRADER_BASE_IMAGE:
        return SolverMachine.CPYTHON
    if requirements.docker_image is not None:
        return None
    if requirements.command_semantics == CommandSemantics.LINUX_PROCESS:
        return SolverMachine.PACKAGES
    return SolverMachine.SHELLSIM


def _facts(machine: SolverMachine | None, packages: str) -> str:
    if machine is None:
        return CONTAINER_FACTS.format(workdir=WORKDIR)
    if machine is SolverMachine.NONE:
        return NO_MACHINE_FACTS
    if machine is SolverMachine.PACKAGES:
        return PACKAGES_FACTS.format(workdir=WORKDIR, packages=packages)
    if machine is SolverMachine.CPYTHON:
        return CPYTHON_FACTS.format(workdir=WORKDIR)
    return SHELLSIM_FACTS.format(workdir=WORKDIR)


def machine_facts(requirements: EnvironmentRequirements) -> str:
    """The solver's machine ``requirements`` describe, for a prompt: ShellSim, CPython on the grader base
    image, CPython with the packages of a package machine, or the task's own container."""
    spec = requirements.package_spec
    listed = "" if spec is None else ", ".join(f"`{package}`" for package in spec.pypi_packages or ())
    return _facts(solver_machine(requirements), f"these PyPI packages: {listed}")


def environment_facts(header: ProposalHeader) -> str:
    """The solver's machine the proposal declares, for a prompt written before the requirements step builds it."""
    return _facts(header.solver_machine, "the PyPI packages you list, installed from wheels with no image to build")


def grader_contract(requirements: EnvironmentRequirements) -> str:
    """What a grader program receives and writes, for a task on ``requirements``."""
    if requirements.docker_image is None:
        return SCRIPT_CONTRACT.format(
            tests=GRADER_RESOURCE_ROOT, script=GRADER_PROGRAM, answer=ANSWER_PATH, verdict=GRADER_VERDICT_PATH
        )
    return CONTAINER_CONTRACT.format(
        tests=GRADER_RESOURCE_ROOT,
        script=GRADER_PROGRAM,
        workdir=WORKDIR,
        answer=ANSWER_PATH,
        verdict=GRADER_VERDICT_PATH,
    )


def task_messages(proposal: TaskProposal, request: str) -> list[Message]:
    """A system message holding the proposal, then ``request`` as the user message."""
    return [
        {"role": "system", "content": TASK_CONTEXT.format(proposal=render(proposal))},
        {"role": "user", "content": request},
    ]


async def structured[T: BaseModel](context: RLEnvBuildContext, messages: Sequence[Message], output_type: type[T]) -> T:
    """One structured call for ``output_type``, recorded under the running step.

    The client repairs a reply that fails ``output_type``'s validation, so checks the model can fix
    belong in its validators. A reply cut off by the output limit is not repaired: the call raises
    ``StructuredOutputError``.
    """
    result = await recorded_structured(
        context.model, messages, output_type, context.sampling, call_ledger(context), {"output": output_type.__name__}
    )
    return result.value


def packages_declared(proposal: TaskProposal) -> Callable[[FixturesDraft], Awaitable[str | None]]:
    """The fixtures check that sends back fixtures the proposal's declared machine cannot hold.

    A proposal that declares ``solver_machine: packages`` needs at least one package; any other
    proposal lists none, since its machine has no packages to install. A proposal that declares
    ``solver_machine: none`` has no machine to install files in, so it ships no solver-visible files.
    """
    machine = proposal.header.solver_machine

    async def problem(submitted: FixturesDraft) -> str | None:
        if machine is SolverMachine.PACKAGES and not submitted.packages:
            return "the proposal declares solver_machine: packages; list the PyPI packages the solver's machine installs"
        if machine is not SolverMachine.PACKAGES and submitted.packages:
            declared = "a container" if machine is None else f"solver_machine: {machine}"
            return f"the proposal declares {declared}, which installs no packages; list none"
        if machine is SolverMachine.NONE and submitted.agent_files:
            return (
                "the proposal declares solver_machine: none, so the solver has no machine and no files; put what "
                "the solver needs in the instructions and list no agent_files"
            )
        return None

    return problem


async def accept(value: BaseModel) -> str | None:
    """A ``structured_until`` check that accepts any valid submission."""
    return None


async def checked_structured[T: BaseModel](
    model: ModelClient,
    sampling: LLMPolicy,
    record: CallLedger,
    messages: Sequence[Message],
    output_type: type[T],
    problem: Callable[[T], Awaitable[str | None]],
) -> T:
    """Ask for ``output_type`` until ``problem`` returns ``None``, sending each problem back.

    Each failed submission joins the conversation as the assistant turn, followed by its problem as
    the next user turn. A check that hands back another check's coroutine, as ``return
    accept(value)`` does, is awaited rather than read as a problem. ``structured_until`` calls this
    with a step's context; validation calls it directly.

    A reply the output limit cut off counts as a failed submission too: its first
    ``CUT_REPLY_CHARS`` go back as the assistant turn with the cut named as the problem, so the model
    sends a shorter submission rather than the workpiece failing on an unhandled error. Any other
    ``StructuredOutputError`` propagates.

    Raises:
        BuildFailure: ``ATTEMPTS`` submissions all had a problem; the message ends with the last one.
    """
    conversation = list(messages)
    issue = "no submission"
    name = output_type.__name__
    for _ in range(ATTEMPTS):
        try:
            result = await recorded_structured(model, conversation, output_type, sampling, record, {"output": name})
        except StructuredOutputError as error:
            if not isinstance(error.__cause__, CutReplyError):
                raise
            cut = error.completions[-1]
            issue = (
                f"the reply was cut off by the output limit after {cut.usage.completion_tokens} tokens, before "
                f"the {name} was complete. Send a shorter submission: fewer or smaller parts, no repetition."
            )
            cut_feedback: list[Message] = [
                {"role": "assistant", "content": f"{cut.content[:CUT_REPLY_CHARS]} ...[cut by the output limit]"},
                {"role": "user", "content": FEEDBACK_PROMPT.format(problem=issue)},
            ]
            conversation += cut_feedback
            continue
        found = await problem(result.value)
        if inspect.isawaitable(found):
            found = await found
        if found is None:
            return result.value
        issue = found
        feedback: list[Message] = [
            {"role": "assistant", "content": result.value.model_dump_json()},
            {"role": "user", "content": FEEDBACK_PROMPT.format(problem=issue)},
        ]
        conversation += feedback
    raise BuildFailure(f"{name}: no acceptable submission after {ATTEMPTS} requests; last problem: {issue}")


async def structured_until[T: BaseModel](
    context: RLEnvBuildContext,
    messages: Sequence[Message],
    output_type: type[T],
    problem: Callable[[T], Awaitable[str | None]],
) -> T:
    """``checked_structured`` with the step's model, sampling policy and call ledger."""
    return await checked_structured(
        context.model, context.sampling, call_ledger(context), messages, output_type, problem
    )


async def agent(context: RLEnvBuildContext, messages: Sequence[Message], tools: Sequence[AgentTool]) -> AgentRun:
    """Run the agent loop (``llm.agent.run_agent``) with ``tools`` under the build's agent limits.

    Each turn sends ``services.agent_limits.max_tokens`` as its output budget, the loop stops after
    ``services.agent_limits.max_turns`` replies, and the assistant turns are replayed as
    ``services.reasoning_replay`` says. Every turn and tool call is recorded under the running step.
    """
    limits = context.services.agent_limits
    return await run_agent(
        context.model,
        replace(context.sampling, max_tokens=limits.max_tokens),
        messages,
        tools,
        limits.max_turns,
        call_ledger(context),
        context.services.reasoning_replay,
    )


def web_calls(turns: Sequence[AgentTurn]) -> tuple[WebCall, ...]:
    """The web tool calls in ``turns`` that executed and returned content rather than an error."""
    return tuple(
        WebCall(tool=result.call.name, arguments=result.call.arguments, output_chars=len(result.output))
        for turn in turns
        for result in turn.tool_results
        if result.outcome is ToolOutcome.EXECUTED
        and result.call.name in WEB_TOOL_NAMES
        and not result.output.startswith(WEB_ERROR_PREFIX)
    )


async def research(context: RLEnvBuildContext, proposal: TaskProposal, index: int) -> ResearchFinding:
    """Look up the proposal's ``research[index]`` on the web with an agent (``web_search``, ``web_fetch``).

    A sources step calls this for every research item, concurrently with ``asyncio.gather``, and
    returns the findings in its ``Sources``. Under ``ResearchRequirement.ENFORCED``, ``run_build``
    fails a build whose sources lack a finding for some item, so a step lets this function's
    failure propagate.

    Raises:
        BuildFailure: ``index`` is out of range, the build has no web tools, the run did not end
            with an answer, or no web tool call executed and returned content.
    """
    items = proposal.header.research
    if not 0 <= index < len(items):
        raise BuildFailure(f"research[{index}]: the proposal lists {len(items)} research items")
    tools = context.services.web_tools
    if not tools:
        raise BuildFailure("web research is not configured for this build")
    item = items[index]
    prompt = RESEARCH_PROMPT.format(proposal=render(proposal), index=index, kind=item.kind, purpose=item.purpose)
    run = await agent(context, [{"role": "user", "content": prompt}], tools)
    if run.stop is not AgentStop.ANSWERED:
        raise BuildFailure(f"research[{index}]: the run ended {run.stop} after {len(run.turns)} turns")
    calls = web_calls(run.turns)
    if not calls:
        raise BuildFailure(f"research[{index}]: no web_search or web_fetch call executed and returned content")
    return ResearchFinding(
        index=index,
        kind=item.kind,
        purpose=item.purpose,
        notes=run.turns[-1].completion.content,
        web_calls=calls,
        turns=len(run.turns),
    )


def grader_package(
    grader: GraderDraft, fixtures: Fixtures, requirements: EnvironmentRequirements
) -> GraderPackage | DecisionGrader:
    """The grader ``grader`` describes.

    A decision_model grader is ``draft.decision_grader`` over the rubric and the reference reply, uncalibrated.
    An llm_as_judge grader is ``draft.judge_grader`` over the rubric and the reference reply.
    The generic kinds become an in-process verifyit grader. A script runs in a verifier machine: on
    the grader base image as a ``python_grader`` for a task without an image, or in a copy of the
    task image as a ``script_grader`` that receives the solver's workdir.
    """
    if grader.kind == "decision_model":
        assert grader.rubric is not None  # GraderDraft.rubric_kind
        return draft.decision_grader(grader.rubric, grader.reference_reply)
    if grader.kind == "llm_as_judge":
        assert grader.rubric is not None  # GraderDraft.rubric_kind
        return draft.judge_grader(grader.rubric, grader.reference_reply)
    if grader.kind == "exact":
        return draft.answer_grader(ExactSpec(expected=(grader.expected,), ignore_case=True, ignore_whitespace=True))
    if grader.kind == "numeric":
        return draft.answer_grader(
            NumericSpec(expected=grader.expected.strip(), tolerance_abs=grader.tolerance, tolerance_rel=0.0)
        )
    if grader.kind == "math":
        return draft.answer_grader(MathSpec(expected=grader.expected.strip()))
    if grader.kind == "mcq":
        return draft.answer_grader(McqSpec(expected=grader.expected.strip()))
    private = (*fixtures.private_files, *(task_file(f) for f in grader.private_files))
    environment = draft.grader_environment(requirements.docker_image)
    if requirements.docker_image is None:
        return draft.python_grader(
            grader.script, {}, environment=environment, answer_path=ANSWER_PATH, timeout=GRADER_TIMEOUT, files=private
        )
    return draft.script_grader(
        ("python3", f"{GRADER_RESOURCE_ROOT}/{GRADER_PROGRAM}"),
        VERDICT_REWARD,
        environment=environment,
        answer_path=ANSWER_PATH,
        timeout=GRADER_TIMEOUT,
        files=(draft.file(GRADER_PROGRAM, grader.script, mode=0o755), *private),
        cwd=WORKDIR,
        artifacts=(VerifierArtifact(source=WORKDIR, target=WORKDIR, kind=ArtifactKind.DIRECTORY),),
    )


def output_paths(package: GraderPackage | DecisionGrader, reference_files: Sequence[TaskResource]) -> tuple[str, ...]:
    """The files a solver is to leave in its machine for a grader program: the reference files' absolute paths.

    They are the task's ``output_paths``: what validation captures for the grader and what a control
    may write. A grader that reads the workspace through an artifact still names them, since the
    controls are written against them. Answer and decision graders read no files and need none.
    """
    if not isinstance(package, GraderPackage) or not isinstance(package.grader, ScriptGrader):
        return ()
    return tuple(sorted(f"/{resource.path}" for resource in reference_files))


def grader_name(package: GraderPackage | DecisionGrader) -> str:
    """The grader's type name, as a build digest reports it."""
    return type(package if isinstance(package, DecisionGrader) else package.grader).__name__


def graded_candidate(reply: str, files: Sequence[TaskResource], grade: GradeResult) -> GradedCandidate:
    """The ``Grader.prototyped`` entry of a candidate ``grade_candidate`` graded."""
    return GradedCandidate(
        reply=reply,
        files=tuple(sorted(files, key=lambda resource: resource.path)),
        status=grade.status,
        reward=grade.reward,
    )


def grade_diagnostics(grade: GradeResult) -> str:
    """A grade's error, detail and diagnostics as JSON, for a prompt.

    The detail holds what a grader program wrote beside its reward: its per-check results and
    anything else it reports there, often with nothing on stdout. The text puts the error and the
    detail first, then the grader's stdout, stderr and exit code. When the JSON runs past
    ``DIAGNOSTICS_CHARS``, the text keeps the first and the last half of that bound around a marker
    giving the count of characters cut, so a long stdout loses its middle while the detail at the
    start and the traceback ending stderr both stay. A detail value without a JSON form, which an
    in-process verifyit grade may hold, appears as its ``str``.
    """
    text = json.dumps({"error": grade.error, "detail": grade.detail, **grade.diagnostics}, default=str)
    if len(text) <= DIAGNOSTICS_CHARS:
        return text
    half = DIAGNOSTICS_CHARS // 2
    return f"{text[:half]} ...[{len(text) - 2 * half} characters cut]... {text[-half:]}"


def instruction_problem(text: str, fixtures: Fixtures, grader: Grader) -> str | None:
    """Why an instruction, system message included, must be rewritten, or ``None`` when it passes."""
    leaked = [value for value in grader.secret_values if value in text]
    if leaked:
        return f"the instruction contains graded answers: {leaked}"
    words = len(text.replace(grader.answer_contract, " ").split())
    if words > INSTRUCTION_WORD_LIMIT:
        return (
            f"the instruction has {words} words outside the answer contract; keep it within "
            f"{INSTRUCTION_WORD_LIMIT} and leave the data in the files"
        )
    inlined = sorted(
        {
            line.strip()
            for resource in fixtures.agent_files
            for line in resource_bytes(resource).decode(errors="replace").splitlines()
            if len(line.strip()) >= INLINED_LINE_CHARS and line.strip() in text
        }
    )
    if inlined:
        return f"the instruction copies solver-visible file lines; reference the file instead: {inlined[:3]}"
    by_hand = HAND_COMPUTATION.search(text)
    if by_hand:
        return f"the instruction asks for work by hand ({by_hand.group(0)!r}); the solver may use its machine's tools"
    grading = grading_language(text)
    if grading:
        return (
            f"the instruction refers to grading ({grading!r}); describe the format as a requirement, not grader "
            "behaviour"
        )
    return None


def grading_language(text: str) -> str | None:
    """The first word of ``text`` that describes the grading machinery (``GRADING_LANGUAGE``), or ``None``."""
    match = GRADING_LANGUAGE.search(text)
    return None if match is None else match.group(0)
