# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draft one task from one task proposal with a short agent session, and no model besides the agent.

``draft_task`` gives the agent the proposal, the example build program and the TaskSpec references, and two tools:
``submit``, which runs only the checks of ``taskforge.agentic.tools`` that need no model, and ``finish``. Nothing
grades, rolls out or reviews the task: a later step validates the drafts. A draft keeps its build program and the
workspace that the program reads, so ``task_instances`` can make more tasks of its kind from other task ids.
"""

import json
import math
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError, validate_call
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness, ToolCallResult, ToolContext
from rolloutengine.models import Model, Rollout, RolloutInterrupted, RolloutLimits, TaskRolloutSpec, TraceMode
from shellbox.machine import Backend, Machine, MachineFactory, PackageSpec
from taskcompendium.grading import grade_answer
from taskcompendium.models import (
    AnswerType,
    Boxed,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    FunctionDefinition,
    Grader,
    GradingAttempt,
    NoGrader,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.grade import numeric_tolerance
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.numeric import numeric_literal
from verifyit.spec import (
    RUBRIC_CHECKLIST,
    RUBRIC_LABELS,
    ExactSpec,
    JudgeSpec,
    MathSpec,
    McqSpec,
    NumericSpec,
    StructuredExactSpec,
    spec_from_table,
)

from taskforge.agentic import grader_examples
from taskforge.agentic.agent import SHELL_OUTPUT_BYTES, example_program, verifyit_reference
from taskforge.agentic.prompts import DRAFT_PROMPT, DRAFT_USER_PROMPT, NO_SOLVER_MACHINE, SOLVER_MACHINES
from taskforge.agentic.tools import (
    COMMAND_SEMANTICS,
    SOLVER_HARNESS,
    WORKDIR,
    ToolError,
    keyword_grader_problems,
    machine_runtime,
    malformed_files,
    needs_machine,
    program_problems,
    read_task,
    rollout_tokens,
    solver_spec,
    unsupported,
    workspace_snapshot,
)

DRAFT_TOOLS = "taskforge-drafts:v1"
"""The action interface of the draft tools, in the agent task's ``tool_providers``."""
DRAFT_HARNESS = "taskforge-draft"
DRAFT_REVISION = "5"
"""The revision of the draft agent task in the ``Source`` of each session."""
DRAFT_MAX_ANSWER_SHARE = 0.625
"""The largest share of the variants of a draft's program that can have one reference answer. It is looser than
the agentic tools' share, so that a task with a small answer space passes: 8 random variants of a four-option task
put 6 on one letter about 1% of the time. The later validation step judges whether such a task is worth training
on."""
VERIFIER_BACKEND = Backend.LOCAL
"""The machines of graders that run in an environment: a bubblewrap sandbox that installs the grader's packages."""
VERIFIER_ROOT = "/tests"
"""Where the engine stages each file ``P`` of ``resources.verifier``: at ``/tests/P``."""
MIN_LEAKED_REFERENCE_CHARS = 40
"""The length from which a judge reference that the solver's prompt or files contain counts as given away."""
MIN_LEAKED_LEAVES = 2
"""The (field, value) pairs of the reference that an example object in the prompt may share before it gives the
answer away."""
PINNED_REQUIREMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9.+!_-]+")
OWN_SCALE = re.compile(
    r"out of 100|\b0\s*(-|\u2013|to)\s*100\b|points?\s*/\s*100|weighted score|\bscore\b[^.]{0,20}\bweighted\b"
    r"|\bweighted (sum|total|average) of (the )?(passed|rubric|criteri|items?|points|scores)"
    r"|\bcap(s|ped)?\b[^.]{0,60}\bat\s+0?\.\d|\"score\"\s*:|\bscore\s*:\s*<"
    r"|\boutput format\b|\b(return|reply with|respond with) (only )?(a|one) json\b",
    re.IGNORECASE,
)
"""Words of a judge question that set its own scale or reply format: a ``reference`` or ``checklist`` judge must end
with verifyit's ``SCORE`` line, so such a question leaves every answer ungraded."""
MAX_CHECKLIST_CRITERIA = 8
# The ways a prompt asks a solver to mark its final answer, and the prose a solver writes before it.
BOXED_ASKED = re.compile(r"\\boxed")
ANSWER_LINE_ASKED = re.compile(r"\banswer:", re.IGNORECASE)
ANSWER_PROSE = "I worked through the problem above."
"""The criteria of a checklist judge: each is one judge call, in sequence, so a long checklist grades one answer in
minutes."""
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
DECIMALS = re.compile(
    r"\b(\d|one|two|three|four|five|six|seven|eight)\s+(?:decimal\s+(?:places?|digits?)|decimals)\b", re.IGNORECASE
)
"""How a prompt asks for an answer to a number of decimals, for example "to four decimal places"."""


class Environment(StrEnum):
    """What a proposal's solver works in. Container proposals bring an image of their own, which drafts cannot run."""

    REASONING = "reasoning"
    SHELLSIM = "shellsim"
    CONTAINER = "container"


class SolverMachine(StrEnum):
    NONE = "none"
    SHELLSIM = "shellsim"
    CPYTHON = "cpython"
    PACKAGES = "packages"


SOLVER_BACKENDS = {
    SolverMachine.NONE: Backend.SHELLSIM,
    SolverMachine.SHELLSIM: Backend.SHELLSIM,
    SolverMachine.CPYTHON: Backend.LOCAL,
    SolverMachine.PACKAGES: Backend.LOCAL,
}
"""The solver backend of each solver machine. A task of a proposal with no machine runs on none, so its backend
only sets the command semantics that ``submit`` expects of a task that declares a shell anyway."""


@dataclass(frozen=True)
class DraftProposal:
    """The header fields of a curriculum proposal that a draft uses, and the proposal document as written."""

    id: str
    environment: Environment
    solver_machine: SolverMachine
    verification: str
    source_hash: str
    """The hash of the capability record that the proposal was written from."""
    document: str = field(repr=False)

    @classmethod
    def from_json(cls, text: str) -> "DraftProposal":
        header = json.loads(text)["header"]
        environment = Environment(header["environment"])
        if environment is Environment.CONTAINER:
            raise ValueError(f"{header['id']}: container proposals have no solver machine for drafts")
        machine = SolverMachine(header["solver_machine"])
        return cls(header["id"], environment, machine, header["verification"], header["source"]["hash"], text)

    @property
    def solver_backend(self) -> Backend:
        return SOLVER_BACKENDS[self.solver_machine]

    @property
    def task_id(self) -> str:
        """A file-safe id for the draft: the proposal id ``capability/slot`` with ``--`` for the slash."""
        return self.id.replace("/", "--")


@dataclass(frozen=True)
class DraftConfig:
    """The models, machines and limits of draft sessions. One config can serve many proposals at the same time."""

    agent_model: Model
    workspace_factory: MachineFactory
    """The agent's machine: a ``LocalMachineFactory`` (bubblewrap) whose mounts give its shell Python with
    taskcompendium and verifyit."""
    solver_factories: Mapping[Backend, MachineFactory]
    """The solver machines that ``RolloutEngine.check`` checks a task against, by backend, with the
    ``VERIFIER_BACKEND`` machines of graders that run in an environment. No solver or grader runs."""
    judge: JudgeConnection
    """The judge that a judge grader would use. ``check`` needs it, and no draft calls it."""
    agent_limits: RolloutLimits
    solver_limits: RolloutLimits
    command_timeout: float


@dataclass(frozen=True)
class Draft:
    """A task that ``submit`` accepted, its build program, and a gzipped tar of the workspace that the program
    reads."""

    task: TaskSpec
    program: str
    workspace: bytes = field(repr=False)


@dataclass(frozen=True)
class DraftResult:
    proposal: DraftProposal
    draft: Draft | None
    report: str | None
    """The agent's report from ``finish``, when it gave up on the proposal."""
    rollout: Rollout
    agent_tokens: int
    submits: int
    failure: str | None
    """The failed operation and its cause, when the session failed."""


def _tool(name: str, description: str, properties: dict[str, Any]) -> FunctionDefinition:
    parameters = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return FunctionDefinition(name=name, description=description, parameters=parameters)


TOOL_DEFINITIONS = (
    _tool(
        "submit",
        "Submit the task with the build program that wrote it. The checks need no model.",
        {
            "path": {"type": "string", "description": "The path of the TaskSpec JSON file in the workspace."},
            "program": {
                "type": "string",
                "description": (
                    "The path of the build program in the workspace: `python3 <program> <task id> <output path>` "
                    "(or `sh <program> ...` for a .sh file) must write this TaskSpec again."
                ),
            },
        },
    ),
    _tool(
        "finish",
        "Stop without a task, with a report on why the proposal cannot become one.",
        {"report": {"type": "string"}},
    ),
)


class DraftTools:
    """The tool provider of one draft session, and its result: ``draft``, ``report`` and ``submits``."""

    def __init__(self, proposal: DraftProposal, config: DraftConfig):
        self.proposal = proposal
        self.config = config
        backend = proposal.solver_backend
        self.backend = backend
        backends = {backend, VERIFIER_BACKEND}
        self.solver_engine = RolloutEngine(
            {name: config.solver_factories[name] for name in backends},
            {SOLVER_HARNESS: ShellHarness(command_timeout=config.command_timeout)},
        )
        self.draft: Draft | None = None
        self.report: str | None = None
        self.submits = 0

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return TOOL_DEFINITIONS

    async def open(self, context: ToolContext) -> "DraftSession":
        if context.machine is None:
            raise ValueError("The draft tools need the agent's workspace machine")
        return DraftSession(self, context.machine)

    def problems(self, task: TaskSpec) -> list[str]:
        """What keeps ``task`` from being a draft: the checks of the agentic tools, with judges, script graders and
        graders with an environment allowed, and ``draft_task_problems``."""
        limits = self.config.solver_limits
        spec = solver_spec(task, self.backend, limits, judge=self.config.judge, verifier_backend=VERIFIER_BACKEND)
        semantics = COMMAND_SEMANTICS[self.backend]
        problems = unsupported(task, self.solver_engine, spec, semantics, draft_grader_problems) + malformed_files(task)
        try:
            return problems + draft_task_problems(task, self.proposal.solver_machine)
        except (TypeError, ValueError) as error:
            return [*problems, f"The grader is invalid: {error}"]


def draft_grader_problems(grader: Grader) -> list[str]:
    """The grader rule of drafts: any grader but none, and no keyword grader."""
    if isinstance(grader, NoGrader):
        return ["The task needs a grader"]
    return keyword_grader_problems(grader)


def draft_task_problems(task: TaskSpec, solver_machine: SolverMachine) -> list[str]:
    """The defects of a draft that validation found and that need no model to see. ``program_problems`` applies
    them to every variant too, so a program cannot write them for other task ids."""
    return [
        *machine_problems(task, solver_machine),
        *script_path_problems(task),
        *judge_problems(task),
        *precision_problems(task),
        *unpinned_package_problems(task),
        *leaked_reference_problems(task),
        *example_answer_problems(task),
        *answer_form_problems(task),
    ]


def machine_problems(task: TaskSpec, solver_machine: SolverMachine) -> list[str]:
    if solver_machine is SolverMachine.NONE and needs_machine(task):
        return ["The proposal's solver_machine is none: the task must give the solver no shell and no files"]
    if solver_machine is not SolverMachine.NONE and not needs_machine(task):
        return [f"The proposal's solver_machine is {solver_machine}: the task must give the solver that machine"]
    return []


def script_path_problems(task: TaskSpec) -> list[str]:
    """The files under ``/tests`` that a script grader's command names and that no verifier resource stages."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader):
        return []
    staged = [f"{VERIFIER_ROOT}/{resource.path}" for resource in task.resources.verifier]
    return [
        f"The grader runs {argument}, but resources.verifier stages no file there: the engine stages a verifier "
        f"file P at {VERIFIER_ROOT}/P, so stage {argument.removeprefix(VERIFIER_ROOT + '/')!r}"
        for argument in grader.argv
        if argument.startswith(VERIFIER_ROOT + "/")
        and not any(path == argument or path.startswith(argument.rstrip("/") + "/") for path in staged)
    ]


def _judge_spec(task: TaskSpec) -> JudgeSpec | None:
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return None
    spec = spec_from_table({"mode": grader.mode, **grader.parameters})
    return spec if isinstance(spec, JudgeSpec) else None


def judge_problems(task: TaskSpec) -> list[str]:
    spec = _judge_spec(task)
    if spec is None:
        return []
    problems = []
    # A thinking judge can run past its first budget of 8,192 tokens, and without a retry that answer is never graded.
    if spec.incomplete_retry_tokens < grader_examples.JUDGE_RETRY_TOKENS:
        problems.append(
            f"Set the judge's incomplete_retry_tokens to {grader_examples.JUDGE_RETRY_TOKENS}: a judge reply that "
            "runs past max_completion_tokens is otherwise never graded"
        )
    scale = OWN_SCALE.search(spec.question) if spec.rubric != RUBRIC_LABELS else None
    if scale is not None:
        problems.append(
            f"The judge question sets its own scale or reply format ({scale.group(0)!r}). "
            f"A {spec.rubric} judge scores only 0, 0.5 or 1 and verifyit adds the reply format itself, so the "
            "question must give no format, scale, weights or caps. For weighted items or caps, use the labels "
            "rubric with its own prompt_template and label_scores, or a checklist with one yes/no criterion each"
        )
    if spec.rubric == RUBRIC_CHECKLIST and len(spec.criteria) > MAX_CHECKLIST_CRITERIA:
        problems.append(
            f"The checklist has {len(spec.criteria)} criteria, and each is one judge call in sequence. Keep at most "
            f"{MAX_CHECKLIST_CRITERIA}, each a decisive check of a number, a decision or a required element, or "
            "grade a long rubric with one labels call and a banded prompt_template"
        )
    return problems


def precision_problems(task: TaskSpec) -> list[str]:
    """A numeric tolerance that does not match the decimals that the prompt asks for: the correctly rounded
    reference must pass, and its neighbors one unit away in the last decimal must fail."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return []
    spec = spec_from_table({"mode": grader.mode, **grader.parameters})
    if not isinstance(spec, NumericSpec):
        return []
    prompt = " ".join(str(event.content) for event in task.context.events if isinstance(event, TextMessage))
    asked = {NUMBER_WORDS.get(match.group(1).lower()) or int(match.group(1)) for match in DECIMALS.finditer(prompt)}
    if len(asked) != 1:
        return []
    [decimals] = asked
    expected = numeric_literal(spec.expected)
    tolerance = numeric_tolerance(spec)
    unit = Fraction(1, 10**decimals)
    rounded = math.floor(expected / unit + Fraction(1, 2)) * unit
    # A reference halfway between two roundings lets both pass at half a unit, which is right.
    neighbors = [abs(rounded + step - expected) for step in (unit, -unit)]
    if abs(rounded - expected) <= tolerance and all(gap > tolerance or gap == unit / 2 for gap in neighbors):
        return []
    return [
        f"The prompt asks for {decimals} decimals, so only {float(rounded)} may pass: set tolerance_abs to "
        f"0.5 * 10**-{decimals} around the unrounded value and tolerance_rel to 0 (now {spec.tolerance_abs} and "
        f"{spec.tolerance_rel})"
    ]


def unpinned_package_problems(task: TaskSpec) -> list[str]:
    """The PyPI requirements of the solver's or the grader's machine without one exact version."""
    environments = [task.environment_requirements]
    if isinstance(task.grader, (ScriptGrader, VerifyitGrader)) and task.grader.environment is not None:
        environments.append(task.grader.environment)
    requirements = [
        requirement
        for environment in environments
        if environment.package_spec is not None
        for requirement in environment.package_spec.pypi_packages or ()
    ]
    return [
        f"Pin the package {requirement!r} to one version, as name==version, so that the machine is fixed"
        for requirement in requirements
        if PINNED_REQUIREMENT.fullmatch(requirement.strip()) is None
    ]


def _collapsed(text: str) -> str:
    return " ".join(text.split()).lower()


def _solver_text(task: TaskSpec) -> str:
    """The prompt and the solver's inline files, which the solver sees."""
    messages = [event.content for event in task.context.events if isinstance(event, TextMessage)]
    files = [resource_bytes(resource).decode(errors="replace") for resource in task.resources.worker]
    return "\n".join(str(text) for text in (*messages, *files))


def leaked_reference_problems(task: TaskSpec) -> list[str]:
    """A long judge reference that the solver's prompt or files contain: the solver sees its own answer."""
    spec = _judge_spec(task)
    if spec is None:
        return []
    seen = _collapsed(_solver_text(task))
    leaked = [
        reference
        for reference in spec.references
        if len(_collapsed(reference)) >= MIN_LEAKED_REFERENCE_CHARS and _collapsed(reference) in seen
    ]
    if not leaked:
        return []
    return [
        "A judge reference is a passage of the prompt or of the solver's files, so the solver sees it. The reference "
        "must be the answer, which the solver does not see, not the task's text"
    ]


def _reference_text(spec: object) -> str | None:
    """The reference answer of a spec that compares a text answer without a model, as a solver would write it."""
    if isinstance(spec, ExactSpec):
        return ", ".join(spec.expected)
    if isinstance(spec, McqSpec | NumericSpec | MathSpec):
        return spec.expected
    return None


def _answer_forms(task: TaskSpec, reference: str) -> list[str]:
    """The reference alone, and the reference written each way the prompt asks for: after prose, in ``\\boxed{}``
    or on a last ``Answer:`` line."""
    prompt = " ".join(str(event.content) for event in task.context.events if isinstance(event, TextMessage))
    forms = [reference]
    if BOXED_ASKED.search(prompt):
        forms.append(f"{ANSWER_PROSE}\n\n\\boxed{{{reference}}}")
    if ANSWER_LINE_ASKED.search(prompt):
        forms.append(f"{ANSWER_PROSE}\n\nAnswer: {reference}")
    return forms


def answer_form_problems(task: TaskSpec) -> list[str]:
    """The reference fails its own grader when written the way the prompt asks. An exact or mcq grader on a
    plain_text answer compares the whole final message, so it fails a right answer after prose or in a box."""
    grader = task.grader
    if (
        not isinstance(grader, VerifyitGrader)
        or grader.environment is not None
        or task.answer_type not in (AnswerType.TEXT, AnswerType.NUMBER)
        or not isinstance(task.answer_format, PlainText | Boxed)
    ):
        return []
    reference = _reference_text(spec_from_table({"mode": grader.mode, **grader.parameters}))
    if reference is None:
        return []
    failing = []
    for form in _answer_forms(task, reference):
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=form)))
        )
        if grade_answer(task, attempt).reward != 1:
            failing.append(form)
    if not failing:
        return []
    if failing[0] == reference:
        return [
            f"The reference {reference!r} fails its own grader. Several expected entries of an exact grader make the "
            "answer a list of that many items, split on commas and newlines, not a set of alternatives: for one "
            "answer, give one entry and define its format in the prompt"
        ]
    return [
        f"The reference fails its own grader when written as {failing[0]!r}, which the prompt asks for. With "
        f"answer_format {task.answer_format.kind}, an exact or mcq grader compares the whole final message. Use "
        "answer_format boxed and ask for \\boxed{...}, or ask for the bare answer alone. For a list, give expected "
        "one entry per item"
    ]


def _leaves(value: object) -> Iterator[tuple[str, str]]:
    """The (field, value) pairs of a JSON value, at any depth, with each value as JSON. Booleans and nulls say too
    little about an answer to count."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                if isinstance(item, list) and item and all(isinstance(x, (str, int, float)) for x in item):
                    yield str(key), json.dumps(item, sort_keys=True)
                yield from _leaves(item)
            elif isinstance(item, (str, int, float)) and not isinstance(item, bool):
                yield str(key), json.dumps(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)


def _json_objects(text: str) -> Iterator[object]:
    """The JSON objects written in ``text``, for example the format example of a prompt."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        yield value
        start = text.find("{", end)


def _reference_values(task: TaskSpec) -> list[object]:
    """The structured reference of the grader: a StructuredExactSpec's expected value, or the JSON files of a
    script grader."""
    grader = task.grader
    if isinstance(grader, VerifyitGrader):
        spec = spec_from_table({"mode": grader.mode, **grader.parameters})
        return [spec.expected] if isinstance(spec, StructuredExactSpec) else []
    if not isinstance(grader, ScriptGrader):
        return []
    values = []
    for resource in task.resources.verifier:
        if resource.path.endswith(".json"):
            try:
                values.append(json.loads(resource_bytes(resource)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    return values


def example_answer_problems(task: TaskSpec) -> list[str]:
    """An example object in the prompt that shares field values with the reference: a solver copies them."""
    reference = {leaf for value in _reference_values(task) for leaf in _leaves(value)}
    if not reference:
        return []
    messages = [str(event.content) for event in task.context.events if isinstance(event, TextMessage)]
    for example in (value for message in messages for value in _json_objects(message)):
        shared = sorted(set(_leaves(example)) & reference)
        if len(shared) >= MIN_LEAKED_LEAVES:
            fields = ", ".join(f"{key}={value}" for key, value in shared[:4])
            return [
                f"An example in the prompt holds values of the reference answer ({fields}). Write format examples "
                'with placeholders such as "<tier>", or with values that are not in the answer'
            ]
    return []


class DraftSession:
    """The tools of one draft attempt, on the agent's workspace machine."""

    def __init__(self, tools: DraftTools, machine: Machine):
        self.tools = tools
        self.machine = machine
        handlers: dict[str, Callable[..., Awaitable[ToolCallResult]]] = {"submit": self.submit, "finish": self.finish}
        assert set(handlers) == {tool.name for tool in TOOL_DEFINITIONS}
        self.handlers = {name: validate_call(handler) for name, handler in handlers.items()}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        handler = self.handlers.get(name)
        if handler is None:
            return _reply({"error": f"Unknown tool: {name}"})
        try:
            return await handler(**arguments)
        except ValidationError as error:
            return _reply({"error": f"Invalid arguments of {name}: {error}"})
        except ToolError as error:
            return _reply({"error": str(error)})

    async def close(self) -> None:
        pass

    async def submit(self, path: str, program: str) -> ToolCallResult:
        self.tools.submits += 1
        timeout = self.tools.config.command_timeout
        task = await read_task(self.machine, path, timeout)
        problems = self.tools.problems(task)
        if not problems:
            problems = await program_problems(
                self.machine, program, task, self.tools.problems, timeout, max_answer_share=DRAFT_MAX_ANSWER_SHARE
            )
        if problems:
            return _reply({"accepted": False, "problems": problems})
        self.tools.draft = Draft(task, program, await workspace_snapshot(self.machine))
        return _reply({"accepted": True}, done=True)

    async def finish(self, report: str) -> ToolCallResult:
        self.tools.report = report
        return _reply({"finished": True}, done=True)


async def draft_task(proposal: DraftProposal, config: DraftConfig) -> DraftResult:
    """Draft the task of ``proposal``. A session without a draft gives the agent's report or its failure."""
    tools = DraftTools(proposal, config)
    harness = ShellHarness(
        command_timeout=config.command_timeout, output_limit_bytes=SHELL_OUTPUT_BYTES, tools={DRAFT_TOOLS: tools}
    )
    workspace = config.workspace_factory
    engine = RolloutEngine({workspace.backend: workspace}, {DRAFT_HARNESS: harness})
    spec = TaskRolloutSpec(
        task=draft_agent_task(proposal, workspace.backend, config.agent_limits),
        harness=DRAFT_HARNESS,
        task_machine=machine_runtime(workspace.backend),
        verifier_machine=None,
        limits=config.agent_limits,
        trace=TraceMode.MESSAGES,
    )
    failure = None
    try:
        rollout = (await engine.run_task(spec, config.agent_model)).rollout
    except RolloutInterrupted as error:
        rollout = error.rollout
        failure = f"{error.operation}: {error.rollout.failure}"
    return DraftResult(proposal, tools.draft, tools.report, rollout, rollout_tokens(rollout), tools.submits, failure)


def draft_agent_task(proposal: DraftProposal, workspace_backend: Backend, agent_limits: RolloutLimits) -> TaskSpec:
    """The agent's own task: the prompts, and a workspace with the proposal, the example program and the schema."""
    backend = proposal.solver_backend
    machine = NO_SOLVER_MACHINE if proposal.solver_machine is SolverMachine.NONE else SOLVER_MACHINES[backend]
    system = DRAFT_PROMPT.format(
        solver_machine=machine, shell_output_kib=SHELL_OUTPUT_BYTES // 1024, turns=agent_limits.max_turns
    )
    workspace = WORKDIR.relative_to("/")
    files = (
        inline_resource(f"{workspace}/proposal.json", proposal.document.encode()),
        inline_resource(f"{workspace}/example_build.py", example_program(backend)),
        inline_resource(
            f"{workspace}/reference/taskspec.schema.json", json.dumps(TaskSpec.model_json_schema(), indent=1).encode()
        ),
        inline_resource(f"{workspace}/reference/verifyit_specs.md", verifyit_reference().encode()),
        inline_resource(f"{workspace}/reference/grader_examples.py", Path(grader_examples.__file__).read_bytes()),
    )
    return TaskSpec(
        id=f"taskforge-draft-{proposal.task_id}",
        context=ConversationInput(
            events=(
                TextMessage(role="system", content=system),
                TextMessage(role="user", content=DRAFT_USER_PROMPT.format(proposal_id=proposal.id)),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=COMMAND_SEMANTICS[workspace_backend],
            # An empty package spec runs the shell on what the factory mounts, for example a Python environment.
            package_spec=PackageSpec(),
            working_directory=str(WORKDIR),
            tool_providers={"taskforge": ProviderRequirement(action_interface=DRAFT_TOOLS, initial_state={})},
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="submit decides whether the agent made a draft"),
        source=Source(
            dataset="taskforge-agentic-drafts",
            revision=proposal.source_hash,
            row=proposal.id,
            importer_revision=DRAFT_REVISION,
        ),
        resources=ResourceGroups(worker=files),
    )


def _reply(value: Mapping[str, Any], *, done: bool = False) -> ToolCallResult:
    return ToolCallResult(json.dumps(value, default=str), done=done)
