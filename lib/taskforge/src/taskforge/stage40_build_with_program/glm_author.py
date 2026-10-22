# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An author that writes a build program for a proposal with one structured GLM call.

``GlmAuthor`` sends the rules (``AUTHOR_SYSTEM``), then the proposal, the SDK reference
(``sdk_reference``, generated from signatures and docstrings) and the standard template's source.
The model replies with a ``ProgramSubmission``, whose validator loads the source with
``program_sdk.load_program``, so a program the loader refuses goes back to the model in the client's
structured repair request, with the loader's message. A reply cut off by the output limit gets no
repair: the call raises ``StructuredOutputError`` at once.

The author lives with the build stage because its prompt documents the build SDK and the template.
It keeps each request as sent, ending with the field instruction ``complete_structured`` appends,
and its completions under ``<workpiece_dir>/author/<n>/``, including the completions of a call that
raised ``StructuredOutputError``. It writes the accepted source to ``<workpiece_dir>/program.py`` before
it loads the program from it. For a revision it reads that file back, since a ``Revision`` carries
no source, and continues the conversation with the previous program as the model's turn and the
revision as the next user turn.
"""

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, field_validator
from rigging.filesystem.atomic import atomic_rename
from shellbox.machine import PackageSpec
from taskcompendium import models as task_models
from taskcompendium.runtime import resources
from verifyit import spec as verifyit_spec

from taskforge.llm.agent import shell_tool
from taskforge.llm.client import Completion, ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_structured
from taskforge.llm.structured import StructuredOutputError, schema_instruction
from taskforge.sandbox.images import GRADER_BASE_IMAGE, DockerBuild
from taskforge.spec import decision, draft, rubric, verdict
from taskforge.stage10_proposal.model import TaskProposal, render
from taskforge.stage30_author import program as program_module
from taskforge.stage30_author.author import PROGRAM_FILE, AuthoredProgram, Revision
from taskforge.stage30_author.loader import ALLOWED_IMPORTS, GRADER_STEP
from taskforge.stage40_build_with_program import context as context_module
from taskforge.stage40_build_with_program import outputs, sandbox_effects, sdk
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.step_store import SDK_VERSION
from taskforge.stage40_build_with_program.template import standard

AUTHOR_DIR = "author"
REQUEST_FILE = "request.json"
COMPLETIONS_FILE = "completions.json"
RECORD_FILE = "record.json"

AUTHOR_SYSTEM = """\
You write build programs for Taskforge. A build program is one Python module that turns an accepted \
task proposal into a TaskCompendium TaskSpec, using the build SDK described below. Rules:
- The module defines, at top level, only imports, output models (classes), steps, the sink and, as \
its last statement, `PROGRAM = rl_env_build_program(<every step but the sink>, sink=assemble)`. \
Define each output model above the steps that return it. No helper functions, constants or other \
statements at top level: a step's memo key covers its own source and its output model alone, so \
whatever a step reads goes in its body or its inputs, or comes from an SDK function. Put prompts and \
checks inside the step that uses them; a nested function inside a step is fine.
- A step is `@rl_env_build_step` (the bare decorator, nothing else) on `async def name(context: \
RLEnvBuildContext, <inputs>) -> Output`. The parameters after the context are the step's inputs, \
each named after the step whose output it receives, or `proposal` for the TaskProposal. The step's \
name is its function name. The return annotation is required: the output is stored as JSON and \
decoded with it, so outputs are dataclasses, pydantic models, tuples, str, int or float.
- The final step is the sink: `@rl_env_build_sink` (in place of `@rl_env_build_step`) on `async def \
assemble(context: RLEnvBuildContext, <inputs>) -> TaskSpec`, returning the TaskSpec that \
`draft.assemble` built. Pass it to `rl_env_build_program` as the keyword `sink=assemble`, not among \
the steps. Every other step must be an input of another step or of the sink.
- Use the standard step names: `sources`, `fixtures`, `requirements`, `grader`, `instructions`, and \
`assemble` for the sink. The program needs a step named `{grader}` returning `Grader`.
- The grader step checks its own grader before it returns: it grades the reference answer with \
`grade_candidate` and raises `BuildFailure` (through `structured_until`, which sends each problem \
back to the model first) when the reference is not `Outcome.GRADED` with at least `FULL_CREDIT` or \
when a reward lies outside the grader's score range (`reward_in_range`). It declares \
`rejection_ceiling`: the highest reward a submission wrong in substance can earn, 0 unless the \
proposal gives partial credit. The library grades the empty answer after every build and fails the \
build when it is `Outcome.GRADED` above that ceiling, so the program need not grade the empty answer \
itself. The grader step keeps every candidate it graded in `Grader.prototyped` (`graded_candidate`). \
Validation writes the controls later and refuses one that repeats a prototyped candidate other than \
the reference and the empty answer, so prototype wrong answers only on candidates you invent for \
that purpose.
- When the proposal lists research, the program needs a step named exactly `sources`, annotated \
`-> Sources`, that calls `research(context, proposal, i)` for every item and returns the findings; \
`run_build` refuses a program without it ("a build that researches its proposal needs a 'sources' step \
returning Sources"), even when you already know the facts. Never catch its `BuildFailure`.
- Build the grader of the kind the proposal's `verification` names; `run_build` refuses a draft whose \
grader is another kind, and refuses a composite proposal outright (no composite grader exists yet). \
decision_model, the usual choice for a reply a person reads: `draft.decision_grader(rubric, reference)`. \
You state a `Rubric` (the deliverable, and one `RubricItem` per property a correct reply has: its meaning \
with the reference values, tolerances and units, the equivalent forms it accepts, how negated or hedged \
statements count, its weight, and any gate or cap) and the reference reply, nothing else: a decision model \
of fixed strength compares the visible output of the solver's final turn with the reference against the rubric (the \
`state` template, `SOLVER_TURN_STATE`; pass another only deliberately), and which model is the host's \
choice. Calibrate it with `calibrate_decision_grader(context, grader, variants)` on three equivalent \
variants of the reference; when its `problem` is set, either the rubric rejects a correct answer or a \
variant is not equivalent. Its instruction states the deliverable the way a requester would, with no \
answer block or labelled fields. llm_as_judge, for a reply whose prose is the product: \
`draft.judge_grader(rubric, reference)` with the same `Rubric` and reference reply: the judge is the \
host's choice. A judge runs `import \
verifyit` in a verifier machine, and `GRADER_BASE_IMAGE` has no verifyit ("No module named \
'verifyit'"); `grade_candidate` calls no model, so a judge can never be prototyped. Never keyword or \
regex checks standing in for a judgement. answer_in_box: `draft.answer_grader` with a verifyit mode \
that grades in process (exact, numeric, mcq, math, ifeval, json_schema, xml_elements, csv_columns, \
structured_exact, predicted_action), for one value in an unambiguous format. code, when the proposal's \
deliverable has a strict structure the requester specified (a JSON record, CSV columns, a config file) and \
every graded part is deterministic, or when grading runs something (solver-written code, executing files); \
never keyword, regex or length checks on prose: a deterministic grader program in a separate verifier \
machine: \
`draft.python_grader(script, config, environment=draft.grader_environment(image), answer_path=..., \
timeout=...)` runs `python3 /tests/grade.py` after the attempt, in the task image for a task with one \
(`publish_image(context, DockerBuild(...))`), otherwise on Taskforge's grader-base image (CPython \
3.12, standard library only). It reads the extracted final answer at `/app/answer.txt` \
(`answer_path=draft.ANSWER_PATH`), the captured `output_paths` at their own absolute paths and its \
private data under `/tests`, never stdin, and writes its verdict to `verdict.GRADER_VERDICT_PATH`: \
`{{"reward": r, "detail": {{"checks": [{{"name": ..., "passed": ..., "weight": ...}}, ...]}}}}`, one \
check per independently gradable property, exiting 0 for every candidate. No grader runs in the \
solver's machine. A private grader file must not have the same bytes as any solver-visible file: \
`assemble` refuses it ("Private verifier file content is also agent-visible"). The verifier machine \
already holds the solver-visible files at their own paths, so the grader reads them there; never copy \
one into the grader's `files`.
- ShellSim, the image-less task machine, has `sh`, coreutils, `awk`, `jq` and `python3` (CPython \
3.13, standard library only; `base64` works, `sort -g` does not), no pip and no network, a compute \
budget of a few seconds of Python per machine and a virtual clock that advances only through `sleep`. \
A task whose solution needs more \
computation than that, or a real clock, runs on `draft.requirements(image=GRADER_BASE_IMAGE, ...)`, \
and one that needs PyPI packages but no system packages runs on `draft.requirements(image=None, \
packages=("numpy", ...), ...)`: CPython 3.12 with those packages, installed from wheels when the \
machine starts, with no image to build. Publish a task image only for apt packages or another base \
image. Never ask the solver to compute by hand.
- Data and fixtures go in workspace files. The instruction stays within 500 words, never inlines \
tables or file contents, and never contains an answer the grader checks. It describes the deliverable and \
its format, never the grading: no grader, checker, parser, autograder, evaluator, rubric or tests; nothing \
about what is graded, scored, checked, read or ignored; no credit, weights or tolerances; never \
"machine-checked". State a format as a requirement ("end with exactly this block", "give values to 3 \
significant figures", "either sign convention, normalized so ..."), not as grader behaviour ("only the block \
is graded", "answers within 1% are accepted", "prose outside is ignored"). Telling the solver how it is \
scored leaks the reward surface and invites writing for the checker instead of doing the task. The \
proposal's Task section may describe grading; do not carry that into the instruction. Every literal the \
answer format requires (headings, field names, units) is in the instruction. RolloutEngine adds the \
shell tool and the submission instruction itself.
- A `TaskResource` holds a `path` and a `source`, an `InlineFile` whose `content_base64` holds the \
bytes. Read the bytes with `resource_bytes(resource)`, or list files as text with `files_text`. \
`EnvironmentRequirements` names a task image `docker_image`. The SDK reference below lists every \
field of both types; use no other attribute. Build a task machine with `draft.requirements` alone: an \
image, packages and a Docker build are exclusive ("mutually exclusive"), and a `DockerBuild` reaches a \
machine only through the pinned reference `publish_image` returns.
- Use model calls (`structured`, `structured_until`, `agent`) for content you cannot compute; compute \
exact values in Python inside steps. When the host fails, the SDK raises \
`BuildInfrastructureFailure`; let it propagate.
- The program runs with restricted builtins (no `exec`, `eval`, `compile`, `open`, `globals`, \
`vars`) and may import only from: {imports}. Import a dotted module's names with `from <module> \
import <name>`: `import taskcompendium.grader` is refused ("use 'from taskcompendium.grader import ...' \
rather than 'import taskcompendium.grader'"), and so is every module outside that list (`inspect`, `os`, \
`verifyit` and `verifyit.judge`, `shellbox`). The SDK constants (`GRADER_TIMEOUT`, `WORKDIR`, `WORKSPACE`, \
`FULL_CREDIT`, ...) live in `taskforge.stage40_build_with_program.sdk`; `draft` has only the names its section of the \
reference lists. Use only the names, keyword arguments and fields the reference shows. It may import steps from \
`taskforge.stage40_build_with_program.template.standard` and pass them to `rl_env_build_program` unchanged.
The template below is the standard program; adapt it to the proposal.
"""

AUTHOR_USER = """\
# Proposal

{proposal}

{reference}

# Template (`taskforge.stage40_build_with_program.template.standard`)

```python
{template}
```

Write the build program for this proposal.
"""

REVISION_PROMPT = """\
That program needs a revision ({kind}):

{detail}

Find the cause, fix it, and write the complete corrected program. Keep the steps that worked \
unchanged, so their memoized outputs are reused.
"""


NOTES_LIMIT = 2000
"""The longest ``ProgramSubmission.notes`` in characters, room for two to five sentences."""


class ProgramSubmission(BaseModel):
    """Submit the build program: a short summary, then the whole module."""

    # The schema lists the fields in declaration order and the model writes them in that order, so
    # the bounded notes come first and the reply ends with the program's last line.
    notes: str = Field(
        max_length=NOTES_LIMIT,
        description=f"At most {NOTES_LIMIT} characters; two to five sentences on what the program builds and "
        "any deviation from the proposal.",
    )
    source: str = Field(
        description="The complete Python module text, the whole program, ending with "
        "PROGRAM = rl_env_build_program(..., sink=assemble)."
    )

    @field_validator("source")
    @classmethod
    def loads(cls, source: str) -> str:
        load_program(source)
        return source


def _describe(name: str, value: Any) -> list[str]:
    if inspect.isclass(value) and issubclass(value, Enum):
        return [f"enum {name}: {', '.join(f'{member.name} = {member.value!r}' for member in value)}", ""]
    if not callable(value):
        return [f"{name} = {value!r}", ""]
    doc = inspect.getdoc(value) or ""
    if inspect.isclass(value) and doc.startswith(f"{value.__name__}("):
        doc = ""  # a dataclass without a docstring: the generated one repeats the signature
    head = f"{'class ' if inspect.isclass(value) else ''}{name}{inspect.signature(value)}"
    return [head, *(f"    {line}" for line in doc.splitlines()), ""]


REFERENCE_SECTIONS: tuple[tuple[str, ModuleType, tuple[str, ...]], ...] = (
    (
        "Steps and the program",
        program_module,
        ("rl_env_build_step", "rl_env_build_sink", "rl_env_build_program", "PROPOSAL_INPUT"),
    ),
    (
        "The step context",
        context_module,
        ("RLEnvBuildContext", "BuildFailure", "BuildInfrastructureFailure", "BuildExpired", "ExpiredClock"),
    ),
    (
        "Step outputs",
        outputs,
        ("Sources", "ResearchFinding", "WebCall", "Fixtures", "Grader", "GradedCandidate", "Instructions"),
    ),
    (
        "Sandbox effects",
        sandbox_effects,
        ("build_machine", "grade_candidate", "calibrate_decision_grader", "DecisionCalibration", "publish_image"),
    ),
    ("Task types", task_models, ("TaskResource", "InlineFile", "EnvironmentRequirements")),
    ("Resource contents", resources, ("resource_bytes",)),
    (
        "SDK functions, schemas and constants",
        sdk,
        (
            "structured",
            "structured_until",
            "accept",
            "agent",
            "research",
            "task_messages",
            "task_file",
            "files_text",
            "machine_facts",
            "environment_facts",
            "packages_declared",
            "grader_contract",
            "grader_package",
            "output_paths",
            "graded_candidate",
            "grade_diagnostics",
            "instruction_problem",
            "FileDraft",
            "FixturesDraft",
            "DockerfileDraft",
            "GraderDraft",
            "InstructionsDraft",
            "WORKDIR",
            "WORKSPACE",
            "FULL_CREDIT",
            "EQUIVALENT_VARIANTS",
            "ATTEMPTS",
            "INSTRUCTION_WORD_LIMIT",
            "GRADER_TIMEOUT",
            "ANSWER_FORMAT",
            "SOURCE_DATASET",
            "SUBMISSION",
            "NUMERIC_LITERALS",
            "DOCKER_IMAGE_REQUIREMENTS",
        ),
    ),
    (
        "Task spec helpers (`from taskforge.spec import draft`)",
        draft,
        (
            "requirements",
            "machine",
            "MachineLimits",
            "file",
            "answer_grader",
            "decision_grader",
            "judge_grader",
            "grader_environment",
            "python_grader",
            "script_grader",
            "reward_file",
            "assemble",
            "ANSWER_PATH",
        ),
    ),
    ("The verdict file (`from taskforge.spec import verdict`)", verdict, ("GRADER_VERDICT_PATH",)),
    ("Rubrics for `draft.decision_grader` and `draft.judge_grader`", rubric, ("Rubric", "RubricItem", "ItemRole")),
    (
        "Decision-model graders",
        decision,
        ("DecisionGrader", "StateSource", "SOLVER_TURN_STATE", "REJECTED_REWARD_MAX", "REFERENCE_CONTROL"),
    ),
    (
        "Verifyit specs for `draft.answer_grader`",
        verifyit_spec,
        ("ExactSpec", "NumericSpec", "MathSpec", "McqSpec", "StructuredExactSpec", "JudgeSpec"),
    ),
)


def sdk_reference() -> str:
    """The build SDK reference the author reads, one section per module, from signatures and docstrings."""
    lines = ["# Build SDK reference", ""]
    for title, module, names in REFERENCE_SECTIONS:
        lines += [f"## {title} (`{module.__name__}`)", ""]
        for name in names:
            lines += _describe(name, getattr(module, name))
    lines += ["## Elsewhere", ""]
    lines += _describe("taskforge.llm.agent.shell_tool", shell_tool)
    lines += _describe("taskforge.sandbox.images.DockerBuild", DockerBuild)
    lines += _describe("taskforge.sandbox.images.GRADER_BASE_IMAGE", GRADER_BASE_IMAGE)
    lines += ["`EnvironmentRequirements.package_spec` (built by `draft.requirements`; not importable):", ""]
    lines += _describe("shellbox.machine.PackageSpec", PackageSpec)
    return "\n".join(lines)


def author_messages(proposal: TaskProposal) -> list[Message]:
    """The author request: the rules, then the proposal, the SDK reference and the template."""
    system = AUTHOR_SYSTEM.format(grader=GRADER_STEP, imports=", ".join(sorted(ALLOWED_IMPORTS)))
    user = AUTHOR_USER.format(proposal=render(proposal), reference=sdk_reference(), template=inspect.getsource(standard))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class GlmAuthor:
    """An ``Author`` for one workpiece and round that asks ``model`` for the program.

    Attributes:
        model: The model that writes the program.
        sampling: Its sampling policy.
        workpiece_dir: The workpiece's directory: ``program.py`` and the ``author/`` records live here.
        record: Where the author's calls are recorded.
    """

    model: ModelClient
    sampling: LLMPolicy
    workpiece_dir: Path
    record: CallLedger

    async def __call__(
        self, proposal: TaskProposal, revision: Revision | None, author_state: Mapping[str, JsonValue]
    ) -> AuthoredProgram:
        """Author the program for ``proposal``, revising the workpiece's ``program.py`` when ``revision`` is given.

        The prompt does not carry ``author_state``.

        Raises:
            FileNotFoundError: ``revision`` is given and the workpiece has no ``program.py`` to revise.
            StructuredOutputError: the output limit cut a reply off, or every submission, repairs
                included, failed to load.
        """
        messages = author_messages(proposal)
        if revision is not None:
            follow_up: list[Message] = [
                {"role": "assistant", "content": (self.workpiece_dir / PROGRAM_FILE).read_text()},
                {"role": "user", "content": REVISION_PROMPT.format(kind=revision.kind, detail=revision.detail)},
            ]
            messages += follow_up
        attempts = self.workpiece_dir / AUTHOR_DIR
        attempts.mkdir(parents=True, exist_ok=True)
        record_dir = attempts / f"{len(list(attempts.iterdir())):02d}"
        record_dir.mkdir()
        # The library appends the field instruction as the final user turn; the record keeps what was sent.
        sent = [*messages, {"role": "user", "content": schema_instruction(ProgramSubmission)}]
        _write(record_dir / REQUEST_FILE, json.dumps(sent, indent=2).encode())
        try:
            result = await recorded_structured(
                self.model,
                messages,
                ProgramSubmission,
                self.sampling,
                self.record,
                {"revision": "none" if revision is None else str(revision.kind)},
            )
        except StructuredOutputError as error:
            _write(record_dir / COMPLETIONS_FILE, _COMPLETIONS.dump_json(error.completions, indent=2))
            raise
        _write(record_dir / COMPLETIONS_FILE, _COMPLETIONS.dump_json(result.completions, indent=2))
        source = result.value.source
        _write(self.workpiece_dir / PROGRAM_FILE, source.encode())
        program = load_program(source)
        record = {
            "program_digest": program.digest,
            "sdk_version": SDK_VERSION,
            "model": self.model.model,
            "revision": None if revision is None else str(revision.kind),
            "notes": result.value.notes,
            "requests": len(result.completions),
        }
        _write(record_dir / RECORD_FILE, json.dumps(record, indent=2).encode())
        return AuthoredProgram(program, source)


def _write(path: Path, content: bytes) -> None:
    with atomic_rename(str(path)) as temp:
        Path(temp).write_bytes(content)


_COMPLETIONS: TypeAdapter[tuple[Completion, ...]] = TypeAdapter(tuple[Completion, ...])
