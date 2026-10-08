# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Author a builder program for a proposal with one structured GLM call.

The model receives the proposal, the SDK reference (``sdk_reference``, generated from docstrings)
and a template's source, and returns a module. The module is compiled in a namespace that starts
with ``SDK_EXPORTS`` only; its imports are limited to ``ALLOWED_IMPORTS`` and its builtins to
``SAFE_BUILTINS``. This keeps a program to the SDK's surface. It is not a security boundary:
programs are trusted model output, run in the builder's process.

A program must define ``async def build(b)``, its own GRADER step, and its own CONTROLS step. A
compile or rule failure is a validation error, so it goes back to the model in the structured
repair request. The accepted source is stored beside the proposal as ``program.py``, with
``program.json`` recording its digest and inputs.
"""

import builtins
import importlib
import inspect
import json
import linecache
import sys
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from taskforge.atomic_file import write_atomic
from taskforge.build.sdk import SDK_EXPORTS, Build, BuildOutput, BuildServices, sdk_reference
from taskforge.build.step import SDK_VERSION, Step, StepRole, code_names
from taskforge.content_hash import sha256_hex
from taskforge.llm.client import Completion
from taskforge.llm.policy import Message
from taskforge.llm.recording import CallLedger, recorded_structured
from taskforge.llm.structured import StructuredTool
from taskforge.proposal.model import TaskProposal, render

PROGRAM_FILE = "program.py"
PROGRAM_RECORD = "program.json"
PROPOSAL_FILE = "proposal.md"
AUTHOR_DIR = "author"
SUBMIT_TOOL = "submit_build_program"

ALLOWED_IMPORTS = frozenset(
    {
        "taskforge.build.sdk",
        "taskforge.build.step",
        "taskforge.build.template.standard",
        "taskforge.spec.draft",
        "taskforge.spec.controls",
        "taskforge.llm.agent",
        "taskforge.llm.policy",
        "taskforge.proposal.model",
        "taskcompendium.environment",
        "taskcompendium.execution",
        "taskcompendium.models",
        "taskcompendium.grading",
        "taskcompendium.grading_result",
        "taskcompendium.submission",
        "verifyit.spec",
        "pydantic",
        "__future__",
        "asyncio",
        "base64",
        "hashlib",
        "collections",
        "collections.abc",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "re",
        "shlex",
        "statistics",
        "string",
        "textwrap",
        "typing",
    }
)
UNSAFE_BUILTINS = frozenset(
    {"open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals", "vars", "help", "exit", "quit"}
)

AUTHOR_SYSTEM = """\
You write builder programs for Taskforge. A builder program is one Python module that turns an \
accepted task proposal into a TaskCompendium TaskSpec plus fixed controls, using only the builder \
SDK described below. Rules:
- Define `async def build(b: Build) -> BuildOutput` and return
  `BuildOutput(task=..., execution=..., convention=..., controls=...)`, where `execution` is the
  `TaskExecution` you passed to `spec.assemble` (`TaskExecution()` when the task sets no deadlines,
  user or stages) and `convention` is the `taskcompendium.submission` convention the solver submits
  under (`PlainText(id="plain_text")` unless the answer needs another). Reference and control
  replies follow that convention.
- Every unit of work is a memoized step: `@step(StepRole.X)` on an `async def name(b, ...) -> Output`.
  Step outputs must be JSON-serializable: dataclasses, pydantic models, tuples, str, int, float.
  Pass everything a step depends on as an argument so the memo key changes when it changes.
- Define your own GRADER step (returns `Grader`) and your own CONTROLS step (returns
  `tuple[Control, ...]`) in this module, specialized to this proposal. They must be separate steps.
  Controls are written, not graded: never replay controls; `validate` does that later.
- Prototype the grader on its reference answer with `b.try_grader` and fail with `b.check` when it
  does not give the reference full credit or gives an empty answer credit. You may also prototype
  it on wrong answers you invent for that purpose, but never on a control's candidate: every
  candidate `b.try_grader` grades is recorded, and the build fails when a control (other than the
  reference and the empty answer) is one of them. Do not share a candidate list between the
  GRADER and CONTROLS steps.
- A partial control (ControlKind.PARTIAL) needs exact reward components, which only a grader that
  writes a JSON reward file reports (`spec.reward_file(path, RewardFileFormat.JSON)` with one key
  per criterion). With a stdout reward, leave partial controls out.
- Task-specific checks belong in the grader script the task ships (a `spec.shell_verifier`, which
  runs inside the task machine after the solver finishes, receives the conversation as JSON on
  stdin, and prints one reward in [0, 1]); generic answer checks use `spec.answer_verifier`.
- A shell verifier needs an executable environment: use EnvironmentKind.SHELLSIM for reasoning
  and shellsim proposals, and EnvironmentKind.DOCKER with a DockerBuild for containers. ShellSim
  has `sh`, coreutils, and a minimal `python3` shim, not CPython: only part of the standard library
  exists (json, re, fractions, math, sys work; traceback does not) and some methods take fewer
  arguments (a compiled regex's `match`/`search` take only the string, no `pos`). Keep grader
  scripts simple, and do not catch broad exceptions around the whole grader: an unexpected error
  should crash so `b.try_grader` shows it. It has no network and no pip.
- The program runs with restricted builtins: `exec`, `eval`, `compile`, `open`, `globals` and
  `vars` do not exist. Compute in plain Python inside steps; run scripts in `b.machine`.
- Use model calls (`b.llm.structured`, `b.llm.complete`, `b.llm.agent`) for content you cannot
  compute; compute exact values (answer keys, arithmetic) in Python inside steps where possible.
- Never put an answer the grader checks into the solver-visible instruction or files.
- You may import from: {imports}. You may reuse helpers and steps from
  `taskforge.build.template.standard`, but the grader and controls steps must be your own.
The template below is the standard session shape; adapt it to the proposal.
"""

AUTHOR_USER = """\
# Proposal

{proposal}

{reference}

# Template (`taskforge.build.template.standard`)

```python
{template}
```

Write the builder program for this proposal and submit it with `{tool}`.
"""

REVISION_PROMPT = """\
Running that program failed:

{failure}

Find the cause, fix it, and submit the complete corrected program with `{tool}`. Keep the steps \
that worked unchanged so their memoized results are reused.
"""


@dataclass(frozen=True)
class BuildProgram:
    """A compiled builder program and the inputs it was authored from."""

    source: str
    digest: str
    proposal_digest: str
    module: ModuleType

    @property
    def build(self) -> Callable[[Build], Awaitable[BuildOutput]]:
        return self.module.build

    @property
    def steps(self) -> dict[str, Step]:
        """Steps defined in this program (not imported), by name."""
        return {
            value.name: value
            for value in vars(self.module).values()
            if isinstance(value, Step) and value.fn.__module__ == self.module.__name__
        }


def _restricted_import(
    name: str,
    globals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
    locals: Mapping[str, object] | None = None,  # noqa: A002 - __import__ signature
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> ModuleType:
    if level != 0 or name not in ALLOWED_IMPORTS:
        raise ImportError(f"builder programs may not import {name!r}; allowed: {sorted(ALLOWED_IMPORTS)}")
    if "." in name and not fromlist:
        raise ImportError(f"use 'from {name} import ...' rather than 'import {name}'")
    return importlib.import_module(name)


SAFE_BUILTINS: dict[str, object] = {
    **{name: value for name, value in vars(builtins).items() if name not in UNSAFE_BUILTINS},
    "__import__": _restricted_import,
}


def module_name(source: str) -> str:
    """The ``sys.modules`` name a program's module is registered under: one per distinct source."""
    return f"taskforge_program_{sha256_hex(source.encode())[:16]}"


def load_module(source: str) -> ModuleType:
    """Execute ``source`` as a module in the restricted namespace, without the program rules.

    A source already loaded in this process returns its registered module, so the classes that
    live programs' step outputs are built from stay the ones pydantic resolves annotations in.

    Raises:
        ValueError: the source does not compile or execute.
    """
    name = module_name(source)
    if name in sys.modules:
        return sys.modules[name]
    digest = sha256_hex(source.encode())
    filename = f"<taskforge-program {digest[:16]}>"
    # inspect.getsource (the step memo key) reads exec'd code through linecache.
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType(name)
    module.__dict__.update({**SDK_EXPORTS, "__builtins__": SAFE_BUILTINS, "__file__": filename})
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except Exception as error:
        del sys.modules[name]
        raise ValueError(f"the program failed to load: {type(error).__name__}: {error}") from error
    return module


def compile_program(source: str, proposal_digest: str) -> BuildProgram:
    """Load a builder program and check its shape.

    A program that breaks a rule is unregistered from ``sys.modules`` unless it was loaded before.

    Raises:
        ValueError: the source does not compile or execute, or breaks a program rule.
    """
    registered = module_name(source) in sys.modules
    module = load_module(source)
    program = BuildProgram(
        source=source, digest=sha256_hex(source.encode()), proposal_digest=proposal_digest, module=module
    )
    try:
        _check_program(program)
    except ValueError:
        if not registered:
            del sys.modules[module.__name__]
        raise
    return program


def _check_program(program: BuildProgram) -> None:
    module = program.module
    build = getattr(module, "build", None)
    if build is None or not inspect.iscoroutinefunction(build):
        raise ValueError("the program must define `async def build(b)`")
    unavailable = sorted(
        name
        for name in code_names(compile(program.source, "<program>", "exec"))
        if name in UNSAFE_BUILTINS and name not in vars(module)
    )
    if unavailable:
        raise ValueError(f"the program uses builtins that do not exist for builder programs: {unavailable}")
    roles = [s.role for s in program.steps.values()]
    for role in (StepRole.GRADER, StepRole.CONTROLS):
        if role not in roles:
            raise ValueError(f"the program must define its own {role.upper()} step")


@dataclass(frozen=True)
class Revision:
    """A program to correct and why: the failure a build or review reported."""

    source: str
    failure: str


class ProgramSubmission(BaseModel):
    """Submit the complete builder program module."""

    source: str = Field(description="The complete Python module source, defining async def build(b).")
    notes: str = Field(description="Two to five sentences: what the program builds and any deviation from the proposal.")

    @field_validator("source")
    @classmethod
    def compiles(cls, source: str) -> str:
        registered = module_name(source) in sys.modules
        compile_program(source, proposal_digest="")
        if not registered:
            del sys.modules[module_name(source)]
        return source


def author_messages(proposal: TaskProposal, template: ModuleType) -> list[Message]:
    """The author request: the rules, then the proposal, the SDK reference, and the template."""
    system = AUTHOR_SYSTEM.format(imports=", ".join(sorted(ALLOWED_IMPORTS)))
    user = AUTHOR_USER.format(
        proposal=render(proposal), reference=sdk_reference(), template=inspect.getsource(template), tool=SUBMIT_TOOL
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def author(
    proposal: TaskProposal,
    template: ModuleType,
    item_dir: Path,
    services: BuildServices,
    item_id: str,
    revision: Revision | None = None,
    round: int = 0,  # noqa: A002 - matches LedgerEntry.round
) -> BuildProgram:
    """Author, compile, and store a builder program for ``proposal``.

    With ``revision``, the request continues with the previous program as the model's turn and the
    failure as the next user turn, and asks for the corrected program. Each request, its
    completions, and the accepted source are kept under ``item_dir/author/<n>/``; the current
    program is ``item_dir/program.py``. ``round`` is the build round the ledger records.
    """
    messages = author_messages(proposal, template)
    if revision is not None:
        follow_up: list[Message] = [
            {"role": "assistant", "content": revision.source},
            {"role": "user", "content": REVISION_PROMPT.format(failure=revision.failure, tool=SUBMIT_TOOL)},
        ]
        messages += follow_up
    tool = StructuredTool(name=SUBMIT_TOOL, description=ProgramSubmission.__doc__ or "", output_type=ProgramSubmission)
    item_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(item_dir / PROPOSAL_FILE, render(proposal).encode())
    attempts = item_dir / AUTHOR_DIR
    attempts.mkdir(exist_ok=True)
    record_dir = attempts / f"{len(list(attempts.iterdir())):02d}"
    record_dir.mkdir()
    write_atomic(record_dir / "request.json", json.dumps(messages, indent=2).encode())
    record = CallLedger(ledger=services.ledger, item_id=item_id, round=round, step="author")
    result = await recorded_structured(
        services.client, messages, services.policy, tool, record, {"revision": str(revision is not None)}
    )
    write_atomic(record_dir / "completions.json", _COMPLETIONS.dump_json(result.completions, indent=2))
    program = compile_program(result.value.source, proposal.digest)
    write_atomic(record_dir / PROGRAM_FILE, program.source.encode())
    write_atomic(item_dir / PROGRAM_FILE, program.source.encode())
    record = {
        "digest": program.digest,
        "proposal_digest": proposal.digest,
        "revises": None if revision is None else sha256_hex(revision.source.encode()),
        "template": template.__name__,
        "template_digest": sha256_hex(inspect.getsource(template).encode()),
        "sdk_version": SDK_VERSION,
        "model": services.client.endpoint.model,
        "notes": result.value.notes,
        "requests": len(result.completions),
    }
    write_atomic(item_dir / PROGRAM_RECORD, json.dumps(record, indent=2).encode())
    return program


def load_program(item_dir: Path, proposal: TaskProposal) -> BuildProgram:
    """Compile the stored ``program.py`` of an item."""
    return compile_program((item_dir / PROGRAM_FILE).read_text(), proposal.digest)


_COMPLETIONS: TypeAdapter[tuple[Completion, ...]] = TypeAdapter(tuple[Completion, ...])
