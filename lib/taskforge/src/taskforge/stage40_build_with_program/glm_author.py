# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An author that writes a build program for a proposal with one structured GLM call.

``GlmAuthor`` sends the rules (``author.system``), then the proposal, the SDK reference
(``sdk_reference``, generated from signatures and docstrings) and the standard template's source.
The model replies with a ``ProgramSubmission``, whose validator loads the source with
``program_sdk.load_program``, so a program the loader refuses goes back to the model in the client's
structured repair request, with the loader's message. A reply cut off by the output limit gets no
repair: the call raises ``IncompleteOutput`` and preserves the partial completion.

The author lives with the build stage because its prompt documents the build SDK and the template.
It keeps each request as sent, ending with the field instruction ``complete_structured``
appends, and its completions under ``author/<n>/`` in the workpiece's directory
(``AuthorContext.workpiece_dir``), including the completions of a call that raised
``StructuredOutputError`` or ``IncompleteOutput``. It writes the accepted source to ``program.py`` in that directory
before it loads the program from it. For a revision it reads that file back, since a ``Revision`` carries
no source, and continues the conversation with the previous program as the model's turn and the
revision as the next user turn.
"""

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, field_validator
from rigging.filesystem.atomic import write_bytes_atomic
from shellbox.machine import PackageSpec
from taskcompendium import grading_result
from taskcompendium import models as task_models
from taskcompendium.runtime import resources
from verifyit import spec as verifyit_spec

from taskforge.ledger.records import Ledger
from taskforge.llm.agent import shell_tool
from taskforge.llm.client import Completion, ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger, recorded_structured
from taskforge.llm.structured import IncompleteOutput, StructuredOutputError, schema_instruction
from taskforge.policy import Policy, Prompt
from taskforge.sandbox.images import GRADER_BASE_IMAGE, DockerBuild
from taskforge.spec import draft, verdict
from taskforge.stage10_proposal.model import TaskProposal, render
from taskforge.stage30_author import program as program_module
from taskforge.stage30_author.author import PROGRAM_FILE, AuthorContext, AuthoredProgram, Revision
from taskforge.stage30_author.loader import ALLOWED_IMPORTS, GRADER_STEP
from taskforge.stage40_build_with_program import context as context_module
from taskforge.stage40_build_with_program import outputs, sandbox_effects, sdk
from taskforge.stage40_build_with_program.program_sdk import load_program
from taskforge.stage40_build_with_program.step_store import SDK_VERSION
from taskforge.stage40_build_with_program.template import standard

AUTHOR_DIR = "author"
AUTHOR_STEP = "author"
"""The ledger step of every author call."""
REQUEST_FILE = "request.json"
COMPLETIONS_FILE = "completions.json"
RECORD_FILE = "record.json"

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
    ("Sandbox effects", sandbox_effects, ("build_machine", "grade_candidate", "publish_image")),
    ("Task types", task_models, ("TaskResource", "InlineFile", "EnvironmentRequirements", "AnswerType")),
    (
        "Grades",
        grading_result,
        ("GradeResult", "Outcome", "GradingFailure"),
    ),
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
            "grader_environment",
            "python_grader",
            "script_grader",
            "reward_file",
            "assemble",
            "ANSWER_PATH",
        ),
    ),
    (
        "The verdict file (`from taskforge.spec import verdict`)",
        verdict,
        ("GRADER_VERDICT_PATH", "GraderCheck", "grader_checks", "reward_in_range"),
    ),
    (
        "Verifyit specs for `draft.answer_grader`",
        verifyit_spec,
        ("ExactSpec", "NumericSpec", "MathSpec", "McqSpec", "StructuredExactSpec", "JudgeSpec"),
    ),
)


def sdk_reference(grading_reference: Prompt) -> str:
    """The build SDK reference the author reads, one section per module, from signatures and docstrings.

    After the sections, ``grading_reference`` describes what ``grade_candidate`` returns.
    """
    lines = ["# Build SDK reference", ""]
    for title, module, names in REFERENCE_SECTIONS:
        lines += [f"## {title} (`{module.__name__}`)", ""]
        for name in names:
            lines += _describe(name, getattr(module, name))
    lines += [*grading_reference.render().splitlines(), ""]
    lines += ["## Elsewhere", ""]
    lines += _describe("taskforge.llm.agent.shell_tool", shell_tool)
    lines += _describe("taskforge.sandbox.images.DockerBuild", DockerBuild)
    lines += _describe("taskforge.sandbox.images.GRADER_BASE_IMAGE", GRADER_BASE_IMAGE)
    lines += ["`EnvironmentRequirements.package_spec` (built by `draft.requirements`; not importable):", ""]
    lines += _describe("shellbox.machine.PackageSpec", PackageSpec)
    return "\n".join(lines)


def author_messages(proposal: TaskProposal, policy: Policy) -> list[Message]:
    """The author request: ``policy``'s rules, then the proposal, the SDK reference and the template."""
    system = policy.prompt("author.system").render(grader=GRADER_STEP, imports=", ".join(sorted(ALLOWED_IMPORTS)))
    user = policy.prompt("author.user").render(
        proposal=render(proposal),
        reference=sdk_reference(policy.prompt("author.grading_reference")),
        template=inspect.getsource(standard),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class GlmAuthor:
    """An ``Author`` that asks ``model`` for the program.

    Attributes:
        model: The model that writes the program.
        sampling: Its sampling policy.
        ledger: Where the author's calls are recorded, under the call's workpiece and round and the
            step ``AUTHOR_STEP``.
        prompts: The text of the request and of a revision.
    """

    model: ModelClient
    sampling: LLMPolicy
    ledger: Ledger
    prompts: Policy

    async def __call__(
        self,
        context: AuthorContext,
        proposal: TaskProposal,
        revision: Revision | None,
        author_state: Mapping[str, JsonValue],
    ) -> AuthoredProgram:
        """Author the program for ``proposal``, revising the workpiece's ``program.py`` when ``revision`` is given.

        ``program.py`` and the ``author/`` records live in ``context.workpiece_dir``. The prompt does
        not carry ``author_state``.

        Raises:
            FileNotFoundError: ``revision`` is given and the workpiece has no ``program.py`` to revise.
            IncompleteOutput: Generation reached its physical context/output boundary.
            StructuredOutputError: Every explicitly configured repair failed to load.
        """
        workpiece_dir = context.workpiece_dir
        messages = author_messages(proposal, self.prompts)
        if revision is not None:
            follow_up: list[Message] = [
                {"role": "assistant", "content": (workpiece_dir / PROGRAM_FILE).read_text()},
                {
                    "role": "user",
                    "content": self.prompts.prompt("author.revision").render(kind=revision.kind, detail=revision.detail),
                },
            ]
            messages += follow_up
        attempts = workpiece_dir / AUTHOR_DIR
        attempts.mkdir(parents=True, exist_ok=True)
        record_dir = attempts / f"{len(list(attempts.iterdir())):02d}"
        record_dir.mkdir()
        # The library appends the field instruction as the final user turn; the record keeps what was sent.
        sent = [*messages, {"role": "user", "content": schema_instruction(ProgramSubmission)}]
        write_bytes_atomic(str(record_dir / REQUEST_FILE), json.dumps(sent, indent=2).encode())
        try:
            result = await recorded_structured(
                self.model,
                messages,
                ProgramSubmission,
                self.sampling,
                CallLedger(self.ledger, context.workpiece_id, context.round, AUTHOR_STEP),
                {"revision": "none" if revision is None else str(revision.kind)},
            )
        except (IncompleteOutput, StructuredOutputError) as error:
            write_bytes_atomic(str(record_dir / COMPLETIONS_FILE), _COMPLETIONS.dump_json(error.completions, indent=2))
            raise
        write_bytes_atomic(str(record_dir / COMPLETIONS_FILE), _COMPLETIONS.dump_json(result.completions, indent=2))
        source = result.value.source
        write_bytes_atomic(str(workpiece_dir / PROGRAM_FILE), source.encode())
        program = load_program(source)
        record = {
            "program_digest": program.digest,
            "sdk_version": SDK_VERSION,
            "model": self.model.model,
            "revision": None if revision is None else str(revision.kind),
            "notes": result.value.notes,
            "requests": len(result.completions),
        }
        write_bytes_atomic(str(record_dir / RECORD_FILE), json.dumps(record, indent=2).encode())
        return AuthoredProgram(program, source)


_COMPLETIONS: TypeAdapter[tuple[Completion, ...]] = TypeAdapter(tuple[Completion, ...])
