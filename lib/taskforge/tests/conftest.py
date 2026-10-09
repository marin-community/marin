# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for taskforge tests.

Tests that run a transcript through RolloutEngine take ``scripted_model``. Live tests write raw
evidence under ``evidence_root``, which is outside the checkout.

``ledger`` is an in-memory ``Ledger`` that keeps every recorded entry in ``entries``.

The standard template's model calls are answered by ``TemplateClient``, a ``builder.sdk.ModelEndpoint``
that returns fixed structured answers for one ShellSim task: multiply the two numbers in
``/workspace/numbers.txt`` and reply ``42``. ``proposal`` is its proposal, ``proposal_source`` proposes
it for any idea, ``draft`` is its build on the laptop's ShellSim factory, and ``solver_models`` gives
every solver trial a scripted model that reads the file and replies ``42``.
"""

import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from rolloutengine.contracts import ModelRequest, ModelTurn

from taskforge.builder.run import TaskDraft, run_build
from taskforge.builder.sdk import BuildServices
from taskforge.builder.template import standard
from taskforge.ledger.records import LedgerEntry
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.recording import CallLedger
from taskforge.loop.program import template_program
from taskforge.proposal.model import TaskProposal, parse
from taskforge.proposal.source import ProposalBatch, SlotProposal
from taskforge.sandbox.factories import MachineHost, machine_factories
from taskforge.validate.solver import ModelFactory

EVIDENCE_DIR_ENV = "TASKFORGE_EVIDENCE_DIR"
DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "taskforge-evidence"


@pytest.fixture(scope="session")
def evidence_root() -> Path:
    """Where live checks write raw evidence: ``TASKFORGE_EVIDENCE_DIR`` when set, else ``<tmp>/taskforge-evidence``.

    The default is the system temp directory, so evidence survives across runs on one machine and never
    lands in the checkout.
    """
    return Path(os.environ.get(EVIDENCE_DIR_ENV) or DEFAULT_EVIDENCE_ROOT).expanduser()


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def ledger() -> ListLedger:
    return ListLedger()


ScriptedModel = Callable[[ModelRequest], Awaitable[ModelTurn]]


def scripted(turns: tuple[dict, ...]) -> ScriptedModel:
    """A model that serves ``turns`` in order, with token ids that extend each request's prefix."""

    async def model(request: ModelRequest) -> ModelTurn:
        served = sum(message["role"] == "assistant" for message in request.messages)
        message = turns[served]
        prompt = (*request.prefix_token_ids, 2 * served + 1)
        stop = "tool_calls" if "tool_calls" in message else "stop"
        return ModelTurn(message, prompt, (2 * served + 2,), None, stop)

    return model


@pytest.fixture
def scripted_model() -> Callable[[tuple[dict, ...]], ScriptedModel]:
    """``scripted_model(turns)`` replays ``turns`` as the policy model of a RolloutEngine rollout."""
    return scripted


PROPOSAL = """---
id: "IDEA/0"
source: {kind: capability, ref: "IDEA", hash: "abc123"}
environment: shellsim
verification: simple
grounding: unverified
research: []
build:
  - "numbers": "the two factors"
resources: ["workspace/numbers.txt"]
null_reason: null
---
## Task
Multiply the two numbers in numbers.txt.

## Realism and workflow
A clerk checks a product.

## Research plan
None.

## Build plan
Write the numbers file.

## Grader design and controls
Numeric, zero tolerance: only 42 passes.

## Risks and null conditions
None.
"""


def _control(control_id: str, kind: str, category: str, concern: str, reply: str, passes: bool) -> dict[str, Any]:
    return {
        "id": control_id,
        "kind": kind,
        "category": category,
        "concern": concern,
        "final_reply": reply,
        "files": [],
        "reward_min": 0.99 if passes else None,
        "reward_max": None if passes else 0.0,
        "rationale": "fixed",
    }


ANSWERS: dict[str, dict[str, Any]] = {
    "submit_fixtures": {
        "agent_files": [{"path": "workspace/numbers.txt", "content": "6 7\n", "executable": False}],
        "private_files": [],
        "facts": "6 * 7 = 42",
    },
    "submit_grader": {
        "kind": "numeric",
        "expected": "42",
        "tolerance": 0.0,
        "answer_contract": "Reply with the product alone.",
        "reference_reply": "42",
        "reference_files": [],
        "secret_values": ["42"],
    },
    "submit_instructions": {
        "system": "",
        "instruction": "Multiply the two numbers in /workspace/numbers.txt. Reply with the product alone.",
    },
    "submit_controls": {
        "controls": [
            _control("reference", "positive", "known_correct", "reference", "42", True),
            _control("sum", "negative", "plausible_wrong", "acceptance", "13", False),
            _control("first", "negative", "task_specific_shortcut", "shortcut", "6", False),
            _control("empty", "malformed", "empty_or_malformed", "extraction", "", False),
        ]
    },
}

SOLVER_TURNS = (
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-0",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": "cat /workspace/numbers.txt"})},
            }
        ],
    },
    {"role": "assistant", "content": "42"},
)


@dataclass(frozen=True)
class ModelId:
    model: str


@dataclass
class TemplateClient:
    """Answers each structured call of ``builder.template.standard`` from ``answers``, by tool name."""

    answers: dict[str, dict[str, Any]]
    endpoint: ModelId = ModelId("scripted-builder")
    calls: list[str] = field(default_factory=list)

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        self.calls.append(name)
        return output_type.model_validate(self.answers[name])


@pytest.fixture
def template_client() -> TemplateClient:
    return TemplateClient(dict(ANSWERS))


@pytest.fixture
def proposal() -> TaskProposal:
    return parse(PROPOSAL)


@dataclass
class FixedSource:
    """Proposes ``proposal`` for any idea and counts its calls."""

    proposal: TaskProposal
    calls: int = 0

    async def propose(self, idea: object, n: int) -> ProposalBatch:
        self.calls += 1
        return ProposalBatch(planning_request=(), planning=(), slots=(SlotProposal(0, self.proposal, (), (), None),))


@pytest.fixture
def proposal_source(proposal: TaskProposal) -> FixedSource:
    return FixedSource(proposal)


@pytest.fixture
def build_services(tmp_path: Path, ledger: ListLedger, template_client: TemplateClient) -> BuildServices:
    factories = machine_factories(MachineHost.LAPTOP, None, tmp_path / "images")
    return BuildServices(
        client=template_client, policy=LLMPolicy(), host=MachineHost.LAPTOP, factories=factories, ledger=ledger
    )


@pytest.fixture
async def draft(tmp_path: Path, proposal: TaskProposal, build_services: BuildServices) -> TaskDraft:
    return await run_build(template_program(standard), proposal, tmp_path / "item", tmp_path / "cache", build_services)


@pytest.fixture
def solver_models() -> ModelFactory:
    def models(record: CallLedger) -> ScriptedModel:
        return scripted(SOLVER_TURNS)

    return models
