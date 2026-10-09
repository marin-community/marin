# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The builder SDK: the ``Build`` context a builder program's ``build(b)`` receives.

A builder program is a module that defines memoized steps (``@step(role)``) and an
``async def build(b: Build) -> BuildOutput``. The program returns the task, its lowered form and
its controls; it never marks itself complete. ``run_build`` checks the output, adds provenance, and
records the draft.

A build reaches its model through ``BuildServices.client``, any ``ModelEndpoint``: an object that
names its model and answers structured calls. Steps call it through ``b.llm.structured``, which
records each call as an ``LLM_CALL`` span under the running step.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec
from shellbox.machine import MachineFactory
from taskcompendium.grader import GraderPackage
from taskcompendium.models import TaskResource, TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from taskforge.builder.step import CURRENT_STEP, Blob, Resource, Step, StepCache
from taskforge.content_hash import digest
from taskforge.ledger.records import EntryKind, Ledger, span
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.proposal.model import TaskProposal, render
from taskforge.sandbox.factories import MachineHost
from taskforge.spec import controls as controls_module
from taskforge.spec import draft as draft_module
from taskforge.spec.controls import Control
from taskforge.spec.draft import MachineSettings

NUMERIC_LITERALS = (
    "A numeric answer's expected value (verifyit `NumericSpec.expected`) is a literal string: an integer, "
    'decimal, scientific-notation number or integer fraction such as "42", "-0.125", "1.5e3" or "1/8". '
    'It is never a float or an expression (not 0.125, not "sqrt(2)").'
)


class BuildFailure(Exception):
    """The program decided the task cannot be built as written; ``step`` names where."""

    def __init__(self, message: str, step: str | None):
        super().__init__(message if step is None else f"{step}: {message}")
        self.message = message
        self.step = step


@dataclass(frozen=True)
class Grader:
    """What a GRADER step returns: the grader and the evidence it was prototyped on.

    Attributes:
        package: The task's grader (``spec.answer_grader``), passed to ``spec.assemble`` as ``grader``.
        answer_contract: The exact output format the solver must follow, for the instructions.
        reference_reply: A correct final reply, used to prototype the grader.
        reference_files: Files a correct solver leaves in the task machine (``spec.file``, paths
            relative to the machine root), if the grader reads any.
        secret_values: Answer strings the instructions must not contain.
    """

    package: GraderPackage
    answer_contract: str
    reference_reply: str
    reference_files: tuple[TaskResource, ...] = ()
    secret_values: tuple[str, ...] = ()


def file_set(files: Sequence[TaskResource]) -> frozenset[tuple[str, bytes]]:
    """``files`` as (path, content) pairs, so two file lists compare by what they install."""
    return frozenset((resource.path, resource_bytes(resource)) for resource in files)


@dataclass(frozen=True)
class BuildOutput:
    """What ``build(b)`` returns. The grader must come from a GRADER step and the controls
    from a CONTROLS step; ``run_build`` checks both.

    ``lowered`` is what ``b.lower(task, ...)`` returned for ``task``: the machine settings (backend,
    network, limits, user, startup timeout) and the session (turn budget, deadlines, verifier
    timeout) RolloutEngine runs it with. Validation replaces the session's turn budget and deadlines
    with its own; the verifier timeout and machine settings stand.

    The task carries its answer format (``spec.assemble(answer_format=...)``): RolloutEngine appends
    its submission instruction to the task prompt and extracts the answer with it. Reference replies
    and control replies follow it.
    """

    task: TaskSpec
    lowered: LoweredTaskSpec
    controls: tuple[Control, ...]


class ModelName(Protocol):
    @property
    def model(self) -> str: ...


class ModelEndpoint(Protocol):
    """The model a build calls: ``endpoint.model`` names it, and ``structured`` forces one call of a strict
    tool named ``name`` whose arguments ``output_type`` validates."""

    @property
    def endpoint(self) -> ModelName: ...

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T: ...


@dataclass(frozen=True)
class BuildServices:
    """Everything a build reaches outside its own directory.

    Attributes:
        client: The model every step calls.
        policy: Sampling policy for every model call in the build; part of every step key.
        host: Where the build runs; ``spec.lower`` picks machine backends for it.
        factories: Machine factories keyed by shellbox ``Backend`` value, as
            ``taskforge.sandbox.factories.machine_factories(host, ...)`` returns them.
        ledger: Where steps and model calls are recorded.
    """

    client: ModelEndpoint
    policy: LLMPolicy
    host: MachineHost
    factories: Mapping[str, MachineFactory]
    ledger: Ledger


class BuildLLM:
    """Model access for steps. Every call is recorded in the ledger under the running step."""

    def __init__(self, client: ModelEndpoint, policy: LLMPolicy, ledger: Ledger, item_id: str, round: int):  # noqa: A002
        self.client = client
        self.policy = policy
        self._ledger = ledger
        self._item_id = item_id
        self._round = round

    @property
    def digest(self) -> str:
        """The policy digest that every step key includes."""
        return digest({"model": self.client.endpoint.model, "policy": self.policy})

    async def structured[T: BaseModel](self, messages: Sequence[Message], output_type: type[T], name: str) -> T:
        """Force one call of a strict tool named ``name`` whose arguments ``output_type`` validates.

        Put checks the model can fix in ``output_type`` validators.
        """
        frame = CURRENT_STEP.get()
        step = "program" if frame is None else frame.name
        with span(self._ledger, EntryKind.LLM_CALL, item_id=self._item_id, round=self._round, step=step) as fields:
            fields.model = self.client.endpoint.model
            fields.attrs["tool"] = name
            return await self.client.structured(messages, output_type, name)


class Build:
    """The context a builder program and each of its steps receive as ``b``.

    Attributes:
        proposal: The accepted proposal being built.
        item_id: The item's file-safe id.
        llm: Model access (``structured``).
        spec: ``taskforge.spec.draft``: ``requirements``, ``machine``, ``session``, ``file``,
            ``answer_grader``, ``assemble`` and friends.
        controls: ``taskforge.spec.controls``: ``Control``, ``Expectation``, ``Transcript``,
            ``Workspace``, ``reply``, ``shell_turn``.
        ledger: The build ledger.
        scratch: A private directory for this build; not part of any output.
    """

    def __init__(
        self,
        proposal: TaskProposal,
        item_id: str,
        services: BuildServices,
        cache: StepCache,
        scratch: Path,
        round: int,  # noqa: A002 - matches LedgerEntry.round
    ):
        self.proposal = proposal
        self.item_id = item_id
        self.llm = BuildLLM(services.client, services.policy, services.ledger, item_id, round)
        self.spec = draft_module
        self.controls = controls_module
        self.ledger = services.ledger
        self.scratch = scratch
        self.round = round
        self.resources: list[Resource] = []
        self._services = services
        self._cache = cache

    @property
    def proposal_text(self) -> str:
        """The proposal's canonical text: front matter plus body."""
        return render(self.proposal)

    async def run_step(self, step: Step, arguments: Mapping[str, object]) -> Any:
        return await self._cache.run(
            step,
            arguments,
            self,
            proposal_digest=self.proposal.digest,
            policy_digest=self.llm.digest,
            ledger=self.ledger,
            round=self.round,
            emitted=self.resources,
        )

    def check(self, condition: bool, message: str) -> None:
        """Fail the build with ``BuildFailure(message)`` unless ``condition`` holds."""
        if not condition:
            raise self.failure(message)

    def failure(self, message: str) -> BuildFailure:
        """A ``BuildFailure(message)`` naming the running step, for ``raise b.failure(...)``."""
        frame = CURRENT_STEP.get()
        return BuildFailure(message, None if frame is None else frame.name)

    def emit(self, name: str, content: bytes) -> Blob:
        """Store ``content`` content-addressed and record it as resource ``name`` of this build."""
        blob = self._cache.blobs.put(content)
        resource = Resource(name=name, blob=blob)
        frame = CURRENT_STEP.get()
        if frame is None:
            self.resources.append(resource)
        else:
            frame.resources.append(resource)
        return blob

    def read(self, blob: Blob) -> bytes:
        """The content of a blob this build or an earlier one stored."""
        return self._cache.blobs.get(blob)

    def lower(
        self,
        task: TaskSpec,
        *,
        task_machine: MachineSettings | None,
        verifier_machine: MachineSettings | None,
        session: TaskSessionSpec,
    ) -> LoweredTaskSpec:
        """``spec.lower`` for this host and its machine factories: what ``BuildOutput.lowered`` holds.

        ``task_machine`` is the task machine's settings (``spec.machine``), ``None`` exactly when the
        task has none; ``verifier_machine`` likewise, given exactly when the grader has an environment
        (``spec.answer_grader`` with one).

        Raises:
            BuildFailure: ``spec.lower`` or RolloutEngine rejects the lowered task.
        """
        try:
            return draft_module.lower(
                task,
                host=self._services.host,
                task_machine=task_machine,
                verifier_machine=verifier_machine,
                session=session,
                factories=self._services.factories,
            )
        except (ValueError, NotImplementedError) as error:
            raise self.failure(f"lower: {error}") from error
