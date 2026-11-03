# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write task proposals for a catalog capability with GLM, as a Taskforge ``ProposalSource``.

``CapabilitySource`` proposes in two phases:

1. ``plan(idea, n)``: a structured call plans ``n`` slots. Each slot gets a workflow, an environment
   the host can run, the solver's machine (``null`` for a container slot), a verification (one of
   ``OFFERED_VERIFICATIONS``, so never the reserved ``composite``) and a research decision.
   The validators of the plan's output type reject a portfolio with the wrong slot count, a solver
   machine that does not fit its environment, a slot in an environment the host lacks, or too many
   non-code slots on one environment x verification pairing, and the model gets one repair.
2. ``write(idea, portfolio, slot)``: for one proposed slot, independent of every other, GLM writes a
   proposal document in the ``stage10_proposal`` JSON format, thinking first. The request decodes under
   schema derived from the proposal dataclasses, with fixed id and source fields. Continuations
   stay off because a continuation would restart the JSON grammar inside the object.

A slot fails alone, as a ``SlotFailure``, when the output limit cuts its document off or when its
model call runs out of retries or is rejected (``ModelUnavailable``, ``ModelRequestRejected``); the
batch keeps the plan and every other slot. When the parser refuses a document on a rule outside the
schema, such as the research citations or the research decision against its list, GLM gets one
repair request under the same schema. The repair request keeps the prior reply as the assistant turn
and quotes the parse error. A slot the plan marks null becomes a proposal with ``null_reason`` set
and costs no model call.

Both prompts quote ``taskforge.capability.capability_idea_record(idea)``, and the system turn of both quotes the
verification rule the source is given, a prompt of the policy (``VERIFICATION_RULE`` names the default).
"""

from collections import Counter
from dataclasses import replace
from enum import StrEnum
from textwrap import indent
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskforge.capability import CapabilityIdea, capability_idea_record
from taskforge.content_hash import pretty_json
from taskforge.llm.client import Completion, FinishReason, ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import ERROR_TEXT_LIMIT, schema_instruction
from taskforge.policy import Prompt, ProposerPolicy
from taskforge.stage10_proposal.model import (
    OFFERED_VERIFICATIONS,
    REQUIRED_HEADINGS,
    Environment,
    Grounding,
    OfferedVerification,
    ProposalFormatError,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    document_request_fields,
    parse,
    render,
    solver_machine_violation,
    source_text,
)
from taskforge.stage10_proposal.source import PlannedSlot as PortfolioSlot
from taskforge.stage10_proposal.source import Portfolio, SlotFailure, SlotOutcome, SlotProposal

CUT_OFF = "cut off by the output limit"
SLOT_MODEL_ERRORS = (ModelUnavailable, ModelRequestRejected)
"""The model errors that fail one slot, rather than the batch."""
VERIFICATION_RULE = "proposer.determinate_first_preferred"
"""The policy key of the default verification rule, the text the 2026-11-02 n=25 curriculum run used."""


class SlotStatus(StrEnum):
    PROPOSE = "propose"
    NULL = "null"


class ExcludedCombination(BaseModel):
    """An environment x verification pairing ruled out for the whole portfolio."""

    model_config = ConfigDict(extra="forbid")

    environment: Environment
    verification: OfferedVerification
    reason: str = Field(min_length=1, description="Why no slot of the portfolio can use this pairing.")


class PlannedSlot(BaseModel):
    """A task slot of the portfolio."""

    model_config = ConfigDict(extra="forbid")

    slot: int = Field(description="The slot number, counting from 1 in order.")
    title: str = Field(description="A short title for the task.")
    workflow: str = Field(description="The real workflow the task reproduces.")
    environment: Environment = Field(description="The solver's workspace and tools.")
    solver_machine: SolverMachine | None = Field(
        description="The solver's machine for a reasoning or shellsim slot (none, no machine, for reasoning only); "
        "null for a container slot."
    )
    verification: OfferedVerification = Field(
        description="How the private evaluator grades the task, by the verification rule."
    )
    distinctive_challenge: str = Field(description="What sets this task apart from the other slots.")
    research_need: ResearchNeed = Field(description="Whether building this task needs a lookup, by the criterion above.")
    research_reason: str = Field(min_length=1, description="A line naming what drives the research decision.")
    status: SlotStatus = Field(
        description="propose for a real, distinct workflow; null for a slot you cannot fill with one."
    )
    reason: str | None = Field(description="For a null slot, the substantive reason; null for a proposed slot.")


UNCAPPED_VERIFICATIONS = frozenset({Verification.CODE})
"""Verifications any number of proposed slots may share with one environment; their slots differ by workflow."""


def max_slots_per_pairing(n: int) -> int:
    """Return how many proposed slots of an ``n``-slot portfolio may share an environment x verification pairing."""
    return -(-n // 3)


class SlotPlan(BaseModel):
    """The portfolio of task slots for the capability above."""

    model_config = ConfigDict(extra="forbid")

    coverage_rationale: str = Field(
        min_length=1, description="How the slots cover the capability, with any limitation of a single slot."
    )
    excluded_combinations: list[ExcludedCombination] = Field(
        description="The pairings ruled out for the whole portfolio, an entry per pairing; no proposed slot uses one."
    )
    slots: list[PlannedSlot] = Field(description="The slots, numbered from 1 in order.")
    research_priorities: list[str] = Field(
        description="For slots marked needed only, what their builders look up first; empty when no slot needs research."
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        numbers = [s.slot for s in self.slots]
        if numbers != list(range(1, len(self.slots) + 1)):
            raise ValueError(f"slots must be numbered 1..{len(self.slots)} in order, got {numbers}")
        excluded = [(c.environment, c.verification) for c in self.excluded_combinations]
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded_combinations lists a pairing twice")
        for s in self.slots:
            violation = solver_machine_violation(s.environment, s.solver_machine)
            if violation is not None:
                raise ValueError(f"slot {s.slot}: {violation}")
            if s.status is SlotStatus.NULL and not (s.reason and s.reason.strip()):
                raise ValueError(f"slot {s.slot} is null and needs a substantive reason")
            if s.status is SlotStatus.PROPOSE and not (s.title.strip() and s.workflow.strip()):
                raise ValueError(f"slot {s.slot} is proposed and needs a title and workflow")
            if s.status is SlotStatus.PROPOSE and (s.environment, s.verification) in excluded:
                raise ValueError(
                    f"slot {s.slot} uses {s.environment} x {s.verification}, which excluded_combinations rules out"
                )
        limit = max_slots_per_pairing(len(self.slots))
        pairings = Counter(
            (s.environment, s.verification)
            for s in self.slots
            if s.status is SlotStatus.PROPOSE and s.verification not in UNCAPPED_VERIFICATIONS
        )
        crowded = [f"{e} x {v} ({count} slots)" for (e, v), count in pairings.items() if count > limit]
        if crowded:
            raise ValueError(
                f"at most {limit} proposed slots may share an environment x verification pairing; "
                f"over the limit: {', '.join(crowded)}"
            )
        return self


def slot_plan_type(n: int, environments: frozenset[Environment]) -> type[SlotPlan]:
    """Build the ``SlotPlan`` subclass for ``n`` slots on a host that runs ``environments``.

    Its schema and its validation take exactly ``n`` slots and put every proposed slot in one of
    ``environments``. A null slot may name any environment, since the host lacking it can be the reason.
    """

    class HostSlotPlan(SlotPlan):
        """The portfolio of task slots for the capability above."""

        slots: list[PlannedSlot] = Field(
            min_length=n, max_length=n, description=f"Exactly {n} slots, numbered 1 through {n} in order."
        )

        @model_validator(mode="after")
        def runnable(self) -> Self:
            for s in self.slots:
                if s.status is SlotStatus.PROPOSE and s.environment not in environments:
                    raise ValueError(
                        f"slot {s.slot} is proposed in {s.environment}, which this host cannot run; "
                        f"use one of {', '.join(sorted(environments))} or mark the slot null"
                    )
            return self

    return HostSlotPlan


def system_prompt(proposer: ProposerPolicy, rule: Prompt) -> str:
    """The system turn of every proposer call, quoting the verification rule ``rule``."""
    return proposer.system.render(solver_machine_rule=solver_machine_rule(proposer), verification_rule=rule.render())


def solver_machine_rule(proposer: ProposerPolicy) -> str:
    """The rule for the ``solver_machine`` header value."""
    return proposer.solver_machine_rule.render(**{machine.value: machine.value for machine in SolverMachine})


def plan_prompt(proposer: ProposerPolicy, idea: CapabilityIdea, n: int, environments: frozenset[Environment]) -> str:
    """Build the planning request for ``n`` slots on a host that runs ``environments``."""
    return proposer.plan.render(
        n=n,
        environments=", ".join(sorted(environments)),
        max_per_pairing=max_slots_per_pairing(n),
        research_criterion=proposer.research_criterion.render(),
        capability_record=pretty_json(capability_idea_record(idea)),
    )


def document_template(proposal_id: str, idea: CapabilityIdea) -> str:
    """Show the JSON object and Markdown sections the proposal must contain."""
    return pretty_json(
        {
            "header": {
                "id": proposal_id,
                "source": {"kind": "capability", "ref": idea.capability_id, "hash": idea.capability_hash},
                "grounding": "unverified",
                "null_reason": None,
                "environment": "<reasoning | shellsim | container>",
                "solver_machine": "<none (reasoning only) | shellsim | cpython | packages, or null for container>",
                "verification": " | ".join(OFFERED_VERIFICATIONS),
                "composite": [],
                "research_need": "<needed | not_needed>",
                "research_reason": "<what does or does not require looking anything up>",
                "research": [{"kind": "<web | github>", "purpose": "<what to find, desired properties and fallback>"}],
                "build": [
                    {"name": "<short-artifact-name>", "description": "<what this artifact is and how it is checked>"}
                ],
                "resources": ["<relative/path/of/a/file/the/build/produces>"],
            },
            "body": "\n".join(f"## {heading}\n<...>\n" for heading in REQUIRED_HEADINGS),
        }
    )


def proposal_prompt(
    proposer: ProposerPolicy, idea: CapabilityIdea, plan: SlotPlan, slot: PlannedSlot, proposal_id: str
) -> str:
    """Build the request for the proposal document of ``slot``, quoting the whole portfolio."""
    return proposer.write.render(
        slot=slot.slot,
        document_template=document_template(proposal_id, idea),
        solver_machine_rule=solver_machine_rule(proposer),
        research_criterion=indent(proposer.research_criterion.render(), "  "),
        headings=", ".join(f"`## {h}`" for h in REQUIRED_HEADINGS),
        capability_record=pretty_json(capability_idea_record(idea)),
        portfolio=pretty_json(plan.model_dump(mode="json")),
        planned_slot=pretty_json(slot.model_dump(mode="json")),
    )


def proposal_id(idea: CapabilityIdea, slot: int) -> str:
    return f"{idea.capability_id}/{slot}"


def source_ref(idea: CapabilityIdea) -> SourceRef:
    return SourceRef(kind=SourceKind.CAPABILITY, ref=idea.capability_id, hash=idea.capability_hash)


def checked_proposal(text: str, idea: CapabilityIdea, slot: int) -> TaskProposal:
    """Parse ``text`` and compare the identity fields with the ones the prompt told the model to copy.

    Raises:
        ProposalFormatError: ``parse`` refused the text, or its id, source or grounding differs.
    """
    proposal = parse(text)
    expected_id = proposal_id(idea, slot)
    if proposal.header.id != expected_id:
        raise ProposalFormatError(f"id: expected {expected_id!r}, got {proposal.header.id!r}")
    if proposal.header.source != source_ref(idea):
        raise ProposalFormatError(
            f"source: expected {source_text(source_ref(idea))}, got {source_text(proposal.header.source)}"
        )
    if proposal.header.grounding is not Grounding.UNVERIFIED:
        raise ProposalFormatError(f"grounding: capability proposals are unverified, got {proposal.header.grounding}")
    return proposal


def null_slot_proposal(idea: CapabilityIdea, slot: PlannedSlot) -> SlotProposal:
    """Write the null proposal for a slot the plan marked null, with no model call."""
    assert slot.reason is not None
    header = ProposalHeader(
        id=proposal_id(idea, slot.slot),
        source=source_ref(idea),
        environment=slot.environment,
        solver_machine=slot.solver_machine,
        verification=slot.verification,
        composite=(),
        grounding=Grounding.UNVERIFIED,
        research_need=ResearchNeed.NOT_NEEDED,
        research_reason=f"null slot: {slot.reason.strip()}",
        research=(),
        build=(),
        resources=(),
        null_reason=slot.reason.strip(),
    )
    body = f"Slot {slot.slot} ({slot.title or 'untitled'}) was planned as null.\n\n{slot.reason.strip()}\n"
    return SlotProposal(slot.slot, parse(render(TaskProposal(header=header, body=body))), (), (), None)


def plan_request(
    proposer: ProposerPolicy, rule: Prompt, idea: CapabilityIdea, n: int, environments: frozenset[Environment]
) -> tuple[Message, ...]:
    """Build the first planning request for ``n`` slots on a host that runs ``environments``, as sent.

    It is the system prompt, ``plan_prompt``, and the final user turn holding the plan type's field
    instruction, which ``complete_structured`` appends to the messages it is given.
    """
    return (
        {"role": "system", "content": system_prompt(proposer, rule)},
        {"role": "user", "content": plan_prompt(proposer, idea, n, environments)},
        {"role": "user", "content": schema_instruction(slot_plan_type(n, environments))},
    )


async def plan_slots(
    model: ModelClient,
    policy: LLMPolicy,
    proposer: ProposerPolicy,
    rule: Prompt,
    idea: CapabilityIdea,
    n: int,
    environments: frozenset[Environment],
) -> Portfolio[SlotPlan]:
    """Plan ``n`` slots for ``idea`` on a host that runs ``environments``, with the client's repair budget.

    The portfolio holds the validated plan, the messages its first request sent and its completions (two
    if repaired), and each slot with the reason the plan gave a null one.

    Raises:
        StructuredOutputError: the output limit cut the plan off, or every reply failed validation.
    """
    request = plan_request(proposer, rule, idea, n, environments)
    # complete_structured appends the field instruction itself, so the call gets the turns before it.
    result = await model.complete_structured(list(request[:-1]), slot_plan_type(n, environments), policy)
    plan = result.value
    slots = tuple(_portfolio_slot(s) for s in plan.slots)
    return Portfolio(request, result.completions, plan, slots)


def _portfolio_slot(slot: PlannedSlot) -> PortfolioSlot:
    """The portfolio's record of a planned slot: null with the plan's reason, or proposing."""
    if slot.status is not SlotStatus.NULL:
        return PortfolioSlot(slot.slot, None)
    assert slot.reason is not None, "the plan's validator requires a reason on every null slot"
    return PortfolioSlot(slot.slot, slot.reason.strip())


async def author_proposal(
    model: ModelClient,
    policy: LLMPolicy,
    proposer: ProposerPolicy,
    rule: Prompt,
    idea: CapabilityIdea,
    plan: SlotPlan,
    slot: PlannedSlot,
) -> SlotOutcome:
    """Have GLM write the proposal document of ``slot`` under the proposal JSON schema, repairing until valid or held.

    Every request carries ``document_request_fields`` and run with continuations off, so a reply the
    output limit cuts off comes back as a ``SlotFailure``. So does a request that raises one of
    ``SLOT_MODEL_ERRORS``, with the error's text and the completions received before it.
    """
    slot_id = proposal_id(idea, slot.slot)
    fields = document_request_fields(slot_id, source_ref(idea))
    single = replace(policy, max_continuations=0)
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(proposer, rule)},
        {"role": "user", "content": proposal_prompt(proposer, idea, plan, slot, slot_id)},
    ]
    request = tuple(messages)
    completions: list[Completion] = []
    first_error: str | None = None
    while True:
        try:
            reply = await model.complete(messages, single, fields)
        except SLOT_MODEL_ERRORS as error:
            prefix = "repair: " if completions else ""
            return SlotFailure(slot.slot, f"{prefix}{error}", request, tuple(completions))
        completions.append(reply)
        if reply.finish_reason is FinishReason.LENGTH:
            return SlotFailure(slot.slot, CUT_OFF, request, tuple(completions))
        try:
            proposal = checked_proposal(reply.content, idea, slot.slot)
        except ProposalFormatError as error:
            if first_error is None:
                first_error = str(error)
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": proposer.repair.render(error=str(error)[:ERROR_TEXT_LIMIT])})
            continue
        return SlotProposal(slot.slot, proposal, request, tuple(completions), first_error)


class CapabilitySource:
    """The ``ProposalSource[CapabilityIdea, SlotPlan]`` for catalog capabilities.

    Args:
        model: The model that plans the slots and writes the documents.
        policy: Its sampling policy for every call.
        proposer: The text of every call.
        rule: The verification rule the system turn quotes, such as ``policy.prompt(VERIFICATION_RULE)``.
        environments: The task environments the build host runs. The plan proposes slots in these alone.
    """

    def __init__(
        self,
        model: ModelClient,
        policy: LLMPolicy,
        proposer: ProposerPolicy,
        rule: Prompt,
        environments: frozenset[Environment],
    ):
        self.model = model
        self.policy = policy
        self.proposer = proposer
        self.rule = rule
        self.environments = environments

    async def plan(self, idea: CapabilityIdea, n: int) -> Portfolio[SlotPlan]:
        """Plan ``n`` slots for ``idea`` on this host's environments.

        Raises:
            StructuredOutputError: the plan stayed invalid after its repair.
            ModelUnavailable: the planning call ran out of retries.
            ModelRequestRejected: the server rejected the planning call.
        """
        return await plan_slots(self.model, self.policy, self.proposer, self.rule, idea, n, self.environments)

    async def write(self, idea: CapabilityIdea, portfolio: Portfolio[SlotPlan], slot: int) -> SlotOutcome:
        """The proposal of ``slot``, or a null proposal with no model call for a slot the plan marked null.

        Parser failures receive repair feedback until the document is valid. A cut-off document or a model
        call raising ``SLOT_MODEL_ERRORS`` returns ``SlotFailure`` with its completions; other slots continue.
        """
        planned = portfolio.plan.slots[slot - 1]
        if planned.status is not SlotStatus.PROPOSE:
            return null_slot_proposal(idea, planned)
        return await author_proposal(self.model, self.policy, self.proposer, self.rule, idea, portfolio.plan, planned)
