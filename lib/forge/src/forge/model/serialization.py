# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JSON file boundary: one reusable adapter per persisted value type."""

import json
from typing import Any

from pydantic import TypeAdapter

from forge.model.controls import Candidate, Control, ControlKind, RewardBand
from forge.model.llm import Completion, FinishReason, Message, ReasoningEffort, Sampling, ToolCall, ToolSchema, Usage
from forge.model.proposal import (
    OFFERED_VERIFICATIONS,
    BuildItem,
    CompositePart,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchItem,
    ResearchKind,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    source_text,
)
from forge.model.review import (
    Accept,
    AdversaryResult,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    ControlResult,
    Decision,
    Incomplete,
    Modify,
    Reject,
    RejectKind,
    Retry,
    TrialResult,
    ValidationResult,
)
from forge.model.source import PlannedSlot, ProposalBatch, SlotFailure, SlotProposal
from forge.model.stage import Stage
from forge.model.verdict import GraderCheck
from forge.model.workpiece import Conversation, Submission, Terminal, Workpiece
from forge.policy import Policy, Prompt

TASK_PROPOSAL = TypeAdapter(TaskProposal)
PROPOSAL_HEADER = TypeAdapter(ProposalHeader)
SOURCE_REF = TypeAdapter(SourceRef)
RESEARCH_ITEM = TypeAdapter(ResearchItem)
BUILD_ITEM = TypeAdapter(BuildItem)
COMPOSITE_PART = TypeAdapter(CompositePart)
SOURCE_KIND = TypeAdapter(SourceKind)
ENVIRONMENT = TypeAdapter(Environment)
SOLVER_MACHINE = TypeAdapter(SolverMachine)
VERIFICATION = TypeAdapter(Verification)
GROUNDING = TypeAdapter(Grounding)
RESEARCH_KIND = TypeAdapter(ResearchKind)
RESEARCH_NEED = TypeAdapter(ResearchNeed)
CANDIDATE = TypeAdapter(Candidate)
CONTROL = TypeAdapter(Control)
CONTROL_KIND = TypeAdapter(ControlKind)
REWARD_BAND = TypeAdapter(RewardBand)
MESSAGE = TypeAdapter(Message)
TOOL_CALL = TypeAdapter(ToolCall)
TOOL_SCHEMA = TypeAdapter(ToolSchema)
SAMPLING = TypeAdapter(Sampling)
USAGE = TypeAdapter(Usage)
COMPLETION = TypeAdapter(Completion)
FINISH_REASON = TypeAdapter(FinishReason)
REASONING_EFFORT = TypeAdapter(ReasoningEffort)
SUBMISSION = TypeAdapter(Submission)
CONVERSATION = TypeAdapter(Conversation)
WORKPIECE = TypeAdapter(Workpiece)
TERMINAL = TypeAdapter(Terminal)
VALIDATION_RESULT = TypeAdapter(ValidationResult)
TRIAL_RESULT = TypeAdapter(TrialResult)
CONTROL_RESULT = TypeAdapter(ControlResult)
ADVERSARY_RESULT = TypeAdapter(AdversaryResult)
DECISION: TypeAdapter[Decision] = TypeAdapter(Accept | Reject | Modify | Retry | Incomplete)
ACCEPT = TypeAdapter(Accept)
REJECT = TypeAdapter(Reject)
MODIFY = TypeAdapter(Modify)
RETRY = TypeAdapter(Retry)
INCOMPLETE = TypeAdapter(Incomplete)
BAND_OUTCOME = TypeAdapter(BandOutcome)
BAND_RULES = TypeAdapter(BandRules)
REJECT_KIND = TypeAdapter(RejectKind)
BUDGET = TypeAdapter(Budget)
CAUSE = TypeAdapter(Cause)
PROMPT = TypeAdapter(Prompt)
POLICY = TypeAdapter(Policy)
PLANNED_SLOT = TypeAdapter(PlannedSlot)
SLOT_PROPOSAL = TypeAdapter(SlotProposal)
SLOT_FAILURE = TypeAdapter(SlotFailure)
PROPOSAL_BATCH = TypeAdapter(ProposalBatch)
STAGE = TypeAdapter(Stage)
GRADER_CHECK = TypeAdapter(GraderCheck)


def proposal_request_fields(proposal_id: str, source: SourceRef) -> dict[str, Any]:
    """Constrain generation to the proposal schema with fixed identity fields."""
    schema = TASK_PROPOSAL.json_schema()
    fields = schema["$defs"]["ProposalHeader"]["properties"]
    fields["id"] = {"const": proposal_id, "type": "string"}
    fields["source"] = {"const": json.loads(source_text(source))}
    fields["verification"] = {"enum": list(OFFERED_VERIFICATIONS), "type": "string"}
    fields["composite"] = {"type": "array", "maxItems": 0, "items": {"$ref": "#/$defs/CompositePart"}}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaskProposal",
                "schema": schema,
                "strict": True,
            },
        }
    }
