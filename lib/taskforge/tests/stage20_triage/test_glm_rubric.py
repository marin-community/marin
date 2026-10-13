# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``GlmRubric`` against a scripted ``CallStore`` stand-in."""

import asyncio
from dataclasses import replace

import pytest

from taskforge.llm.client import Completion, FinishReason, StructuredResult, Usage
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError
from taskforge.stage10_proposal.model import ProposalFormatError, TaskProposal, document_regex, render
from taskforge.stage20_triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.stage20_triage.program import REASK_STAGE, REPAIR_STAGE, RUBRIC_STAGE, GlmRubric, RubricReport, evaluate
from taskforge.stage20_triage.verdict import RubricAxis, RubricResult, TriageDecision, Verdict

POLICY = LLMPolicy()
CTX = CheckContext(allowed_combinations=ALL_COMBINATIONS)
ACCEPTING = RubricReport.model_validate(
    {axis.value: 5 for axis in RubricAxis}
    | {"critical_failures": [], "issues": [], "required_changes": [], "recommendation": "accept"}
)


def completion(content: str = "", finish_reason: FinishReason = FinishReason.STOP) -> Completion:
    return Completion(
        content=content,
        reasoning="",
        tool_calls=(),
        finish_reason=finish_reason,
        usage=Usage(100, 50, 0, 0),
        wall_time=1.0,
        ttft=0.1,
        decode_time=0.5,
        continuations=0,
        attempts=(),
    )


class ScriptedStore:
    """Answers ``structured`` per ``(stage, sample)``; a key in ``failing`` raises ``StructuredOutputError``.

    ``complete`` returns ``replies`` in order and records each request.
    """

    def __init__(self, failing=frozenset(), replies=()):
        self.failing = failing
        self.replies = list(replies)
        self.draws = []
        self.requests = []
        self.policies = []

    async def complete(self, stage, messages, policy, *, request_fields, sample=0):
        self.requests.append((stage, messages, request_fields))
        self.policies.append(policy)
        return self.replies.pop(0)

    async def structured(self, stage, messages, output_type, policy, *, sample=0):
        self.draws.append((stage, sample))
        if (stage, sample) in self.failing:
            raise StructuredOutputError(ValueError(f"bad object at {stage}"), (completion(), completion()))
        return StructuredResult(ACCEPTING, (completion(),))


def glm_rubric(store, proposal: TaskProposal) -> GlmRubric:
    return GlmRubric(store, POLICY, samples=3, source_records={proposal.header.source: {"name": "a capability"}})


def run_evaluate(store, proposal):
    return asyncio.run(evaluate(proposal, CHECKS, glm_rubric(store, proposal), CTX))


def repair_verdict(p: TaskProposal) -> Verdict:
    scores = tuple((axis, 3 if axis is RubricAxis.SPECIFICITY else 5) for axis in RubricAxis)
    sample = RubricResult(scores, (), ("vague build plan",), ("name the fixtures",), TriageDecision.REPAIR)
    reasons = ("0 accept, 1 repair, 0 reject of 1 samples", "specificity scored 3 (< 4)")
    return Verdict(p.header.id, p.digest, TriageDecision.REPAIR, (), (sample,), reasons, ())


def run_repair(store, p: TaskProposal):
    return asyncio.run(glm_rubric(store, p).repair(p, repair_verdict(p)))


def test_rubric_sample_that_fails_twice_is_dropped_and_the_rest_decide(proposal):
    stage = f"{RUBRIC_STAGE}.1"
    store = ScriptedStore(failing={(stage, 0), (stage, 1)})
    verdict = run_evaluate(store, proposal)
    assert verdict.decision is TriageDecision.ACCEPT
    assert len(verdict.rubric) == 2
    assert verdict.reasons == (
        "2 accept, 0 repair, 0 reject of 2 samples",
        "rubric sample 1 dropped: structured output invalid after 1 repairs: bad object at triage.rubric.1",
    )
    assert len(verdict.calls) == 2 + 2 * 2


def test_rubric_sample_resampled_once_after_a_structured_failure(proposal):
    store = ScriptedStore(failing={(f"{RUBRIC_STAGE}.0", 0)})
    verdict = run_evaluate(store, proposal)
    assert len(verdict.rubric) == 3
    assert sorted(store.draws) == [
        (f"{RUBRIC_STAGE}.0", 0),
        (f"{RUBRIC_STAGE}.0", 1),
        (f"{RUBRIC_STAGE}.1", 0),
        (f"{RUBRIC_STAGE}.2", 0),
    ]
    assert verdict.reasons == ("3 accept, 0 repair, 0 reject of 3 samples",)


def test_all_samples_failing_raises(proposal):
    store = ScriptedStore(failing={(f"{RUBRIC_STAGE}.{i}", s) for i in range(3) for s in range(2)})
    with pytest.raises(StructuredOutputError):
        run_evaluate(store, proposal)


def revised(p: TaskProposal) -> TaskProposal:
    return replace(p, body=p.body.replace("two real lab handouts", "three real lab handouts"))


def test_repair_request_carries_the_document_regex(proposal):
    store = ScriptedStore(replies=[completion(render(revised(proposal)))])
    repair = run_repair(store, proposal)
    [(stage, _, fields)] = store.requests
    assert stage == REPAIR_STAGE
    assert fields == {"structured_outputs": {"regex": document_regex(proposal.header.id, proposal.header.source)}}
    assert [policy.max_continuations for policy in store.policies] == [0]
    assert repair.proposal == revised(proposal)


def test_repair_reasks_once_on_a_consistency_error(proposal):
    bad = render(proposal).replace("research[0] finds", "research[3] finds")
    good = render(revised(proposal))
    store = ScriptedStore(replies=[completion(bad), completion(good)])
    repair = run_repair(store, proposal)
    assert repair.proposal == revised(proposal) and len(repair.calls) == 2
    (_, first, fields), (stage, reask, reask_fields) = store.requests
    assert stage == REASK_STAGE and reask_fields == fields
    assert reask[: len(first)] == first and reask[len(first)] == {"role": "assistant", "content": bad}
    assert "research[3]" in reask[-1]["content"]


def test_repair_failing_after_the_reask_raises(proposal):
    bad = render(proposal).replace("research[0] finds", "research[3] finds")
    store = ScriptedStore(replies=[completion(bad), completion(bad)])
    with pytest.raises(ProposalFormatError, match=r"research\[3\]"):
        run_repair(store, proposal)
    assert len(store.requests) == 2


def test_repair_cut_off_by_the_output_limit_is_not_reasked(proposal):
    store = ScriptedStore(replies=[completion(render(proposal)[:200], FinishReason.LENGTH)])
    with pytest.raises(ProposalFormatError, match="repair cut off by the output limit"):
        run_repair(store, proposal)
    assert len(store.requests) == 1
