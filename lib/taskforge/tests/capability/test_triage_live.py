# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live validation of triage against GLM-5.3 on the interactive tier.

Evaluates every proposal the proposal builder saved under ``.evidence/proposal/`` with
``RUBRIC_SAMPLES`` rubric samples each, then repairs up to ``REPAIRS`` of the REPAIR verdicts and
re-evaluates them. Evidence goes to
``lib/taskforge/.evidence/triage/run-<timestamp>/``: ``calls/`` is the CallStore (every request and
full response), ``verdicts/`` one verdict per proposal, ``repairs/`` each rewritten document and its
new verdict, and ``summary.json`` the distributions, wall time, and tokens per evaluation.
"""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import pytest

from taskforge.llm.client import Completion, GlmClient, GlmEndpoint, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.store import CallStore
from taskforge.llm.structured import StructuredOutputError
from taskforge.proposal.model import ProposalFormatError, TaskProposal, parse, render
from taskforge.proposal.sources.capability import capability_prompt_record, load_capability_ideas, source_ref
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext, CheckStatus
from taskforge.triage.program import GlmRubric, evaluate, sample_decision
from taskforge.triage.verdict import RubricAxis, TriageDecision, Verdict

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / ".evidence"
PROPOSALS = EVIDENCE_ROOT / "proposal"
REPAIRS = 3
RUBRIC_SAMPLES = 3


def completion_record(completion: Completion) -> dict[str, object]:
    return {
        "content": completion.content,
        "reasoning": completion.reasoning,
        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in completion.tool_calls],
        "finish_reason": completion.finish_reason,
        "usage": vars(completion.usage),
        "wall_time": completion.wall_time,
    }


def evaluation_record(name: str, verdict: Verdict, wall_time: float) -> dict[str, object]:
    return {
        "proposal": name,
        "decision": verdict.decision,
        "failed_checks": [r.name for r in verdict.structural if r.status is CheckStatus.FAIL],
        "sample_decisions": [sample_decision(r)[0] for r in verdict.rubric],
        "sample_scores": [{str(a): s for a, s in r.scores} for r in verdict.rubric],
        "recommendations": [r.recommendation for r in verdict.rubric],
        "wall_time": wall_time,
        "calls": len(verdict.calls),
        "prompt_tokens": sum(c.usage.prompt_tokens for c in verdict.calls),
        "completion_tokens": sum(c.usage.completion_tokens for c in verdict.calls),
        "reasoning_tokens": sum(c.usage.reasoning_tokens for c in verdict.calls),
        "finish_reasons": [c.finish_reason for c in verdict.calls],
    }


async def timed_evaluate(p: TaskProposal, rubric: GlmRubric, ctx: CheckContext) -> tuple[Verdict, float]:
    started = time.monotonic()
    verdict = await evaluate(p, CHECKS, rubric, ctx)
    return verdict, time.monotonic() - started


async def recorded_evaluate(
    name: str, p: TaskProposal, rubric: GlmRubric, ctx: CheckContext, out: Path
) -> tuple[Verdict, float] | dict[str, object]:
    """Evaluate ``p``; when the rubric's structured output fails, save its completions and return the error."""
    started = time.monotonic()
    try:
        return await timed_evaluate(p, rubric, ctx)
    except StructuredOutputError as error:
        (out / f"{name}.completions.json").write_text(
            json.dumps([completion_record(c) for c in error.completions], indent=2, ensure_ascii=False)
        )
        return {"proposal": name, "error": str(error), "wall_time": time.monotonic() - started}


async def repair_and_reevaluate(
    name: str, p: TaskProposal, verdict: Verdict, rubric: GlmRubric, ctx: CheckContext, out: Path
) -> dict[str, object]:
    started = time.monotonic()
    try:
        repair = await rubric.repair(p, verdict)
    except ProposalFormatError as error:
        return {"proposal": name, "repair_error": str(error), "repair_wall_time": time.monotonic() - started}
    repair_wall = time.monotonic() - started
    (out / f"{name}.md").write_text(render(repair.proposal))
    reverdict, wall = await timed_evaluate(repair.proposal, rubric, ctx)
    (out / f"{name}.verdict.json").write_text(reverdict.to_json())
    return {
        "proposal": name,
        "repair_wall_time": repair_wall,
        "repair_call": {
            "wall_time": repair.call.wall_time,
            "finish_reason": repair.call.finish_reason,
            "usage": vars(repair.call.usage),
        },
        "before": verdict.decision,
        "after": evaluation_record(name, reverdict, wall),
    }


@pytest.mark.live_glm
@pytest.mark.timeout(7200)
def test_triage_on_proposal_evidence(glm_settings, capability_catalog):
    ideas = load_capability_ideas(capability_catalog)
    sources = {source_ref(idea): capability_prompt_record(idea) for idea in ideas.values()}
    paths = sorted(PROPOSALS.rglob("*.md"))
    assert paths, f"no proposals under {PROPOSALS}; run the proposal live test first"
    names = ["__".join(path.relative_to(PROPOSALS).with_suffix("").parts) for path in paths]
    proposals = [parse(path.read_text()) for path in paths]
    out = EVIDENCE_ROOT / "triage" / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    (out / "verdicts").mkdir(parents=True)
    (out / "repairs").mkdir()
    (out / "errors").mkdir()
    ctx = CheckContext(allowed_combinations=ALL_COMBINATIONS)

    async def run() -> tuple[dict[str, tuple[Verdict, float]], list[dict[str, object]], list[dict[str, object]], float]:
        endpoint = GlmEndpoint(base_url=glm_settings.base_url, token=glm_settings.token, pool=Pool.HIGH)
        async with GlmClient(endpoint) as client:
            rubric = GlmRubric(CallStore(out / "calls", client), LLMPolicy(), RUBRIC_SAMPLES, sources)
            started = time.monotonic()
            outcomes = await asyncio.gather(
                *(recorded_evaluate(n, p, rubric, ctx, out / "errors") for n, p in zip(names, proposals, strict=True))
            )
            evaluate_wall = time.monotonic() - started
            results = {n: o for n, o in zip(names, outcomes, strict=True) if isinstance(o, tuple)}
            errors = [o for o in outcomes if isinstance(o, dict)]
            for name, (verdict, _) in results.items():
                (out / "verdicts" / f"{name}.json").write_text(verdict.to_json())
            to_repair = [
                (name, p, results[name][0])
                for name, p in zip(names, proposals, strict=True)
                if name in results and results[name][0].decision is TriageDecision.REPAIR
            ][:REPAIRS]
            repairs = await asyncio.gather(
                *(repair_and_reevaluate(n, p, v, rubric, ctx, out / "repairs") for n, p, v in to_repair)
            )
            return results, errors, list(repairs), evaluate_wall

    results, errors, repairs, evaluate_wall = asyncio.run(run())
    records = [evaluation_record(n, v, w) for n, (v, w) in results.items()]
    rubric_records = [r for r in records if r["sample_decisions"]]
    samples = [s for v, _ in results.values() for s in v.rubric]
    summary = {
        "proposals": len(paths),
        "evaluated": len(records),
        "rubric_output_errors": errors,
        "evaluate_wall_time_all_concurrent": evaluate_wall,
        "decisions": Counter(str(r["decision"]) for r in records),
        "rubric_samples": RUBRIC_SAMPLES,
        "sample_decisions": Counter(str(sample_decision(s)[0]) for s in samples),
        "sample_recommendations": Counter(str(s.recommendation) for s in samples),
        "unanimous_evaluations": sum(len(set(r["sample_decisions"])) == 1 for r in rubric_records),
        "structural": {
            check.name: Counter(
                str(next(s.status for s in v.structural if s.name == check.name)) for v, _ in results.values()
            )
            for check in CHECKS
        },
        "axis_score_counts": {str(axis): Counter(str(s.score(axis)) for s in samples) for axis in RubricAxis},
        "evaluations": records,
        "repairs": repairs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    assert errors == []
    assert len(records) == len(paths)
    assert all(len(v.calls) >= RUBRIC_SAMPLES for v, _ in results.values() if v.rubric)
    assert [r for r in repairs if "repair_error" in r] == []
