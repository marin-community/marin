# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 schemas for the disagreement primitive pipeline.

Six record types fixed by the Codex plan in `executable_specs_codex.md`
(2026-04-30, "Experiment plan for disagreement primitive and refinement
loop"). Field names track the plan exactly so artifacts can be replayed
across phases.

Project rule: never use reasoning, or use the lowest reasoning tier.
Records that hold a model output must record the reasoning/thinking
setting used so audits stay honest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

# ----- vocab -----------------------------------------------------------------

InferredRole = Literal[
    "requirement_like",
    "guideline_like",
    "meta_rule",
    "style_rule",
    "unclear",
]

CandidateSource = Literal[
    "lm_topk",
    "embedding_neighbor",
    "atlas_known",
    "random_control",
    "all_pair_backtest",
]

PredictedRelation = Literal[
    "dominance",
    "bidirectional_tradeoff",
    "modifier",
    "independent",
    "unclear",
]

ScenarioVariant = Literal["neutral", "biased_to_a", "biased_to_b", "opposite_mode"]

DisagreementLabel = Literal[
    "model_behavior",
    "cross_tension_needed",
    "spec_ambiguity",
    "oracle_unsatisfiable",
    "scenario_bug",
]

PatchType = Literal[
    "add_example",
    "edit_statement_text",
    "add_cross_tension_rubric",
    "add_dominance_rule",
    "add_exception",
    "reclassify_statement_role",
    "split_statement",
    "needs_human_decision",
    "scenario_bug",
]


# ----- records ---------------------------------------------------------------


@dataclass
class StatementAnalysis:
    """One row per statement in `statement_analysis.jsonl`.

    The analyzer sees statement text + examples but NOT
    `authority_level`. The backtest compares `inferred_role` to the
    hidden hierarchy collapse afterwards.
    """

    statement_id: str
    summary: str
    inferred_role: InferredRole
    role_confidence: float  # 0.0 - 1.0
    non_negotiables: list[str]
    soft_preferences: list[str]
    examples_used: list[str]  # short labels / descriptions of examples cited
    likely_tension_targets: list[str]  # other statement_ids likely to tense with this
    likely_supersedes: list[str]
    likely_subordinated_by: list[str]
    rationale_quotes: list[str]  # verbatim substrings of statement text or examples
    # bookkeeping
    analyzer_model: str
    reasoning_setting: str  # e.g. "thinking_budget=0", "reasoning_effort=none"
    temperature: float


@dataclass
class PairCandidate:
    """One row per candidate pair in `pair_candidate.jsonl`."""

    statement_a_id: str
    statement_b_id: str
    candidate_source: CandidateSource
    predicted_relation: PredictedRelation
    predicted_controller: str | None  # statement_id when dominance-like; None otherwise
    why_pair_matters: str
    expected_failure_mode: str
    confidence: float  # 0.0 - 1.0
    # bookkeeping
    classifier_model: str
    reasoning_setting: str
    temperature: float


@dataclass
class ScenarioProbe:
    """One row per scenario in `scenario_probe.jsonl`."""

    pair_id: str  # canonical "<a_id>__<b_id>" with a < b lexicographically
    statement_a_id: str
    statement_b_id: str
    scenario_id: str
    scenario_text: str
    variant: ScenarioVariant
    intended_tension: str
    expected_satisfiability: bool


@dataclass
class OracleResponse:
    """One row per (scenario, generator) pair in `oracle_response.jsonl`."""

    scenario_id: str
    generator_model: str
    generator_mode: str  # e.g. "default", "high_thinking_oracle_search"
    response: str
    self_declared_controlling_statement: str | None
    reasoning_setting: str
    temperature: float


@dataclass
class JudgePanelScore:
    """One row per (oracle_response, judge) pair in `judge_panel_score.jsonl`."""

    scenario_id: str
    oracle_response_id: str  # stable hash of (scenario_id, generator_model, generator_mode)
    judge_model: str
    compliance_score: float  # 0-10
    controlling_statement: str | None
    cited_rubric_clauses: list[str]
    cited_spec_clauses: list[str]
    failure_reason: str | None
    confidence: float
    reasoning_setting: str
    temperature: float


@dataclass
class RepairProposal:
    """One row per compiler proposal in `repair_proposal.jsonl`."""

    scenario_id: str
    disagreement_label: DisagreementLabel
    proposed_patch_type: PatchType
    target_statement_ids: list[str]
    diff: str | None  # for edit_statement_text / add_dominance_rule etc.
    new_example: dict[str, Any] | None  # for add_example
    predicted_downstream_effect: str
    needs_human_decision: bool
    compiler_model: str
    reasoning_setting: str
    temperature: float


# ----- helpers ---------------------------------------------------------------


def to_jsonl_row(obj: Any) -> str:
    """Serialize a dataclass record to a JSONL line (no trailing newline)."""
    return json.dumps(asdict(obj), ensure_ascii=False)


def write_jsonl(records: list[Any], path: str) -> None:
    """Append-friendly write of a list of dataclass records to JSONL."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(to_jsonl_row(rec))
            fh.write("\n")


__all__ = [
    "CandidateSource",
    "DisagreementLabel",
    "InferredRole",
    "JudgePanelScore",
    "OracleResponse",
    "PairCandidate",
    "PatchType",
    "PredictedRelation",
    "RepairProposal",
    "ScenarioProbe",
    "ScenarioVariant",
    "StatementAnalysis",
    "to_jsonl_row",
    "write_jsonl",
]
