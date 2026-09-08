# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen metric meanings and sources; raw optimization reward is never implicit accuracy."""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Metric:
    name: str
    kind: Literal["quality", "performance"]
    producer: str
    key_or_field: str
    source_file_line: str
    denominator: str
    notes: str


def _metric(name, producer, source, denominator, notes, *, kind="quality"):
    return Metric(name, kind, producer, name, source, denominator, notes)


TRAINER = "MarinSkyRL/skyrl-train/skyrl_train/utils/trainer_utils.py:182"
STOPS = "MarinSkyRL/skyrl-train/skyrl_train/rollout_observability.py:75"
AUDIT = "experiments/post_training/async_rl_audit.py:652"
SCORER = "experiments/post_training/math_eval/scoring.py:34"
CURRICULUM = "MarinSkyRL/skyrl-train/skyrl_train/curriculum.py:593"

METRICS = (
    _metric(
        "eval/{ds}/avg_score",
        "wandb|dump_aggregate",
        TRAINER,
        "all evaluated response sequences",
        "Mean token-reward sum; preserves signed/fractional optimization scores.",
    ),
    _metric(
        "eval/{ds}/pass_at_{n}",
        "wandb|dump_aggregate",
        TRAINER,
        "unique prompt UIDs",
        "Any positive native final reward among n responses; fractional RG rewards can count.",
    ),
    *(
        _metric(f"eval/{{ds}}/{key}", "wandb|dump_aggregate", TRAINER, denominator, notes, kind="performance")
        for key, denominator, notes in (
            ("response_tokens", "none; token count", "Finalized response tokens summed over sequences."),
            ("response_tokens_mean", "all response sequences", "Finalized token count per sequence."),
            ("response_tokens_max", "none; maximum", "Largest finalized response length."),
            ("sequences", "none; count", "All finalized evaluated response sequences."),
            ("length_stop_count", "none; count", "Engine or runner length stop; not an answer-quality judgment."),
            ("known_stop_count", "none; count", "Nonempty known stop label."),
            ("unknown_stop_count", "none; count", "Missing or empty stop label."),
            ("stop_reason_coverage", "all response sequences", "Known labels divided by sequences."),
            ("length_stop_fraction", "all response sequences", "Only emitted with complete stop coverage."),
            ("completed_stop_fraction", "all response sequences", "Accepted stop label, irrespective of correctness."),
        )
    ),
    *(
        _metric(
            f"eval/{{ds}}/{key}_stop_score_contribution",
            "wandb|dump_aggregate",
            TRAINER,
            "all response sequences",
            "Native sequence reward sum restricted to this stop class; not conditional accuracy.",
        )
        for key in ("length", "completed")
    ),
    *(
        _metric(f"consumed/{key}", "wandb", STOPS, denominator, notes, kind="performance")
        for key, denominator, notes in (
            ("sequences", "none; count", "Admitted sequences before padding/sharding."),
            ("length_stop_count", "none; count", "Admitted length-stop count."),
            ("known_stop_count", "none; count", "Admitted sequences with known stop label."),
            ("unknown_stop_count", "none; count", "Admitted sequences missing stop label."),
            ("stop_reason_coverage", "admitted sequences", "Known divided by admitted sequences."),
            ("length_stop_fraction", "admitted sequences", "Only emitted when all admitted labels are known."),
        )
    ),
    _metric(
        "trainer/global_step",
        "wandb",
        "experiments/post_training/curriculum_rl/report.py:166",
        "none; update index",
        "Optimizer global step.",
        kind="performance",
    ),
    _metric(
        "generate/avg_num_tokens",
        "wandb",
        "experiments/post_training/curriculum_rl/report.py:167",
        "generated responses",
        "Legacy generated mean length; distinct from consumed loss tokens.",
        kind="performance",
    ),
    _metric(
        "grade_weighted_pass1",
        "quality",
        "experiments/post_training/curriculum_rl/report.py:264",
        "sum of (1 + grade) across every validation bin",
        "Legacy grade-weighted positive-reward pass1; all bins required.",
    ),
    *(
        _metric(f"curriculum/{{ds}}/{key}", "wandb", CURRICULUM, denominator, notes)
        for key, denominator, notes in (
            ("weight", "sum of sampling weights", "Current per-bin probability."),
            ("informative_frac", "per-bin observed groups", "Groups informative to curriculum policy."),
            ("pass_rate", "per-bin observed samples", "Solved samples over observed samples; zero if no samples."),
            ("groups", "none; count", "Per-bin groups in the current update."),
        )
    ),
    _metric(
        "curriculum/level",
        "wandb",
        "MarinSkyRL/skyrl-train/skyrl_train/curriculum.py:433",
        "none; grade",
        "Active level for level-based curriculum policies.",
    ),
    _metric(
        "async_phase_window",
        "finelog",
        "MarinSkyRL/skyrl-train/skyrl_train/rollout_observability.py:64",
        "one phase window",
        "Wall-time start/end/duration, phase/step/role/outcome attributes.",
        kind="performance",
    ),
    *(
        _metric(
            name, "audit", AUDIT, "ordered finalized response rows", "Identity or stop/engine provenance, not accuracy."
        )
        for name in (
            "ordered_prompt_sha256",
            "ordered_response_sha256",
            "ordered_result_sha256",
            "stop_counts",
            "generator_engine_index_counts",
        )
    ),
    *(
        _metric(
            name,
            "audit",
            "experiments/post_training/async_rl_audit.py:924",
            "equal-weight paired questions after averaging observed training seeds",
            "Conditional on observed training seeds; question bootstrap percentile interval.",
        )
        for name in ("final_reward_delta", "initial_to_final_change_delta")
    ),
    *(
        _metric(
            name,
            "quality",
            "experiments/post_training/async_rl_quality_audit.py:285",
            "none; row count",
            "Legacy numeric answer audit; separately report all-row denominator.",
        )
        for name in (
            "raw_correct",
            "extracted_correct",
            "extracted_correct_and_stop",
            "boundary/{status}",
            "status/{status}",
        )
    ),
    *(
        _metric(name, "harness", SCORER, denominator, notes)
        for name, denominator, notes in (
            ("score_contract", "one response; sequence mean when aggregated", "Raw signed/fractional verifier score."),
            (
                "contract_correct",
                "one response; sequence mean when aggregated",
                "Task-native correctness: GSM1, AIME+1, RG>=1.",
            ),
            (
                "score_contract_completed",
                "all response sequences",
                "Binary correctness times accepted stop; ranking/MDE/CI primary.",
            ),
            ("native_reward_tokens", "one response", "Original reward list preserved verbatim; null for scalar dumps."),
            ("native_reward_reduction", "one response", "token_reward_sum or scalar_identity, matching the auditor."),
            (
                "score_semantic",
                "resolved responses only if summarized",
                "Unresolved is null, never silently treated as wrong.",
            ),
            ("semantic_status", "one response", "Extraction or boundary outcome, including unresolved cases."),
            ("semantic_engine", "one response", "Pinned fraction-exact or Math-Verify parser."),
            ("truncated", "all response sequences", "Stop reason is length."),
            ("thinking_closed", "all response sequences", "Token-boundary closure; nonthinking prompts are true."),
            ("response_tokens", "one response", "Finalized token length."),
            ("stop_reason", "one response", "Original runner stop label, including null."),
            ("format_gap", "resolved semantic responses", "Semantic score minus binary contract correctness."),
            ("lm_eval_strict", "GSM responses only", "Pinned strict lm-eval regex/string score."),
            ("lm_eval_flexible", "GSM responses only", "Pinned flexible lm-eval regex/string score."),
            ("prompt_sha256", "one question", "Frozen normalized question identity; legacy mode explicitly marked."),
            ("uid", "one prompt UID", "Native dump sample-group identifier."),
            ("model", "one record table", "Pinned model identity supplied by evaluator."),
            ("contract_rule", "one response", "Named frozen native parser rule."),
        )
    ),
    _metric(
        "pass_rate_k",
        "harness",
        "experiments/post_training/math_eval/pool.py:1",
        "k responses to one question",
        "Planned KE7 ratings field; not yet produced. Uses completed correctness.",
    ),
)


def lookup(name: str) -> Metric:
    for metric in METRICS:
        pattern = re.escape(metric.name)
        for placeholder in ("ds", "status"):
            pattern = pattern.replace(re.escape("{" + placeholder + "}"), "[^/]+")
        pattern = pattern.replace(re.escape("{n}"), "[1-9][0-9]*")
        if re.fullmatch(pattern, name):
            return metric
    raise KeyError(f"Unregistered metric: {name}")
