# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the scaling_laws module.

These tests focus on integration and behavioral validation, particularly
the snapshot test which ensures reproducibility of config generation.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from marin.scaling_laws.isoflop_analysis import (
    DEFAULT_SEQ_LEN,
    CandidateConfig,
    fit_scaling_laws,
    robust_quad_logx,
)


@dataclass
class FakeWandbRun:
    name: str
    group: str
    tags: list[str]
    summary: dict
    config: dict
    state: str = "finished"
    path: list[str] | None = None
    created_at: str | None = None
    updated_at: str | None = None


# --- Run name parsing tests ---


def test_parse_isoflop_run_name():
    """Test parsing isoflop run names extracts experiment names."""
    from experiments.isoflop_sweep import parse_isoflop_run_name

    # New format: isoflop-{budget}-N{params}-B{batch}-{experiment_name}
    assert parse_isoflop_run_name("isoflop-1e+18-N1e+08-B128-nemo-wider-depth-adapt") == "nemo-wider-depth-adapt"
    assert parse_isoflop_run_name("isoflop-1e+18-N1e+08-B128-dclm-a1b2c3") == "dclm"  # hash stripped

    # Legacy format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    assert parse_isoflop_run_name("isoflop-1e+18-d512-L8-B128-dclm-a1b2c3") == "dclm"
    assert parse_isoflop_run_name("isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt") == "nemo-wider-depth-adapt"

    # Invalid formats return None
    assert parse_isoflop_run_name("not-a-valid-name") is None
    assert parse_isoflop_run_name("") is None


# --- Candidate config tests ---


def test_candidate_configs_within_tolerance():
    """Test that generated configs achieve the target FLOP budget within tolerance."""
    from experiments.scaling_law_sweeps.c_adamc import CAdamCHeuristic

    heuristic = CAdamCHeuristic()
    budget = 1e19
    flop_tolerance = 0.01
    seq_len = DEFAULT_SEQ_LEN

    # Generate candidates using the public API
    for candidate in heuristic.candidates_for_budget(budget, seq_len):
        flops_per_token = candidate.model_config.flops_per_token(heuristic.vocab_size, seq_len)

        # Compute training FLOPs inline: 3 * flops_per_token * batch * steps * seq_len
        achieved = 3 * flops_per_token * candidate.batch_size * candidate.train_steps * seq_len
        relative_error = abs(achieved - budget) / budget
        assert relative_error <= flop_tolerance


# --- Curve fitting tests ---


def test_robust_quad_logx_fits_quadratic():
    """Test that robust_quad_logx recovers known coefficients from synthetic data."""
    x = jnp.array([1e9, 1e10, 1e11, 1e12])
    L = jnp.log10(x)
    # y = 0.1 * L^2 - 2 * L + 20
    y = 0.1 * L**2 - 2 * L + 20

    a, b, c = robust_quad_logx(x, y)

    assert abs(a - 0.1) < 0.01
    assert abs(b - (-2)) < 0.1
    assert abs(c - 20) < 0.5


# --- Snapshot test for config generation ---

# Snapshot of expected output for candidates_for_budget with budget=3e18 training FLOPs.
EXPECTED_ISOFLOP_CONFIGS_3E18 = [
    {"batch_size": 32, "train_steps": 32844, "flops_budget": 3e18},
    {"batch_size": 16, "train_steps": 46274, "flops_budget": 3e18},
    {"batch_size": 16, "train_steps": 33965, "flops_budget": 3e18},
    {"batch_size": 8, "train_steps": 48105, "flops_budget": 3e18},
    {"batch_size": 8, "train_steps": 37335, "flops_budget": 3e18},
]


def test_candidates_for_budget_snapshot():
    """Snapshot test: verify candidates_for_budget produces expected configs.

    This ensures reproducibility of the config generation algorithm.
    """
    from experiments.scaling_law_sweeps.c_adamc import CAdamCHeuristic

    recipe = CAdamCHeuristic()
    result = list(recipe.candidates_for_budget(budget=3e18))

    assert len(result) == len(EXPECTED_ISOFLOP_CONFIGS_3E18)

    for i, (candidate, expected) in enumerate(zip(result, EXPECTED_ISOFLOP_CONFIGS_3E18, strict=True)):
        assert isinstance(candidate, CandidateConfig)
        assert candidate.batch_size == expected["batch_size"], f"Config {i}: batch_size mismatch"
        assert candidate.train_steps == expected["train_steps"], f"Config {i}: train_steps mismatch"
        assert candidate.flops_budget == expected["flops_budget"], f"Config {i}: flops_budget mismatch"


# --- End-to-end integration test ---

# Sample tracker_metrics.jsonl data simulating real runs
SAMPLE_METRICS_DATA = [
    # 1e18 budget - 3 runs with U-shaped loss curve
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d1024-L11-B8-nemo",
        "config": {"model": {"hidden_dim": 1024, "num_layers": 11}, "trainer": {"train_batch_size": 8}},
        "summary": {
            "throughput/total_tokens": 1e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.25,
            "parameter_count": 4e8,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d768-L8-B16-nemo",
        "config": {"model": {"hidden_dim": 768, "num_layers": 8}, "trainer": {"train_batch_size": 16}},
        "summary": {
            "throughput/total_tokens": 2.5e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.12,
            "parameter_count": 2.7e8,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+18-d512-L6-B32-nemo",
        "config": {"model": {"hidden_dim": 512, "num_layers": 6}, "trainer": {"train_batch_size": 32}},
        "summary": {
            "throughput/total_tokens": 5e9,
            "throughput/total_gflops": 1e9,
            "eval/paloma/c4_en/bpb": 1.18,
            "parameter_count": 1.5e8,
        },
    },
    # 1e19 budget - 3 runs
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d2048-L21-B16-nemo",
        "config": {"model": {"hidden_dim": 2048, "num_layers": 21}, "trainer": {"train_batch_size": 16}},
        "summary": {
            "throughput/total_tokens": 3e9,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 1.05,
            "parameter_count": 1.8e9,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1536-L16-B32-nemo",
        "config": {"model": {"hidden_dim": 1536, "num_layers": 16}, "trainer": {"train_batch_size": 32}},
        "summary": {
            "throughput/total_tokens": 8e9,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 0.98,
            "parameter_count": 1e9,
        },
    },
    {
        "run_path": "gs://marin/checkpoints/isoflop-1e+19-d1024-L11-B64-nemo",
        "config": {"model": {"hidden_dim": 1024, "num_layers": 11}, "trainer": {"train_batch_size": 64}},
        "summary": {
            "throughput/total_tokens": 2e10,
            "throughput/total_gflops": 1e10,
            "eval/paloma/c4_en/bpb": 1.02,
            "parameter_count": 4e8,
        },
    },
]


def test_end_to_end_analysis_pipeline():
    """Integration test: transform metrics and fit scaling laws end-to-end.

    Uses SAMPLE_METRICS_DATA (simulating real wandb metrics) to verify the full
    pipeline: metrics transformation -> curve fitting -> scaling law extraction.
    """
    from marin.scaling_laws import round_flops_to_bucket
    from experiments.isoflop_sweep import transform_levanter_metrics

    # Transform metrics using the Levanter transform function
    records = transform_levanter_metrics(SAMPLE_METRICS_DATA, "eval/paloma/c4_en/bpb")
    assert len(records) == 6

    # Fit scaling laws
    fit_result = fit_scaling_laws(records)

    # Should find two minima (one per budget: ~1e18 and ~1e19)
    # FLOP values are bucketed by round_flops_to_bucket
    assert len(fit_result.minima_records) == 2

    # Get expected bucketed values
    bucket_1e18 = round_flops_to_bucket(1e18)
    bucket_1e19 = round_flops_to_bucket(1e19)
    assert {rec.flops for rec in fit_result.minima_records} == {bucket_1e18, bucket_1e19}

    # Verify fitted minima are near expected optimal points
    minima_by_flops = {rec.flops: rec for rec in fit_result.minima_records}

    # At ~1e18: raw data optimal at 2.5B tokens (loss=1.12)
    assert abs(minima_by_flops[bucket_1e18].optimal_tokens - 2.6e9) < 0.2e9
    assert abs(minima_by_flops[bucket_1e18].loss_at_optimal - 1.12) < 0.01

    # At ~1e19: raw data optimal at 8B tokens (loss=0.98)
    assert abs(minima_by_flops[bucket_1e19].optimal_tokens - 8.8e9) < 0.2e9
    assert abs(minima_by_flops[bucket_1e19].loss_at_optimal - 0.98) < 0.01


def test_iteration_02_summary_parses_irregular_rerun_name():
    from experiments.grug.moe_scaling_iteration_02.summary.summary import _run_to_cell

    run = FakeWandbRun(
        name="isoflop-moe-adamh-gatednorm-v5p16-eval16-r1-1e+19-d2048",
        group="isoflop-moe-adamh-gatednorm-v5p16-eval16-r1",
        tags=[
            "grug",
            "iteration-02",
            "isoflop",
            "attn-mlp-lmh-adamh",
            "gatednorm",
            "gated_norm_rank=16",
            "budget=1e+19",
            "d=2048",
        ],
        summary={
            "eval/paloma/c4_en/bpb": 1.031,
            "_step": 1000,
            "parameter_count": 123456789,
        },
        config={},
        path=["marin-community", "dial_moe", "abc123"],
        updated_at="2026-03-22T06:11:03Z",
    )

    row = _run_to_cell(run, "eval/paloma/c4_en/bpb")

    assert row is not None
    assert row.variant == "gatednorm"
    assert row.budget == 1e19
    assert row.hidden_dim == 2048
    assert row.metric == 1.031
    assert row.path == "marin-community/dial_moe/abc123"


def test_iteration_02_summary_parses_legacy_adamh_family_name():
    from experiments.grug.moe_scaling_iteration_02.summary.summary import _run_to_cell

    run = FakeWandbRun(
        name="ifv3-it02-attnmlplmh-ahscale-xp1-3e+18-d1024",
        group="ifv3-it02-attnmlplmh-ahscale-xp1",
        tags=[
            "grug",
            "isoflop",
            "attn-mlp-lmh-adamh",
            "budget=3e+18",
            "d=1024",
        ],
        summary={"eval/paloma/c4_en/bpb": 1.0482, "_step": 1000, "parameter_count": 4.0e8},
        config={},
        path=["marin-community", "dial_moe", "legacy"],
        updated_at="2026-03-20T20:00:00Z",
    )

    row = _run_to_cell(run, "eval/paloma/c4_en/bpb")

    assert row is not None
    assert row.variant == "adamh"
    assert row.budget == 3e18
    assert row.hidden_dim == 1024
    assert row.metric == 1.0482


def test_iteration_02_summary_selects_best_duplicate():
    from experiments.grug.moe_scaling_iteration_02.summary.summary import _run_to_cell, select_best_runs

    canonical_failed = FakeWandbRun(
        name="isoflop-moe-adamh-r3-gatedschema-1e+19-d1536-v5p16",
        group="isoflop-moe-adamh-r3-gatedschema",
        tags=[
            "grug",
            "iteration-02",
            "isoflop",
            "attn-mlp-lmh-adamh",
            "budget=1e+19",
            "d=1536",
        ],
        summary={"eval/paloma/c4_en/bpb": 1.02, "_step": 1000, "parameter_count": 1.0e9},
        config={},
        state="failed",
        path=["marin-community", "dial_moe", "old"],
        updated_at="2026-03-21T21:00:00Z",
    )
    retry_finished = FakeWandbRun(
        name="isoflop-moe-adamh-r3-gatedschema-v5p8-retry2-missing-1e+19-d1536",
        group="isoflop-moe-adamh-r3-gatedschema-v5p8-retry2-missing",
        tags=[
            "grug",
            "iteration-02",
            "isoflop",
            "attn-mlp-lmh-adamh",
            "retry",
            "retry2",
            "source=isoflop-moe-adamh-r3-gatedschema",
            "budget=1e+19",
            "d=1536",
        ],
        summary={"eval/paloma/c4_en/bpb": 1.01, "_step": 1000, "parameter_count": 1.0e9},
        config={},
        state="finished",
        path=["marin-community", "dial_moe", "new"],
        updated_at="2026-03-22T21:00:00Z",
    )

    rows = [
        _run_to_cell(canonical_failed, "eval/paloma/c4_en/bpb"),
        _run_to_cell(retry_finished, "eval/paloma/c4_en/bpb"),
    ]
    selected, grouped = select_best_runs([row for row in rows if row is not None])

    assert len(grouped) == 1
    assert len(selected) == 1
    assert selected[0].name == retry_finished.name
