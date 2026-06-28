# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the FP8 ragged-dot autotuner's jax-free support modules.

Covers the measurement-uncertainty machinery (bootstrap median CI, ratio-of-medians CI), the
shared-memory pruning of the block-config candidate space, and the multi-GPU orchestrator's pure
planning/selection logic. These must be correct independent of any GPU; the GPU timing/tuning path
itself runs only on H100. The modules import cleanly without jax, so these stay fast.
"""

import pathlib
import sys

import numpy as np

_BENCH = pathlib.Path(__file__).parents[1] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH))

import fp8_autotune_configs as cfg  # noqa: E402
import fp8_autotune_stats as stats  # noqa: E402
import orchestrate_fp8_autotune as orch  # noqa: E402


# ---- statistics ----


def test_summarize_times_basic_stats():
    s = stats.summarize_times(np.array([1.0, 1.0, 1.0, 1.0]))
    assert s["median_s"] == 1.0 and s["min_s"] == 1.0 and s["std_s"] == 0.0 and s["n"] == 4
    assert s["ci95_low_s"] == 1.0 and s["ci95_high_s"] == 1.0  # zero-variance => point CI


def test_bootstrap_median_ci_brackets_true_median():
    times = np.random.default_rng(0).normal(10.0, 1.0, size=400)
    median, lo, hi = stats.bootstrap_median_ci(times)
    assert lo < median < hi and lo < 10.0 < hi and (hi - lo) < 0.5


def test_ratio_median_ci_recovers_known_speedup():
    rng = np.random.default_rng(1)
    fp8 = rng.normal(1.0, 0.02, size=300)
    bf16 = rng.normal(2.0, 0.04, size=300)
    speedup, lo, hi = stats.ratio_median_ci(fp8, bf16)
    assert lo < speedup < hi and 1.9 < speedup < 2.1 and lo > 1.0  # CI excludes "no speedup"


def test_ci_width_shrinks_with_more_samples():
    rng = np.random.default_rng(2)
    wide = stats.bootstrap_median_ci(rng.normal(10, 1, size=20))
    narrow = stats.bootstrap_median_ci(rng.normal(10, 1, size=2000))
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])


# ---- config space / smem pruning ----


def test_smem_pruning_drops_infeasible_configs():
    huge = cfg.smem_per_stage_f8(256, 256, 256) * 8
    assert huge > cfg.H100_SMEM_BYTES
    for c in cfg.mosaic_candidate_dicts() + cfg.wgrad_candidate_dicts():
        used = cfg.smem_per_stage_f8(c["block_m"], c["block_n"], c["block_k"]) * c["max_concurrent_steps"]
        assert used <= cfg.H100_SMEM_BYTES


def test_candidate_lists_nonempty_and_deduped():
    mosaic, wgrad, bf16 = cfg.mosaic_candidate_dicts(), cfg.wgrad_candidate_dicts(), cfg.bf16_candidate_dicts()
    assert len(mosaic) >= 8 and len(wgrad) >= 6 and len(bf16) >= 8
    for configs in (mosaic, wgrad):
        keys = [tuple(c.values()) for c in configs]
        assert len(keys) == len(set(keys))


def test_shape_grid_matches_known_target():
    t = cfg.SHAPE_GRID["target"]
    assert (t.tokens, t.hidden, t.intermediate, t.experts) == (8192, 2048, 5632, 8)


# ---- orchestrator planning / selection ----


def test_chunk_covers_all_items_in_n_pieces():
    items = list(range(10))
    chunks = orch._chunk(items, 3)
    assert len(chunks) == 3 and sum(chunks, []) == items
    # asking for more chunks than items never yields empty chunks
    assert all(len(c) >= 1 for c in orch._chunk(items, 100))


def test_plan_units_single_shape_and_bounded():
    reqs = {"target": [{"id": f"t{i}"} for i in range(19)], "small": [{"id": f"s{i}"} for i in range(11)]}
    units = orch.plan_units(reqs, num_gpus=32, max_reqs_per_worker=4)
    # every unit is single-shape and within the per-worker bound
    for u in units:
        assert len({r["id"][0] for r in u["requests"]}) == 1
        assert 1 <= len(u["requests"]) <= 4
    # all requests are covered exactly once
    covered = [r["id"] for u in units for r in u["requests"]]
    assert sorted(covered) == sorted(r["id"] for rs in reqs.values() for r in rs)


def test_plan_units_respects_max_when_few_gpus():
    reqs = {"target": [{"id": f"t{i}"} for i in range(19)]}
    units = orch.plan_units(reqs, num_gpus=1, max_reqs_per_worker=4)
    assert all(len(u["requests"]) <= 4 for u in units)
    assert sum(len(u["requests"]) for u in units) == 19


def test_best_picks_min_median_and_skips_errors():
    rows = [
        {"request_id": "a|x", "kind": "bf16", "error": None, "steady_state_time_s": 0.5},
        {"request_id": "a|y", "kind": "bf16", "error": None, "steady_state_time_s": 0.3},
        {"request_id": "a|z", "kind": "bf16", "error": "boom", "steady_state_time_s": 0.1},  # failed: ignore
        {"request_id": "a|w", "kind": "fp8", "error": None, "steady_state_time_s": 0.2},
    ]
    best = orch._best(rows, lambda r: r["kind"] == "bf16")
    assert best["request_id"] == "a|y"
    assert orch._best(rows, lambda r: r["kind"] == "nope") is None
