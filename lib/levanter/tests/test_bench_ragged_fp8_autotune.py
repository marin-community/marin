# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the FP8 ragged-dot autotuning benchmark's statistics and config pruning.

These cover the measurement-uncertainty machinery (bootstrap median CI, ratio-of-medians CI, summary
stats) and the shared-memory pruning of block-config candidates — the parts that must be correct
independent of any GPU. The GPU timing/tuning path itself runs only on H100.
"""

import importlib.util
import os
import pathlib

import numpy as np

# Import the script module without triggering the GPU CUDA-toolchain bootstrap.
os.environ["BENCH_SKIP_CUDA_BOOTSTRAP"] = "1"
_SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "bench" / "bench_ragged_fp8_autotune.py"
_spec = importlib.util.spec_from_file_location("bench_ragged_fp8_autotune", _SCRIPT)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_summarize_times_basic_stats():
    times = np.array([1.0, 1.0, 1.0, 1.0])
    s = bench.summarize_times(times)
    assert s["median_s"] == 1.0
    assert s["min_s"] == 1.0
    assert s["std_s"] == 0.0
    assert s["n"] == 4
    # Degenerate (zero-variance) sample: CI collapses to the point.
    assert s["ci95_low_s"] == 1.0 and s["ci95_high_s"] == 1.0


def test_bootstrap_median_ci_brackets_true_median():
    rng = np.random.default_rng(0)
    times = rng.normal(loc=10.0, scale=1.0, size=400)
    median, lo, hi = bench._bootstrap_median_ci(times)
    assert lo < median < hi
    # A 400-sample CI for the median should be tight and contain the true value.
    assert lo < 10.0 < hi
    assert (hi - lo) < 0.5


def test_ratio_median_ci_recovers_known_speedup():
    # bf16 ~2x slower than fp8 -> speedup median(den)/median(num) ~ 2.0.
    rng = np.random.default_rng(1)
    fp8 = rng.normal(loc=1.0, scale=0.02, size=300)
    bf16 = rng.normal(loc=2.0, scale=0.04, size=300)
    speedup, lo, hi = bench.ratio_median_ci(fp8, bf16)
    assert lo < speedup < hi
    assert 1.9 < speedup < 2.1
    assert lo > 1.0  # CI excludes "no speedup" -> a significant win


def test_ci_width_shrinks_with_more_samples():
    rng = np.random.default_rng(2)
    wide = bench._bootstrap_median_ci(rng.normal(10, 1, size=20))
    narrow = bench._bootstrap_median_ci(rng.normal(10, 1, size=2000))
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])


def test_smem_pruning_drops_infeasible_configs():
    # A 256/256/256 tile at 8 stages is ~1MB of f8 smem -> must be pruned from candidates.
    huge = bench._smem_per_stage_f8(256, 256, 256) * 8
    assert huge > bench._H100_SMEM_BYTES
    for cfg in bench._mosaic_candidates():
        used = bench._smem_per_stage_f8(cfg.block_m, cfg.block_n, cfg.block_k) * cfg.max_concurrent_steps
        assert used <= bench._H100_SMEM_BYTES


def test_candidate_lists_nonempty_and_deduped():
    mosaic = bench._mosaic_candidates()
    wgrad = bench._wgrad_candidates()
    assert len(mosaic) >= 8 and len(wgrad) >= 6
    for configs in (mosaic, wgrad):
        keys = [(c.block_m, c.block_n, c.block_k, c.max_concurrent_steps, c.grid_block_n) for c in configs]
        assert len(keys) == len(set(keys))


def test_select_best_skips_rejected_and_failed():
    results = [
        {"cfg": "a", "result": {"error": "numerics", "summary": {"median_s": 0.1}}},  # rejected
        {"cfg": "b", "result": {"error": None, "summary": {"median_s": 0.5}}},
        {"cfg": "c", "result": {"error": None, "summary": {"median_s": 0.3}}},  # best viable
        {"cfg": "d", "result": {"error": "compile", "summary": None}},  # failed
    ]
    best = bench._select_best(results)
    assert best["cfg"] == "c"


def test_select_best_none_when_all_fail():
    results = [{"cfg": "a", "result": {"error": "x", "summary": None}}]
    assert bench._select_best(results) is None
