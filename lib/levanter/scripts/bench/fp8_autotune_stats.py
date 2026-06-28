# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy timing statistics for the FP8 ragged-dot autotuner.

Factored out of the benchmark/worker (which imports jax) so the multi-GPU orchestrator can aggregate
results without importing jax — keeping the parent process off the GPUs it is handing to its children.
"""

import numpy as np


def bootstrap_median_ci(times, *, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the median of ``times``. Returns (median, ci_low, ci_high)."""
    times = np.asarray(times, dtype=np.float64)
    median = float(np.median(times))
    if times.size < 2:
        return median, median, median
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, times.size, size=(n_boot, times.size))
    boot_medians = np.median(times[idx], axis=1)
    lo = float(np.quantile(boot_medians, alpha / 2))
    hi = float(np.quantile(boot_medians, 1 - alpha / 2))
    return median, lo, hi


def summarize_times(times):
    """Summary stats for a per-step time sample: median + bootstrap 95% CI, min, mean, std, n."""
    times = np.asarray(times, dtype=np.float64)
    median, lo, hi = bootstrap_median_ci(times)
    return {
        "median_s": median,
        "ci95_low_s": lo,
        "ci95_high_s": hi,
        "ci95_rel_width": (hi - lo) / median if median > 0 else float("nan"),
        "min_s": float(np.min(times)),
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
        "n": int(times.size),
    }


def ratio_median_ci(num_times, den_times, *, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap CI for ``median(den)/median(num)`` — the speedup of num over den (den slower => >1).

    num/den are independent samples (e.g. fp8 vs bf16 per-step times), resampled independently.
    Returns (speedup, ci_low, ci_high) where speedup = median(den)/median(num).
    """
    num = np.asarray(num_times, dtype=np.float64)
    den = np.asarray(den_times, dtype=np.float64)
    speedup = float(np.median(den) / np.median(num))
    rng = np.random.default_rng(seed)
    ni = rng.integers(0, num.size, size=(n_boot, num.size))
    di = rng.integers(0, den.size, size=(n_boot, den.size))
    boot = np.median(den[di], axis=1) / np.median(num[ni], axis=1)
    return speedup, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
