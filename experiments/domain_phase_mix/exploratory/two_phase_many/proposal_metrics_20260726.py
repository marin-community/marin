# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metrics for judging a surrogate as a *proposer*, not as a regressor.

Pooled RMSE and global Spearman are actively misleading here. At 3e18 every model
reaches Spearman near 0.88 on the archive while its top-1 pick inside the
kappa=0.5 ball lands at the 51st to 64th percentile, because the top 5 percent of
that ball spans 0.0026 BPB against a run sigma of 0.00096. Ranking bad policies
below good ones is easy; resolving the top is the job.

Five measures, each answering a question the others cannot.

``top_k_percentile``
    Where the best of a model's k best feasible picks sits in the feasible
    distribution. Rank-based, so immune to the noise scale, and k>1 because a
    real campaign trains a handful of candidates rather than one.

``regret_in_sigma``
    What that pick costs in BPB, divided by run sigma so Uncheatable (0.00096)
    and Table-9 (0.0031) are comparable. Referenced to the 5th percentile rather
    than the minimum: a noise-selected winner averages 32 percent of the
    best-to-p5 spread below the true best on Uncheatable.

``phase_decision_skill``
    The question this project exists to answer, and the one the two ranking
    metrics cannot see. On exact aggregate-matched pairs the estimand
    ``Delta = L(a,d) - L(a,0)`` is observed directly. Scored by realized value
    against the always-tied policy, because 40 to 52 percent of two-phase
    policies beat their tied twin so accuracy alone is nearly uninformative, and
    because the payoff is dominated by a fat left tail: the best observed Delta
    is -0.032 BPB at 3e18 Uncheatable and -0.094 at Table-9.

``proposal_stability``
    Whether the proposal survives resampling the panel it was fitted on. A single
    number you would act on is worthless if it moves under a bootstrap.

``optimum_sanity``
    A pass/fail gate, never a score. Ranking well and being safe to solve are
    independent: the real DSP picks at the 5th percentile under kappa=0.25 while
    its continuous optimum collapses onto 1.93 of 39 buckets.
"""

from __future__ import annotations

import numpy as np

# A proposal round trains a few candidates, so the operative quantity is the best
# of the top k rather than a single argmin.
DEFAULT_TOP_K = 5
REFERENCE_QUANTILE = 0.05
# Gate thresholds. Effective buckets counts 1/sum(w^2); 39 buckets are available
# and the proportional policy scores 39, so 5 is already extremely concentrated.
MIN_EFFECTIVE_BUCKETS = 5.0
MAX_SIMULATED_EPOCHS = 50.0


def top_k_percentile(
    predicted: np.ndarray, observed: np.ndarray, feasible: np.ndarray, k: int = DEFAULT_TOP_K
) -> dict[str, float]:
    """Where the best of the model's k best feasible picks lands, 0 = best available."""
    p, o = predicted[feasible], observed[feasible]
    order = np.argsort(p)[:k]
    realized = float(np.min(o[order]))
    return {
        "n_feasible": int(feasible.sum()),
        "top_k": k,
        "picked_observed_bpb": realized,
        "picked_percentile": float((o < realized).mean()),
        "top1_percentile": float((o < o[int(np.argmin(p))]).mean()),
    }


def regret_in_sigma(
    predicted: np.ndarray, observed: np.ndarray, feasible: np.ndarray, sigma: float, k: int = DEFAULT_TOP_K
) -> dict[str, float]:
    """Cost of the model's pick against the feasible 5th percentile, in run sigma."""
    p, o = predicted[feasible], observed[feasible]
    realized = float(np.min(o[np.argsort(p)[:k]]))
    reference = float(np.quantile(o, REFERENCE_QUANTILE))
    return {
        "feasible_p5_observed_bpb": reference,
        "feasible_best_observed_bpb": float(np.min(o)),
        "regret_vs_p5_bpb": realized - reference,
        "regret_vs_p5_sigma": (realized - reference) / sigma,
    }


def phase_decision_skill(predicted_delta: np.ndarray, observed_delta: np.ndarray) -> dict[str, float]:
    """Realized value of the model's two-phase-versus-tied calls, against always-tied.

    Acting on the model means running two phases exactly when it predicts
    ``Delta < 0``. Realized value is the observed ``Delta`` on those rows and zero
    on the rest, so always-tied scores 0 and a clairvoyant scores
    ``mean(min(Delta, 0))``. The skill score normalizes between them: 1 is oracle,
    0 is no better than never using two phases, negative is worse than that.

    Reported alongside raw accuracy because accuracy is inflated by the base rate
    and does not weight the fat left tail that carries the payoff.
    """
    chosen = predicted_delta < 0.0
    realized = float(np.mean(np.where(chosen, observed_delta, 0.0)))
    oracle = float(np.mean(np.minimum(observed_delta, 0.0)))
    beats_tied = observed_delta < 0.0
    return {
        "n_pairs": len(observed_delta),
        "base_rate_two_phase_wins": float(beats_tied.mean()),
        "always_tied_realized_bpb": 0.0,
        "oracle_realized_bpb": oracle,
        "model_realized_bpb": realized,
        # Guard the degenerate case where no pair benefits from two phases.
        "phase_skill_score": float(realized / oracle) if oracle < 0.0 else float("nan"),
        "decision_accuracy": float((chosen == beats_tied).mean()),
        "always_tied_accuracy": float((~beats_tied).mean()),
        "fraction_proposed_two_phase": float(chosen.mean()),
        "delta_correlation": float(np.corrcoef(predicted_delta, observed_delta)[0, 1]),
    }


def proposal_stability(picks: list[np.ndarray], percentiles: list[float]) -> dict[str, float]:
    """Agreement between the top-k sets recovered from resampled panels."""
    overlaps = [len(np.intersect1d(a, b)) / len(np.union1d(a, b)) for i, a in enumerate(picks) for b in picks[i + 1 :]]
    stacked = np.concatenate([np.asarray(p) for p in picks])
    _, counts = np.unique(stacked, return_counts=True)
    return {
        "draws": len(picks),
        "mean_pairwise_jaccard": float(np.mean(overlaps)) if overlaps else float("nan"),
        "distinct_policies_ever_picked": len(counts),
        "modal_pick_frequency": float(counts.max() / len(picks)),
        "percentile_mean": float(np.mean(percentiles)),
        "percentile_sd": float(np.std(percentiles, ddof=1)) if len(percentiles) > 1 else 0.0,
        "percentile_p90": float(np.quantile(percentiles, 0.9)),
    }


def optimum_sanity(aggregate: np.ndarray, epochs: np.ndarray, support_distance: float) -> dict[str, float | bool]:
    """Pass/fail gate on a continuous optimum. Not a score; a model either ships or does not."""
    effective_buckets = float(1.0 / (aggregate**2).sum())
    max_epochs = float(epochs.max())
    return {
        "effective_buckets": effective_buckets,
        "max_simulated_epochs": max_epochs,
        "support_distance": support_distance,
        "passes_concentration": effective_buckets >= MIN_EFFECTIVE_BUCKETS,
        "passes_epochs": max_epochs <= MAX_SIMULATED_EPOCHS,
        "passes_gate": effective_buckets >= MIN_EFFECTIVE_BUCKETS and max_epochs <= MAX_SIMULATED_EPOCHS,
    }
