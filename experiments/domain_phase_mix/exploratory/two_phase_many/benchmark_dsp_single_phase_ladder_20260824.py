# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Build upward from canonical single-phase DSP, one mechanism at a time (ATOM-033).

The general mixture surrogate wins on this task but carries twelve searched dimensions and a two-tier
ridge, which is a lot of machinery to justify. This asks the cheaper question instead: start from the
model the programme actually uses, ablated to one phase, and add only the pieces that earn their place.

Canonical DSP is per-domain and has NO family structure -- at most four parameters per domain and phase
handled by fixed global scalars (`fit_dsp_canonical_variants_300m.py`). Its `PhaseMode.NONE` control sets
the phase multipliers to (0, 1, 1), so with one phase the saturation and penalty exposures both collapse
to total epochs and the model is

    yhat = b0 - sum_b a_b (1 - exp(-rho_b E_b)) + sum_b p_b softplus(log(1 + E_b) - tau_b)^2

with E_b = (c0_b + c1_b) w_b, per-domain rate rho_b and threshold tau_b searched, and amplitudes a_b, p_b
non-negative from the profiled linear head. This is NOT `effective_exposure_dsp` from the model zoo, which
adds a family-summed benefit block canonical DSP does not have, drops the per-domain rate and threshold it
does have, and is absent from the packet cross-check.

Every rung shares one fold structure and one optimiser so a difference is the mechanism and not the
harness; rung `canonical` is therefore canonical DSP fitted with our folds rather than its own. Bucket
classes come only from domain classification and quality splits: the thirteen dolma3_cc topics each carry
a high and a low split, giving a balanced high / low / unsplit partition and thirteen same-domain pairs.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_single_phase_surrogates_20260824 as base  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402

LINEAR_REG = 1e-6
LOG_RATE_BOUND = (float(np.log(1e-4)), float(np.log(2.0)))
THRESHOLD_BOUND = (-2.0, 8.0)
LOG_EXPONENT_BOUND = (float(np.log(0.2)), float(np.log(10.0)))
DAMAGE_KNEE = 105.0
SHRINKAGE = 1.0
N_FOLDS = 3
RESTARTS = 2


def softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(value, 0.0)


def quality_class(bucket: str) -> str:
    """high / low / unsplit -- the only balanced partition the quality splits support (13/13/13)."""
    for suffix in base.QUALITY_SUFFIXES:
        if bucket.endswith(suffix):
            return suffix.lstrip("_")
    return "unsplit"


def benefit(exposure: np.ndarray, log_rate: np.ndarray) -> np.ndarray:
    """Canonical DSP's saturating benefit, one rate per domain."""
    return 1.0 - np.exp(-np.exp(log_rate)[None, :] * exposure)


def canonical_penalty(exposure: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Canonical DSP's over-exposure harm: unbounded, quadratic in a softplus of log epochs."""
    return softplus(np.log1p(exposure) - threshold[None, :]) ** 2


def bounded_penalty(exposure: np.ndarray, log_exponent: np.ndarray) -> np.ndarray:
    """The general surrogate's harm instead: same one-parameter-per-domain budget, but bounded in [0, 1).

    Swapping this in is the sharpest single test available of why DSP mis-selects. Its own penalty grows
    without bound while these panels reach 91 epochs at the median policy and 283x oversampling, and the
    head normalises columns by their TRAINING norm, so a test row past that range is amplified
    quadratically. That is the shape of a model that ranks acceptably and still puts its minimum in the
    wrong place, which is exactly DSP's measured failure.
    """
    unit = np.maximum(exposure - 1.0, 0.0) / DAMAGE_KNEE
    powered = unit ** np.exp(log_exponent)[None, :]
    return powered / (1.0 + powered)


def solve_head(design: np.ndarray, response: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> tuple[float, np.ndarray]:
    """Non-negative amplitudes with an optional same-domain tie, and a free intercept."""
    centre = design.mean(axis=0, keepdims=True)
    target_mean = float(response.mean())
    rows = design - centre
    target = response - target_mean
    width = design.shape[1]
    rows = np.vstack([rows, np.sqrt(LINEAR_REG) * np.eye(width)])
    target = np.concatenate([target, np.zeros(width)])
    if pairs:
        tie = np.zeros((len(pairs), width))
        for row, (first, second) in zip(tie, pairs, strict=True):
            row[first] = np.sqrt(SHRINKAGE)
            row[second] = -np.sqrt(SHRINKAGE)
        rows = np.vstack([rows, tie])
        target = np.concatenate([target, np.zeros(len(pairs))])
    coefficients, _ = nnls(rows, target, maxiter=200 * width)
    return target_mean - float((centre @ coefficients).item()), coefficients


class Rung:
    """One step up from canonical DSP. Only the named mechanism changes."""

    def __init__(self, name: str, per_domain: bool, penalty: str, tie_pairs: bool, note: str):
        self.name = name
        self.per_domain = per_domain
        self.penalty = penalty
        self.tie_pairs = tie_pairs
        self.note = note


LADDER = (
    Rung("canonical", True, "canonical", False, "canonical single-phase DSP, per-domain rate and threshold"),
    Rung("shared_shape", False, "canonical", False, "ablation down: one rate and one threshold for all buckets"),
    Rung("bounded_harm", True, "bounded", False, "swap the unbounded quadratic harm for a bounded one"),
    Rung("canonical+pairs", True, "canonical", True, "tie amplitudes of the two quality splits of a domain"),
    Rung("bounded_harm+pairs", True, "bounded", True, "both additions together"),
)


def rung_design(exposure: np.ndarray, vector: np.ndarray, rung: Rung, n_buckets: int) -> np.ndarray:
    if rung.per_domain:
        rate, harm = vector[:n_buckets], vector[n_buckets:]
    else:
        rate, harm = np.full(n_buckets, vector[0]), np.full(n_buckets, vector[1])
    signal = benefit(exposure, rate)
    penalty = canonical_penalty(exposure, harm) if rung.penalty == "canonical" else bounded_penalty(exposure, harm)
    return np.hstack([-signal, penalty])


def rung_bounds(rung: Rung, n_buckets: int) -> list[tuple[float, float]]:
    harm_bound = THRESHOLD_BOUND if rung.penalty == "canonical" else LOG_EXPONENT_BOUND
    count = n_buckets if rung.per_domain else 1
    return [LOG_RATE_BOUND] * count + [harm_bound] * count


def fit_rung(exposure, response, rung: Rung, folds, pairs, seed: int, maxiter: int, restarts: int = RESTARTS):
    n_buckets = exposure.shape[1]
    box = rung_bounds(rung, n_buckets)
    tied = pairs if rung.tie_pairs else ()

    def objective(vector: np.ndarray) -> float:
        total = 0.0
        for train, test in folds:
            intercept, coefficients = solve_head(
                rung_design(exposure[train], vector, rung, n_buckets), response[train], tied
            )
            predicted = intercept + rung_design(exposure[test], vector, rung, n_buckets) @ coefficients
            residual = predicted - response[test]
            if not np.isfinite(residual).all():
                return 1e6
            total += float(residual @ residual)
        return total

    rng = np.random.default_rng(20260824 + seed)
    lows = np.array([low for low, _ in box])
    highs = np.array([high for _, high in box])
    starts = [0.5 * (lows + highs)]
    starts.extend(rng.uniform(lows, highs) for _ in range(restarts - 1))
    best_vector, best_value = None, np.inf
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=box, options={"maxiter": maxiter})
        if float(result.fun) < best_value:
            best_value, best_vector = float(result.fun), np.asarray(result.x, dtype=float)
    intercept, coefficients = solve_head(rung_design(exposure, best_vector, rung, n_buckets), response, tied)
    return best_vector, intercept, coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="delphi_3e18")
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--restarts", type=int, default=RESTARTS)
    args = parser.parse_args()

    rows = []
    for scale in args.scales.split(","):
        held = base.single_phase_heldout(scale)
        fit_panels = [("two-phase", swarm39.load_scale(scale)[0])]
        if scale in base.ONE_PHASE_DATASET:
            fit_panels.append(("one-phase", base.one_phase_panel(scale)))
        for target in (swarm39.UNCHEATABLE, swarm39.TABLE9):
            query = held.subset(np.isfinite(held.targets[target]))
            measured = query.targets[target]
            query_exposure = (query.c0 + query.c1)[None, :] * query.phase0
            for fit_name, panel in fit_panels:
                usable = np.isfinite(panel.targets[target])
                train = panel.subset(usable)
                exposure = (train.c0 + train.c1)[None, :] * train.phase0
                response = train.targets[target]
                folds = swarm39.mixture_blocked_splits(train, N_FOLDS, seed=0)
                groups: dict[str, list[int]] = {}
                for position, bucket in enumerate(train.buckets):
                    groups.setdefault(base.domain_of(bucket), []).append(position)
                pairs = tuple((members[0], members[1]) for members in groups.values() if len(members) == 2)
                for rung in LADDER:
                    vector, intercept, coefficients = fit_rung(
                        exposure, response, rung, folds, pairs, seed=0, maxiter=args.maxiter, restarts=args.restarts
                    )
                    predicted = intercept + rung_design(query_exposure, vector, rung, exposure.shape[1]) @ coefficients
                    rows.append(
                        {"cell": f"{scale}/{target.split('_')[0]}", "fitted_on": fit_name, "rung": rung.name}
                        | base.score(predicted, measured)
                    )
                    print(f"  done {scale}/{target.split('_')[0]} {fit_name} {rung.name}", flush=True)
    table = pd.DataFrame(rows)
    for cell, group in table.groupby("cell"):
        print(f"\n=== {cell} ===")
        print(group.drop(columns=["cell"]).to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print("\n=== mean across cells ===")
    summary = table.groupby(["rung", "fitted_on"]).agg(
        mean_rho=("spearman", "mean"), mean_regret=("regret@1", "mean"), mean_calibration=("calibration", "mean")
    )
    print(summary.sort_values("mean_rho", ascending=False).to_string(float_format=lambda v: f"{v:+.5f}"))


if __name__ == "__main__":
    main()
