# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimize a pessimistic bound so the proposal cannot walk off the panel's support.

The failure that has blocked deployment throughout this project is not scoring, it is
optimizing. Every surrogate here scores observed policies well and then, pointed at the
simplex, returns a mixture predicted 0.04 to 0.09 BPB below anything ever observed in the
same KL ball -- 43 to 93 run sigma of optimism the panel cannot contradict. The optimizer
walks into a region where the model is unconstrained and the model has no way to say so.

The band ensemble supplies the missing signal for free. Its members are configurations the
selection criterion cannot separate, so they agree wherever the panel pins the response down
and diverge wherever it does not. Measured: the member spread is a median 0.00101 BPB across
the 280 fit rows, about one run sigma, and 0.09195 BPB at a policy that puts all mass on one
bucket, about 96 run sigma. That is a usable out-of-support detector obtained without fitting
anything new.

So optimize the pessimistic objective

    mean_band(policy) + kappa * spread_band(policy)

instead of the mean alone. On the manifold the penalty is worth roughly one sigma and barely
moves the solution; off it the penalty explodes and pushes the optimizer back. ``kappa`` is
the number of standard deviations of model disagreement the proposal is charged for, so it
sets how much the campaign is willing to bet on regions the panel does not constrain.

The test is whether any ``kappa`` yields an optimum that is *credible*: predicted better than
the best observed policy in the same feasible ball, but not absurdly so. A proposal predicted
40 sigma below everything ever measured is not a proposal, it is an extrapolation artifact.
One predicted a few sigma below the best observed is a bet worth placing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from band_ensemble_20260726 import build_band  # noqa: E402
from dual_objective_harness_20260726 import RUN_SIGMA, aggregate_of, build_benchmark, fit_on  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "pessimistic_optimum_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
KL_BUDGETS = (0.05, 0.25)
RESTARTS = 4
# Cross-entropy method settings. A population search is used rather than Nelder-Mead because
# the objective evaluates 15 band members per policy, and scoring a batch of policies in one
# call is 41x cheaper per policy than scoring them one at a time. Nelder-Mead is sequential
# and cannot exploit that; the full sweep would take about four hours where this takes under
# a minute, for a better optimizer on a 78-dimensional non-smooth objective.
POPULATION = 400
ELITE_FRACTION = 0.10
CEM_ITERATIONS = 60
INITIAL_SIGMA = 0.6
SIGMA_FLOOR = 1e-3
SEED = 20260726
L2_GRID = (0.0, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class BandScorer:
    """Evaluates the band's mean and disagreement at arbitrary policies."""

    fits: tuple
    reference: Panel

    def __call__(self, phase0: np.ndarray, phase1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase0 = np.atleast_2d(np.asarray(phase0, dtype=float))
        phase1 = np.atleast_2d(np.asarray(phase1, dtype=float))
        rows = len(phase0)
        reference = self.reference
        probe = Panel(
            scale="proposal",
            split="proposal",
            alpha=reference.alpha,
            buckets=reference.buckets,
            c0=reference.c0,
            c1=reference.c1,
            family_index=reference.family_index,
            family_names=reference.family_names,
            phase0=phase0,
            phase1=phase1,
            targets={},
            series=np.array(["proposal"] * rows),
            policy_class=np.array(["two_phase"] * rows),
            group=np.arange(rows),
            row_id=np.array([f"proposal_{i}" for i in range(rows)]),
        )
        predictions = np.stack([fit.predict(probe) for fit in self.fits])
        return predictions.mean(axis=0), predictions.std(axis=0)


def kl_to(weights: np.ndarray, reference: np.ndarray) -> float:
    safe = np.clip(weights, 1e-12, None)
    return float((safe * np.log(safe / np.clip(reference, 1e-12, None))).sum())


def propose(
    scorer: BandScorer, reference: np.ndarray, kl_budget: float, kappa: float, tied: bool, alpha: float
) -> dict[str, float]:
    """Minimize mean + kappa * spread over policies inside a KL ball, by cross-entropy method.

    Candidates are parameterized by logits and mapped through a softmax, so the simplex
    constraints hold exactly. The KL budget enters as a quadratic penalty that is raised over
    the run, and the returned policy is rejected unless it satisfies the budget.
    """
    n = len(reference)
    size = n if tied else 2 * n
    base = np.log(np.clip(reference, 1e-12, None))
    rng = np.random.default_rng(SEED)

    def unpack(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shifted = population - population.max(axis=1, keepdims=True)
        if tied:
            weights = np.exp(shifted)
            weights = weights / weights.sum(axis=1, keepdims=True)
            return weights, weights
        first, second = population[:, :n], population[:, n:]
        phase0 = np.exp(first - first.max(axis=1, keepdims=True))
        phase1 = np.exp(second - second.max(axis=1, keepdims=True))
        return phase0 / phase0.sum(axis=1, keepdims=True), phase1 / phase1.sum(axis=1, keepdims=True)

    def score(population: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase0, phase1 = unpack(population)
        mean, spread = scorer(phase0, phase1)
        aggregate = alpha * phase0 + (1.0 - alpha) * phase1
        safe = np.clip(aggregate, 1e-12, None)
        kl = (safe * np.log(safe / np.clip(reference, 1e-12, None)[None, :])).sum(axis=1)
        excess = np.maximum(0.0, kl - kl_budget)
        return mean + kappa * spread + penalty * excess**2, kl, mean + kappa * spread

    best: tuple[float, np.ndarray] | None = None
    for restart in range(RESTARTS):
        mean_vector = np.concatenate([base] * (1 if tied else 2))
        if restart:
            mean_vector = mean_vector + 0.3 * rng.standard_normal(size)
        sigma = np.full(size, INITIAL_SIGMA)
        n_elite = max(4, int(ELITE_FRACTION * POPULATION))
        for iteration in range(CEM_ITERATIONS):
            penalty = 10.0 ** (1.0 + 3.0 * iteration / max(CEM_ITERATIONS - 1, 1))
            population = mean_vector[None, :] + sigma[None, :] * rng.standard_normal((POPULATION, size))
            objective, kl, clean = score(population, penalty)
            elite = population[np.argsort(objective)[:n_elite]]
            mean_vector = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), SIGMA_FLOOR)
        objective, kl, clean = score(mean_vector[None, :], 0.0)
        if kl[0] > kl_budget * 1.02:
            continue
        if best is None or clean[0] < best[0]:
            best = (float(clean[0]), mean_vector)

    assert best is not None, "no restart satisfied the KL budget"
    phase0, phase1 = unpack(best[1][None, :])
    mean, spread = scorer(phase0, phase1)
    aggregate = alpha * phase0[0] + (1.0 - alpha) * phase1[0]
    return {
        "predicted_mean_bpb": float(mean[0]),
        "band_spread_bpb": float(spread[0]),
        "pessimistic_bpb": float(mean[0] + kappa * spread[0]),
        "aggregate_kl": kl_to(aggregate, reference),
        "effective_buckets": float(1.0 / (aggregate**2).sum()),
        "phase_tv": float(0.5 * np.abs(phase1[0] - phase0[0]).sum()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    panel = benchmark.fit_300m
    model = Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True), l2_grid=L2_GRID)
    reference = panel.proportional
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        band = build_band(panel, model, target, weighting="uniform")
        scorer = BandScorer(tuple(fit_on(panel, model, target, m.shape, m.l2) for m in band.members), panel)
        observed = panel.targets[target]
        klv = np.array([kl_to(row, reference) for row in aggregate_of(panel)])
        print(f"{target}: band {band.size} members")

        for kl_budget in KL_BUDGETS:
            inside = klv <= kl_budget
            best_observed = float(np.min(observed[inside])) if inside.any() else float("nan")
            for kappa in KAPPAS:
                for tied in (True, False):
                    result = propose(scorer, reference, kl_budget, kappa, tied, panel.alpha)
                    gap = result["predicted_mean_bpb"] - best_observed
                    rows.append(
                        {
                            "target": target,
                            "kl_budget": kl_budget,
                            "kappa": kappa,
                            "policy_class": "tied" if tied else "two_phase",
                            "n_observed_in_ball": int(inside.sum()),
                            "best_observed_in_ball": best_observed,
                            **result,
                            "gap_to_best_observed_bpb": gap,
                            "gap_in_run_sigma": gap / RUN_SIGMA[target],
                        }
                    )
                print(f"  {target} kl={kl_budget} kappa={kappa} done")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "pessimistic_optima.csv", index=False)

    print("\n=== does a pessimistic objective produce a credible optimum? ===")
    print("gap_in_run_sigma is predicted minus best observed in the same ball; very negative means")
    print("the model claims an improvement the panel cannot support.\n")
    columns = [
        "kappa",
        "policy_class",
        "predicted_mean_bpb",
        "band_spread_bpb",
        "best_observed_in_ball",
        "gap_in_run_sigma",
        "effective_buckets",
        "phase_tv",
    ]
    for target in TARGETS:
        for kl_budget in KL_BUDGETS:
            block = frame[(frame.target == target) & (frame.kl_budget == kl_budget)]
            print(f"### {target}  KL <= {kl_budget}  (n observed in ball = {block.n_observed_in_ball.iloc[0]})")
            print(block[columns].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
            print()

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps({"kappas": list(KAPPAS), "kl_budgets": list(KL_BUDGETS), "restarts": RESTARTS}, indent=2)
    )


if __name__ == "__main__":
    main()
