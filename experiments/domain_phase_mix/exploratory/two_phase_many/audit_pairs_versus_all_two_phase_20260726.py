# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""At a fixed number of training runs, is it better to train pairs or all two-phase policies?

This is the one design question left open by the decoupling experiment. There, a joint head
given 140 exposure-matched pairs reached phase decision skill 0.932 on Uncheatable and 0.864
on Table-9, far above anything else measured in this project. That number was not comparable
to the 0.784 incumbent figure, because the incumbent's phase arm is fitted at 60M and scored
on all 238 pairs while the pairs arm was fitted at 300M and scored on a 75-pair holdout. The
comparison here removes that confound by putting both designs through one protocol.

The trade is about what a run buys. A two-phase row costs one run and yields one observation
of ``L(a, d)``. A pair costs two runs and yields two observations, ``L(a, d)`` and
``L(a, 0)``, and therefore also the difference ``Delta`` that the phase decision depends on.
Spending the budget on pairs halves the number of distinct aggregates seen but makes the
phase contrast directly observable rather than something the model has to disentangle from
policies that all carry contrast.

Two designs at an identical run budget, both fitting one joint head with the same selection
protocol:

``all_two_phase``  every run is a two-phase policy. This is how the existing 280-row panel
                   was built.
``pairs``          half the runs are two-phase policies and half are their exposure-matched
                   tied twins.

Evaluation is identical for both and uses rows neither design ever trains on: a censored set
of the best policies by observed loss, scored on both members so neither design is favoured
by the policy class it happens to see, and a held-out set of pairs for phase decision skill.
Allocations are redrawn several times so the comparison is not about which particular rows
were sampled.
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

from dual_objective_harness_20260726 import build_benchmark  # noqa: E402
from phase_order_spine_20260725 import load_paired_panel  # noqa: E402
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Panel, fit_head, grouped_splits  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "pairs_versus_all_two_phase_20260726"

TARGETS = (UNCHEATABLE, TABLE9)
TIED_PREFIX = r"^singleavg_"
CENSOR_FRACTION = 0.10
PHASE_HOLDOUT = 60
# Sweep the budget so the comparison is not read off a single run count. None means "as
# many runs as the pool allows", which is the largest matched budget available.
BUDGETS = (60, 100, 140, None)
DRAWS = 10
SEED = 20260726
L2 = 0.0
N_SPLITS = 5
SPLIT_SEED = 0


@dataclass(frozen=True)
class Pairs:
    phase0: np.ndarray
    phase1: np.ndarray
    aggregate: np.ndarray
    tied: dict[str, np.ndarray]
    two_phase: dict[str, np.ndarray]
    delta: dict[str, np.ndarray]
    reference: Panel

    def __len__(self) -> int:
        return len(self.phase0)


def load_pairs(reference: Panel) -> Pairs:
    paired = load_paired_panel("300m", TIED_PREFIX)
    assert paired.buckets == reference.buckets, "bucket order mismatch"
    return Pairs(
        phase0=paired.phase0,
        phase1=paired.phase1,
        aggregate=paired.aggregate,
        tied=dict(paired.tied_bpb),
        two_phase=dict(paired.two_phase_bpb),
        delta=dict(paired.delta),
        reference=reference,
    )


def panel_of(pairs: Pairs, phase0: np.ndarray, phase1: np.ndarray, target: str, values: np.ndarray) -> Panel:
    rows = len(phase0)
    reference = pairs.reference
    return Panel(
        scale="300m",
        split="design",
        alpha=reference.alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0,
        phase1=phase1,
        targets={target: values},
        series=np.array(["design"] * rows),
        policy_class=np.array(["two_phase"] * rows),
        group=np.arange(rows),
        row_id=np.array([f"row_{i}" for i in range(rows)]),
    )


def training_panel(pairs: Pairs, target: str, design: str, index: np.ndarray, budget: int) -> Panel:
    """Build the training panel a design buys with ``budget`` runs."""
    if design == "all_two_phase":
        chosen = index[:budget]
        return panel_of(pairs, pairs.phase0[chosen], pairs.phase1[chosen], target, pairs.two_phase[target][chosen])
    chosen = index[: budget // 2]
    phase0 = np.vstack([pairs.phase0[chosen], pairs.aggregate[chosen]])
    phase1 = np.vstack([pairs.phase1[chosen], pairs.aggregate[chosen]])
    values = np.concatenate([pairs.two_phase[target][chosen], pairs.tied[target][chosen]])
    return panel_of(pairs, phase0, phase1, target, values)


def select_shape(panel: Panel, target: str) -> dict:
    """Grouped out-of-fold RMSE over the shape grid, on the design's own training rows."""
    observed = panel.targets[target]
    best: tuple[float, dict] | None = None
    for shape in _state_shapes(True):
        design = build_hierarchical_phase_replay(panel, shape).matrix
        prediction = np.full(len(observed), np.nan)
        for train, test in grouped_splits(panel, N_SPLITS, SPLIT_SEED):
            if train.sum() < 2:
                continue
            intercept, coefficients = fit_head(design[train], observed[train], L2)
            prediction[test] = intercept + design[test] @ coefficients
        finite = np.isfinite(prediction)
        score = float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2)))
        if best is None or score < best[0]:
            best = (score, shape)
    assert best is not None
    return best[1]


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def evaluate(
    pairs: Pairs,
    target: str,
    design: str,
    index: np.ndarray,
    budget: int,
    censored: np.ndarray,
    phase_holdout: np.ndarray,
) -> dict[str, float]:
    panel = training_panel(pairs, target, design, index, budget)
    shape = select_shape(panel, target)
    matrix = build_hierarchical_phase_replay(panel, shape).matrix
    intercept, coefficients = fit_head(matrix, panel.targets[target], L2)

    def predict(phase0: np.ndarray, phase1: np.ndarray) -> np.ndarray:
        probe = panel_of(pairs, phase0, phase1, target, np.zeros(len(phase0)))
        return intercept + build_hierarchical_phase_replay(probe, shape).matrix @ coefficients

    scores: dict[str, float] = {"n_runs": float(budget), "n_train_rows": float(len(panel.row_id))}
    # Censored extrapolation, scored on BOTH members so neither design is favoured by the
    # policy class it happens to train on.
    for label, phase0, phase1, truth in (
        ("tied", pairs.aggregate[censored], pairs.aggregate[censored], pairs.tied[target][censored]),
        ("two", pairs.phase0[censored], pairs.phase1[censored], pairs.two_phase[target][censored]),
    ):
        prediction = predict(phase0, phase1)
        residual = prediction - truth
        scores[f"cens_{label}_rmse"] = float(np.sqrt(np.mean(residual**2)))
        scores[f"cens_{label}_bias"] = float(np.mean(residual))
        scores[f"cens_{label}_spearman"] = float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1])

    predicted_delta = predict(pairs.phase0[phase_holdout], pairs.phase1[phase_holdout]) - predict(
        pairs.aggregate[phase_holdout], pairs.aggregate[phase_holdout]
    )
    skill = phase_decision_skill(predicted_delta, pairs.delta[target][phase_holdout])
    scores["phase_skill"] = float(skill["phase_skill_score"])
    scores["phase_accuracy"] = float(skill["decision_accuracy"])
    scores["phase_delta_corr"] = float(skill["delta_correlation"])
    return scores


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(build_benchmark().fit_300m)
    total = len(pairs)
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        n_censored = max(1, int(CENSOR_FRACTION * total))
        censored = np.argsort(pairs.tied[target])[:n_censored]
        remaining = np.setdiff1d(np.arange(total), censored)

        for draw in range(DRAWS):
            rng = np.random.default_rng(SEED + draw)
            shuffled = rng.permutation(remaining)
            phase_holdout = shuffled[:PHASE_HOLDOUT]
            pool = shuffled[PHASE_HOLDOUT:]
            # Budget is the largest even number of runs both designs can afford: the
            # all-two-phase design is capped by the pool size.
            for requested in BUDGETS:
                budget = (len(pool) // 2) * 2 if requested is None else requested
                if budget > len(pool):
                    continue
                for design in ("all_two_phase", "pairs"):
                    rows.append(
                        {
                            "target": target,
                            "design": design,
                            "budget": budget,
                            "draw": draw,
                            **evaluate(pairs, target, design, pool, budget, censored, phase_holdout),
                        }
                    )
            print(f"  {target} draw {draw + 1}/{DRAWS}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "design_comparison.csv", index=False)

    print("\n=== mean over draws, identical run budget and identical evaluation rows ===")
    columns = [
        "n_runs",
        "n_train_rows",
        "cens_tied_rmse",
        "cens_tied_bias",
        "cens_tied_spearman",
        "cens_two_rmse",
        "cens_two_bias",
        "cens_two_spearman",
        "phase_skill",
        "phase_accuracy",
    ]
    print(frame.groupby(["target", "budget", "design"])[columns].mean().to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n=== paired over draws: pairs minus all_two_phase ===")
    summary = []
    metrics = [c for c in columns if c.startswith(("cens_", "phase_"))]
    for (target, budget), group in frame.groupby(["target", "budget"]):
        a = group[group.design == "all_two_phase"].set_index("draw")
        b = group[group.design == "pairs"].set_index("draw")
        for metric in metrics:
            delta = (b[metric] - a[metric]).to_numpy()
            better = (
                np.abs(b[metric]) < np.abs(a[metric])
                if "bias" in metric
                else (b[metric] > a[metric] if ("spearman" in metric or "phase" in metric) else b[metric] < a[metric])
            )
            summary.append(
                {
                    "target": target,
                    "budget": budget,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "ci95_low": float(np.quantile(delta, 0.025)),
                    "ci95_high": float(np.quantile(delta, 0.975)),
                    "fraction_better": float(np.mean(better)),
                }
            )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    print(summary_frame.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "n_pairs_available": total,
                "censor_fraction": CENSOR_FRACTION,
                "phase_holdout": PHASE_HOLDOUT,
                "draws": DRAWS,
                "budgets": [b for b in BUDGETS if b is not None] + ["pool_max"],
                "note": (
                    "both designs spend the same number of RUNS; a pair costs two runs and yields "
                    "two observations, a two-phase row costs one run and yields one"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
