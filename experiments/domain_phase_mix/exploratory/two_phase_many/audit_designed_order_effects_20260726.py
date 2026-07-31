# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Can the surrogate predict the ODD phase-order effect on designed directions?

The new 60M fixed-aggregate panel changes what the phase question is. My earlier bound --
that the incumbent already captures 79 percent of the always-tied-to-clairvoyant span, leaving
1.5 run sigma for any model -- was computed on the 238-pair 300M panel, whose contrasts come
from one qsplit design and are effectively arbitrary directions. The new panel holds the
aggregate, compute, model, seeds, and phase fractions fixed and varies *designed* directions
antithetically, so it isolates the odd part of the response:

    o = (L_plus - L_minus) / 2,     c = (L_plus + L_minus) / 2 - L_tied,

with ``o`` the order effect and ``c`` the cost of the asymmetry itself. It reports order
half-effects of 0.017 to 0.040 BPB in specific mechanistic directions, against an
implied order-half-effect noise of 0.0008 (Uncheatable) and 0.0049 (Table-9), and the
orientation sign reproduces in 4 of 4 fresh-seed sentinel repeats on both targets. That is
signal far larger than my bound contemplated, and it is direction-specific: curated non-CC
later helps, while CC-high later *hurts*, a counterexample to any universal quality-late rule.

The panel's own conclusion is that total phase total variation is insufficient and optimization
must learn mechanism-specific order effects. That is a direct statement about a gap in the
incumbent design, which carries exactly one phase-sensitive column, ``phase_shift_tv``, equal
to half the L1 norm of the contrast. That column is *even*: it takes the same value for both
orientations of an antithetic pair, so it cannot express an order effect at all. Whatever odd
signal the incumbent captures has to arrive through the asymmetry of the retained state.

This script measures three things on the 33 antithetic pairs per anchor and target:

1. how much odd signal the incumbent's design can express at all, by predicting both
   orientations and reading off the implied ``o``;
2. whether the sign of the predicted order effect matches the observed one, which is the
   decision a proposer actually makes;
3. whether the realized value of acting on the model's orientation choice beats always taking
   a fixed orientation.

Nothing here is fitted to this panel. The models are fitted on the 60M swarm fit panel exactly
as elsewhere, and this panel is used only for evaluation, so it is a genuine holdout of designed
directions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from band_ensemble_20260726 import build_band  # noqa: E402
from dual_objective_harness_20260726 import build_benchmark, fit_on, select_by  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model, Panel  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "60m_fixed_aggregate_phase_order_results_20260726"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "designed_order_effects_20260726"

TARGET_COLUMN = {UNCHEATABLE: "uncheatable", TABLE9: "table9"}
L2_GRID = (0.0, 0.01, 0.1, 1.0)
# Implied order-half-effect noise from the panel's fresh-seed sentinel repeats.
ORDER_NOISE = {UNCHEATABLE: 0.000798, TABLE9: 0.004908}


def load_designed_pairs(reference: Panel) -> pd.DataFrame:
    """Join the antithetic decomposition to both orientations' phase weights."""
    observed = pd.read_csv(PANEL_DIR / "observed_results.csv", low_memory=False)
    decomposition = pd.read_csv(PANEL_DIR / "pair_decomposition.csv", low_memory=False)

    phase0 = [f"phase_0_{bucket}" for bucket in reference.buckets]
    phase1 = [f"phase_1_{bucket}" for bucket in reference.buckets]
    missing = [column for column in phase0 + phase1 if column not in observed.columns]
    assert not missing, f"panel is missing weight columns: {missing[:4]}"

    by_candidate = observed.set_index("candidate_id")
    rows = []
    for record in decomposition.itertuples():
        plus, minus = by_candidate.loc[record.plus_candidate_id], by_candidate.loc[record.minus_candidate_id]
        rows.append(
            {
                "target": record.target,
                "anchor_id": record.anchor_id,
                "pair_id": record.pair_id,
                "direction_family": record.direction_family,
                "direction_id": record.direction_id,
                "phase_tv": record.phase_tv,
                "observed_order": record.order_half_effect_plus_minus,
                "observed_cost": record.symmetric_asymmetry_cost,
                "plus_phase0": plus[phase0].to_numpy(float),
                "plus_phase1": plus[phase1].to_numpy(float),
                "minus_phase0": minus[phase0].to_numpy(float),
                "minus_phase1": minus[phase1].to_numpy(float),
            }
        )
    return pd.DataFrame(rows)


def probe_panel(reference: Panel, phase0: np.ndarray, phase1: np.ndarray) -> Panel:
    rows = len(phase0)
    return Panel(
        scale="60m",
        split="designed",
        alpha=reference.alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0,
        phase1=phase1,
        targets={},
        series=np.array(["designed"] * rows),
        policy_class=np.array(["two_phase"] * rows),
        group=np.arange(rows),
        row_id=np.array([f"designed_{i}" for i in range(rows)]),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = build_benchmark()
    fit_panel = benchmark.fit_60m
    model = Model("hpr", build_hierarchical_phase_replay, lambda: _state_shapes(True), l2_grid=L2_GRID)
    pairs = load_designed_pairs(fit_panel)
    rows: list[dict[str, Any]] = []

    for target, label in TARGET_COLUMN.items():
        block = pairs[pairs.target == label].reset_index(drop=True)
        if block.empty:
            continue
        plus = probe_panel(fit_panel, np.stack(block.plus_phase0.to_numpy()), np.stack(block.plus_phase1.to_numpy()))
        minus = probe_panel(fit_panel, np.stack(block.minus_phase0.to_numpy()), np.stack(block.minus_phase1.to_numpy()))

        # Confirm the claim that the only phase-sensitive column is even in the contrast.
        shape, l2, _ = select_by(fit_panel, model, target, "rmse")
        design_plus = build_hierarchical_phase_replay(plus, shape)
        design_minus = build_hierarchical_phase_replay(minus, shape)
        column_gap = np.abs(design_plus.matrix - design_minus.matrix).max(axis=0)
        tv_index = design_plus.names.index("phase_shift_tv")

        predictors: dict[str, np.ndarray] = {}
        fitted = fit_on(fit_panel, model, target, shape, l2)
        predictors["argmin"] = 0.5 * (fitted.predict(plus) - fitted.predict(minus))
        band = build_band(fit_panel, model, target, weighting="stacked")
        predictors["band_stacked"] = 0.5 * (band.predict(fit_panel, None, plus) - band.predict(fit_panel, None, minus))

        observed_order = block.observed_order.to_numpy(float)
        for name, predicted in predictors.items():
            # Realized value of acting on the model's orientation call, against the two fixed
            # policies of always taking plus and always taking minus.
            realized = float(np.mean(np.where(predicted < 0.0, observed_order, -observed_order)))
            oracle = float(np.mean(-np.abs(observed_order)))
            rows.append(
                {
                    "target": label,
                    "predictor": name,
                    "n_pairs": len(block),
                    "phase_shift_tv_column_gap": float(column_gap[tv_index]),
                    "max_column_gap": float(column_gap.max()),
                    "predicted_order_sd": float(np.std(predicted)),
                    "observed_order_sd": float(np.std(observed_order)),
                    "order_noise_sd": ORDER_NOISE[target],
                    "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(observed_order))),
                    "pearson": float(np.corrcoef(predicted, observed_order)[0, 1]),
                    "spearman": float(
                        np.corrcoef(np.argsort(np.argsort(predicted)), np.argsort(np.argsort(observed_order)))[0, 1]
                    ),
                    "realized_order_bpb": realized,
                    "oracle_order_bpb": oracle,
                    "always_plus_bpb": float(np.mean(observed_order)),
                    "order_skill": float(realized / oracle) if oracle < 0.0 else float("nan"),
                }
            )

        # Per-family breakdown of the observed effect, for context on what must be learned.
        for (family, direction), group in block.groupby(["direction_family", "direction_id"]):
            rows.append(
                {
                    "target": label,
                    "predictor": f"observed:{family}/{direction}",
                    "n_pairs": len(group),
                    "observed_order_sd": float(np.std(group.observed_order)),
                    "realized_order_bpb": float(np.mean(group.observed_order)),
                    "order_noise_sd": ORDER_NOISE[target],
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "order_effect_prediction.csv", index=False)

    print("=== is the only phase column even in the contrast? ===")
    check = frame[frame.predictor.isin(["argmin", "band_stacked"])]
    print(
        check[["target", "predictor", "phase_shift_tv_column_gap", "max_column_gap"]].to_string(
            index=False, float_format=lambda v: f"{v:.3e}"
        )
    )
    print("\n=== can the model predict the designed order effect? ===")
    columns = [
        "target",
        "predictor",
        "n_pairs",
        "predicted_order_sd",
        "observed_order_sd",
        "order_noise_sd",
        "sign_accuracy",
        "pearson",
        "spearman",
        "realized_order_bpb",
        "oracle_order_bpb",
        "order_skill",
    ]
    print(check[columns].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n=== observed effect by designed direction (what would have to be learned) ===")
    observed_rows = frame[frame.predictor.str.startswith("observed:")]
    print(
        observed_rows[["target", "predictor", "n_pairs", "realized_order_bpb", "observed_order_sd"]].to_string(
            index=False, float_format=lambda v: f"{v:.5f}"
        )
    )

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(
            {
                "source_panel": str(PANEL_DIR.name),
                "order_noise_sd": ORDER_NOISE,
                "note": "models are fitted on the 60M swarm fit panel only; this designed panel is evaluation",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
