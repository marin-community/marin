# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Does the frozen MARINER fit predict the frontier factorial's measured local effects?

Refits the paper's MARINER objective fits from the reference package (as
`materialize_delphi_comparator_proposals_20260909` does) and predicts every design of the 2^(5-1) frontier factorial
(`delphi_frontier_factorial_design_20260906`). The same effects model that the factorial report fits to the measured
corners is fitted to the predictions, so each main effect and two-factor interaction has a predicted and a measured
value. Also records calibration at the centre and the corners.

Writes `predicted_vs_measured.csv`, `effects_comparison.csv` and `summary.json` to
`reference_outputs/factorial_vs_mariner_20260928/`.

usage: uv run --offline --no-sync python analyze_factorial_vs_mariner_20260928.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import analyze_delphi_frontier_factorial_20260906 as factorial  # noqa: E402
import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402
from uncheatable_objective import uncheatable_weights  # noqa: E402

DESIGN = SCRIPT_DIR / "reference_outputs" / "delphi_frontier_factorial_design_20260906"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "factorial_vs_mariner_20260928"
TARGETS = {"table9": "table9_macro_bpb", "uncheatable": "uncheatable_bpb"}
# Repeat SD per objective at the centre, from the factorial report (26-run centre; Uncheatable centre pair).
REPEAT_SD = {"table9": factorial.CENTRE_SD, "uncheatable": 0.0009}


def measured_responses() -> pd.DataFrame:
    table = pd.read_csv(DESIGN / "measured_results.csv")
    table = table[table.status.eq("measured")]
    fixed = sum(w * table[f"uncheatable_{key.split('/')[2]}_bpb"] for key, w in uncheatable_weights().items())
    if (fixed - table.uncheatable_bpb).abs().max() > 1e-6:
        raise ValueError("factorial Uncheatable values are not the fixed weighting")
    return table.groupby("candidate_id")[list(TARGETS.values())].first()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = mat.bench.load_panel(mat.PANEL)
    buckets = [str(bucket) for bucket in panel.buckets]
    swarm, to_panel = mat.reference_swarm_in_panel_order(panel)
    from_panel = np.argsort(to_panel)
    weights = pd.read_csv(DESIGN / "candidate_weights.csv").pivot(
        index="candidate_id", columns="domain", values="weight"
    )
    weights = weights[buckets]
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("factorial weights do not sum to one")
    design = pd.read_csv(DESIGN / "design.csv")
    measured = measured_responses().loc[weights.index]
    factors = [column for column in design.columns if column.startswith("factor_")]
    is_corner = design.set_index("candidate_id")[factors].abs().eq(1).all(axis=1).loc[weights.index]

    rows, effects, summary = [], [], {}
    for target, column in TARGETS.items():
        fit = mat.reference_fit(swarm, target, None)
        predicted = pd.Series(fit.predict(weights.to_numpy()[:, from_panel]), index=weights.index, name="predicted")
        for candidate in weights.index:
            rows.append(
                {
                    "target": target,
                    "candidate_id": candidate,
                    "measured_bpb": float(measured.loc[candidate, column]),
                    "predicted_bpb": float(predicted[candidate]),
                    "is_corner": bool(is_corner[candidate]),
                }
            )
        measured_effects = factorial.effects_table(design, measured[column], REPEAT_SD[target])
        predicted_effects = factorial.effects_table(design, predicted, REPEAT_SD[target])
        merged = measured_effects.merge(
            predicted_effects[["term", "effect_bpb"]].rename(columns={"effect_bpb": "predicted_effect_bpb"}), on="term"
        )
        merged.insert(0, "target", target)
        effects.append(merged)
        main = merged[merged.kind.eq("main")]
        corners = predicted.index[is_corner]
        centre = weights.index[~is_corner]
        m, p = measured.loc[corners, column].to_numpy(), predicted[corners].to_numpy()
        i, j = np.triu_indices(len(corners), 1)
        summary[target] = {
            "centre_measured_mean": float(measured.loc[centre, column].mean()),
            "centre_predicted_mean": float(predicted[centre].mean()),
            "corner_rmse": float(np.sqrt(np.mean((m - p) ** 2))),
            "corner_bias_predicted_minus_measured": float(np.mean(p - m)),
            "corner_spearman": float(stats.spearmanr(m, p).statistic),
            "corner_pair_sign_accuracy": float(np.mean(np.sign(m[i] - m[j]) == np.sign(p[i] - p[j]))),
            "main_effect_sign_agreement": int((np.sign(main.effect_bpb) == np.sign(main.predicted_effect_bpb)).sum()),
            "main_effects_detected_at_2se": int((main.t.abs() >= factorial.MAIN_EFFECT_T).sum()),
            "detected_effects_with_agreeing_sign": int(
                (
                    (main.t.abs() >= factorial.MAIN_EFFECT_T)
                    & (np.sign(main.effect_bpb) == np.sign(main.predicted_effect_bpb))
                ).sum()
            ),
            "predicted_over_measured_slope": float(
                np.polyfit(main.effect_bpb.to_numpy(), main.predicted_effect_bpb.to_numpy(), 1)[0]
            ),
        }
        print(f"== {target}")
        print(merged[["term", "kind", "effect_bpb", "se", "t", "predicted_effect_bpb"]].round(5).to_string(index=False))
        print(json.dumps(summary[target], indent=1))
    pd.DataFrame(rows).to_csv(OUTPUT / "predicted_vs_measured.csv", index=False)
    pd.concat(effects, ignore_index=True).to_csv(OUTPUT / "effects_comparison.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
