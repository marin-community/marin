# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Re-adjudicate every candidate on phase-decision skill, fit at 300M, judged at 3e18.

The candidate set was previously ranked on low-predicted-tail RMSE. That metric
just proved misleading for this purpose: crs_plus wins low-tail in four of six
scale-by-target cells yet ranks last on Table-9 phase skill, while
hierarchical_phase_replay captures about 70 percent of the oracle two-phase value
against 35 to 41 percent for everything else. The crs_plus extensions in
particular were screened on low-tail alone and never on phase skill, and one of
them adds explicit phase-divergence features of the same kind HPR carries.

Protocol matches the operating plan: fit on the 300M panel, evaluate on the
3e18 aggregate-matched pairs. That is a genuine transfer test, not a resubstitution
one, so it also reports whether phase skill survives the scale jump at all.
Within-scale out-of-fold skill at 300M is reported alongside to separate "cannot
learn the phase response" from "learns it but does not transfer".

Delta is observed directly on aggregate-matched pairs, so this carries no
aggregate-model error: the surrogate is judged only on the phase decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_proposal_metric_suite_20260726 import paired_subset, tied_twin
from dsp_exact_baseline_20260726 import MODEL_NAME as DSP_NAME
from dsp_exact_baseline_20260726 import _as_fit, _fit_once, dsp_exact_model
from phase_order_spine_20260725 import load_paired_panel
from proposal_metrics_20260726 import phase_decision_skill
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Panel,
    fit_model,
    grouped_splits,
    load_scale,
    provenance,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "crossscale_phase_skill_20260726"
FIT_SCALE = "300m"
JUDGE_SCALE = "delphi_3e18"
TARGETS = (UNCHEATABLE, TABLE9)
TIED_PREFIX = {"300m": r"^singleavg_", "delphi_3e18": r"^singleavg_fit_\d+_"}
CV_SPLITS = 5


def predicted_delta(fit, model, panel: Panel) -> np.ndarray:
    """Predicted L(a,d) - L(a,0) using the exact aggregate-matched tied twin."""
    return fit.predict(panel, model) - fit.predict(tied_twin(panel), model)


def fit_any(panel: Panel, name: str, catalogue: dict, target: str):
    if name == DSP_NAME:
        fitted, shape = _fit_once(panel, target)
        return _as_fit(fitted, shape, panel, oof_rmse=float("nan")), dsp_exact_model()
    return fit_model(panel, catalogue[name], target), catalogue[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-within-scale", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    panels = {scale: load_scale(scale)[0] for scale in (FIT_SCALE, JUDGE_SCALE)}
    paired = {scale: load_paired_panel(scale, TIED_PREFIX[scale]) for scale in (FIT_SCALE, JUDGE_SCALE)}
    catalogue = {
        m.name: m for m in observatory_baselines(panels[FIT_SCALE]) + nested_candidates() + crs_plus_extensions()
    }
    names = [*catalogue, DSP_NAME] if DSP_NAME not in catalogue else list(catalogue)

    rows, deltas = [], []
    for target in TARGETS:
        fit_panel = panels[FIT_SCALE].subset(np.isfinite(panels[FIT_SCALE].targets[target]))
        judge_panel, judge_order = paired_subset(
            panels[JUDGE_SCALE].subset(np.isfinite(panels[JUDGE_SCALE].targets[target])), paired[JUDGE_SCALE]
        )
        judge_delta = paired[JUDGE_SCALE].delta[target][np.argsort(judge_order)]
        pair_300m, order_300m = paired_subset(fit_panel, paired[FIT_SCALE])
        delta_300m = paired[FIT_SCALE].delta[target][np.argsort(order_300m)]

        for name in names:
            fit, model = fit_any(fit_panel, name, catalogue, target)
            transfer = predicted_delta(fit, model, judge_panel)
            keep = np.isfinite(transfer) & np.isfinite(judge_delta)
            record = {
                "model": name,
                "target": target,
                "arm": "fit_300m_judged_3e18",
                **phase_decision_skill(transfer[keep], judge_delta[keep]),
            }
            rows.append(record)
            deltas.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "target": target,
                        "predicted_delta": transfer[keep],
                        "observed_delta": judge_delta[keep],
                    }
                )
            )
            print(f"transfer {target} {name}: skill={record['phase_skill_score']:.3f}", flush=True)

            if args.skip_within_scale:
                continue
            oof = np.full(len(pair_300m), np.nan)
            for train, test in grouped_splits(pair_300m, CV_SPLITS, 0):
                fold_fit, fold_model = fit_any(pair_300m.subset(train), name, catalogue, target)
                held = pair_300m.subset(test)
                oof[test] = predicted_delta(fold_fit, fold_model, held)
            keep = np.isfinite(oof) & np.isfinite(delta_300m)
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "arm": "oof_within_300m",
                    **phase_decision_skill(oof[keep], delta_300m[keep]),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "crossscale_phase_skill.csv", index=False)
    # Per-pair deltas are kept so the skill score can be bootstrapped without refitting.
    pd.concat(deltas, ignore_index=True).to_csv(output / "transfer_deltas.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "fit_scale": FIT_SCALE,
                "judge_scale": JUDGE_SCALE,
                "estimand": "Delta = L(a,d) - L(a,0) on exact aggregate-matched pairs; carries no aggregate-model error",
                "skill": "value-weighted against always-tied; 1 = oracle, 0 = never use two phases",
                "arms": ["fit_300m_judged_3e18 (transfer)", "oof_within_300m"],
                "cv_splits": CV_SPLITS,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    for arm in frame["arm"].unique():
        block = frame[frame["arm"] == arm]
        print(f"\n=== {arm}: phase skill (1 = oracle) ===")
        print(
            block.pivot_table(index="model", columns="target", values="phase_skill_score")
            .sort_values(UNCHEATABLE, ascending=False)
            .round(4)
            .to_string()
        )
        print(f"\n=== {arm}: decision accuracy vs always-tied baseline ===")
        print(block.pivot_table(index="model", columns="target", values="decision_accuracy").round(4).to_string())


if __name__ == "__main__":
    main()
