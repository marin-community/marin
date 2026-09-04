# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-bucket repetition knees from the Delphi dose-response curves, against candidate onset covariates.

For each bucket the conditional dose curve (multipliers 0.25 to 32 of its proportional share, other buckets
proportional) is summarized by the optimal multiplier of a quadratic in log multiplier (as in Scaling Domain Data
Repetition) and by the first multiplier whose change from the anchor exceeds a noise margin. Both are compared
with log inventory, the declared quality rank and the bucket's deletion value. Development evidence: the dose
runs are in the bank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_dose_anatomy_20260903 as anatomy,
)

PANEL = "delphi_3e18_39bucket"
ANCHOR_EPOCHS = 0.9053525469339763
NOISE_MARGIN = 3.0  # multiples of the target's repeat SD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = harness.load_panel(PANEL)
    table, _components = anatomy.dose_table(args.recovery)
    covariates = pd.DataFrame(
        {
            "bucket": panel.buckets,
            "log_inventory": models.onset_covariate(panel.features, "log_inventory"),
            "quality": models.onset_covariate(panel.features, "quality"),
        }
    )
    rows = []
    for target, column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
        anchor = float(table.loc[table["run_name"].eq(anatomy.ANCHOR), column].iloc[0])
        margin = NOISE_MARGIN * panel.repeat_sd.get(target, float("nan"))
        for bucket, group in table.groupby("focal_domain"):
            curve = group[group[column].notna() & group["epoch_multiplier"].gt(0)].sort_values("epoch_multiplier")
            if len(curve) < 4:
                continue
            multipliers = curve["epoch_multiplier"].to_numpy(float)
            delta = curve[column].to_numpy(float) - anchor
            log_m = np.log2(multipliers)
            coefficients = np.polyfit(log_m, delta, 2)
            argmin = float(2 ** (-coefficients[1] / (2 * coefficients[0]))) if coefficients[0] > 0 else float("nan")
            crossing = multipliers[delta > margin]
            deletion = group.loc[group["epoch_multiplier"].eq(0.0), column]
            rows.append(
                {
                    "target": target,
                    "bucket": bucket,
                    "points": len(curve),
                    "max_multiplier": float(multipliers.max()),
                    "quadratic_optimum_multiplier": argmin,
                    "quadratic_optimum_epochs": argmin * ANCHOR_EPOCHS if np.isfinite(argmin) else float("nan"),
                    "first_harmful_multiplier": float(crossing.min()) if len(crossing) else float("inf"),
                    "delta_at_max": float(delta[-1]),
                    "deletion_delta": float(deletion.iloc[0] - anchor) if len(deletion) else float("nan"),
                }
            )
    knees = pd.DataFrame(rows).merge(covariates, on="bucket")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    knees.to_csv(args.output_dir / "dose_knees.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    for target, subset in knees.groupby("target"):
        finite = subset[np.isfinite(subset["quadratic_optimum_epochs"]) & subset["max_multiplier"].ge(8)]
        print(f"\n=== {target}: {len(subset)} buckets, {len(finite)} with a finite quadratic optimum and max multiplier >= 8")
        for covariate in ("log_inventory", "quality", "deletion_delta"):
            valid = finite[np.isfinite(finite[covariate])]
            if len(valid) >= 5:
                rho, p_value = stats.spearmanr(valid[covariate], np.log(valid["quadratic_optimum_epochs"]))
                rho_first, _ = stats.spearmanr(valid[covariate], np.log(valid["first_harmful_multiplier"].replace(np.inf, 64.0)))
                print(f"  Spearman(log optimum epochs, {covariate}) = {rho:+.3f} (p = {p_value:.3f}, n = {len(valid)}); with first harmful multiplier {rho_first:+.3f}")
        print(
            subset.sort_values("quadratic_optimum_epochs")[
                ["bucket", "points", "max_multiplier", "quadratic_optimum_epochs", "first_harmful_multiplier", "delta_at_max", "deletion_delta", "log_inventory", "quality"]
            ]
            .round(3)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
