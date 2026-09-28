# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Bound measurement-error attenuation in matched phase-effect transfer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round70_cross_scale_measurement_error"
TRANSFER = OUTPUT_ROOT / "round61_cross_scale_variance_decomposition/component_scale_transfer.csv"
VARIANCE = OUTPUT_ROOT / "round61_cross_scale_variance_decomposition/variance_decomposition.csv"
DELPHI_NOISE = OUTPUT_ROOT / "round67_confirmation_power/repeat_noise_estimates.csv"
METRICS_300M = (
    TWO_PHASE_ROOT
    / "reference_outputs/two_phase_solver_gap_collaborator_packet_20260701/data/all_300m_checkpoint_metrics.csv"
)
TARGET_COLUMNS = {
    "uncheatable": "eval_uncheatable_eval_bpb",
    "table9": "table9_macro_bpb",
}


def noise_300m() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_300M, low_memory=False)
    repeats = metrics.loc[metrics["run_name"].astype(str).str.match(r"propvar_300m_6b_trainer_seed_\d+$")].copy()
    if len(repeats) != 10:
        raise ValueError(f"Expected 10 independent 300M proportional repeats, found {len(repeats)}")
    records = []
    degrees_of_freedom = len(repeats) - 1
    for target, column in TARGET_COLUMNS.items():
        values = repeats[column].to_numpy(dtype=float)
        variance = float(np.var(values, ddof=1))
        variance_high = degrees_of_freedom * variance / chi2.ppf(0.025, degrees_of_freedom)
        records.append(
            {
                "target": target,
                "scale": "300m",
                "repeat_groups": 1,
                "repeat_rows": len(repeats),
                "degrees_of_freedom": degrees_of_freedom,
                "per_run_sd": float(np.sqrt(variance)),
                "per_run_sd_95pct_high": float(np.sqrt(variance_high)),
            }
        )
    return pd.DataFrame(records)


def noise_delphi() -> pd.DataFrame:
    source = pd.read_csv(DELPHI_NOISE)
    return pd.DataFrame(
        {
            "target": source["target"],
            "scale": "delphi",
            "repeat_groups": source["independent_repeat_groups"],
            "repeat_rows": source["repeat_rows"],
            "degrees_of_freedom": source["degrees_of_freedom"],
            "per_run_sd": source["pooled_within_policy_sd"],
            "per_run_sd_95pct_high": source["pooled_sd_95pct_high"],
        }
    )


def reliability(observed_phase_variance: float, per_run_sd: float) -> float:
    phase_difference_noise_variance = 2.0 * per_run_sd**2
    return float(max(0.0, 1.0 - phase_difference_noise_variance / observed_phase_variance))


def audit_table(noise: pd.DataFrame) -> pd.DataFrame:
    variance = pd.read_csv(VARIANCE)
    transfer = pd.read_csv(TRANSFER)
    phase_transfer = transfer.loc[transfer["component"].eq("phase_delta")].set_index("target")
    records = []
    for target in sorted(phase_transfer.index):
        target_noise = noise.loc[noise["target"].eq(target)].set_index("scale")
        target_variance = variance.loc[variance["target"].eq(target)].set_index("scale")
        observed_pearson = float(phase_transfer.loc[target, "pearson"])
        observed_slope = float(phase_transfer.loc[target, "delphi_on_300m_slope"])
        for noise_bound, column in (("point", "per_run_sd"), ("upper_95pct", "per_run_sd_95pct_high")):
            reliability_300m = reliability(
                float(target_variance.loc["300m", "standard_deviation_phase_delta"]) ** 2,
                float(target_noise.loc["300m", column]),
            )
            reliability_delphi = reliability(
                float(target_variance.loc["delphi", "standard_deviation_phase_delta"]) ** 2,
                float(target_noise.loc["delphi", column]),
            )
            records.append(
                {
                    "target": target,
                    "noise_bound": noise_bound,
                    "observed_phase_transfer_pearson": observed_pearson,
                    "phase_reliability_300m": reliability_300m,
                    "phase_reliability_delphi": reliability_delphi,
                    "deattenuated_phase_transfer_pearson": min(
                        1.0,
                        observed_pearson / np.sqrt(reliability_300m * reliability_delphi),
                    ),
                    "observed_delphi_on_300m_slope": observed_slope,
                    "errors_in_variables_slope": observed_slope / reliability_300m,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    noise = pd.concat([noise_300m(), noise_delphi()], ignore_index=True)
    audit = audit_table(noise)
    noise.to_csv(ROUND_DIR / "cross_scale_noise_inputs.csv", index=False)
    audit.to_csv(ROUND_DIR / "phase_transfer_deattenuation.csv", index=False)

    report = "\n".join(
        [
            "# Round 70: cross-scale phase-transfer measurement-error bound",
            "",
            "This audit asks whether weak matched-policy phase-effect transfer can be explained by training/evaluation noise. It uses ten 300M proportional repeats and the ten exact-policy Delphi repeat groups from Round 67. For a paired phase effect $Y_{2p}-Y_{1p}$, the conservative independent-run noise variance is $2\\sigma^2$. It fits no surrogate and reads no sealed confirmation outcome.",
            "",
            "## Noise inputs",
            "",
            noise.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Errors-in-variables bound",
            "",
            audit.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Even under the upper 95% noise bounds, deattenuation changes the phase-transfer correlations only modestly. It cannot turn weak Table-9 transfer or the attenuated Delphi-on-300M slopes into a scale-invariant relation. The mismatch is therefore structural at the resolution of these data, not primarily a consequence of run noise. The bound remains approximate because proportional-repeat noise is used as a nuisance estimate for all policies.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
