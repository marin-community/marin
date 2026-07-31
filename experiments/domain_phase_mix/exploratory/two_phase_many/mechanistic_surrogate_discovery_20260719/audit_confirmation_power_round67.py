# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Audit repeat noise and power for the inactive future confirmation design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import brentq
from scipy.stats import chi2, nct, t

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round67_confirmation_power"
PROVENANCE_DIR = OUTPUT_ROOT / "round58_heldout_provenance"
ALPHA = 0.05
TARGET_POWER = 0.80
EFFECTS = (0.002, 0.005, 0.010)
REPEATS_PER_ARM = tuple(range(2, 13))
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def independent_repeat_rows() -> pd.DataFrame:
    provenance = pd.read_csv(PROVENANCE_DIR / "heldout_provenance_index.csv")
    groups = pd.read_csv(PROVENANCE_DIR / "coordinate_repeat_groups.csv")
    independent_hashes = set(
        groups.loc[groups["repeat_kind"].eq("independent_seed_repeat"), "mixture_sha256"].astype(str)
    )
    rows = provenance.loc[provenance["mixture_sha256"].astype(str).isin(independent_hashes)].copy()
    if rows["mixture_sha256"].nunique() != 10 or len(rows) != 26:
        raise ValueError("Expected 26 rows in 10 independent-seed repeat groups")
    return rows


def pooled_noise(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for target, column in (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb")):
        sum_squares = 0.0
        degrees_of_freedom = 0
        group_standard_deviations = []
        for _, group in rows.groupby("mixture_sha256", sort=True):
            values = group[column].to_numpy(dtype=float)
            sum_squares += float(np.sum((values - values.mean()) ** 2))
            degrees_of_freedom += len(values) - 1
            group_standard_deviations.append(float(np.std(values, ddof=1)))
        variance = sum_squares / degrees_of_freedom
        standard_deviation = float(np.sqrt(variance))
        variance_low = degrees_of_freedom * variance / chi2.ppf(0.975, degrees_of_freedom)
        variance_high = degrees_of_freedom * variance / chi2.ppf(0.025, degrees_of_freedom)
        output.append(
            {
                "target": target,
                "independent_repeat_groups": rows["mixture_sha256"].nunique(),
                "repeat_rows": len(rows),
                "degrees_of_freedom": degrees_of_freedom,
                "pooled_within_policy_sd": standard_deviation,
                "pooled_sd_95pct_low": float(np.sqrt(variance_low)),
                "pooled_sd_95pct_high": float(np.sqrt(variance_high)),
                "median_group_sd": float(np.median(group_standard_deviations)),
                "maximum_group_sd": float(np.max(group_standard_deviations)),
            }
        )
    return pd.DataFrame(output)


def two_sample_power(effect: float, standard_deviation: float, repeats_per_arm: int) -> float:
    degrees_of_freedom = 2 * repeats_per_arm - 2
    critical = t.ppf(1.0 - ALPHA, degrees_of_freedom)
    noncentrality = effect / (standard_deviation * np.sqrt(2.0 / repeats_per_arm))
    return float(1.0 - nct.cdf(critical, degrees_of_freedom, noncentrality))


def minimum_detectable_effect(standard_deviation: float, repeats_per_arm: int) -> float:
    return float(
        brentq(
            lambda effect: two_sample_power(effect, standard_deviation, repeats_per_arm) - TARGET_POWER,
            0.0,
            0.1,
        )
    )


def power_table(noise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for noise_row in noise.itertuples(index=False):
        for repeats_per_arm in REPEATS_PER_ARM:
            detectable = minimum_detectable_effect(float(noise_row.pooled_within_policy_sd), repeats_per_arm)
            for effect in EFFECTS:
                rows.append(
                    {
                        "target": noise_row.target,
                        "repeats_per_arm": repeats_per_arm,
                        "effect_bpb": effect,
                        "one_sided_alpha": ALPHA,
                        "power": two_sample_power(effect, float(noise_row.pooled_within_policy_sd), repeats_per_arm),
                        "minimum_detectable_effect_at_80pct_power": detectable,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    noise = pooled_noise(independent_repeat_rows())
    power = power_table(noise)
    noise.to_csv(ROUND_DIR / "repeat_noise_estimates.csv", index=False)
    power.to_csv(ROUND_DIR / "confirmation_power.csv", index=False)

    figure = px.line(
        power,
        x="repeats_per_arm",
        y="power",
        color="effect_bpb",
        markers=True,
        facet_col="target",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Future paired-policy confirmation: power under independent-arm repeat noise",
        labels={
            "repeats_per_arm": "Independent training repeats per policy arm",
            "power": "One-sided power",
            "effect_bpb": "True BPB improvement",
        },
    )
    figure.add_hline(y=TARGET_POWER, line_dash="dash", line_color="#4d5963")
    figure.update_layout(template="plotly_white", height=520, width=1120)
    figure.write_html(ROUND_DIR / "confirmation_power.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    effect_rows = power.loc[power["effect_bpb"].eq(0.005) & power["repeats_per_arm"].isin([3, 6])]
    required_rows = []
    for target, group in power.loc[power["effect_bpb"].eq(0.005)].groupby("target"):
        passing = group.loc[group["power"].ge(TARGET_POWER)].sort_values("repeats_per_arm")
        required_rows.append(
            {
                "target": target,
                "effect_bpb": 0.005,
                "minimum_repeats_per_arm_for_80pct_power": int(passing.iloc[0]["repeats_per_arm"]),
            }
        )
    required = pd.DataFrame(required_rows)
    required.to_csv(ROUND_DIR / "required_repeats.csv", index=False)

    report = "\n".join(
        [
            "# Round 67: future confirmation repeat-power audit",
            "",
            "This audit reads only development-archive policies with genuinely independent training seeds. It fits no response surrogate, reads no sealed confirmation outcome, and changes no candidate decision. The test model is a conservative independent-arm comparison; matched seed covariance is not assumed.",
            "",
            "## Empirical repeat noise",
            "",
            noise.to_markdown(index=False, floatfmt=".6f"),
            "",
            "The pooled estimates use 26 runs across 10 exact-policy groups and 16 residual degrees of freedom. Their confidence intervals are wide enough that the power calculation should be treated as design guidance rather than a universal noise constant.",
            "",
            "## Frozen 0.005-BPB threshold",
            "",
            effect_rows.to_markdown(index=False, floatfmt=".5f"),
            "",
            required.to_markdown(index=False),
            "",
            "Three repeats per arm are adequate for a 0.005-BPB Uncheatable effect but provide only about 51% power for Table-9. Six repeats per arm provide about 85% Table-9 power at the pooled point estimate. The inactive future confirmation design is therefore revised from three to six independent seeds for each raw-optimum/tied-control pair and incumbent frontier. This revision is based only on pre-existing repeat variance and the already frozen 0.005-BPB acceptance threshold.",
            "",
            "The upper confidence bound on Table-9 noise remains materially larger than the point estimate. Before an expensive future confirmation launch, the exact candidate-specific repeat allocation should be frozen with either six runs as the minimum or a blinded nuisance-variance re-estimation rule. Outcomes must not be inspected by treatment arm during that reassessment.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
