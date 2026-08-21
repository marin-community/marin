# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Audit multiplicity and power for the inactive confirmation design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import nct, t

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round74_confirmation_multiplicity"
NOISE = OUTPUT_ROOT / "round67_confirmation_power/repeat_noise_estimates.csv"
FAMILYWISE_ALPHA = 0.05
SUPERIORITY_TARGET_COUNT = 2
WORST_HOLM_ALPHA = FAMILYWISE_ALPHA / SUPERIORITY_TARGET_COUNT
EFFECT_BPB = 0.005
TARGET_POWER = 0.80
REPEAT_COUNTS = tuple(range(2, 21))
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def two_sample_power(effect: float, standard_deviation: float, repeats_per_arm: int, alpha: float) -> float:
    degrees_of_freedom = 2 * repeats_per_arm - 2
    critical = t.ppf(1.0 - alpha, degrees_of_freedom)
    noncentrality = effect / (standard_deviation * np.sqrt(2.0 / repeats_per_arm))
    return float(1.0 - nct.cdf(critical, degrees_of_freedom, noncentrality))


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    noise = pd.read_csv(NOISE)
    rows: list[dict[str, object]] = []
    for noise_row in noise.itertuples(index=False):
        for noise_bound, standard_deviation in (
            ("pooled_point", float(noise_row.pooled_within_policy_sd)),
            ("upper_95pct", float(noise_row.pooled_sd_95pct_high)),
        ):
            for repeats_per_arm in REPEAT_COUNTS:
                rows.append(
                    {
                        "target": noise_row.target,
                        "noise_bound": noise_bound,
                        "standard_deviation": standard_deviation,
                        "repeats_per_arm": repeats_per_arm,
                        "effect_bpb": EFFECT_BPB,
                        "one_sided_alpha": WORST_HOLM_ALPHA,
                        "power": two_sample_power(
                            EFFECT_BPB,
                            standard_deviation,
                            repeats_per_arm,
                            WORST_HOLM_ALPHA,
                        ),
                    }
                )
    power = pd.DataFrame(rows)
    required_rows = []
    for (target, noise_bound), group in power.groupby(["target", "noise_bound"], sort=True):
        passing = group.loc[group["power"].ge(TARGET_POWER)].sort_values("repeats_per_arm")
        required_rows.append(
            {
                "target": target,
                "noise_bound": noise_bound,
                "standard_deviation": float(group["standard_deviation"].iloc[0]),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "worst_holm_alpha": WORST_HOLM_ALPHA,
                "effect_bpb": EFFECT_BPB,
                "minimum_repeats_per_arm_for_80pct_power": int(passing.iloc[0]["repeats_per_arm"]),
            }
        )
    required = pd.DataFrame(required_rows)
    power.to_csv(ROUND_DIR / "multiplicity_adjusted_power.csv", index=False)
    required.to_csv(ROUND_DIR / "multiplicity_adjusted_required_repeats.csv", index=False)

    figure = px.line(
        power,
        x="repeats_per_arm",
        y="power",
        color="noise_bound",
        facet_col="target",
        markers=True,
        title="Confirmation power after two-target family-wise correction",
        labels={
            "repeats_per_arm": "Independent repeats per policy arm",
            "power": "One-sided power at alpha=0.025",
            "noise_bound": "Noise assumption",
        },
    )
    figure.add_hline(y=TARGET_POWER, line_dash="dash", line_color="#4d5963")
    figure.update_layout(template="plotly_white", height=520, width=1120)
    figure.write_html(ROUND_DIR / "multiplicity_adjusted_power.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = "\n".join(
        [
            "# Round 74: untouched-confirmation multiplicity audit",
            "",
            "This audit changes no surrogate decision and reads no sealed outcome. It uses only the frozen 0.005-BPB effect threshold and the development repeat-noise estimates.",
            "",
            "The confirmation claim that a phase-varying optimum beats its tied control on at least one of two targets is a union claim. Control its family-wise type-I error with Holm's procedure across the two one-sided superiority tests. A design that must retain 80% power when only one target improves should therefore power the first Holm threshold at alpha=0.05/2=0.025. The simultaneous noninferiority claim against the frontier on both targets is an intersection-union test; requiring both one-sided 95% bounds to pass already controls its size and needs no Bonferroni penalty.",
            "",
            "## Required repeats for a 0.005-BPB superiority effect",
            "",
            required.to_markdown(index=False, floatfmt=".6f"),
            "",
            "At the pooled Table-9 noise estimate, six repeats per arm provide 0.735 power after the correction and seven provide 0.812. At the upper 95% nuisance bound, 15 repeats are required. Because the future panel is inactive and should be decisive if ever activated, the conservative fixed design uses 15 repeats for each decisive arm rather than allowing outcome-dependent sample-size changes.",
            "",
            "The many single-seed contrast-ray policies are descriptive checks of the frozen surface, not candidates from which a winner may be selected. The only confirmatory superiority tests are the two frozen raw-optimum versus tied-control contrasts. Regularized optima and rays cannot replace a failed raw optimum after outcomes are unsealed.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
