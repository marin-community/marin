# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Audit policy-group influence on the future-confirmation noise estimate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from audit_confirmation_power_round67 import independent_repeat_rows, pooled_noise, two_sample_power

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round69_repeat_noise_influence"
EFFECT_BPB = 0.005
POWER_TARGET = 0.80
MAX_REPEATS_PER_ARM = 30
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def required_repeats(standard_deviation: float) -> int:
    for repeats_per_arm in range(2, MAX_REPEATS_PER_ARM + 1):
        if two_sample_power(EFFECT_BPB, standard_deviation, repeats_per_arm) >= POWER_TARGET:
            return repeats_per_arm
    raise ValueError(f"No repeat count up to {MAX_REPEATS_PER_ARM} reaches target power")


def leave_one_group_out(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    full_noise = pooled_noise(rows).set_index("target")
    for omitted_hash in sorted(rows["mixture_sha256"].astype(str).unique()):
        retained = rows.loc[rows["mixture_sha256"].astype(str).ne(omitted_hash)]
        omitted_names = "|".join(
            sorted(rows.loc[rows["mixture_sha256"].astype(str).eq(omitted_hash), "wandb_run_name"].astype(str))
        )
        for estimate in pooled_noise(retained).itertuples(index=False):
            standard_deviation = float(estimate.pooled_within_policy_sd)
            full_standard_deviation = float(full_noise.loc[estimate.target, "pooled_within_policy_sd"])
            records.append(
                {
                    "target": estimate.target,
                    "omitted_mixture_sha256": omitted_hash,
                    "omitted_run_names": omitted_names,
                    "remaining_groups": int(estimate.independent_repeat_groups),
                    "remaining_degrees_of_freedom": int(estimate.degrees_of_freedom),
                    "pooled_within_policy_sd": standard_deviation,
                    "relative_sd_change": standard_deviation / full_standard_deviation - 1.0,
                    "power_at_six_repeats_per_arm": two_sample_power(EFFECT_BPB, standard_deviation, 6),
                    "minimum_repeats_per_arm_for_80pct_power": required_repeats(standard_deviation),
                }
            )
    return pd.DataFrame(records)


def summary_table(influence: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target, group in influence.groupby("target", sort=True):
        records.append(
            {
                "target": target,
                "leave_one_group_out_fits": len(group),
                "minimum_pooled_sd": float(group["pooled_within_policy_sd"].min()),
                "median_pooled_sd": float(group["pooled_within_policy_sd"].median()),
                "maximum_pooled_sd": float(group["pooled_within_policy_sd"].max()),
                "maximum_absolute_relative_sd_change": float(group["relative_sd_change"].abs().max()),
                "minimum_power_at_six_repeats": float(group["power_at_six_repeats_per_arm"].min()),
                "maximum_required_repeats_per_arm": int(group["minimum_repeats_per_arm_for_80pct_power"].max()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    influence = leave_one_group_out(independent_repeat_rows())
    summary = summary_table(influence)
    influence.to_csv(ROUND_DIR / "repeat_noise_leave_one_group_out.csv", index=False)
    summary.to_csv(ROUND_DIR / "repeat_noise_influence_summary.csv", index=False)

    figure = px.scatter(
        influence,
        x="omitted_mixture_sha256",
        y="pooled_within_policy_sd",
        color="power_at_six_repeats_per_arm",
        facet_col="target",
        hover_data=["omitted_run_names", "minimum_repeats_per_arm_for_80pct_power"],
        color_continuous_scale="RdYlGn_r",
        title="Future-confirmation noise: leave-one-policy-group-out influence",
        labels={
            "omitted_mixture_sha256": "Omitted exact-policy group",
            "pooled_within_policy_sd": "Remaining pooled within-policy SD",
            "power_at_six_repeats_per_arm": "Power at six repeats",
        },
    )
    figure.update_xaxes(showticklabels=False)
    figure.update_layout(template="plotly_white", height=520, width=1120)
    figure.write_html(ROUND_DIR / "repeat_noise_influence.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = "\n".join(
        [
            "# Round 69: repeat-noise influence audit",
            "",
            "This audit removes one of the ten independent exact-policy repeat groups at a time and recomputes the pooled nuisance variance and the already frozen 0.005-BPB power calculation. It fits no surrogate, reads no sealed confirmation outcome, and does not alter a model or response threshold.",
            "",
            summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "The purpose is to detect whether a single repeat policy determines the proposed six-seed confirmation budget. A robust allocation must retain at least 80% point-estimate power under every leave-one-group-out estimate; otherwise the future design must use the worst-case repeat count or a blinded variance rule rather than the pooled point estimate.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
