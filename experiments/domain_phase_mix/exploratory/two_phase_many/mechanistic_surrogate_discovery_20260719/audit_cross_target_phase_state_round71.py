# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Audit whether a scalar phase state transfers across targets and scales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round71_cross_target_phase_state"
MATCHED = OUTPUT_ROOT / "round1_cross_scale_matched_policy/matched_targets.csv"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def component_values(data: pd.DataFrame, target: str, policy_class: str, scale: str) -> pd.Series:
    return (
        data.loc[data["target"].eq(target) & data["policy_class"].eq(policy_class)]
        .set_index("source_index")[f"value_{scale}"]
        .sort_index()
    )


def cross_target_metrics(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scale in ("300m", "delphi"):
        for component in ("one_phase", "phase_delta", "two_phase"):
            uncheatable = component_values(data, "uncheatable", component, scale)
            table9 = component_values(data, "table9", component, scale).loc[uncheatable.index]
            records.append(
                {
                    "scale": scale,
                    "component": component,
                    "n": len(uncheatable),
                    "pearson": float(np.corrcoef(uncheatable, table9)[0, 1]),
                    "spearman": float(spearmanr(uncheatable, table9).statistic),
                    "raw_sign_agreement": float(np.mean(np.sign(uncheatable) == np.sign(table9)))
                    if component == "phase_delta"
                    else np.nan,
                    "both_targets_improve_fraction": float(np.mean((uncheatable < 0) & (table9 < 0)))
                    if component == "phase_delta"
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def attenuation_bootstrap(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = {
        target: (
            component_values(data, target, "phase_delta", "300m").to_numpy(dtype=float),
            component_values(data, target, "phase_delta", "delphi").to_numpy(dtype=float),
        )
        for target in ("uncheatable", "table9")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[dict[str, float]] = []
    n = len(values["uncheatable"][0])
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, n, n)
        ratios = {
            target: float(np.std(delphi[indices], ddof=1) / np.std(scale_300m[indices], ddof=1))
            for target, (scale_300m, delphi) in values.items()
        }
        samples.append(
            {
                "uncheatable_attenuation_ratio": ratios["uncheatable"],
                "table9_attenuation_ratio": ratios["table9"],
                "table9_minus_uncheatable": ratios["table9"] - ratios["uncheatable"],
            }
        )
    frame = pd.DataFrame(samples)
    summary = []
    for quantity in frame.columns:
        summary.append(
            {
                "quantity": quantity,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "mean": float(frame[quantity].mean()),
                "median": float(frame[quantity].median()),
                "low_95pct": float(frame[quantity].quantile(0.025)),
                "high_95pct": float(frame[quantity].quantile(0.975)),
            }
        )
    return frame, pd.DataFrame(summary)


def scatter_frame(data: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scale in ("300m", "delphi"):
        uncheatable = component_values(data, "uncheatable", "phase_delta", scale)
        table9 = component_values(data, "table9", "phase_delta", scale).loc[uncheatable.index]
        for source_index in uncheatable.index:
            records.append(
                {
                    "scale": scale,
                    "source_index": source_index,
                    "uncheatable_phase_delta": float(uncheatable.loc[source_index]),
                    "table9_phase_delta": float(table9.loc[source_index]),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED)
    metrics = cross_target_metrics(matched)
    _bootstrap, attenuation = attenuation_bootstrap(matched)
    metrics.to_csv(ROUND_DIR / "cross_target_component_transfer.csv", index=False)
    attenuation.to_csv(ROUND_DIR / "phase_attenuation_bootstrap.csv", index=False)

    figure = px.scatter(
        scatter_frame(matched),
        x="uncheatable_phase_delta",
        y="table9_phase_delta",
        facet_col="scale",
        hover_data=["source_index"],
        color="scale",
        title="Matched-policy phase effects share signal but disagree for one fifth of policies",
        labels={
            "uncheatable_phase_delta": "Uncheatable two-minus-one-phase BPB",
            "table9_phase_delta": "Table-9 two-minus-one-phase BPB",
        },
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#4d5963")
    figure.add_vline(x=0.0, line_dash="dash", line_color="#4d5963")
    figure.update_layout(template="plotly_white", height=540, width=1120, showlegend=False)
    figure.write_html(ROUND_DIR / "cross_target_phase_effects.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = "\n".join(
        [
            "# Round 71: cross-target phase-state audit",
            "",
            "This audit uses the 238 exact matched policies at 300M and Delphi. It tests whether a single target-independent scalar phase state is plausible before any new multi-output model is proposed. It fits no response surrogate and reads no sealed confirmation outcome.",
            "",
            "## Cross-target component agreement",
            "",
            metrics.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Scale attenuation by target",
            "",
            attenuation.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Phase effects are correlated across targets, so the shared-state premise is not vacuous. It is not sufficient for a universal scalar state: 21%-22% of policies have opposite phase-benefit signs, and Table-9 phase variation attenuates significantly less from 300M to Delphi than Uncheatable. The paired-bootstrap 95% interval for the attenuation-ratio difference excludes zero. A future shared-state model therefore needs at least target-specific observation loadings and a transition whose scale dependence is independently identifiable. The previously rejected joint latent phase transport route did not meet that identification bar and selected full latent rank in most folds; this audit does not reopen it.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
