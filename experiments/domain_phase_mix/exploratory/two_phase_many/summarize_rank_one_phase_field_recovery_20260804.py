# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Stratify the frozen rank-one recovery audit without changing its decision."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "rank_one_phase_field_recovery_20260804"
PRIMARY_SIGNAL_RMS = 0.0039


def q90(values: pd.Series) -> float:
    return float(values.quantile(0.9))


def grouped_summary(noisy: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        noisy.groupby(columns, sort=True)
        .agg(
            rows=("test_signal_rmse_ratio", "size"),
            factor_cases=("factor_case", "nunique"),
            matrix_cosine_median=("matrix_cosine", "median"),
            signal_rmse_ratio_median=("test_signal_rmse_ratio", "median"),
            signal_rmse_ratio_q90=("test_signal_rmse_ratio", q90),
            sign_accuracy_median=("test_sign_accuracy", "median"),
        )
        .reset_index()
    )


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(formatted.columns) + " |"
    divider = "|" + "|".join("---" for _ in formatted.columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in formatted.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    protocol = json.loads((OUTPUT_DIR / "protocol.json").read_text())
    decision = json.loads((OUTPUT_DIR / "decision.json").read_text())
    noisy = pd.read_csv(OUTPUT_DIR / "noisy_recovery.csv")

    by_kind = grouped_summary(noisy, ["basis", "target_noise", "signal_rms", "factor_kind"])
    by_case = grouped_summary(
        noisy,
        ["basis", "target_noise", "signal_rms", "factor_kind", "factor_case"],
    )
    by_kind.to_csv(OUTPUT_DIR / "posthoc_factor_kind_summary.csv", index=False)
    by_case.to_csv(OUTPUT_DIR / "posthoc_factor_case_summary.csv", index=False)

    primary = by_kind.loc[by_kind["signal_rms"].eq(PRIMARY_SIGNAL_RMS)].copy()
    table9 = primary.loc[primary["target_noise"].eq("table9")].copy()
    amplitude = grouped_summary(noisy, ["basis", "target_noise", "signal_rms"])
    gates = protocol["gates"]
    family_pooled = amplitude.loc[
        amplitude["basis"].eq("declared_family_masses")
        & amplitude["target_noise"].eq("table9")
        & amplitude["signal_rms"].eq(PRIMARY_SIGNAL_RMS)
    ].iloc[0]
    family_margin = (
        float(family_pooled["signal_rmse_ratio_median"]) - gates["primary_noisy_signal_rmse_ratio_median_max"]
    )

    report = f"""# Post-hoc rank-one recovery stratification

Protocol: `{protocol['protocol_hash']}`

The preregistered decision is unchanged: `{decision['decision']}`. Neither basis
passes both endpoint-noise gates, and no endpoint model is promoted.

## Primary signal by factor kind

{markdown_table(table9)}

The geometry-stress and random cases fail similarly. The negative result is not
caused by a small hidden stress subset. The full 76-degree-of-freedom field is
not recoverable at the Table-9 noise level. The 40-degree-of-freedom declared
family basis is a near miss: its pooled median signal-RMSE ratio exceeds the
frozen `0.5000` limit by `{family_margin:.6f}`. The threshold is not relaxed.

## Signal-amplitude dependence

{markdown_table(amplitude)}

Both bases recover a 0.0100-BPB synthetic signal under both noise levels, but
that does not rescue the failed 0.0039-BPB primary gate. The result says the
current 238-row design can resolve a large rank-one phase field; it cannot
reliably resolve a field at the primary Table-9 effect-to-noise scale.

## Scope

This is a descriptive post-hoc stratification of already-generated rows. It
does not alter factors, folds, gates, or the preregistered decision. The truth
was rank one by construction, so even a pass would not show that the physical
phase field is rank one or identify meaningful factors.
"""
    (OUTPUT_DIR / "posthoc_analysis.md").write_text(report)


if __name__ == "__main__":
    main()
