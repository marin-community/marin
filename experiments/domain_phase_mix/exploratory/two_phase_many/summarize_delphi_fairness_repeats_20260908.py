# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Seed-matched comparison of the frozen procedure with Olmix's best 3e18 policies.

Reads the collector's tables of the fairness repeats (trainer seeds 0 to 2 for the Olmix policies, 1 and 2 for
ours), of the frozen-procedure validation (our trainer-seed-0 controls) and of the Olmix KL 0.05 ladder-policy
repeats (Table 2's Uncheatable row, trainer seeds 0 to 2), and writes `fairness_summary.csv`
(per policy: mean and SD over seeds; per pair: the paired difference with its standard error) and
`fairness_summary.md` into the repeats package.

usage: uv run summarize_delphi_fairness_repeats_20260908.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = Path(__file__).resolve().parent / "reference_outputs"
REPEATS = REFERENCE / "delphi_fairness_repeats_3e18_20260908"
CONTROLS = REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908"
LADDER_REPEATS = REFERENCE / "delphi_olmix_kl005_repeats_3e18_20260908"  # Table 2's Uncheatable ladder policy
POLICIES = (  # candidate, objective column, label
    ("olmix_u_kl0p1_cap04", "uncheatable_bpb", "Olmix, KL 0.1, cap 4"),
    ("olmix_u_kl0p05_cap04", "uncheatable_bpb", "Olmix, KL 0.05, cap 4 (ladder policy)"),
    ("lwspu_u_snc_cap06", "uncheatable_bpb", "ours, unconstrained (5.3 epochs)"),
    ("olmix_t9_kl0p005_cap05", "table9_macro_bpb", "Olmix, KL 0.005, cap 4"),
    ("lwspu_t9_snc_cap08", "table9_macro_bpb", "ours, unconstrained (7.5 epochs)"),
    ("lwspu_t9_snc_cap06", "table9_macro_bpb", "ours, cap 6"),
)
PAIRS = (
    ("lwspu_u_snc_cap06", "olmix_u_kl0p1_cap04", "uncheatable_bpb"),
    ("lwspu_u_snc_cap06", "olmix_u_kl0p05_cap04", "uncheatable_bpb"),
    ("lwspu_t9_snc_cap08", "olmix_t9_kl0p005_cap05", "table9_macro_bpb"),
    ("lwspu_t9_snc_cap06", "olmix_t9_kl0p005_cap05", "table9_macro_bpb"),
    ("lwspu_t9_snc_cap06", "lwspu_t9_snc_cap08", "table9_macro_bpb"),
)


def measured_by_seed() -> pd.DataFrame:
    repeats = pd.read_csv(REPEATS / "measured_results.csv")
    controls = pd.read_csv(CONTROLS / "measured_results.csv").assign(trainer_seed=0)
    ladder_repeats = pd.read_csv(LADDER_REPEATS / "measured_results.csv")
    frame = pd.concat([repeats, controls, ladder_repeats], ignore_index=True)
    frame = frame[frame.status == "measured"]
    return frame[["candidate_id", "trainer_seed", "uncheatable_bpb", "table9_macro_bpb"]].sort_values(
        ["candidate_id", "trainer_seed"]
    )


def main() -> None:
    frame = measured_by_seed()
    rows = []
    for candidate, column, label in POLICIES:
        values = frame[frame.candidate_id == candidate].set_index("trainer_seed")[column]
        rows.append(
            {
                "kind": "policy",
                "candidate_id": candidate,
                "label": label,
                "metric": column,
                "seeds": ";".join(str(int(seed)) for seed in values.index),
                "values": ";".join(f"{value:.4f}" for value in values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "n": len(values),
            }
        )
    for ours, other, column in PAIRS:
        a = frame[frame.candidate_id == ours].set_index("trainer_seed")[column]
        b = frame[frame.candidate_id == other].set_index("trainer_seed")[column]
        diff = (a - b).dropna()
        rows.append(
            {
                "kind": "paired_difference",
                "candidate_id": f"{ours} - {other}",
                "label": f"{ours} minus {other}",
                "metric": column,
                "seeds": ";".join(str(int(seed)) for seed in diff.index),
                "values": ";".join(f"{value:+.4f}" for value in diff),
                "mean": float(diff.mean()),
                "sd": float(diff.std(ddof=1)),
                "n": len(diff),
                "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(REPEATS / "fairness_summary.csv", index=False)
    lines = [
        "# Seed-matched comparison at 3e18 (trainer seeds 0, 1, 2; data seed 666200 Uncheatable, 662009 Table 9)",
        "",
    ]
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    (REPEATS / "fairness_summary.md").write_text("\n".join(lines) + "\n")
    print(summary[["kind", "label", "metric", "values", "mean", "sd", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
