# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Materialize tied controls for the validated separate-heads frontiers.

The separate-heads plus geometry fit is exactly the plain separate-heads fit:
all nonnegative geometry coefficients are zero on the full original and matched
panels. There is therefore no distinct augmented optimum to validate.

This panel instead tests the phase ordering of the existing KL=0.1 frontiers.
For each objective, the tied control repeats the two-phase candidate's 80/20
aggregate mixture in both phases. Aggregate exposure is exactly preserved.
Three repeat pairs receive shared data seeds in the launcher.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "separate_heads_frontier_tied_panel_20260710"
SOURCE_DIR = REFERENCE_OUTPUTS / "sep_lf_kl_sweep_panel_20260706"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
REPEATS = 3
DATA_SEEDS = {
    "uncheatable": (680000, 680001, 680002),
    "table9": (680100, 680101, 680102),
}
SOURCE_CANDIDATES = {
    "uncheatable": "seplf_unch_sep_kl0p1",
    "table9": "seplf_t9_sep_kl0p1",
}
KEY_PREFIXES = {
    "uncheatable": "sepfront_unch",
    "table9": "sepfront_t9",
}


def tied_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    aggregate = PHASE_0_FRACTION * frame["phase_0_weight"].to_numpy(dtype=float) + PHASE_1_FRACTION * frame[
        "phase_1_weight"
    ].to_numpy(dtype=float)
    proportional = frame["proportional"].to_numpy(dtype=float)
    frame["phase_0_weight"] = aggregate
    frame["phase_1_weight"] = aggregate
    frame["aggregate_weight"] = aggregate
    frame["phase_0_epoch_multiplier"] = aggregate / proportional
    frame["phase_1_epoch_multiplier"] = aggregate / proportional
    frame["phase_0_delta"] = aggregate - proportional
    frame["phase_1_delta"] = aggregate - proportional
    frame["max_abs_delta"] = np.abs(aggregate - proportional)
    return frame


def validate_pair(two_phase: pd.DataFrame, tied: pd.DataFrame, objective: str) -> None:
    if two_phase["domain"].tolist() != tied["domain"].tolist():
        raise ValueError(f"{objective}: domain order differs")
    for column in ("phase_0_weight", "phase_1_weight"):
        if not np.isclose(two_phase[column].sum(), 1.0, atol=1e-10):
            raise ValueError(f"{objective}: source {column} does not sum to one")
        if not np.isclose(tied[column].sum(), 1.0, atol=1e-10):
            raise ValueError(f"{objective}: tied {column} does not sum to one")
    source_aggregate = PHASE_0_FRACTION * two_phase["phase_0_weight"].to_numpy(
        dtype=float
    ) + PHASE_1_FRACTION * two_phase["phase_1_weight"].to_numpy(dtype=float)
    tied_aggregate = PHASE_0_FRACTION * tied["phase_0_weight"].to_numpy(dtype=float) + PHASE_1_FRACTION * tied[
        "phase_1_weight"
    ].to_numpy(dtype=float)
    if not np.allclose(source_aggregate, tied_aggregate, atol=1e-12, rtol=0.0):
        raise ValueError(f"{objective}: tied control changes aggregate exposure")
    if not np.allclose(tied["phase_0_weight"], tied["phase_1_weight"], atol=1e-12, rtol=0.0):
        raise ValueError(f"{objective}: tied phases differ")
    if not np.allclose(two_phase["simulated_epochs"], tied["simulated_epochs"], atol=1e-10):
        raise ValueError(f"{objective}: tied control changes simulated epochs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    mixture_dir = args.output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for objective, source_candidate in SOURCE_CANDIDATES.items():
        source_path = SOURCE_DIR / source_candidate / "proposed_mixture_weights.csv"
        two_phase = pd.read_csv(source_path)
        tied = tied_frame(two_phase)
        validate_pair(two_phase, tied, objective)

        prefix = KEY_PREFIXES[objective]
        two_phase_path = mixture_dir / f"{prefix}_2p.csv"
        tied_path = mixture_dir / f"{prefix}_tied.csv"
        two_phase.to_csv(two_phase_path, index=False)
        tied.to_csv(tied_path, index=False)

        source_aggregate = two_phase["aggregate_weight"].to_numpy(dtype=float)
        phase_tv = float(
            0.5
            * np.abs(
                two_phase["phase_0_weight"].to_numpy(dtype=float) - two_phase["phase_1_weight"].to_numpy(dtype=float)
            ).sum()
        )
        summary_rows.append(
            {
                "objective": objective,
                "source_candidate": source_candidate,
                "kl_reg": 0.1,
                "phase_tv": phase_tv,
                "aggregate_hhi": float(np.sum(source_aggregate**2)),
                "max_simulated_epoch": float(two_phase["simulated_epochs"].max()),
                "aggregate_preserved_max_abs_error": float(
                    np.max(np.abs(source_aggregate - tied["aggregate_weight"].to_numpy(dtype=float)))
                ),
            }
        )
        for repeat, data_seed in enumerate(DATA_SEEDS[objective]):
            for policy, weights_path in (("2p", two_phase_path), ("tied", tied_path)):
                manifest_rows.append(
                    {
                        "candidate": f"{prefix}_{policy}_s{repeat}",
                        "objective": objective,
                        "policy": policy,
                        "repeat": repeat,
                        "data_seed": data_seed,
                        "trainer_seed": 0,
                        "source_candidate": source_candidate,
                        "kl_reg": 0.1,
                        "weights_csv": str(weights_path.relative_to(args.output_dir)),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    summary.to_csv(args.output_dir / "pair_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(manifest.to_string(index=False))
    print(f"Wrote {len(manifest)} paired validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
