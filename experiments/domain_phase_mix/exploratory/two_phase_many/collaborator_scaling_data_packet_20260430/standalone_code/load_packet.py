#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the core tables in the collaborator scaling packet.

This file intentionally has no Marin imports. Run from the packet root:

    python standalone_code/load_packet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PACKET_ROOT = Path(__file__).resolve().parents[1]


def phase_domains(columns: list[str]) -> list[str]:
    domains = sorted(c.removeprefix("phase_0_") for c in columns if c.startswith("phase_0_"))
    missing = [domain for domain in domains if f"phase_1_{domain}" not in columns]
    if missing:
        raise ValueError(f"Missing phase_1 columns for {len(missing)} domains: {missing[:5]}")
    if not domains:
        raise ValueError("No phase_0_* columns found")
    return domains


def normalized_phase_arrays(df: pd.DataFrame, domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    w0 = df[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = df[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float)
    for name, weights in (("phase_0", w0), ("phase_1", w1)):
        if np.any(~np.isfinite(weights)):
            raise ValueError(f"{name} contains non-finite weights")
        if np.any(weights < -1e-12):
            raise ValueError(f"{name} contains negative weights")
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError(f"{name} contains a row with non-positive mass")
        weights /= row_sums[:, None]
    return w0, w1


def summarize_table(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    domains = phase_domains(list(df.columns))
    w0, w1 = normalized_phase_arrays(df, domains)
    return {
        "path": str(path.relative_to(PACKET_ROOT)),
        "rows": len(df),
        "columns": len(df.columns),
        "domains": len(domains),
        "phase_0_sum_min": float(w0.sum(axis=1).min()),
        "phase_0_sum_max": float(w0.sum(axis=1).max()),
        "phase_1_sum_min": float(w1.sum(axis=1).min()),
        "phase_1_sum_max": float(w1.sum(axis=1).max()),
        "scale_counts": (
            df["scale_display_label"].value_counts(dropna=False).to_dict()
            if "scale_display_label" in df
            else df.get("scale", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
        ),
    }


def main() -> None:
    tables = [
        PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_runs.csv",
        PACKET_ROOT / "data" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv",
        PACKET_ROOT / "data" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_noise.csv",
    ]
    summaries = [summarize_table(path) for path in tables if path.exists()]

    npz_path = PACKET_ROOT / "data" / "analysis_dataset" / "nd_scale_packet.npz"
    if npz_path.exists():
        packet = np.load(npz_path, allow_pickle=True)
        summaries.append(
            {
                "path": str(npz_path.relative_to(PACKET_ROOT)),
                "arrays": {key: list(packet[key].shape) for key in packet.files},
            }
        )

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
