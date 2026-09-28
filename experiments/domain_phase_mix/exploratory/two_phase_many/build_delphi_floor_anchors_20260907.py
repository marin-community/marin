# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task anchors for the fitted-floor link: proportional mean and repeat SD of every Delphi component.

Sources: `table9_reliability_20260905/snr_fit_components_delphi.csv` (51 OlmoBaseEval Easy tasks) and
`table9_reliability_20260905/proportional_uncheatable_components.csv` (the eleven proportional runs, seven
Uncheatable components). Output `reference_outputs/delphi_floor_anchors_20260907/anchors.csv` with the benchmark's
full component names, read by the registry entry `weibull_softplus_unscaled@fitted_floor_link`.

usage: uv run python build_delphi_floor_anchors_20260907.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RELIABILITY = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_floor_anchors_20260907" / "anchors.csv"
PANEL = "delphi_3e18_39bucket"
PANEL_NPZ = SCRIPT_DIR / "reference_outputs" / "delphi_offline_selection_20260906" / "inputs" / "panel.npz"


def main() -> None:
    table9 = pd.read_csv(RELIABILITY / "snr_fit_components_delphi.csv").dropna(subset=["component"])
    panel = np.load(PANEL_NPZ, allow_pickle=True)
    # The benchmark names most tasks `olmo_base_eval/easy_bpb/<task>/bpb` and the MMLU categories bare.
    names = {}
    for full in (str(c) for c in panel["table9_components"]):
        parts = full.split("/")
        names[parts[2] if len(parts) >= 4 else full] = full
    missing = sorted(set(table9.component) - set(names))
    if missing:
        raise ValueError(f"reliability components absent from the panel: {missing}")
    rows = [
        {
            "panel": PANEL,
            "target": "table9",
            "component": names[row.component],
            "proportional_bpb": float(row.proportional_mean),
            "repeat_sd": float(row.repeat_sd),
        }
        for row in table9.itertuples()
    ]
    proportional = pd.read_csv(RELIABILITY / "proportional_uncheatable_components.csv")
    columns = [c for c in proportional.columns if c.startswith("eval/uncheatable_eval/") and c.endswith("/bpb")]
    for column in columns:
        if column == "eval/uncheatable_eval/bpb":
            continue
        rows.append(
            {
                "panel": PANEL,
                "target": "uncheatable",
                "component": column,
                "proportional_bpb": float(proportional[column].mean()),
                "repeat_sd": float(proportional[column].std(ddof=1)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 58 or frame.component.duplicated().any():
        raise ValueError(f"expected 58 distinct components, found {len(frame)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(frame.groupby("target").agg(components=("component", "count"), median_sd=("repeat_sd", "median")))


if __name__ == "__main__":
    main()
