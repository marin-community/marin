# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Floor anchors for the two Llama swarms (300m_39bucket = Llama 200M/6B, 60m_39bucket = Llama 160M/1.2B).

The floored surrogate anchors each task's floor to the proportional mixture's mean and repeat SD. The Delphi
anchors come from `build_delphi_floor_anchors_20260907.py`; this script appends the Llama panels to the same
file so the final model can be fitted on all three swarms. Sources, by decreasing completeness:

- 300M Table 9: the ten trainer-seed proportional repeats plus the baseline run of
  `one_phase_swarm_scores_export_300m_20260630/proportional_reference_uncheatable_table9_scores_300m.csv`
  (per-component mean and SD over the eleven runs).
- 300M Uncheatable and both 60M targets: no per-component repeats exist, so the anchor is the panel's own
  proportional run (one run) and the noise SD is the panel's aggregate repeat SD for that target, the value the
  observatory harness uses as the panel noise floor. The margin only matters for tasks the swarm barely moved.

usage: uv run python build_llama_floor_anchors_20260907.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)

ANCHORS = SCRIPT_DIR / "reference_outputs" / "delphi_floor_anchors_20260907" / "anchors.csv"
THREE_HUNDRED_M_REFERENCE = (
    SCRIPT_DIR
    / "reference_outputs"
    / "one_phase_swarm_scores_export_300m_20260630"
    / "proportional_reference_uncheatable_table9_scores_300m.csv"
)
PANELS = ("300m_39bucket", "60m_39bucket")
PROPORTIONAL_RUN = "singleavg_baseline_proportional"


def panel_rows(name: str) -> list[dict[str, object]]:
    panel = harness.load_panel(name)
    runs = list(panel.runs)
    if runs.count(PROPORTIONAL_RUN) != 1:
        raise ValueError(f"{name}: expected one proportional run, found {runs.count(PROPORTIONAL_RUN)}")
    index = runs.index(PROPORTIONAL_RUN)
    reference = pd.read_csv(THREE_HUNDRED_M_REFERENCE) if name == "300m_39bucket" else None
    rows: list[dict[str, object]] = []
    for group in panel.groups:
        aggregate_sd = float(panel.repeat_sd[group.name])
        for column, component in enumerate(group.components):
            proportional = float(group.outcomes[index, column])
            source = "panel_proportional_run+aggregate_repeat_sd"
            repeat_sd = aggregate_sd
            if reference is not None and group.name == "table9":
                key = component
                if key not in reference.columns:
                    raise ValueError(f"{name}: component {component!r} missing from the 300M proportional reference")
                values = reference[key].to_numpy(float)
                proportional = float(np.mean(values))
                repeat_sd = float(np.std(values, ddof=1))
                source = "300m_proportional_reference_11_runs"
            rows.append(
                {
                    "panel": name,
                    "target": group.name,
                    "component": component,
                    "proportional_bpb": proportional,
                    "repeat_sd": repeat_sd,
                    "source": source,
                }
            )
    return rows


def main() -> None:
    existing = pd.read_csv(ANCHORS)
    if "source" not in existing.columns:
        existing["source"] = "delphi_reliability_package"
    kept = existing[~existing["panel"].isin(PANELS)]
    added = pd.DataFrame([row for name in PANELS for row in panel_rows(name)])
    frame = pd.concat([kept, added], ignore_index=True)
    frame.to_csv(ANCHORS, index=False)
    print(frame.groupby(["panel", "target"]).agg(components=("component", "count"), median_sd=("repeat_sd", "median")))


if __name__ == "__main__":
    main()
