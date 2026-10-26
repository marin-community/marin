# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the seed-replicate panel of the SFT-probe arms B1 and C at Delphi 3e18 (2026-10-23).

The panel copies the phase weights of arm B1 (two-phase ``dphase_t9b075_can_e0p005``) and arm C (its single-phase
average) from ``sft_midtraining_probe_20261014/arms_source_panel.csv`` and crosses each with four new data seeds and
two trainer seeds: sixteen runs that measure how far seed replicates of one fixed curriculum diverge, compared with
the existing init-only replicates at data seed 662009. Output: ``reference_outputs/delphi_seed_replicates_20261023/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS_PANEL = SCRIPT_DIR / "reference_outputs" / "sft_midtraining_probe_20261014" / "arms_source_panel.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_seed_replicates_20261023"
TEMPLATE_RUNS = {"B1": "sftp_b1_t0", "C": "sftp_c_t0"}
DATA_SEEDS = (10_230_000, 10_230_001, 10_230_002, 10_230_003)
TRAINER_SEEDS = (0, 1)
RUN_ID_BASE = 7_520_000
COPIED_COLUMNS = ("arm", "policy_class", "source_candidate", "phase_tv", "aggregate_max_abs_error")


def build_panel(arms: pd.DataFrame) -> pd.DataFrame:
    phase_columns = [c for c in arms.columns if c.startswith(("phase_0_", "phase_1_"))]
    rows = []
    for arm, template_id in TEMPLATE_RUNS.items():
        template = arms.loc[arms["candidate_id"] == template_id].iloc[0]
        for data_seed in DATA_SEEDS:
            for trainer_seed in TRAINER_SEEDS:
                row = {
                    "run_order": len(rows),
                    "run_id": RUN_ID_BASE + len(rows),
                    "candidate_id": f"sdrep_{arm.lower()}_d{data_seed}_t{trainer_seed}",
                    **{column: template[column] for column in COPIED_COLUMNS},
                    "data_seed": data_seed,
                    "trainer_seed": trainer_seed,
                    **{column: template[column] for column in phase_columns},
                }
                rows.append(row)
    ordered = [
        "run_order",
        "run_id",
        "candidate_id",
        "arm",
        "policy_class",
        "source_candidate",
        "data_seed",
        "trainer_seed",
        "phase_tv",
        "aggregate_max_abs_error",
        *phase_columns,
    ]
    return pd.DataFrame(rows)[ordered]


def main() -> None:
    arms = pd.read_csv(ARMS_PANEL)
    panel = build_panel(arms)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "source_panel.csv"
    panel.to_csv(panel_path, index=False)
    summary = {
        "source_panel": str(panel_path.relative_to(SCRIPT_DIR)),
        "source_panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "arms_panel_sha256": hashlib.sha256(ARMS_PANEL.read_bytes()).hexdigest(),
        "template_runs": TEMPLATE_RUNS,
        "data_seeds": DATA_SEEDS,
        "trainer_seeds": TRAINER_SEEDS,
        "runs": len(panel),
    }
    (OUTPUT_DIR / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
