# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Table 5 rows with three trainer seeds for the baselines that had one run at 3e18 FLOPs.

Combines each baseline's trainer-seed-0 run with its two repeats from `delphi_table5_repeats_3e18_20260928` and pairs
every run with MARINER's run at the same trainer seed (frozen-procedure validation for seed 0, fairness repeats for
seeds 1 and 2), as `summarize_delphi_comparator_proposals_20260909.py` does for the other rows. Seed-0 sources: the
comparator proposals launch (natural cubic spline, tuned RegMix, Hellinger kernel ridge), the released-RegMix
endpoints rescored to the fixed Uncheatable weighting from their component columns, and the swarm panel's
measurement of run_00125 (the lowest single-run aggregate of the 280 swarm designs on both objectives) for the best
swarm run. Uncheatable is the fixed seven-component weighting throughout.

Writes `table5_three_seed_rows.csv` and `table5_rows.tex` to `reference_outputs/delphi_table5_repeats_3e18_20260928/`.

usage: uv run --offline --no-sync python summarize_table5_repeats_20260928.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import summarize_delphi_comparator_proposals_20260909 as paired  # noqa: E402
from uncheatable_objective import uncheatable_weights  # noqa: E402

REFERENCE = SCRIPT_DIR / "reference_outputs"
REPEATS = REFERENCE / "delphi_table5_repeats_3e18_20260928"
COMPARATORS = REFERENCE / "delphi_comparator_proposals_3e18_20260909" / "measured_results.csv"
RELEASED_REGMIX = REFERENCE / "delphi_regmix_reference_3e18_20260913" / "measured_results.csv"
PANEL = REFERENCE / "delphi_offline_selection_20260906" / "inputs" / "panel.npz"
SWARM_RUN = "singleavg_fit_125_run_00125"
PANEL_AGGREGATE = {"uncheatable": "uncheatable_aggregate", "table9": "table9_aggregate"}
SEEDS = (0, 1, 2)
# Table rows: label, candidate per objective, and where the trainer-seed-0 measurement comes from.
ROWS = (
    ("Natural cubic spline", {"uncheatable": "cmp_u_spline_cap08", "table9": "cmp_t9_spline_cap12"}, "comparators"),
    ("RegMix (tuned)", {"uncheatable": "cmp_u_lgbm_cap08", "table9": "cmp_t9_lgbm_cap12"}, "comparators"),
    ("RegMix (released)", {"uncheatable": "rgref_u_endpoint_cap64", "table9": "rgref_t9_endpoint_cap64"}, "released"),
    ("Hellinger kernel ridge", {"uncheatable": "cmp_u_krr_cap12", "table9": "cmp_t9_krr_cap16"}, "comparators"),
    ("Best of the 280 swarm runs", {"uncheatable": "swarm_run00125_cap08", "table9": "swarm_run00125_cap08"}, "panel"),
)


@dataclass(frozen=True)
class Cell:
    label: str
    target: str
    candidate_id: str
    measured: tuple[float, float, float]
    mean: float
    std: float
    difference_mean: float
    difference_se: float


def fixed_weighting(table: pd.DataFrame) -> pd.Series:
    """The paper's Uncheatable objective recomputed from the seven component columns."""
    total = pd.Series(0.0, index=table.index)
    for key, weight in uncheatable_weights().items():
        total += weight * table[f"uncheatable_{key.split('/')[2]}_bpb"].astype(float)
    return total


def measured_objective(table: pd.DataFrame, target: str) -> pd.Series:
    """Objective column per target; Uncheatable from the fixed weighting so legacy and schema-2 rows agree."""
    if target == "table9":
        return table["table9_macro_bpb"].astype(float)
    fixed = fixed_weighting(table)
    legacy = ~table.get("uncheatable_bpb_schema", pd.Series(np.nan, index=table.index)).notna()
    if ((fixed - table["uncheatable_bpb"]).abs()[~legacy] > 1e-9).any():
        raise ValueError("a schema-2 collection does not carry the fixed weighting in uncheatable_bpb")
    return fixed.astype(float)


def seed_zero(source: str, candidate_id: str, target: str) -> float:
    if source == "panel":
        panel = np.load(PANEL, allow_pickle=True)
        runs = [str(run) for run in panel["runs"]]
        aggregate = panel[PANEL_AGGREGATE[target]]
        if runs.index(SWARM_RUN) != int(np.nanargmin(aggregate)):
            raise ValueError(f"{SWARM_RUN} is not the swarm minimum on {target}")
        return float(aggregate[runs.index(SWARM_RUN)])
    path = COMPARATORS if source == "comparators" else RELEASED_REGMIX
    table = pd.read_csv(path)
    table = table[table.status.eq("measured") & table.candidate_id.eq(candidate_id)]
    if len(table) != 1:
        raise ValueError(f"expected one seed-0 run of {candidate_id} in {path.parent.name}, found {len(table)}")
    return float(measured_objective(table, target).iloc[0])


def repeats(candidate_id: str, target: str) -> pd.Series:
    table = pd.read_csv(REPEATS / "measured_results.csv")
    table = table[table.status.eq("measured") & table.candidate_id.eq(candidate_id)].copy()
    table["measured"] = measured_objective(table, target)
    series = table.set_index(table.trainer_seed.astype(int))["measured"].sort_index()
    if series.index.tolist() != [1, 2]:
        raise ValueError(f"expected trainer seeds 1 and 2 for {candidate_id}, found {series.index.tolist()}")
    return series


def cell(label: str, target: str, candidate_id: str, source: str, mariner: pd.Series) -> Cell:
    series = pd.concat([pd.Series({0: seed_zero(source, candidate_id, target)}), repeats(candidate_id, target)])
    series.index = series.index.astype(int)
    if series.index.tolist() != list(SEEDS) or mariner.index.tolist() != list(SEEDS):
        raise ValueError(f"{candidate_id}: seeds {series.index.tolist()} against MARINER {mariner.index.tolist()}")
    differences = series - mariner
    return Cell(
        label=label,
        target=target,
        candidate_id=candidate_id,
        measured=tuple(float(value) for value in series),
        mean=float(series.mean()),
        std=float(series.std(ddof=1)),
        difference_mean=float(differences.mean()),
        difference_se=float(differences.std(ddof=1) / np.sqrt(len(series))),
    )


def latex_row(cells: dict[str, Cell]) -> str:
    parts = [cells["uncheatable"].label]
    for target in ("uncheatable", "table9"):
        c = cells[target]
        parts.append(f"${c.mean:.4f} \\pm {c.std:.4f}$")
        parts.append(f"${c.difference_mean:+.3f} \\pm {c.difference_se:.3f}$")
    return " & ".join(parts) + " \\\\"


def main() -> None:
    mariner = paired.mariner_runs()
    records, lines = [], []
    for label, candidates, source in ROWS:
        cells = {target: cell(label, target, candidates[target], source, mariner[target]) for target in candidates}
        lines.append(latex_row(cells))
        for c in cells.values():
            records.append(
                {
                    "label": c.label,
                    "target": c.target,
                    "candidate_id": c.candidate_id,
                    "seed0_source": source,
                    "measured_seed0": c.measured[0],
                    "measured_seed1": c.measured[1],
                    "measured_seed2": c.measured[2],
                    "mean": c.mean,
                    "std": c.std,
                    "difference_vs_mariner_mean": c.difference_mean,
                    "difference_vs_mariner_se": c.difference_se,
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(REPEATS / "table5_three_seed_rows.csv", index=False)
    (REPEATS / "table5_rows.tex").write_text("\n".join(lines) + "\n")
    print("MARINER:", {t: [round(v, 5) for v in s] for t, s in mariner.items()})
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
