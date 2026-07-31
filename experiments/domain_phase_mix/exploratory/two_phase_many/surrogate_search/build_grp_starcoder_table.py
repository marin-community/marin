# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the comparison table for 2-phase StarCoder GRP vs DS-RE-CEQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_dsre_ceq import (
    compute_starcoder_dsre_ceq_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    compute_starcoder_grp_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_OUTPUT_CSV = SCRIPT_DIR / "grp_starcoder_evaluation_table_full.csv"
LATEX_OUTPUT = SCRIPT_DIR / "grp_starcoder_evaluation_table_rows.tex"
SUMMARY_OUTPUT = SCRIPT_DIR / "grp_starcoder_evaluation_summary.json"


def _display_float(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "--"
    return format(float(value), fmt)


def _grp_row(cv_seed: int, tune_seed: int) -> dict[str, Any]:
    row = compute_starcoder_grp_metrics(cv_seed=cv_seed, tune_seed=tune_seed)
    row["source"] = "in_repo_starcoder_grp"
    row["notes"] = "Tuned full-packet GRP parameters; 5-fold CV refits only the linear head."
    return row


def _dsre_row(cv_seed: int, fit_seed: int) -> dict[str, Any]:
    row = compute_starcoder_dsre_ceq_metrics(cv_seed=cv_seed, fit_seed=fit_seed)
    row["source"] = "in_repo_scaling_models.fit_dsre_ceq"
    row["notes"] = (
        "Live DS-RE-CEQ fit on the completed 2-phase StarCoder packet. "
        "March 2 slide used mean-per-fold CV; old reference was "
        f"R^2={row['old_cv_r2_meanfold']:.3f}, RMSE={row['old_cv_rmse_meanfold']:.4f}, "
        f"Spearman={row['old_cv_spearman_meanfold']:.3f}, Mean Regret@1={row['old_cv_regret_meanfold']:.4f}."
    )
    return row


def _latex_rows(table: pd.DataFrame) -> str:
    lines = []
    for row in table.itertuples(index=False):
        lines.append(
            f"{row.model} & "
            f"{int(row.n_params)} & "
            f"{_display_float(row.train_r2, '.3f')} & "
            f"{_display_float(row.cv_r2, '.3f')} & "
            f"{_display_float(row.cv_rmse, '.4f')} & "
            f"{_display_float(row.cv_spearman, '.3f')} & "
            f"{_display_float(row.cv_regret_at_1, '.4f')} & "
            f"{_display_float(row.cv_foldmean_regret_at_1, '.4f')} \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument("--tune-seed", type=int, default=0)
    parser.add_argument("--dsre-fit-seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=FULL_OUTPUT_CSV)
    parser.add_argument("--latex-output", type=Path, default=LATEX_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    args = parser.parse_args()

    grp_row = _grp_row(args.cv_seed, args.tune_seed)
    dsre_row = _dsre_row(args.cv_seed, args.dsre_fit_seed)
    full = pd.DataFrame([grp_row, dsre_row])
    full["model"] = pd.Categorical(full["model"], categories=["GRP", "DS-RE-CEQ"], ordered=True)
    full = full.sort_values("model").reset_index(drop=True)

    args.output_csv.write_text(full.to_csv(index=False))
    args.latex_output.write_text(_latex_rows(full))
    args.summary_output.write_text(
        json.dumps(
            {
                "dataset": "two_phase_starcoder",
                "rows": full.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(full.to_markdown(index=False))
    print(f"\nWrote {args.output_csv}")
    print(f"Wrote {args.latex_output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
