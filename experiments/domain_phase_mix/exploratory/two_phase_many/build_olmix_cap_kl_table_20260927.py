# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Fill the 90-cell matched-Olmix epoch-cap x KL grid at 3e18 FLOPs and write its appendix table.

Reads the sweep's `grid_map.csv` (every cell, the run that measures it, and whether that run trained the cell's own
mixture) and the collector's `measured_results.csv` (`collect_delphi_3e18_validation_results_20260906.py --launch
olmix_cap_kl_sweep`). Uncheatable cells use the fixed seven-component weighting; OlmoBaseEval Easy cells use the
native 51-component mean. A cell whose mixture lies within total variation 0.01 of an earlier cell's, or any cap-1
cell above KL weight 0, shows the value of the run that measures it, set in italics.

Writes `filled_grid.csv`, `table_rows.tex` (the tabular body) and `table_summary.json` next to the inputs.

usage: uv run --offline --no-sync python build_olmix_cap_kl_table_20260927.py
"""

import json
import math
from pathlib import Path

import pandas as pd
from uncheatable_objective import uncheatable_weights

SWEEP = Path(__file__).resolve().parent / "reference_outputs" / "delphi_olmix_cap_kl_sweep_3e18_20260926"
CAPS = ("1", "4", "8", "12", "none")
KLS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5)
METRIC: dict[str, str] = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
DEPLOYED = {"uncheatable": ("4", 0.05), "table9": ("4", 0.005)}
MARINER_SEED0 = SWEEP.parent / "delphi_frozen_procedure_validation_3e18_20260908" / "measured_results.csv"
CAPTION = (
    "Measured BPB of matched Olmix proposals at Qwen3 360M/1.6B across the epoch cap and the KL weight $\\lambda$ of "
    "Olmix's optimizer, each trained once at the data and trainer seeds of MARINER's seed-0 runs ({u:.4f} on "
    "Uncheatable, {t:.4f} on OlmoBaseEval Easy). Italic entries repeat the run of an earlier cell whose optimized "
    "mixture lies within total variation 0.01; cap-1 entries above $\\lambda=0$ repeat the $\\lambda=0$ run, since a "
    "one-epoch cap keeps every mixture within total variation 0.1 of proportional. Bold marks the lowest loss per "
    "objective; $\\dagger$ marks the deployed Olmix setting."
)


def filled_grid(grid: pd.DataFrame, measured: pd.DataFrame) -> pd.DataFrame:
    """Every cell with the objective's measured loss of the run that measures it."""
    values = measured.set_index("candidate_id")
    rows = []
    for cell in grid.itertuples():
        run = values.loc[cell.run]
        value = float(run[METRIC[str(cell.target)]]) if run["status"] == "measured" else math.nan
        rows.append(
            {
                "target": cell.target,
                "cap": str(cell.cap),
                "kl": float(cell.kl),
                "run": cell.run,
                "run_directly": bool(cell.run_directly),
                "tv_to_run": float(cell.tv_to_run),
                "measured": value,
            }
        )
    return pd.DataFrame(rows)


def cell_text(cell: pd.Series, best: float, deployed: bool) -> str:
    if math.isnan(cell.measured):
        return "--"
    text = f"{cell.measured:.4f}"
    if not cell.run_directly:
        text = f"\\textit{{{text}}}"
    if math.isclose(cell.measured, best, rel_tol=0.0, abs_tol=5e-5):
        text = f"\\textbf{{{text}}}"
    if deployed:
        text = f"{text}\\rlap{{$^\\dagger$}}"  # the dagger does not widen its column
    return text


def table_rows(filled: pd.DataFrame) -> str:
    best = {target: filled[filled.target == target].measured.min() for target in METRIC}
    lines = []
    for kl in KLS:
        cells = []
        for target in METRIC:
            for cap in CAPS:
                (cell,) = [
                    row for _, row in filled.iterrows() if row.target == target and row.cap == cap and row.kl == kl
                ]
                cells.append(cell_text(cell, best[target], DEPLOYED[target] == (cap, kl)))
        lines.append(f"{kl:g} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def mariner_seed0() -> tuple[float, float]:
    """MARINER's seed-0 losses at the sweep's seeds: fixed-weight Uncheatable and the OlmoBaseEval Easy mean."""
    runs = pd.read_csv(MARINER_SEED0).set_index("candidate_id")
    u = runs.loc["lwspu_u_snc_cap06"]
    # Metric keys are eval/uncheatable_eval/<component>/bpb; the collector writes uncheatable_<component>_bpb.
    uncheatable = sum(w * float(u[f"uncheatable_{key.split('/')[2]}_bpb"]) for key, w in uncheatable_weights().items())
    return uncheatable, float(runs.loc["lwspu_t9_snc_cap08", "table9_macro_bpb"])


def table_latex(rows: str, uncheatable: float, table9: float) -> str:
    return (
        "\\begin{table}[H]\n"
        f"\\caption{{{CAPTION.format(u=uncheatable, t=table9)}}}\n"
        "\\label{tab:a-olmix-cap-kl}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
        "& \\multicolumn{5}{c}{Uncheatable, by epoch cap} & \\multicolumn{5}{c}{OlmoBaseEval Easy, by epoch cap} \\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
        "$\\lambda$ & 1 & 4 & 8 & 12 & none & 1 & 4 & 8 & 12 & none \\\\\n\\midrule\n"
        f"{rows}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    grid = pd.read_csv(SWEEP / "grid_map.csv", dtype={"cap": str})
    measured = pd.read_csv(SWEEP / "measured_results.csv")
    filled = filled_grid(grid, measured)
    if len(filled) != 90:
        raise ValueError(f"Expected 90 cells, found {len(filled)}")
    filled.to_csv(SWEEP / "filled_grid.csv", index=False)
    rows = table_rows(filled)
    (SWEEP / "table_rows.tex").write_text(rows)
    (SWEEP / "table.tex").write_text(table_latex(rows, *mariner_seed0()))
    summary: dict[str, object] = {"cells": len(filled), "measured_cells": int(filled.measured.notna().sum())}
    for target in METRIC:
        part = filled[filled.target == target]
        best = part.loc[part.measured.idxmin()] if part.measured.notna().any() else None
        cap, kl = DEPLOYED[target]
        deployed = part[(part.cap == cap) & (part.kl == kl)].iloc[0]
        summary[target] = {
            "best": None
            if best is None
            else {"cap": best.cap, "kl": best.kl, "measured": best.measured, "run": best.run},
            "deployed": {"cap": cap, "kl": kl, "measured": deployed.measured},
            "directly_trained_cells": int(part.run_directly.sum()),
            "max_tv_to_measuring_run": float(part.tv_to_run.max()),
        }
    (SWEEP / "table_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
