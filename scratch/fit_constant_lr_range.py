# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import re
from pathlib import Path

import numpy as np
import wandb

RUN_PATTERN = re.compile(r"^AUG-LRC-TPU-\d{3}-d512-(30|60|150|300)x-lr(0\.7|0\.85|1|1\.2|1\.4)$")
PROJECT = "marin-community/marin_moe"
OUTPUT_DIR = Path(__file__).parent
RAW_OUTPUT = OUTPUT_DIR / "20260827_constant_lr_terminal_sweep.csv"
FIT_OUTPUT = OUTPUT_DIR / "20260827_constant_lr_range_fit.json"
RECOMMENDED_GRID = (0.1, 0.2, 0.32, 0.45, 0.7)


def fixed_budget_fit(rows: list[dict[str, float]], budgets: tuple[int, ...]) -> tuple[float, float, float]:
    design = []
    losses = []
    for row in rows:
        budget = int(row["token_budget_x"])
        if budget not in budgets:
            continue
        log_multiplier = np.log(row["lr_multiplier"])
        design.append([*(float(candidate == budget) for candidate in budgets), log_multiplier, log_multiplier**2])
        losses.append(row["paloma_macro_loss"])
    coefficients = np.linalg.lstsq(np.asarray(design), np.asarray(losses), rcond=None)[0]
    slope, curvature = coefficients[-2:]
    optimum = float(np.exp(-slope / (2 * curvature)))
    return optimum, float(slope), float(curvature)


def main() -> None:
    api = wandb.Api(timeout=60)
    rows = []
    runs = api.runs(PROJECT, filters={"display_name": {"$regex": "^AUG-LRC-TPU-"}})
    for run in runs:
        match = RUN_PATTERN.match(run.name)
        if match is None:
            continue
        summary = dict(run.summary)
        rows.append(
            {
                "token_budget_x": int(match.group(1)),
                "lr_multiplier": float(match.group(2)),
                "paloma_macro_loss": float(summary["eval/paloma/macro_loss"]),
                "run_name": run.name,
                "state": run.state,
            }
        )

    rows.sort(key=lambda row: (row["token_budget_x"], row["lr_multiplier"]))
    assert len(rows) == 20
    assert len({(row["token_budget_x"], row["lr_multiplier"]) for row in rows}) == 20
    assert {row["state"] for row in rows} == {"finished"}

    with RAW_OUTPUT.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    budgets = (30, 60, 150, 300)
    individual = {}
    for budget in budgets:
        budget_rows = [row for row in rows if row["token_budget_x"] == budget]
        x = np.log([row["lr_multiplier"] for row in budget_rows])
        y = [row["paloma_macro_loss"] for row in budget_rows]
        curvature, slope, _ = np.polyfit(x, y, 2)
        individual[str(budget)] = float(np.exp(-slope / (2 * curvature)))

    shared_optimum, shared_slope, shared_curvature = fixed_budget_fit(rows, budgets)
    leave_one_out = {
        str(omitted): fixed_budget_fit(rows, tuple(budget for budget in budgets if budget != omitted))[0]
        for omitted in budgets
    }
    result = {
        "source_project": PROJECT,
        "source_run_count": len(rows),
        "metric": "eval/paloma/macro_loss",
        "fit": "budget fixed effects plus shared quadratic in log(lr_multiplier)",
        "individual_optimum_multiplier": individual,
        "shared_optimum_multiplier": shared_optimum,
        "shared_log_multiplier_slope": shared_slope,
        "shared_log_multiplier_curvature": shared_curvature,
        "leave_one_budget_out_optimum_multiplier": leave_one_out,
        "leave_one_budget_out_range": [min(leave_one_out.values()), max(leave_one_out.values())],
        "recommended_grid": RECOMMENDED_GRID,
    }
    FIT_OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
