# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main effects and two-factor interactions of the frontier factorial from its measured results.

Reads `measured_results.csv` written by `collect_delphi_3e18_validation_results_20260906.py --launch factorial`
and `design.csv`. On the sixteen corners the 2^(5-1) design is orthogonal, so every effect is the difference
between the mean at +1 and the mean at -1 (the regression coefficient is half of it); the resolution-V
aliasing makes each two-factor interaction confounded only with a three-factor one. The standard error of
an effect is the run-to-run SD over two (eight runs per level); the repeat SD of the Table-9 mean (0.0038)
and the centre pair's difference are both reported. Missing corners are tolerated: effects are then least-squares
estimates on the available corners and marked as such.

When all sixteen corners are measured the script also proposes the next runs: the fitted first-order model with
two-factor interactions is evaluated on all 32 corners of the factor box (the 16 unmeasured ones form the
complementary half fraction), on the main-effect-only step, and on the same steps at 1.5x and 2x the factor
deltas (extrapolation, flagged). Each proposal is rounded to the runtime grid, checked against the 16-epoch cap,
compared with the bank kernel's neighbour forecast, and written to `proposal_candidate_weights.csv` in the
launcher schema. Nothing is launched.

usage: uv run python analyze_delphi_frontier_factorial_20260906.py [--repeat-sd 0.0038]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    design_delphi_frontier_factorial_20260906 as design_script,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_one_phase_surrogate_challengers_20260831 as grid,
)

DESIGN_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_frontier_factorial_design_20260906"
CENTRE_MEAN = 1.0639
CENTRE_SD = 0.0041
CENTRE_RUNS = 26
STEP_SCALES = (1.0, 1.5, 2.0)
KERNEL_BANDWIDTH = 0.05
MAIN_EFFECT_T = 2.0


def effects_table(design: pd.DataFrame, response: pd.Series, repeat_sd: float) -> pd.DataFrame:
    factors = [column for column in design.columns if column.startswith("factor_")]
    letters = [column.split("_")[1] for column in factors]
    corners = design[design[factors].abs().eq(1).all(axis=1)].copy()
    corners = corners[corners.candidate_id.isin(response.index)]
    y = response.loc[corners.candidate_id].to_numpy(float)
    columns, names = [np.ones(len(corners))], ["intercept"]
    for column, letter in zip(factors, letters, strict=True):
        columns.append(corners[column].to_numpy(float))
        names.append(letter)
    for (column_a, letter_a), (column_b, letter_b) in itertools.combinations(zip(factors, letters, strict=True), 2):
        columns.append((corners[column_a] * corners[column_b]).to_numpy(float))
        names.append(f"{letter_a}{letter_b}")
    matrix = np.column_stack(columns)
    complete = len(corners) == 16
    if complete:
        coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        # Effect = mean(+1) - mean(-1) = 2 * coefficient; SE of a half-mean difference at eight runs per level.
        effect = 2 * coefficients
        standard_error = np.full(len(names), repeat_sd / 2)
    else:
        # Without all sixteen corners the interaction columns are not identifiable; fit main effects only.
        main = matrix[:, : 1 + len(letters)]
        if len(corners) <= main.shape[1]:
            raise ValueError(f"Only {len(corners)} corners measured; main effects need more than {main.shape[1]}")
        coefficients, *_ = np.linalg.lstsq(main, y, rcond=None)
        covariance = np.linalg.pinv(main.T @ main) * repeat_sd**2
        effect = np.full(len(names), np.nan)
        standard_error = np.full(len(names), np.nan)
        effect[: main.shape[1]] = 2 * coefficients
        standard_error[: main.shape[1]] = 2 * np.sqrt(np.diag(covariance))
    table = pd.DataFrame({"term": names, "effect_bpb": effect, "se": standard_error})
    table["t"] = table.effect_bpb / table.se
    table["kind"] = ["intercept"] + ["main"] * len(letters) + ["interaction"] * (len(names) - 1 - len(letters))
    table["corners_used"] = len(corners)
    table.loc[0, ["effect_bpb", "se", "t"]] = [coefficients[0], np.nan, np.nan]
    return table


def predict_box(table: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Fitted first-order-plus-interactions model at coded points (rows x factors), from the effects table."""
    letters = [term for term in table.term if len(term) == 1 and term != "intercept"]
    coefficient = table.set_index("term").effect_bpb
    value = np.full(len(points), float(coefficient["intercept"]))
    for index, letter in enumerate(letters):
        value += 0.5 * float(coefficient[letter]) * points[:, index]
    for (i, a), (j, b) in itertools.combinations(enumerate(letters), 2):
        value += 0.5 * float(coefficient[f"{a}{b}"]) * points[:, i] * points[:, j]
    return value


def proposals(design: pd.DataFrame, table: pd.DataFrame, repeat_sd: float, output: Path) -> pd.DataFrame:
    """Rank every corner and main-effect step by the fitted model; write the launcher table of the best ones."""
    factors = [column for column in design.columns if column.startswith("factor_")]
    corners = design[design[factors].abs().eq(1).all(axis=1)]
    measured_signs = {tuple(int(v) for v in row) for row in corners[factors].to_numpy()}
    main = table[table.kind.eq("main")].set_index("term").effect_bpb
    main_t = table[table.kind.eq("main")].set_index("term").t
    points, labels = [], []
    for signs in itertools.product((-1, 1), repeat=len(factors)):
        points.append(np.array(signs, float))
        labels.append(("measured corner" if signs in measured_signs else "unmeasured corner", 1.0))
    step = np.array([-np.sign(main[letter]) if abs(main_t[letter]) >= MAIN_EFFECT_T else 0.0 for letter in main.index])
    for scale in STEP_SCALES:
        points.append(step * scale)
        labels.append(("main-effect step", scale))
    for scale in STEP_SCALES[1:]:
        best_corner = points[int(np.argmin(predict_box(table, np.stack(points[:32]))))]
        points.append(best_corner * scale)
        labels.append(("best corner scaled", scale))
    coded = np.stack(points)
    predicted = predict_box(table, coded)
    data = benchmark.read_npz(design_script.FROZEN / "inputs" / "panel.npz")
    buckets = [str(b) for b in data["buckets"]]
    inventory = pd.Series(data["inventory"], index=buckets)
    bank = benchmark.read_npz(design_script.FROZEN / "inputs" / "table9_bank_features.npz")
    bank_labels = pd.read_csv(design_script.FROZEN / "inputs" / "table9_bank_labels.csv").set_index("coordinate_id")
    frame = pd.DataFrame(bank["weights"], columns=buckets, index=bank["coordinate_id"].astype(str))
    centre = frame.loc[design_script.CENTRE_ID]
    measured = bank_labels.loc[frame.index, "measured_mean_bpb"].to_numpy(float)
    maximum = np.floor(
        np.minimum(1.0, design_script.EPOCH_CAP / inventory.to_numpy()) * design_script.BLOCK_SIZE + 1e-12
    ).astype(np.int64)
    rows, weight_rows = [], []
    for point, (kind, scale), value in zip(coded, labels, predicted, strict=True):
        signs = tuple(float(v) for v in point)
        name = (
            "prop_"
            + "".join("p" if v > 0 else ("m" if v < 0 else "z") for v in signs)
            + f"_x{scale:g}".replace(".", "p")
        )
        try:
            weights = design_script.perturb(centre, signs) if any(signs) else centre.copy()
        except ValueError as error:
            rows.append(
                {
                    "candidate_id": name,
                    "kind": kind,
                    "scale": scale,
                    "predicted_bpb": value,
                    "feasible": False,
                    "reason": str(error)[:80],
                }
            )
            continue
        counts = grid.prefix_materializer.constrained_counts(weights.to_numpy(), maximum)
        feasible = int(counts.sum()) == design_script.BLOCK_SIZE and not np.any(counts > maximum)
        runtime = counts / design_script.BLOCK_SIZE
        distance = np.abs(frame.to_numpy() - runtime[None, :]).sum(axis=1) / 2
        kernel = np.exp(-0.5 * (distance / KERNEL_BANDWIDTH) ** 2)
        rows.append(
            {
                "candidate_id": name,
                "kind": kind,
                "scale": scale,
                **{f"x_{letter}": float(v) for letter, v in zip("ABCDE", signs, strict=True)},
                "predicted_bpb": float(value),
                "prediction_se": float(repeat_sd),
                "feasible": bool(feasible),
                "tv_to_centre": float(np.abs(runtime - centre.to_numpy()).sum() / 2),
                "max_materialized_epoch": float((runtime * inventory.to_numpy()).max()),
                "nearest_bank_tv": float(distance.min()),
                "kernel_forecast": float(kernel @ measured / kernel.sum()),
                "kernel_mass": float(kernel.sum()),
            }
        )
        if feasible and kind != "measured corner":
            weight_rows.extend(
                {
                    "candidate_id": name + "_cap16",
                    "target": "table9",
                    "target_label": "Table-9 macro",
                    "epoch_cap": int(design_script.EPOCH_CAP),
                    "domain": bucket,
                    "runtime_count": int(count),
                    "weight": float(count / design_script.BLOCK_SIZE),
                    "materialized_epochs": float(count / design_script.BLOCK_SIZE * inventory[bucket]),
                }
                for bucket, count in zip(buckets, counts, strict=True)
            )
    ranking = pd.DataFrame(rows).sort_values("predicted_bpb")
    ranking.to_csv(output / "proposal_ranking.csv", index=False)
    weights_table = pd.DataFrame(weight_rows)
    keep = ranking[ranking.feasible & ranking.kind.ne("measured corner")].candidate_id.head(6) + "_cap16"
    weights_table[weights_table.candidate_id.isin(keep)].to_csv(output / "proposal_candidate_weights.csv", index=False)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--repeat-sd", type=float, default=0.0038, help="run-to-run SD of the Table-9 mean")
    args = parser.parse_args()
    design = pd.read_csv(args.design_dir / "design.csv")
    measured = pd.read_csv(args.design_dir / "measured_results.csv")
    measured = measured[measured.status.eq("measured")]
    lines = [f"# Frontier factorial results ({len(measured)} of {len(design)} runs measured)", ""]
    pd.set_option("display.width", 220)
    for response_name, sd in (("table9_macro_bpb", args.repeat_sd), ("uncheatable_bpb", 0.0009)):
        if response_name not in measured:
            lines.append(f"## {response_name}: nothing measured yet")
            continue
        response = measured.dropna(subset=[response_name]).set_index("candidate_id")[response_name]
        if response.empty:
            lines.append(f"## {response_name}: nothing measured yet")
            continue
        centre = response[response.index.str.startswith("centre_")]
        corners_measured = int(response.index.str.startswith("fac_").sum())
        if corners_measured < 7:
            lines.append(f"## {response_name}: {corners_measured} corners measured, too few for main effects")
            lines.append("")
            print(lines[-2])
            continue
        table = effects_table(design, response, sd)
        table.to_csv(args.design_dir / f"effects_{response_name}.csv", index=False)
        lines.append(f"## {response_name}")
        lines.append("")
        if len(centre):
            reference = (
                f"; {CENTRE_RUNS}-run centre mean {CENTRE_MEAN:.4f} (SD {CENTRE_SD:.4f})"
                if response_name == "table9_macro_bpb"
                else ""
            )
            lines.append(
                f"Centre runs: {', '.join(f'{index} {value:.4f}' for index, value in centre.items())}{reference}."
                + (f" Centre pair difference {abs(centre.iloc[0] - centre.iloc[1]):.4f}." if len(centre) == 2 else "")
            )
        lines.append("")
        lines.append(table.round(5).to_string(index=False))
        lines.append("")
        big = table[table.kind.eq("interaction") & table.t.abs().ge(2.5)]
        lines.append(
            "Interactions beyond 2.5 SE: "
            + (", ".join(f"{row.term} {row.effect_bpb:+.4f}" for row in big.itertuples()) or "none")
        )
        lines.append("")
        print("\n".join(lines[-len(table) - 8 :]))
        if response_name == "table9_macro_bpb" and int(table.corners_used.iloc[0]) == 16:
            ranking = proposals(design, table, sd, args.design_dir)
            shown = ranking.head(10)[
                [
                    "candidate_id",
                    "kind",
                    "scale",
                    "predicted_bpb",
                    "feasible",
                    "tv_to_centre",
                    "max_materialized_epoch",
                    "kernel_forecast",
                    "kernel_mass",
                ]
            ]
            lines.append("## Proposals (fitted model over the factor box; `proposal_ranking.csv`)")
            lines.append("")
            lines.append(shown.round(4).to_string(index=False))
            lines.append("")
            print("\n".join(lines[-4:]))
    (args.design_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.design_dir / 'report.md'}")


if __name__ == "__main__":
    main()
