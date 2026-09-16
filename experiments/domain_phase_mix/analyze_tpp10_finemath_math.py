# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect verified FineMath math likelihoods without launching evaluation or training."""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.domain_phase_mix import evaluate_tpp10_finemath_math as evaluation
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

ARCHIVE = evaluation.DIRECTORY.parent / "plots_20260912"
UNCHEATABLE = "eval/uncheatable_eval/macro_bpb"
DATASETS = {"math500": "MATH-500", "gsm8k": "GSM8K"}
COLORS = {"math500": "#0072B2", "gsm8k": "#009E73", "uncheatable": "#555555"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def archived_points(spec: dict, archive: Path) -> dict[int, dict]:
    """Match archived epoch counts and broad evaluation to the exact proxy grid."""
    preflight = read_json(archive / "preflight.json")
    curves = read_json(archive / "curves.json")["curves"]
    selected = [c for c in curves if c["domain"] == "finemath_3plus" and c["arm"] == "matched"]
    if len(selected) != 1:
        raise ValueError("Expected one archived FineMath matched curve")
    points = {p["percent"]: p for p in selected[0]["points"]}
    if len(points) != len(selected[0]["points"]) or tuple(sorted(points)) != evaluation.GRID:
        raise ValueError("Archived FineMath curve does not contain the unique complete grid")
    coordinates = {p["percent"]: p for p in preflight["allocation"]["coordinates"]}
    pool_tokens = preflight["data"]["caches"]["finemath_3plus/matched"]["tokens"]
    for endpoint in spec["endpoints"]:
        request = endpoint["request"]
        percent = request["percent"]
        point = points[percent]
        expected_name = "shared p0" if percent == 0 else request["run_name"]
        if point["run_name"] != expected_name:
            raise ValueError(f"Archived run identity differs at p={percent}")
        if not math.isclose(point["epochs"], coordinates[percent]["matched_epochs"], abs_tol=1e-12):
            raise ValueError(f"Archived epoch coordinates disagree at p={percent}")
        if percent:
            if request["support_sequences"] * evaluation.experiment.SEQ_LEN != pool_tokens:
                raise ValueError(f"FineMath pool size differs at p={percent}")
            allocated = coordinates[percent]["proxy_allocation"]["starcoder"]
            if not math.isclose(point["epochs"], allocated / request["support_sequences"], abs_tol=1e-12):
                raise ValueError(f"Realized allocation does not reproduce epochs at p={percent}")
        if not math.isfinite(point["metrics"][UNCHEATABLE]):
            raise ValueError(f"Archived broad-evaluation metric is invalid at p={percent}")
    return points


def run_identities(spec: dict, path: Path) -> dict[str, dict]:
    """Preserve the Fieldbook evaluation and original training run identifiers."""
    records = json.loads(path.read_text())
    identities = {r["name"]: r for r in records}
    names = {e["request"]["run_name"] + "_math_likelihood" for e in spec["endpoints"]}
    if len(records) != len(identities) or set(identities) != names:
        raise ValueError("Fieldbook identities differ from the eight evaluation endpoints")
    if any(not r["parent_run_id"] or not r["id"] for r in records):
        raise ValueError("Missing training or evaluation Fieldbook run identity")
    return {name.removesuffix("_math_likelihood"): record for name, record in identities.items()}


def collect(spec: dict, points: dict[int, dict], identities: dict[str, dict]) -> list[dict]:
    """Read verified receipts only, retaining all endpoints and missing-result status."""
    population_sha256 = canonical_sha256(spec["population_counts"])
    rows = []
    for endpoint in spec["endpoints"]:
        request = endpoint["request"]
        run_name = request["run_name"]
        point = points[request["percent"]]
        result = evaluation.verified_result(spec, endpoint)
        if result is not None and result["population_sha256"] != population_sha256:
            raise ValueError(f"Scored population differs from the frozen specification: {run_name}")
        rows.append(
            {
                "percent": request["percent"],
                "epochs": point["epochs"],
                "status": "verified" if result is not None else "missing",
                "run_name": run_name,
                "training_run_id": identities[run_name]["parent_run_id"],
                "evaluation_run_id": identities[run_name]["id"],
                "checkpoint": endpoint["checkpoint"],
                "receipt_uri": f"{evaluation.output_root(spec)}/{run_name}.json",
                "uncheatable_bpb": point["metrics"][UNCHEATABLE],
                "result": result,
            }
        )
    return rows


def observed_minimum(rows: list[dict], dataset: str) -> dict:
    """Describe the observed grid minimum and gaps, without extrapolating a turnover."""
    available = [r for r in rows if r["result"] is not None]
    if not available:
        return {"verified_points": 0, "minimum": None}

    def loss(row: dict) -> float:
        return row["result"]["metrics"][f"eval/{dataset}/loss"]

    def detail(row: dict) -> dict:
        return {
            "percent": row["percent"],
            "epochs": row["epochs"],
            "loss": loss(row),
            "perplexity": row["result"]["perplexity"][dataset],
            "run_name": row["run_name"],
        }

    ranked = sorted(available, key=lambda r: (loss(r), r["percent"]))
    best = ranked[0]
    ties = [r for r in ranked if loss(r) == loss(best)]
    boundary = best["percent"] in (evaluation.GRID[0], evaluation.GRID[-1])
    neighbors = []
    index = evaluation.GRID.index(best["percent"])
    for neighbor in (index - 1, index + 1):
        if 0 <= neighbor < len(rows):
            row = rows[neighbor]
            neighbors.append(
                {
                    "percent": row["percent"],
                    "epochs": row["epochs"],
                    "status": row["status"],
                    "loss_gap_from_minimum": loss(row) - loss(best) if row["result"] is not None else None,
                    "perplexity_gap_from_minimum": (
                        row["result"]["perplexity"][dataset] - best["result"]["perplexity"][dataset]
                        if row["result"] is not None
                        else None
                    ),
                }
            )
    return {
        "verified_points": len(available),
        "complete_grid": len(available) == len(evaluation.GRID),
        "minimum": detail(best),
        "tied_minima": [detail(r) for r in ties],
        "at_prespecified_grid_boundary": boundary,
        "second_best": detail(ranked[1]) if len(ranked) > 1 else None,
        "second_best_loss_gap": loss(ranked[1]) - loss(best) if len(ranked) > 1 else None,
        "second_best_perplexity_gap": (
            ranked[1]["result"]["perplexity"][dataset] - best["result"]["perplexity"][dataset]
            if len(ranked) > 1
            else None
        ),
        "adjacent_grid_points": neighbors,
        "interpretation": (
            "Minimum among verified grid points. A boundary minimum does not establish an interior turnover "
            "or locate the optimum beyond the sampled range. One checkpoint per mixture; gaps have no seed uncertainty."
        ),
    }


def write_csv(rows: list[dict], spec: dict, destination: Path) -> None:
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"result", "checkpoint"}}
        flat["spec_sha256"] = spec["spec_sha256"]
        flat["checkpoint_path"] = row["checkpoint"]["path"]
        flat["checkpoint_metadata_sha256"] = row["checkpoint"]["metadata_sha256"]
        for dataset in DATASETS:
            result = row["result"]
            flat[f"{dataset}_loss"] = result["metrics"][f"eval/{dataset}/loss"] if result is not None else None
            flat[f"{dataset}_perplexity"] = result["perplexity"][dataset] if result is not None else None
            flat[f"{dataset}_scored_tokens"] = spec["population_counts"][dataset]["scored_tokens"]
            flat[f"{dataset}_problems"] = spec["population_counts"][dataset]["problems"]
        flat_rows.append(flat)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def plot_results(rows: list[dict], destination: Path, include_uncheatable: bool) -> list[str]:
    """Plot measured perplexities and their minima, leaving missing-grid gaps visible."""
    completed = sum(r["result"] is not None for r in rows)
    if not completed:
        return []
    panels = list(DATASETS)
    if include_uncheatable:
        panels.append("uncheatable")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.8), layout="constrained", sharex=True)
    epochs = np.array([r["epochs"] for r in rows])
    for ax, dataset in zip(axes, panels, strict=True):
        if dataset == "uncheatable":
            values = np.array([r["uncheatable_bpb"] for r in rows])
            ax.set_title("Uncheatable · existing evaluation")
            ax.set_ylabel("Mean component BPB")
        else:
            values = np.array([r["result"]["perplexity"][dataset] if r["result"] is not None else np.nan for r in rows])
            role = "primary" if dataset == "math500" else "secondary"
            ax.set_title(f"{DATASETS[dataset]} · {role}")
            ax.set_ylabel("Reference-solution perplexity")
        ax.plot(epochs, values, "o-", color=COLORS[dataset], linewidth=1.4, markersize=4)
        minima = np.flatnonzero(values == np.nanmin(values))
        ax.scatter(epochs[minima], values[minima], marker="*", s=125, c="#E69F00", edgecolors="black", zorder=3)
        ax.set_xlabel("Materialized FineMath epochs")
        ax.set_xlim(-0.4, epochs[-1] + 0.4)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        selected_epochs = ", ".join(f"{epochs[i]:.2f}" for i in minima)
        ax.text(0.04, 0.95, f"Observed minimum: {selected_epochs} epochs", transform=ax.transAxes, va="top", fontsize=8)
        ax.margins(y=0.20)
    fig.suptitle(f"FineMath matched proxy · {completed}/8 math evaluations verified", fontsize=12)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"math_likelihood.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=evaluation.DIRECTORY / "spec.json")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--run-identities", type=Path, default=evaluation.DIRECTORY / "fieldbook_runs.json")
    parser.add_argument("--output", type=Path, default=evaluation.DIRECTORY / "results")
    parser.add_argument("--include-uncheatable", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    spec = read_json(args.spec)
    evaluation.validate_spec(spec)
    points = archived_points(spec, args.archive)
    rows = collect(spec, points, run_identities(spec, args.run_identities))
    completed = sum(r["result"] is not None for r in rows)
    destination = args.output / spec["spec_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {dataset: observed_minimum(rows, dataset) for dataset in DATASETS}
    receipt = {
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "complete" if completed == len(rows) else "partial",
        "verified_points": completed,
        "expected_points": len(rows),
        "missing_percent": [r["percent"] for r in rows if r["result"] is None],
        "spec_sha256": spec["spec_sha256"],
        "source_objects": spec["sources"],
        "population_counts": spec["population_counts"],
        "output_root": evaluation.output_root(spec),
        "local_sources_sha256": {
            str(path): file_sha256(path)
            for path in (args.spec, args.archive / "preflight.json", args.archive / "curves.json", args.run_identities)
        },
        "analyzer_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "summaries": summaries,
    }
    write_csv(rows, spec, destination / "points.csv")
    receipt["figures"] = plot_results(rows, destination, args.include_uncheatable)
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination.resolve()), "verified_points": completed, "summaries": summaries}))
    if args.require_complete and completed != len(rows):
        raise SystemExit("The prescribed eight-checkpoint grid is incomplete; partial artifacts were retained")


if __name__ == "__main__":
    main()
