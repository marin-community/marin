# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot the existing Wikipedia proxy and target on Wikipedia-English BPB."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.domain_phase_mix import plot_tpp10_domain_sweeps as plots
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

DIRECTORY = Path(".agents/projects/starcoder_tpp10/domain_sweeps")
METRIC = "eval/uncheatable_eval/wikipedia_english/bpb"
LABELS = {"matched": "Epoch-matched proxy", "target": "Target"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DIRECTORY / "plots_20260912")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "wikipedia_eval_20260912")
    args = parser.parse_args()
    result = json.loads((args.snapshot / "corrected_results.json").read_text())
    source = json.loads((args.snapshot / "source.json").read_text())
    if canonical_sha256(result) != source["results_sha256"] or not source["all_three_checkpoint_audits_passed"]:
        raise ValueError("Expected the audited BPB result snapshot")
    preflight = json.loads((args.snapshot / "preflight.json").read_text())
    curves = [c for c in plots.build_curves(result, preflight) if c["domain"] == "wikipedia"]
    if {c["arm"] for c in curves} != set(LABELS) or len(curves) != 2 or any(c["missing_percent"] for c in curves):
        raise ValueError("Expected complete Wikipedia proxy and target curves")
    args.output.mkdir(parents=True, exist_ok=True)
    flat, minima = [], []
    colors = plt.get_cmap("RdYlGn_r")([0.15, 0.85])
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), sharex=True)
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.77, wspace=0.26)
        for ax, curve, color in zip(axes, curves, colors, strict=True):
            points = curve["points"]
            values = np.array([p["metrics"][METRIC] for p in points])
            if not np.all(np.isfinite(values)):
                raise ValueError("Nonfinite Wikipedia loss")
            ranked = sorted(points, key=lambda p: p["metrics"][METRIC])
            best, second = ranked[:2]
            minimum = {
                "arm": curve["arm"],
                "percent": best["percent"],
                "epochs": best["epochs"],
                "bpb": best["metrics"][METRIC],
                "second_percent": second["percent"],
                "second_epochs": second["epochs"],
                "second_gap_bpb": second["metrics"][METRIC] - best["metrics"][METRIC],
                "boundary_minimum": best["percent"] in (0, 100),
            }
            minima.append(minimum)
            flat.extend(
                {
                    "arm": curve["arm"],
                    "percent": p["percent"],
                    "epochs": p["epochs"],
                    "wikipedia_english_bpb": p["metrics"][METRIC],
                    "run_name": p["run_name"],
                }
                for p in points
            )
            ax.plot([p["epochs"] for p in points], values, "o-", color=color, linewidth=1.8, markersize=4.5)
            ax.scatter(
                best["epochs"], best["metrics"][METRIC], marker="*", s=190, c="#E69F00", edgecolor="#222222", zorder=3
            )
            ax.set_title(LABELS[curve["arm"]], fontsize=13, pad=28)
            ax.text(
                0.5,
                1.035,
                f"Observed minimum: {best['epochs']:.2f} epochs · {best['percent']}% Wikipedia",
                ha="center",
                transform=ax.transAxes,
                fontsize=9.5,
            )
            ax.set_xlabel("Materialized Wikipedia epochs")
            ax.set_ylabel("Wikipedia English BPB")
            ax.set_xlim(-0.4, 16.3)
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.margins(y=0.15)
            ax.grid(alpha=0.20)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Wikipedia training sweeps on the Wikipedia-English evaluation", fontsize=14, y=0.98)
        fig.text(
            0.5,
            0.045,
            "TPP ≈ 10 at both scales · one trainer seed · stars mark observed grid minima · separate y-axis ranges",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        for extension in ("png", "pdf"):
            fig.savefig(args.output / f"wikipedia_eval.{extension}", dpi=180)
        plt.close(fig)
    with (args.output / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    receipt = {
        "metric": METRIC,
        "evaluation_path": preflight["evaluation_paths"]["uncheatable_eval/wikipedia_english"],
        "evaluation_sequences": preflight["evaluation_populations"]["uncheatable_eval/wikipedia_english"],
        "snapshot_directory": str(args.snapshot.resolve()),
        "source": source,
        "preflight_sha256": file_sha256(args.snapshot / "preflight.json"),
        "plot_source_sha256": file_sha256(Path(__file__)),
        "minima": minima,
        "points": flat,
        "scope": "Existing completed evaluations only; no new training or evaluation jobs.",
    }
    plots.write_json(args.output / "receipt.json", receipt)
    print(json.dumps({"output": str(args.output.resolve()), "minima": minima}))


if __name__ == "__main__":
    main()
