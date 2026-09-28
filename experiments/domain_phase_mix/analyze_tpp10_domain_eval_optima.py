# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare observed proxy minima across every recorded BPB evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.domain_phase_mix import plot_tpp10_domain_sweeps as plots
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256

METRICS = {
    plots.MACRO: "Uncheatable mean",
    plots.repair.experiment.PRIMARY_METRIC: "PALOMA Programming Languages",
    **{f"eval/uncheatable_eval/{name}/bpb": label for name, label in plots.COMPONENT_NAMES.items()},
}
DISPLAY_METRICS = (
    plots.repair.experiment.PRIMARY_METRIC,
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
)


def proxy_minima(curves: list[dict]) -> list[dict]:
    """Retain all evaluation choices and loss penalties at the other domain's choice."""
    proxies = {curve["domain"]: curve for curve in curves if curve["arm"] == "matched"}
    if any(curve["missing_percent"] for curve in proxies.values()):
        raise ValueError("Both full proxy grids must be complete")
    rows = []
    for metric, label in METRICS.items():
        ordered = {name: sorted(curve["points"], key=lambda p: p["metrics"][metric]) for name, curve in proxies.items()}
        for name, points in ordered.items():
            best, second = points[:2]
            other_domain = next(domain for domain in ordered if domain != name)
            other_choice = ordered[other_domain][0]["percent"]
            other_point = next(point for point in points if point["percent"] == other_choice)
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "domain": name,
                    "best_percent": best["percent"],
                    "best_epochs": best["epochs"],
                    "best_bpb": best["metrics"][metric],
                    "second_percent": second["percent"],
                    "second_gap_bpb": second["metrics"][metric] - best["metrics"][metric],
                    "other_domain_choice_percent": other_choice,
                    "other_domain_choice_penalty_bpb": other_point["metrics"][metric] - best["metrics"][metric],
                }
            )
    return rows


def plot_differences(curves: list[dict], rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
    figure.subplots_adjust(left=0.085, right=0.98, bottom=0.155, top=0.81, hspace=0.42, wspace=0.27)
    for axis, metric in zip(axes.flat, DISPLAY_METRICS, strict=True):
        for curve in curves:
            if curve["arm"] != "matched":
                continue
            points = [p for p in curve["points"] if p["percent"] <= 70]
            name = curve["domain"]
            best = next(row for row in rows if row["metric"] == metric and row["domain"] == name)
            axis.plot(
                [p["epochs"] for p in points],
                [p["metrics"][metric] for p in points],
                "o-",
                color=plots.COLORS[name],
                label=plots.DOMAINS[name],
                linewidth=1.7,
                markersize=4,
            )
            axis.plot(best["best_epochs"], best["best_bpb"], "*", color=plots.COLORS[name], markersize=13, zorder=5)
        axis.set_title(METRICS[metric], fontsize=12)
        axis.set_xlabel("Materialized epochs of the varied domain")
        axis.set_ylabel("Evaluation BPB")
        axis.set_xlim(-0.3, 11.5)
        axis.set_xticks([0, 2, 4, 6, 8, 10])
        axis.margins(y=0.12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.17)
    figure.suptitle("Some shared evaluations select different proxy mixtures", y=0.97, fontsize=15, weight="bold")
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2)
    figure.text(
        0.085,
        0.025,
        "Stars: observed full-grid minima. Detail through 70% domain weight; 100% points remain in the data.\n"
        "Exploratory comparison, one trainer seed. Full table includes all eight evaluations and the Uncheatable mean.",
        fontsize=9,
        color="#46515C",
    )
    figure.savefig(output / "proxy_eval_differences.png", dpi=180)
    figure.savefig(output / "proxy_eval_differences.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.snapshot / "corrected_results.json").read_text())
    source = json.loads((args.snapshot / "source.json").read_text())
    if source["results_sha256"] != canonical_sha256(result) or not source["all_three_checkpoint_audits_passed"]:
        raise ValueError("Results do not match the verified BPB snapshot")
    preflight = json.loads((args.snapshot / "preflight.json").read_text())
    curves = plots.build_curves(result, preflight)
    rows = proxy_minima(curves)
    args.output.mkdir(parents=True, exist_ok=True)
    plots.write_json(args.output / "source.json", {**source, "snapshot_directory": str(args.snapshot.resolve())})
    plots.write_json(args.output / "proxy_eval_optima.json", {"rows": rows})
    with (args.output / "proxy_eval_optima.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        plot_differences(curves, rows, args.output)
    for metric, label in METRICS.items():
        selected = [row for row in rows if row["metric"] == metric]
        print(label, [(row["domain"], row["best_percent"], row["best_epochs"]) for row in selected])


if __name__ == "__main__":
    main()
