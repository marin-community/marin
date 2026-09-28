# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot completed domain-survey endpoints from the audited BPB repair."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.domain_phase_mix import repair_tpp10_evaluation as repair
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256

DOMAINS = {"wikipedia": "Wikipedia", "finemath_3plus": "FineMath-3+"}
COLORS = {"wikipedia": "#0072B2", "finemath_3plus": "#D55E00"}
MACRO = repair.original_eval.METRIC
COMPONENT_NAMES = {
    "wikipedia_english": "Wikipedia English",
    "github_python": "GitHub Python",
    "github_cpp": "GitHub C++",
    "bbc_news": "BBC News",
    "arxiv_physics": "arXiv physics",
    "arxiv_computer_science": "arXiv computer science",
    "ao3_english": "AO3 English",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refresh(plan: dict, output: Path) -> None:
    """Archive a consistent, read-only snapshot and its numerical audit receipts."""
    root = repair.root(plan)
    result = repair.read_json(root + "/corrected_results.json")
    if result["repair_sha256"] != plan["repair_sha256"]:
        raise ValueError("Corrected results belong to a different repair")
    for item in plan["audits"]:
        name = item["request"]["run_name"]
        audit = repair.read_json(root + "/checkpoint_audits/" + name + ".json")
        if not (
            audit["verified"]
            and audit["repair_sha256"] == plan["repair_sha256"]
            and audit["counts_sha256"] == canonical_sha256(result["counts"])
            and audit["request"] == item["request"]
            and audit["metadata"] == item["checkpoint"]["metadata"]
        ):
            raise ValueError(f"Unverified numerical audit: {name}")
        write_json(output / f"{name}_audit.json", audit)
    survey = plan["artifacts"]["plan"]
    preflight = repair.read_json(
        f"{survey['marin_prefix']}/experiments/tpp10_domain_sweeps/{survey['plan_sha256']}/preflight.json"
    )
    write_json(output / "corrected_results.json", result)
    write_json(output / "preflight.json", preflight)
    write_json(
        output / "source.json",
        {
            "checked_at": datetime.now(UTC).isoformat(),
            "repair_sha256": plan["repair_sha256"],
            "results_sha256": canonical_sha256(result),
            "source": root + "/corrected_results.json",
            "all_three_checkpoint_audits_passed": True,
        },
    )


def build_curves(result: dict, preflight: dict) -> list[dict]:
    """Include shared controls and completed endpoints without filling missing points."""
    allocations = {row["percent"]: row for row in preflight["allocation"]["coordinates"]}
    curves = []
    for arm in ("matched", "target"):
        for domain in DOMAINS:
            rows = [row for row in result["rows"] if row["request"]["domain"] == domain and row["request"]["arm"] == arm]
            points = [{"percent": 0, "epochs": 0.0, "metrics": result["controls"][arm], "run_name": "shared p0"}]
            for row in sorted(rows, key=lambda row: row["request"]["percent"]):
                request = row["request"]
                points.append(
                    {
                        "percent": request["percent"],
                        "epochs": allocations[request["percent"]][f"{arm}_epochs"],
                        "metrics": row["metrics"],
                        "run_name": request["run_name"],
                    }
                )
            if len({p["percent"] for p in points}) != len(points):
                raise ValueError(f"Duplicate coordinate: {domain}/{arm}")
            if not all(np.isfinite(p["metrics"][MACRO]) for p in points):
                raise ValueError("Nonfinite Uncheatable metric")
            best = min(points, key=lambda p: (p["metrics"][MACRO], p["percent"]))
            missing = [p for p in (0, 5, 10, 20, 30, 50, 70, 100) if p not in {q["percent"] for q in points}]
            curves.append({"domain": domain, "arm": arm, "points": points, "best": best, "missing_percent": missing})
    return curves


def summarize(curves: list[dict]) -> dict:
    minima = []
    for curve in curves:
        best = curve["best"]
        minima.append(
            {
                "domain": curve["domain"],
                "arm": curve["arm"],
                "minimum_percent": best["percent"],
                "minimum_epochs": best["epochs"],
                "minimum_bpb": best["metrics"][MACRO],
                "missing_percent": curve["missing_percent"],
                "scope": "lowest completed point" if curve["missing_percent"] else "observed full-grid minimum",
                "delta_bpb_from_minimum": {
                    str(p["percent"]): p["metrics"][MACRO] - best["metrics"][MACRO] for p in curve["points"]
                },
            }
        )
    return {"metric": MACRO, "minima": minima, "scope": "One trainer seed; observed grid values, without fitted curves."}


def format_axes(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_xlabel("Materialized epochs of the varied domain")
    axis.set_ylabel("Uncheatable BPB")


def overview(curves: list[dict], result: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    figure.subplots_adjust(left=0.085, right=0.98, bottom=0.15, top=0.84, hspace=0.43, wspace=0.25)
    titles = {"matched": "Epoch-matched proxy · 16.6M parameters", "target": "Target · 301.2M parameters"}
    for column, arm in enumerate(("matched", "target")):
        selected = [curve for curve in curves if curve["arm"] == arm]
        for row, max_percent in enumerate((100, 70)):
            axis = axes[row, column]
            format_axes(axis)
            axis.set_title(titles[arm] if row == 0 else "Valley detail · through 70% domain weight", fontsize=12, pad=12)
            for curve in selected:
                points = [p for p in curve["points"] if p["percent"] <= max_percent]
                color = COLORS[curve["domain"]]
                axis.plot(
                    [p["epochs"] for p in points],
                    [p["metrics"][MACRO] for p in points],
                    "o-",
                    color=color,
                    markersize=4.4,
                    linewidth=1.7,
                    label=DOMAINS[curve["domain"]],
                )
                best = curve["best"]
                if best["percent"] <= max_percent:
                    axis.plot(
                        best["epochs"],
                        best["metrics"][MACRO],
                        marker="*",
                        markersize=13,
                        markerfacecolor="white" if curve["missing_percent"] else color,
                        markeredgecolor=color,
                        markeredgewidth=1.2,
                        zorder=5,
                    )
                    if row == 1 and not curve["missing_percent"]:
                        axis.annotate(
                            f"{best['percent']}% · {best['epochs']:.2f} epochs",
                            xy=(best["epochs"], best["metrics"][MACRO]),
                            xytext=(0, 16),
                            textcoords="offset points",
                            ha="center",
                            fontsize=9,
                            color=color,
                        )
            axis.set_xlim(-0.2, 16.3 if row == 0 else 11.6)
            axis.set_xticks([0, 4, 8, 12, 16] if row == 0 else [0, 2, 4, 6, 8, 10])
            axis.margins(y=0.13)
            if row == 0:
                axis.legend(frameon=False, loc="upper left", fontsize=10)
            if arm == "target" and any(curve["missing_percent"] for curve in selected):
                pending = next(curve for curve in selected if curve["missing_percent"])
                text = (
                    f"{DOMAINS[pending['domain']]} target incomplete\n"
                    + ", ".join(f"{p}%" for p in pending["missing_percent"])
                    + " still running"
                )
                axis.text(
                    0.98 if row else 0.35,
                    0.97 if row else 0.94,
                    text,
                    transform=axis.transAxes,
                    ha="right" if row else "left",
                    va="top",
                    fontsize=9,
                    color=COLORS[pending["domain"]],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
                )
    figure.suptitle("Completed Wikipedia and FineMath sweeps", y=0.965, fontsize=17, weight="bold")
    figure.text(
        0.5,
        0.915,
        f"{len(result['rows'])}/{result['expected_rows']} training points + two shared controls"
        " · TPP 10 · same seven-component objective",
        ha="center",
        fontsize=10.5,
        color="#46515C",
    )
    figure.text(
        0.085,
        0.03,
        "Stars: lowest observed loss. "
        "One seed; segments join measurements, with no fitted curve.\n"
        "Corrected BPB: total scored loss bits / total scored bytes. All three saved-checkpoint audits passed.",
        fontsize=9,
        color="#46515C",
    )
    figure.savefig(output / "completed_sweeps.png", dpi=180)
    figure.savefig(output / "completed_sweeps.pdf")
    plt.close(figure)


def components(curves: list[dict], output: Path) -> None:
    metrics = {MACRO: "Uncheatable mean", **{f"eval/uncheatable_eval/{k}/bpb": v for k, v in COMPONENT_NAMES.items()}}
    figure, axes = plt.subplots(2, 4, figsize=(14, 7.1))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.84, hspace=0.40, wspace=0.30)
    for axis, (metric, title) in zip(axes.flat, metrics.items(), strict=True):
        for curve in sorted(curves, key=lambda c: c["arm"], reverse=True):
            points = [p for p in curve["points"] if p["percent"] <= 70]
            baseline = points[0]["metrics"][metric]
            axis.plot(
                [p["epochs"] for p in points],
                [p["metrics"][metric] - baseline for p in points],
                linestyle="--" if curve["arm"] == "target" else "-",
                marker="s" if curve["arm"] == "target" else "o",
                markersize=3.3,
                color=COLORS[curve["domain"]],
                linewidth=1.5,
            )
        axis.axhline(0, color="#666666", linewidth=0.7)
        axis.set_title(title, fontsize=11, pad=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
        axis.set_xlim(-0.2, 11.5)
        axis.set_xticks([0, 4, 8])
        axis.set_xlabel("Materialized epochs", fontsize=9)
        axis.tick_params(labelsize=9)
    figure.supylabel("BPB change from shared web-only control", x=0.015, fontsize=11)
    handles = [Line2D([0], [0], color=COLORS[d], linewidth=2, label=DOMAINS[d]) for d in DOMAINS]
    handles += [
        Line2D([0], [0], color="#444444", marker="o", label="Matched proxy"),
        Line2D([0], [0], color="#444444", marker="s", linestyle="--", label="Target"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False)
    figure.suptitle("Component trade-offs behind the aggregate minimum", y=0.97, fontsize=16, weight="bold")
    figure.text(
        0.5, 0.91, "Negative values improve on the corresponding scale's web-only control", ha="center", fontsize=11
    )
    figure.text(
        0.065,
        0.025,
        "Detail through 70% domain weight; the overview includes the 100% endpoints.",
        fontsize=9,
        color="#46515C",
    )
    figure.savefig(output / "component_tradeoffs.png", dpi=170)
    figure.savefig(output / "component_tradeoffs.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.repair_plan.read_text())
    if args.refresh:
        refresh(plan, args.output)
    result = json.loads((args.output / "corrected_results.json").read_text())
    source = json.loads((args.output / "source.json").read_text())
    if source["results_sha256"] != canonical_sha256(result) or source["repair_sha256"] != plan["repair_sha256"]:
        raise ValueError("Cached results do not match the audited snapshot")
    preflight = json.loads((args.output / "preflight.json").read_text())
    curves = build_curves(result, preflight)
    write_json(args.output / "curves.json", {"curves": curves})
    write_json(args.output / "summary.json", summarize(curves))
    with (args.output / "completed_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "arm", "percent", "epochs", "uncheatable_bpb", "run_name"])
        writer.writeheader()
        for curve in curves:
            for point in curve["points"]:
                writer.writerow(
                    {
                        "domain": curve["domain"],
                        "arm": curve["arm"],
                        "percent": point["percent"],
                        "epochs": point["epochs"],
                        "uncheatable_bpb": point["metrics"][MACRO],
                        "run_name": point["run_name"],
                    }
                )
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        overview(curves, result, args.output)
        components(curves, args.output)
    print(json.dumps(summarize(curves), indent=2))


if __name__ == "__main__":
    main()
