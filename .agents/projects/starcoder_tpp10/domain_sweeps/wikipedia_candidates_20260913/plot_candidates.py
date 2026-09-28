# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2"]
# ///
"""Plot all saved Wikipedia evaluation candidates from corrected local receipts."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

DIRECTORY = Path(__file__).resolve().parent
BASE = DIRECTORY.parent
NATIVE = BASE / "native_closeout_20260912/plots/curves.json"
CORRECTED = BASE / "native_closeout_20260912/plots/corrected_results.json"
MATH = (
    BASE
    / "target_math_complete_20260912/results"
    / "c788d33ddd3b357a7f80de6ed1d5ecb632104e4beabd5cc0934d580ecac77d39/receipt.json"
)
GRID = [0, 5, 10, 20, 30, 50, 70, 100]
STYLES = {"matched": ("Simulated-epoching proxy", "#0072B2", "o", "-"), "target": ("Target", "#D55E00", "s", "--")}
# Ordered by subject matter, not by shape or minimum location.
CANDIDATES = [
    ("wikipedia_english", "Wikipedia English", "Same-domain evaluation", "bpb"),
    ("bbc_news", "BBC News", "Related English prose; news", "bpb"),
    ("ao3_english", "AO3 English", "Related English prose; fiction", "bpb"),
    ("macro", "Uncheatable aggregate", "Equal mean of all seven components", "bpb"),
    ("arxiv_physics", "arXiv Physics", "Scientific prose", "bpb"),
    ("arxiv_computer_science", "arXiv Computer Science", "Scientific prose", "bpb"),
    ("github_python", "GitHub Python", "Code; cross-domain diagnostic", "bpb"),
    ("github_cpp", "GitHub C++", "Code; cross-domain diagnostic", "bpb"),
    ("paloma", "PALOMA Programming Languages", "Code; cross-domain diagnostic", "bpb"),
    ("math500", "MATH-500", "Reference-solution likelihood", "ppl"),
    ("gsm8k", "GSM8K", "Reference-solution likelihood", "ppl"),
]


def load_candidates():
    native = json.loads(NATIVE.read_text())
    corrected = json.loads(CORRECTED.read_text())
    math = json.loads(MATH.read_text())
    assert corrected["complete"] and math["all_target_grids_complete"]
    curves = {c["arm"]: c["points"] for c in native["curves"] if c["domain"] == "wikipedia"}
    records, summaries = [], []
    for key, label, scope, unit in CANDIDATES:
        for arm in STYLES:
            if unit == "ppl":
                points = math["rows"][f"wikipedia/{arm}"]
                values = [p["result"]["perplexity"][key] for p in points]
                metric = f"reference_solution_perplexity/{key}"
            else:
                points = curves[arm]
                metric = (
                    "eval/uncheatable_eval/macro_bpb"
                    if key == "macro"
                    else (
                        "eval/paloma/dolma_100_programing_languages-tpp10/bpb"
                        if key == "paloma"
                        else f"eval/uncheatable_eval/{key}/bpb"
                    )
                )
                values = [p["metrics"][metric] for p in points]
            assert [p["percent"] for p in points] == GRID
            assert all(np.isfinite(v) and v > 0 for v in values)
            best = int(np.argmin(values))
            minimum = values[best]
            summary = {
                "candidate": key,
                "benchmark": label,
                "scope": scope,
                "unit": unit,
                "arm": arm,
                "minimum_percent": points[best]["percent"],
                "minimum_epochs": points[best]["epochs"],
                "minimum_value": minimum,
                "gain_from_zero_percent": 100 * (values[0] - minimum) / values[0],
                "excess_at_p100_percent": 100 * (values[-1] / minimum - 1),
                "second_best_gap_percent": 100 * (sorted(values)[1] / minimum - 1),
                "boundary_minimum": best in (0, len(values) - 1),
            }
            summaries.append(summary)
            for p, value in zip(points, values, strict=True):
                records.append(
                    {
                        "candidate": key,
                        "benchmark": label,
                        "arm": arm,
                        "percent": p["percent"],
                        "epochs": p["epochs"],
                        "metric": metric,
                        "unit": unit,
                        "value": value,
                        "excess_percent": 100 * (value / minimum - 1),
                        "run_name": p["run_name"],
                    }
                )
    return records, summaries


def decorate(ax):
    ax.set_xlim(-0.25, 16.2)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Materialized Wikipedia epochs")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def draw(ax, records, key, arm, excess=False):
    points = [r for r in records if r["candidate"] == key and r["arm"] == arm]
    label, color, marker, linestyle = STYLES[arm]
    x = np.array([r["epochs"] for r in points])
    y = np.array([r["excess_percent"] if excess else r["value"] for r in points])
    ax.plot(
        x,
        y,
        linestyle=linestyle,
        marker=marker,
        color=color,
        lw=1.9,
        ms=4.5,
        mfc=color if arm == "matched" else "white",
        label=label,
    )
    best = int(np.argmin(y))
    ax.scatter(x[best], y[best], marker="*", s=145, c=color, edgecolor="black", lw=0.6, zorder=5)
    return x, y


def legend_handles():
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=line,
            lw=1.8,
            markerfacecolor=color if arm == "matched" else "white",
            label=label,
        )
        for arm, (label, color, marker, line) in STYLES.items()
    ]


def draw_excess(ax, records, key, cap):
    for arm in STYLES:
        draw(ax, records, key, arm, excess=True)
    ax.set_ylim(-cap * 0.08, cap)
    ax.set_ylabel("Excess over own minimum (%)")
    ax.axhline(0, color="#777777", lw=0.6, zorder=0)
    decorate(ax)


def main():
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    records, summaries = load_candidates()
    for filename, rows in [("points.csv", records), ("minima.csv", summaries)]:
        with (DIRECTORY / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pdf = DIRECTORY / "wikipedia_benchmark_candidates.pdf"
    with PdfPages(pdf) as pages:
        for page_index, start in enumerate(range(0, len(CANDIDATES), 2), 1):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
            fig.subplots_adjust(left=0.065, right=0.985, bottom=0.15, top=0.82, hspace=0.55, wspace=0.31)
            fig.suptitle(
                f"Wikipedia training sweeps: all saved benchmark candidates ({page_index}/6)", y=0.975, fontsize=15
            )
            fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
            for row_index, candidate in enumerate(CANDIDATES[start : start + 2]):
                key, label, scope, unit = candidate
                fig.text(0.066, 0.862 - row_index * 0.391, f"{label}  |  {scope}", fontsize=12, weight="bold")
                for col, arm in enumerate(STYLES):
                    ax = axes[row_index, col]
                    draw(ax, records, key, arm)
                    decorate(ax)
                    s = next(s for s in summaries if s["candidate"] == key and s["arm"] == arm)
                    ax.set_title(f"{STYLES[arm][0]}: minimum {s['minimum_epochs']:.2f} epochs", fontsize=10.5)
                    ax.set_ylabel("BPB" if unit == "bpb" else "Reference-solution perplexity")
                    ax.margins(y=0.13)
                ax = axes[row_index, 2]
                draw_excess(ax, records, key, cap=5)
                ax.set_title("Near the minimum (upper limit 5%)")
            if len(CANDIDATES[start : start + 2]) == 1:
                for ax in axes[1]:
                    ax.set_visible(False)
            fig.text(
                0.5,
                0.064,
                "Left / middle: absolute values, full range. Right: 100 \u00d7 (loss / that curve's minimum - 1); "
                "values above 5% are clipped.",
                ha="center",
                fontsize=10,
            )
            fig.text(
                0.5,
                0.035,
                "Every measured point is retained. Lines join observations; stars mark grid minima. "
                "One trainer seed and one matched subset; no smoothing.",
                ha="center",
                fontsize=10,
                color="#555555",
            )
            pages.savefig(fig)
            fig.savefig(DIRECTORY / f"gallery_{page_index}.png", dpi=130)
            plt.close(fig)
    # Compact language-only contact sheet for the interactive discussion.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.82, hspace=0.39, wspace=0.26)
    fig.suptitle("Wikipedia training: language-evaluation candidates", y=0.98, fontsize=15)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    for ax, key in zip(axes.flat, ["wikipedia_english", "bbc_news", "ao3_english", "macro"], strict=True):
        draw_excess(ax, records, key, cap=5)
        candidate = next(c for c in CANDIDATES if c[0] == key)
        minima = [s for s in summaries if s["candidate"] == key]
        ax.set_title(
            candidate[1]
            + f"\nMinima: proxy {minima[0]['minimum_epochs']:.2f} / target {minima[1]['minimum_epochs']:.2f} epochs",
            fontsize=11,
        )
    fig.text(
        0.5,
        0.047,
        "100 \u00d7 (loss / each curve's minimum - 1); values above 5% clipped. Stars: measured grid minima.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.02,
        "Measured points, no smoothing. Absolute curves and all 11 benchmark views are in the PDF gallery.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.savefig(DIRECTORY / "language_candidates.png", dpi=150)
    plt.close(fig)
    provenance = {
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [NATIVE, CORRECTED, MATH, Path(__file__)]
        },
        "benchmark_views": len(CANDIDATES),
        "curve_count": len(summaries),
        "points": len(records),
        "transformation": "100 * (value / min(value for same benchmark and scale) - 1)",
        "source_units": ["corrected BPB", "reference-solution perplexity"],
        "raw_view": "full linear y range",
        "excess_view_ymax_percent": 5,
        "new_evaluations": False,
        "smoothing": False,
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [pdf, DIRECTORY / "points.csv", DIRECTORY / "minima.csv", DIRECTORY / "language_candidates.png"]
        },
    }
    (DIRECTORY / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "curves": len(summaries), "points": len(records)}, indent=2))


if __name__ == "__main__":
    main()
