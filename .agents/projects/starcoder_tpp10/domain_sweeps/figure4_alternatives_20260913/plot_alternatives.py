# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2"]
# ///
"""Render absolute-BPB Figure 4 alternatives from verified local receipts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DIRECTORY = Path(__file__).resolve().parent
BASE = DIRECTORY.parent
REPO = next(p for p in DIRECTORY.parents if (p / "pyproject.toml").exists())
NATIVE = BASE / "native_closeout_20260912/plots/curves.json"
MATH = (
    BASE
    / "target_math_complete_20260912/results"
    / "c788d33ddd3b357a7f80de6ed1d5ecb632104e4beabd5cc0934d580ecac77d39/receipt.json"
)
DESIGN = REPO / "experiments/domain_phase_mix/starcoder_tpp10_assets/design.json"
COLORS = {"wikipedia": "#0072B2", "finemath_3plus": "#D55E00"}
ARMS = {"matched": ("-", "o"), "target": ("--", "s")}
Y_LIMITS = (0.70, 2.02)


def load_curves():
    native = json.loads(NATIVE.read_text())
    math = json.loads(MATH.read_text())
    assert math["all_target_grids_complete"]
    curves = {}
    for arm in ARMS:
        source = next(c for c in native["curves"] if c["domain"] == "wikipedia" and c["arm"] == arm)
        for benchmark in ["ao3_english", "wikipedia_english"]:
            curves["wikipedia", arm, benchmark] = [
                {
                    "epochs": p["epochs"],
                    "percent": p["percent"],
                    "bpb": p["metrics"][f"eval/uncheatable_eval/{benchmark}/bpb"],
                    "run_name": p["run_name"],
                }
                for p in source["points"]
            ]
        for domain in COLORS:
            for benchmark in ["math500", "gsm8k"]:
                rows = math["rows"][f"{domain}/{arm}"]
                points = []
                for row in rows:
                    metrics = row["result"]["metrics"]
                    assert metrics["eval/bpb_schema_version"] == 2
                    points.append(
                        {
                            "epochs": row["epochs"],
                            "percent": row["percent"],
                            "bpb": metrics[f"eval/{benchmark}/bpb"],
                            "run_name": row["run_name"],
                        }
                    )
                # Changing from perplexity to BPB preserves the selected point.
                assert np.argmin([p["bpb"] for p in points]) == np.argmin(
                    [r["result"]["perplexity"][benchmark] for r in rows]
                )
                curves[domain, arm, benchmark] = points
    for points in curves.values():
        assert [p["percent"] for p in points] == [0, 5, 10, 20, 30, 50, 70, 100]
        assert all(np.isfinite(p["bpb"]) and p["bpb"] > 0 for p in points)
    return curves


def draw_curve(ax, points, domain, arm, benchmark, fit=None):
    color = COLORS[domain]
    line, marker = ARMS[arm]
    x = np.array([p["epochs"] for p in points])
    y = np.array([p["bpb"] for p in points])
    ax.plot(
        x,
        y,
        color=color,
        ls=line if fit is None else "none",
        marker=marker,
        lw=1.65,
        ms=4,
        mfc=color if arm == "matched" else "white",
        mew=0.9,
        zorder=2,
    )
    minimum = int(np.argmin(y))
    minimum_x, minimum_y = x[minimum], y[minimum]
    if fit is not None:
        np.testing.assert_allclose(fit["observed_epochs"], x)
        np.testing.assert_allclose(fit["observed"], y)
        ax.plot(fit["dense_epochs"], fit["dense_prediction"], color=color, ls=line, lw=1.65, zorder=1)
        minimum_x, minimum_y = fit["predicted_minimum_epochs"], fit["predicted_minimum_bpb"]
    ax.scatter(
        minimum_x,
        minimum_y,
        marker="*",
        s=135,
        color=color,
        edgecolor="#222222",
        lw=0.65,
        zorder=4,
    )
    text = f"{minimum_x:.1f} ep."
    if minimum == len(y) - 1:
        offset, align = (-8, 10), "right"
    elif domain == "wikipedia" and benchmark == "ao3_english":
        offset, align = ((0, -19), "center") if arm == "matched" else ((6, 11), "left")
    elif domain == "finemath_3plus" and arm == "matched":
        offset, align = (-6, -19), "center"
    elif domain == "wikipedia" and benchmark == "wikipedia_english" and arm == "target":
        offset, align = (-18, -18), "center"
    elif domain == "wikipedia":
        offset, align = (4, 12 if arm == "matched" else -21), "left"
    else:
        offset, align = (0, 12 if arm == "matched" else -21), "center"
    ax.annotate(
        text,
        (minimum_x, minimum_y),
        xytext=offset,
        textcoords="offset points",
        ha=align,
        fontsize=8.7,
        color=color,
        zorder=6,
    )
    clipped = [{"domain": domain, "arm": arm, **p} for p in points if not Y_LIMITS[0] <= p["bpb"] <= Y_LIMITS[1]]
    return clipped


def make_figure(curves, left_math, left_wikipedia, design, fits=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.05), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.69, bottom=0.15, wspace=0.20)
    handles = [
        Line2D([], [], color=color, lw=2, label=label)
        for (domain, color), label in zip(COLORS.items(), ["Wikipedia", "FineMath-3+"], strict=True)
    ]
    handles.append(
        Line2D(
            [],
            [],
            ls="none",
            marker="*",
            ms=10,
            color="#555555",
            label="MARINER fitted minimum" if fits is not None else "Observed minimum",
        )
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=10.5)
    proxy_flops = design["models"]["unmatched"]["training_flops"]
    target_flops = design["models"]["target"]["training_flops"]
    scale_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            lw=1.65,
            marker="o",
            ms=4,
            label=rf"Simulated-epoching proxy ($ {proxy_flops / 1e16:.2f} \times 10^{{16}}$ FLOPs)",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            lw=1.65,
            marker="s",
            ms=4,
            mfc="white",
            ls="--",
            label=rf"Target ($ {target_flops / 1e18:.2f} \times 10^{{18}}$ FLOPs)",
        ),
    ]
    fig.legend(
        handles=scale_handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False, fontsize=10
    )
    left_label = "GSM8K" if left_math == "gsm8k" else "MATH-500"
    wikipedia_label = "AO3 English" if left_wikipedia == "ao3_english" else "Wikipedia English"
    axes[0].set_title("Different benchmarks", pad=29, fontsize=12, weight="bold")
    axes[0].text(
        0.5,
        1.06,
        f"Wikipedia: {wikipedia_label}  |  FineMath: {left_label}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9.8,
    )
    axes[1].set_title("Same benchmark: MATH-500", pad=29, fontsize=12, weight="bold")
    clipped = []
    exported = []
    for panel, ax in enumerate(axes):
        for domain in COLORS:
            benchmark = "math500" if panel == 1 else left_wikipedia if domain == "wikipedia" else left_math
            for arm in ARMS:
                points = curves[domain, arm, benchmark]
                fit = None if fits is None else fits[domain, arm, benchmark]
                clipped.extend(
                    {"panel": panel, "benchmark": benchmark, **p}
                    for p in draw_curve(ax, points, domain, arm, benchmark, fit)
                )
                exported.extend(
                    {"panel": panel, "domain": domain, "benchmark": benchmark, "arm": arm, **p} for p in points
                )
        ax.set_xlim(-0.2, 16.4)
        ax.set_ylim(*Y_LIMITS)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_yticks(np.arange(0.8, 2.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Materialized epochs of varied training domain", fontsize=10)
    axes[0].set_ylabel("Evaluation loss (bits per byte)", fontsize=10.5)
    return fig, exported, clipped


def compact_paper_layout(fig):
    """Fit two narrow panels beside the manuscript caption."""
    fig.set_size_inches(5.5, 3.0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.62, bottom=0.19, wspace=0.16)
    for panel, ax in enumerate(fig.axes):
        ax.set_xlabel("Materialized domain epochs", fontsize=8.5)
        ax.set_title(ax.get_title(), fontsize=9, pad=27, weight="bold")
        ax.tick_params(labelsize=8)
        ax.set_ylim(0.68, Y_LIMITS[1])
        for line in ax.lines:
            line.set_linewidth(1.3)
            line.set_markersize(3.0)
            line.set_markeredgewidth(0.7)
        for collection in ax.collections:
            collection.set_sizes([85])
            collection.set_linewidth(0.55)
        for annotation in ax.texts:
            if annotation.get_text().startswith("Wikipedia:"):
                annotation.set_text(annotation.get_text().replace("  |  ", "\n"))
                annotation.set_fontsize(7.8)
            elif annotation.get_text().endswith(" ep."):
                annotation.set_fontsize(8)
                _, minimum_loss = annotation.xy
                if minimum_loss < 0.85:
                    annotation.set_position((-4, 7))
                elif minimum_loss < 1.1 and panel == 0:
                    annotation.set_position((-24, -11))
                elif minimum_loss < 1.3:
                    annotation.set_position((-3, -11))
                else:
                    annotation.set_position((4, 7))
    fig.axes[0].set_ylabel("Evaluation loss (BPB)", fontsize=8.5)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferred", action="store_true", help="Render only the preferred observed and fitted pair.")
    parser.add_argument("--paper-directory", type=Path, help="Export the compact fitted figure into this directory.")
    args = parser.parse_args()
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.unicode_minus": False})
    curves = load_curves()
    design = json.loads(DESIGN.read_text())
    if args.paper_directory is not None:
        results = json.loads((DIRECTORY / "mariner_fits/summary.json").read_text())["curves"]
        fits = {(r["domain"], r["arm"], r["benchmark"]): r for r in results}
        fig, rows, clipped = make_figure(curves, "math500", "wikipedia_english", design, fits)
        compact_paper_layout(fig)
        output = args.paper_directory.resolve()
        output.mkdir(parents=True, exist_ok=True)
        stem = "figure4_left_wikipedia_english_math500_mariner"
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"{stem}.{extension}", dpi=200)
        plt.close(fig)
        receipt = {
            "layout": {
                "size_inches": [5.5, 3.0],
                "panel_gap": 0.16,
                "y_limits": [0.68, Y_LIMITS[1]],
                "paper_width_fraction": 0.64,
                "caption_width_fraction": 0.33,
                "printed_axis_width_ratio_to_previous": (0.87 / 2.16 * 0.64) / (0.915 / 2.075),
            },
            "rows": rows,
            "clipped": clipped,
            "fits_sha256": hashlib.sha256((DIRECTORY / "mariner_fits/summary.json").read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "output_sha256": {
                extension: hashlib.sha256((output / f"{stem}.{extension}").read_bytes()).hexdigest()
                for extension in ["pdf", "png"]
            },
        }
        (output / f"{stem}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"output": str(output), "points": len(rows), "fits": len(fits)}))
        return
    manifests = {}
    versions = [
        ("gsm8k", "gsm8k", "ao3_english"),
        ("math500", "math500", "ao3_english"),
        ("wikipedia_english_math500", "math500", "wikipedia_english"),
    ]
    if args.preferred:
        versions = [versions[-1]]
    for version, left_math, left_wikipedia in versions:
        fig, rows, clipped = make_figure(curves, left_math, left_wikipedia, design)
        stem = f"figure4_left_{version}"
        fig.savefig(DIRECTORY / f"{stem}.pdf")
        fig.savefig(DIRECTORY / f"{stem}.png", dpi=180)
        plt.close(fig)
        with (DIRECTORY / f"{stem}_points.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifests[version] = {"pdf": stem + ".pdf", "png": stem + ".png", "points": len(rows), "clipped": clipped}
    if args.preferred:
        results = json.loads((DIRECTORY / "mariner_fits/summary.json").read_text())["curves"]
        fits = {(r["domain"], r["arm"], r["benchmark"]): r for r in results}
        fig, rows, clipped = make_figure(curves, "math500", "wikipedia_english", design, fits)
        stem = "figure4_left_wikipedia_english_math500_mariner"
        fig.savefig(DIRECTORY / f"{stem}.pdf")
        fig.savefig(DIRECTORY / f"{stem}.png", dpi=180)
        plt.close(fig)
        manifests["wikipedia_english_math500_mariner"] = {
            "pdf": stem + ".pdf",
            "png": stem + ".png",
            "points": len(rows),
            "clipped": clipped,
            "fit_results": "mariner_fits/summary.json",
            "minima": [
                {k: r[k] for k in ["domain", "arm", "benchmark", "predicted_minimum_epochs", "predicted_minimum_bpb"]}
                for r in results
            ],
        }
    receipt = {
        "metric": "ratio of total scored loss bits to total scored bytes",
        "math_scoring": "Reference-solution tokens only; native BPB schema v2, already present in result receipts",
        "axes": {"scale": "linear", "shared_y_limits": Y_LIMITS},
        "no_normalization": True,
        "lines": "MARINER fits for the fitted companion; measured point joins otherwise",
        "compute": design["models"],
        "versions": manifests,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [NATIVE, MATH, DESIGN, Path(__file__)] + (
                [DIRECTORY / "mariner_fits/summary.json"] if args.preferred else []
            )
        },
        "output_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in DIRECTORY.iterdir()
            if p.suffix in [".pdf", ".png", ".csv"]
        },
    }
    receipt_name = "preferred_provenance.json" if args.preferred else "provenance.json"
    (DIRECTORY / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
