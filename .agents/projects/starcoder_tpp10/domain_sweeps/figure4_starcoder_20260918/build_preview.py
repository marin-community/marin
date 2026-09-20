# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Preview of Figure 4 with StarCoder added to the different-benchmarks panel.

StarCoder reuses Figure 3's twelve-point TPP-10 curves on the Paloma programming-languages
evaluation: the single-seed target and the matched proxy averaged over three subsets and two
trainer seeds. The MATH-500 panel is unchanged because StarCoder was never scored on MATH-500.

Run from the repository root: PYTHONPATH=. uv run --offline --no-sync python <this file>
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DIRECTORY = Path(__file__).resolve().parent
ALTERNATIVES = DIRECTORY.parent / "figure4_alternatives_20260913"
sys.path.insert(0, str(ALTERNATIVES))

import plot_alternatives as base  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_starcoder_tpp10_mariner_20260911 as fitting,
)

PAPER = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin"
    "/data_mixing_paper_one_phase"
)
ANALYSIS = PAPER / "revision_notes/20260912_outline_figures/data/epoch_matching_analysis.json"
ALLOCATION = PAPER / "revision_notes/20260912_outline_figures/data/allocation_audit.json"
DOMAIN_FITS = ALTERNATIVES / "mariner_fits/summary.json"
FITS = DIRECTORY / "mariner_fits"
STARCODER = "starcoder"
STARCODER_COLOR = "#009E73"
BENCHMARK = "programming_languages"
METRIC = "eval/paloma/dolma_100_programing_languages-tpp10/bpb"
GRID = [0, 10, 30, 40, 50, 55, 60, 65, 70, 80, 90, 100]
STEM = "figure4_left_starcoder_wikipedia_english_math500_mariner"
LEFT_DOMAINS = ["wikipedia", "finemath_3plus", STARCODER]
RIGHT_DOMAINS = ["wikipedia", "finemath_3plus"]


def load_starcoder_curves():
    analysis = json.loads(ANALYSIS.read_text())
    definition = analysis["metric_definition"]
    assert definition["id"] == "scored_byte_bpb_from_token_loss_v1"
    assert definition["schema_version"] == 2 and definition["total_records"] == 102
    assert analysis["complete_common_grid_analysis"]["grid_percent"] == GRID
    allocation = json.loads(ALLOCATION.read_text())
    assert allocation["status"] == "passed"
    epochs = {row["percent"]: row for row in allocation["coordinates"]}
    target = next(c for c in analysis["curves"] if c["arm"] == "target")
    matched = [c for c in analysis["curves"] if c["arm"] == "matched"]
    assert len(matched) == 3 and [p["percent"] for p in target["points"]] == GRID
    curves = {}
    curves[STARCODER, "target", BENCHMARK] = [
        {
            "epochs": epochs[p["percent"]]["target_epochs"],
            "percent": p["percent"],
            "bpb": p["value"],
            "run_name": p["run_names"][0],
        }
        for p in target["points"]
    ]
    points = []
    for index, percent in enumerate(GRID):
        seed_values = []
        run_names = []
        for curve in matched:
            point = curve["points"][index]
            assert point["percent"] == percent and len(point["seed_values"]) == 2
            assert np.isclose(point["value"], np.mean(point["seed_values"]))
            seed_values.extend(point["seed_values"])
            run_names.extend(point["run_names"])
        if percent == 0:
            # The three subsets share the same two web-only runs.
            assert len(set(run_names)) == 2
            seed_values, run_names = seed_values[:2], run_names[:2]
        points.append(
            {
                "epochs": epochs[percent]["matched_epochs"],
                "percent": percent,
                "bpb": float(np.mean(seed_values)),
                "run_name": "+".join(sorted(set(run_names))),
                "runs": len(run_names),
            }
        )
    curves[STARCODER, "matched", BENCHMARK] = points
    return curves


def fit_starcoder(curves, design, hashes):
    FITS.mkdir(exist_ok=True)
    results = {}
    for arm in base.ARMS:
        points = curves[STARCODER, arm, BENCHMARK]
        name = f"{STARCODER}_{arm}_{BENCHMARK}"
        output = FITS / f"{name}.json"
        pool_tokens = design["matched_sequences" if arm == "matched" else "parent_sequences"] * fitting.SEQUENCE_LENGTH
        horizon = design["models"]["unmatched" if arm == "matched" else "target"]["tokens"]
        scale = horizon / pool_tokens
        x = np.array([p["epochs"] for p in points])
        share = x / scale
        assert share[0] == 0 and np.isclose(share[-1], 1, atol=1e-4)
        share[-1] = 1.0
        identity = {"sources": hashes, "key": [STARCODER, arm, BENCHMARK], "points": points}
        input_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if output.exists():
            existing = json.loads(output.read_text())
            if existing["input_sha256"] == input_hash:
                results[STARCODER, arm, BENCHMARK] = existing
                print(f"Reusing {name}", flush=True)
                continue
        role = "descriptive_subset_and_seed_mean" if arm == "matched" else "descriptive_single_seed"
        curve = fitting.Curve(name, arm, np.array([p["bpb"] for p in points]), role)
        print(f"Fitting {name} ({len(points)} points)", flush=True)
        result = fitting.fit_curve(curve, share, design, METRIC)
        result.update(
            {
                "domain": STARCODER,
                "benchmark": BENCHMARK,
                "input_sha256": input_hash,
                "observed_epochs": x.tolist(),
                "realized_share": share.tolist(),
                "predicted_minimum_epochs": result["predicted_minimum_share"] * scale,
                "observed_minimum_epochs": points[int(np.argmin(curve.response))]["epochs"],
                "dense_epochs": (np.array(result["dense_share"]) * scale).tolist(),
                "feasible_epoch_range": [0, scale],
                "source_sha256": hashes,
            }
        )
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results[STARCODER, arm, BENCHMARK] = result
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "curve",
                        "observed_minimum_epochs",
                        "predicted_minimum_epochs",
                        "fit_rmse_bpb",
                        "shape",
                        "ridge",
                    ]
                }
            ),
            flush=True,
        )
    return results


def make_figure(curves, design, fits):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.05), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.69, bottom=0.15, wspace=0.20)
    labels = {"wikipedia": "Wikipedia", "finemath_3plus": "FineMath-3+", STARCODER: "StarCoder"}
    handles = [Line2D([], [], color=base.COLORS[domain], lw=2, label=labels[domain]) for domain in LEFT_DOMAINS]
    handles.append(Line2D([], [], ls="none", marker="*", ms=10, color="#555555", label="MARINER fitted minimum"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=1.6,
        columnspacing=1.3,
        handletextpad=0.5,
    )
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
    axes[0].set_title("Different benchmarks", pad=29, fontsize=12, weight="bold")
    axes[0].text(
        0.5,
        1.06,
        "Wikipedia: Wikipedia English  |  FineMath: MATH-500  |  StarCoder: Paloma code",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9.8,
    )
    axes[1].set_title("Same benchmark: MATH-500", pad=29, fontsize=12, weight="bold")
    clipped = []
    exported = []
    for panel, (ax, domains) in enumerate(zip(axes, [LEFT_DOMAINS, RIGHT_DOMAINS], strict=True)):
        for domain in domains:
            if panel == 1:
                benchmark = "math500"
            else:
                benchmark = {"wikipedia": "wikipedia_english", "finemath_3plus": "math500", STARCODER: BENCHMARK}[domain]
            for arm in base.ARMS:
                points = curves[domain, arm, benchmark]
                fit = fits[domain, arm, benchmark]
                clipped.extend(
                    {"panel": panel, "benchmark": benchmark, **p}
                    for p in base.draw_curve(ax, points, domain, arm, benchmark, fit)
                )
                exported.extend(
                    {"panel": panel, "domain": domain, "benchmark": benchmark, "arm": arm, **p} for p in points
                )
        ax.set_xlim(-0.2, 16.4)
        ax.set_ylim(*base.Y_LIMITS)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_yticks(np.arange(0.8, 2.01, 0.2))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Materialized epochs of varied training domain", fontsize=10)
    axes[0].set_ylabel("Evaluation loss (bits per byte)", fontsize=10.5)
    return fig, exported, clipped


# Offsets in points for the compact layout, keyed by the fitted-minimum star each label belongs to.
COMPACT_Y_BOTTOM = 0.62
COMPACT_LABEL_OFFSETS = {
    (STARCODER, "matched"): ((-16, -5.5), "right"),
    (STARCODER, "target"): ((0, -13), "center"),
}


def adjust_left_labels(fig, fits):
    """Move StarCoder's minimum labels off the neighbouring curves and split the panel subtitle."""
    stars = {
        (round(fits[domain, arm, BENCHMARK]["predicted_minimum_epochs"], 6),
         round(fits[domain, arm, BENCHMARK]["predicted_minimum_bpb"], 6)): key
        for key in COMPACT_LABEL_OFFSETS
        for domain, arm in [key]
    }
    moved = set()
    for annotation in fig.axes[0].texts:
        text = annotation.get_text()
        if text.startswith("Wikipedia:"):
            annotation.set_text(
                "Wikipedia: Wikipedia English  |  FineMath: MATH-500\nStarCoder: Paloma programming languages"
            )
            continue
        if not text.endswith(" ep."):
            continue
        key = stars.get((round(annotation.xy[0], 6), round(annotation.xy[1], 6)))
        if key is None:
            continue
        offset, align = COMPACT_LABEL_OFFSETS[key]
        annotation.set_position(offset)
        annotation.set_ha(align)
        moved.add(key)
    assert moved == set(COMPACT_LABEL_OFFSETS), moved


def main():
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10, "axes.unicode_minus": False})
    curves = base.load_curves()
    curves.update(load_starcoder_curves())
    base.COLORS[STARCODER] = STARCODER_COLOR
    design = json.loads(base.DESIGN.read_text())
    # The fit identity excludes this script so that layout edits reuse the cached fits.
    sources = [ANALYSIS, ALLOCATION, base.DESIGN, Path(fitting.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    domain_results = json.loads(DOMAIN_FITS.read_text())["curves"]
    fits = {(r["domain"], r["arm"], r["benchmark"]): r for r in domain_results}
    assert len(fits) == 6
    fits.update(fit_starcoder(curves, design, hashes))
    fig, rows, clipped = make_figure(curves, design, fits)
    base.compact_paper_layout(fig)
    adjust_left_labels(fig, fits)
    for ax in fig.axes:
        # Room for the StarCoder target label under the flat 0.77-BPB basin.
        ax.set_ylim(COMPACT_Y_BOTTOM, base.Y_LIMITS[1])
    for extension in ["pdf", "png"]:
        fig.savefig(DIRECTORY / f"{STEM}.{extension}", dpi=200)
    plt.close(fig)
    receipt = {
        "starcoder_source": "Figure 3 twelve-point TPP-10 curves; matched proxy averaged over three subsets and two trainer seeds",
        "starcoder_metric": METRIC,
        "rows": rows,
        "clipped": clipped,
        "starcoder_fits": {
            f"{arm}": {
                k: fits[STARCODER, arm, BENCHMARK][k]
                for k in ["observed_minimum_epochs", "predicted_minimum_epochs", "predicted_minimum_bpb", "fit_rmse_bpb"]
            }
            for arm in base.ARMS
        },
        "domain_fits_sha256": hashlib.sha256(DOMAIN_FITS.read_bytes()).hexdigest(),
        "source_sha256": {**hashes, str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "output_sha256": {
            extension: hashlib.sha256((DIRECTORY / f"{STEM}.{extension}").read_bytes()).hexdigest()
            for extension in ["pdf", "png"]
        },
    }
    (DIRECTORY / f"{STEM}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"points": len(rows), "clipped": len(clipped), "starcoder_fits": receipt["starcoder_fits"]}, indent=1))


if __name__ == "__main__":
    main()
