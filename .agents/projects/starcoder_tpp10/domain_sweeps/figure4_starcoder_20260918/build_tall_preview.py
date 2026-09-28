# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Tall Figure 4 preview: three-domain curves on the left, fixed-epoch excess panels on the right.

The right column stacks one panel per domain in the style of Figure 3's target panel: the fitted
MARINER target curve, its fitted minimum, and the excess target loss when the domain is trained
for a fixed epoch count versus the epoch count selected by simulated epoching (the matched
proxy's fitted minimum at the same domain fraction). Excess values are read from the fitted
target curves. The evaluation used for Wikipedia and FineMath is selectable; StarCoder has only
the Paloma programming-languages evaluation.

Run from the repository root:
PYTHONPATH=. uv run --offline --no-sync python <this file> --wikipedia-eval gsm8k --finemath-eval gsm8k --caps 4,8,12
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(DIRECTORY.parent / "figure4_alternatives_20260913"))

import build_preview as previous  # noqa: E402
import plot_alternatives as base  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_starcoder_tpp10_mariner_20260911 as fitting,
)

PREFLIGHT = DIRECTORY.parent / "native_closeout_20260912/plots/preflight.json"
FITS = DIRECTORY / "mariner_fits"
FIXED_COLOR = "#6b6b6b"
MINIMUM_LINE_COLOR = "#909090"
UNCHEATABLE = ["ao3_english", "arxiv_computer_science", "arxiv_physics", "bbc_news", "github_cpp", "github_python", "wikipedia_english"]
MATH = ["math500", "gsm8k"]
EVAL_LABELS = {
    "wikipedia_english": "Wikipedia English",
    "bbc_news": "BBC News",
    "ao3_english": "AO3 English",
    "arxiv_computer_science": "arXiv CS",
    "arxiv_physics": "arXiv physics",
    "github_cpp": "GitHub C++",
    "github_python": "GitHub Python",
    "math500": "MATH-500",
    "gsm8k": "GSM8K",
    previous.BENCHMARK: "Paloma code",
}
LEFT_Y_LIMITS = (0.62, base.Y_LIMITS[1])
RIGHT_X_LIMITS = (3.0, 16.4)


def domain_rows(wikipedia_eval, finemath_eval):
    return [
        ("wikipedia", wikipedia_eval, "Wikipedia"),
        ("finemath_3plus", finemath_eval, "FineMath-3+"),
        (previous.STARCODER, previous.BENCHMARK, "StarCoder"),
    ]


def load_uncheatable_curves():
    """Per-component Uncheatable curves for both domains and arms from the native closeout."""
    native = json.loads(base.NATIVE.read_text())
    curves = {}
    for curve in native["curves"]:
        for component in UNCHEATABLE:
            curves[curve["domain"], curve["arm"], component] = [
                {
                    "epochs": p["epochs"],
                    "percent": p["percent"],
                    "bpb": p["metrics"][f"eval/uncheatable_eval/{component}/bpb"],
                    "run_name": p["run_name"],
                }
                for p in curve["points"]
            ]
    for points in curves.values():
        assert [p["percent"] for p in points] == [0, 5, 10, 20, 30, 50, 70, 100]
    return curves


def fit_domain_curve(curves, design, preflight, hashes, domain, arm, benchmark):
    """Descriptive MARINER fit of one domain-sweep curve, as in fit_mariner.py, cached by input hash."""
    FITS.mkdir(exist_ok=True)
    name = f"{domain}_{arm}_{benchmark}"
    output = FITS / f"{name}.json"
    points = curves[domain, arm, benchmark]
    pool_key = f"{domain}/{'matched' if arm == 'matched' else 'parent'}"
    pool_tokens = preflight["data"]["caches"][pool_key]["tokens"]
    assert pool_tokens == design["matched_sequences" if arm == "matched" else "parent_sequences"] * fitting.SEQUENCE_LENGTH
    horizon = design["models"]["unmatched" if arm == "matched" else "target"]["tokens"]
    scale = horizon / pool_tokens
    x = np.array([p["epochs"] for p in points])
    share = x / scale
    assert share[0] == 0 and np.isclose(share[-1], 1, atol=1e-6)
    share[-1] = 1.0
    identity = {"sources": hashes, "key": [domain, arm, benchmark], "points": points}
    input_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output.exists():
        existing = json.loads(output.read_text())
        if existing["input_sha256"] == input_hash:
            print(f"Reusing {name}", flush=True)
            return existing
    curve = fitting.Curve(name, arm, np.array([p["bpb"] for p in points]), "descriptive_single_seed")
    metric = f"eval/{benchmark}/bpb" if benchmark in MATH else f"eval/uncheatable_eval/{benchmark}/bpb"
    print(f"Fitting {name}", flush=True)
    result = fitting.fit_curve(curve, share, design, metric)
    result.update(
        {
            "domain": domain,
            "benchmark": benchmark,
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
    print(
        json.dumps({k: result[k] for k in ["curve", "observed_minimum_epochs", "predicted_minimum_epochs", "fit_rmse_bpb"]}),
        flush=True,
    )
    return result


def bracket(ax, x, bottom, top, color, cap=0.22, lw=3.0):
    ax.plot([x, x], [bottom, top], color=color, linewidth=lw, solid_capstyle="butt", zorder=5)
    ax.hlines([bottom, top], x - cap, x + cap, color=color, linewidth=1.6, zorder=5)


def label_collides_above(dense_x, dense_y, x, y, reach, half_width=0.95):
    """True when the fitted curve passes through the box a centred label above (x, y) would occupy."""
    box = (np.abs(dense_x - x) <= half_width) & (dense_y > y + 0.04 * reach) & (dense_y < y + 0.22 * reach)
    return bool(box.any())


def excess_table(fits, rows, caps):
    """Excess target loss at the fixed epoch counts and at the simulated-epoching selection."""
    table = []
    for domain, benchmark, _ in rows:
        target = fits[domain, "target", benchmark]
        matched = fits[domain, "matched", benchmark]
        dense_x = np.array(target["dense_epochs"])
        dense_y = np.array(target["dense_prediction"])
        minimum = target["predicted_minimum_bpb"]
        selection = matched["predicted_minimum_share"] * target["feasible_epoch_range"][1]
        policies = [(f"{e:g} epochs", float(e)) for e in caps] + [("simulated epoching", selection)]
        for policy, epochs in policies:
            loss = float(np.interp(epochs, dense_x, dense_y))
            table.append(
                {
                    "domain": domain,
                    "benchmark": benchmark,
                    "policy": policy,
                    "epochs": epochs,
                    "target_bpb": loss,
                    "excess_bpb": loss - minimum,
                    "excess_percent": 100 * (loss - minimum) / minimum,
                }
            )
    return table


def place_left_labels(ax, fits, rows):
    """Put each fitted-minimum label above or below its star, whichever side is clear of the other curves."""
    stars = {}
    for domain, benchmark, _ in rows:
        for arm in base.ARMS:
            fit = fits[domain, arm, benchmark]
            stars[round(fit["predicted_minimum_epochs"], 6), round(fit["predicted_minimum_bpb"], 6)] = (domain, arm)
    dense = {
        (domain, arm): (np.array(fits[domain, arm, benchmark]["dense_epochs"]), np.array(fits[domain, arm, benchmark]["dense_prediction"]))
        for domain, benchmark, _ in rows
        for arm in base.ARMS
    }
    span = LEFT_Y_LIMITS[1] - LEFT_Y_LIMITS[0]
    moved = set()
    for annotation in ax.texts:
        key = stars.get((round(annotation.xy[0], 6), round(annotation.xy[1], 6)))
        if key is None:
            continue
        x, y = annotation.xy
        clearance = {}
        for side, (low, high) in {"above": (0.02, 0.10), "below": (-0.11, -0.03)}.items():
            crossing = 0.0
            for other, (dx, dy) in dense.items():
                if other == key:
                    continue
                inside = (np.abs(dx - x) <= 1.3) & (dy > y + low * span) & (dy < y + high * span)
                crossing += inside.sum()
            clearance[side] = crossing
        side = "above" if clearance["above"] <= clearance["below"] else "below"
        annotation.set_position((0, 8) if side == "above" else (0, -13))
        annotation.set_ha("center")
        annotation.set_fontsize(7.5)
        moved.add(key)
    assert len(moved) == 6, moved


def draw_left(ax, curves, fits, rows):
    clipped = []
    for domain, benchmark, _ in rows:
        for arm in base.ARMS:
            points = curves[domain, arm, benchmark]
            clipped.extend(base.draw_curve(ax, points, domain, arm, benchmark, fits[domain, arm, benchmark]))
    for line in ax.lines:
        line.set_linewidth(1.3)
        line.set_markersize(3.0)
        line.set_markeredgewidth(0.7)
    for collection in ax.collections:
        collection.set_sizes([85])
        collection.set_linewidth(0.55)
    place_left_labels(ax, fits, rows)
    ax.set_xlim(-0.2, 16.4)
    ax.set_ylim(*LEFT_Y_LIMITS)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_yticks(np.arange(0.8, 2.01, 0.2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Materialized domain epochs", fontsize=8)
    ax.set_ylabel("Evaluation loss (BPB)", fontsize=8)
    ax.set_title("Proxy and target sweeps", fontsize=9, weight="bold", pad=22)
    labels = [f"{label}: {EVAL_LABELS[benchmark]}" for _, benchmark, label in rows]
    ax.text(
        0.5,
        1.015,
        "  |  ".join(labels[:2]) + "\n" + labels[2],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
    )
    return clipped


def draw_right(ax, fits, table, domain, benchmark, label, bottom_panel):
    color = base.COLORS[domain]
    target = fits[domain, "target", benchmark]
    dense_x = np.array(target["dense_epochs"])
    dense_y = np.array(target["dense_prediction"])
    minimum = target["predicted_minimum_bpb"]
    ax.plot(dense_x, dense_y, color=color, ls="--", lw=1.3, zorder=2)
    ax.axhline(minimum, color=MINIMUM_LINE_COLOR, ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.scatter(
        target["predicted_minimum_epochs"], minimum, marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=6
    )
    panel_rows = [r for r in table if r["domain"] == domain]
    reach = max(r["target_bpb"] for r in panel_rows) - minimum
    for row in panel_rows:
        fixed = row["policy"] != "simulated epoching"
        bar_color = FIXED_COLOR if fixed else color
        bracket(ax, row["epochs"], minimum, row["target_bpb"], bar_color)
        ax.plot(
            row["epochs"],
            row["target_bpb"],
            marker="D",
            ms=4.2,
            color=bar_color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ls="none",
            zorder=7,
        )
        text = f"+{row['excess_percent']:.1f}%"
        if fixed:
            beside = label_collides_above(dense_x, dense_y, row["epochs"], row["target_bpb"], reach)
            ax.annotate(
                text,
                (row["epochs"], row["target_bpb"]),
                xytext=(5, 2) if beside else (0, 5),
                textcoords="offset points",
                ha="left" if beside else "center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color=FIXED_COLOR,
                zorder=8,
            )
        else:
            ax.annotate(
                text,
                (row["epochs"], minimum),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.8,
                fontweight="bold",
                color=color,
                zorder=8,
            )
    ax.set_xlim(*RIGHT_X_LIMITS)
    ax.set_ylim(minimum - 0.32 * reach, minimum + 1.30 * reach)
    ax.set_xticks([4, 8, 12, 16])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=7, length=3, width=0.6)
    ax.set_title(f"{label} ({EVAL_LABELS[benchmark]})", fontsize=7.6, loc="left", pad=4)
    if bottom_panel:
        ax.set_xlabel("Materialized domain epochs", fontsize=8)
    else:
        ax.tick_params(labelbottom=False)


def make_figure(curves, fits, design, rows, caps):
    table = excess_table(fits, rows, caps)
    fig = plt.figure(figsize=(5.6, 6.6))
    left = fig.add_axes([0.09, 0.075, 0.43, 0.725])
    panel_height = 0.2155
    gap = 0.0395
    rights = [fig.add_axes([0.645, 0.075 + (2 - i) * (panel_height + gap), 0.32, panel_height]) for i in range(3)]
    clipped = draw_left(left, curves, fits, rows)
    for ax, (domain, benchmark, label), i in zip(rights, rows, range(3), strict=True):
        draw_right(ax, fits, table, domain, benchmark, label, bottom_panel=i == 2)
    rights[1].set_ylabel("Target loss (BPB)", fontsize=8)
    header_x = 0.645 + 0.16
    fig.text(header_x, 0.851, "Fixed epoch count vs.\nsimulated epoching", ha="center", va="bottom", fontsize=9, weight="bold")
    fig.text(header_x, 0.833, "Excess target loss over the fitted minimum", ha="center", va="bottom", fontsize=6.9)
    domain_handles = [Line2D([], [], color=base.COLORS[d], lw=2, label=label) for d, _, label in rows]
    fig.legend(
        handles=domain_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=1.6,
    )
    proxy_flops = design["models"]["unmatched"]["training_flops"]
    target_flops = design["models"]["target"]["training_flops"]
    scale_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            lw=1.3,
            marker="o",
            ms=3,
            label=rf"Simulated-epoching proxy ($ {proxy_flops / 1e16:.2f} \times 10^{{16}}$ FLOPs)",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            lw=1.3,
            marker="s",
            ms=3,
            mfc="white",
            ls="--",
            label=rf"Target ($ {target_flops / 1e18:.2f} \times 10^{{18}}$ FLOPs)",
        ),
    ]
    fig.legend(
        handles=scale_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.966),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.6,
    )
    marker_handles = [
        Line2D([], [], ls="none", marker="*", ms=9, color="#555555", label="Fitted minimum"),
        Line2D(
            [],
            [],
            ls="none",
            marker="D",
            ms=5,
            color=FIXED_COLOR,
            markeredgecolor="white",
            label=f"Fixed epoch count ({', '.join(f'{c:g}' for c in caps)})",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="D",
            ms=5,
            color="#444444",
            markeredgecolor="white",
            label="Simulated-epoching selection",
        ),
    ]
    fig.legend(
        handles=marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.934),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    return fig, table, clipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikipedia-eval", default="wikipedia_english", choices=UNCHEATABLE + MATH)
    parser.add_argument("--finemath-eval", default="math500", choices=UNCHEATABLE + MATH)
    parser.add_argument("--caps", default="4,8,12", help="Comma-separated fixed epoch counts.")
    args = parser.parse_args()
    caps = [float(c) for c in args.caps.split(",")]
    rows = domain_rows(args.wikipedia_eval, args.finemath_eval)
    stem = f"figure4_tall_w-{args.wikipedia_eval}_f-{args.finemath_eval}_caps-{'-'.join(f'{c:g}' for c in caps)}"
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.unicode_minus": False})
    curves = base.load_curves()
    curves.update(load_uncheatable_curves())
    curves.update(previous.load_starcoder_curves())
    base.COLORS[previous.STARCODER] = previous.STARCODER_COLOR
    design = json.loads(base.DESIGN.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    sources = [previous.ANALYSIS, previous.ALLOCATION, base.DESIGN, base.NATIVE, base.MATH, PREFLIGHT, Path(fitting.__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    fits = {(r["domain"], r["arm"], r["benchmark"]): r for r in json.loads(previous.DOMAIN_FITS.read_text())["curves"]}
    assert len(fits) == 6
    fits.update(previous.fit_starcoder(curves, design, hashes))
    for domain, benchmark, _ in rows[:2]:
        for arm in base.ARMS:
            if (domain, arm, benchmark) not in fits:
                fits[domain, arm, benchmark] = fit_domain_curve(curves, design, preflight, hashes, domain, arm, benchmark)
    fig, table, clipped = make_figure(curves, fits, design, rows, caps)
    for extension in ["pdf", "png"]:
        fig.savefig(DIRECTORY / f"{stem}.{extension}", dpi=200)
    plt.close(fig)
    receipt = {
        "wikipedia_eval": args.wikipedia_eval,
        "finemath_eval": args.finemath_eval,
        "caps": caps,
        "excess_source": "fitted MARINER target curves (dense 2001-point scan, linear interpolation)",
        "excess_rows": table,
        "clipped_left": clipped,
        "fits": {
            f"{d}/{a}/{b}": {k: fits[d, a, b][k] for k in ["observed_minimum_epochs", "predicted_minimum_epochs", "predicted_minimum_bpb", "fit_rmse_bpb"]}
            for d, b, _ in rows
            for a in base.ARMS
        },
        "domain_fits_sha256": hashlib.sha256(previous.DOMAIN_FITS.read_bytes()).hexdigest(),
        "source_sha256": {
            **hashes,
            str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            str(Path(previous.__file__)): hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
        },
        "output_sha256": {
            extension: hashlib.sha256((DIRECTORY / f"{stem}.{extension}").read_bytes()).hexdigest()
            for extension in ["pdf", "png"]
        },
    }
    (DIRECTORY / f"{stem}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(stem)
    for row in table:
        print(f"{row['domain']:<15} {row['policy']:<18} {row['epochs']:6.2f} ep  +{row['excess_bpb']:.4f} BPB  {row['excess_percent']:+.2f}%")


if __name__ == "__main__":
    main()
