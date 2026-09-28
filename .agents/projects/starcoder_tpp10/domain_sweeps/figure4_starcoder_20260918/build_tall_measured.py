# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "matplotlib"]
# ///
"""Tall Figure 4 preview built from measured curves only, in the style of Figure 3.

Left: proxy and target sweeps for Wikipedia, FineMath-3+ and StarCoder, measured points joined,
stars at observed minima. Right: one panel per domain with the measured target curve, its grid
minimum, gray bars for fixed epoch counts taken from the shared grid, and a colored bar for the
simulated-epoching selection (the matched proxy's grid minimum; for StarCoder the paper's
averaged selection with its validated target run). Excess values are measured, not fitted.

Run from the repository root:
PYTHONPATH=. uv run --offline --no-sync python <this file> --wikipedia-eval gsm8k --finemath-eval gsm8k
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
import build_tall_preview as tall  # noqa: E402
import plot_alternatives as base  # noqa: E402

FOLLOWUP = previous.PAPER / "revision_notes/20260913_figure3_mean_selection_result/sources"
FIXED_COLOR = "#6b6b6b"
MINIMUM_LINE_COLOR = "#909090"
GRID_CAPS = [1.58, 4.77, 7.94, 11.10]
LEFT_Y_LIMITS = (0.62, base.Y_LIMITS[1])
RIGHT_X_LIMITS = (0.0, 16.4)


def measured(points):
    return np.array([p["epochs"] for p in points]), np.array([p["bpb"] for p in points])


def starcoder_followup(curves):
    """The averaged simulated-epoching selection of Figure 3 and its validated target loss."""
    plan = json.loads((FOLLOWUP / "plan.json").read_text())
    result = json.loads((FOLLOWUP / "result.json").read_text())
    assert result["plan_sha256"] == plan["plan_sha256"]
    assert result["final_native_record"]["eval/bpb_schema_version"] == 2
    x, y = measured(curves[previous.STARCODER, "target", previous.BENCHMARK])
    assert np.isclose(y.min(), plan["reference_losses"]["target_grid_minimum_bpb"], atol=1e-12, rtol=0)
    assert np.isclose(result["normalized_bpb"] - y.min(), result["target_regret_bpb"], atol=1e-12, rtol=0)
    return {"epochs": plan["allocation"]["starcoder_epochs"], "target_bpb": result["normalized_bpb"]}


def excess_table(curves, rows, caps, followup):
    table = []
    for domain, benchmark, _ in rows:
        x, y = measured(curves[domain, "target", benchmark])
        minimum = float(y.min())
        px, py = measured(curves[domain, "matched", benchmark])
        policies = []
        for cap in caps:
            index = int(np.argmin(np.abs(x - cap)))
            assert abs(x[index] - cap) < 0.03, (domain, cap, x[index])
            policies.append((f"{cap:g} epochs", float(x[index]), float(y[index])))
        if domain == previous.STARCODER:
            policies.append(("simulated epoching", followup["epochs"], followup["target_bpb"]))
        else:
            pick = int(np.argmin(py))
            index = int(np.argmin(np.abs(x - px[pick])))
            assert abs(x[index] - px[pick]) < 0.03
            policies.append(("simulated epoching", float(x[index]), float(y[index])))
        for policy, epochs, loss in policies:
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


def draw_left(ax, curves, rows):
    span = LEFT_Y_LIMITS[1] - LEFT_Y_LIMITS[0]
    polylines = {}
    minima = {}
    for domain, benchmark, _ in rows:
        for arm in base.ARMS:
            x, y = measured(curves[domain, arm, benchmark])
            polylines[domain, arm] = (x, y)
            line, marker = base.ARMS[arm]
            color = base.COLORS[domain]
            ax.plot(x, y, color=color, ls=line, marker=marker, lw=1.3, ms=3.0, mfc=color if arm == "matched" else "white", mew=0.7, zorder=2)
            i = int(np.argmin(y))
            minima[domain, arm] = (x[i], y[i])
            ax.scatter(x[i], y[i], marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=4)
    for key, (x, y) in minima.items():
        clearance = {}
        for side, (low, high) in {"above": (0.02, 0.10), "below": (-0.11, -0.03)}.items():
            crossing = 0.0
            for other, (ox, oy) in polylines.items():
                if other == key:
                    continue
                grid = np.linspace(max(x - 1.3, 0), min(x + 1.3, 16), 60)
                oy_interp = np.interp(grid, ox, oy)
                crossing += ((oy_interp > y + low * span) & (oy_interp < y + high * span)).sum()
            clearance[side] = crossing
        side = "above" if clearance["above"] <= clearance["below"] else "below"
        ax.annotate(
            f"{x:.1f} ep.",
            (x, y),
            xytext=(0, 8) if side == "above" else (0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=base.COLORS[key[0]],
            zorder=6,
        )
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
    labels = [f"{label}: {tall.EVAL_LABELS[benchmark]}" for _, benchmark, label in rows]
    ax.text(0.5, 1.015, "  |  ".join(labels[:2]) + "\n" + labels[2], transform=ax.transAxes, ha="center", va="bottom", fontsize=6.9)
    return {f"{d}/{a}": {"epochs": float(x), "bpb": float(y)} for (d, a), (x, y) in minima.items()}


def draw_right(ax, curves, table, domain, benchmark, label, bottom_panel):
    color = base.COLORS[domain]
    x, y = measured(curves[domain, "target", benchmark])
    minimum = float(y.min())
    ax.plot(x, y, color=color, ls="--", marker="s", lw=1.3, ms=3.0, mfc="white", mew=0.7, zorder=2)
    ax.axhline(minimum, color=MINIMUM_LINE_COLOR, ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.scatter(x[np.argmin(y)], minimum, marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=6)
    panel_rows = [r for r in table if r["domain"] == domain]
    reach = max(r["target_bpb"] for r in panel_rows) - minimum
    for row in panel_rows:
        fixed = row["policy"] != "simulated epoching"
        bar_color = FIXED_COLOR if fixed else color
        if row["excess_bpb"] > 0:
            tall.bracket(ax, row["epochs"], minimum, row["target_bpb"], bar_color)
        ax.plot(row["epochs"], row["target_bpb"], marker="D", ms=4.2, color=bar_color, markeredgecolor="white", markeredgewidth=0.7, ls="none", zorder=7)
        text = f"+{row['excess_percent']:.1f}%"
        if fixed:
            grid = np.linspace(row["epochs"] - 0.95, row["epochs"] + 0.95, 40)
            nearby = np.interp(grid, x, y)
            beside = bool(((nearby > row["target_bpb"] + 0.04 * reach) & (nearby < row["target_bpb"] + 0.22 * reach)).any())
            ax.annotate(text, (row["epochs"], row["target_bpb"]), xytext=(5, 2) if beside else (0, 5), textcoords="offset points", ha="left" if beside else "center", va="bottom", fontsize=6.8, fontweight="bold", color=FIXED_COLOR, zorder=8)
        else:
            ax.annotate(text, (row["epochs"], minimum), xytext=(0, -5), textcoords="offset points", ha="center", va="top", fontsize=6.8, fontweight="bold", color=color, zorder=8)
    ax.set_xlim(*RIGHT_X_LIMITS)
    ax.set_ylim(minimum - 0.32 * reach, minimum + 1.30 * reach)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=7, length=3, width=0.6)
    ax.set_title(f"{label} ({tall.EVAL_LABELS[benchmark]})", fontsize=7.6, loc="left", pad=4)
    if bottom_panel:
        ax.set_xlabel("Materialized domain epochs", fontsize=8)
    else:
        ax.tick_params(labelbottom=False)


def make_figure(curves, design, rows, caps, followup):
    table = excess_table(curves, rows, caps, followup)
    fig = plt.figure(figsize=(5.6, 6.6))
    left = fig.add_axes([0.09, 0.075, 0.43, 0.725])
    panel_height = 0.2155
    gap = 0.0395
    rights = [fig.add_axes([0.645, 0.075 + (2 - i) * (panel_height + gap), 0.32, panel_height]) for i in range(3)]
    minima = draw_left(left, curves, rows)
    for ax, (domain, benchmark, label), i in zip(rights, rows, range(3), strict=True):
        draw_right(ax, curves, table, domain, benchmark, label, bottom_panel=i == 2)
    rights[1].set_ylabel("Target loss (BPB)", fontsize=8)
    header_x = 0.645 + 0.16
    fig.text(header_x, 0.851, "Fixed epoch count vs.\nsimulated epoching", ha="center", va="bottom", fontsize=9, weight="bold")
    fig.text(header_x, 0.833, "Excess target loss over measured optimum", ha="center", va="bottom", fontsize=6.6)
    fig.legend(handles=[Line2D([], [], color=base.COLORS[d], lw=2, label=label) for d, _, label in rows], loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=3, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.6)
    proxy_flops = design["models"]["unmatched"]["training_flops"]
    target_flops = design["models"]["target"]["training_flops"]
    scale_handles = [
        Line2D([], [], color="#444444", lw=1.3, marker="o", ms=3, label=rf"Simulated-epoching proxy ($ {proxy_flops / 1e16:.2f} \times 10^{{16}}$ FLOPs)"),
        Line2D([], [], color="#444444", lw=1.3, marker="s", ms=3, mfc="white", ls="--", label=rf"Target ($ {target_flops / 1e18:.2f} \times 10^{{18}}$ FLOPs)"),
    ]
    fig.legend(handles=scale_handles, loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=2, frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.6)
    marker_handles = [
        Line2D([], [], ls="none", marker="*", ms=9, color="#555555", label="Measured minimum"),
        Line2D([], [], ls="none", marker="D", ms=5, color=FIXED_COLOR, markeredgecolor="white", label=f"Fixed epoch count ({', '.join(f'{c:.1f}' for c in caps)})"),
        Line2D([], [], ls="none", marker="D", ms=5, color="#444444", markeredgecolor="white", label="Simulated-epoching selection"),
    ]
    fig.legend(handles=marker_handles, loc="upper center", bbox_to_anchor=(0.5, 0.934), ncol=3, frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=1.4, handletextpad=0.5)
    return fig, table, minima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikipedia-eval", default="gsm8k", choices=tall.UNCHEATABLE + tall.MATH)
    parser.add_argument("--finemath-eval", default="gsm8k", choices=tall.UNCHEATABLE + tall.MATH)
    parser.add_argument("--caps", default=",".join(str(c) for c in GRID_CAPS), help="Fixed epoch counts on the shared grid.")
    args = parser.parse_args()
    caps = [float(c) for c in args.caps.split(",")]
    rows = tall.domain_rows(args.wikipedia_eval, args.finemath_eval)
    stem = f"figure4_tall_measured_w-{args.wikipedia_eval}_f-{args.finemath_eval}"
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.unicode_minus": False})
    curves = base.load_curves()
    curves.update(tall.load_uncheatable_curves())
    curves.update(previous.load_starcoder_curves())
    base.COLORS[previous.STARCODER] = previous.STARCODER_COLOR
    design = json.loads(base.DESIGN.read_text())
    followup = starcoder_followup(curves)
    fig, table, minima = make_figure(curves, design, rows, caps, followup)
    for extension in ["pdf", "png"]:
        fig.savefig(DIRECTORY / f"{stem}.{extension}", dpi=200)
    plt.close(fig)
    sources = [previous.ANALYSIS, previous.ALLOCATION, base.DESIGN, base.NATIVE, base.MATH, FOLLOWUP / "plan.json", FOLLOWUP / "result.json", Path(__file__)]
    receipt = {
        "wikipedia_eval": args.wikipedia_eval,
        "finemath_eval": args.finemath_eval,
        "caps": caps,
        "excess_source": "measured target curves at shared grid epochs; StarCoder selection from the validated averaged-selection run",
        "excess_rows": table,
        "left_minima": minima,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "output_sha256": {extension: hashlib.sha256((DIRECTORY / f"{stem}.{extension}").read_bytes()).hexdigest() for extension in ["pdf", "png"]},
    }
    (DIRECTORY / f"{stem}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(stem)
    for row in table:
        print(f"{row['domain']:<15} {row['policy']:<18} {row['epochs']:6.2f} ep  +{row['excess_bpb']:.4f} BPB  {row['excess_percent']:+.2f}%")


if __name__ == "__main__":
    main()
