# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Two-domain Figure 4 preview from measured curves: Dolmino FLAN and FineMath-3+.

Left: proxy and target sweeps for both domains on one on-target evaluation each, measured points
joined, stars at observed minima. Right: one panel per domain with the measured target curve, its
grid minimum, gray bars for fixed epoch counts from the shared grid, and a colored bar for the
simulated-epoching selection (the matched proxy's grid minimum). Run from the repository root:

    PYTHONPATH=. uv run --offline --no-sync python <this file> --flan-eval sciq_bpb --finemath-eval gsm8k
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(DIRECTORY.parent / "figure4_alternatives_20260913"))
import build_tall_preview as tall  # noqa: E402
import plot_alternatives as base  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
FLAN = "dolmino_flan"
FLAN_COLOR = "#5D3A9B"
FLAN_RESULTS = {
    "matched": REPO / ".agents/projects/starcoder_tpp10/instruction_sweep/results",
    "target": REPO / ".agents/projects/starcoder_tpp10/instruction_sweep/results_target",
}
FLAN_EVALS = ["sciq_bpb", "openbookqa_bpb", "arc_easy_bpb", "arc_challenge_bpb", "qasc_bpb"]
UNMATCHED_RESULTS = {
    FLAN: REPO / ".agents/projects/starcoder_tpp10/instruction_sweep/results_unmatched/measurements.csv",
    "finemath_3plus": REPO / ".agents/projects/starcoder_tpp10/domain_sweeps/finemath_unmatched/results/measurements.csv",
}
UNMATCHED_COLUMNS = {FLAN: {"sciq_bpb": "sciq_bpb", "openbookqa_bpb": "openbookqa_bpb", "qasc_bpb": "qasc_bpb", "arc_easy_bpb": "arc_easy_bpb", "arc_challenge_bpb": "arc_challenge_bpb"},
                     "finemath_3plus": {"gsm8k": "gsm8k_bpb", "math500": "math500_bpb"}}
ARM_STYLES = {"matched": ("-", "o", "Simulated-epoching proxy"), "target": ("--", "s", "Target"), "unmatched": (":", "^", "Proxy without simulated epoching")}
EVAL_LABELS = {**tall.EVAL_LABELS, "sciq_bpb": "SciQ", "openbookqa_bpb": "OpenBookQA", "arc_easy_bpb": "ARC-Easy", "arc_challenge_bpb": "ARC-Challenge", "qasc_bpb": "QASC"}
FIXED_COLOR = "#6b6b6b"
EXCESS_COLOR = "#D62728"  # the cross-domain cap bar, its diamond and its label, on both panels
MINIMUM_LINE_COLOR = "#909090"
RIGHT_X_LIMITS = (0.0, 16.4)


def measured(points):
    return np.array([p["epochs"] for p in points]), np.array([p["bpb"] for p in points])


def load_flan_curves():
    """Measured FLAN curves for both arms from the sweep collections, epochs from their analyses."""
    curves = {}
    for arm, directory in FLAN_RESULTS.items():
        frame = pd.read_csv(directory / "measurements.csv").sort_values("percent")
        analysis = json.loads((directory / "analysis.json").read_text())["metrics"]
        epochs = dict(zip(analysis["sciq_bpb"]["grid_percent"], analysis["sciq_bpb"]["epochs"], strict=True))
        assert list(frame["percent"]) == [0, 5, 10, 20, 30, 50, 70, 100]
        for benchmark in FLAN_EVALS:
            curves[FLAN, arm, benchmark] = [
                {"epochs": epochs[int(row.percent)], "percent": int(row.percent), "bpb": float(getattr(row, benchmark)), "run_name": row.run_name}
                for row in frame.itertuples()
            ]
    return curves


def load_unmatched_curves(curves, design, rows):
    """Unmatched-proxy curves: the proxy on the full parent pool, so epochs = p * proxy tokens / parent tokens."""
    parent_tokens = design["parent_sequences"] * 2048
    proxy_tokens = design["models"]["unmatched"]["tokens"]
    for domain, benchmark, _ in rows:
        frame = pd.read_csv(UNMATCHED_RESULTS[domain]).sort_values("percent")
        column = UNMATCHED_COLUMNS[domain][benchmark]
        points = {int(r.percent): float(getattr(r, column)) for r in frame.itertuples()}
        if 0 not in points:  # FineMath's p=0 is the shared web-only proxy run of the matched arm
            points[0] = next(pt["bpb"] for pt in curves[domain, "matched", benchmark] if pt["percent"] == 0)
        curves[domain, "unmatched", benchmark] = [
            {"epochs": percent / 100 * proxy_tokens / parent_tokens, "percent": percent, "bpb": points[percent], "run_name": f"{domain}_unmatched_p{percent:03d}"}
            for percent in sorted(points)
        ]
    return curves


def excess_table(curves, rows, optima):
    """Per domain: the target loss at the other domain's optimum used as the cap, and at the simulated-epoching pick."""
    table = []
    for i, (domain, benchmark, _) in enumerate(rows):
        x, y = measured(curves[domain, "target", benchmark])
        minimum = float(y.min())
        px, py = measured(curves[domain, "matched", benchmark])
        policies = []
        other = rows[1 - i][0]
        cap = optima[other]
        index = int(np.argmin(np.abs(x - cap)))
        assert abs(x[index] - cap) < 0.03, (domain, cap, x[index])
        policies.append((f"{other} optimum", float(x[index]), float(y[index])))
        pick = int(np.argmin(py))
        index = int(np.argmin(np.abs(x - px[pick])))
        assert abs(x[index] - px[pick]) < 0.03
        policies.append(("simulated epoching", float(x[index]), float(y[index])))
        for policy, epochs, loss in policies:
            table.append({"domain": domain, "benchmark": benchmark, "policy": policy, "epochs": epochs, "target_bpb": loss,
                          "excess_bpb": loss - minimum, "excess_percent": 100 * (loss - minimum) / minimum})
    return table


def draw_left(ax, curves, rows, arms=("matched", "target"), x_axis="epochs"):
    """``x_axis``: "epochs" (materialized domain epochs) or "weight" (domain share p); stars carry their own epoch count."""
    polylines, minima, star_epochs = {}, {}, {}
    for domain, benchmark, _ in rows:
        for arm in arms:
            points = curves[domain, arm, benchmark]
            epochs, y = measured(points)
            full = curves[domain, "target", benchmark][-1]["epochs"] / 100  # target-scale epochs per weight percent
            if x_axis == "weight":
                x = np.array([pt["percent"] for pt in points], dtype=float)
            elif arm == "unmatched":
                x = np.array([pt["percent"] * full for pt in points])  # same weight, same x as the matched tracks
            else:
                x = epochs
            polylines[domain, arm] = (x, y)
            line, marker, _ = ARM_STYLES[arm]
            color = base.COLORS[domain]
            ax.plot(x, y, color=color, ls=line, marker=marker, lw=1.3, ms=3.0, mfc=color if arm == "matched" else "white", mew=0.7, zorder=2)
            i = int(np.argmin(y))
            minima[domain, arm] = (x[i], y[i])
            star_epochs[domain, arm] = float(epochs[i])
            ax.scatter(x[i], y[i], marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=4)
    x_max = 100.0 if x_axis == "weight" else 16.4
    low = min(y.min() for _, y in polylines.values())
    high = max(y.max() for _, y in polylines.values())
    span = high - low
    strip = 0.11 if (x_axis == "epochs" and "unmatched" in arms) else 0.06  # room for the epoch labels at the axes edges
    limits = (low - strip * span, high + strip * span)
    guides = x_axis == "epochs" and "unmatched" in arms
    if guides:
        span = limits[1] - limits[0]
        pad = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
        for domain, _, _ in rows:
            color = base.COLORS[domain]
            (xm, ym), (xt, yt) = minima[domain, "matched"], minima[domain, "target"]
            xo = xt  # matched and target share the optimum epoch count
            ax.plot([xo, xo], [limits[0], max(ym, yt)], color=color, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
            ax.text(xo, limits[0] + 0.012 * span, f"{star_epochs[domain, "target"]:.1f} ep.", ha="center", va="bottom", fontsize=7.5, color=color, zorder=6, bbox=pad)
            xu, yu = minima[domain, "unmatched"]
            ax.plot([xu, xu], [yu, limits[1]], color=color, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
            ax.text(xu, limits[1] - 0.012 * span, f"{star_epochs[domain, "unmatched"]:.1f} ep.", ha="center", va="top", fontsize=7.5, color=color, zorder=6, bbox=pad)
    labelled = set()
    connected = set()
    occupied = []  # boxes of connector labels, in data units, that other labels must avoid
    if x_axis == "weight":
        for domain, _, _ in rows:
            (xm, ym), (xt, yt) = minima[domain, "matched"], minima[domain, "target"]
            if abs(xm - xt) < 0.5:  # same weight, hence the same materialized epochs: one label serves both stars
                color = base.COLORS[domain]
                ax.plot([xm, xt], [ym, yt], color=color, lw=0.9, ls=(0, (1, 1.5)), zorder=3)
                ax.annotate(f"{star_epochs[domain, "target"]:.1f} ep.", (xm, (ym + yt) / 2), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=7.5, color=color, zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
                occupied.append((xm, (ym + yt) / 2))  # filled in data units below once the label size is known
                connected.update({(domain, "matched"), (domain, "target")})
    x_span, y_span = x_max + 0.2, limits[1] - limits[0]
    axes_w_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    axes_h_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    label_w = 26 / axes_w_pt * x_span  # about 26 pt of text
    label_h = 8 / axes_h_pt * y_span
    occupied = [(cx + 6 / axes_w_pt * x_span, cy - label_h / 2, label_w, label_h) for cx, cy in occupied]
    placements = {"below": ((0, -13), "center", "top"), "above": ((0, 8), "center", "bottom"), "right": ((9, -3), "left", "center"), "left": ((-9, -3), "right", "center"),
                  "above-left": ((-6, 7), "right", "bottom"), "above-right": ((6, 7), "left", "bottom"), "below-left": ((-6, -8), "right", "top"), "below-right": ((6, -8), "left", "top")}
    for key in sorted(minima, key=lambda k: (k[0], k[1] != "target")):  # target first: a shared optimum is labelled on the target track
        if key in connected or guides:
            continue
        x, y = minima[key]
        partner = (key[0], "target" if key[1] == "matched" else "matched")
        if x_axis == "epochs" and partner in labelled and abs(minima[partner][0] - x) < 0.03:
            continue  # the two arms share the optimum; one label serves both stars
        labelled.add(key)
        best = None
        for name, ((dx, dy), ha, va) in placements.items():
            cx = x + dx / axes_w_pt * x_span
            cy = y + dy / axes_h_pt * y_span
            x0 = cx if ha == "left" else cx - label_w if ha == "right" else cx - label_w / 2
            y0 = cy if va == "bottom" else cy - label_h if va == "top" else cy - label_h / 2
            crossings = 0
            for ox, oy in polylines.values():
                grid = np.linspace(max(x0 - 0.3, 0), min(x0 + label_w + 0.3, x_max), 60)
                oy_interp = np.interp(grid, ox, oy)
                crossings += int(((oy_interp > y0 - 0.2 * label_h) & (oy_interp < y0 + 1.2 * label_h)).sum())
            if x0 < -0.1 or x0 + label_w > x_max or y0 < limits[0] or y0 + label_h > limits[1]:
                crossings += 1000  # keep the label inside the axes
            for bx, by, bw, bh in occupied:
                if x0 < bx + bw and x0 + label_w > bx and y0 < by + bh and y0 + label_h > by:
                    crossings += 50  # do not sit on a connector label
            if best is None or crossings < best[0]:
                best = (crossings, name)
        (dx, dy), ha, va = placements[best[1]]
        # A faint white pad keeps the label legible where curves converge on the optimum.
        ax.annotate(f"{star_epochs[key]:.1f} ep.", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.5, color=base.COLORS[key[0]], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_ylim(*limits)
    if x_axis == "weight":
        ax.set_xlim(-1.5, 102)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(["0", "20", "40", "60", "80", "100%"])
        full = curves[rows[0][0], "target", rows[0][1]][-1]["epochs"] / 100  # epochs per weight percent at the shared scale
        top = ax.secondary_xaxis("top", functions=(lambda w: w * full, lambda e: e / full))
        top.set_xticks([0, 4, 8, 12, 16])
        top.tick_params(labelsize=7, length=2.5)
        top.set_xlabel("Materialized epochs (target and simulated-epoching proxy)", fontsize=6.6, labelpad=2)
        top.spines["top"].set_visible(True)
    else:
        ax.set_xlim(-0.2, 16.4)
        ax.set_xticks([0, 4, 8, 12, 16])
        if guides:
            ratio = curves[rows[0][0], "unmatched", rows[0][1]][-1]["epochs"] / curves[rows[0][0], "target", rows[0][1]][-1]["epochs"]
            top = ax.secondary_xaxis("top", functions=(lambda e: e * ratio, lambda u: u / ratio))
            top.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            top.tick_params(labelsize=7, length=2.5)
            top.set_xlabel("Epochs without simulated epoching", fontsize=6.6, labelpad=2)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Domain weight in the mixture" if x_axis == "weight" else ("Materialized domain epochs\n(target and simulated-epoching proxy)" if guides else "Materialized domain epochs"), fontsize=8 if not guides else 7.4)
    ax.set_ylabel("Evaluation loss (BPB)", fontsize=8)
    if x_axis != "weight" and not guides:
        ax.text(0.5, 1.012, "  |  ".join(f"{label}: {EVAL_LABELS[benchmark]}" for _, benchmark, label in rows),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.9)
    return {f"{d}/{a}": {"epochs": star_epochs[d, a], "bpb": float(y)} for (d, a), (x, y) in minima.items()}


def draw_right(ax, curves, table, domain, benchmark, label, bottom_panel, other):
    """``other`` = (epochs, color, label) of the other domain's optimum; that fixed cap is drawn in the other domain's colour."""
    color = base.COLORS[domain]
    other_epochs, other_color, other_label = other
    x, y = measured(curves[domain, "target", benchmark])
    minimum = float(y.min())
    ax.plot(x, y, color=color, ls="--", marker="s", lw=1.3, ms=3.0, mfc="white", mew=0.7, zorder=2)
    ax.axhline(minimum, color=MINIMUM_LINE_COLOR, ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.scatter(x[np.argmin(y)], minimum, marker="*", s=85, color=color, edgecolor="#222222", lw=0.55, zorder=6)
    panel_rows = [r for r in table if r["domain"] == domain]
    reach = max(r["target_bpb"] for r in panel_rows) - minimum
    for row in panel_rows:
        fixed = row["policy"] != "simulated epoching"
        cross = fixed and abs(row["epochs"] - other_epochs) < 0.03
        bar_color = EXCESS_COLOR if cross else (FIXED_COLOR if fixed else color)
        if row["excess_bpb"] > 0:
            tall.bracket(ax, row["epochs"], minimum, row["target_bpb"], bar_color, lw=4.2 if cross else 3.0)
        ax.plot(row["epochs"], row["target_bpb"], marker="D", ms=5.4 if cross else 4.2, color=bar_color, markeredgecolor="#222222" if cross else "white", markeredgewidth=0.7, ls="none", zorder=7)
        text = f"+{row['excess_percent']:.1f}%"
        if cross:
            # One block to the right of the diamond: the excess in bold, then whose optimum this cap is.
            pad = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85)
            ax.annotate(text, (row["epochs"], row["target_bpb"]), xytext=(6, 17), textcoords="offset points", ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=EXCESS_COLOR, zorder=8, bbox=pad)
            ax.annotate(f"{other_label}'s\noptimum", (row["epochs"], row["target_bpb"]), xytext=(6, 1), textcoords="offset points", ha="left", va="bottom", fontsize=6.2, color=EXCESS_COLOR, zorder=8, linespacing=1.0, bbox=pad)
        elif fixed:
            grid = np.linspace(row["epochs"] - 0.95, row["epochs"] + 0.95, 40)
            nearby = np.interp(grid, x, y)
            beside = bool(((nearby > row["target_bpb"] + 0.04 * reach) & (nearby < row["target_bpb"] + 0.22 * reach)).any())
            ax.annotate(text, (row["epochs"], row["target_bpb"]), xytext=(5, 2) if beside else (0, 5), textcoords="offset points",
                        ha="left" if beside else "center", fontsize=6.8, color=FIXED_COLOR, zorder=8)
        # The simulated-epoching pick coincides with the minimum on both domains; the diamond on the star says so.
    ax.set_xlim(*RIGHT_X_LIMITS)
    ax.set_ylim(minimum - 0.32 * reach, minimum + 1.75 * reach)  # headroom for the label block above the cap diamond
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDE2E7", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=7, length=3, width=0.6)
    ax.set_title(f"{label} ({EVAL_LABELS[benchmark]})", fontsize=7.6, loc="left", pad=4)
    if bottom_panel:
        ax.set_xlabel("Materialized domain epochs", fontsize=8)
    else:
        ax.tick_params(labelbottom=False)


def make_figure(curves, design, rows, figsize=(5.6, 5.2), compact=False, arms=("matched", "target"), x_axis="epochs"):
    """``compact``: print-size layout with two legend rows and no FLOPs in the legend (they go in the caption)."""
    fig = plt.figure(figsize=figsize)
    bottom = (0.125 if "unmatched" in arms and x_axis == "epochs" else 0.10) if compact else 0.095
    top_axis = x_axis == "weight" or "unmatched" in arms
    axes_top = (0.785 if top_axis else 0.835) if compact else (0.78 if top_axis else 0.83)  # one legend row above; headers sit just under it
    gap = 0.075
    panel_height = (axes_top - bottom - gap) / 2
    left = fig.add_axes([0.11 if compact else 0.10, bottom, 0.41, axes_top - bottom])
    rights = [fig.add_axes([0.655 if compact else 0.645, bottom + (1 - i) * (panel_height + gap), 0.32, panel_height]) for i in range(2)]
    minima = draw_left(left, curves, rows, arms=arms, x_axis=x_axis)
    optima = {domain: minima[f"{domain}/target"]["epochs"] for domain, _, _ in rows}
    table = excess_table(curves, rows, optima)
    for ax, (domain, benchmark, label), i in zip(rights, rows, range(2), strict=True):
        other_domain, _, other_label = rows[1 - i]
        draw_right(ax, curves, table, domain, benchmark, label, bottom_panel=i == 1, other=(optima[other_domain], base.COLORS[other_domain], other_label))
        ax.set_ylabel("Target loss (BPB)", fontsize=8)
    header_y = axes_top + ((0.095 if top_axis else 0.048) if compact else (0.09 if top_axis else 0.04))
    left_box = left.get_position()
    fig.text(left_box.x0 + left_box.width / 2, header_y, "Proxy and target sweeps", ha="center", va="bottom", fontsize=9, weight="bold")
    if top_axis:
        fig.text(left_box.x0 + left_box.width / 2, header_y - 0.028, "  |  ".join(f"{label}: {EVAL_LABELS[benchmark]}" for _, benchmark, label in rows), ha="center", va="bottom", fontsize=6.9)
    fig.text((0.655 if compact else 0.645) + 0.16, header_y, "Excess loss", ha="center", va="bottom", fontsize=9, weight="bold")
    domain_handles = [Line2D([], [], color=base.COLORS[d], lw=2, label=label) for d, _, label in rows]
    proxy_flops = design["models"]["unmatched"]["training_flops"]
    target_flops = design["models"]["target"]["training_flops"]
    scale_handles = [
        Line2D([], [], color="#444444", lw=1.3, marker="o", ms=3, label="Simulated-epoching proxy" if compact else rf"Simulated-epoching proxy ($ {proxy_flops / 1e16:.2f} \times 10^{{16}}$ FLOPs)"),
        Line2D([], [], color="#444444", lw=1.3, marker="s", ms=3, mfc="white", ls="--", label="Target" if compact else rf"Target ($ {target_flops / 1e18:.2f} \times 10^{{18}}$ FLOPs)"),
    ]
    if "unmatched" in arms:
        scale_handles.append(Line2D([], [], color="#444444", lw=1.3, marker="^", ms=3, mfc="white", ls=":", label="Proxy without simulated epoching (< 1 epoch)"))
    fig.legend(handles=domain_handles + scale_handles, loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=5 if "unmatched" in arms else 4, frameon=False, fontsize=(6.6 if "unmatched" in arms else 7.4) if compact else 8, handlelength=1.5, columnspacing=0.9 if compact else 1.6)
    return fig, table, minima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flan-eval", default="sciq_bpb", choices=FLAN_EVALS)
    parser.add_argument("--finemath-eval", default="gsm8k", choices=tall.UNCHEATABLE + tall.MATH)
    parser.add_argument("--figsize", default="5.6,5.2", help="Figure size in inches; the paper version is rendered at its print size.")
    parser.add_argument("--stem", default=None, help="Output stem (default figure4_two_f-<flan>_m-<math>).")
    parser.add_argument("--compact", action="store_true", help="Print-size layout: two legend rows, FLOPs left to the caption.")
    parser.add_argument("--unmatched", action="store_true", help="Add the no-simulated-epoching proxy tracks to the left panel.")
    parser.add_argument("--x-axis", choices=["epochs", "weight"], default="epochs")
    args = parser.parse_args()
    rows = [(FLAN, args.flan_eval, "Dolmino FLAN"), ("finemath_3plus", args.finemath_eval, "FineMath-3+")]
    stem = args.stem or f"figure4_two_f-{args.flan_eval.removesuffix('_bpb')}_m-{args.finemath_eval}"
    figsize = tuple(float(v) for v in args.figsize.split(","))
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.unicode_minus": False})
    curves = base.load_curves()
    curves.update(load_flan_curves())
    base.COLORS[FLAN] = FLAN_COLOR
    design = json.loads(base.DESIGN.read_text())
    arms = ("matched", "target", "unmatched") if args.unmatched else ("matched", "target")
    if args.unmatched:
        load_unmatched_curves(curves, design, rows)
    fig, table, minima = make_figure(curves, design, rows, figsize=figsize, compact=args.compact, arms=arms, x_axis=args.x_axis)
    for extension in ["pdf", "png"]:
        fig.savefig(DIRECTORY / f"{stem}.{extension}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    sources = [base.DESIGN, base.NATIVE, base.MATH, *(d / "measurements.csv" for d in FLAN_RESULTS.values()), Path(__file__)]
    receipt = {"flan_eval": args.flan_eval, "finemath_eval": args.finemath_eval, "excess_rows": table, "left_minima": minima,
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "output_sha256": {e: hashlib.sha256((DIRECTORY / f"{stem}.{e}").read_bytes()).hexdigest() for e in ["pdf", "png"]}}
    (DIRECTORY / f"{stem}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(stem)
    for row in table:
        print(f"{row['domain']:<15} {row['policy']:<18} {row['epochs']:6.2f} ep  +{row['excess_bpb']:.4f} BPB  {row['excess_percent']:+.2f}%")


if __name__ == "__main__":
    main()
