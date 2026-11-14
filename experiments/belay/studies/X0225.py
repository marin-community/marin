# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""X0225: which layers grow the largest output outliers (max|h| vs rms(h)) over training, and when.

Reads the per-layer output-stream statistics the run logged with ``layer_output_stats_every``
(``train/layer_out/max_abs_L<i>``, ``train/layer_out/rms_L<i>`` and the ``_final`` normed hidden state) from the W&B
history. The outlier ratio of a layer is max|h| / rms(h). The report gives each layer's ratio at the first and last
logged step, its peak and the step it was reached, and the first step the ratio came within 10% of that peak.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

STAT_KEY = re.compile(r"^train/layer_out/(max_abs|rms)_(L\d+|final)$")
NEAR_PEAK = 0.9


def layer_order(layer: str) -> tuple[int, int]:
    return (1, 0) if layer == "final" else (0, int(layer.removeprefix("L")))


def logged_series(history: list[dict]) -> dict[str, dict[str, dict[int, float]]]:
    """Each layer's logged statistics: layer -> stat (``max_abs`` / ``rms``) -> step -> value."""
    series: dict[str, dict[str, dict[int, float]]] = {}
    for row in history:
        for key, value in row.items():
            match = STAT_KEY.match(key)
            if match is None or value is None:
                continue
            stat, layer = match.groups()
            series.setdefault(layer, {}).setdefault(stat, {})[int(row["_step"])] = float(value)
    return dict(sorted(series.items(), key=lambda item: layer_order(item[0])))


def outlier_ratio(stats: dict[str, dict[int, float]]) -> dict[int, float]:
    """max|h| / rms(h) at every step where both were logged."""
    max_abs, rms = stats.get("max_abs", {}), stats.get("rms", {})
    return {step: max_abs[step] / rms[step] for step in sorted(max_abs.keys() & rms.keys()) if rms[step] > 0}


def first_near_peak(ratio: dict[int, float]) -> int:
    peak = max(ratio.values())
    return next(step for step, value in ratio.items() if value >= NEAR_PEAK * peak)


def plot(series: dict[str, dict[str, dict[int, float]]], ratios: dict[str, dict[int, float]], path: Path) -> None:
    panels = [("max_abs", "max|h|"), ("rms", "rms(h)")]
    figure = Figure(figsize=(14, 4), layout="constrained")
    axes = figure.subplots(1, 3)
    for axis, (stat, label) in zip(axes, panels, strict=False):
        for layer, stats in series.items():
            values = stats.get(stat, {})
            axis.plot(list(values), list(values.values()), label=layer)
        axis.set_yscale("log")
        axis.set_xlabel("step")
        axis.set_ylabel(label)
    for layer, ratio in ratios.items():
        axes[2].plot(list(ratio), list(ratio.values()), label=layer)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("max|h| / rms(h)")
    axes[2].legend(fontsize="small")
    figure.savefig(path, dpi=120)


def ratio_table(ratios: dict[str, dict[int, float]]) -> str:
    lines = [
        "| Layer | First step | Ratio at first | Ratio at last | Peak ratio | Peak step | First within 10% of peak |",
        "|---|---|---|---|---|---|---|",
    ]
    for layer, ratio in ratios.items():
        steps = list(ratio)
        peak_step = max(ratio, key=ratio.__getitem__)
        lines.append(
            f"| {layer} | {steps[0]} | {ratio[steps[0]]:.1f} | {ratio[steps[-1]]:.1f} | {ratio[peak_step]:.1f} "
            f"| {peak_step} | {first_near_peak(ratio)} |"
        )
    return "\n".join(lines)


def stat_table(series: dict[str, dict[str, dict[int, float]]]) -> str:
    lines = ["| Layer | max\\|h\\| first | max\\|h\\| last | rms(h) first | rms(h) last |", "|---|---|---|---|---|"]
    for layer, stats in series.items():
        cells = []
        for stat in ("max_abs", "rms"):
            values = list(stats.get(stat, {}).values())
            cells += [f"{values[0]:.3g}", f"{values[-1]:.3g}"] if values else ["n/a", "n/a"]
        lines.append(f"| {layer} | {' | '.join(cells)} |")
    return "\n".join(lines)


def findings(ratios: dict[str, dict[int, float]]) -> str:
    if not ratios:
        return "The history logged no step with both max|h| and rms(h), so no outlier ratio could be computed."
    last = {layer: list(ratio.values())[-1] for layer, ratio in ratios.items()}
    peak = {layer: max(ratio.values()) for layer, ratio in ratios.items()}
    growth = {layer: last[layer] / next(iter(ratio.values())) for layer, ratio in ratios.items()}
    top_last = max(last, key=last.__getitem__)
    top_peak = max(peak, key=peak.__getitem__)
    top_growth = max(growth, key=growth.__getitem__)
    median_last = float(np.median(list(last.values())))
    return (
        f"- Largest ratio at the last logged step: {top_last} ({last[top_last]:.1f}; median over layers "
        f"{median_last:.1f}).\n"
        f"- Largest ratio at any step: {top_peak} ({peak[top_peak]:.1f} at step "
        f"{max(ratios[top_peak], key=ratios[top_peak].__getitem__)}).\n"
        f"- Largest growth from the first to the last logged step: {top_growth} ({growth[top_growth]:.1f}x)."
    )


def analyze(checkpoints: Mapping[int, dict[str, np.ndarray]], history: list[dict], out_dir: Path) -> None:
    series = logged_series(history)
    ratios = {layer: ratio for layer, stats in series.items() if (ratio := outlier_ratio(stats))}
    figure = ""
    if series:
        plot(series, ratios, out_dir / "layer_out.png")
        figure = "![max|h|, rms(h) and their ratio per layer over training](layer_out.png)\n\n"
    steps = sorted({step for stats in series.values() for values in stats.values() for step in values})
    span = f"{len(steps)} logged steps ({steps[0]} to {steps[-1]})" if steps else "no logged steps"
    (out_dir / "report.md").write_text(
        "# X0225: layer-output max vs RMS over training\n\n"
        "Question: in the current recipe at d512, which layers grow the largest output outliers (max|h| vs rms(h)) "
        "over training, and when?\n\n"
        f"Source: the run's logged `train/layer_out/*` statistics, {span}, over {len(series)} layer outputs "
        f"(`L<i>` is layer i's output stream, `final` the normed hidden state before the head); "
        f"{len(checkpoints)} saved checkpoint(s), not read. "
        "max|h| and rms(h) are taken over every element of a training batch.\n\n"
        f"## Findings\n\n{findings(ratios)}\n\n"
        f"{figure}"
        f"## Outlier ratio max|h| / rms(h)\n\n{ratio_table(ratios)}\n\n"
        f"## Logged statistics\n\n{stat_table(series)}\n"
    )
