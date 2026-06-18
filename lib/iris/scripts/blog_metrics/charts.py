# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline step 3: render blog charts from the daily CSVs.

Reads ``<data_dir>/daily/*.csv`` and writes SVG + PNG charts to
``<data_dir>/charts``. Uses matplotlib only (no pandas); run with
``uv run --with matplotlib``. Time-axis charts overlay the milestones from
``config.MILESTONES`` that fall inside the data window.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os

import matplotlib as mpl

mpl.use("Agg")  # headless backend; must precede pyplot import
import config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Generation-ordered family palette so stacks/legends stay stable across charts.
FAMILY_ORDER = ["v4", "v5e", "v5p", "v6e", "v3"]
FAMILY_COLOR = {
    "v4": "#4C72B0",
    "v5e": "#55A868",
    "v5p": "#C44E52",
    "v6e": "#8172B3",
    "v3": "#CCB974",
}


def _load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dates(rows: list[dict[str, str]]) -> list[dt.date]:
    return [dt.date.fromisoformat(r["day"]) for r in rows]


def _floats(rows: list[dict[str, str]], col: str) -> list[float]:
    return [float(r[col]) if r[col] not in ("", None) else 0.0 for r in rows]


def _save(fig: plt.Figure, charts_dir: str, name: str) -> None:
    for ext in ("svg", "png"):
        out = os.path.join(charts_dir, f"{name}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote %s.{svg,png}", os.path.join(charts_dir, name))


def _add_milestones(ax: plt.Axes, day_min: dt.date, day_max: dt.date) -> None:
    """Draw vertical markers for milestones inside [day_min, day_max].

    Labels run vertically just inside the top of the plot so they never collide
    with the title or each other.
    """
    top = ax.get_ylim()[1]
    for iso, label in config.MILESTONES:
        day = dt.date.fromisoformat(iso)
        if not (day_min <= day <= day_max):
            continue
        ax.axvline(day, color="0.35", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
        ax.text(
            day,
            top * 0.98,
            f" {label}",
            rotation=90,
            fontsize=6.5,
            color="0.3",
            ha="right",
            va="top",
            rotation_mode="anchor",
            zorder=3,
        )


def _style_time_axis(ax: plt.Axes, *, long_range: bool = False) -> None:
    if long_range:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _cumulative(values: list[float]) -> list[float]:
    out, acc = [], 0.0
    for v in values:
        acc += v
        out.append(acc)
    return out


def _pivot_family(rows: list[dict[str, str]], value_col: str) -> tuple[list[dt.date], dict[str, list[float]]]:
    """Pivot the long by-family CSV into per-family series aligned on a shared date axis."""
    days = sorted({dt.date.fromisoformat(r["day"]) for r in rows})
    index = {d: i for i, d in enumerate(days)}
    series: dict[str, list[float]] = {}
    for r in rows:
        fam = r["family"]
        series.setdefault(fam, [0.0] * len(days))[index[dt.date.fromisoformat(r["day"])]] = float(r[value_col])
    ordered = {fam: series[fam] for fam in FAMILY_ORDER if fam in series}
    for fam in series:  # any family not in FAMILY_ORDER, appended at the end
        ordered.setdefault(fam, series[fam])
    return days, ordered


def _stacked_family(
    ax: plt.Axes,
    days: list[dt.date],
    series: dict[str, list[float]],
    peak_days: list[dt.date] | None = None,
    peak_values: list[float] | None = None,
) -> None:
    labels = list(series.keys())
    ax.stackplot(
        days,
        *[series[f] for f in labels],
        labels=labels,
        colors=[FAMILY_COLOR.get(f, "0.6") for f in labels],
        alpha=0.9,
    )
    if peak_values is not None and peak_days is not None:
        ax.plot(peak_days, peak_values, color="0.15", linewidth=1.0, label="peak (all families)")


def chart_accelerators(daily_dir: str, charts_dir: str) -> None:
    fam_rows = _load_csv(os.path.join(daily_dir, "accelerators_by_family_daily.csv"))
    tot_rows = _load_csv(os.path.join(daily_dir, "accelerators_daily.csv"))
    days, series = _pivot_family(fam_rows, "mean_chips")
    tot_days, peak = _dates(tot_rows), _floats(tot_rows, "peak_chips")

    fig, ax = plt.subplots(figsize=(11, 5))
    _stacked_family(ax, days, series, tot_days, peak)
    ax.set_title("Active TPU accelerators (chips) — Iris cluster")
    ax.set_ylabel("chips (mean concurrent / day)")
    _style_time_axis(ax)
    ax.set_ylim(bottom=0)
    _add_milestones(ax, min(days), max(days))
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    _save(fig, charts_dir, "accelerators_chips")


def chart_flops(daily_dir: str, charts_dir: str) -> None:
    fam_rows = _load_csv(os.path.join(daily_dir, "accelerators_by_family_daily.csv"))
    tot_rows = _load_csv(os.path.join(daily_dir, "accelerators_daily.csv"))
    days, series = _pivot_family(fam_rows, "mean_pflops")
    tot_days, peak = _dates(tot_rows), _floats(tot_rows, "peak_pflops")

    fig, ax = plt.subplots(figsize=(11, 5))
    _stacked_family(ax, days, series, tot_days, peak)
    ax.set_title("Peak bf16 compute capacity (PFLOP/s) — Iris cluster")
    ax.set_ylabel("PFLOP/s (mean concurrent / day)")
    _style_time_axis(ax)
    ax.set_ylim(bottom=0)
    _add_milestones(ax, min(days), max(days))
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    _save(fig, charts_dir, "flops_pflops")


def chart_users(daily_dir: str, charts_dir: str) -> None:
    rows = _load_csv(os.path.join(daily_dir, "users_daily.csv"))
    days = _dates(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(days, _floats(rows, "active_users_human"), color="#4C72B0", linewidth=1.8, marker="o", markersize=3)
    ax.set_title("Active users per day (ran >=1 task)")
    ax.set_ylabel("distinct users")
    _style_time_axis(ax)
    ax.set_ylim(bottom=0)
    _add_milestones(ax, min(days), max(days))
    _save(fig, charts_dir, "users")


def chart_tasks(daily_dir: str, charts_dir: str) -> None:
    rows = _load_csv(os.path.join(daily_dir, "accelerators_daily.csv"))
    user_rows = _load_csv(os.path.join(daily_dir, "users_daily.csv"))
    days = _dates(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(days, _floats(rows, "mean_tasks"), color="#55A868", linewidth=1.8, label="mean concurrent")
    ax.plot(days, _floats(rows, "peak_tasks"), color="#C44E52", linewidth=1.0, alpha=0.8, label="peak concurrent")
    ax.set_title("Running tasks — Iris cluster")
    ax.set_ylabel("concurrent tasks")
    _style_time_axis(ax)
    ax.set_ylim(bottom=0)
    _add_milestones(ax, min(days), max(days))

    ax2 = ax.twinx()
    ax2.plot(_dates(user_rows), _floats(user_rows, "active_tasks_ran"), color="0.5", linewidth=1.0, linestyle=":")
    ax2.set_ylabel("distinct tasks run / day (dotted)", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")
    ax2.spines["top"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _save(fig, charts_dir, "tasks")


def chart_regions(daily_dir: str, charts_dir: str) -> None:
    rows = _load_csv(os.path.join(daily_dir, "accelerators_by_region_daily.csv"))
    days = sorted({dt.date.fromisoformat(r["day"]) for r in rows})
    index = {d: i for i, d in enumerate(days)}
    # Order regions by total footprint so the largest sit at the bottom of the stack.
    totals: dict[str, float] = {}
    series: dict[str, list[float]] = {}
    for r in rows:
        region = r["region"] or "(unknown)"
        series.setdefault(region, [0.0] * len(days))[index[dt.date.fromisoformat(r["day"])]] = float(r["mean_chips"])
        totals[region] = totals.get(region, 0.0) + float(r["mean_chips"])
    order = sorted(series, key=lambda reg: totals[reg], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(days, *[series[reg] for reg in order], labels=order, alpha=0.9)
    ax.set_title("Active TPU chips by region — Iris cluster")
    ax.set_ylabel("chips (mean concurrent / day)")
    _style_time_axis(ax)
    ax.set_ylim(bottom=0)
    _add_milestones(ax, min(days), max(days))
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    _save(fig, charts_dir, "accelerators_by_region")


def chart_fleet_utilization(daily_dir: str, charts_dir: str) -> None:
    """Fleet occupancy: provisioned TPU chips running a task vs idle, over time.

    The stacked area is the standing fleet split into busy (>=1 task scheduled)
    and idle chips; the idle band is the "idle accelerator is a waste" gap. The
    dotted line is occupancy %. This is allocation, not efficiency — a busy chip
    may still run at low MFU (that is the calibration chart's job).
    """
    rows = _load_csv(os.path.join(daily_dir, "utilization_daily.csv"))
    if not rows:
        return
    days = _dates(rows)
    busy = _floats(rows, "mean_busy_chips")
    idle = _floats(rows, "mean_idle_chips")
    total = _floats(rows, "mean_total_chips")
    occupancy = [100 * v for v in _floats(rows, "occupancy")]
    overall = 100 * sum(busy) / sum(total) if sum(total) else 0.0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(days, busy, idle, labels=["running a task", "idle"], colors=["#55A868", "#D9D9D9"], alpha=0.95)
    ax.set_ylabel("TPU chips (mean concurrent)")
    ax.set_title(f"Fleet occupancy: {overall:.0f}% of provisioned chip-time was running a task")
    ax.set_ylim(bottom=0)
    _style_time_axis(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _add_milestones(ax, min(days), max(days))

    ax2 = ax.twinx()
    ax2.plot(days, occupancy, color="0.25", linewidth=1.2, linestyle=":", label="occupancy %")
    ax2.set_ylabel("occupancy (%)", color="0.25")
    ax2.tick_params(axis="y", colors="0.25")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    _save(fig, charts_dir, "fleet_utilization")


def chart_preemptible(daily_dir: str, charts_dir: str) -> None:
    """Preemptible (non-v4) vs reserved (v4) TPU capacity over time.

    v4 is the reserved (guaranteed) pool; everything else (v5e/v5p/v6e) is
    preemptible. Built from the real per-device iris data, so it only spans the
    structured window (~May 6 on); W&B carries no device generation, so this
    split cannot be reconstructed for the pre-iris (Jan-Apr) history.
    """
    rows = _load_csv(os.path.join(daily_dir, "accelerators_by_family_daily.csv"))
    if not rows:
        return
    days, by_family = _pivot_family(rows, "mean_pflops")
    reserved = [0.0] * len(days)
    preempt = [0.0] * len(days)
    for family, values in by_family.items():
        target = reserved if family == "v4" else preempt
        for i, v in enumerate(values):
            target[i] += v
    total = sum(preempt) + sum(reserved)
    overall = 100 * sum(preempt) / total if total else 0.0
    share = [100 * p / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(preempt, reserved, strict=True)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(
        days,
        preempt,
        reserved,
        labels=["preemptible (non-v4)", "reserved (v4)"],
        colors=["#4C72B0", "#DD8452"],
        alpha=0.9,
    )
    ax.set_ylabel("TPU capacity (mean PFLOP/s)")
    ax.set_title(f"Preemptible vs reserved capacity - {overall:.0f}% of compute was preemptible (non-v4)")
    ax.set_ylim(bottom=0)
    _style_time_axis(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _add_milestones(ax, min(days), max(days))

    ax2 = ax.twinx()
    ax2.plot(days, share, color="0.25", linewidth=1.2, linestyle=":", label="preemptible share")
    ax2.set_ylabel("preemptible share (%)", color="0.25")
    ax2.tick_params(axis="y", colors="0.25")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    _save(fig, charts_dir, "preemptible")


def chart_compute_history(daily_dir: str, charts_dir: str) -> None:
    """Realized training compute over the full W&B history, back to 2024.

    FLOPs are ``6*N*D`` capped at each run's physical hardware capacity (see
    ``config.REALIZED_FLOPS_CEILING_*``). Top: FLOPs/day (log) with the Iris
    stats window shaded. Bottom: cumulative FLOPs in units of 1e24 (the blog's
    final-run scale).
    """
    rows = _load_csv(os.path.join(daily_dir, "wandb_compute_daily.csv"))
    days = _dates(rows)
    flops = _floats(rows, "realized_flops")
    flops_tpu = _floats(rows, "realized_flops_tpu")
    cumulative = [c / 1e24 for c in _cumulative(flops)]
    iris_start = dt.date.fromisoformat(config.CALIBRATION_START)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_top.plot(days, flops, color="#4C72B0", linewidth=1.2, label="all backends")
    ax_top.plot(days, flops_tpu, color="#C44E52", linewidth=1.0, alpha=0.8, label="TPU only")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("training FLOPs / day (capped 6*N*D)")
    ax_top.set_title("Realized training compute over time - W&B run history")
    ax_top.axvspan(iris_start, max(days), color="0.85", alpha=0.5, zorder=0, label="Iris stats window")
    ax_top.legend(loc="upper left", fontsize=8, frameon=False)
    _add_milestones(ax_top, min(days), max(days))

    ax_bot.fill_between(days, cumulative, color="#55A868", alpha=0.85)
    ax_bot.set_ylabel(r"cumulative FLOPs ($\times 10^{24}$)")
    ax_bot.set_title(rf"Cumulative realized compute: {cumulative[-1]:.2f}$\times 10^{{24}}$ FLOPs to date")

    for ax in (ax_top, ax_bot):
        _style_time_axis(ax, long_range=True)
    ax_bot.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, charts_dir, "compute_history")


def chart_calibration(daily_dir: str, charts_dir: str) -> None:
    """Calibrate W&B realized compute vs iris provisioned capacity (all-on-iris window).

    Primary axis: realized (W&B-logged training) vs provisioned (whole fleet)
    FLOPs/day — the gap between the lines is the un-logged / non-training work.
    Secondary axis: **on-active MFU**, the efficiency of the logged jobs while
    they ran. We deliberately do *not* plot "effective MFU vs total fleet"
    (realized/capacity): it is on-active MFU times coverage, so it craters
    whenever coverage dips even though efficiency is steady.
    """
    rows = _load_csv(os.path.join(daily_dir, "calibration_daily.csv"))
    if not rows:
        return
    days = _dates(rows)
    realized = _floats(rows, "wandb_tpu_realized_flops")
    capacity = _floats(rows, "iris_capacity_flops")
    on_active = [100 * v for v in _floats(rows, "on_active_mfu")]
    # Coverage-weighted headline MFU + mean coverage over the window.
    realized_sum, capacity_sum = sum(realized), sum(capacity)
    devhrs = _floats(rows, "wandb_tpu_device_hours")
    iris_devhrs = _floats(rows, "iris_device_hours")
    cov_pct = 100 * sum(devhrs) / sum(iris_devhrs) if sum(iris_devhrs) else 0.0
    mfu_pct = 100 * realized_sum * sum(iris_devhrs) / (capacity_sum * sum(devhrs)) if sum(devhrs) else 0.0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(days, capacity, color="#8172B3", linewidth=1.6, label="iris provisioned capacity (whole fleet)")
    ax.plot(days, realized, color="#C44E52", linewidth=1.6, label="W&B-logged training (on iris)")
    ax.set_yscale("log")
    ax.set_ylabel("FLOPs / day")
    ax.set_title(
        f"Realized vs provisioned compute - W&B-logged training covered {cov_pct:.0f}% "
        f"of fleet device-hours at {mfu_pct:.0f}% MFU"
    )
    _style_time_axis(ax)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    ax2 = ax.twinx()
    ax2.plot(days, on_active, color="0.4", linewidth=1.2, marker="o", markersize=2.5, label="on-active MFU")
    ax2.set_ylabel("on-active MFU (%)", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")
    ax2.set_ylim(0, 100)
    ax2.spines["top"].set_visible(False)
    _save(fig, charts_dir, "calibration")


def chart_overview(daily_dir: str, charts_dir: str) -> None:
    """One 2x2 figure tying the four series together for an at-a-glance summary."""
    fam_rows = _load_csv(os.path.join(daily_dir, "accelerators_by_family_daily.csv"))
    tot_rows = _load_csv(os.path.join(daily_dir, "accelerators_daily.csv"))
    user_rows = _load_csv(os.path.join(daily_dir, "users_daily.csv"))
    tot_days = _dates(tot_rows)
    day_min, day_max = min(tot_days), max(tot_days)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))

    days, chips = _pivot_family(fam_rows, "mean_chips")
    _stacked_family(axes[0][0], days, chips, tot_days, _floats(tot_rows, "peak_chips"))
    axes[0][0].set_title("Active TPU chips")
    axes[0][0].set_ylabel("chips (mean/day)")
    axes[0][0].legend(loc="upper left", fontsize=7, ncol=3, frameon=False)

    _, pflops = _pivot_family(fam_rows, "mean_pflops")
    _stacked_family(axes[0][1], days, pflops, tot_days, _floats(tot_rows, "peak_pflops"))
    axes[0][1].set_title("Peak bf16 capacity")
    axes[0][1].set_ylabel("PFLOP/s (mean/day)")

    axes[1][0].plot(
        _dates(user_rows), _floats(user_rows, "active_users_human"), color="#4C72B0", marker="o", markersize=3
    )
    axes[1][0].set_title("Active users/day")
    axes[1][0].set_ylabel("distinct users")

    axes[1][1].plot(tot_days, _floats(tot_rows, "mean_tasks"), color="#55A868", label="mean")
    axes[1][1].plot(tot_days, _floats(tot_rows, "peak_tasks"), color="#C44E52", alpha=0.8, linewidth=1.0, label="peak")
    axes[1][1].set_title("Concurrent running tasks")
    axes[1][1].set_ylabel("tasks")
    axes[1][1].legend(loc="upper left", fontsize=7, frameon=False)

    for ax in axes.flat:
        _style_time_axis(ax)
        ax.set_ylim(bottom=0)
        _add_milestones(ax, day_min, day_max)

    fig.suptitle("Iris cluster usage — Marin", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, charts_dir, "overview")


def run(paths: config.Paths) -> None:
    """Render every chart from the daily CSVs."""
    os.makedirs(paths.charts_dir, exist_ok=True)
    chart_accelerators(paths.daily_dir, paths.charts_dir)
    chart_flops(paths.daily_dir, paths.charts_dir)
    chart_users(paths.daily_dir, paths.charts_dir)
    chart_tasks(paths.daily_dir, paths.charts_dir)
    chart_fleet_utilization(paths.daily_dir, paths.charts_dir)
    chart_preemptible(paths.daily_dir, paths.charts_dir)
    chart_regions(paths.daily_dir, paths.charts_dir)
    chart_overview(paths.daily_dir, paths.charts_dir)
    if os.path.exists(os.path.join(paths.daily_dir, "wandb_compute_daily.csv")):
        chart_compute_history(paths.daily_dir, paths.charts_dir)
        chart_calibration(paths.daily_dir, paths.charts_dir)
