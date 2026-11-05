# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-of-run summary figures for the 535B-A23B 18T-token hero (issue #8435), pulled from W&B.

Writes five figures to ``hero_summary_figs/``:

* ``hero_train_loss.png`` -- training cross-entropy over the whole run, with the near-dropless EP switch, the
  datamix changes and the context-length changes marked.
* ``hero_vs_ladder_train_loss.png`` -- the same curve against the scaling-ladder rungs, by run progress.
* ``hero_evals_vs_forecast.png`` -- the ladder's dropless Paloma macro evals, the hero forecast from the per-5%
  ladder fits (``plot_scaling_ladder.py``), and the hero's own evals at 4K context.
* ``hero_ptb_4k.png`` -- the Paloma PTB eval at 4K, with the training-stream reads of contaminated PTB text marked
  (issue #9874).
* ``hero_mfu.png`` -- model FLOPs utilization over the run.

The hero spans eleven W&B runs joined at checkpoint handoffs; each run's data is kept from its handoff step up to
the next run's. Evals at 4K are the run's own up to the 16K context switch (it trained at 4K), and after it the
re-evaluation of every 6K checkpoint at 4K (W&B ``hero-evals-at-4k``, same ``eval_dropless/*`` names).

Run, from the repository root (about a minute; each W&B run's pull is cached in ``--cache-dir`` for redraws):

    uv run --with matplotlib python -m experiments.grug.moe_hero_ep.plot_hero_summary --cache-dir /tmp/hero_summary
"""

import argparse
import concurrent.futures
import functools
import os
import pickle
import time

import matplotlib as mpl
import numpy as np
import pandas as pd
import wandb

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.grug.moe_hero_ep.plot_scaling_ladder import (
    COLOR,
    D2048_CORRECTION,
    DROPLESS_MACRO,
    ENTITY_PROJECT,
    RUNS,
    TAG,
    TRAIN_CE,
    _fit_and_hero,
    _footer,
    fetch_wandb,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "hero_summary_figs")
FINAL_STEP = 390251
# 4,194,304 tokens per rack x 11 racks; the batch was resized at each context switch to hold this fixed (it matches
# W&B's throughput/total_tokens at every phase end).
TOKENS_PER_STEP = 46_137_344
# The hero's W&B runs, each with the checkpoint step it resumed from.
HERO_RUNS = (
    ("hero-12d8b6f0-dee637", 0),
    ("hero-wd-gate-router-p02-step58k", 58014),
    ("hero-ragged_a2a-nccl2307-ep-step81k", 81716),
    ("hero-mix-996f4891-step108k", 108000),
    ("hero-nopdl-step108k", 108778),
    ("hero-main-step121638", 121638),
    ("hero-fa4sm100-nomask-step146k", 146139),
    ("hero-mfu30-step231k", 231464),
    ("hero-ctx16k-step257k", 257553),
    ("hero-gradclip-step356k", 356531),
    ("hero-ctx65k-step372k", 372000),
)
EVALS_AT_4K_RUN = "hero-evals-at-4k"
CONTEXT_16K_STEP = 257553
PTB_LOSS = "eval_dropless/paloma/ptb-llama3/loss"
PALOMA_SETS = 16  # the Paloma macro is the plain mean over these, so PTB carries 1/16 of its weight
MFU = "throughput/mfu"

# Run events, by step: (step, label, kind).
EP_SWITCH = (81716, "near-dropless EP\n(ragged all-to-all)", "ep")
DATAMIX = ((108000, "datamix change", "mix"), (312192, "datamix phase 2", "mix"))
CONTEXT = ((CONTEXT_16K_STEP, "4K → 16K context", "ctx"), (372000, "16K → 65K context", "ctx"))
EVENTS = (EP_SWITCH, *DATAMIX, *CONTEXT)
EVENT_COLOR = {"ep": "#6a3d9a", "mix": "#1b7837", "ctx": "#b35806"}

# MFU changes at the hero's cutovers: (step, change, median MFU % from the cutover's paired trial, or None). From the
# Belay app's Hero MFU page (infra/marina/apps/belay/web/mfu.json, "production").
MFU_CHANGES = (
    (58014, "kernel improvements", 21.5),
    (81716, "kernel improvements", 23.1),
    (108778, "kernel improvements", 23.6),
    (121638, "kernel improvements", 24.1),
    (146139, "kernel improvements", 26.8),
    (231464, "kernel improvements", 29.4),
    (257553, "16K context", 30.2),
    (372000, "65K context", None),
)

# Training-stream reads of the torch-nplm document (issue #9874): (first step, last step, carries the eval text).
PTB_READS = (
    (176080, 176355, False),
    (209390, 209669, True),
    (327205, 327493, False),
    (362280, 362578, True),
)
HERO_COLOR = "#0b5d1e"
FORECAST_COLOR = "#6a3d9a"


def _history(run, keys, samples):
    """``keys`` against ``_step``, each fetched alone (W&B returns no rows for a key set the run lacks one of), then
    joined on step; a run without a key has no column for it."""
    frames = []
    for key in keys:
        frame = run.history(keys=["_step", key], samples=samples, pandas=True)
        if len(frame):
            frames.append(frame[["_step", key]].dropna().set_index("_step"))
    if not frames:
        return pd.DataFrame(columns=["_step", *keys])
    return pd.concat(frames, axis=1).reset_index().sort_values("_step")


SCAN_ATTEMPTS = 5
API_TIMEOUT = 180  # seconds; W&B's default is too short for long history scans
SCAN_WINDOW = 2000


SCAN_WORKERS = 8


def _scan_window(run, key, lo, hi):
    """``key`` at every step in ``[lo, hi)``, retried on W&B's busy-server timeout."""
    for attempt in range(1, SCAN_ATTEMPTS + 1):
        try:
            scan = run.scan_history(keys=["_step", key], min_step=lo, max_step=hi, page_size=2000)
            return [(row["_step"], row[key]) for row in scan]
        except wandb.errors.CommError:
            if attempt == SCAN_ATTEMPTS:
                raise
            print(f"W&B timed out scanning {run.name} steps {lo}-{hi}; retrying ({attempt}/{SCAN_ATTEMPTS})")
            time.sleep(30)
    raise AssertionError("unreachable")


def _every_step(run, key, start, end):
    """``key`` at every logged step in ``[start, end)``. ``history`` only samples, so this reads ``scan_history`` in
    ``SCAN_WINDOW``-step windows, ``SCAN_WORKERS`` at a time: one long scan hits W&B's busy-server timeout."""
    windows = [(lo, min(lo + SCAN_WINDOW, end)) for lo in range(start, end, SCAN_WINDOW)]
    with concurrent.futures.ThreadPoolExecutor(SCAN_WORKERS) as pool:
        parts = list(pool.map(lambda w: _scan_window(run, key, *w), windows))
    print(f"{run.name}: {sum(map(len, parts))} steps of {key} in {len(windows)} windows", flush=True)
    rows = [row for part in parts for row in part]
    return pd.DataFrame(rows, columns=["_step", key]).dropna().sort_values("_step")


def fetch_hero_run(api, name, start, end):
    """One hero run's train loss (every step), MFU and, while it trained at 4K, its own dropless evals, each kept to
    the steps it owns: ``[start, end)``, its handoff step up to the next run's."""
    run = api.runs(ENTITY_PROJECT, filters={"display_name": name})[0]
    train = _every_step(run, TRAIN_CE, start, end)
    frame = _history(run, [MFU], samples=20000)
    mfu = frame[(frame["_step"] >= start) & (frame["_step"] < end) & (frame[MFU] > 0)] if MFU in frame else None
    evals = None
    if start < CONTEXT_16K_STEP:  # it trained at 4K, so its own evals are at 4K
        frame = _history(run, [DROPLESS_MACRO, PTB_LOSS], samples=5000)
        evals = frame[(frame["_step"] >= start) & (frame["_step"] < min(end, CONTEXT_16K_STEP))]
    return train, mfu, evals


def _hero_spans():
    """Each hero run's name and the steps it owns."""
    ends = [start for _, start in HERO_RUNS[1:]] + [FINAL_STEP + 1]
    return [(name, start, end) for (name, start), end in zip(HERO_RUNS, ends, strict=True)]


def _cached(cache_dir, key, fetch):
    """``fetch()``'s result, read from ``cache_dir/<key>.pkl`` when present, else fetched and written there."""
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    value = fetch()
    with open(path, "wb") as f:
        pickle.dump(value, f)
    print(f"cached {key}", flush=True)
    return value


def fetch_all(cache_dir):
    """The ladder, the hero's stitched train loss and MFU, and its dropless evals at 4K, each piece cached."""
    api = wandb.Api(timeout=API_TIMEOUT)
    ladder = _cached(cache_dir, "ladder", fetch_wandb)
    train, mfu, evals = [], [], []
    for name, start, end in _hero_spans():
        fetch = functools.partial(fetch_hero_run, api, name, start, end)
        run_train, run_mfu, run_evals = _cached(cache_dir, name, fetch)
        train.append(run_train)
        mfu += [run_mfu] if run_mfu is not None else []
        evals += [run_evals] if run_evals is not None else []
    rerun = _cached(
        cache_dir,
        EVALS_AT_4K_RUN,
        lambda: _history(
            api.runs(ENTITY_PROJECT, filters={"display_name": EVALS_AT_4K_RUN})[0],
            [DROPLESS_MACRO, PTB_LOSS],
            samples=5000,
        ),
    )
    evals.append(rerun)
    return (
        ladder,
        pd.concat(train).drop_duplicates("_step"),
        pd.concat(mfu).drop_duplicates("_step"),
        pd.concat(evals).drop_duplicates("_step"),
    )


def _rolling(series, window):
    return series.rolling(window, center=True, min_periods=1).mean()


EVENT_GRAY = "0.55"


def _mark_events(ax, events, top):
    """Thin gray lines with a label at the top: the swap points, kept behind the data."""
    for step, label, _kind in events:
        ax.axvline(step, color=EVENT_GRAY, ls=(0, (4, 4)), lw=0.8, zorder=0)
        ax.text(step, top, f" {label}", rotation=90, va="top", ha="right", fontsize=8, color=EVENT_GRAY)


FULL_RESOLUTION_NOTE = "Every training step plotted, no smoothing"


def _note_full_resolution(ax, y=0.02):
    """The full-resolution note in the lower left, at height ``y`` (axes fraction)."""
    ax.text(
        0.01,
        y,
        FULL_RESOLUTION_NOTE,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="0.35",
    )


def _annotate_under_curve(ax, x, y_curve, label):
    """A short tick under the curve at ``(x, y_curve)`` and a small label beneath it, so the event reads with the
    change it caused."""
    ax.plot([x, x], [y_curve - 0.03, y_curve - 0.08], color=EVENT_GRAY, lw=1)
    ax.text(x, y_curve - 0.09, label, ha="center", va="top", fontsize=7.5, color=EVENT_GRAY, linespacing=1.1)


def _annotate_events(ax, events):
    """A short tick under the top edge and a small label per swap point, instead of a full-height line."""
    for x, label in events:
        ax.plot([x, x], [1.0, 0.975], color=EVENT_GRAY, lw=1, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(
            x,
            0.97,
            label,
            rotation=90,
            va="top",
            ha="center",
            fontsize=7.5,
            color=EVENT_GRAY,
            transform=ax.get_xaxis_transform(),
        )


def _shade_long_context(ax, x0, x1, label, y=0.10):
    """Lightly shade ``[x0, x1]`` and label it at height ``y`` (axes fraction), clear of the curves."""
    ax.axvspan(x0, x1, color="0.5", alpha=0.08, lw=0)
    ax.text(x0, y, f"  {label}", transform=ax.get_xaxis_transform(), va="bottom", fontsize=8.5, color="0.35")


def plot_train_loss(train):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    tokens_t = train["_step"] * TOKENS_PER_STEP / 1e12
    ax.plot(tokens_t, train[TRAIN_CE], color=HERO_COLOR, lw=0.35, label="hero train loss")
    _note_full_resolution(ax)
    lo, hi = train[TRAIN_CE].min() - 0.02, 1.8
    ax.set_ylim(lo, hi)
    _mark_events(ax, [(step * TOKENS_PER_STEP / 1e12, label, kind) for step, label, kind in EVENTS], hi - 0.01)
    ax.set_xlim(0, FINAL_STEP * TOKENS_PER_STEP / 1e12)  # 18.005T, so the 18T tick falls inside
    ax.set_xticks(range(0, 19, 3))
    ax.set_xlabel("tokens trained (T)")
    ax.set_ylabel("training cross-entropy loss")
    ax.set_title("535B-A23B 18T tok hero: training loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="center right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_train_loss.png"), dpi=130)


def plot_vs_ladder(train, ladder):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for r in RUNS:
        d = ladder[r]
        ax.plot(d["ce_step"] / d["nts"] * 100, d["ce"], "-", color=COLOR[r], lw=0.55, alpha=0.85, label=TAG[r])
    progress = train["_step"] / FINAL_STEP * 100
    ax.plot(progress, train[TRAIN_CE], color=HERO_COLOR, lw=0.35, label=f"hero · {TAG['d6144']}")
    _note_full_resolution(ax)
    for step, label in (
        (DATAMIX[0][0], "datamix\nchange"),
        (CONTEXT[0][0], "4K → 16K\ncontext"),
        (DATAMIX[1][0], "hero: phase-2 mix B\n(loss down)"),
        (CONTEXT[1][0], "16K → 65K\ncontext"),
    ):
        # The curve's low edge around the event (its 1st percentile over 2,000 steps either side), so a drop at
        # the event does not run the tick through the noise after it.
        near = train[(train["_step"] >= step - 2000) & (train["_step"] < step + 2000)][TRAIN_CE]
        _annotate_under_curve(ax, step / FINAL_STEP * 100, float(near.quantile(0.01)), label)
    # The d2048 rung stopped short of 100%: say why at its last point.
    d2048 = ladder["d2048"]
    end_x, end_y = d2048["ce_step"][-1] / d2048["nts"] * 100, float(np.median(d2048["ce"][-200:]))
    ax.annotate(
        "crashed at 81% (hardware);\nnot resumed",
        xy=(end_x, end_y),
        xytext=(8, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
        color=COLOR["d2048"],
        linespacing=1.1,
    )
    # The rungs switch to their own phase-2 mix at 80% too, with different weights: their loss steps up there.
    top_rung = ladder["d768"]
    near = top_rung["ce"][
        (top_rung["ce_step"] / top_rung["nts"] >= 0.79) & (top_rung["ce_step"] / top_rung["nts"] < 0.81)
    ]
    ax.plot([80, 80], [np.quantile(near, 0.99) + 0.03, np.quantile(near, 0.99) + 0.08], color=EVENT_GRAY, lw=1)
    ax.text(
        80,
        np.quantile(near, 0.99) + 0.09,
        "ladder: phase-2 mix A\n(loss up)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=EVENT_GRAY,
        linespacing=1.1,
    )
    ax.set_ylim(round(train[TRAIN_CE].min() - 0.25, 2), 3.0)  # room under the hero curve for its event labels
    ax.set_xlim(0, 100)
    ax.set_xlabel("run progress (%)")
    ax.set_ylabel("training cross-entropy loss")
    ax.set_title("535B-A23B 18T tok hero against its scaling ladder")
    ax.grid(alpha=0.25)
    ax.set_xticks(range(0, 101, 10))
    for line in ax.legend(fontsize=9, loc="upper right").get_lines():
        line.set_linewidth(2.5)
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_vs_ladder_train_loss.png"), dpi=130)


def plot_evals_vs_forecast(evals, ladder):
    grid, _fits, hero, slope, intercept, d2048_max, *_ = _fit_and_hero(ladder)
    fig, ax = plt.subplots(figsize=(12, 7))
    for r in ["d768", "d1024", "d1536"]:
        gs = sorted(ladder[r]["grid"])
        ax.plot(
            [g * 100 for g in gs], [ladder[r]["grid"][g] for g in gs], "-o", color=COLOR[r], lw=1.6, ms=4, label=TAG[r]
        )
    g2048 = ladder["d2048"]["grid"]
    measured = [g for g in sorted(g2048) if g <= d2048_max]
    ax.plot(
        [g * 100 for g in measured],
        [g2048[g] for g in measured],
        "-o",
        color=COLOR["d2048"],
        lw=1.6,
        ms=4,
        label=TAG["d2048"],
    )
    ext = [d2048_max, *(g for g in grid if g > d2048_max)]
    ax.plot(
        [g * 100 for g in ext],
        [g2048[d2048_max], *(slope * g + intercept + D2048_CORRECTION for g in ext[1:])],
        "--o",
        color=COLOR["d2048"],
        lw=1.6,
        ms=6,
        mfc="white",
        mec=COLOR["d2048"],
        label="d2048 extrapolated (pinned)",
    )
    ax.plot([g * 100 for g in grid], [hero[g][1] for g in grid], "--", color=FORECAST_COLOR, lw=2.2, zorder=5)
    ax.plot(
        [g * 100 for g in grid],
        [hero[g][1] for g in grid],
        ls="none",
        marker="*",
        ms=13,
        mfc="white",
        mec=FORECAST_COLOR,
        mew=1.6,
        zorder=6,
        label="hero forecast (per-5% ladder fits)",
    )
    macro = evals.dropna(subset=[DROPLESS_MACRO])
    progress = macro["_step"] / FINAL_STEP * 100
    actual = macro[DROPLESS_MACRO]
    ax.plot(progress, actual, "-", color=HERO_COLOR, lw=1.4, zorder=7, label="hero actual (eval at 4K)")
    final_x, final, forecast = float(progress.iloc[-1]), float(actual.iloc[-1]), hero[1.0][1]
    # Fill the forecast's 100% star: the forecast number labels it.
    ax.plot(100, forecast, ls="none", marker="*", ms=16, mfc=FORECAST_COLOR, mec=FORECAST_COLOR, mew=1.6, zorder=7)
    ax.plot(final_x, final, "o", color=HERO_COLOR, ms=11, mec="white", mew=1.5, zorder=8)
    # The two numbers the chart is about: the forecast's final point and the run's final value.
    ax.annotate(
        f"{forecast:.3f}",
        xy=(100, forecast),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=FORECAST_COLOR,
        zorder=8,
    )
    ax.annotate(
        f"{final:.3f}",
        xy=(final_x, final),
        xytext=(-10, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=HERO_COLOR,
        zorder=8,
    )
    label = (
        "hero extended to 16K, then 65K context\ntraining with an improved datamix,\n"
        "giving a boost over the scaling ladder"
    )
    _shade_long_context(ax, CONTEXT_16K_STEP / FINAL_STEP * 100, 100, label, y=0.19)  # just above the hero curve
    ax.set_xlim(2, 103)
    ax.set_ylim(1.85, 4.0)  # the first evals, near step 0, sit far above; room below for the final label
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel("dropless Paloma macro loss (eval at 4K)")
    ax.set_title("535B-A23B 18T tok hero: forecast and actual, every eval at 4K context")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_evals_vs_forecast.png"), dpi=130)
    return hero[1.0][1], float(actual.iloc[-1])


def plot_ptb(evals):
    fig, ax = plt.subplots(figsize=(12, 6))
    ptb = evals.dropna(subset=[PTB_LOSS, DROPLESS_MACRO])
    final_ptb, final_macro = float(ptb[PTB_LOSS].iloc[-1]), float(ptb[DROPLESS_MACRO].iloc[-1])
    share = final_ptb / PALOMA_SETS / final_macro
    ax.text(
        0.01,
        0.12,
        f"PTB is 1 of {PALOMA_SETS} sets in the Paloma macro (a plain mean), so it is {share:.1%} of the final macro:\n"
        f"{final_ptb:.3f} / {PALOMA_SETS} of {final_macro:.3f}. A 0.30 drop in PTB moves the macro by "
        f"{0.30 / PALOMA_SETS:.3f}.",
        transform=ax.transAxes,
        fontsize=9,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.8"},
    )
    ax.plot(ptb["_step"], ptb[PTB_LOSS], "-o", color="#2b5c8a", lw=1.3, ms=3, label="Paloma PTB loss at 4K (dropless)")
    for first, last, eval_text in PTB_READS:
        color, alpha = ("#c0392b", 0.35) if eval_text else ("#e67e22", 0.18)
        ax.axvspan(first, last + 600, color=color, alpha=alpha, lw=0)
    ax.plot([], [], color="#c0392b", alpha=0.6, lw=6, label="trained on the PTB eval text (valid split)")
    ax.plot([], [], color="#e67e22", alpha=0.4, lw=6, label="trained on PTB test-split text only")
    _shade_long_context(ax, CONTEXT_16K_STEP, FINAL_STEP, "hero at 16K, then 65K; evals re-run at 4K")
    ax.set_xlim(0, FINAL_STEP)
    ax.set_ylim(1.4, 2.6)  # the first evals, near step 0, sit far above
    ax.set_xlabel("step")
    ax.set_ylabel("Paloma PTB loss")
    ax.set_title(
        f"Paloma PTB (1/{PALOMA_SETS} of the Paloma macro) at 4K context, with the contaminated training reads (#9874)"
    )
    ax.text(
        0.01,
        0.02,
        "Reads mapped from the 4K data stream; after the 16K switch the stream was repacked, so the 362K read is "
        "approximate. Bands widened for visibility.",
        transform=ax.transAxes,
        fontsize=8.5,
        style="italic",
        color="0.35",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_ptb_4k.png"), dpi=130)


def plot_mfu(mfu):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mfu["_step"], mfu[MFU], color="#4a6fa5", lw=0.4, alpha=0.3)  # W&B logs MFU in percent
    ax.plot(mfu["_step"], _rolling(mfu[MFU], 50), color="#4a6fa5", lw=1.4, label="MFU (smoothed)")
    top = float(np.nanpercentile(mfu[MFU], 99.5)) * 1.5  # headroom for the change labels above the curve
    ax.set_ylim(0, top)
    _annotate_events(
        ax, [(step, change + (f" · {median:.1f}%" if median else "")) for step, change, median in MFU_CHANGES]
    )
    ax.set_xlim(0, FINAL_STEP)
    ax.set_xlabel("step")
    ax.set_ylabel("model FLOPs utilization (%)")
    ax.set_title("535B-A23B 18T tok hero: MFU over the run (11 GB200 NVL72 racks)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    _footer(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, "hero_mfu.png"), dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True, help="one pickle per W&B run pulled; present ones are reused")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    ladder, train, mfu, evals = fetch_all(args.cache_dir)
    plot_train_loss(train)
    plot_vs_ladder(train, ladder)
    forecast, actual = plot_evals_vs_forecast(evals, ladder)
    plot_ptb(evals)
    plot_mfu(mfu)
    print(f"wrote 5 figures to {FIG_DIR}; forecast {forecast:.4f}, actual at 4K {actual:.4f}")


if __name__ == "__main__":
    main()
