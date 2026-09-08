# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render audited Snowball weight-sync timing and token-counter evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_report(inputs: Path, output: Path) -> dict:
    filenames = [
        "snowball-durable-audit.json",
        "snowball-history.json",
        "snowball-finelog-audit.json",
        "snowball-token-summary.json",
        "snowball-age-raw.json",
        "snowball-eval-intervals.json",
        "snowball-task-cost.json",
    ]
    source = {name: json.loads((inputs / name).read_text()) for name in filenames}
    audit = source["snowball-durable-audit.json"]
    assert audit["clean_end_to_end"]
    history = [row for row in source["snowball-history.json"]["rows"] if "trainer/global_step" in row]
    assert [row["trainer/global_step"] for row in history] == list(range(1, 6))
    trace = source["snowball-finelog-audit.json"]
    token = source["snowball-token-summary.json"]
    ages = source["snowball-age-raw.json"]["summary"]
    assert ages["group_count"] == 160 and ages["coverage"] == 1.0
    assert len(trace["publication_stage_checks"]) == 25
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    color = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, 5))
    whole = [
        ("Pause", "timing/weight_pause"),
        ("Weight broadcast", "timing/weight_broadcast"),
        ("Whole weight sync", "timing/sync_weights"),
        ("Derived stall proxy", "timing/publication_stall_seconds"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for index, (_label, key) in enumerate(whole):
        values = [row[key] for row in history]
        ax.scatter(values, np.full(5, index) + np.linspace(-0.12, 0.12, 5), c=color, s=48, zorder=3)
        ax.plot([np.median(values)] * 2, [index - 0.22, index + 0.22], color="black", lw=2)
    ax.set(yticks=range(4), yticklabels=[x[0] for x in whole], xlabel="Wall seconds per sync", xlim=(0, 22))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Snowball baseline: five post-update weight syncs")
    fig.supxlabel(
        "One point per sync; black tick = median. Fixed five-second pause retained.\n"
        "Nested spans overlap; do not add them. Stall is a derived wall-time proxy, not GPU idle time.",
        fontsize=9,
    )
    fig.savefig(output / "whole-sync.png", dpi=180)
    fig.savefig(output / "whole-sync.svg")
    plt.close(fig)

    stages = ["export", "nccl_send", "rpc_wait", "barrier", "reload_finalize", "recv", "load", "finalize"]
    gpu = {}
    for event in trace["stages"]:
        attributes = json.loads(event["attributes_json"])
        body = json.loads(event["body_json"])
        step = int(attributes["step"])
        if step > 0:
            key = (attributes["stage"], step)
            gpu[key] = max(gpu.get(key, 0), body["gpu_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout="constrained")
    for index, stage in enumerate(stages):
        wall = [row["timing/weight_broadcast/" + stage] for row in history]
        axes[0].scatter(wall, np.full(5, index) + np.linspace(-0.12, 0.12, 5), c=color, s=35)
        if all((stage, step) in gpu for step in range(1, 6)):
            axes[1].scatter(
                [gpu[stage, step] for step in range(1, 6)],
                np.full(5, index) + np.linspace(-0.12, 0.12, 5),
                c=color,
                s=35,
            )
        else:
            axes[1].text(
                0.04,
                index,
                "Receiver GPU receipts missing",
                transform=axes[1].get_yaxis_transform(),
                va="center",
                color="#a43d32",
            )
    axes[0].set(yticks=range(len(stages)), yticklabels=stages, xlabel="Wall seconds (log scale)", xscale="log")
    axes[1].set(xlabel="CUDA-event milliseconds (log scale)", xscale="log")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Stage distributions: maximum rank or receiver per sync")
    fig.supxlabel(
        "Five observations per stage. Export/barrier fold 32 ranks; other learner stages use rank zero.\n"
        "Receiver walls are W&B folds. CUDA and host spans overlap and measure different waits; neither is additive.",
        fontsize=9,
    )
    fig.savefig(output / "sync-stages.png", dpi=180)
    fig.savefig(output / "sync-stages.svg")
    plt.close(fig)

    intervals = token["intervals"]
    begin = min(row["start_ms"] for rows in intervals.values() for row in rows)
    end = max(row["end_ms"] for rows in intervals.values() for row in rows)
    series = []
    for start in range(int(begin // 1000) * 1000, int(end), 1000):
        total, coverage = 0.0, []
        for rows in intervals.values():
            covered = 0.0
            for row in rows:
                overlap = max(0, min(start + 1000, row["end_ms"]) - max(start, row["start_ms"]))
                total += overlap / 1000 * row["tokens_per_second"]
                covered += overlap
            coverage.append(covered)
        series.append({"timestamp_ms": start + 500, "tokens_per_second": total if min(coverage) == 1000 else None})
    windows = sorted(token["windows"], key=lambda row: row["step"])
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharey=True, layout="constrained")
    for ax, window in zip(axes, windows, strict=True):
        anchor = window["start_ms"]
        duration = (window["end_ms"] - anchor) / 1000
        selected = [row for row in series if anchor - 60000 <= row["timestamp_ms"] <= window["end_ms"] + 60000]
        ax.plot(
            [(row["timestamp_ms"] - anchor) / 1000 for row in selected],
            [row["tokens_per_second"] if row["tokens_per_second"] is not None else np.nan for row in selected],
            lw=1.3,
            color="#1968a6",
        )
        for other in windows:
            ax.axvspan(
                (other["start_ms"] - anchor) / 1000, (other["end_ms"] - anchor) / 1000, color="#d98550", alpha=0.2
            )
        ax.axvspan(0, duration, color="#d98550", alpha=0.3)
        if end < window["end_ms"] + 60000:
            ax.axvspan((end - anchor) / 1000, duration + 60, color="gray", alpha=0.3)
            ax.text(0.98, 0.8, "Missing final coverage", transform=ax.transAxes, ha="right", fontsize=9)
        ax.set(xlim=(-60, duration + 60), ylabel="Tokens/s", title=f"Sync after update {window['step']}")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Seconds relative to target sync start")
    fig.suptitle("Eight-engine token counters around weight sync (±60 seconds)")
    fig.supxlabel(
        "Rates derive from counter differences over actual intervals, averaged into one-second bins.\n"
        "Orange: sync; gray: unavailable. Windows overlap other syncs/evaluation and are not independent comparisons.",
        fontsize=9,
    )
    fig.savefig(output / "token-windows.png", dpi=180)
    fig.savefig(output / "token-windows.svg")
    plt.close(fig)
    report = {
        "task_cost": source["snowball-task-cost.json"],
        "training_source": {
            "marin": "d0601b1ccebfb22f6ea14f99ec8d55f144a86509",
            "marinskyrl": "cf4b0d8cf92d97f9354bd374a2df765555a9dd31",
        },
        "input_sha256": {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest() for name in filenames},
        "whole_sync_rows": history,
        "stage_summary": trace["stage_summary"],
        "age_summary": ages,
        "token_windows": windows,
        "token_rate_one_second_bins": series,
        "lag_summary": trace["lag_summary"],
        "lag_violations": trace["lag_violations"],
        "eval_intervals": source["snowball-eval-intervals.json"],
        "limitations": [
            "Receiver GPU receipts missing",
            "Stage timings overlap and are non-additive",
            "Five within-run syncs, no effect confidence interval",
            "Adjacent token windows overlap; final after-window incomplete",
            "Strict maximum-lag gate failed",
        ],
    }
    (output / "weight-sync-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("E3_REPORT_ARTIFACT_PASS five_syncs=5 native_stage_crosschecks=25 age_groups=160 plots=3")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.inputs, args.output)
