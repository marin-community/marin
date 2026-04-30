#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dual-batch monitor for the full-atlas N=3 BCG pipeline (M0 + M1).

Polls 2 gpt-5.1 score batches every 15 minutes. When each reaches a terminal
state, auto-runs score-collect and compute for that job root. When BOTH are
done and computed, subsamples oracle scores to N=3 (for apples-to-apples
comparison), recomputes oracle bcg_summary, then regenerates the scatter
comparison and radar plots at full scale.

Each stdout line is one event for the Monitor tool.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("stage4_full_dual_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TERMINAL = {"completed", "failed", "expired", "cancelled"}

BATCHES = {
    "M0_full": os.environ.get(
        "BCG_FULL_M0_BATCH", "batch_69e6563e060c8190ba474d9cc6717cb0"
    ),
    "M1_full": os.environ.get(
        "BCG_FULL_M1_BATCH", "batch_69e6564cdbb481908f61bb597445292c"
    ),
}
JOB_ROOTS = {
    "M0_full": Path("experiments/posttrain/stage4_output/bcg_M0_full"),
    "M1_full": Path("experiments/posttrain/stage4_output/bcg_M1_full"),
}
RUBRICS = Path("experiments/posttrain/stage3_output/paired_rubrics_full.jsonl")
ORACLE_JOB_ROOT = Path("experiments/posttrain/stage4_output/full_oracle")
ORACLE_N3_JOB_ROOT = Path("experiments/posttrain/stage4_output/full_oracle_n3")
STAGE4_OUTPUT_DIR = Path("experiments/posttrain/stage4_output")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 15 * 60))
MAX_POLLS = int(os.environ.get("MAX_POLLS", 96))  # 24h at 15-min interval


def emit(kind: str, msg: str = "") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {kind} {msg}".rstrip(), flush=True)


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        tail = (p.stdout + p.stderr).splitlines()[-10:]
        return p.returncode, "\n".join(tail)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def collect_and_compute(key: str, client: OpenAI) -> bool:
    emit(f"{key}_COLLECTING")
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "score-collect",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOTS[key]),
    ])
    if rc != 0:
        emit(f"{key}_COLLECT_ERR", tail)
        return False
    emit(f"{key}_COLLECT_OK")

    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "compute",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOTS[key]),
        "--threshold", "7",
    ])
    if rc != 0:
        emit(f"{key}_COMPUTE_ERR", tail)
        return False

    # Summary line
    summary_path = JOB_ROOTS[key] / "bcg_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        agg = s.get("aggregate", {})
        emit(
            f"{key}_COMPUTED",
            f"n={agg.get('n_tension_points')} BCG={agg.get('mean_bcg')} "
            f"joint={agg.get('mean_joint_satisfaction')} "
            f"A={agg.get('mean_marginal_A')} B={agg.get('mean_marginal_B')}",
        )
    return True


def subsample_oracle_to_n3() -> bool:
    """Step 5: copy oracle's scores to full_oracle_n3, drop sample_idx=3, recompute."""
    emit("ORACLE_SUBSAMPLE_START")
    ORACLE_N3_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    src_scores = ORACLE_JOB_ROOT / "scores.jsonl"
    dst_scores = ORACLE_N3_JOB_ROOT / "scores.jsonl"
    if not src_scores.exists():
        emit("ORACLE_SUBSAMPLE_ERR", f"{src_scores} missing")
        return False

    kept = 0
    dropped = 0
    with open(src_scores) as src, open(dst_scores, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["sample_idx"] < 3:
                dst.write(line + "\n")
                kept += 1
            else:
                dropped += 1

    # Also copy generations so compute can read them (it doesn't actually need
    # them since scores.jsonl is what matters, but keep the dir consistent).
    src_gens = ORACLE_JOB_ROOT / "generations.jsonl"
    if src_gens.exists():
        dst_gens = ORACLE_N3_JOB_ROOT / "generations.jsonl"
        with open(src_gens) as src, open(dst_gens, "w") as dst:
            for line in src:
                r = json.loads(line.strip()) if line.strip() else None
                if r is None:
                    continue
                if r["sample_idx"] < 3:
                    dst.write(json.dumps(r) + "\n")

    emit("ORACLE_SUBSAMPLE_OK", f"kept={kept} dropped={dropped}")

    # Recompute oracle BCG on the N=3 scores.
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "compute",
        "--rubrics", str(RUBRICS),
        "--job-root", str(ORACLE_N3_JOB_ROOT),
        "--threshold", "7",
    ])
    if rc != 0:
        emit("ORACLE_N3_COMPUTE_ERR", tail)
        return False
    summary_path = ORACLE_N3_JOB_ROOT / "bcg_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        agg = s.get("aggregate", {})
        emit(
            "ORACLE_N3_COMPUTED",
            f"n={agg.get('n_tension_points')} BCG={agg.get('mean_bcg')} "
            f"joint={agg.get('mean_joint_satisfaction')}",
        )
    return True


def regenerate_plots() -> bool:
    """Step 6: run stage4_compare.py + plot_bcg_radar.py on full-atlas data.

    Both scripts point at `stage4_output/{bcg_M0,bcg_M1,bcg_gpt51}` by default.
    We need them to point at the full-atlas dirs instead. The simplest thing
    that works: symlink the full-atlas job roots to the paths the scripts
    expect, regenerate, then point back. But cleaner: patch the scripts to
    accept alternate paths, or write a small wrapper.

    Simpler still: write a minimal comparison+radar wrapper here that reads
    the three full-atlas bcg_summary.json files and produces
    comparison_full.{md,png} + comparison_radar_full.png.
    """
    emit("REGEN_PLOTS_START")

    # Approach: write a tiny inline script that calls the plotting scripts
    # with env overrides. Since stage4_compare.py and plot_bcg_radar.py have
    # MODELS hardcoded, easiest is to run them with env MODELS_OVERRIDE — but
    # they don't support that. Simplest: symlink the full dirs into the paths
    # the plotters expect, run, then remove symlinks.
    #
    # Even simpler: just regenerate locally via embedded comparison logic.
    rc, tail = run([
        "uv", "run", "--with", "matplotlib", "python",
        "experiments/posttrain/stage4_full_plots.py",
    ])
    if rc != 0:
        emit("REGEN_PLOTS_ERR", tail)
        return False
    emit("REGEN_PLOTS_OK", "wrote comparison_full.* + comparison_radar_full.png")
    return True


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        emit("ERR", "OPENAI_API_KEY not set")
        return 2

    client = OpenAI()
    emit("STARTED", f"poll_every={POLL_INTERVAL}s max_polls={MAX_POLLS}")
    for k, bid in BATCHES.items():
        emit("TRACKING", f"{k}={bid}")

    prev: dict[str, str | None] = {k: None for k in BATCHES}
    final: dict[str, str] = {}
    done_collect: set[str] = set()

    for i in range(1, MAX_POLLS + 1):
        current: dict[str, str] = {}
        for k, bid in BATCHES.items():
            try:
                b = client.batches.retrieve(bid)
                current[k] = b.status
                counts = getattr(b, "request_counts", None)
                done = getattr(counts, "completed", None) if counts else None
                total = getattr(counts, "total", None) if counts else None
                failed = getattr(counts, "failed", None) if counts else None
                if current[k] != prev[k]:
                    emit(
                        f"{k}_STATE",
                        f"status={current[k]} done={done}/{total} failed={failed}",
                    )
                    prev[k] = current[k]
            except Exception as e:
                emit(f"{k}_POLL_ERR", str(e))

        # Opportunistically collect any that just went terminal
        for k, st in current.items():
            if st == "completed" and k not in done_collect:
                if collect_and_compute(k, client):
                    done_collect.add(k)
                    final[k] = "completed"
            elif st in TERMINAL and k not in done_collect and st != "completed":
                emit(f"{k}_TERMINAL_BAD", f"status={st}; not collecting")
                done_collect.add(k)
                final[k] = st

        if len(done_collect) == len(BATCHES):
            emit("ALL_COLLECTED")
            break

        if i < MAX_POLLS:
            time.sleep(POLL_INTERVAL)
    else:
        emit("BUDGET_EXHAUSTED", f"done={list(done_collect)}")
        return 1

    if all(final.get(k) == "completed" for k in BATCHES):
        emit("PHASE_5_ORACLE_SUBSAMPLE")
        subsample_oracle_to_n3()

        emit("PHASE_6_REGEN_PLOTS")
        regenerate_plots()

        emit("FULL_PIPELINE_DONE")
    else:
        emit("FATAL", f"not all batches completed cleanly: {final}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
