#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chained monitor for the full-atlas oracle feasibility filter.

Phase 1: wait for generate batch to terminate.
Phase 2: run generate-collect.
Phase 3: submit score batch.
Phase 4: wait for score batch to terminate.
Phase 5: run score-collect.
Phase 6: run compute.

Env vars:
    STAGE4_FULL_ORACLE_GEN_BATCH_ID   (submitted above)
    STAGE4_FULL_ORACLE_JOB_ROOT       (default: stage4_output/full_oracle)
    STAGE4_FULL_RUBRICS               (default: stage3_output/paired_rubrics_full.jsonl)
    STAGE4_POLL_INTERVAL              (seconds; default 1800 = 30 min)
    STAGE4_MAX_POLLS                  (default 48 = 24h)

Emits one stdout line per event for the Monitor tool to display.
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

logger = logging.getLogger("stage4_oracle_chain_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TERMINAL = {"completed", "failed", "expired", "cancelled"}

GEN_BATCH_ID = os.environ.get(
    "STAGE4_FULL_ORACLE_GEN_BATCH_ID",
    "batch_69e5e4820b348190b53726d387177cd7",
)
JOB_ROOT = Path(os.environ.get(
    "STAGE4_FULL_ORACLE_JOB_ROOT",
    "experiments/posttrain/stage4_output/full_oracle",
))
RUBRICS = Path(os.environ.get(
    "STAGE4_FULL_RUBRICS",
    "experiments/posttrain/stage3_output/paired_rubrics_full.jsonl",
))

POLL_INTERVAL = int(os.environ.get("STAGE4_POLL_INTERVAL", 30 * 60))
MAX_POLLS = int(os.environ.get("STAGE4_MAX_POLLS", 48))


def emit(kind: str, msg: str = "") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} {kind} {msg}".rstrip(), flush=True)


def run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        tail = (p.stdout + p.stderr).splitlines()[-10:]
        return p.returncode, "\n".join(tail)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def wait_for_batch(client: OpenAI, batch_id: str, phase: str) -> str:
    prev = None
    for i in range(1, MAX_POLLS + 1):
        try:
            b = client.batches.retrieve(batch_id)
            status = b.status
            counts = getattr(b, "request_counts", None)
            done = getattr(counts, "completed", None) if counts else None
            failed = getattr(counts, "failed", None) if counts else None
            total = getattr(counts, "total", None) if counts else None
        except Exception as e:
            emit(f"{phase}_POLL_ERR", str(e))
            if i < MAX_POLLS:
                time.sleep(POLL_INTERVAL)
            continue
        if status != prev:
            emit(f"{phase}_STATE", f"status={status} done={done}/{total} failed={failed}")
            prev = status
        if status in TERMINAL:
            return status
        if i < MAX_POLLS:
            time.sleep(POLL_INTERVAL)
    emit(f"{phase}_BUDGET_EXHAUSTED", f"status={prev}")
    return prev or "unknown"


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        emit("ERR", "OPENAI_API_KEY not set")
        return 2

    client = OpenAI()
    emit("STARTED", f"gen_batch={GEN_BATCH_ID} job_root={JOB_ROOT}")

    # Phase 1: wait for generate
    emit("PHASE", "1/6 wait-generate")
    status = wait_for_batch(client, GEN_BATCH_ID, "GEN")
    if status != "completed":
        emit("FATAL", f"generate ended status={status}")
        return 1

    # Phase 2: collect generations
    emit("PHASE", "2/6 generate-collect")
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "generate-collect",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOT),
    ])
    if rc != 0:
        emit("FATAL", f"generate-collect rc={rc}\n{tail}")
        return 1
    emit("GEN_COLLECT_OK")

    # Phase 3: submit score batch
    emit("PHASE", "3/6 score-submit")
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "score-submit",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOT),
        "--judge-model", "gpt-5.1",
    ])
    if rc != 0:
        emit("FATAL", f"score-submit rc={rc}\n{tail}")
        return 1

    # Find the score batch id from the job state.
    score_state_path = JOB_ROOT / "score" / "batch_state.json"
    if not score_state_path.exists():
        emit("FATAL", f"score batch_state.json missing at {score_state_path}")
        return 1
    state = json.loads(score_state_path.read_text())
    score_batch_id = state.get("batch_id")
    if not score_batch_id:
        emit("FATAL", "could not read score batch_id from state")
        return 1
    emit("SCORE_SUBMITTED", f"batch={score_batch_id}")

    # Phase 4: wait for score
    emit("PHASE", "4/6 wait-score")
    status = wait_for_batch(client, score_batch_id, "SCORE")
    if status != "completed":
        emit("FATAL", f"score ended status={status}")
        return 1

    # Phase 5: score-collect
    emit("PHASE", "5/6 score-collect")
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "score-collect",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOT),
    ])
    if rc != 0:
        emit("FATAL", f"score-collect rc={rc}\n{tail}")
        return 1
    emit("SCORE_COLLECT_OK")

    # Phase 6: compute
    emit("PHASE", "6/6 compute")
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "compute",
        "--rubrics", str(RUBRICS),
        "--job-root", str(JOB_ROOT),
        "--threshold", "7",
    ])
    if rc != 0:
        emit("FATAL", f"compute rc={rc}\n{tail}")
        return 1

    # Read and summarize
    summary_path = JOB_ROOT / "bcg_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        agg = s.get("aggregate") or {}
        emit(
            "RESULT",
            f"oracle n={agg.get('n_tension_points')} "
            f"BCG={agg.get('mean_bcg')} joint={agg.get('mean_joint_satisfaction')} "
            f"A={agg.get('mean_marginal_A')} B={agg.get('mean_marginal_B')}",
        )

    emit("ALL_DONE", str(summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
