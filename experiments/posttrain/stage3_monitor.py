#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor the Stage 3 full-atlas batch; when complete, auto-collect.

Similar event stream pattern to stage4_monitor.py, for the stage3
paired-rubric elicitation batch on all 2573 tension points. When the
batch terminates, runs `stage3_paired_rubrics.py --batch-mode collect`.

Usage (via Monitor tool with persistent=True):
    source .env && uv run python experiments/posttrain/stage3_monitor.py
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

logger = logging.getLogger("stage3_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TERMINAL = {"completed", "failed", "expired", "cancelled"}

BATCH_ID = os.environ.get("STAGE3_BATCH_ID", "batch_69e5dcc3ebe48190ac6fc4a83b99be60")
JOB_DIR = Path(os.environ.get("STAGE3_JOB_DIR", "experiments/posttrain/stage3_output/batch_jobs/rubrics_full"))
RUBRICS_INPUT = Path(os.environ.get("STAGE3_INPUT", "experiments/posttrain/stage3_output/tension_atlas_full_2570.jsonl"))
RUBRICS_OUTPUT = Path(os.environ.get("STAGE3_OUTPUT", "experiments/posttrain/stage3_output/paired_rubrics_full.jsonl"))
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")

POLL_INTERVAL_SECONDS = int(os.environ.get("STAGE3_POLL_INTERVAL", 30 * 60))
MAX_POLLS = int(os.environ.get("STAGE3_MAX_POLLS", 48))


def emit(kind: str, msg: str = "") -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if msg:
        print(f"{now} {kind} {msg}", flush=True)
    else:
        print(f"{now} {kind}", flush=True)


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        emit("COLLECT_ERR", "OPENAI_API_KEY not set")
        return 2

    client = OpenAI()
    prev = None

    emit("STARTED", f"batch={BATCH_ID} poll_every={POLL_INTERVAL_SECONDS}s max_polls={MAX_POLLS}")

    for i in range(1, MAX_POLLS + 1):
        try:
            b = client.batches.retrieve(BATCH_ID)
            status = b.status
            counts = getattr(b, "request_counts", None)
            done = getattr(counts, "completed", None) if counts else None
            failed = getattr(counts, "failed", None) if counts else None
            total = getattr(counts, "total", None) if counts else None
        except Exception as e:
            emit("POLL_ERR", str(e))
            if i < MAX_POLLS:
                time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if status != prev:
            emit("STATE_CHANGE", f"status={status} done={done}/{total} failed={failed}")
            prev = status

        if status in TERMINAL:
            break
        if i < MAX_POLLS:
            time.sleep(POLL_INTERVAL_SECONDS)
    else:
        emit("BUDGET_EXHAUSTED", f"status={prev}")
        return 1

    if status != "completed":
        emit("COLLECT_ERR", f"batch ended status={status}; not collecting")
        return 1

    # Auto-collect.
    emit("COLLECTING")
    cmd = [
        "uv", "run", "python", "experiments/posttrain/stage3_paired_rubrics.py",
        "--batch-mode", "collect",
        "--input", str(RUBRICS_INPUT),
        "--spec", str(SPEC_PATH),
        "--output", str(RUBRICS_OUTPUT),
        "--job-dir", str(JOB_DIR),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
        if p.returncode == 0:
            # Count successful rubrics.
            n = 0
            if RUBRICS_OUTPUT.exists():
                with open(RUBRICS_OUTPUT) as f:
                    for line in f:
                        if line.strip():
                            n += 1
            emit("COLLECT_OK", f"wrote {n} paired-rubric records -> {RUBRICS_OUTPUT}")
        else:
            tail = (p.stdout + p.stderr).splitlines()[-5:]
            emit("COLLECT_ERR", f"rc={p.returncode} {' | '.join(tail)}")
    except subprocess.TimeoutExpired:
        emit("COLLECT_ERR", "collect timeout")

    emit("READY_FOR_STAGE4", "run experiments/posttrain/stage4_oracle_feasibility.py next")
    emit("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
