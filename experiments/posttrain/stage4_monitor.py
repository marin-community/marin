#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor 3 OpenAI batch jobs (M0, M1, oracle scoring), auto-collect + compute + compare when all terminal.

Runs as a foreground process; emits one stdout line per event (suitable for
the Monitor tool). Event types emitted to stdout:

    STATE_CHANGE  <key>=<status>   (one per state change since last poll)
    ALL_TERMINAL  all batches reached terminal state
    COLLECTING    <key>            (score-collect started for a job root)
    COLLECT_OK    <key>
    COLLECT_ERR   <key>: <msg>
    COMPUTED      <key>            (bcg_summary.json written)
    COMPARE_OK                     (comparison.md / .csv / .png generated)
    SUMMARY       M0=... M1=... oracle=...
    RESULTS_READY <path>           (user should read this file)
    EXIT_OK                        (final clean exit)

Event types NOT always emitted:
    HEARTBEAT     every 30 minutes even if no state change (commented out
                  by default to keep the notification stream sparse).

Batch IDs come from env (BCG_M0_BATCH, BCG_M1_BATCH, BCG_ORACLE_BATCH) OR
fallback defaults baked in. Poll interval + max polls via env with defaults
of 30 min and 48 (= 24h).

Usage (normally launched via Monitor tool with persistent=True):
    source .env && uv run python experiments/posttrain/stage4_monitor.py
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

logger = logging.getLogger("stage4_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TERMINAL = {"completed", "failed", "expired", "cancelled"}

BATCHES = {
    "M0": os.environ.get("BCG_M0_BATCH", "batch_69e5cc30bd20819092d4afee541b4c78"),
    "M1": os.environ.get("BCG_M1_BATCH", "batch_69e5cc323c7c819085cea3dd004ce7f0"),
    "oracle": os.environ.get("BCG_ORACLE_BATCH", "batch_69e5cc33b328819089dfc5e98b381ec7"),
}
JOB_ROOTS = {
    "M0": Path("experiments/posttrain/stage4_output/bcg_M0"),
    "M1": Path("experiments/posttrain/stage4_output/bcg_M1"),
    "oracle": Path("experiments/posttrain/stage4_output/bcg_gpt51"),
}
RUBRICS_PATH = Path("experiments/posttrain/stage3_output/paired_rubrics_50.jsonl")

POLL_INTERVAL_SECONDS = int(os.environ.get("STAGE4_POLL_INTERVAL", 30 * 60))  # default 30 min
MAX_POLLS = int(os.environ.get("STAGE4_MAX_POLLS", 48))  # default 24 hours

COMPARISON_DIR = Path("experiments/posttrain/stage4_output")


def emit(kind: str, msg: str = "") -> None:
    """Emit one line to stdout. Each line is an event/notification."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if msg:
        print(f"{now} {kind} {msg}", flush=True)
    else:
        print(f"{now} {kind}", flush=True)


def run(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess; return (returncode, combined stdout/stderr tail)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=1800)
        tail = (p.stdout + p.stderr).splitlines()[-5:]
        return p.returncode, " | ".join(tail)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def run_score_collect(key: str) -> bool:
    emit("COLLECTING", key)
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "score-collect",
        "--rubrics", str(RUBRICS_PATH),
        "--job-root", str(JOB_ROOTS[key]),
    ])
    if rc == 0:
        emit("COLLECT_OK", key)
        return True
    emit("COLLECT_ERR", f"{key}: rc={rc} {tail}")
    return False


def run_compute(key: str) -> bool:
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_bcg_eval.py",
        "compute",
        "--rubrics", str(RUBRICS_PATH),
        "--job-root", str(JOB_ROOTS[key]),
        "--threshold", "7",
    ])
    if rc == 0:
        emit("COMPUTED", key)
        return True
    emit("COLLECT_ERR", f"compute/{key}: rc={rc} {tail}")
    return False


def run_compare() -> bool:
    rc, tail = run([
        "uv", "run", "python", "experiments/posttrain/stage4_compare.py",
        "--output-dir", str(COMPARISON_DIR),
    ])
    if rc == 0:
        emit("COMPARE_OK")
        return True
    emit("COLLECT_ERR", f"compare: rc={rc} {tail}")
    return False


def summary_line() -> str:
    """Read the three bcg_summary.json and emit a compact line."""
    parts = []
    for key in ("M0", "M1", "oracle"):
        path = JOB_ROOTS[key] / "bcg_summary.json"
        if not path.exists():
            parts.append(f"{key}=(missing)")
            continue
        try:
            s = json.loads(path.read_text())
            agg = s.get("aggregate") or {}
            parts.append(
                f"{key}=BCG:{agg.get('mean_bcg', '?'):.2f}"
                f" joint:{agg.get('mean_joint_satisfaction', '?'):.2f}"
                f" A:{agg.get('mean_marginal_A', '?'):.2f}"
                f" B:{agg.get('mean_marginal_B', '?'):.2f}"
            )
        except Exception as e:
            parts.append(f"{key}=(err:{e})")
    return " ".join(parts)


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        emit("COLLECT_ERR", "OPENAI_API_KEY not set")
        return 2

    client = OpenAI()
    prev_state: dict[str, str | None] = {k: None for k in BATCHES}

    emit("STARTED", f"poll_every={POLL_INTERVAL_SECONDS}s max_polls={MAX_POLLS}")
    for k, bid in BATCHES.items():
        emit("TRACKING", f"{k}={bid}")

    final_states: dict[str, str] = {}

    for i in range(1, MAX_POLLS + 1):
        current: dict[str, str] = {}
        for k, bid in BATCHES.items():
            try:
                b = client.batches.retrieve(bid)
                current[k] = b.status
            except Exception as e:
                current[k] = f"ERROR:{e}"
        changes = [k for k in BATCHES if current[k] != prev_state[k]]
        for k in changes:
            emit("STATE_CHANGE", f"{k}={current[k]} (was {prev_state[k]})")
        prev_state = current

        if all(s in TERMINAL for s in current.values()):
            final_states = current
            emit("ALL_TERMINAL", " ".join(f"{k}={v}" for k, v in current.items()))
            break

        if i < MAX_POLLS:
            time.sleep(POLL_INTERVAL_SECONDS)
    else:
        # Polling budget exhausted without all batches terminating.
        emit(
            "BUDGET_EXHAUSTED",
            " ".join(f"{k}={v}" for k, v in prev_state.items()),
        )
        return 1

    # All terminal. Collect the ones that completed.
    for key, status in final_states.items():
        if status == "completed":
            if not run_score_collect(key):
                continue
            run_compute(key)
        else:
            emit("COLLECT_ERR", f"{key}: non-completed terminal status {status}; skipping")

    # Compare whatever we have.
    run_compare()

    emit("SUMMARY", summary_line())
    emit("RESULTS_READY", str(COMPARISON_DIR / "comparison.md"))
    emit("EXIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
