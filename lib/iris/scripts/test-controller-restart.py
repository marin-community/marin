#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual test: start smoke cluster, restart controller, validate health.

Usage:
    cd lib/iris && uv run python scripts/test-controller-restart.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("test-restart")

IRIS_ROOT = Path(__file__).parent.parent
CONFIG = str(IRIS_ROOT / "examples" / "smoke-gcp.yaml")


def iris(*args: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "iris", "--config", CONFIG, *args]
    logger.info("$ %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(IRIS_ROOT), timeout=timeout)
    for line in (r.stdout + r.stderr).strip().splitlines():
        logger.info("  %s", line)
    r.check_returncode()
    return r


def iris_nofail(*args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "iris", "--config", CONFIG, *args]
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(IRIS_ROOT), timeout=timeout)


def main():
    logger.info("=== 1. Stop existing cluster ===")
    iris_nofail("cluster", "stop", timeout=300)

    logger.info("=== 2. Start cluster ===")
    iris("cluster", "start")

    logger.info("=== 3. Check status ===")
    iris("cluster", "status")

    logger.info("=== 4. List jobs (pre-restart) ===")
    r = iris_nofail("rpc", "controller", "list-jobs")
    pre_jobs = json.loads(r.stdout).get("jobs", []) if r.returncode == 0 else []
    logger.info("Jobs before restart: %d", len(pre_jobs))

    logger.info("=== 5. Controller restart (checkpoint + stop + start) ===")
    iris("cluster", "controller", "restart")

    logger.info("=== 6. Check status (post-restart) ===")
    iris("cluster", "status")

    logger.info("=== 7. Validate restored jobs ===")
    r = iris_nofail("rpc", "controller", "list-jobs")
    if r.returncode != 0:
        logger.error("FAIL: could not list jobs after restart: %s", r.stderr)
        sys.exit(1)
    post_jobs = json.loads(r.stdout).get("jobs", [])
    logger.info("Jobs after restart: %d", len(post_jobs))
    if len(post_jobs) < len(pre_jobs):
        logger.error("FAIL: lost jobs after restart (%d -> %d)", len(pre_jobs), len(post_jobs))
        sys.exit(1)

    logger.info("=== 8. Cleanup ===")
    iris("cluster", "stop")

    logger.info("=== PASS ===")


if __name__ == "__main__":
    main()
