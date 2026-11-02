# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Laptop preflight for the proposal shards; sends no completion. Exits 1 when any check fails.

Checks: the driver imports, the S3 catalog loads, each relay's recent logs name ``glm-5.3``, and the router's
``/health`` (through ``kubectl -n open-athena port-forward svc/glm53-router 18000:8000``) reports at least
``MIN_BULK_WORKERS`` bulk workers. Run through ``launch.sh smoke``::

    uv run --frozen --package marin-taskforge runs/curriculum-n25/smoke.py CATALOG RELAY [RELAY...]
"""

import subprocess
import sys

import httpx
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.capability_driven_envs.propose_curriculum import read_catalog, shard_ids

HEALTH_URL = "http://127.0.0.1:18000/health"
MIN_BULK_WORKERS = 8
MODEL = "glm-5.3"


def relay_mentions(relay: str) -> int:
    """How many of the relay's last log lines name the model; 0 when the logs are empty or unreadable."""
    logs = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "iris",
            "--cluster=marin",
            "job",
            "logs",
            relay,
            "--max-lines",
            "5",
            "--since-seconds",
            "300",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return sum(MODEL in line for line in logs.stdout.splitlines())


def bulk_workers() -> int | None:
    try:
        workers = httpx.get(HEALTH_URL, timeout=20).json().get("workers", {})
    except httpx.HTTPError as error:
        print(f"health: {HEALTH_URL} unreachable ({error!r}); is the port-forward up?")
        return None
    print("health workers", workers)
    bulk = workers.get("bulk")
    return bulk if isinstance(bulk, int) else None


def main() -> int:
    catalog, relays = sys.argv[1], sys.argv[2:]
    failures = []
    configure_coreweave_s3()
    ideas = read_catalog(catalog)
    print("catalog capabilities", len(ideas), "shard 0 of 8 has", len(shard_ids(ideas, 0, 8)))
    if not ideas:
        failures.append("catalog is empty")
    for relay in relays:
        count = relay_mentions(relay)
        print(f"relay {relay}: {count} of the last 5 log lines name {MODEL}")
        if count == 0:
            failures.append(f"relay {relay} logs do not name {MODEL}")
    bulk = bulk_workers()
    if bulk is None or bulk < MIN_BULK_WORKERS:
        failures.append(f"bulk workers {bulk}, need >= {MIN_BULK_WORKERS}")
    print("FAILED: " + "; ".join(failures) if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
