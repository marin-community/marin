# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny Zephyr TPU fanout smoke.

Run a local dry-run first:

    uv run python experiments/tutorials/zephyr_tpu_fanout_smoke.py --dry-run

Run on Iris as a batch job:

    uv run iris --cluster marin job run --priority batch --no-preemptible -- \
      python experiments/tutorials/zephyr_tpu_fanout_smoke.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from typing import Any

from fray import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from zephyr import Dataset, ZephyrContext

# Edit these constants when trying different fleets.
TPU_TYPES = ["v4-8", "v5p-8", "v6e-4", "v5litepod-4"]
WORKER_REGIONS = ["europe-west4", "us-central1", "us-central2", "us-east1", "us-east5", "us-west4"]

# Keep the output small and placement-focused.
ENV_KEYS = [
    "IRIS_TASK_ID",
    "IRIS_NUM_TASKS",
    "IRIS_WORKER_ID",
    "IRIS_WORKER_REGION",
    "IRIS_JOB_CONSTRAINTS",
    "JAX_PLATFORMS",
    "PJRT_DEVICE",
]


def infer_worker_placement(worker_id: str | None) -> dict[str, str]:
    """Infer placement hints from the Iris worker id."""
    if not worker_id:
        return {}

    placement = {}

    # Example worker id fragment:
    #   marin-tpu-v5p-preemptible-8-us-east5-a-...
    for tpu_type in TPU_TYPES:
        family, chip_count = tpu_type.rsplit("-", maxsplit=1)
        if f"-{family}-" in worker_id and f"-{chip_count}-" in worker_id:
            placement["inferred_tpu_type"] = tpu_type
            break

    for region in WORKER_REGIONS:
        marker = f"-{region}-"
        if marker not in worker_id:
            continue
        zone_suffix = worker_id.split(marker, maxsplit=1)[1].split("-", maxsplit=1)[0]
        placement["inferred_region"] = region
        placement["inferred_zone"] = f"{region}-{zone_suffix}"
        break

    return placement


def process_context() -> dict[str, Any]:
    """Return the Iris/local context for this process."""
    job_info = get_job_info()
    worker_id = job_info.worker_id if job_info else None
    return {
        "hostname": socket.gethostname(),
        "iris_job_id": str(job_info.job_id) if job_info else None,
        "iris_task_index": job_info.task_index if job_info else None,
        "iris_attempt_id": job_info.attempt_id if job_info else None,
        "iris_worker_id": worker_id,
        "iris_worker_region": job_info.worker_region if job_info else None,
        **infer_worker_placement(worker_id),
        "env": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
    }


def worker_probe(record_id: str) -> dict[str, Any]:
    """Minimal worker body: report placement context, then sleep."""
    # Keep the TPU worker alive long enough to inspect placement.
    time.sleep(600)

    # Put real vLLM work here.

    return {
        "record_id": record_id,
        **process_context(),
    }


def main() -> None:
    # One record becomes one Zephyr shard. Keep this large enough that Iris
    # may need to spill demand into more than one region or TPU pool.
    records = [f"probe-{idx:04d}" for idx in range(256)]

    # These resources apply to Zephyr workers, not the outer Iris job.
    worker_resources = ResourceConfig.with_tpu(TPU_TYPES, regions=WORKER_REGIONS)

    # The coordinator child job uses Zephyr's CPU-only default resources.
    ctx = ZephyrContext(name="tpu-fanout-smoke", max_workers=len(records), resources=worker_resources)

    result = ctx.execute(
        Dataset.from_list(records).map(worker_probe),
        verbose=True,
        dry_run="--dry-run" in sys.argv,
    )

    print(
        json.dumps(
            {
                "driver": process_context(),
                "requested_tpu_types": TPU_TYPES,
                "requested_worker_regions": WORKER_REGIONS,
                "results": result.results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
