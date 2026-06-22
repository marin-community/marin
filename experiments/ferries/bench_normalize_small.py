# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast small-input normalize benchmark for Zephyr shuffle perf iteration.

Calls ``normalize_to_parquet`` directly (no StepRunner) on a 3-file (~6GB)
subset of FineWeb-Edu sample/10BT, pre-staged at
``gs://marin-us-central1/datakit-shuffle-bench/download-small``. The
treatment/baseline *ratio* of normalize shard wall time is stable across scale,
so this gives a faithful A/B much faster than the full 28.5GB job.

Launch (interactive, us-central1):
  uv run iris --config lib/iris/config/marin.yaml job run --region us-central1 \
    --memory 3GB --no-wait -e MARIN_PREFIX gs://marin-us-central1 \
    -e BENCH_RUN_ID small-XXXX -- python -m experiments.ferries.bench_normalize_small
"""

import logging
import os

from fray import ResourceConfig
from marin.datakit.normalize import normalize_to_parquet
from rigging.log_setup import configure_logging
from rigging.timing import log_time

logger = logging.getLogger(__name__)

_INPUT = "gs://marin-us-central1/datakit-shuffle-bench/download-small/sample/10BT"


def main() -> None:
    configure_logging()
    marin_prefix = os.environ["MARIN_PREFIX"]
    run_id = os.environ["BENCH_RUN_ID"]
    output = f"{marin_prefix}/datakit-shuffle-bench/normalize-small/{run_id}"
    logger.info("normalize-small: %s -> %s", _INPUT, output)

    kwargs = {}
    if os.environ.get("BENCH_COLUMNAR") == "1":
        kwargs["columnar"] = True
        logger.info("COLUMNAR path enabled")
    with log_time(f"normalize-small bench {run_id}"):
        result = normalize_to_parquet(
            input_path=_INPUT,
            output_path=output,
            worker_resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
            **kwargs,
        )
    logger.info("counters: %s", result.counters)


if __name__ == "__main__":
    main()
