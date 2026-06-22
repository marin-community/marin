# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated normalize-only benchmark for Zephyr shuffle perf work.

Runs only download (FineWeb-Edu sample/10BT, cached at a fixed path) + normalize
(the shuffle-heavy group_by stage). Download caches across runs via a stable
``override_output_path``; normalize re-runs each invocation into a fresh path
keyed by ``BENCH_RUN_ID`` so we can A/B treatments against the same input.

Launch on Iris (interactive band, us-central1, co-located with marin-us-central1):

  MARIN_PREFIX=gs://marin-us-central1 BENCH_RUN_ID=base-$(date +%s) \
  uv run iris --config lib/iris/config/marin.yaml job run \
    --region us-central1 --memory 5GB --no-wait \
    -e MARIN_PREFIX gs://marin-us-central1 -e BENCH_RUN_ID base-XXXX \
    -- python -m experiments.ferries.bench_normalize_only
"""

import logging
import os

from fray import ResourceConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging
from rigging.timing import log_time

logger = logging.getLogger(__name__)

# Stable download path so the ~10BT download is paid for once and reused.
_DOWNLOAD_PATH = "datakit-shuffle-bench/download-fineweb-edu-10BT"


def main() -> None:
    configure_logging()
    marin_prefix = os.environ["MARIN_PREFIX"]
    run_id = os.environ["BENCH_RUN_ID"]
    logger.info("MARIN_PREFIX=%s BENCH_RUN_ID=%s", marin_prefix, run_id)

    downloaded = download_hf_step(
        "datakit-shuffle-bench/download",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision="87f0914",
        hf_urls_glob=["sample/10BT/*.parquet"],
        zephyr_max_parallelism=14,
        override_output_path=_DOWNLOAD_PATH,
    )

    normalized = normalize_step(
        name="datakit-shuffle-bench/normalize",
        download=downloaded,
        relative_input_path="sample/10BT",
        worker_resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
        override_output_path=f"datakit-shuffle-bench/normalize/{run_id}",
    )

    with log_time(f"normalize-only bench {run_id}"):
        StepRunner().run([downloaded, normalized])


if __name__ == "__main__":
    main()
