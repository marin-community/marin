# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for full science conversion, audit it, prepare its token store, and launch SFT."""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass

from iris.client.client import iris_ctx
from iris.cluster.types import JobName
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.science_sft_conversion.audit import audit
from experiments.datakit.science_sft_conversion.conversion import OUTPUT_MAIN_DIR, OUTPUT_ROOT, _work_items
from experiments.grug.science_sft.launch import launch
from experiments.grug.science_sft.prepare import SOURCE_NAME, TOKENIZER, prepare_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandoffConfig:
    producer_job: str
    store_path: str
    num_shards: int
    max_workers: int
    timeout_hours: float
    version: str


def finish_conversion(config: HandoffConfig) -> None:
    """Advance only after the producer succeeds and every expected output passes audit."""
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Run the conversion handoff as an Iris task")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required before waiting for the SFT handoff")
    producer = client.job(JobName.from_string(config.producer_job))
    producer.wait(timeout=config.timeout_hours * 3600, poll_interval=300)
    result = audit(_work_items(), OUTPUT_ROOT, workers=config.max_workers)
    fs, path = filesystem_for(prefix_join(OUTPUT_ROOT, "audit", "final-coverage.json"))
    with atomic_rename(path, filesystem=fs) as temporary_path:
        with fs.open(temporary_path, "w") as stream:
            json.dump(asdict(result), stream, indent=2)
    if not result.complete:
        raise ValueError(f"Full conversion audit failed: {result}")
    logger.info("Full conversion passed audit: %s", result)
    store = prepare_store(
        prefix_join(OUTPUT_ROOT, OUTPUT_MAIN_DIR), config.store_path, TOKENIZER, config.num_shards, config.max_workers
    )
    if store.sources[SOURCE_NAME].conversations != result.output_rows:
        raise ValueError("Packed store conversation count differs from the audited converted corpus")
    launch(config.store_path, config.version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-job", required=True)
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--timeout-hours", type=float, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", action="store_true", help="Execute the handoff; otherwise print its plan")
    args = parser.parse_args()
    if min(args.num_shards, args.max_workers, args.timeout_hours) <= 0:
        parser.error("num-shards, max-workers, and timeout-hours must be positive")
    config = HandoffConfig(
        producer_job=args.producer_job,
        store_path=args.store_path,
        num_shards=args.num_shards,
        max_workers=args.max_workers,
        timeout_hours=args.timeout_hours,
        version=args.version,
    )
    logging.basicConfig(level=logging.INFO)
    if args.run:
        finish_conversion(config)
        return
    print(
        json.dumps({**asdict(config), "source_path": prefix_join(OUTPUT_ROOT, OUTPUT_MAIN_DIR), "tokenizer": TOKENIZER})
    )


if __name__ == "__main__":
    main()
