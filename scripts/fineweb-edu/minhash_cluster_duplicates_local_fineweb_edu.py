#!/usr/bin/env python3
"""
This script runs _only_ stage 3 of the deduplication pipeline.

```
# Install marin
pip install -e ".[extras]"
# Install deps for this script
pip install 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging' spacy cupy-cuda12x==13.3.0
```

Cluster duplicates for FineWeb-Edu

```
python scripts/fineweb-edu/minhash_cluster_duplicates_local_fineweb_edu.py \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_logs'
```
"""
import logging
from dataclasses import dataclass

import draccus
import fsspec
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupCluster
from datatrove.utils.hashing import HashConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinhashDeduplicateFineWebEduConfig:
    minhash_base_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash"
    minhash_logs_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_logs"


def minhash_deduplicate_fineweb_edu(
    minhash_base_path: str,
    minhash_logs_path: str,
    minhash_config: MinhashConfig,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # NOTE: Stage 1 and stage 2 should have been already completed before running this script.
    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_base_path}/buckets",
                output_folder=f"{minhash_base_path}/remove_ids",
                config=minhash_config,
                lines_to_buffer=1000,
            ),
        ],
        tasks=1,
        logging_dir=f"{minhash_logs_path}/clusters",
    )
    stage3.run()


@draccus.wrap()
def minhash_cluster_duplicates_fineweb_edu_driver(cfg: MinhashDeduplicateFineWebEduConfig):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    # Do everything in a remote task
    minhash_deduplicate_fineweb_edu(cfg.minhash_base_path, cfg.minhash_logs_path, minhash_config)


if __name__ == "__main__":
    minhash_cluster_duplicates_fineweb_edu_driver()
