#!/usr/bin/env python3
"""
python marin/run/ray_run.py \
    --pip_deps 'datatrove[all]' \
    --no_wait -- \
    python scripts/fineweb-edu/minhash_deduplicate_fineweb_edu.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*.parquet' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_minhash_logs'
"""
import logging
from dataclasses import dataclass

import draccus
import fsspec
import ray
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinhashDeduplicateFineWebEduConfig:
    input_pattern: str = "gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*.parquet"
    parquets_paths_file: str = "gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_paths.txt"
    minhash_base_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash"
    minhash_logs_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_minhash_logs"


@ray.remote(memory=300 * 1024 * 1024 * 1024, num_cpus=101)
def minhash_deduplicate_fineweb_edu(
    fineweb_edu_pattern: str,
    parquets_paths_file: str,
    minhash_base_path: str,
    minhash_logs_path: str,
    minhash_config: MinhashConfig,
):
    if not fsspec_exists(parquets_paths_file):
        # Create the pathfile for FineWeb-Edu, removing the base bucket prefix
        fineweb_edu_parquet_paths = [
            path.removeprefix("gs://marin-us-central2/") for path in fsspec_glob(fineweb_edu_pattern)
        ]
        with fsspec.open(parquets_paths_file) as f:
            for path in tqdm(fineweb_edu_parquet_paths, desc="Writing parquets paths file"):
                f.write(path + "\n")

    # this is the original data that we want to deduplicate
    INPUT_READER = ParquetReader("gs://marin-us-central2/", paths_file=parquets_paths_file)
    TOTAL_TASKS = 1000
    NUM_WORKERS = 100

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{minhash_base_path}/signatures", config=minhash_config, language=Languages.english
            ),
        ],
        tasks=TOTAL_TASKS,
        WORKERS=NUM_WORKERS,
        logging_dir=f"{minhash_logs_path}/signatures",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{minhash_base_path}/signatures",
                output_folder=f"{minhash_base_path}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{minhash_logs_path}/buckets",
        depends=stage1,
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_base_path}/buckets",
                output_folder=f"{minhash_base_path}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"{minhash_logs_path}/clusters",
        depends=stage2,
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
            MinhashDedupFilter(
                input_folder=f"{minhash_base_path}/remove_ids",
                exclusion_writer=JsonlWriter(f"{minhash_base_path}/removed"),
            ),
            JsonlWriter(output_folder=f"{minhash_base_path}/deduplicated_output"),
        ],
        tasks=TOTAL_TASKS,
        WORKERS=NUM_WORKERS,
        logging_dir=f"{minhash_logs_path}/filter",
        depends=stage3,
    )
    stage4.run()


@draccus.wrap()
def process_fineweb_edu(cfg: MinhashDeduplicateFineWebEduConfig):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    # Do everything in a remote task
    ray.get(
        minhash_deduplicate_fineweb_edu.remote(
            cfg.input_pattern, cfg.parquets_paths_file, cfg.minhash_base_path, cfg.minhash_logs_path, minhash_config
        )
    )


if __name__ == "__main__":
    minhash_deduplicate_fineweb_edu()
