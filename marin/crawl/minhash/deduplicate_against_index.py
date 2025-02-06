#!/usr/bin/env python3
"""
Use MinHash deduplication to fuzzily deduplicate an input dataset against an indexed dataset.
More specifically, given an input dataset D (e.g., 10M new crawled pages),
and an indexed dataset I, this scripts produces a dataset D' by removing items in D that:

- Are duplicated in D
- Are duplicated in I

Note that the output deduplicated dataset D' is a subset of the input dataset D,
and does not include any elements from the indexed dataset I.

We use datatrove's built-in minhash deduplication to match the fineweb-edu
curation process.

Deduplicating fineweb-edu-10M against fineweb-edu:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_passing_minhash_against_fineweb_edu_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_passing_minhash_against_fineweb_edu/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_passing_minhash_against_fineweb_edu
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_passing_minhash_against_fineweb_edu_logs
```

Deduplicating open-web-math-10M-passing against open-web-math:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0,orjson' \
    --no_wait -- \
    python marin/crawl/minhash/deduplicate_against_index.py \
    --index_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index/index' \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/*_text_and_scores.passing.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/open-web-math-fde8ef8-10M-passing_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_passing_minhash_against_open_web_math/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_10M_passing_minhash_against_open_web_math_logs
```

"""  # noqa: E501
import logging
from dataclasses import dataclass

import draccus
import fsspec
import ray
from datatrove.executor import RayPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinhashDeduplicateAgainstIndexConfig:
    index_path: str
    input_patterns: list[str]
    parquets_paths_file: str
    minhash_base_path: str
    minhash_logs_path: str


@ray.remote(memory=32 * 1024 * 1024 * 1024, num_cpus=8)
def minhash_deduplicate_against_index(
    index_path: str,
    input_patterns: list[str],
    parquets_paths_file: str,
    minhash_base_path: str,
    minhash_logs_path: str,
    minhash_config: MinhashConfig,
):
    """
    Given a path to a previously-produced index (with `index_for_deduplication.py`),
    deduplicate an input dataset against this index.

    Args:
        index_path (str): path to directory with index files
        input_patterns (list[str]): pattern(s) to input files to deduplicate. Ignored if
            `parquets_path_file` already exists, since it's assumed that `parquest_path_file`
            is a cache of the files matched by the `input_patterns`.
        parquets_path_file (str): path to write a file with the resolved paths of
            the files specified by the `input_patterns`. If this file already exists,
            its contents are just used as-is.
        minhash_base_path (str): path to write minhash output
        minhash_logs_path (str): path to write minhash logs
        minhash_config (MinhashConfig): configuration to use for minhash deduplication
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not fsspec_exists(parquets_paths_file):
        # Create the pathfile for the input, removing the base bucket prefix
        input_parquet_paths = []
        for pattern in input_patterns:
            for path in fsspec_glob(pattern):
                assert path.startswith("gs://marin-us-central2/")
                input_parquet_paths.append(path.removeprefix("gs://marin-us-central2/"))
        with fsspec.open(parquets_paths_file, "w", block_size=1 * 1024 * 1024 * 1024) as f:
            for path in tqdm(input_parquet_paths, desc="Writing parquets paths file"):
                f.write(path + "\n")

    # this is the original data that we want to deduplicate
    # NOTE: neither the base folder path or the paths in the pathfile should
    # include the leading "/"
    INPUT_READER = ParquetReader(
        "gs://marin-us-central2",
        paths_file=parquets_paths_file,
    )
    TOTAL_TASKS = 2000
    NUM_WORKERS = 1000

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = RayPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{minhash_base_path}/signatures", config=minhash_config, language=Languages.english
            ),
        ],
        tasks=TOTAL_TASKS,
        workers=NUM_WORKERS,
        logging_dir=f"{minhash_logs_path}/signatures",
        memory_bytes_per_task=8 * 1024 * 1024 * 1024,
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = RayPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{minhash_base_path}/signatures",
                output_folder=f"{minhash_base_path}/buckets",
                # Remove items from the input dataset with signatures that match the index
                index_folder=index_path,
                # Ensures we also catch duplicates within the input dataset
                only_dedup_in_index=False,
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets * 50,
        randomize_start_duration=180,
        logging_dir=f"{minhash_logs_path}/buckets",
        depends=stage1,
        memory_bytes_per_task=16 * 1024 * 1024 * 1024,
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = RayPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_base_path}/buckets",
                output_folder=f"{minhash_base_path}/remove_ids",
                config=minhash_config,
                lines_to_buffer=1000,
                ignore_index_matches=False,
            ),
        ],
        tasks=1,
        memory_bytes_per_task=300 * 1024 * 1024 * 1024,
        logging_dir=f"{minhash_logs_path}/clusters",
        depends=stage2,
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = RayPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{minhash_base_path}/remove_ids",
                exclusion_writer=JsonlWriter(f"{minhash_base_path}/removed"),
            ),
            JsonlWriter(output_folder=f"{minhash_base_path}/deduplicated_output"),
        ],
        tasks=TOTAL_TASKS,
        workers=NUM_WORKERS,
        logging_dir=f"{minhash_logs_path}/filter",
        depends=stage3,
        memory_bytes_per_task=8 * 1024 * 1024 * 1024,
    )
    stage4.run()


@draccus.wrap()
def minhash_deduplicate_against_index_driver(cfg: MinhashDeduplicateAgainstIndexConfig):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    # Do everything in a remote task
    ray.get(
        minhash_deduplicate_against_index.remote(
            cfg.index_path,
            cfg.input_patterns,
            cfg.parquets_paths_file,
            cfg.minhash_base_path,
            cfg.minhash_logs_path,
            minhash_config,
        )
    )


if __name__ == "__main__":
    minhash_deduplicate_against_index_driver()
