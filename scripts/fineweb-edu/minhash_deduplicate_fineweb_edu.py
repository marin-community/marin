#!/usr/bin/env python3
"""

Deduplicating fineweb-edu-10M

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python scripts/fineweb-edu/minhash_deduplicate_fineweb_edu.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_minhash_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-minhash/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_minhash_logs
```

Deduplicating FineWeb-Edu

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python scripts/fineweb-edu/minhash_deduplicate_fineweb_edu.py \
    --input_patterns '["gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_logs'

# Move the deduplicated content
gcloud storage mv gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash/deduplicated_output/* gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-minhash/
# Remove the logs and intermediate output
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_logs
```

Deduplicating the union of FineWeb-Edu and fineweb-edu-10M

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python scripts/fineweb-edu/minhash_deduplicate_fineweb_edu.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.parquet", "gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_fineweb_edu_10M_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_fineweb_edu_10M_minhash' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_fineweb_edu_10M_minhash_logs'
```
"""
import logging
from dataclasses import dataclass, field

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
class MinhashDeduplicateFineWebEduConfig:
    input_patterns: list[str] = field(
        default_factory=lambda: ["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.parquet"]
    )
    parquets_paths_file: str = "gs://marin-us-central2/scratch/nfliu/fineweb_edu_10M_paths.txt"
    minhash_base_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_10M_minhash"
    minhash_logs_path: str = "gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_10M_minhash_logs"


@ray.remote(memory=32 * 1024 * 1024 * 1024, num_cpus=8)
def minhash_deduplicate_fineweb_edu(
    fineweb_edu_patterns: list[str],
    parquets_paths_file: str,
    minhash_base_path: str,
    minhash_logs_path: str,
    minhash_config: MinhashConfig,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not fsspec_exists(parquets_paths_file):
        # Create the pathfile for FineWeb-Edu, removing the base bucket prefix
        fineweb_edu_parquet_paths = []
        for pattern in fineweb_edu_patterns:
            for path in fsspec_glob(pattern):
                assert path.startswith("gs://marin-us-central2/")
                fineweb_edu_parquet_paths.append(path.removeprefix("gs://marin-us-central2/"))
        with fsspec.open(parquets_paths_file, "w") as f:
            for path in tqdm(fineweb_edu_parquet_paths, desc="Writing parquets paths file"):
                f.write(path + "\n")

    # this is the original data that we want to deduplicate
    # NOTE: neither the base folder path or the paths in the pathfile should
    # include the leading "/"
    INPUT_READER = ParquetReader(
        "gs://marin-us-central2",
        paths_file=parquets_paths_file,
    )
    TOTAL_TASKS = 2000
    NUM_WORKERS = 500

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
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{minhash_logs_path}/buckets",
        depends=stage1,
        memory_bytes_per_task=8 * 1024 * 1024 * 1024,
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = RayPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_base_path}/buckets",
                output_folder=f"{minhash_base_path}/remove_ids",
                config=minhash_config,
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
def minhash_deduplicate_fineweb_edu_driver(cfg: MinhashDeduplicateFineWebEduConfig):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    # Do everything in a remote task
    ray.get(
        minhash_deduplicate_fineweb_edu.remote(
            cfg.input_patterns, cfg.parquets_paths_file, cfg.minhash_base_path, cfg.minhash_logs_path, minhash_config
        )
    )


if __name__ == "__main__":
    minhash_deduplicate_fineweb_edu_driver()
