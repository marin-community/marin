#!/usr/bin/env python3
"""
Index fineweb-edu for eventual bipartite deduplication.

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove[io,processing] @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python scripts/fineweb-edu/index_fineweb_edu_for_deduplication.py \
    --input_patterns '["gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_index_logs'
```
"""
import logging
from dataclasses import dataclass

import draccus
import fsspec
import ray
from datatrove.executor import RayPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashBuildIndex
from datatrove.pipeline.readers import ParquetReader
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexFineWebEduForMinHashDeduplicationConfig:
    input_patterns: list[str]
    parquets_paths_file: str
    minhash_base_path: str
    minhash_logs_path: str


@ray.remote(memory=32 * 1024 * 1024 * 1024, num_cpus=8)
def index_fineweb_edu_for_minhash_deduplication(
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
        doc_progress=True,
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

    # stage 2 creates an index from the signatures
    stage2 = RayPipelineExecutor(
        pipeline=[
            MinhashBuildIndex(
                input_folder=f"{minhash_base_path}/signatures",
                output_folder=f"{minhash_base_path}/index",
                index_name="fineweb-edu-index",
                config=minhash_config,
                lines_to_buffer=1000,
            ),
        ],
        tasks=minhash_config.num_buckets,
        memory_bytes_per_task=16 * 1024 * 1024 * 1024,
        logging_dir=f"{minhash_logs_path}/index",
        depends=stage1,
    )
    stage2.run()


@draccus.wrap()
def index_fineweb_edu_for_minhash_deduplication_driver(cfg: IndexFineWebEduForMinHashDeduplicationConfig):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            hash_fc="sha1",
            precision=64,
        ),
        num_buckets=14,
        hashes_per_bucket=8,
    )
    # Do everything in a remote task
    ray.get(
        index_fineweb_edu_for_minhash_deduplication.remote(
            cfg.input_patterns, cfg.parquets_paths_file, cfg.minhash_base_path, cfg.minhash_logs_path, minhash_config
        )
    )


if __name__ == "__main__":
    index_fineweb_edu_for_minhash_deduplication_driver()
