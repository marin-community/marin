#!/usr/bin/env python3
"""
Create an index of a dataset so that we can deduplicate other datasets against it.

Indexing fineweb-edu:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python marin/crawl/minhash/index_for_deduplication.py \
    --input_patterns '["gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/fineweb_edu_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/fineweb_edu_minhash_index_logs' \
    --index_name fineweb-edu-index

# Remove intermediate outputs
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index/signatures/
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/fineweb_edu_minhash_index/buckets/
```

Indexing open-web-math:

```
python marin/run/ray_run.py \
    --pip_deps 'datatrove @ git+https://github.com/nelson-liu/datatrove@ray_executor_dedup_logging,spacy,cupy-cuda12x==13.3.0' \
    --no_wait -- \
    python marin/crawl/minhash/index_for_deduplication.py \
    --input_patterns '["gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/*.parquet"]' \
    --parquets_paths_file 'gs://marin-us-central2/scratch/nfliu/open_web_math_paths.txt' \
    --minhash_base_path 'gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index' \
    --minhash_logs_path 'gs://marin-us-central2/scratch/nfliu/minhash/logs/open_web_math_minhash_index_logs' \
    --index_name open-web-math-index

# Remove intermediate outputs
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index/signatures/
gcloud storage rm --recursive gs://marin-us-central2/scratch/nfliu/minhash/open_web_math_minhash_index/buckets/
```
"""  # noqa: E501
import json
import logging
from dataclasses import dataclass

import draccus
import fsspec
import ray
from datatrove.executor import RayPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.readers import ParquetReader
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexForDeduplicationConfig:
    input_patterns: list[str]
    parquets_paths_file: str
    minhash_base_path: str
    minhash_logs_path: str
    index_name: str


def str_metadata_adapter(self, data: dict, path: str, id_in_file: int | str):
    """
    The function takes input data and transforms it into the
    datatrove Document format. However, the default adapter assumes that
    the metadata is a `dict`.
    This is not always the case (e.g., open-web-math uses a string-serialized JSON dict),
    so this adapter handles the case that it's a string by trying to parse it as JSON.

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    if "metadata" in data:
        metadata = data.pop("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif isinstance(metadata, dict):
            pass
        else:
            raise ValueError(f"Got invalid metadata of type {type(metadata)}: {metadata}")
    else:
        metadata = {}

    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        # remaining data goes into metadata
        "metadata": metadata | data,
    }


@ray.remote(memory=32 * 1024 * 1024 * 1024, num_cpus=8)
def index_for_minhash_deduplication(
    input_patterns: list[str],
    parquets_paths_file: str,
    minhash_base_path: str,
    minhash_logs_path: str,
    index_name: str,
    minhash_config: MinhashConfig,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not fsspec_exists(parquets_paths_file):
        # Create the pathfile for the input dataset, removing the base bucket prefix
        input_parquet_paths = []
        for pattern in input_patterns:
            for path in fsspec_glob(pattern):
                assert path.startswith("gs://marin-us-central2/")
                input_parquet_paths.append(path.removeprefix("gs://marin-us-central2/"))
        with fsspec.open(parquets_paths_file, "w") as f:
            for path in tqdm(input_parquet_paths, desc="Writing parquets paths file"):
                f.write(path + "\n")

    # this is the original data that we want to deduplicate
    # NOTE: neither the base folder path or the paths in the pathfile should
    # include the leading "/"
    INPUT_READER = ParquetReader(
        "gs://marin-us-central2",
        paths_file=parquets_paths_file,
        doc_progress=True,
        adapter=str_metadata_adapter,
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
            MinhashDedupBuckets(
                input_folder=f"{minhash_base_path}/signatures",
                output_folder=f"{minhash_base_path}/buckets",
                index_folder=f"{minhash_base_path}/index",
                create_index_name=index_name,
                # Index should be empty, so we aren't actually removing anything
                only_dedup_in_index=True,
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets * 50,
        randomize_start_duration=180,
        logging_dir=f"{minhash_logs_path}/index",
        depends=stage1,
        memory_bytes_per_task=16 * 1024 * 1024 * 1024,
    )
    stage2.run()


@draccus.wrap()
def index_for_minhash_deduplication_driver(cfg: IndexForDeduplicationConfig):
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
        index_for_minhash_deduplication.remote(
            cfg.input_patterns,
            cfg.parquets_paths_file,
            cfg.minhash_base_path,
            cfg.minhash_logs_path,
            cfg.index_name,
            minhash_config,
        )
    )


if __name__ == "__main__":
    index_for_minhash_deduplication_driver()
