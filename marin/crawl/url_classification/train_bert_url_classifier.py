#!/usr/bin/env python3
"""
```
python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/libtpu-releases/index.html,--find-links https://storage.googleapis.com/libtpu-wheels/index.html,datasets,filelock,torch,torch_xla[tpu],accelerate,scikit-learn' \
    --env_vars WANDB_API_KEY 'ca4e321fd237f65236ab95e92724934b47264b1c' \
    --no_wait -- \
    python marin/crawl/url_classification/train_bert_url_classifier.py \
    --urls_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/links.*.parquet' \
    --fetched_urls_pattern 'gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/links.*.parquet' \
    --output_path gs://marin-us-central2/scratch/nfliu/url_classification_models/bert-base-uncased-open-web-math-fde8ef8-10M/
```

For multi-GPU training:

```
pip install datasets filelock torch accelerate scikit-learn
export WANDB_API_KEY='ca4e321fd237f65236ab95e92724934b47264b1c'

python marin/crawl/url_classification/train_bert_url_classifier.py \
    --urls_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/links.*.parquet' \
    --fetched_urls_pattern 'gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/links.*.parquet' \
    --output_path gs://marin-us-central2/scratch/nfliu/url_classification_models/bert-base-uncased-open-web-math-fde8ef8-10M/
```
"""  # noqa: E501
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass

import draccus
import fsspec
import pandas as pd
import ray
from tqdm_loggable.auto import tqdm

from marin.classifiers.bert.training import BertTrainingArguments, _mp_fn, tokenize_json_save_as_arrow
from marin.crawl.url_classification.metrics import url_classifier_compute_eval_metrics
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob, remove_tpu_lockfile_on_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainBertUrlClassifierConfig:
    urls_pattern: str
    fetched_urls_pattern: str
    output_path: str
    val_frac: float = 0.1
    batch_size: int = 64
    lr: float = 2e-5
    hf_model: str = "bert-base-uncased"
    num_epochs: int = 3
    seed: int = 0
    max_length: int = 512
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int | None = None


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def make_url_classification_dataset(
    urls_pattern: str,
    fetched_urls_pattern: str,
    train_output_path: str,
    val_output_path: str,
    val_frac: float,
    seed: int,
):
    """
    Writes a jsonl.gz file at train_output_path and val_output_path with the train and dev
    datasets for BERT training. Each record has a 'text' and 'label' field.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(seed)

    train_success_path = train_output_path + ".SUCCESS"
    val_success_path = val_output_path + ".SUCCESS"
    if fsspec_exists(train_success_path) and fsspec_exists(val_success_path):
        logger.info(
            f"Found train success path {train_success_path} and val success path {val_success_path}, skipping..."
        )
        return

    # Get the input shard paths
    fetched_shard_paths: list[str] = list(fsspec_glob(fetched_urls_pattern))
    logger.info(f"Found {len(fetched_shard_paths)} fetched shards to process")
    fetched_shard_paths = sorted(fetched_shard_paths)
    logger.info(f"Sorted {len(fetched_shard_paths)} fetched shards")

    url_shard_paths: list[str] = list(fsspec_glob(urls_pattern))
    logger.info(f"Found {len(url_shard_paths)} URL shards to process")
    url_shard_paths = sorted(url_shard_paths)
    logger.info(f"Sorted {len(url_shard_paths)} URL shards")

    # First pass: Count total lines and build a set of all fetched URLs
    failing_fetched_urls = set()
    passing_fetched_urls = set()
    for shard_path in tqdm(fetched_shard_paths, desc="Collecting URLs from fetched input records"):
        df = pd.read_parquet(shard_path)
        for record_metadata in df["metadata"]:
            if record_metadata["passes_all_filters"]:
                passing_fetched_urls.add(record_metadata["url"])
            else:
                failing_fetched_urls.add(record_metadata["url"])

    unfetched_urls = set()
    for shard_path in tqdm(url_shard_paths, desc="Collecting URLs from unfetched input records"):
        # Read no columns => minimal overhead, but we still get the row count
        df = pd.read_parquet(shard_path)
        for link_target in df["link_target"]:
            if link_target not in failing_fetched_urls and link_target not in passing_fetched_urls:
                unfetched_urls.add(link_target)

    examples = []
    for url in tqdm(failing_fetched_urls, desc="Converting failing fetched URLs to examples"):
        examples.append({"text": url, "label": False})
    for url in tqdm(passing_fetched_urls, desc="Converting passing fetched URLs to examples"):
        examples.append({"text": url, "label": True})
    for url in tqdm(unfetched_urls, desc="Converting unfetched URLs to examples"):
        examples.append({"text": url, "label": False})
    random.shuffle(examples)

    total_examples = len(examples)
    logger.info(f"Total URLs across all shards = {total_examples:,}")
    assert total_examples > 0

    # Compute how many lines go to validation (integer)
    num_val_lines = int(total_examples * val_frac)
    logger.info(f"# of validation lines = {num_val_lines:,}")

    train_examples_written = 0
    val_examples_written = 0
    with fsspec.open(val_output_path, "w", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as f_val:
        for example in examples[:num_val_lines]:
            f_val.write(json.dumps(example) + "\n")
            val_examples_written += 1
    with fsspec.open(train_output_path, "w", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as f_train:
        for example in examples[num_val_lines:]:
            f_train.write(json.dumps(example) + "\n")
            train_examples_written += 1

    assert val_examples_written == num_val_lines
    assert (train_examples_written + val_examples_written) == total_examples

    # Write success files
    with fsspec.open(train_success_path, "w", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"examples_written": train_examples_written}, f)
    with fsspec.open(val_success_path, "w", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"examples_written": val_examples_written}, f)
    logger.info("Finished writing train and validation splits.")


def train_bert_url_classifier(
    urls_pattern: str,
    fetched_urls_pattern: str,
    output_path: str,
    val_frac: float,
    batch_size: int,
    lr: float,
    hf_model: str,
    num_epochs: int,
    max_length: int,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int | None,
    seed: int = 0,
):
    # Hash this dataset configuration so we can skip dataset generation if it already exists.
    dataset_hash: str = hashlib.md5(f"{fetched_urls_pattern}{val_frac}{seed}".encode()).hexdigest()
    train_dataset_path = os.path.join(output_path, "data", f"train_{dataset_hash}.jsonl.gz")
    val_dataset_path = os.path.join(output_path, "data", f"val_{dataset_hash}.jsonl.gz")
    hf_format_dataset_path = os.path.join(output_path, "data", "train_val_hf")

    ray.get(
        make_url_classification_dataset.remote(
            urls_pattern=urls_pattern,
            fetched_urls_pattern=fetched_urls_pattern,
            train_output_path=train_dataset_path,
            val_output_path=val_dataset_path,
            val_frac=val_frac,
            seed=seed,
        )
    )

    ray.get(
        tokenize_json_save_as_arrow.remote(
            train_input_path=train_dataset_path,
            val_input_path=val_dataset_path,
            hf_model=hf_model,
            max_length=max_length,
            output_path=hf_format_dataset_path,
        )
    )

    ray.get(
        train_model.remote(
            dataset_path=hf_format_dataset_path,
            output_path=output_path,
            batch_size=batch_size,
            lr=lr,
            hf_model=hf_model,
            num_epochs=num_epochs,
            max_length=max_length,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
        )
    )


@ray.remote(
    memory=350 * 1024 * 1024 * 1024,
    num_cpus=8,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
)
@remove_tpu_lockfile_on_exit
def train_model_tpu(*args, **kwargs):
    return train_model(*args, **kwargs, device="tpu")


def train_model_gpu(*args, **kwargs):
    return train_model(*args, **kwargs, device="gpu")


def train_model(
    dataset_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    hf_model: str,
    num_epochs: int,
    max_length: int = 512,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int | None = None,
    device: str = "tpu",
) -> None:
    logger.info(f"Training BERT model for experiment {output_path}")
    success_path = os.path.join(output_path, ".SUCCESS")

    if fsspec_exists(success_path):
        logger.info(f"Found success file at {success_path}, skipping training.")
        return

    train_start_time = time.time()
    gcs_checkpoint_path = os.path.join(output_path, "gcs_checkpoints")
    gcs_model_output_path = os.path.join(output_path, "model_output")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_output_path = os.path.join(tmp_dir, "model_output")
        local_trainer_output_path = os.path.join(tmp_dir, "trainer_output")
        os.makedirs(local_model_output_path, exist_ok=True)
        os.makedirs(local_trainer_output_path, exist_ok=True)

        bert_args = BertTrainingArguments(
            output_dir=local_trainer_output_path,
            remove_unused_columns=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            report_to="wandb",
            logging_steps=10,
            eval_steps=0.1,
            eval_accumulation_steps=None,
            eval_strategy="steps",
            save_steps=0.1,
            save_strategy="steps",
            gcs_checkpoint_path=gcs_checkpoint_path,
            max_length=max_length,
        )

        if device == "tpu":
            import torch_xla.distributed.xla_multiprocessing as xmp

            xmp.spawn(
                _mp_fn,
                args=(
                    hf_model,
                    dataset_path,
                    local_model_output_path,
                    bert_args,
                    url_classifier_compute_eval_metrics,
                ),
            )
        elif device == "gpu":
            # When training on GPU, we set up only one Trainer instance.
            index = 0
            _mp_fn(
                index,
                hf_model,
                dataset_path,
                local_model_output_path,
                bert_args,
                url_classifier_compute_eval_metrics,
            )
        else:
            raise ValueError(f"Got invalid value for device: {device}")

        fsspec_cpdir(local_model_output_path, gcs_model_output_path)
        try:
            fsspec_cpdir(local_trainer_output_path, os.path.join(output_path, "trainer_output"))
        except FileNotFoundError:
            logger.info(
                f"Local trainer output path {local_trainer_output_path} appears to be empty, "
                f"skipping copy. Contents: {os.listdir(local_trainer_output_path)}"
            )

    train_end_time = time.time()
    elapsed_seconds = train_end_time - train_start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    logger.info(f"Training BERT for experiment {output_path} completed; total time = {elapsed_str}.")
    with fsspec.open(success_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"training_elapsed_seconds": elapsed_seconds}, f)


@draccus.wrap()
def train_bert_url_classifier_driver(cfg: TrainBertUrlClassifierConfig):
    train_bert_url_classifier(
        urls_pattern=cfg.urls_pattern,
        fetched_urls_pattern=cfg.fetched_urls_pattern,
        output_path=cfg.output_path,
        val_frac=cfg.val_frac,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        hf_model=cfg.hf_model,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
        max_length=cfg.max_length,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
    )


if __name__ == "__main__":
    train_bert_url_classifier_driver()
