#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass

import draccus
import fsspec
import ray
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm_loggable.auto import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from marin.classifiers.bert.utils import BertDataset
from marin.classifiers.utils import shuffle
from marin.utils import fsspec_exists, fsspec_glob, remove_tpu_lockfile_on_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainBertUrlClassifierConfig:
    input_pattern: str
    output_path: str
    val_frac: float = 0.1
    batch_size: int = 1
    lr: float = 2e-5
    hf_model: str = "bert-base-uncased"
    num_epochs: int = 1
    seed: int = 0


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def make_url_classification_dataset(
    input_pattern: str, train_output_path: str, val_output_path: str, val_frac: float, seed: int
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
    shard_paths: list[str] = list(fsspec_glob(input_pattern))
    logger.info(f"Found {len(shard_paths)} shards to process")
    shard_paths = sorted(shard_paths)
    logger.info(f"Sorted {len(shard_paths)} shards")

    # First pass: Count total lines
    total_lines = 0
    for shard_path in tqdm(shard_paths, desc="Counting lines in input records"):
        with fsspec.open(shard_path, mode="rt", compression="infer", block_size=1 * 1024 * 1024 * 1024) as f:
            for _ in f:
                total_lines += 1
    logger.info(f"Total lines across all shards = {total_lines:,}")
    assert total_lines > 0

    # Compute how many lines go to validation (integer)
    val_lines_target = int(total_lines * val_frac)
    logger.info(f"Target # of validation lines = {val_lines_target:,}")

    # Second pass: Assign each line to train or val
    lines_left = total_lines
    val_left = val_lines_target

    # Store the number of train/val examples written to sanity-check
    train_examples_written = 0
    val_examples_written = 0
    with (
        fsspec.open(train_output_path, "w", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as f_train,
        fsspec.open(val_output_path, "w", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as f_val,
    ):
        for shard_path in tqdm(shard_paths, desc="Converting shards to examples"):
            with fsspec.open(shard_path, mode="rt", compression="infer", block_size=1 * 1024 * 1024 * 1024) as f:
                for line in f:
                    record = json.loads(line)
                    out = {"text": record["metadata"]["url"], "label": record["metadata"]["passes_all_filters"]}

                    # Probability that this line should go to val
                    # Ensures we end with exactly val_lines_target lines in val
                    if random.random() < (val_left / lines_left) if lines_left else 0:
                        f_val.write(json.dumps(out) + "\n")
                        val_left -= 1
                        val_examples_written += 1
                    else:
                        f_train.write(json.dumps(out) + "\n")
                        train_examples_written += 1

                    lines_left -= 1

    assert val_examples_written == val_lines_target
    assert (train_examples_written + val_examples_written) == total_lines
    logger.info("Shuffling training dataset")
    shuffle(train_output_path, train_output_path, seed=seed)
    logger.info("Shuffling validation dataset")
    shuffle(val_output_path, val_output_path, seed=seed)

    # Write success files
    with fsspec.open(train_success_path, "w", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"examples_written": train_examples_written}, f)
    with fsspec.open(val_success_path, "w", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"examples_written": val_examples_written}, f)
    logger.info("Finished writing train and validation splits.")


def train_bert_url_classifier(
    input_pattern: str,
    output_path: str,
    val_frac: float,
    batch_size: int,
    lr: float,
    hf_model: str,
    num_epochs: int,
    seed: int = 0,
):
    # Hash this dataset configuration so we can skip dataset generation if it already exists.
    dataset_hash = hashlib.md5(f"{input_pattern}{val_frac}{seed}".encode())
    train_dataset_path = os.path.join(output_path, "data", f"train_.{dataset_hash}jsonl.gz")
    val_dataset_path = os.path.join(output_path, "data", f"val_{dataset_hash}.jsonl.gz")
    _ = ray.get(
        make_url_classification_dataset.remote(
            input_pattern=input_pattern,
            train_output_path=train_dataset_path,
            val_output_path=val_dataset_path,
            val_frac=val_frac,
            seed=seed,
        )
    )

    _ = ray.get(
        train_model.remote(
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            output_path=output_path,
            batch_size=batch_size,
            lr=lr,
            hf_model=hf_model,
            num_epochs=num_epochs,
        )
    )


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
)
@remove_tpu_lockfile_on_exit
def train_model(
    train_dataset_path: str,
    val_dataset_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    hf_model: str,
    num_epochs: int,
) -> None:
    logger.info(f"Training BERT model for experiment {output_path}")
    success_path = os.path.join(output_path, ".SUCCESS")

    if fsspec_exists(success_path):
        logger.info(f"Found success file at {success_path}, skipping training.")
        return

    train_start_time = time.time()
    xmp.spawn(_mp_fn, args=(hf_model, train_dataset_path, val_dataset_path, output_path, lr, batch_size, num_epochs))
    train_end_time = time.time()

    elapsed_seconds = train_end_time - train_start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    logger.info(f"Training BERT for experiment {output_path} completed; total time = {elapsed_str}.")
    with fsspec.open(success_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024) as f:
        json.dump({"training_elapsed_seconds": elapsed_seconds}, f)


def train_epochs(
    model: BertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    index: int | None = None,
):
    """
    Train a model for a number of epochs.

    Args:
        model (BertForSequenceClassification): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train for.
        index: (Optional[int]): Index of the TPU device (if called from _mp_fn).

    Returns:
        bool: True if the process is successful.
    """
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch}")
        total_loss = 0

        for t, batch in enumerate(data_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)
            xm.mark_step()

            total_loss += loss.item()
            if t % 10 == 0:
                logger.info(f"Step {t} on device {index}, Loss: {loss.item()}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")


def _mp_fn(
    index: int,
    hf_model: str,
    train_dataset_path: str,
    val_dataset_path: str,
    output_path: str,
    lr: float,
    batch_size: int,
    num_epochs: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    tokenizer = BertTokenizer.from_pretrained(hf_model)
    # Train dataset/loader
    train_dataset = BertDataset(train_dataset_path, tokenizer)
    train_sampler = DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    device = xm.xla_device()
    device_train_loader = pl.MpDeviceLoader(train_loader, device)

    # Validation dataset/loader
    val_dataset = BertDataset(val_dataset_path, tokenizer)
    val_sampler = DistributedSampler(val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    device_val_loader = pl.MpDeviceLoader(val_loader, device)

    model = BertForSequenceClassification.from_pretrained(hf_model, num_labels=train_dataset.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    xm.broadcast_master_param(model)

    # Train
    train_epochs(model, optimizer, device_train_loader, num_epochs, index)

    # Validate
    evaluate_model(model, device_val_loader, index)

    if index == 0:
        # Save entire model + tokenizer for easy reloading
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)


def evaluate_model(model: BertForSequenceClassification, data_loader, index: int | None = None) -> float:
    model.eval()
    local_correct = 0
    local_total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            local_correct += (preds == labels).sum().item()
            local_total += labels.size(0)

    # Each process now has a local_correct/local_total from just its chunk
    # Use mesh_reduce to sum across processes
    global_correct = xm.mesh_reduce("correct_reduce", local_correct, sum)
    global_total = xm.mesh_reduce("total_reduce", local_total, sum)

    if global_total == 0:
        global_accuracy = 0.0
    else:
        global_accuracy = global_correct / global_total

    # Log only on one process
    if index == 0:
        logger.info(f"Validation Accuracy (global across all TPUs) = {global_accuracy:.4f}")

    return global_accuracy


@draccus.wrap()
def train_bert_url_classifier_driver(cfg: TrainBertUrlClassifierConfig):
    train_bert_url_classifier(
        input_pattern=cfg.input_pattern,
        output_path=cfg.output_path,
        val_frac=cfg.val_frac,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        hf_model=cfg.hf_model,
        num_epochs=cfg.num_epochs,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    train_bert_url_classifier_driver()
