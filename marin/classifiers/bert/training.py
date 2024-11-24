"""
training.py

Train BERT models.
"""

import logging
import os
import tempfile
import time
from datetime import datetime

import ray
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from marin.classifiers.bert.utils import BertDataset, format_example
from marin.classifiers.utils import format_dataset, merge_shards, shuffle, split_dataset
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob, fsspec_rm, remove_tpu_lockfile_on_exit

logger = logging.getLogger("ray")


def train_epochs(
    model: BertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    index: int,
) -> bool:
    """
    Train a model for a number of epochs.

    Args:
        model (BertForSequenceClassification): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train for.

    Returns:
        bool: True if the process is successful.
    """
    model.train()
    for epoch in range(num_epochs):
        logger.info("Training epoch %d", epoch)
        total_loss = 0
        start = time.time()
        recent_losses = []
        recent_times = []
        recent_load_times = []
        stats_window = 1

        device = xm.xla_device()

        for batch, t in zip(data_loader, range(len(data_loader)), strict=False):
            if t > 0:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

        logger.info("starting actual TRAINING")
        iteration_start_time = time.time()
        for _batch, t in zip(data_loader, range(len(data_loader)), strict=False):
            # for t in range(len(data_loader)):

            iteration_load_time = time.time() - iteration_start_time
            recent_load_times.append(iteration_load_time)

            input_ids = _batch["input_ids"]  # .to(device)
            attention_mask = _batch["attention_mask"]  # .to(device)
            labels = _batch["labels"]  # .to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)

            # iteration_step_time = time.time()

            loss = torch.tensor(0.0)
            total_loss += loss.item()

            recent_losses.append(loss.item())
            iteration_total_time = time.time() - iteration_start_time
            recent_times.append(iteration_total_time)

            if len(recent_losses) > stats_window:
                recent_losses.pop(0)
                recent_times.pop(0)
                recent_load_times.pop(0)

            if t % stats_window == 0:
                # logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {t}/{len(data_loader)}, Loss: {loss.item():.4f}")

                avg_time = sum(recent_times) / len(recent_times)
                # max_time = max(recent_times)
                # min_time = min(recent_times)

                avg_load_time = sum(recent_load_times) / len(recent_load_times)
                # max_load_time = max(recent_load_times)
                # min_load_time = min(recent_load_times)

                logger.info(
                    f"Total Time: {avg_time:.4f} s, Load Time: {avg_load_time:.4f} s, Iteration: {t}, \
                        Device: {index}, Shape: {input_ids.shape}"
                )

                # logger.info(
                #     f"Step taken at time \
                # {datetime.fromtimestamp(iteration_step_time).strftime('%Y-%m-%d %H:%M:%S.%f')} \
                #     for Iteration {t} on Device {index}"
                # )

                # logger.info(
                #     f"Model parameter at Iteration {t} on Device {index} \
                # is {model.state_dict()['classifier.weight'][0][0]}"
                # )

            if t >= 50:
                return

            iteration_start_time = time.time()

        end = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")
        print(f"Took {end - start} time")


def _mp_fn(index, hf_model, train_path, save_path, lr, batch_size, num_epochs):
    """
    Function to run on each TPU device for BERT classifier training.

    Args:
        index (int): Index of the TPU device.
        hf_model (str): Pretrained BERT model to use (from Huggingface).
        train_path (str): Path to the training dataset.
        save_path (str): Path to save the trained model.
        lr (float): Learning rate for training.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train for.

    Returns:
        bool: True if the process is successful.
    """
    logger.info(f"Training on TPU device {index}")

    tokenizer = BertTokenizer.from_pretrained(hf_model)
    train_dataset = BertDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    logger.info("Loading model and optimizer")

    device = xm.xla_device()
    device_loader = pl.MpDeviceLoader(train_loader, device)

    model = BertForSequenceClassification.from_pretrained(hf_model, num_labels=train_dataset.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    xm.broadcast_master_param(model)

    logger.info("Model loaded")

    train_epochs(model, optimizer, device_loader, num_epochs, index)

    if index == 0:
        xm.save(model.state_dict(), save_path)


def train_model(
    input_path: str,
    output_path: str,
    seed: int,
    val_frac: float,
    memory_req: int = 10,
    batch_size: int = 1,
    lr: float = 2e-5,
    hf_model: str = "bert-base-uncased",
    num_epochs: int = 1,
) -> None:
    """
    Train a BERT model.

    Args:
        input_path (str): Path for input training data.
        output_path (str): Path to save the trained model (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        hf_model (str): Pretrained BERT model to use (from Huggingface).
        num_epochs (int): Number of epochs to train for.

    Returns:
        None: No return value.
    """

    logger.info(f"Training BERT model for experiment {output_path}")
    datetime_start = datetime.utcnow()

    # run training on remote worker, not head node
    @ray.remote(
        memory=memory_req * 1024 * 1024 * 1024,
        resources={"TPU": 4, "TPU-v4-8-head": 1},
    )
    @remove_tpu_lockfile_on_exit
    def run():
        if fsspec_exists(f"{output_path}/model.bin"):
            logger.info(f"Model already exists at {output_path}/model.bin. Skipping training.")
            # return

        shard_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
        logger.info(f"Received input paths: {shard_paths}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "data.full")
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards(shard_paths, merge_path)
            format_dataset(merge_path, format_example)
            split_dataset(merge_path, train_path, val_path, val_frac, seed)
            shuffle(train_path, train_path, seed)

            logger.info("About to spawn XMP processes")
            logger.info(f"TPU ENV: {[env for env in os.environ if 'TPU' in env]}")
            xmp.spawn(_mp_fn, args=(hf_model, train_path, model_path, lr, batch_size, num_epochs))

            logger.info("Spawning XMP processes complete")

            fsspec_rm(merge_path)
            fsspec_cpdir(tmp_dir, output_path)

    response = run.remote()
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise

    datetime_end = datetime.utcnow()
    logger.info(f"Training BERT for experiment {output_path} completed in {datetime_end - datetime_start}.")
