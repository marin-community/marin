"""
training.py

Train BERT models.
"""

import logging
import os
import tempfile
from datetime import datetime

import ray
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

from marin.classifiers.bert.utils import BertDataset, format_example
from marin.classifiers.utils import merge_shards_and_split, shuffle
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob


def train_epochs(
    model: BertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int,
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
        total_loss = 0
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")


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
    tokenizer = BertTokenizer.from_pretrained(hf_model)
    train_dataset = BertDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    device = xm.xla_device()
    device_loader = pl.MpDeviceLoader(train_loader, device)

    model = BertForSequenceClassification.from_pretrained(hf_model, num_labels=train_dataset.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    xm.broadcast_master_param(model)

    train_epochs(model, optimizer, device_loader, num_epochs)

    if index == 0:
        xm.save(model.state_dict(), save_path)
    return True


def train_model(
    input_path: str,
    output_path: str,
    seed: int,
    val_split: float,
    memory_req: int = 10,
    batch_size: int = 1,
    lr: float = 2e-5,
    hf_model: str = "bert-base-uncased",
    num_epochs: int = 1,
) -> None:
    """
    Train a fastText model.

    Args:
        input_path (str): Path for input training data.
        output_path (str): Path to save the trained model (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        hf_model (str): Pretrained BERT model to use (from Huggingface).
        num_epochs (int): Number of epochs to train for.

    Returns:
        None: No return value.
    """
    logger = logging.getLogger("bert")

    logger.info(f"Training BERT model for experiment {output_path}")
    datetime_start = datetime.utcnow()

    # run training on remote worker, not head node
    @ray.remote(
        memory=memory_req * 1024 * 1024 * 1024,
        resources={"TPU": 4},
    )
    def run():
        if fsspec_exists(f"{output_path}/model.bin"):
            logger.info(f"Model already exists at {output_path}/model.bin. Skipping training.")
            return True

        shard_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards_and_split(shard_paths, train_path, val_path, val_split, seed, format_example)
            shuffle(train_path, train_path, seed)

            xmp.spawn(_mp_fn, args=(hf_model, train_path, model_path, lr, batch_size, num_epochs))

            fsspec_cpdir(tmp_dir, output_path)

        return True

    response = run.remote()
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise

    datetime_end = datetime.utcnow()
    logger.info(f"Training BERT for experiment {output_path} completed in {datetime_end - datetime_start}.")

    return None
