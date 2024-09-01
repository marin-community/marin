
import os
import logging
from datetime import datetime
import tempfile

import fsspec
import ray
import json

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from marin.utils import fsspec_glob
from marin.classifiers.utils import merge_shards_and_split, shuffle
from marin.classifiers.bert.utils import format_example, BertDataset

def train_epochs(
        model: BertForSequenceClassification, 
        optimizer: torch.optim.Optimizer, 
        data_loader: torch.utils.data.DataLoader, 
        num_epochs: int
    ) -> bool:
    """
    Train a model for a number of epochs.

    Attributes:
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
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)

            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}')

# TODO: document and also **model_args
def _mp_fn(index,train_loader,bert_model,lr,num_epochs,model_path,train_dataset):
    device = xm.xla_device()
    device_loader = pl.MpDeviceLoader(train_loader, device)

    model = BertForSequenceClassification.from_pretrained(bert_model,num_labels=train_dataset.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    xm.broadcast_master_param(model)

    train_epochs(model, optimizer, device_loader, num_epochs)

    if index == 0:
        xm.save(model.state_dict(), model_path)
    return True

def train_model(
        base_path: str, 
        experiment: str, 
        seed: int, 
        val_split: float, 
        memory_req: int = 10,
        batch_size: int = 1,
        lr: float = 2e-5,
        bert_model: str = 'bert-base-uncased',
        num_epochs: int = 1
    ) -> bool:
    """
    Train a fastText model.

    Attributes:
        base_path (str): Base path for input and output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        num_cpus (int): Number of CPUs allocated for remote training process.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        bert_model (str): Pretrained BERT model to use.
        num_epochs (int): Number of epochs to train for.
    
    Returns:
        bool: True if the process is successful.
    """
    logger = logging.getLogger("ray")

    logger.info(f"Training BERT model for experiment {experiment}")
    datetime_start = datetime.utcnow()

    # run training on remote worker, not head node
    @ray.remote(memory=memory_req * 1024 * 1024 * 1024, resources={"TPU": 4},)
    def run():
        experiment_path = f'{base_path}/classifiers/{experiment}'
        shard_paths = fsspec_glob(os.path.join(f'{experiment_path}/data', "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards_and_split(shard_paths,train_path,val_path,val_split,seed,format_example)

            shuffle(train_path,train_path,seed)
            shuffle(val_path,val_path,seed)

            tokenizer = BertTokenizer.from_pretrained(bert_model)
            train_dataset = BertDataset(train_path, tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)

            xmp.spawn(_mp_fn, args=(train_loader,bert_model,lr,num_epochs,model_path,train_dataset))

            fs = fsspec.core.get_fs_token_paths(experiment_path, mode="wb")[0]
            fs.put(os.path.join(tmp_dir, "*"), experiment_path, recursive=True)
        
        return True
    
    response = run.remote()
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise
    
    datetime_end = datetime.utcnow()
    logger.info(f"Training BERT for experiment {experiment} completed in {datetime_end - datetime_start}.")

    return True