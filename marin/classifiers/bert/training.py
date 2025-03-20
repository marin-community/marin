"""
training.py

Train BERT models.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime

import ray
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from marin.classifiers.bert.utils import format_example
from marin.classifiers.utils import format_dataset, merge_dataset_shards, shuffle, split_dataset
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob, fsspec_rm, remove_tpu_lockfile_on_exit

logger = logging.getLogger("ray")


@dataclass
class BertTrainingArguments:
    # Training arguments
    output_dir: str
    remove_unused_columns: bool = False
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int = 1
    report_to: str = "wandb"
    logging_steps: float = 0.1
    eval_steps: float = 0.1
    save_strategy: str = "no"

    # Collation arguments
    max_length: int = 128

    def get_hf_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            remove_unused_columns=self.remove_unused_columns,
            per_device_train_batch_size=self.per_device_train_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            report_to=self.report_to,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
        )


def _mp_fn(
    index: int,
    hf_model: str,
    train_path: str,
    val_path: str,
    model_path: str,
    bert_args: BertTrainingArguments,
):
    """
    Function to run on each TPU device for BERT classifier training.

    Args:
        index (int): Index of the TPU device.
        hf_model (str): Pretrained BERT model to use (from Huggingface).
        train_path (str): Path to the training dataset.
        val_path (str): Path to the validation dataset.
        model_path (str): Path to save the trained model.
        bert_args (BertTrainingArguments): Arguments for training the BERT model.
    Returns:
        None: No return value.
    """

    tokenizer = BertTokenizer.from_pretrained(hf_model)
    train_dataset = load_dataset("json", data_files=train_path)
    val_dataset = load_dataset("json", data_files=train_path)

    # Tokenize the dataset (padding is added by collator)
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            return_tensors="pt",
            max_length=bert_args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    model = BertForSequenceClassification.from_pretrained(hf_model, num_labels=train_dataset.num_labels)

    trainer = Trainer(
        model,
        bert_args.get_hf_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    if index == 0:
        os.makedirs(model_path, exist_ok=True)
        model.cpu().save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)


def train_model(
    input_path: str,
    output_path: str,
    seed: int,
    val_frac: float,
    memory_req: int = 10,
    batch_size: int = 64,
    lr: float = 2e-5,
    hf_model: str = "bert-base-uncased",
    num_epochs: int = 1,
    max_length: int = 128,
    dataloader_num_workers: int = 8,
    dataloader_prefetch_factor: int = 4,
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
        max_length (int): Maximum sequence length for training.
        dataloader_num_workers (int): Number of workers for data loading.
        dataloader_prefetch_factor (int): Prefetch factor for data loading.
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
        if fsspec_exists(f"{output_path}/model"):
            logger.info(f"Model already exists at {output_path}/model. Skipping training.")
            return

        shard_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
        logger.info(f"Received input paths: {shard_paths}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_path = os.path.join(tmp_dir, "data.full")
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")

            model_path = os.path.join(tmp_dir, "model")
            trainer_path = os.path.join(tmp_dir, "trainer_output")

            merge_dataset_shards(shard_paths, merge_path)
            format_dataset(merge_path, format_example)
            split_dataset(merge_path, train_path, val_path, val_frac, seed)
            shuffle(train_path, train_path, seed)
            shuffle(val_path, val_path, seed)

            os.makedirs(trainer_path, exist_ok=True)
            bert_args = BertTrainingArguments(
                output_dir=trainer_path,
                remove_unused_columns=False,
                per_device_train_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                dataloader_num_workers=dataloader_num_workers,
                dataloader_prefetch_factor=dataloader_prefetch_factor,
                report_to="wandb",
                logging_steps=0.1,
                eval_steps=0.1,
                save_strategy="no",
                max_length=max_length,
            )

            import torch_xla.distributed.xla_multiprocessing as xmp

            xmp.spawn(_mp_fn, args=(hf_model, train_path, val_path, model_path, bert_args))

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
