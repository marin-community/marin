# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
training.py

Train BERT models.
"""

import json
import logging
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import fsspec
import fsspec.generic
import ray
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
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
    per_device_eval_batch_size: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    dataloader_num_workers: int = 1
    dataloader_prefetch_factor: int | None = 1
    report_to: str = "wandb"
    logging_steps: float = 0.1
    eval_steps: float = 0.1
    eval_strategy: str = "steps"

    # Collation arguments
    max_length: int = 128

    # Serialization arguments
    save_strategy: str = "steps"
    save_steps: float = 0.1
    gcs_checkpoint_path: str | None = None
    save_total_limit: int | None = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    def get_hf_training_args(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            remove_unused_columns=self.remove_unused_columns,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            report_to=self.report_to,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
        )


def _mp_fn(
    index: int,
    hf_model: str,
    dataset_path: str,
    model_path: str,
    bert_args: BertTrainingArguments,
    compute_eval_metrics: Callable[[dict[str, int], EvalPrediction], dict] | None = None,
):
    """
    Function to run on each TPU device for BERT classifier training.

    Args:
        index (int): Index of the TPU device.
        hf_model (str): Pretrained BERT model to use (from Huggingface).
        dataset_path (str): Path to arrow dataset with training and validation splits
        model_path (str): Path to save the trained model.
        bert_args (BertTrainingArguments): Arguments for training the BERT model.
        compute_eval_metrics (Callable[[dict[str, int], EvalPrediction], dict] | None): function to use to
            compute evaluation metrics on model predictions.
    Returns:
        None: No return value.
    """
    # Load the JSONL data and index into the DatasetDict to get a Dataset
    dataset: DatasetDict = load_from_disk(dataset_path)
    class_label_feature = dataset["train"].features["label"]
    labels2id = {label: class_label_feature.str2int(label) for label in class_label_feature.names}
    logger.info(f"Labels to ID mapping: {labels2id}")

    # Print training label distribution
    train_label_counts = Counter(dataset["train"]["label"])
    logger.info("Training Label Counts (label_id -> count):")
    for label_id, count in sorted(train_label_counts.items()):
        label_name = class_label_feature.int2str(label_id)
        logger.info(f"  {label_id} ({label_name}): {count} ({count/len(dataset['train']):.2%})")

    # Print val label distribution
    val_label_counts = Counter(dataset["val"]["label"])
    logger.info("Validation Label Counts (label_id -> count):")
    for label_id, count in sorted(val_label_counts.items()):
        label_name = class_label_feature.int2str(label_id)
        logger.info(f"  {label_id} ({label_name}): {count} ({count/len(dataset['val']):.2%})")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model, num_labels=dataset["train"].features["label"].num_classes
    )
    hf_training_args = bert_args.get_hf_training_args()

    # Possibly download existing checkpoint from GCS
    local_trainer_output_dir = hf_training_args.output_dir
    resume_from_checkpoint = None

    fs = fsspec.filesystem("gs")
    if bert_args.gcs_checkpoint_path:
        checkpoint_glob_pattern = f"{bert_args.gcs_checkpoint_path.rstrip('/')}/checkpoint-*"
        try:
            # List possible checkpoint dirs in the GCS path
            checkpoint_dirs = fs.glob(checkpoint_glob_pattern)
            checkpoint_dirs = [p.rstrip("/") for p in checkpoint_dirs]
            # ensure it ends with 'checkpoint-<step>'
            checkpoint_dirs = [p for p in checkpoint_dirs if re.match(r".*checkpoint-\d+$", p)]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split("checkpoint-")[-1]))
                latest_ckpt = checkpoint_dirs[-1]

                # Sanity-check that 'pytorch_model.bin' (or another required file) is present
                try:
                    remote_files = fs.ls(latest_ckpt)
                    remote_files_basenames = [os.path.basename(f) for f in remote_files]
                    if "pytorch_model.bin" in remote_files_basenames:
                        # Download it to local trainer_output/<checkpoint_name>
                        ckpt_name = os.path.basename(latest_ckpt)
                        local_ckpt_dir = os.path.join(local_trainer_output_dir, ckpt_name)
                        logger.info(f"TPU worker {index} - found checkpoint {latest_ckpt}, downloading...")
                        fs.get(latest_ckpt, local_ckpt_dir, recursive=True)
                        resume_from_checkpoint = local_ckpt_dir
                        logger.info(f"TPU worker {index} - will resume from local checkpoint {local_ckpt_dir}")
                    else:
                        logger.info(
                            f"TPU worker {index} - latest checkpoint {latest_ckpt} is missing pytorch_model.bin?"
                        )
                except Exception as e:
                    logger.warning(f"Error listing checkpoint dir {latest_ckpt}: {e}")
            else:
                logger.info(f"TPU worker {index} - no directories match pattern {checkpoint_glob_pattern}")
        except Exception as e:
            logger.warning(f"TPU worker {index} - error scanning for checkpoints: {e}")

    class GCSCheckpointCallback(TrainerCallback):
        def __init__(
            self,
            gcs_output_dir: str,
        ) -> None:
            self.gcs_output_dir = gcs_output_dir
            if index == 0:
                logger.info(f"Creating output directory {gcs_output_dir}...")
                with fsspec.open(os.path.join(gcs_output_dir, ".keep"), "w") as f:
                    json.dump({"creation_time": time.time()}, f)

        def on_save(self, args, state, control, **kwargs):
            """
            Called every time a checkpoint is saved locally.
            We'll:
              1. Upload the new checkpoint to GCS
              2. Remove any older checkpoints on GCS that were rotated out locally
            """
            if not state.is_world_process_zero:
                return

            try:
                logger.info(f"TPU worker {index} - rsyncing {args.output_dir} to {self.gcs_output_dir}")
                fsspec.generic.rsync(args.output_dir, self.gcs_output_dir, delete_missing=True)
            except Exception as ex:
                logger.error(f"TPU worker {index} - Error rsyncing {args.output_dir} to {self.gcs_output_dir}: {ex}")

    callbacks = []
    if bert_args.gcs_checkpoint_path:
        callbacks.append(
            GCSCheckpointCallback(
                bert_args.gcs_checkpoint_path,
            )
        )
    trainer = Trainer(
        model,
        bert_args.get_hf_training_args(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks,
        compute_metrics=partial(compute_eval_metrics, labels2id) if compute_eval_metrics else None,
    )
    # Start training, resuming if we found a checkpoint
    if resume_from_checkpoint:
        logger.info(f"TPU worker {index}: beginning training from checkpoint")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        logger.info(f"TPU worker {index}: beginning training from scratch")
        trainer.train()

    if index == 0:
        os.makedirs(model_path, exist_ok=True)
        model.cpu().save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        with open(os.path.join(model_path, "label_index.json"), "w") as f:
            json.dump(labels2id, f)


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
