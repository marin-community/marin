"""Train an encoder model based on HuggingFace's Trainer API.

This code was adapted from the FinewebEdu training script.
"""

import glob
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import datasets
import fsspec
import fsspec.generic
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from marin.evaluation.utils import upload_to_gcs

logger = logging.getLogger(__name__)


def find_and_download_latest_checkpoint(gcs_checkpoint_path: str, local_tmp_dir: str, rank: int) -> str | None:
    """
    Finds the latest checkpoint in a GCS path, downloads it to a local directory,
    and returns the local path.

    Args:
        gcs_checkpoint_path (str): The GCS path where checkpoints are stored.
        local_tmp_dir (str): A local temporary directory to download the checkpoint to.
        rank (int): The rank of the current process.

    Returns:
        Optional[str]: The local checkpoint directory to resume from, or
        None if no valid checkpoint is found.
    """
    if not gcs_checkpoint_path:
        return None

    fs = fsspec.core.url_to_fs(gcs_checkpoint_path)[0]
    checkpoint_glob_pattern = f"{gcs_checkpoint_path.rstrip('/')}/checkpoint-*"

    try:
        checkpoint_dirs = fs.glob(checkpoint_glob_pattern)
        checkpoint_dirs = [p.rstrip("/") for p in checkpoint_dirs]
        # ensure it ends with 'checkpoint-<step>'
        checkpoint_dirs = [p for p in checkpoint_dirs if re.match(r".*checkpoint-\d+$", p)]

        if not checkpoint_dirs:
            logger.info(f"Rank {rank} - no directories match pattern {checkpoint_glob_pattern}")
            return None

        checkpoint_dirs.sort(key=lambda x: int(x.split("checkpoint-")[-1]))
        latest_ckpt = checkpoint_dirs[-1]

        # Sanity-check that 'pytorch_model.bin' (or another required file) is present
        try:
            remote_files = fs.ls(latest_ckpt)
            remote_files_basenames = [os.path.basename(f) for f in remote_files]
            if "model.safetensors" in remote_files_basenames:
                # Download it to a local directory
                ckpt_name = os.path.basename(latest_ckpt)
                local_ckpt_dir = os.path.join(local_tmp_dir, ckpt_name)
                logger.info(f"Rank {rank} - found latest checkpoint {latest_ckpt}, downloading to {local_ckpt_dir}...")
                fs.get(latest_ckpt, local_ckpt_dir, recursive=True)
                logger.info(f"Rank {rank} - will resume from local checkpoint {local_ckpt_dir}")
                return local_ckpt_dir
            else:
                logger.info(f"Rank {rank} - latest checkpoint {latest_ckpt} is missing model.safetensors")
                return None
        except Exception as e:
            logger.warning(f"Rank {rank} - Error listing/checking checkpoint dir {latest_ckpt}: {e}")
            return None
    except Exception as e:
        logger.warning(f"Rank {rank} - error scanning for checkpoints in {gcs_checkpoint_path}: {e}")
        return None


class GCSCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        gcs_output_dir: str,
        rank: int,
    ) -> None:
        self.gcs_output_dir = gcs_output_dir
        self.rank = rank
        if self.rank == 0:
            logger.info(f"Creating output directory {gcs_output_dir}...")
            # Use fsspec to create directory implicitly by writing a file
            with fsspec.open(os.path.join(gcs_output_dir, ".keep"), "w") as f:
                json.dump({"creation_time": time.time()}, f)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called every time a checkpoint is saved locally.
        We'll:
          1. Upload the new checkpoint to GCS
          2. Remove any older checkpoints on GCS that were rotated out locally
        """
        if not state.is_world_process_zero:
            return

        try:
            logger.info(f"TPU worker {self.rank} - rsyncing {args.output_dir} to {self.gcs_output_dir}")
            fsspec.generic.rsync(args.output_dir, self.gcs_output_dir, delete_missing=True)
        except Exception as ex:
            logger.error(f"TPU worker {self.rank} - Error rsyncing {args.output_dir} to {self.gcs_output_dir}: {ex}")


@dataclass
class ScriptArguments(TrainingArguments):
    model_name: str = field(default="Alibaba-NLP/gte-base-en-v1.5")
    max_length: int = field(default=8192)
    train_dataset: str = field(default="", metadata={"help": "Path to the training dataset"})
    # Use a regression task for now
    num_labels: int = field(default=1)
    target_column: str = field(default="label")
    output_dir: str = field(default="", metadata={"help": "Path to the output directory"})


@dataclass
class HFTrainingConfig:
    model_name: str = field(default="Alibaba-NLP/gte-base-en-v1.5")
    max_length: int = field(default=8192)
    train_dataset: str = field(default="", metadata={"help": "Path to the training dataset"})
    val_dataset: str = field(default="", metadata={"help": "Path to the validation dataset"})
    # Use a regression task for now
    num_labels: int = field(default=1)
    target_column: str = field(default="label")
    output_dir: str = field(default="", metadata={"help": "Path to the output directory"})
    tpu_num_cores: int = field(default=1, metadata={"help": "Number of TPU cores"})
    train_size: float = field(default=0.9, metadata={"help": "Fraction of training data to use"})
    eval_steps: int = field(default=200, metadata={"help": "Number of evaluation steps"})
    save_steps: int = field(default=200, metadata={"help": "Number of save steps"})
    logging_steps: int = field(default=50, metadata={"help": "Number of logging steps"})
    per_device_train_batch_size: int = field(default=128, metadata={"help": "Batch size per device"})
    run_name: str = field(default="", metadata={"help": "Name of the run"})
    min_label: int = field(default=0, metadata={"help": "Minimum label value"})
    max_label: int = field(default=5, metadata={"help": "Maximum label value"})
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.10)
    learning_rate: float = field(default=5e-5)
    save_total_limit: int = field(default=1, metadata={"help": "Number of total checkpoints to save"})
    load_best_model_at_end: bool = field(
        default=False, metadata={"help": "Whether to load the best model at the end of training"}
    )
    per_device_eval_batch_size: int = field(default=128, metadata={"help": "Batch size per device for evaluation"})


class DataCollator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = self.args.max_length

    @torch.no_grad()
    def __call__(self, items) -> dict[str, Any]:
        batch = self.tokenizer(
            [item["text"] for item in items],
            truncation=True,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
        )

        labels = torch.tensor([item["label"] for item in items])

        batch["labels"] = labels
        return batch


def create_compute_metrics_fn(min_label: int, max_label: int):
    def compute_metrics(eval_pred):
        # NOTE(chris): lazy import because main training cluster may not have these dependencies
        import evaluate
        from sklearn.metrics import classification_report, confusion_matrix

        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")

        logits, labels = eval_pred
        preds = np.round(logits.squeeze()).clip(min_label, max_label).astype(int)
        labels = np.round(labels.squeeze()).astype(int)
        precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
        recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

        # Metrics for classifying labels >= 3
        # Binary: 1 if label >= 3, else 0
        labels_3_plus = (labels >= 3).astype(int)
        preds_3_plus = (preds >= 3).astype(int)
        precision_3_plus = precision_metric.compute(
            predictions=preds_3_plus, references=labels_3_plus, average="binary"
        )["precision"]
        recall_3_plus = recall_metric.compute(predictions=preds_3_plus, references=labels_3_plus, average="binary")[
            "recall"
        ]
        accuracy_3_plus = accuracy_metric.compute(predictions=preds_3_plus, references=labels_3_plus)["accuracy"]

        report = classification_report(labels, preds)
        cm = confusion_matrix(labels, preds)
        print("Validation Report:\n" + report)
        print("Confusion Matrix:\n" + str(cm))

        return {
            "precision": precision,
            "recall": recall,
            "f1_macro": f1,
            "accuracy": accuracy,
            "precision_3_plus": precision_3_plus,
            "recall_3_plus": recall_3_plus,
            "accuracy_3_plus": accuracy_3_plus,
        }

    return compute_metrics


def load_dataset(input_path: str, split: str):
    import tempfile

    from datasets import Features, Value

    features = Features({"text": Value("string"), "label": Value("int32")})

    # Create a temporary directory and copy the entire input path to it
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use fsspec to copy the entire directory/file to temp location
        fs = fsspec.core.url_to_fs(input_path)[0]
        temp_path = os.path.join(temp_dir, "data")

        if fs.isdir(input_path):
            fs.get(input_path, temp_path, recursive=True)
            # Get all jsonl.gz files from the temp directory
            data_files = glob.glob(os.path.join(temp_path, "**/*.jsonl.gz"), recursive=True)
        else:
            # If it's a file, copy just the file
            fs.get(input_path, temp_path)
            data_files = temp_path

        # Load dataset without cache
        dataset = datasets.load_dataset(
            "json",
            data_files=data_files,
            split=split,
            features=features,
            keep_in_memory=True,
            num_proc=16,
        )
        return dataset


def train_classifier(rank: int, hf_script_args: HFTrainingConfig, train_dataset, eval_dataset, local_output_dir: str):
    # NOTE(chris): We NEED to instantiate the ScriptArugments here because instantiating it will
    # result in usage of the XLA backend, which will not allow us to call xmp.spawn.
    resume_from_checkpoint = find_and_download_latest_checkpoint(
        gcs_checkpoint_path=hf_script_args.output_dir, local_tmp_dir=local_output_dir, rank=rank
    )

    args = ScriptArguments(
        model_name=hf_script_args.model_name,
        max_length=hf_script_args.max_length,
        train_dataset=hf_script_args.train_dataset,
        num_labels=hf_script_args.num_labels,
        target_column=hf_script_args.target_column,
        output_dir=local_output_dir,
        tpu_num_cores=hf_script_args.tpu_num_cores,
        remove_unused_columns=False,
        per_device_train_batch_size=hf_script_args.per_device_train_batch_size,
        per_device_eval_batch_size=hf_script_args.per_device_eval_batch_size,
        # NOTE(chris): gradient accumulation steps actually slow down training A LOT for TPUs
        # gradient_accumulation_steps=16,
        report_to="wandb",
        logging_steps=hf_script_args.logging_steps,
        eval_strategy="steps",
        eval_steps=hf_script_args.eval_steps,
        save_strategy="steps",
        save_steps=hf_script_args.save_steps,
        load_best_model_at_end=hf_script_args.load_best_model_at_end,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        run_name=hf_script_args.run_name,
        lr_scheduler_type=hf_script_args.lr_scheduler_type,
        warmup_ratio=hf_script_args.warmup_ratio,
        learning_rate=hf_script_args.learning_rate,
        save_total_limit=hf_script_args.save_total_limit,
    )

    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, trust_remote_code=True, num_labels=args.num_labels, output_hidden_states=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    callbacks = []
    callbacks.append(GCSCheckpointCallback(gcs_output_dir=hf_script_args.output_dir, rank=rank))

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollator(args, tokenizer),
        compute_metrics=create_compute_metrics_fn(hf_script_args.min_label, hf_script_args.max_label),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if rank == 0:
        # NOTE(chris): Cannot run trainer.save_model() because the model is located on the TPU
        # This will lead to a RuntimeError and hang the program without us knowing why.
        # The final model is saved by the GCSCheckpointCallback on the last save.
        # Here we just save the final tokenizer.
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            model.cpu().save_pretrained(temp_dir)
            upload_to_gcs(temp_dir, hf_script_args.output_dir)
