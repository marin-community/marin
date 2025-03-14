"""Train an encoder model based on HuggingFace's Trainer API.

This code was adapted from the FinewebEdu training script.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import datasets
import fsspec
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from marin.evaluation.utils import upload_to_gcs
from marin.utils import fsspec_glob


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


def compute_metrics(eval_pred):
    # NOTE(chris): lazy import because main training cluster may not have these dependencies
    import evaluate
    from sklearn.metrics import classification_report, confusion_matrix

    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def load_dataset(input_path: str, split: str):
    # Check if input_path is a directory or file
    fs = fsspec.core.url_to_fs(input_path)[0]
    if fs.isdir(input_path):
        # If directory, get all jsonl.gz files
        data_files = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
    else:
        # If file, use directly
        data_files = input_path

    dataset = datasets.load_dataset("json", data_files=data_files, split=split)
    return dataset


def train_classifier(rank: int, hf_script_args: HFTrainingConfig, train_dataset, eval_dataset):
    # NOTE(chris): We NEED to instantiate the ScriptArugments here because instantiating it will
    # result in usage of the XLA backend, which will not allow us to call xmp.spawn.
    args = ScriptArguments(
        model_name=hf_script_args.model_name,
        max_length=hf_script_args.max_length,
        train_dataset=hf_script_args.train_dataset,
        num_labels=hf_script_args.num_labels,
        target_column=hf_script_args.target_column,
        output_dir=hf_script_args.output_dir,
        tpu_num_cores=hf_script_args.tpu_num_cores,
        remove_unused_columns=False,
        per_device_train_batch_size=hf_script_args.per_device_train_batch_size,
        # NOTE(chris): gradient accumulation steps actually slow down training A LOT for TPUs
        # gradient_accumulation_steps=16,
        report_to="wandb",
        logging_steps=hf_script_args.logging_steps,
        eval_steps=hf_script_args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=hf_script_args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        run_name=hf_script_args.run_name,
    )

    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, trust_remote_code=True, num_labels=args.num_labels, output_hidden_states=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollator(args, tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if rank == 0:
        # NOTE(chris): Cannot run trainer.save_model() because the model is located on the TPU
        # This will lead to a RuntimeError and hang the program without us knowing why.

        with tempfile.TemporaryDirectory() as temp_dir:
            model.cpu().save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            upload_to_gcs(temp_dir, args.output_dir)
