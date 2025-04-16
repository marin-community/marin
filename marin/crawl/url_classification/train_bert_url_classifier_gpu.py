#!/usr/bin/env python3
"""
```
pip install transformers wandb datasets filelock torch accelerate scikit-learn
export WANDB_API_KEY='ca4e321fd237f65236ab95e92724934b47264b1c'

mkdir -p /scr/biggest/nfliu/cache/hf_home/
export HF_HOME=/scr/biggest/nfliu/cache/hf_home/
```

Training BERT base uncased on open-web-math URLs

```
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    marin/crawl/url_classification/train_bert_url_classifier_gpu.py \
    --model_name_or_path "bert-base-uncased" \
    --dataset_path "./url_classification_datasets/bert-base-uncased-open-web-math-fde8ef8-10M/train_val_hf/" \
    --output_dir "./url_classification_models/bert-base-uncased-open-web-math-fde8ef8-10M-num_train_epochs-5/" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --report_to "wandb" \
    --logging_steps 10 \
    --eval_steps 0.1 \
    --eval_strategy "steps" \
    --save_steps 0.1 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_binary_f1" \
    --greater_is_better True
```


Training BERT large uncased on open-web-math URLs

```
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=4 \
    marin/crawl/url_classification/train_bert_url_classifier_gpu.py \
    --model_name_or_path "bert-large-uncased" \
    --dataset_path "./url_classification_datasets/bert-base-uncased-open-web-math-fde8ef8-10M/train_val_hf/" \
    --output_dir "./url_classification_models/bert-large-uncased-open-web-math-fde8ef8-10M-num_train_epochs-5/" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --report_to "wandb" \
    --logging_steps 10 \
    --eval_steps 0.1 \
    --eval_strategy "steps" \
    --save_steps 0.1 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_binary_f1" \
    --greater_is_better True
```
"""
import json
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import partial

from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from marin.crawl.url_classification.metrics import url_classifier_compute_eval_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    dataset_path: str | None = field(
        default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        + f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load the JSONL data and index into the DatasetDict to get a Dataset
    dataset: DatasetDict = load_from_disk(data_args.dataset_path)
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

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=dataset["train"].features["label"].num_classes
    )

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]

    if training_args.do_eval:
        if "val" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["val"]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=partial(url_classifier_compute_eval_metrics, labels2id),
        processing_class=tokenizer,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        # Saves the tokenizer too for easy upload
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # Save the label to ID mapping
        with open(os.path.join(training_args.output_dir, "label_index.json"), "w") as f:
            json.dump(labels2id, f)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
