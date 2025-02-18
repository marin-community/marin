from dataclasses import dataclass, field
from typing import Any

import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed


@dataclass
class ScriptArguments(TrainingArguments):
    model_name: str = field(default="Alibaba-NLP/gte-base-en-v1.5")
    max_length: int = field(default=8192)
    train_dataset: str = field(default="", metadata={"help": "Path to the training dataset"})
    # Use a regression task for now
    num_labels: int = field(default=1)
    target_column: str = field(default="label")
    output_dir: str = field(default="", metadata={"help": "Path to the output directory"})


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


def load_dataset(input_path: str, split: str):
    dataset = datasets.load_dataset("json", data_files=input_path, split=split)
    print(dataset)
    return dataset


def train_classifier(rank: int, args: ScriptArguments):
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, trust_remote_code=True, num_labels=args.num_labels, output_hidden_states=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(args.train_dataset, "train")
    dataset = dataset.train_test_split(train_size=0.9, seed=42)
    trainer = Trainer(
        model,
        args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollator(args, tokenizer),
    )
    trainer.train()

    trainer.save_model(args.output_dir)


# def train_classifier_distributed(args: ScriptArguments):
#     import torch_xla.distributed.xla_multiprocessing as xmp

#     xmp.spawn(train_classifier, args=(args,), start_method="fork")
