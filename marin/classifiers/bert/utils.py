"""
utils.py

Utility functions for training BERT models.
"""

import json

import fsspec
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def format_example(data: dict) -> str:
    """
    Converts example to BERT training data format.
    """

    example = {"text": data["text"], "label": data["label"]}

    return json.dumps(example)


class BertDataset(Dataset):
    """
    Dataset subclass for BERT quality classifier training data.
    """

    def __init__(self, dataset_path: str, tokenizer: BertTokenizer, max_len: int = 128, labels: list[str] | None = None):
        """
        __init__ method for BertDataset.

        Args:
            dataset_path (str): Path to the dataset file.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            max_len (int): Maximum length (in tokens) of sequences in dataset.
            labels (Optional[list]): List of possible labels.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        labels = [] if labels is None else labels
        self.label_set = set(labels)

        self.texts = []
        self.labels = []

        with fsspec.open(dataset_path, "rb", compression="infer", block_size=1 * 1024 * 1024 * 1024) as f:
            for line in f:
                example = json.loads(line)

                self.texts.append(example["text"])
                self.labels.append(example["label"])
                if len(labels) > 0 and example["label"] not in self.label_set:
                    raise ValueError(f"Label {example['label']} not in provided label set")
                self.label_set.add(
                    example["label"]
                )  # if input labels is not empty then this should typically be a no-op

        self.num_labels = len(self.label_set)
        self.label_set = sorted(self.label_set)
        self.label_index = {
            label: i for label, i in zip(self.label_set, range(self.num_labels), strict=False)
        }  # map label to index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        # TODO: make some of these args configurable
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.label_index[label], dtype=torch.long),
        }
