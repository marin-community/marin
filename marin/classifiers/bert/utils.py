"""
utils.py

Utility functions for training BERT models.
"""

import json
import fsspec

import torch
from torch.utils.data import Dataset

def format_example(data: dict) -> str:
    """
    Converts example to BERT training data format.
    """

    example = {
        "text": data["text"],
        "label": data["label"]
    }

    return json.dumps(example) + "\n"

# TODO: document
class BertDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128, labels=[]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_set = set(labels)

        self.construct_dataset(path)
    
    def construct_dataset(self, path):
        self.texts = []
        self.labels = []

        with fsspec.open(path, 'rb') as f:
            for line in f:
                example = json.loads(line)

                self.texts.append(example['text'])
                self.labels.append(example['label'])
                self.label_set.add(example['label'])
        
        self.num_labels = len(self.label_set)
        self.label_set = sorted(self.label_set)
        self.label_index = {label : i for label,i in zip(self.label_set,range(self.num_labels))}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_index[label], dtype=torch.long)
        }