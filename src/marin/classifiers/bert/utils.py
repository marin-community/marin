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
utils.py

Utility functions for training BERT models.
"""

import json

import fsspec
from torch.utils.data import Dataset


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

    def __init__(self, dataset_path: str, labels: list[str] | None = None):
        """
        __init__ method for BertDataset.

        Args:
            dataset_path (str): Path to the dataset file.
            labels (Optional[list]): List of possible labels.
        """
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
        return {
            "text": self.texts[idx],
            "label": self.label_index[self.labels[idx]],
        }
