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
train_bert.py

Training script for BERT quality classifiers.
"""

import os
from dataclasses import dataclass, field

import draccus

from marin.classifiers.bert.training import train_model
from marin.classifiers.utils import CreateDatasetConfig, DatasetConfig, create_dataset


@dataclass
class TrainBertClassifierConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        datasets (list[DatasetConfig]): List of configurations for converting Dolma documents into
            labeled training datasets.
        bert_args (dict): Arguments for the fastText training process (see fastText docs for list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
    """

    output_path: str | None = field(default=None)
    datasets: list[DatasetConfig] = field(default_factory=list)
    bert_args: dict = field(default_factory=dict)
    seed: int = 0
    val_frac: float = 0.1
    memory: int = 1


def train(cfg: TrainBertClassifierConfig):
    for dataset in cfg.datasets:
        create_dataset(
            config=CreateDatasetConfig(
                input_doc_path=dataset.input_doc_path,
                output_dataset_path=cfg.output_path,
                label_func=lambda doc, attrs, dataset=dataset: dataset.label,
                seed=cfg.seed,
                sampling_rate=dataset.sampling_rate,
                max_sample_size=dataset.max_sample_size,
            )
        )

    input_dataset_path = os.path.join(cfg.output_path, "data")
    train_model(
        input_path=input_dataset_path,
        output_path=cfg.output_path,
        seed=cfg.seed,
        val_frac=cfg.val_frac,
        memory_req=cfg.memory,
        **cfg.bert_args,
    )


@draccus.wrap()
def main(cfg: TrainBertClassifierConfig):
    train(cfg)


if __name__ == "__main__":
    main()
