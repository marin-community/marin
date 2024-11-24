"""
train_bert.py

Training script for BERT quality classifiers.
"""

import os
from dataclasses import dataclass, field

import draccus

from marin.classifiers.bert.training import train_model
from marin.classifiers.utils import DatasetConfig


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
    # for dataset in cfg.datasets:
    #     attr_path = os.path.join(cfg.output_path, "tmp")
    #     create_label_attribute(input_doc_path=dataset.input_doc_path, output_attr_path=attr_path, label=dataset.label)
    #     attributes_to_dataset(
    #         output_path=cfg.output_path,
    #         doc_path=dataset.input_doc_path,
    #         attr_path=attr_path,
    #         sampling_rate=dataset.sampling_rate,
    #         seed=cfg.seed,
    #     )
    #     fsspec_rm(attr_path)

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
