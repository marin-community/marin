"""
train_fasttext.py

Training script for fastText quality classifiers.
"""

import os
from dataclasses import dataclass, field

import draccus

from marin.classifiers.fasttext.training import train_model
from marin.classifiers.utils import DatasetConfig, attributes_to_dataset, label_docs
from marin.utils import fsspec_rm


@dataclass
class TrainFasttextClassifierConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        datasets (list[DatasetConfig]): List of configurations for converting Dolma documents into
            labeled training datasets.
        fasttext_args (dict): Arguments for the fastText training process (see fastText docs for list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
    """

    output_path: str | None = field(default=None)
    datasets: list[DatasetConfig] = field(default_factory=list)
    fasttext_args: dict = field(default_factory=dict)
    seed: int = 0
    val_frac: float = 0.1
    memory: int = 1


def train(cfg: TrainFasttextClassifierConfig):
    for dataset in cfg.datasets:
        attr_path = os.path.join(cfg.output_path, "tmp")
        label_docs(
            input_doc_path=dataset.input_doc_path,
            output_attr_path=attr_path,
            attribute_func=lambda _, dataset=dataset: {"label": dataset.label},
        )
        attributes_to_dataset(
            output_path=cfg.output_path,
            doc_path=dataset.input_doc_path,
            attr_path=attr_path,
            sampling_rate=dataset.sampling_rate,
            seed=cfg.seed,
            max_sample_size=dataset.max_sample_size,
        )
        fsspec_rm(attr_path)

    input_dataset_path = os.path.join(cfg.output_path, "data")
    train_model(
        input_path=input_dataset_path,
        output_path=cfg.output_path,
        seed=cfg.seed,
        val_frac=cfg.val_frac,
        memory_req=cfg.memory,
        **cfg.fasttext_args,
    )


@draccus.wrap()
def main(cfg: TrainFasttextClassifierConfig):
    train(cfg)


if __name__ == "__main__":
    main()
