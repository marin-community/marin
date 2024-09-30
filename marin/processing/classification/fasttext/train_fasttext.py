"""
train_fasttext.py

Training script for fastText quality classifiers.
"""

from dataclasses import dataclass, field
from typing import Enum

import draccus

from marin.classifiers.fasttext.training import train_model
from marin.classifiers.utils import attributes_to_dataset


class DatasetFormat(Enum):
    DOLMA_FORMATTED_JSONL = "dolma_formatted_jsonl"
    FASTTEXT = "fasttext"


@dataclass
class DatasetCurationConfig:
    input_doc_path: str
    label: str
    sampling_rate: float = 1.0
    format: DatasetFormat


@dataclass
class TrainFasttextClassifierConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        pos_doc_path (str): Path to experiment with positive examples (i.e., gs://$BUCKET/documents/../$EXPERIMENT).
        neg_doc_path (str): Path to experiment with negative examples (i.e., gs://$BUCKET/documents/../$EXPERIMENT).
        pos_sampling_rate (float): Fraction of positive examples to include the training dataset.
        neg_sampling_rate (float): Fraction of negative examples to include the training dataset.
        fasttext_args (dict): Arguments for the fastText training process (see fastText docs for list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
    """

    output_path: str
    input_doc_paths: list[DatasetCurationConfig]
    fasttext_args: dict = field(default_factory=dict)
    seed: int = 0
    val_frac: float = 0.1
    memory: int = 1


def train(cfg: TrainFasttextClassifierConfig):
    for input_doc_path in cfg.input_doc_paths:
        attributes_to_dataset(
            output_path=cfg.output_path,
            doc_path=input_doc_path.input_doc_path,
            sampling_rate=input_doc_path.sampling_rate,
            seed=cfg.seed,
            label=input_doc_path.label,
            file_format=input_doc_path.format,
        )

    train_model(
        input_path=f"{cfg.output_path}/data",
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
