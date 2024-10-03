"""
train_fasttext.py

Training script for fastText quality classifiers.
"""

from dataclasses import dataclass, field

import draccus

from marin.classifiers.fasttext.training import train_model
from marin.classifiers.utils import attributes_to_dataset
from marin.processing.classification.types import DatasetFormat


@dataclass(frozen=True)
class DatasetCurationConfig:
    """Configuration for curating a dataset for training a quality classfier

    Attributes:
        input_doc_path (str): Path to the input dataset which can be a directory or a file.
            If it is a directory, the function will glob all the files in the directory and sample from each file.
            The files can be formatted in jsonl or fasttext format.
        label (str): Label for the dataset. This should be in the format "__label__<label>"
            where <label> is the label for the dataset. For example, "__label__hq" or "__label__lq", respectively.
        absolute_sampling_rate (Optional[int]): Number of examples to sample from the dataset where each example
            is sampled with probability 1/N.
        relative_sampling_rate (Optional[float]): Fraction of the dataset to sample.
        format (DatasetFormat): Format of the dataset.
    """

    input_doc_path: str
    label: str
    format: DatasetFormat
    absolute_sampling_rate: int | None = None
    relative_sampling_rate: float | None = None


@dataclass
class TrainFasttextClassifierConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        input_doc_paths (list[DatasetCurationConfig]): List of configurations for converting input datasets into
            labeled datasets. The input datasets can be a directory or a file.
            If it is a directory, the function will glob all the files in the directory and sample from each file.
            The files can be formatted in jsonl or fasttext format.
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
            absolute_sampling_rate=input_doc_path.absolute_sampling_rate,
            relative_sampling_rate=input_doc_path.relative_sampling_rate,
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
