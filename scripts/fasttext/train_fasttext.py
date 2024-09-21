"""
train_fasttext.py

Training script for fastText quality classifiers.
"""

from dataclasses import dataclass, field

import draccus
import os

from marin.utils import fsspec_rm
from marin.classifiers.utils import create_label_attribute, attributes_to_dataset
from marin.classifiers.fasttext.training import train_model


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
        fasttext_args (dict): Arguments for the fastText training process (see fastText docs for the full list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
    """

    output_path: str
    pos_doc_path: str
    neg_doc_path: str
    pos_sampling_rate: float = 1.0
    neg_sampling_rate: float = 1.0
    fasttext_args: dict = field(default_factory=dict)
    seed: int = 0
    val_frac: float = 0.1
    memory: int = 1


@draccus.wrap()
def main(cfg: TrainFasttextClassifierConfig):
    ray.init()

    pos_attr_path = os.path.join(cfg.output_path, "tmp", "positives")
    neg_attr_path = os.path.join(cfg.output_path, "tmp", "negatives")

    create_label_attribute(input_doc_path=cfg.pos_doc_path, output_attr_path=pos_attr_path, label="hq")
    attributes_to_dataset(
        output_path=cfg.output_path,
        doc_path=cfg.pos_doc_path,
        attr_path=pos_attr_path,
        sampling_rate=cfg.pos_sampling_rate,
        seed=cfg.seed,
    )

    create_label_attribute(input_doc_path=cfg.neg_doc_path, output_attr_path=neg_attr_path, label="lq")
    attributes_to_dataset(
        output_path=cfg.output_path,
        doc_path=cfg.neg_doc_path,
        attr_path=neg_attr_path,
        sampling_rate=cfg.neg_sampling_rate,
        seed=cfg.seed,
    )

    train_model(
        input_path=f"{cfg.output_path}/data",
        output_path=cfg.output_path,
        seed=cfg.seed,
        val_frac=cfg.val_frac,
        memory_req=cfg.memory,
        **cfg.fasttext_args,
    )

    fsspec_rm(pos_attr_path)
    fsspec_rm(neg_attr_path)


if __name__ == "__main__":
    main()
