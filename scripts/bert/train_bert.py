"""
train_bert.py

Training script for BERT quality classifiers.
"""

from dataclasses import dataclass, field

import ray
import draccus

from marin.classifiers.utils import create_label_attribute, attribute_to_dataset
from marin.classifiers.bert.training import train_model

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        output_base_path (str): Base path for output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment name.
        pos_doc_path (str): Path to experiment with positive examples (i.e., gs://{BUCKET}/documents/../$EXPERIMENT).
        neg_doc_path (str): Path to experiment with negative examples (i.e., gs://{BUCKET}/documents/../$EXPERIMENT).
        pos_sampling_rate (float): Fraction of positive examples to include the training dataset.
        neg_sampling_rate (float): Fraction of negative examples to include the training dataset.
        bert_args (dict): Arguments for the BERT training process.
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
        num_cpus (int): Number of CPUs allocated for remote training process.
    """
    output_base_path: str
    experiment: str
    pos_doc_path: str
    neg_doc_path: str
    pos_sampling_rate: float = 1.0
    neg_sampling_rate: float = 1.0
    bert_args: dict = field(default_factory=dict)
    seed: int = 0
    val_split: float = 0.1
    memory: int = 1

def get_attr_path(doc_path: str, attr_experiment: str) -> str:
    """
    Utility function to get the attribute experiment path for a given document experiment path.
    """
    base_path,experiment_path = doc_path.split('/documents/')
    doc_experiment = experiment_path.split('/')[-2]

    return f"{base_path}/attributes/{experiment_path.split(doc_experiment)[0]}{attr_experiment}"

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()

    pos_attr_path = get_attr_path(cfg.pos_doc_path, cfg.experiment)
    neg_attr_path = get_attr_path(cfg.neg_doc_path, cfg.experiment)

    create_label_attribute(input_doc_path=cfg.pos_doc_path, output_attr_path=pos_attr_path, label="hq")
    attribute_to_dataset(output_base_path=cfg.output_base_path, experiment=cfg.experiment, doc_path=cfg.pos_doc_path, attr_path=pos_attr_path, sampling_rate=cfg.pos_sampling_rate, seed=cfg.seed)

    create_label_attribute(input_doc_path=cfg.neg_doc_path, output_attr_path=neg_attr_path, label="lq")
    attribute_to_dataset(output_base_path=cfg.output_base_path, experiment=cfg.experiment, doc_path=cfg.neg_doc_path, attr_path=neg_attr_path, sampling_rate=cfg.neg_sampling_rate, seed=cfg.seed)

    train_model(base_path=cfg.output_base_path, experiment=cfg.experiment, seed=cfg.seed, val_split=cfg.val_split, memory_req=cfg.memory, **cfg.bert_args)

if __name__ == '__main__':
    main()