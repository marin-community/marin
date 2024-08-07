"""
train_model.py

Trains a fastText model on a dataset (e.g., a result of running create_dataset.py).
"""

from dataclasses import dataclass
import os
import random
from datetime import datetime
from typing import List

import draccus
import fsspec
import ray
import logging

from marin.utils import fsspec_glob
import fasttext

def merge_shards(shard_paths: List[str], train_path: str, val_path: str, val_split: float, seed: int) -> bool:
    """
    Merges multiple shard files into training and validation datasets.

    Args:
        shard_paths (List[str]): List of paths to shard files.
        train_path (str): Path to the output training dataset file.
        val_path (str): Path to the output validation dataset file.
        val_split (float): Fraction of data to be used for validation.
        seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        bool: True if the process is successful.
    """
    random.seed(seed)
    with fsspec.open(train_path, "wt") as f_train, fsspec.open(val_path, "wt") as f_val:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression = "gzip") as f_in:
                for line in f_in:
                    if random.random() < val_split:
                        f_val.write(line)
                    else:
                        f_train.write(line)

    return True

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        path (str): Base path for input and output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        training_args (dict): Arguments for the fastText training process (see fastText docs for the full list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
        num_cpus (int): Number of CPUs allocated for remote training process.
    """
    path: str
    experiment: str
    training_args: dict
    seed: int
    val_split: float
    memory: int
    num_cpus: int

logger = logging.getLogger("ray")

@draccus.wrap()
def main(cfg: MainConfig):
    ray.init()

    logger.info(f"Training fasText model for experiment {cfg.experiment}")
    datetime_start = datetime.utcnow()

    cfg.training_args['thread'] = cfg.num_cpus # tell fasttext trainer to use all available CPUs

    # run training on remote worker, not head node
    @ray.remote(memory=cfg.memory * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=cfg.num_cpus)
    def run(cfg):
        experiment_dir = f'{cfg.path}/classifiers/{cfg.experiment}'
        shard_paths = fsspec_glob(os.path.join(f'{experiment_dir}/data', "**/*.jsonl.gz"))
        merge_shards(shard_paths,"data.train","data.val",cfg.val_split,cfg.seed)

        model = fasttext.train_supervised("data.train",**cfg.training_args)
        model.save_model("model.bin")

        # fasttext can't handle gs:// paths, so we copy everything from local worker disk to experiment directory at the end
        with fsspec.open("model.bin","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/model.bin',"wb") as f_out:
                f_out.write(f_in.read())
        with fsspec.open("data.train","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/data.train',"wb") as f_out:
                f_out.write(f_in.read())
        with fsspec.open("data.val","rb") as f_in, \
            fsspec.open(f'{experiment_dir}/data.val',"wb") as f_out:
                f_out.write(f_in.read())
        
        return True
    
    response = run.remote(cfg)
    try:
        ray.get(response)
    except Exception as e:
        print(f"Error processing: {e}")
    
    datetime_end = datetime.utcnow()
    logger.info(f"Training fastText for experiment {cfg.experiment} completed in {datetime_end - datetime_start}.")


if __name__ == '__main__':
    main()