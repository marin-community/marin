"""
train_model.py

Trains a fastText model on a dataset (e.g., a result of running create_dataset.py).
"""

from dataclasses import dataclass
import os
from datetime import datetime
from typing import List
import tempfile

import draccus
import fsspec
import ray
import numpy as np
import logging
import json

from marin.utils import fsspec_glob
# import fasttext (TODO: add fasttext to cluster setup and just import it here instead of using runtime_env)

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
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(train_path, "wt") as f_train, fsspec.open(val_path, "wt") as f_val:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression = "gzip") as f_in:
                for input_line in f_in:
                    data = json.loads(input_line)

                    label_string = ''
                    if data["label"] is not None:
                        label_string += f' __label__{data["label"]}'
                    output_line = label_string + " " + data["text"] + "\n"

                    if rng.random() < val_split:
                        f_val.write(output_line)
                    else:
                        f_train.write(output_line)

    return True

@dataclass
class MainConfig:
    """
    Configuration class for main process.

    Attributes:
        base_path (str): Base path for input and output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        training_args (dict): Arguments for the fastText training process (see fastText docs for the full list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
        num_cpus (int): Number of CPUs allocated for remote training process.
    """
    base_path: str
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

    logger.info(f"Training fastText model for experiment {cfg.experiment}")
    datetime_start = datetime.utcnow()

    cfg.training_args['thread'] = cfg.num_cpus # tell fasttext trainer to use all available CPUs

    # run training on remote worker, not head node
    @ray.remote(memory=cfg.memory * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs","fasttext"]}, num_cpus=cfg.num_cpus)
    def run(cfg):
        import fasttext

        experiment_path = f'{cfg.base_path}/classifiers/{cfg.experiment}'
        shard_paths = fsspec_glob(os.path.join(f'{experiment_path}/data', "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            merge_shards(shard_paths,os.path.join(tmp_dir, "data.train"),os.path.join(tmp_dir, "data.val"),cfg.val_split,cfg.seed)

            model = fasttext.train_supervised(os.path.join(tmp_dir, "data.train"),**cfg.training_args)
            model.save_model(os.path.join(tmp_dir, "model.bin"))

            fs = fsspec.core.get_fs_token_paths(experiment_path, mode="wb")[0]
            fs.put(os.path.join(tmp_dir, "*"), experiment_path, recursive=True)
        
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