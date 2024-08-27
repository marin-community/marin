"""
training.py

Train fastText models.
"""
import os
import logging
from datetime import datetime
import tempfile

import fsspec
import ray

from marin.utils import fsspec_glob
from marin.classifiers.utils import merge_shards_and_split, shuffle
from marin.classifiers.fasttext.utils import format_example

def train_model(base_path: str, experiment: str, training_args: dict, seed: int, val_split: float, memory_req: int, num_cpus: int) -> bool:
    """
    Train a fastText model.

    Attributes:
        base_path (str): Base path for input and output data (i.e., gs://{BUCKET}).
        experiment (str): Experiment identifier.
        training_args (dict): Arguments for the fastText training process (see fastText docs for the full list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        num_cpus (int): Number of CPUs allocated for remote training process.
    
    Returns:
        bool: True if the process is successful.
    """
    logger = logging.getLogger("ray")

    logger.info(f"Training fastText model for experiment {experiment}")
    datetime_start = datetime.utcnow()

    training_args['thread'] = num_cpus # tell fasttext trainer to use all available CPUs

    # run training on remote worker, not head node
    @ray.remote(memory=memory_req * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs","fasttext"]}, num_cpus=num_cpus)
    def run(base_path, experiment, training_args, seed, val_split):
        import fasttext

        experiment_path = f'{base_path}/classifiers/{experiment}'
        shard_paths = fsspec_glob(os.path.join(f'{experiment_path}/data', "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards_and_split(shard_paths,train_path,val_path,val_split,seed,format_example)

            shuffle(train_path,train_path,seed)
            shuffle(val_path,val_path,seed)

            model = fasttext.train_supervised(train_path,**training_args)
            model.save_model(model_path)

            fs = fsspec.core.get_fs_token_paths(experiment_path, mode="wb")[0]
            fs.put(os.path.join(tmp_dir, "*"), experiment_path, recursive=True)
        
        return True
    
    response = run.remote(base_path, experiment, training_args, seed, val_split)
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise
    
    datetime_end = datetime.utcnow()
    logger.info(f"Training fastText for experiment {experiment} completed in {datetime_end - datetime_start}.")

    return True