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

def train_model(
        experiment_path: str, 
        seed: int, 
        val_split: float, 
        memory_req: int,
        **fasttext_args: dict
    ) -> None:
    """
    Train a fastText model.

    Args:
        experiment_path (str): Path for input (i.e., training data) and output (i.e., trained model) data (i.e., gs://{BUCKET}).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        fasttext_args (dict): Arguments for the fastText training process (see fastText docs for the full list of options).
    
    Returns:
        None: No return value.
    """
    logger = logging.getLogger("ray")

    logger.info(f"Training fastText model for experiment {experiment_path}")
    datetime_start = datetime.utcnow()

    num_cpus = fasttext_args['thread'] if 'thread' in fasttext_args else 1

    # run training on remote worker, not head node
    @ray.remote(memory=memory_req * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs","fasttext"]}, num_cpus=num_cpus)
    def run():
        import fasttext

        shard_paths = fsspec_glob(os.path.join(f'{experiment_path}/data', "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards_and_split(shard_paths,train_path,val_path,val_split,seed,format_example)
            shuffle(train_path,train_path,seed)

            model = fasttext.train_supervised(train_path,**fasttext_args)
            model.save_model(model_path)

            fs = fsspec.core.get_fs_token_paths(experiment_path, mode="wb")[0]
            fs.put(os.path.join(tmp_dir, "*"), experiment_path, recursive=True)
        
        return None
    
    response = run.remote()
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise
    
    datetime_end = datetime.utcnow()
    logger.info(f"Training fastText for experiment {experiment_path} completed in {datetime_end - datetime_start}.")

    return True