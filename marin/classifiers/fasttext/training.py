"""
training.py

Train fastText models.
"""

import logging
import os
import tempfile
from datetime import datetime

import ray

from marin.classifiers.fasttext.utils import format_example
from marin.classifiers.utils import merge_shards_and_split, shuffle
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob


def train_model(
    input_path: str, output_path: str, seed: int, val_split: float, memory_req: int, **fasttext_args: dict
) -> None:
    """
    Train a fastText model.

    Args:
        input_path (str): Path for input training data.
        output_path (str): Path to save the trained model (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_split (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        fasttext_args (dict): Arguments for the fastText training process
            (see fastText docs for the full list of options).

    Returns:
        None: No return value.
    """
    logger = logging.getLogger("ray")

    logger.info(f"Training fastText model for experiment {output_path}")
    datetime_start = datetime.utcnow()

    num_cpus = fasttext_args["thread"] if "thread" in fasttext_args else 1

    # run training on remote worker, not head node
    @ray.remote(memory=memory_req * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs", "fasttext"]}, num_cpus=num_cpus)
    def run():
        if fsspec_exists(f"{output_path}/model.bin"):
            logger.info(f"Model already exists at {output_path}/model.bin. Skipping training.")
            return True

        import fasttext

        shard_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = os.path.join(tmp_dir, "data.train")
            val_path = os.path.join(tmp_dir, "data.val")
            model_path = os.path.join(tmp_dir, "model.bin")

            merge_shards_and_split(shard_paths, train_path, val_path, val_split, seed, format_example)
            shuffle(train_path, train_path, seed)

            model = fasttext.train_supervised(train_path, **fasttext_args)
            model.save_model(model_path)

            fsspec_cpdir(tmp_dir, output_path)

        return True

    response = run.remote()
    try:
        ray.get(response)
    except Exception as e:
        logger.exception(f"Error processing: {e}")
        raise

    datetime_end = datetime.utcnow()
    logger.info(f"Training fastText for experiment {output_path} completed in {datetime_end - datetime_start}.")

    return None
