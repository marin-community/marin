import logging
import tempfile
from dataclasses import dataclass

import ray

from experiments.evals.resource_configs import ResourceConfig
from marin.classifiers.hf.train_classifier import HFTrainingConfig, load_dataset, train_classifier

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    training_config: HFTrainingConfig
    resource_config: ResourceConfig


def launch_training_with_ray(launch_config: LaunchConfig):
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        xmp = None
        logger.warning("torch_xla is not installed, so we will not be able to train the quality filter.")

    # NOTE(chris): Important to set the PJRT_DEVICE or else sometimes it won't launch correctly because it
    # does not recognize that there is a TPU device available. Also, must use the TPU-v6e-8-head or else it
    # may not recognize the topology of the device correctly.
    @ray.remote(
        num_cpus=8,
        resources={"TPU": launch_config.resource_config.num_tpu, f"{launch_config.resource_config.tpu_type}-head": 1},
        runtime_env={"env_vars": {"PJRT_DEVICE": "TPU"}},
        memory=64 * 1024 * 1024 * 1024,
    )
    def train_classifier_distributed(config: HFTrainingConfig):
        # No validation dataset provided, so we split the training dataset into train and val
        if not config.val_dataset:
            dataset = load_dataset(config.train_dataset, "train")
            dataset = dataset.train_test_split(train_size=config.train_size, seed=42)
            train_dataset = dataset["train"]
            val_dataset = dataset["test"]
        else:
            train_dataset = load_dataset(config.train_dataset, "train")
            val_dataset = load_dataset(config.val_dataset, "train")

        with tempfile.TemporaryDirectory() as local_output_dir:
            xmp.spawn(train_classifier, args=(config, train_dataset, val_dataset, local_output_dir))

    ray.get(train_classifier_distributed.remote(launch_config.training_config))
