import logging
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
        resources={"TPU": launch_config.resource_config.num_tpu, f"{launch_config.resource_config.tpu_type}-head": 1},
        runtime_env={"env_vars": {"PJRT_DEVICE": "TPU"}},
    )
    def train_classifier_distributed(config: HFTrainingConfig):
        dataset = load_dataset(config.train_dataset, "train")
        dataset = dataset.train_test_split(train_size=config.train_size, seed=42)
        xmp.spawn(train_classifier, args=(config, dataset["train"], dataset["test"]))

    ray.get(train_classifier_distributed.remote(launch_config.training_config))
