# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass

import ray
from fray.cluster import ResourceConfig
from fray.cluster.ray import as_remote_kwargs
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

    resource_config = launch_config.resource_config

    @ray.remote(**as_remote_kwargs(resource_config))
    def train_classifier_distributed(config: HFTrainingConfig):
        dataset = load_dataset(config.train_dataset, "train")
        dataset = dataset.train_test_split(train_size=config.train_size, seed=42)
        xmp.spawn(train_classifier, args=(config, dataset["train"], dataset["test"]))

    ray.get(train_classifier_distributed.remote(launch_config.training_config))
