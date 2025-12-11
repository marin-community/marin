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

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from marin.classifiers.hf.train_classifier import HFTrainingConfig, load_dataset, train_classifier

logger = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    training_config: HFTrainingConfig
    resource_config: ResourceConfig


def _train_classifier_distributed(config: HFTrainingConfig):
    """Inner training function that runs on TPU."""
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError as e:
        raise RuntimeError("torch_xla is not installed, so we cannot train the quality filter.") from e

    dataset = load_dataset(config.train_dataset, "train")
    dataset = dataset.train_test_split(train_size=config.train_size, seed=42)
    xmp.spawn(train_classifier, args=(config, dataset["train"], dataset["test"]))


def launch_training_with_ray(launch_config: LaunchConfig):
    """Launch HuggingFace classifier training via Fray.

    Despite the name (kept for backward compatibility), this now uses Fray's
    JobRequest pattern instead of Ray directly.
    """
    job_request = JobRequest(
        name="train-hf-classifier",
        entrypoint=Entrypoint.from_callable(
            _train_classifier_distributed,
            args=[launch_config.training_config],
        ),
        resources=launch_config.resource_config,
        environment=EnvironmentConfig.create(),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
