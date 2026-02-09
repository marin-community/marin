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

"""
This file uses Levanter to compute validation losses and entropies.
"""

import dataclasses
import os
from dataclasses import dataclass

from fray.v1.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from fray.v1.cluster.base import TpuConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.main.eval_lm import EvalLmConfig as LevanterEvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.models.lm_model import LmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name


@dataclass
class EvalLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    name: str | None
    checkpoint_path: str
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore
    checkpoint_is_hf: bool = False
    """Whether the checkpoint is in HF format."""

    log_entropy: bool = True
    """ Whether to log entropies of the model. """

    max_samples_per_dataset: int | None = None

    wandb_tags: list[str] | None = None
    """Tags to add to the wandb run."""


def default_lm_log_probs(
    checkpoint: str | InputName,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> ExecutorStep:
    """
    Creates a step to evaluate log probabilities of a language model.
    Args:
        checkpoint:  The checkpoint to evaluate.
        model:  The model configuration.
        data: The data to evaluate on.
        resource_config: The resource configuration.
        checkpoint_is_hf:  Whether the checkpoint is in HF format.
    """
    if not name:
        name = ckpt_path_to_step_name(checkpoint)
    executor_name = f"analysis/log_probs/{name}"
    return ExecutorStep(
        name=executor_name,
        fn=evaluate_lm_log_probs,
        config=EvalLmConfig(
            name=name,
            checkpoint_path=checkpoint,  # type: ignore
            model=model,
            datasets=data,
            log_entropy=True,
            resource_config=resource_config,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
        ),
    )


def do_eval_lm(config: LevanterEvalLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """
    # Levanter can read `gs://` checkpoints directly via fsspec/tensorstore, and HF
    # checkpoints via fsspec as well. Avoid staging large directories locally.
    eval_lm_main(config)


def evaluate_lm_log_probs(config: EvalLmConfig) -> None:
    """
    Evaluate log probabilities of a language model on a mixture, and optionally entropies.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """

    if not config.name:
        name = os.path.basename(config.output_path)
        name = f"ppl-eval-{name}"
    else:
        name = config.name.replace("/", "-")

    if config.max_samples_per_dataset is None:
        max_eval_batches = None
    else:
        max_eval_batches = config.max_samples_per_dataset // config.per_device_batch_size

    wandb_tags = ["eval_lm", *(config.wandb_tags or [])]
    levanter_config = LevanterEvalLmConfig(
        checkpoint_path=config.checkpoint_path if not config.checkpoint_is_hf else None,
        hf_checkpoint=RepoRef.from_string(config.checkpoint_path) if config.checkpoint_is_hf else None,
        model=config.model,
        data=config.datasets,
        trainer=TrainerConfig(
            tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
            ray=RayConfig(auto_start_cluster=False),
            per_device_eval_parallelism=config.per_device_batch_size,
            max_eval_batches=max_eval_batches,
        ),
        log_entropy=config.log_entropy,
    )

    assert isinstance(config.resource_config.device, TpuConfig), "evaluate_lm_log_probs requires TPU resource config"

    cluster = current_cluster()
    job_request = JobRequest(
        name=f"eval-lm-{name}",
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_eval_lm, args=[levanter_config]),
        environment=EnvironmentConfig.create(),
    )
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
