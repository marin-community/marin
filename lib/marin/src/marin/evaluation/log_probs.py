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
import hashlib
import os
import shutil
from dataclasses import dataclass

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from fray.cluster.base import TpuConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.main.eval_lm import EvalLmConfig as LevanterEvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.main.eval_ensemble import EvalEnsembleConfig as LevanterEvalEnsembleConfig
from levanter.main.eval_ensemble import main as eval_ensemble_main
from levanter.models.lm_model import LmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.evaluation.utils import discover_levanter_checkpoints, download_from_gcs, is_remote_path
from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name

HUGGINGFACE_CACHE_PATH = "/tmp/huggingface-cache"
GCSFUSE_MOUNT_POINT = "/opt/gcsfuse_mount"


def _wandb_id(name: str) -> str:
    digest = hashlib.md5(name.encode()).hexdigest()[:8]
    base = name[-52:] if len(name) > 52 else name
    return f"{base}-{digest}"


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

@dataclass
class EvalEnsembleLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    checkpoint_paths: list[str | InputName]
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore
    run_prefix: str = "ppl-eval"
    key: str = None,
    checkpoint_is_hf: bool = False
    """Whether the checkpoint is in HF format."""

    log_entropy: bool = False
    """ Whether to log entropies of the model. """

    max_samples_per_dataset: int | None = None
    resource_overrides: dict[str, float] | None = None
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

def default_ensemble_log_probs(
    checkpoints: list[str | InputName],
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    run_prefix: str = "ppl-eval",
    name_prefix: str = "ensemble-",
    key: str = None,
    resource_overrides: dict[str, float] | None = None,
) -> ExecutorStep:
    """
    Creates a step to evaluate log probabilities of an ensemble language model.
    Args:
        checkpoint:  The checkpoint to evaluate.
        model:  The model configuration.
        data: The data to evaluate on.
        checkpoint_is_hf:  Whether the checkpoint is in HF format.
    """
    name = ckpt_path_to_step_name(checkpoints[0])
    name = f"analysis/log_probs/data-efficiency/{name_prefix}{len(checkpoints)}x-{name}"
    resources = resource_overrides.copy() if resource_overrides is not None else {"TPU": 4, "TPU-v4-8-head": 1}
    return ExecutorStep(
        name=name,
        fn=evaluate_ensemble_log_probs,
        config=EvalEnsembleLmConfig(
            checkpoint_paths=checkpoints,  # type: ignore
            model=model,
            datasets=data,
            log_entropy=False,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            run_prefix=run_prefix,
            key=key,
            resource_overrides=resources,
        ),
    )


def do_eval_lm(config: LevanterEvalLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """
    try:
        local_path = None
        # for hf checkpoints, levanter can read hf://, gs:// directly
        # but for non-gcs hf checkpoints, we download to gcs fuse for now.
        if config.hf_checkpoint and is_remote_path(config.hf_checkpoint.model_name_or_path):
            pass
        elif config.hf_checkpoint:
            # Use GCSFuse directly so that we don't have to download the checkpoint to the local filesystem
            checkpoint_ref = str(config.hf_checkpoint)
            local_path = os.path.join(config.local_model_dir, ckpt_path_to_step_name(checkpoint_ref))
            download_from_gcs(
                gcs_path=checkpoint_ref,
                destination_path=local_path,
            )
            config.hf_checkpoint = RepoRef.from_string(local_path)
            print(f"Downloaded model checkpoint to {local_path}: {os.listdir(local_path)}")
        elif config.checkpoint_path and is_remote_path(config.checkpoint_path):
            local_path = os.path.join(config.local_model_dir, ckpt_path_to_step_name(config.checkpoint_path))
            download_from_gcs(
                gcs_path=config.checkpoint_path,
                destination_path=local_path,
            )
            config.checkpoint_path = discover_levanter_checkpoints(local_path)[-1]
        eval_lm_main(config)
    finally:
        if config.hf_checkpoint:
            hf_checkpoint_path = str(config.hf_checkpoint)
            if not os.path.exists(hf_checkpoint_path):
                shutil.rmtree(HUGGINGFACE_CACHE_PATH, ignore_errors=True)
                print(f"Deleted HuggingFace cache at {HUGGINGFACE_CACHE_PATH}.")
        if local_path and not local_path.startswith(GCSFUSE_MOUNT_POINT):
            shutil.rmtree(local_path, ignore_errors=True)
            print(f"Deleted local checkpoint at {local_path}.")

def do_eval_ensemble(config: LevanterEvalEnsembleConfig) -> None:
    """
    Evaluate log probabilities of a language model on a mixture, and optionally entropies.
    """
    eval_ensemble_main(config)

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
            tracker=WandbConfig(project="suhas-eval-data-efficiency", tags=wandb_tags, name=name, id=_wandb_id(name)),
            # tracker=WandbConfig(project="marin", tags=wandb_tags, name=name),
            ray=RayConfig(auto_start_cluster=False),
            per_device_eval_parallelism=config.per_device_batch_size,
            max_eval_batches=max_eval_batches,
        ),
        log_entropy=config.log_entropy,
    )

    assert isinstance(config.resource_config.device, TpuConfig), "evaluate_lm_log_probs requires TPU resource config"

    env_vars = {"HF_HOME": HUGGINGFACE_CACHE_PATH}
    cluster = current_cluster()
    job_request = JobRequest(
        name=f"eval-lm-{name}",
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_eval_lm, args=[levanter_config]),
        environment=EnvironmentConfig.create(env_vars=env_vars),
    )
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


def evaluate_ensemble_log_probs(config: EvalEnsembleLmConfig) -> None:
    """
    Evaluate log probabilities of a language model on a mixture, and optionally entropies.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """

    name = os.path.basename(config.output_path)
    name = f"{config.run_prefix}-{name}"

    if config.max_samples_per_dataset is None:
        max_eval_batches = None
    else:
        max_eval_batches = config.max_samples_per_dataset // config.per_device_batch_size

    levanter_config = LevanterEvalEnsembleConfig(
        checkpoint_paths=config.checkpoint_paths if not config.checkpoint_is_hf else [],
        hf_checkpoints=config.checkpoint_paths if config.checkpoint_is_hf else None,
        model=config.model,
        data=config.datasets,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="suhas-eval-data-efficiency",
                tags=["eval_lm"] + [config.key] if config.key else [],
                name=name,
                id=_wandb_id(name),
            ),
            ray=RayConfig(auto_start_cluster=False),
            per_device_eval_parallelism=config.per_device_batch_size,
            max_eval_batches=max_eval_batches,
        ),
        log_entropy=config.log_entropy,
    )
    
    assert isinstance(config.resource_config.device, TpuConfig), "evaluate_ensemble_log_probs requires TPU resource config"

    env_vars = {"HF_HOME": HUGGINGFACE_CACHE_PATH}
    cluster = current_cluster()
    job_request = JobRequest(
        name=name,
        resources=config.resource_config,
        entrypoint=Entrypoint.from_callable(do_eval_ensemble, args=[levanter_config]),
        environment=EnvironmentConfig.create(env_vars=env_vars),
    )
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
