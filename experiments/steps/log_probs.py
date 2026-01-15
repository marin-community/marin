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
Step wrappers for evaluating log probabilities using Levanter.

This module provides step definitions that wrap the library functions in
marin.evaluation.log_probs.
"""

from fray.cluster import ResourceConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

from marin.evaluation.log_probs import EvalLmConfig
from marin.evaluation.log_probs import evaluate_lm_log_probs as _evaluate_lm_log_probs
from marin.execution import StepRef, deferred, output, step
from marin.utilities.executor_utils import ckpt_path_to_step_name

# Mark library function as deferred
evaluate_lm_log_probs = deferred(_evaluate_lm_log_probs)


def _extract_name_from_checkpoint(checkpoint: str | StepRef) -> str:
    """
    Extract a name from a checkpoint path or StepRef for use in step naming.

    Args:
        checkpoint: Either a string path or a StepRef

    Returns:
        A derived name suitable for use in step names
    """
    if isinstance(checkpoint, str):
        return ckpt_path_to_step_name(checkpoint)
    elif isinstance(checkpoint, StepRef):
        # For StepRef, use the step's name
        return checkpoint.name.split("/")[-1]
    else:
        raise ValueError(f"Unknown type for checkpoint: {type(checkpoint)}")


@step(name="analysis/log_probs/{name}")
def log_probs_step(
    name: str,
    checkpoint: StepRef,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    wandb_tags: list[str] | None = None,
):
    """
    Create a step that evaluates log probabilities from a checkpoint step.

    Use this when the checkpoint is an output of another step.
    For raw checkpoint paths, use log_probs_from_path instead.

    Args:
        name: Name for this evaluation step
        checkpoint: The checkpoint step to evaluate
        model: The model configuration
        data: The data to evaluate on
        resource_config: The resource configuration
        checkpoint_is_hf: Whether the checkpoint is in HF format
        per_device_batch_size: Batch size per device
        max_samples_per_dataset: Optional limit on samples per dataset
        wandb_tags: Optional tags to add to the wandb run
    """
    return evaluate_lm_log_probs(
        EvalLmConfig(
            name=name,
            checkpoint_path=checkpoint,
            model=model,
            datasets=data,
            log_entropy=True,
            resource_config=resource_config,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
            output_path=output(),
        )
    )


@step(name="analysis/log_probs/{name}")
def log_probs_from_path(
    name: str,
    checkpoint_path: str,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    wandb_tags: list[str] | None = None,
):
    """
    Create a step that evaluates log probabilities from a raw checkpoint path.

    Use this for evaluating existing checkpoint paths (GCS, local).
    For checkpoints that are outputs of another step, use log_probs_step instead.

    Args:
        name: Name for this evaluation step
        checkpoint_path: The checkpoint path to evaluate (GCS or local path)
        model: The model configuration
        data: The data to evaluate on
        resource_config: The resource configuration
        checkpoint_is_hf: Whether the checkpoint is in HF format
        per_device_batch_size: Batch size per device
        max_samples_per_dataset: Optional limit on samples per dataset
        wandb_tags: Optional tags to add to the wandb run
    """
    return evaluate_lm_log_probs(
        EvalLmConfig(
            name=name,
            checkpoint_path=checkpoint_path,
            model=model,
            datasets=data,
            log_entropy=True,
            resource_config=resource_config,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
            output_path=output(),
        )
    )


def default_lm_log_probs(
    checkpoint: str | StepRef,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> StepRef:
    """
    Creates a step to evaluate log probabilities of a language model.

    This is a convenience wrapper that automatically chooses between log_probs_step
    and log_probs_from_path based on the type of the checkpoint parameter.

    Args:
        checkpoint: The checkpoint to evaluate (StepRef or path string)
        model: The model configuration
        data: The data to evaluate on
        resource_config: The resource configuration
        checkpoint_is_hf: Whether the checkpoint is in HF format
        per_device_batch_size: Batch size per device
        max_samples_per_dataset: Optional limit on samples per dataset
        name: Optional name for the step (auto-generated from checkpoint if not provided)
        wandb_tags: Optional tags to add to the wandb run
    """
    if not name:
        name = _extract_name_from_checkpoint(checkpoint)

    if isinstance(checkpoint, StepRef):
        return log_probs_step(
            name=name,
            checkpoint=checkpoint,
            model=model,
            data=data,
            resource_config=resource_config,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
        )
    else:
        return log_probs_from_path(
            name=name,
            checkpoint_path=checkpoint,
            model=model,
            data=data,
            resource_config=resource_config,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
        )
