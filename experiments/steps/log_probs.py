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
    """Extract a name from a checkpoint path or StepRef for use in step naming."""
    if isinstance(checkpoint, str):
        return ckpt_path_to_step_name(checkpoint)
    elif isinstance(checkpoint, StepRef):
        return checkpoint.name.split("/")[-1]
    else:
        raise ValueError(f"Unknown type for checkpoint: {type(checkpoint)}")


@step(name="analysis/log_probs/{name}")
def log_probs(
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
    Evaluate log probabilities from a checkpoint.

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
    if name is None:
        name = _extract_name_from_checkpoint(checkpoint)

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


# Backward compatibility alias
default_lm_log_probs = log_probs
