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
Step wrappers for converting Levanter checkpoints to HuggingFace format.

This module provides step definitions that wrap the library functions in
marin.export.levanter_checkpoint.
"""

from typing import Any

from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.execution import StepRef, deferred, output, step
from marin.export.levanter_checkpoint import ConvertCheckpointStepConfig
from marin.export.levanter_checkpoint import convert_checkpoint_to_hf as _convert_checkpoint_to_hf

# Mark library function as deferred
convert_checkpoint_to_hf = deferred(_convert_checkpoint_to_hf)


@step(name="{name}")
def convert_checkpoint_to_hf_step(
    name: str,
    checkpoint_path: str,
    trainer: TrainerConfig,
    model: LmConfig,
    resources: ResourceConfig | None = None,
    upload_to_hf: bool | str | RepoRef = False,
    tokenizer: str | None = None,
    override_vocab_size: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    save_tokenizer: bool = True,
    use_cpu: bool = False,
    discover_latest: bool = False,
):
    """
    Create a step that converts a Levanter checkpoint to HuggingFace format.

    Args:
        name: Step name. Commonly prefixed with "hf/" to keep outputs organized.
        checkpoint_path: Path to the Levanter checkpoint directory.
        trainer: TrainerConfig matching the topology the checkpoint was saved with.
        model: Model configuration that produced the checkpoint.
        resources: Hardware resources for conversion. Defaults to CPU-only.
        upload_to_hf: Optional HuggingFace repo reference (bool, repo-id string, or RepoRef).
        tokenizer: Optional tokenizer override.
        override_vocab_size: If provided, resizes the vocabulary before exporting.
        config_overrides: Optional dict merged into the HF config prior to saving.
        save_tokenizer: Whether to emit tokenizer files alongside the model weights.
        use_cpu: Force conversion to run on CPU instead of the configured device mesh.
        discover_latest: If True, resolves checkpoint_path to the most recent checkpoint.
    """
    return convert_checkpoint_to_hf(
        ConvertCheckpointStepConfig(
            checkpoint_path=checkpoint_path,
            trainer=trainer,
            model=model,
            output_path=output(),
            resources=resources or ResourceConfig.with_cpu(),
            upload_to_hf=upload_to_hf,
            tokenizer=tokenizer,
            override_vocab_size=override_vocab_size,
            config_overrides=config_overrides,
            save_tokenizer=save_tokenizer,
            use_cpu=use_cpu,
            discover_latest=discover_latest,
        )
    )


@step(name="{name}")
def convert_checkpoint_from_step(
    name: str,
    training_step: StepRef,
    checkpoint_subpath: str,
    trainer: TrainerConfig,
    model: LmConfig,
    resources: ResourceConfig | None = None,
    upload_to_hf: bool | str | RepoRef = False,
    tokenizer: str | None = None,
    override_vocab_size: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    save_tokenizer: bool = True,
    use_cpu: bool = False,
    discover_latest: bool = False,
):
    """
    Create a step that converts a checkpoint from a training step to HuggingFace format.

    This is for converting checkpoints that are outputs of another step.
    Use convert_checkpoint_to_hf_step for raw paths.

    Args:
        name: Step name. Commonly prefixed with "hf/" to keep outputs organized.
        training_step: The training step whose checkpoint to convert.
        checkpoint_subpath: Subpath within the training step output (e.g., "checkpoints/step-80000").
        trainer: TrainerConfig matching the topology the checkpoint was saved with.
        model: Model configuration that produced the checkpoint.
        resources: Hardware resources for conversion. Defaults to CPU-only.
        upload_to_hf: Optional HuggingFace repo reference (bool, repo-id string, or RepoRef).
        tokenizer: Optional tokenizer override.
        override_vocab_size: If provided, resizes the vocabulary before exporting.
        config_overrides: Optional dict merged into the HF config prior to saving.
        save_tokenizer: Whether to emit tokenizer files alongside the model weights.
        use_cpu: Force conversion to run on CPU instead of the configured device mesh.
        discover_latest: If True, resolves checkpoint_path to the most recent checkpoint.
    """
    checkpoint_ref = training_step / checkpoint_subpath
    return convert_checkpoint_to_hf(
        ConvertCheckpointStepConfig(
            checkpoint_path=checkpoint_ref,
            trainer=trainer,
            model=model,
            output_path=output(),
            resources=resources or ResourceConfig.with_cpu(),
            upload_to_hf=upload_to_hf,
            tokenizer=tokenizer,
            override_vocab_size=override_vocab_size,
            config_overrides=config_overrides,
            save_tokenizer=save_tokenizer,
            use_cpu=use_cpu,
            discover_latest=discover_latest,
        )
    )
