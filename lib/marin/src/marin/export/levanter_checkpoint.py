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

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any
import levanter.infra.cli_helpers
import ray
from levanter.checkpoint import discover_latest_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.infra.ray_tpu import run_on_pod_resumable
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import (
    ExecutorStep,
    InputName,
    VersionedValue,
    ensure_versioned,
    this_output_path,
)
from marin.resources import CpuOnlyConfig, ResourceConfig, TpuPodConfig
from marin.training.training import _add_default_env_variables, _add_run_env_variables

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertCheckpointStepConfig:
    """
    Configuration for converting a single Levanter checkpoint into HuggingFace format.
    """

    checkpoint_path: str | InputName | VersionedValue[str]
    trainer: TrainerConfig
    model: LmConfig
    resources: ResourceConfig = dataclasses.field(default_factory=CpuOnlyConfig)
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore[arg-type]
    upload_to_hf: bool | str | RepoRef = False
    tokenizer: str | None = None
    override_vocab_size: int | None = None
    config_overrides: dict[str, Any] | None = None
    save_tokenizer: bool = True
    use_cpu: bool = False
    discover_latest: bool = False


def convert_checkpoint_to_hf(config: ConvertCheckpointStepConfig) -> None:
    """
    Executor entry point that shells into Levanter's export_lm_to_hf utility.
    """

    default_launch_config = levanter.infra.cli_helpers.load_config()

    checkpoint_path = config.checkpoint_path  # type: ignore[assignment]
    if config.discover_latest:
        discovered = discover_latest_checkpoint(checkpoint_path)
        if not discovered:
            raise FileNotFoundError(f"Could not discover checkpoint under '{checkpoint_path}'.")
        checkpoint_path = discovered

    use_cpu = config.use_cpu or isinstance(config.resources, CpuOnlyConfig)

    convert_config = ConvertLmConfig(
        trainer=config.trainer,
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        output_dir=config.output_path,
        upload_to_hf=config.upload_to_hf,
        model=config.model,
        save_tokenizer=config.save_tokenizer,
        tokenizer=config.tokenizer,
        override_vocab_size=config.override_vocab_size,
        config_overrides=config.config_overrides,
        use_cpu=use_cpu,
    )

    env = _add_default_env_variables(
        config.resources.runtime_env.get("env_vars", {}),
        default_launch_config.env_for_accel(config.resources.accelerator_descriptor() or ""),
    )
    env = _add_run_env_variables(env)

    hw_config = config.resources.with_env_vars(env)

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def convert_task():
        export_lm_to_hf.main(convert_config)

    if isinstance(hw_config, TpuPodConfig):
        assert (hw_config.slice_count == 1, "Export currently works on single slices at present.")
        return run_on_pod_resumable(convert_task, config.resources.accelerator_descriptor(), max_retries_failure=10)

    return ray.get(convert_task.remote())


def convert_checkpoint_to_hf_step(
    name: str,
    checkpoint_path: InputName | str,
    *,
    trainer: TrainerConfig,
    model: LmConfig,
    resources: ResourceConfig | None = None,
    upload_to_hf: bool | str | RepoRef = False,
    tokenizer: str | None = None,
    override_vocab_size: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    save_tokenizer: bool = True,
    use_cpu: bool = False,
    override_output_path: str | None = None,
    pip_dependency_groups: list[str] | None = None,
    discover_latest: bool = False,
) -> ExecutorStep:
    """
    Creates an ExecutorStep that materializes a HuggingFace checkpoint from a saved Levanter checkpoint.

    Args:
        name: Step name. Commonly prefixed with ``hf/`` to keep outputs organized.
        checkpoint_path: Path (or InputName) pointing to a Levanter checkpoint directory, e.g.
            ``train_step.cd("checkpoints/ckpt-210388")``.
        trainer: TrainerConfig that matches the topology the checkpoint was saved with.
        model: Model configuration that produced the checkpoint.
        resources: Hardware resources to use when running the conversion. Defaults to CPU-only execution.
        upload_to_hf: Optional HuggingFace repo reference (bool, repo-id string, or RepoRef).
        tokenizer: Optional tokenizer override. Defaults to the tokenizer specified by ``model``.
        override_vocab_size: If provided, resizes the vocabulary before exporting.
        config_overrides: Optional dict merged into the HF config prior to saving.
        save_tokenizer: Whether to emit tokenizer files alongside the model weights.
        use_cpu: Force conversion to run on CPU instead of the configured device mesh. When False, CPU mode is enabled
            automatically if the provided resources do not expose an accelerator.
        override_output_path: Explicit output path override. Useful when aligning with pre-existing directories.
        pip_dependency_groups: Optional executor dependency groups. Defaults to ``["tokenize_train"]`` so the
            Levanter/JAX stack is available.
        discover_latest: If True, resolves ``checkpoint_path`` to the most recent checkpoint in that directory.
    """

    checkpoint_value: InputName | VersionedValue[str]
    if isinstance(checkpoint_path, InputName):
        checkpoint_value = checkpoint_path
    else:
        checkpoint_value = ensure_versioned(checkpoint_path)

    config = ConvertCheckpointStepConfig(
        checkpoint_path=checkpoint_value,
        trainer=trainer,
        model=model,
        resources=resources or CpuOnlyConfig(),
        upload_to_hf=upload_to_hf,
        tokenizer=tokenizer,
        override_vocab_size=override_vocab_size,
        config_overrides=config_overrides,
        save_tokenizer=save_tokenizer,
        use_cpu=use_cpu,
        discover_latest=discover_latest,
    )

    dependency_groups = pip_dependency_groups or ["tokenize_train"]

    return ExecutorStep(
        name=name,
        fn=convert_checkpoint_to_hf,
        config=config,
        override_output_path=override_output_path,
        pip_dependency_groups=dependency_groups,
    )
