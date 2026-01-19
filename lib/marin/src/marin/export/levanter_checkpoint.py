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
Library code for converting Levanter checkpoints to HuggingFace format.

This module contains pure processing functions that work with concrete paths.
For step wrappers that handle dependencies, see experiments/steps/levanter_checkpoint.py
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import levanter.infra.cli_helpers
from fray.cluster import (
    CpuConfig,
    Entrypoint,
    EnvironmentConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    current_cluster,
)
from levanter.checkpoint import discover_latest_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.training.training import _add_default_env_variables, _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertCheckpointStepConfig:
    """Configuration for converting a Levanter checkpoint into HuggingFace format."""

    checkpoint_path: str
    """Path to the Levanter checkpoint directory."""

    trainer: TrainerConfig
    """TrainerConfig matching the topology the checkpoint was saved with."""

    model: LmConfig
    """Model configuration that produced the checkpoint."""

    output_path: str
    """Output directory for the HuggingFace checkpoint."""

    resources: ResourceConfig = dataclasses.field(default_factory=ResourceConfig.with_cpu)
    """Hardware resources for conversion."""

    upload_to_hf: bool | str | RepoRef = False
    """Optional HuggingFace repo reference (bool, repo-id string, or RepoRef)."""

    tokenizer: str | None = None
    """Optional tokenizer override."""

    override_vocab_size: int | None = None
    """If provided, resizes the vocabulary before exporting."""

    config_overrides: dict[str, Any] | None = None
    """Optional dict merged into the HF config prior to saving."""

    save_tokenizer: bool = True
    """Whether to emit tokenizer files alongside the model weights."""

    use_cpu: bool = False
    """Force conversion to run on CPU instead of the configured device mesh."""

    discover_latest: bool = False
    """If True, resolves checkpoint_path to the most recent checkpoint in that directory."""


def convert_checkpoint_to_hf(config: ConvertCheckpointStepConfig) -> None:
    """
    Convert a Levanter checkpoint to HuggingFace format.

    This is the library function that performs the actual conversion.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    checkpoint_path = config.checkpoint_path
    if config.discover_latest:
        discovered = discover_latest_checkpoint(checkpoint_path)
        if not discovered:
            raise FileNotFoundError(f"Could not discover checkpoint under '{checkpoint_path}'.")
        checkpoint_path = discovered

    use_cpu = config.use_cpu or isinstance(config.resources.device, CpuConfig)

    convert_config = ConvertLmConfig(
        trainer=config.trainer,
        checkpoint_path=checkpoint_path,
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
        {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    env = _add_run_env_variables(env)

    def convert_task():
        export_lm_to_hf.main(convert_config)

    def _run_with_lockfile():
        with remove_tpu_lockfile_on_exit():
            convert_task()

    if isinstance(config.resources.device, TpuConfig):
        assert config.resources.replicas == 1, "Export currently works on single slices at present."

    job_request = JobRequest(
        name="convert-checkpoint-to-hf",
        entrypoint=Entrypoint.from_callable(_run_with_lockfile),
        resources=config.resources,
        environment=EnvironmentConfig.create(env_vars=env),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)
