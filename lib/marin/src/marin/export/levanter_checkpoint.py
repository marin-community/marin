# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import levanter.infra.cli_helpers
from fray.current_client import current_client
from fray.types import (
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    create_environment,
)
from levanter.checkpoint import discover_latest_checkpoint
from levanter.compat.hf_checkpoints import DEFAULT_MAX_SHARD_SIZE, RepoRef
from levanter.main import export_lm_to_hf
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.training.run_environment import add_run_env_variables
from marin.training.training import _add_default_env_variables

logger = logging.getLogger(__name__)


# Default CPU resources for checkpoint export. The HF conversion streams one weight tensor
# at a time, so a moderate RAM/disk budget is sufficient; `ResourceConfig.with_cpu()`'s
# 128m/1g defaults are too small though.
def _default_export_resources() -> ResourceConfig:
    return ResourceConfig.with_cpu(cpu=8, ram="64g", disk="64g")


@dataclass(frozen=True)
class ConvertCheckpointStepConfig:
    """
    Configuration for converting a single Levanter checkpoint into HuggingFace format.
    """

    checkpoint_path: str
    trainer: TrainerConfig
    model: LmConfig
    checkpoint_subpath: str = "model"
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE
    resources: ResourceConfig = dataclasses.field(default_factory=_default_export_resources)
    output_path: str = ""
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

    checkpoint_path = config.checkpoint_path
    if config.discover_latest:
        discovered = discover_latest_checkpoint(checkpoint_path)
        if not discovered:
            raise FileNotFoundError(f"Could not discover checkpoint under '{checkpoint_path}'.")
        checkpoint_path = discovered

    use_cpu = config.use_cpu or isinstance(config.resources.device, CpuConfig)

    convert_config = ConvertLmConfig(
        trainer=config.trainer,
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        output_dir=config.output_path,
        upload_to_hf=config.upload_to_hf,
        checkpoint_subpath=config.checkpoint_subpath,
        max_shard_size=config.max_shard_size,
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
    env = add_run_env_variables(env)

    def convert_task():
        export_lm_to_hf.main(convert_config)

    if isinstance(config.resources.device, TpuConfig):
        assert config.resources.replicas == 1, "Export currently works on single slices at present."

    extras: list[str] = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")
    elif isinstance(config.resources.device, GpuConfig):
        extras.append("gpu")

    client = current_client()
    job_request = JobRequest(
        name="convert-checkpoint-to-hf",
        entrypoint=Entrypoint.from_callable(convert_task),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)
