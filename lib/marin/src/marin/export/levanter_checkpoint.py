# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import levanter.infra.cli_helpers
from fray import current_client
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
from levanter.compat.hf_checkpoints import RepoRef
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
from marin.training.run_environment import add_run_env_variables
from marin.training.training import _add_default_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

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

    checkpoint_path: str | InputName | VersionedValue[str]
    trainer: TrainerConfig
    model: LmConfig
    resources: ResourceConfig = dataclasses.field(default_factory=_default_export_resources)
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

    use_cpu = config.use_cpu or isinstance(config.resources.device, CpuConfig)

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
        {},
        default_launch_config.env_for_accel(config.resources.device.variant),
    )
    env = add_run_env_variables(env)

    def convert_task():
        export_lm_to_hf.main(convert_config)

    def _run_with_lockfile():
        with remove_tpu_lockfile_on_exit():
            convert_task()

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
        entrypoint=Entrypoint.from_callable(_run_with_lockfile),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


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
        resources: Hardware resources to use when running the conversion. Defaults to CPU-only
            execution (8 CPU, 64g RAM, 64g disk); override when using an accelerator.
        upload_to_hf: Optional HuggingFace repo reference (bool, repo-id string, or RepoRef).
        tokenizer: Optional tokenizer override. Defaults to the tokenizer specified by ``model``.
        override_vocab_size: If provided, resizes the vocabulary before exporting.
        config_overrides: Optional dict merged into the HF config prior to saving.
        save_tokenizer: Whether to emit tokenizer files alongside the model weights.
        use_cpu: Force conversion to run on CPU instead of the configured device mesh. When False, CPU mode is enabled
            automatically if the provided resources do not expose an accelerator.
        override_output_path: Explicit output path override. Useful when aligning with pre-existing directories.
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
        resources=resources or _default_export_resources(),
        upload_to_hf=upload_to_hf,
        tokenizer=tokenizer,
        override_vocab_size=override_vocab_size,
        config_overrides=config_overrides,
        save_tokenizer=save_tokenizer,
        use_cpu=use_cpu,
        discover_latest=discover_latest,
    )

    return ExecutorStep(
        name=name,
        fn=convert_checkpoint_to_hf,
        config=config,
        override_output_path=override_output_path,
    )
