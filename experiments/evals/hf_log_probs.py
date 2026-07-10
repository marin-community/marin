# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log-probs eval for HuggingFace checkpoints with HF config fetched at runtime.

``default_hf_lm_log_probs`` defers the HF config fetch into the step's runtime
``fn``. The DAG is constructed without network I/O; the config is read only when
the step actually executes (which is also when the model itself is downloaded).
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from fray.types import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text.datasets import LMMixtureDatasetConfig
from marin.evaluation.log_probs import EvalLmConfig, evaluate_lm_log_probs
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import mixture
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint


@dataclass
class HfLogProbsConfig:
    """Like ``EvalLmConfig`` but the model config is fetched at runtime."""

    name: str
    hf_repo_id: str
    hf_revision: str
    checkpoint_path: str
    datasets: LMMixtureDatasetConfig
    resource_config: ResourceConfig
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default="")
    log_entropy: bool = True
    max_samples_per_dataset: int | None = None
    wandb_tags: list[str] | None = None


def evaluate_hf_log_probs(config: HfLogProbsConfig) -> None:
    """Fetch HF model config, then dispatch to ``evaluate_lm_log_probs``."""
    model_identifier = f"{config.hf_repo_id}@{config.hf_revision}"
    hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)
    eval_config = EvalLmConfig(
        name=config.name,
        checkpoint_path=config.checkpoint_path,
        model=hf_model_config,
        datasets=config.datasets,
        resource_config=config.resource_config,
        per_device_batch_size=config.per_device_batch_size,
        output_path=config.output_path,
        checkpoint_is_hf=True,
        log_entropy=config.log_entropy,
        max_samples_per_dataset=config.max_samples_per_dataset,
        wandb_tags=config.wandb_tags,
    )
    evaluate_lm_log_probs(eval_config)


def default_hf_lm_log_probs(
    *,
    hf_repo_id: str,
    hf_revision: str,
    checkpoint: ArtifactStep[LevanterCheckpoint] | str,
    validation_datasets: Sequence[ArtifactStep[TokenizedCache]],
    resource_config: ResourceConfig,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
    name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> ArtifactStep[Artifact]:
    """Build a log-probs eval artifact that fetches its HF model config at runtime."""
    if name is None:
        if isinstance(checkpoint, str):
            # Strip trailing slash and infer a name from the path.
            path = checkpoint.rstrip("/")
            name = "_".join(path.split("/")[-2:]) if "/" in path else path
        else:
            name = checkpoint.name.replace("/", "--")

    step_name = f"analysis/log_probs/{name}"
    deps: tuple[ArtifactStep, ...] = (
        (checkpoint, *validation_datasets) if isinstance(checkpoint, ArtifactStep) else tuple(validation_datasets)
    )

    def build_config(ctx: StepContext) -> HfLogProbsConfig:
        checkpoint_path = ctx.artifact_path(checkpoint) if isinstance(checkpoint, ArtifactStep) else checkpoint
        data = mixture(ctx, {}, validation=list(validation_datasets), shuffle=False)
        return HfLogProbsConfig(
            name=name,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            checkpoint_path=checkpoint_path,
            datasets=data,
            resource_config=resource_config,
            per_device_batch_size=per_device_batch_size,
            output_path=ctx.output_path,
            log_entropy=True,
            max_samples_per_dataset=max_samples_per_dataset,
            wandb_tags=wandb_tags,
        )

    return ArtifactStep(
        name=step_name,
        version="2026.06.28",
        artifact_type=Artifact,
        run=evaluate_hf_log_probs,
        build_config=build_config,
        deps=deps,
    )
