# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local speedrun support for standalone Qwen3 optimizer submissions."""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from dataclasses import dataclass
from enum import Enum

import wandb
from levanter.models.lm_model import LmConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import open_url, prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.prebuilt_caches import fineweb_edu_10B_dataset, fineweb_edu_10M_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.marin_tokenizer import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

_C4_EN_BPB_KEYS = (
    "eval/paloma/c4_en/bpb",
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/c4_en-marin/bpb",
)


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str
    url: str | None = None


@dataclass
class SpeedrunConfig:
    author: Author
    description: str
    model_config: LmConfig
    train_config: SimpleTrainConfig
    tokenized_dataset: str

    @property
    def vocab_size(self) -> int:
        return llama3_tokenizer_vocab_size

    @property
    def fixed_train_batch_size(self) -> int:
        batch_size = self.train_config.train_batch_size
        if not isinstance(batch_size, int):
            raise ValueError("Speedrun submissions require a fixed integer train batch size.")
        return batch_size

    def as_json_dict(self) -> dict[str, object]:
        def _make_serializable(obj: object) -> object:
            if isinstance(obj, Enum):
                return obj.name
            if isinstance(obj, dict):
                return {key: _make_serializable(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(value) for value in obj]
            return obj

        train_config_dict = dataclasses.asdict(self.train_config)
        resources_dict = train_config_dict.pop("resources")
        model_config_dict = dataclasses.asdict(self.model_config)
        return {
            "author": dataclasses.asdict(self.author),
            "description": self.description,
            "model_config": _make_serializable(model_config_dict),
            "train_config": _make_serializable(train_config_dict),
            "tokenized_dataset": self.tokenized_dataset,
            "resources": _make_serializable(resources_dict),
        }

    def print_run_info(self) -> None:
        logger.info("----- START OF PRINT RUN INFO -----")
        logger.info("Speedrun Configuration:")
        logger.info(json.dumps(self.as_json_dict(), indent=2))
        logger.info("Model FLOPs: %.2e FLOP", self.compute_model_flops())
        logger.info("----- END OF PRINT RUN INFO -----")

    def compute_model_flops(self) -> float:
        context_length = self.train_config.train_seq_len or self.model_config.max_seq_len
        total_tokens = self.fixed_train_batch_size * self.train_config.num_train_steps * context_length
        flops_per_token = self.model_config.flops_per_token(self.vocab_size, context_length)
        if flops_per_token is None:
            raise ValueError("Model config must provide flops_per_token to compute model FLOPs.")
        return flops_per_token * 3 * total_tokens


@dataclass(frozen=True)
class SpeedrunResultsConfig:
    wandb_run_id: str
    wandb_entity: str | None
    wandb_project: str
    speedrun_config: SpeedrunConfig
    output_path: str


def _get_step_times(run_id: str, entity: str, project: str) -> list[float]:
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    return [
        row["throughput/duration"]
        for row in run.scan_history(keys=["throughput/duration"])
        if "throughput/duration" in row
    ]


def _resolve_wandb_entity(entity: str | None) -> str:
    if entity is not None:
        return entity

    settings_entity = wandb.Settings().entity
    if settings_entity is not None:
        return settings_entity

    try:
        default_entity = wandb.Api().default_entity
    except wandb.errors.UsageError as exc:
        raise ValueError("Could not infer a W&B entity. Set tracker.entity or configure a default W&B entity.") from exc
    if default_entity is not None:
        return default_entity

    raise ValueError("Could not infer a W&B entity. Set tracker.entity or configure a default W&B entity.")


def speedrun_results(config: SpeedrunResultsConfig) -> None:
    wandb_run_id = config.wandb_run_id.split("/")[-1]
    wandb_entity = _resolve_wandb_entity(config.wandb_entity)
    step_times = _get_step_times(wandb_run_id, wandb_entity, config.wandb_project)
    if not step_times:
        raise ValueError(
            f"No throughput/duration history found for {wandb_entity}/{config.wandb_project}/{wandb_run_id}."
        )

    run = wandb.Api().run(f"{wandb_entity}/{config.wandb_project}/{wandb_run_id}")
    context_length = config.speedrun_config.train_config.train_seq_len or config.speedrun_config.model_config.max_seq_len
    flops_per_token = config.speedrun_config.model_config.flops_per_token(
        config.speedrun_config.vocab_size, context_length
    )
    model_flops = config.speedrun_config.compute_model_flops()
    model_size = config.speedrun_config.model_config.total_trainable_params(config.speedrun_config.vocab_size)
    device_flops = config.speedrun_config.train_config.resources.device_flops()
    if flops_per_token is None or device_flops is None:
        raise ValueError("Speedrun submission requires model flops_per_token and resource device_flops.")

    runtime_seconds = sum(step_times)
    start_time = datetime.datetime.fromisoformat(run.createdAt.replace("Z", "+00:00"))
    end_time = start_time + datetime.timedelta(seconds=run.summary["_runtime"])
    c4_en_bpb = None
    for key in _C4_EN_BPB_KEYS:
        c4_en_bpb = run.summary.get(key)
        if c4_en_bpb is not None:
            break
    if c4_en_bpb is None:
        raise ValueError(f"No C4-en BPB metric found for {wandb_entity}/{config.wandb_project}/{wandb_run_id}.")
    run_info = {
        **config.speedrun_config.as_json_dict(),
        "model_size": model_size,
        "total_tokens": (
            config.speedrun_config.fixed_train_batch_size
            * config.speedrun_config.train_config.num_train_steps
            * context_length
        ),
        "model_flops": model_flops,
        "model_flops_per_token": flops_per_token,
        "num_devices": run.summary.get("num_devices"),
        "num_chips": config.speedrun_config.train_config.resources.chip_count(),
        "device_flops": device_flops,
        "training_time": runtime_seconds,
        "training_hardware_flops": (
            runtime_seconds * config.speedrun_config.train_config.resources.chip_count() * device_flops
        ),
        "eval/paloma/c4_en/bpb": c4_en_bpb,
        "run_completion_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "wandb_run_link": f"https://wandb.ai/{wandb_entity}/{config.wandb_project}/runs/{wandb_run_id}",
    }
    with open_url(config.output_path, "w") as f:
        json.dump({"runs": [{"run_info": run_info}]}, f, indent=2, sort_keys=True)


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    *,
    tags: list[str] | None = None,
    version: str = "dev",
) -> tuple[ArtifactStep[LevanterCheckpoint], ArtifactStep[Artifact]]:
    """Build one lazy speedrun training step and its dependent result collector."""
    train_config = config.train_config
    optimizer = train_config.optimizer_config
    if optimizer is None:
        raise ValueError("Speedrun submissions must provide train_config.optimizer_config explicitly.")
    batch_size = config.fixed_train_batch_size

    tokenized_dataset = _resolve_tokenized_dataset(config.tokenized_dataset)
    validation = [
        *paloma_datasets(tokenizer=marin_tokenizer).values(),
        *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
    ]
    train_step = train_lm(
        name=f"checkpoints/speedrun/{name}",
        version=version,
        run_id=f"{name}-{version}",
        model=config.model_config,
        optimizer=optimizer,
        datasets={tokenized_dataset: 1.0},
        validation=validation,
        batch_size=batch_size,
        seq_len=train_config.train_seq_len or config.model_config.max_seq_len,
        num_train_steps=train_config.num_train_steps,
        z_loss_weight=train_config.z_loss_weight,
        evals=None,
        resources=train_config.resources,
        tensor_parallel_size=train_config.tensor_parallel_size,
        steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
        tags=["speedrun", *(tags or [])],
        env_vars=train_config.env_vars,
    )

    def build_results_config(ctx: StepContext) -> SpeedrunResultsConfig:
        return SpeedrunResultsConfig(
            wandb_run_id=f"{name}-{version}",
            wandb_entity=None,
            wandb_project="marin",
            speedrun_config=config,
            output_path=prefix_join(ctx.artifact_path(train_step), "speedrun_results.json"),
        )

    results_step = ArtifactStep(
        name=f"{train_step.name}-speedrun-results",
        version=version,
        artifact_type=Artifact,
        run=speedrun_results,
        build_config=build_results_config,
        deps=(train_step,),
    )
    return train_step, results_step


def _resolve_tokenized_dataset(tokenized_dataset: str) -> ArtifactStep[TokenizedCache]:
    if tokenized_dataset.startswith("gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-"):
        return fineweb_edu_10B_dataset(pin=tokenized_dataset)
    if tokenized_dataset.startswith("gs://marin-us-central1/tokenized/subcache/fineweb-edu-10M-"):
        return fineweb_edu_10M_dataset(pin=tokenized_dataset)
    raise ValueError(
        f"Unsupported raw tokenized dataset path {tokenized_dataset!r}. "
        "Use a prebuilt cache step or dataset config with tokenizer metadata."
    )
