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
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from rigging.filesystem import open_url

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)


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
    tokenized_dataset: str | InputName

    @property
    def vocab_size(self) -> int:
        return llama3_tokenizer_vocab_size

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
        total_tokens = self.train_config.train_batch_size * self.train_config.num_train_steps * context_length
        flops_per_token = self.model_config.flops_per_token(self.vocab_size, context_length)
        if flops_per_token is None:
            raise ValueError("Model config must provide flops_per_token to compute model FLOPs.")
        return flops_per_token * 3 * total_tokens


@dataclass(frozen=True)
class SpeedrunResultsConfig:
    wandb_run_id: str
    wandb_entity: str
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

    default_entity = wandb.Api().default_entity
    if default_entity is not None:
        return default_entity

    raise ValueError("Could not infer a W&B entity. Set tracker.entity or configure a default W&B entity.")


def speedrun_results(config: SpeedrunResultsConfig) -> None:
    wandb_run_id = config.wandb_run_id.split("/")[-1]
    step_times = _get_step_times(wandb_run_id, config.wandb_entity, config.wandb_project)
    if not step_times:
        raise ValueError(
            f"No throughput/duration history found for {config.wandb_entity}/{config.wandb_project}/{wandb_run_id}."
        )

    run = wandb.Api().run(f"{config.wandb_entity}/{config.wandb_project}/{wandb_run_id}")
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
    run_info = {
        **config.speedrun_config.as_json_dict(),
        "model_size": model_size,
        "total_tokens": (
            config.speedrun_config.train_config.train_batch_size
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
        "eval/paloma/c4_en/bpb": run.summary.get("eval/paloma/c4_en/bpb"),
        "run_completion_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "wandb_run_link": f"https://wandb.ai/{config.wandb_entity}/{config.wandb_project}/runs/{wandb_run_id}",
    }
    with open_url(config.output_path, "w") as f:
        json.dump({"runs": [{"run_info": run_info}]}, f, indent=2, sort_keys=True)


def default_speedrun(name: str, config: SpeedrunConfig, *, tags: list[str] | None = None) -> list[ExecutorStep]:
    resolved_tokenized_dataset = _resolve_tokenized_dataset(config.tokenized_dataset)
    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=resolved_tokenized_dataset,
        model_config=config.model_config,
        train_config=config.train_config,
        tags=["speedrun", *(tags or [])],
        use_default_validation=True,
        eval_harness_tasks=(),
    )
    tracker = train_step.config.train_config.trainer.tracker
    results_step = ExecutorStep(
        name=f"speedrun/{name}-speedrun_results",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_results,
        config=SpeedrunResultsConfig(
            wandb_run_id=train_step,
            wandb_entity=_resolve_wandb_entity(tracker.entity),
            wandb_project=tracker.project or "marin",
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_results.json"),
        ),
    )
    return [train_step, results_step]


def _resolve_tokenized_dataset(tokenized_dataset: str | InputName) -> InputName:
    if not isinstance(tokenized_dataset, str):
        return tokenized_dataset
    return InputName.hardcoded(tokenized_dataset)
