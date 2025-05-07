"""
Default functions, configurations and utilities for Marin speedruns to use.

default_speedrun() is the function a user should call to run a speedrun; example is
in experiments/speedrun/sample_run.py
"""
import os
import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
import wandb
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import compute_num_parameters, llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")


### Configuration classes ###

@dataclass
class HardwareConfig:
    device_type: str  # a string describing the device e.g. "v4-128", or "h100"
    num_devices: int
    device_flops: float  # Peak FLOPs/s per device


@dataclass
class SpeedrunConfig:
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    hardware_config: HardwareConfig
    mfu_estimate: float = 0.5

    # by default, this is fineweb_edu_tokenized
    tokenized_dataset: InputName | LMMixtureDatasetConfig = fineweb_edu_tokenized

    @property
    def vocab_size(self) -> int:

        # TODO (Nikil): this doesn't interact well with different types (InputName, LMMixtureDatasetConfig, ExecutorStep)
        # Need to change this to automatically figure out vocab size
        # return load_tokenizer(unwrap_versioned_value(self.tokenized_dataset.tokenizer)).vocab_size
        return llama3_tokenizer_vocab_size

    def estimate_flops_via_6nd(self) -> float:
        N = compute_num_parameters(self.model_config, self.vocab_size)

        # TODO (Nikil): make this a helper and handle edge-cases
        if isinstance(self.train_config.train_batch_size, list) and len(self.train_config.train_batch_size) > 0:
            from levanter.schedule import BatchSchedule

            batch_schedule = BatchSchedule(self.train_config.train_batch_size)
            total_tokens = (
                batch_schedule.global_data_offset_by_step(self.train_config.num_train_steps) * self.model_config.seq_len
            )
        else:
            # integer batch size
            total_tokens = (
                self.train_config.train_batch_size * self.model_config.seq_len * self.train_config.num_train_steps
            )

        return 6.0 * N * total_tokens

    def validate(self) -> tuple[bool, str]:

        # estimate model FLOPs as 6*N*D, and calculate estimated compute using this and (a reasonable estimate of) MFU
        model_flops = self.estimate_flops_via_6nd()
        estimated_compute = model_flops / self.mfu_estimate

        logger.info(f"Estimated {estimated_compute:.2e} FLOPs")
        return estimated_compute


@dataclass
class SpeedrunResultsConfig:
    speedrun_train_step: ExecutorStep
    speedrun_config: SpeedrunConfig
    output_path: str


### Utils and analysis functions ###


def get_wandb_run_info_from_step(step: ExecutorStep) -> tuple[str, str, str]:
    """
    Get the wandb entity, project, and run id from a given ExecutorStep.
    """

    # defaults
    wandb_entity = "stanford-mercury"
    wandb_project = "marin"

    if isinstance(step, str):
        return wandb_entity, wandb_project, step
    elif hasattr(step, "config") and hasattr(step.config, "trainer") and hasattr(step.config.trainer, "tracker"):
        wandb_entity = step.config.trainer.tracker.entity
        if wandb_entity is None:
            wandb_entity = "stanford-mercury"
        wandb_project = step.config.trainer.tracker.project
        if wandb_project is None:
            wandb_project = "marin"
        return wandb_entity, wandb_project, step.config.trainer.tracker.id
    
    return wandb_entity, wandb_project, None

def get_step_times_from_wandb(run_id: str, entity: str = "stanford-mercury", project: str = "marin") -> list[float]:
    try:
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        return run.history(keys=["throughput/duration"])["throughput/duration"].tolist()
    except Exception as e:
        logger.error(f"Failed to fetch step times: {e}")
        return []


def get_wandb_metrics(run_id: str, entity: str = "stanford-mercury", project: str = "marin") -> dict:
    try:
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        summary = run.summary
        return {
            "eval/paloma/c4_en/bpb": summary.get("eval/paloma/c4_en/bpb", None),
        }
    except Exception as e:
        logger.error(f"Failed to fetch wandb metrics: {e}")
        return {"eval/paloma/c4_en/bpb": None}


def speedrun_results(config: SpeedrunResultsConfig):
    """Compute and store metrics and stats for the speedrun."""

    wandb_entity, wandb_project, run_id = get_wandb_run_info_from_step(config.speedrun_train_step)

    assert run_id is not None, "Run ID not found while creating speedrun results file"

    step_times = get_step_times_from_wandb(run_id, wandb_entity, wandb_project)
    if not step_times:
        logger.error("No step times available; analysis aborted.")
        return

    six_nd_flops = config.speedrun_config.estimate_flops_via_6nd()
    num_params = compute_num_parameters(config.speedrun_config.model_config, config.speedrun_config.vocab_size)
    total_tokens = (
        config.speedrun_config.train_config.train_batch_size
        * config.speedrun_config.model_config.seq_len
        * config.speedrun_config.train_config.num_train_steps
    )
    total_time = sum(step_times)
    total_training_flops = (
        total_time
        * config.speedrun_config.hardware_config.num_devices
        * config.speedrun_config.hardware_config.device_flops
    )

    # get wandb metrics
    wandb_metrics = get_wandb_metrics(run_id, wandb_entity, wandb_project)
    full_wandb_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{run_id}"

    run_stats = {
        "run_related_info": {
            "num_parameters": num_params,
            "total_tokens": total_tokens,
            "model_config": dataclasses.asdict(config.speedrun_config.model_config),
            "train_config": dataclasses.asdict(config.speedrun_config.train_config),
            "tokenized_dataset": str(config.speedrun_config.tokenized_dataset),
            "hardware_config": dataclasses.asdict(config.speedrun_config.hardware_config),
            "wandb_link": full_wandb_url,
        },
        "run_stats": {
            "pre_run_flops_estimate": six_nd_flops,
            "training_time_in_minutes": total_time,
            "total_training_flops": total_training_flops,
            "eval/paloma/c4_en/bpb": float(wandb_metrics.get("eval/paloma/c4_en/bpb")) 
                        if wandb_metrics.get("eval/paloma/c4_en/bpb") is not None else None,
        },
    }

    logger.info(f"Speedrun stats: {run_stats}")

    output_data = {"runs": [run_stats]}
    with fsspec.open(config.output_path, "w") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)
    logger.info(f"Speedrun stats written to {config.output_path}")


### Default speedrun function ###


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
    override_output_path: str | None = None,
) -> Sequence[ExecutorStep]:
    """
    Speedrun is a mechanism for submitting a PoC run for a new model/optimizer configuration. It consists of both a
    training step and a step that computes and stores metrics/metadata for the speedrun. See `experiments/speedrun/`
    for examples.

    Args:
        name: name of the training run. Will form the basis of the output path for the executor step.
        config: SpeedrunConfig containing model, training, and dataset configuration
        tags: Optional additional tags for tracking

    Returns:
        training step configured for the speedrun, and a step that computes and stores metrics/metadata
        for the speedrun.

    Raises:
        ValueError: If the configuration is invalid
    """

    logger.info(f"Running speedrun {name} with config {config}")
    logger.info(f"Estimated FLOPs: {config.validate()}")

    run_tags = ["speedrun"] + (tags or [])

    train_config = dataclasses.replace(config.train_config, data_seed=42)
    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=config.tokenized_dataset,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=None,
        override_output_path=override_output_path,
    )

    results_step = ExecutorStep(
        name=f"speedrun/{name}_speedrun_results",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_results,
        config=SpeedrunResultsConfig(
            speedrun_train_step=train_step,
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_results.json"),
        ),
    )

    return [train_step, results_step]
