"""
Default functions, configurations and utilities for Marin speedruns to use.

default_speedrun() is the function a user should call to run a speedrun; example is
in experiments/speedrun/sample_run.py
"""

import dataclasses
import datetime
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
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

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

    def estimate_flops_before_speedrun(self) -> tuple[bool, str]:

        # estimate model FLOPs as 6*N*D, and calculate estimated compute using this and (a reasonable estimate of) MFU
        model_flops = self.estimate_flops_via_6nd()
        estimated_compute = model_flops / self.mfu_estimate

        logger.info(f"Estimated {estimated_compute:.2e} FLOPs")
        return estimated_compute


@dataclass
class SpeedrunResultsConfig:
    wandb_run_id: str
    wandb_entity: str
    wandb_project: str
    speedrun_config: SpeedrunConfig
    output_path: str


### Utils and analysis functions ###

def get_step_times_from_wandb(run_id: str, entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT) -> list[float]:
    try:
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        return run.history(keys=["throughput/duration"])["throughput/duration"].tolist()
    except Exception as e:
        logger.error(f"Failed to fetch step times: {e}")
        return []


def speedrun_results(config: SpeedrunResultsConfig):
    """Compute and store metrics and stats for the speedrun."""

    # get the last part of the path (i.e. last part of gs://.../checkpoints/speedrun/<wandb_run_id>)
    wandb_run_id = config.wandb_run_id.split("/")[-1]

    step_times = get_step_times_from_wandb(
        run_id=wandb_run_id, entity=config.wandb_entity, project=config.wandb_project
    )
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

    # get wandb run and metrics
    run = wandb.Api().run(f"{config.wandb_entity}/{config.wandb_project}/{wandb_run_id}")
    wandb_metrics = {
        "eval/paloma/c4_en/bpb": run.summary.get("eval/paloma/c4_en/bpb", None),
    }

    # Get timestamps in UTC
    start_time = datetime.datetime.fromisoformat(run.createdAt.replace("Z", "+00:00"))  # Convert ISO string to datetime
    runtime_seconds = run.summary["_runtime"]
    end_time = start_time + datetime.timedelta(seconds=runtime_seconds)

    full_wandb_url = f"https://wandb.ai/{config.wandb_entity}/{config.wandb_project}/runs/{wandb_run_id}"

    run_stats = {
        "run_related_info": {
            "num_parameters": num_params,
            "total_tokens": total_tokens,
            "model_config": dataclasses.asdict(config.speedrun_config.model_config),
            "train_config": dataclasses.asdict(config.speedrun_config.train_config),
            "tokenized_dataset": str(config.speedrun_config.tokenized_dataset),
            "hardware_config": dataclasses.asdict(config.speedrun_config.hardware_config),
            "run_completion_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "wandb_run_link": full_wandb_url,
        },
        "run_stats": {
            "pre_run_flops_estimate": six_nd_flops,
            "training_time_in_minutes": total_time,
            "total_training_flops": total_training_flops,
            "eval/paloma/c4_en/bpb": (
                float(wandb_metrics.get("eval/paloma/c4_en/bpb"))
                if wandb_metrics.get("eval/paloma/c4_en/bpb") is not None
                else None
            ),
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
    logger.info(f"Estimated FLOPs: {config.estimate_flops_before_speedrun()}")

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

    # Extract wandb info from train step
    wandb_run_id = None  # None by default

    if (
        train_step.config
        and train_step.config.train_config
        and train_step.config.train_config.trainer
        and train_step.config.train_config.trainer.tracker
    ):

        wandb_entity = train_step.config.train_config.trainer.tracker.entity or WANDB_ENTITY
        wandb_project = train_step.config.train_config.trainer.tracker.project or WANDB_PROJECT

        # (Nikil) this is a hack (the ExecutorStep isn't populated when using an override path, so can't get configs when
        # we do an override or if it is None for some reason, but after some investigation found that in those cases we
        # can fall back to the fact that we set the wandb run ID as the last part of the path anyway, so can use that)
        if override_output_path:
            wandb_run_id = override_output_path.split("/")[-1]
        else:
            wandb_run_id = train_step # gets converted to output path when passing it into the results step
            
    assert wandb_run_id is not None, "Could not extract wandb run ID from train step"

    results_step = ExecutorStep(
        name=f"speedrun/{name}-speedrun_results-new",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_results,
        config=SpeedrunResultsConfig(
            wandb_run_id=wandb_run_id,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_results.json"),
        ),
    )

    return [train_step, results_step]
