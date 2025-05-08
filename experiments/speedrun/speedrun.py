"""
Default functions, configurations and utilities for Marin speedruns to use.

default_speedrun() is the function a user should call to run a speedrun; example is
in experiments/speedrun/sample_run.py
"""

import dataclasses
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import fsspec
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

import wandb
from experiments.defaults import default_train
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
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


class ComputeBudget(Enum):
    # target compute budget in FLOPs that the user opts to stay within
    TINY = 3e18
    SMALL = 6e18
    MEDIUM = 3e19


@dataclass
class SpeedrunConfig:
    compute_budget: ComputeBudget
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    hardware_config: HardwareConfig
    tokenized_dataset: InputName | LMMixtureDatasetConfig = fineweb_edu_tokenized
    hyperparameter_scaling: Callable[[LmConfig, ComputeBudget], dict] | None = None
    mfu_estimate: float = 0.5

    @property
    def vocab_size(self) -> int:

        # TODO (Nikil): this doesn't interact well with different types (InputName, LMMixtureDatasetConfig, ExecutorStep)
        # Need to change this to automatically figure out vocab size
        # return load_tokenizer(unwrap_versioned_value(self.tokenized_dataset.tokenizer)).vocab_size
        return llama3_tokenizer_vocab_size

    # def estimate_training_flops_using_levanter(self) -> float:
    #     flops_per_token = self.model_config.flops_per_token(self.vocab_size)
    #     total_tokens = self.train_config.train_batch_size * self.model_config.seq_len *
    #                       self.train_config.num_train_steps
    #     return flops_per_token * 6  # 6ND standard: forward + backward

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

    def adjust_to_exact_budget(self):
        # TODO (Nikil): at the moment, we have this function but don't apply it- need to figure out user flow for
        # adjusting to exact budget
        flops_per_token = self.model_config.flops_per_token(self.vocab_size) * 6
        total_tokens = self.compute_budget.value / flops_per_token

        # TODO (Nikil): handle batch schedules
        self.train_config.num_train_steps = int(
            total_tokens / (self.train_config.train_batch_size * self.model_config.seq_len)
        )
        logger.info(
            f"""Adjusted to {self.train_config.num_train_steps} steps for
            {self.estimate_flops_via_6nd():.2e} FLOPs"""
        )

    def apply_scaling(self):
        if self.hyperparameter_scaling:
            params = self.hyperparameter_scaling(self.model_config, self.compute_budget)
            self.train_config = dataclasses.replace(
                self.train_config, **{k: params.get(k, v) for k, v in self.train_config.__dict__.items()}
            )

    def validate(self) -> tuple[bool, str]:

        # estimate model FLOPs as 6*N*D, and calculate estimated compute using this and (a reasonable estimate of) MFU
        model_flops = self.estimate_flops_via_6nd()
        estimated_compute = model_flops / self.mfu_estimate

        # check if estimated compute is within user's requested budget
        is_valid = estimated_compute <= self.compute_budget.value
        return is_valid, f"Estimated {estimated_compute:.2e} FLOPs vs. budget {self.compute_budget.value:.2e}"


@dataclass
class SpeedrunAnalysisConfig:
    speedrun_train_step: ExecutorStep
    speedrun_config: SpeedrunConfig
    output_path: str


### Utils and analysis functions ###


def get_wandb_run_id_from_step(step: ExecutorStep) -> str:
    if isinstance(step, str):
        return step
    if hasattr(step, "config") and hasattr(step.config, "trainer") and hasattr(step.config.trainer, "tracker"):
        return step.config.trainer.tracker.id
    return ""


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
            "lm_eval/averages/macro_avg_acc": summary.get("lm_eval/averages/macro_avg_acc", None),
            "eval/paloma/c4_en/bpb": summary.get("eval/paloma/c4_en/bpb", None),
            "eval/fineweb-edu/loss": summary.get("eval/fineweb-edu/loss", None),
        }
    except Exception as e:
        logger.error(f"Failed to fetch wandb metrics: {e}")
        return {"lm_eval/averages/macro_avg_acc": None, "eval/paloma/c4_en/bpb": None, "eval/fineweb-edu/loss": None}


def get_compute_budgets():
    return {budget.name: budget.value for budget in ComputeBudget}


def speedrun_analysis(config: SpeedrunAnalysisConfig):
    """Compute and store metrics and stats for the speedrun."""

    run_id = get_wandb_run_id_from_step(config.speedrun_train_step)
    step_times = get_step_times_from_wandb(run_id)
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
    total_hardware_flops = (
        total_time
        * config.speedrun_config.hardware_config.num_devices
        * config.speedrun_config.hardware_config.device_flops
    )

    # Get wandb metrics
    run_id = get_wandb_run_id_from_step(config.speedrun_train_step)
    wandb_metrics = get_wandb_metrics(run_id)

    stats = {
        "compute_budget": {
            "track": config.speedrun_config.compute_budget.name,
            "flops_budget_for_track": config.speedrun_config.compute_budget.value,
            "flops_6nd": f"{six_nd_flops:.2e}",
            "within_budget_6nd": six_nd_flops <= config.speedrun_config.compute_budget.value,
        },
        "run_related_info": {
            "num_parameters": num_params,
            "total_tokens": total_tokens,
            "model_config": dataclasses.asdict(config.speedrun_config.model_config),
            "train_config": dataclasses.asdict(config.speedrun_config.train_config),
            "tokenized_dataset": str(config.speedrun_config.tokenized_dataset),
            "hardware_config": dataclasses.asdict(config.speedrun_config.hardware_config),
            "wandb_run_id": run_id,
        },
        "actual_stats": {
            "training_time_in_minutes": total_time,
            "final_flops_estimate": f"{total_hardware_flops:.2e}",
            "within_budget_hardware": total_hardware_flops <= config.speedrun_config.compute_budget.value,
            "lm_eval/averages/macro_avg_acc": wandb_metrics["lm_eval/averages/macro_avg_acc"],
            "eval/paloma/c4_en/bpb": wandb_metrics["eval/paloma/c4_en/bpb"],
            "eval/fineweb-edu/loss": wandb_metrics["eval/fineweb-edu/loss"],
        },
    }

    logger.info(f"Speedrun stats: {stats}")

    stats["speedrun_analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
    with fsspec.open(config.output_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    logger.info(f"Speedrun stats written to {config.output_path}")


### Default speedrun function ###


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
) -> Sequence[ExecutorStep]:
    """

    Speedrun is a mechanism for submitting a PoC run for a new model/optimizer configuration. It consists of both a
    training step and an analysis step that computes and stores metrics/metadata for the speedrun. See `sample_run.py`
    for an example.

    Args:
        name: name of the training run. Will form the basis of the output path for the executor step.
        config: SpeedrunConfig containing model, training, and dataset configuration
        tags: Optional additional tags for tracking

    Returns:
        training step configured for the speedrun, and an analysis step that
        computes and stores metrics/metadata for the speedrun.

    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate configuration
    is_valid, error = config.validate()
    logger.info(f"Speedrun validation: {is_valid}, {error}")
    if not is_valid:
        raise ValueError(f"Invalid speedrun configuration: {error}")

    run_tags = ["speedrun", f"budget_{config.compute_budget.name}"] + (tags or [])

    # TODO (Nikil): need to have a better mechanism to enforce/validate fixed dataset and seed
    # instead of just overriding the configs
    train_config = dataclasses.replace(config.train_config, data_seed=42)
    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=config.tokenized_dataset,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    )

    analysis_step = ExecutorStep(
        name=f"{name}_speedrun_analysis",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_analysis,
        config=SpeedrunAnalysisConfig(
            speedrun_train_step=train_step,
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_analysis.json"),
        ),
    )

    return [train_step, analysis_step]
