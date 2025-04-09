"""
Default configurations and utilities for the Marin speedrun experiment.
"""

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import os

from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import compute_num_parameters
from experiments.defaults import default_train
from levanter.compat.hf_checkpoints import load_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    unwrap_versioned_value,
)
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")

@dataclass
class HardwareConfig:
    device_type: str  # e.g., "v4-128", "A100"
    num_devices: int
    device_flops: float  # Theoretical peak FLOPs/s per device


class ComputeBudget(Enum):
    """Predefined compute budgets for speed run tracks, in FLOPs."""
    TINY = 3e18    # Smallest track, ~3e18 FLOPs
    SMALL = 6e18   # Small track, ~6e18 FLOPs
    MEDIUM = 3e19  # Medium track, ~3e19 FLOPs
    

@dataclass
class SpeedrunConfig:
    """Configuration for a speedrun submission with compute budget constraints."""


    compute_budget: ComputeBudget
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    tokenized_dataset: InputName | ExecutorStep | LMMixtureDatasetConfig
    hyperparameter_scaling: Optional[Callable[[LmConfig, ComputeBudget], dict]] = None
    output_path: str | None = None

    @property
    def vocab_size(self) -> int:
        return load_tokenizer(unwrap_versioned_value(self.tokenized_dataset.tokenizer)).vocab_size

    def estimate_training_flops_using_levanter(self) -> float:
        flops_per_token = self.model_config.flops_per_token(self.vocab_size)
        batch_size = self.train_config.train_batch_size
        total_tokens = batch_size * self.model_config.seq_len * self.train_config.num_train_steps
        return float(flops_per_token * total_tokens * 3)

    def estimate_flops_via_6nd(self) -> float:
        """Estimate training FLOPs using 6 times N times D, where N is the number of parameters, and D is the number of tokens."""
        
        N = compute_num_parameters(self.model_config, self.vocab_size)
        D = (
            self.train_config.train_batch_size
            * self.train_config.seq_len
            * self.train_config.num_train_steps
        )
        return 6.0 * N * D

    def adjust_to_exact_budget(self):
        """Adjust num_train_steps to match compute budget exactly."""
        flops_per_token = self.model_config.flops_per_token(self.vocab_size) * 3
        target_flops = self.compute_budget.value
        batch_size = self.train_config.train_batch_size
        seq_len = self.train_config.seq_len
        total_tokens = target_flops / flops_per_token
        self.train_config.num_train_steps = int(total_tokens / (batch_size * seq_len))
        actual_flops = self.estimate_training_flops_using_levanter()
        logger.info(f"Adjusted to {self.train_config.num_train_steps} steps for {actual_flops:e} FLOPs "
                    f"(target: {target_flops:e})")

    def apply_scaling(self):
        """
        Apply user-defined hyperparameter scaling if provided.

        Updates train_config with values from hyperparameter_scaling function, if present.
        """
        if self.hyperparameter_scaling:
            params = self.hyperparameter_scaling(self.model_config, self.compute_budget)
            self.train_config.learning_rate = params.get("learning_rate", self.train_config.learning_rate)
            self.train_config.train_batch_size = params.get("train_batch_size", self.train_config.train_batch_size)
            self.train_config.weight_decay = params.get("weight_decay", self.train_config.weight_decay)
            # Add more as needed (e.g., seq_len, warmup steps)

    def validate(self) -> tuple[bool, str | None]:
        """Validates the configuration against compute budget constraints, and makes adjustments as needed."""
        # Step 1: Calculate model FLOPs (6ND)
        model_flops = self.calculate_6nd()
        
        # Step 2: Estimate effective compute capacity
        effective_flops_per_sec = (
            self.hardware_config.num_devices
            * self.hardware_config.device_flops
            * mfu_estimate
        )
        
        # Step 3: Estimate wall-clock time
        estimated_time = model_flops / effective_flops_per_sec
        
        # Step 4: Calculate Total Compute metric
        estimated_compute = (
            estimated_time
            * self.hardware_config.num_devices
            * self.hardware_config.device_flops
        )
        
        # Step 5: Check against budget
        is_valid = estimated_compute <= self.compute_budget
        
        return is_valid, estimated_compute


def get_wandb_run_id_from_step(step: ExecutorStep) -> str:
    """Get the wandb run id from a given ExecutorStep."""
    return step.config.trainer.tracker.id


def get_step_times_from_wandb(run_id: str, entity: str = "stanford-mercury", project: str = "marin") -> list[float]:
    """Get step durations from wandb run.
    
    Args:
        run_id: The wandb run ID
        entity: The wandb entity (organization)
        project: The wandb project name
        
    Returns:
        List of step durations in seconds
    """
    import wandb
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=["throughput/duration"])
        durations = history["throughput/duration"].tolist()
        return durations

    except Exception as e:
        logger.error(f"Failed to get step times from wandb: {e}")
        return []


def get_compute_budgets():
    return {budget.name: budget.value for budget in ComputeBudget}


def speedrun_analysis(config: SpeedrunConfig, step_times: list[float]):
    """Compute and store metrics and stats for the speedrun.

    Args:
        config: SpeedrunConfig containing model, training, and dataset configuration
        step_times: List of step times in seconds
    """
    
    # Calculate actual training FLOPs using both methods
    six_nd_flops = config.estimate_flops_via_6nd()
    
    # Get the model parameters
    num_params = compute_num_parameters(config.model_config, config.vocab_size)
    
    # Calculate total tokens processed
    total_tokens = (
        config.train_config.train_batch_size 
        * config.model_config.seq_len 
        * config.train_config.num_train_steps
    )

    # Hardware-based FLOPs
    total_time = sum(step_times)
    total_hardware_flops = (
        total_time
        * config.hardware_config.num_devices
        * config.hardware_config.device_flops
    )
    
    # Compare with pre-training estimates
    effective_flops_per_sec = config.hardware_config.num_devices * config.hardware_config.device_flops * 0.5  # Using 0.5 as baseline MFU
    estimated_time = six_nd_flops / effective_flops_per_sec
    estimated_compute = estimated_time * config.hardware_config.num_devices * config.hardware_config.device_flops
    
    logger.info(
        f"Pre-training estimate vs actual:\n"
        f"  - Wall time: {estimated_time:.1f}s vs {total_time:.1f}s\n"
        f"  - Compute: {estimated_compute:.2e} vs {hardware_flops:.2e} FLOPs"
    )
    
    # Prepare stats dictionary
    stats = {
        "compute_budget": {
            "track": config.compute_budget.name,
            "budget_flops": config.compute_budget.value,
            "flops_6nd": six_nd_flops,
        },
        "model_stats": {
            "num_parameters": num_params,
            "total_tokens": total_tokens,
            "model_config": json.dumps(dataclasses.asdict(config.model_config)),
            "train_config": json.dumps(dataclasses.asdict(config.train_config)),
            "hardware_config": json.dumps(dataclasses.asdict(config.hardware_config)),
        },
        "estimated_stats_before_training": {
            "estimated_training_time": estimated_time,
            "estimated_compute": estimated_compute,
            "assumed_mfu": 0.5
        },

        # these 
        "actual_stats": {
            "actual_training_time": total_time, # sum of training step times- needs to be obtained from wandb
            "actual_compute": total_hardware_flops,
            "within_budget": total_hardware_flops <= config.compute_budget.value
        }
    }
    
    output_path = os.path.join(config.output_path, "speedrun_stats.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Speedrun stats written to {output_path}")

    if not stats["compute_budget"]["within_budget_6nd"]:
        logger.warning(
            f"Training exceeded compute budget using 6ND estimate: "
            f"{six_nd_flops:.2e} > {config.compute_budget.value:.2e} FLOPs"
        )

    if not stats["actual_stats"]["within_budget"]:
        logger.warning(
            f"Training exceeded compute budget according to (estimated) FLOPs: "
            f"{total_hardware_flops:.2e} > {config.compute_budget.value:.2e} FLOPs"
        )
    
    logger.info(f"Speedrun completed and stats written to {output_path}")


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
) -> Sequence[ExecutorStep]:
    """Returns an ExecutorStep for a speedrun with the given configuration.

    Args:
        name: name of the training run. Will form the basis of the output path for the executor step.
        config: SpeedrunConfig containing model, training, and dataset configuration
        tags: Optional additional tags for tracking

    Returns:
        training step configured for the speedrun

    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate configuration
    is_valid, error = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid speedrun configuration: {error}")

    # Set up tags
    run_tags = ["speedrun", f"budget_{config.compute_budget.name}"]
    if tags:
        run_tags.extend(tags)

    # Create training step
    model_config = dataclasses.replace(config.model_config)

    steps = []

    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=config.tokenized_dataset,
        model_config=model_config,
        train_config=config.train_config,
        tags=run_tags,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    )

    steps.append(train_step)

    # Get step times from training step via wandb
    run_id = get_wandb_run_id_from_step(train_step)
    step_times = get_step_times_from_wandb(run_id)

    post_process_speedrun = ExecutorStep(
        name=f"speedrun/{name}",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_analysis,
        config=config,
    )

    steps.append(post_process_speedrun)

    return steps
