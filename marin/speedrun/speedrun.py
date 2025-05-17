"""
Default functions, configurations and utilities for Marin speedruns to use.

NOTE: If you are submitting a speedrun, you shouldn't modify this code (unless there is a very strong reason to do so).
You should just call default_speedrun() to run a speedrun; examples can be found in marin/experiments/speedrun/.
"""

import dataclasses
import datetime
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token

import wandb
from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import compute_num_parameters, llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.training.training import TrainLmOnPodConfig
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

logger = logging.getLogger("ray")


@dataclass
class SpeedrunConfig:

    author: str  # name of the author/individual submitting the speedrun
    affiliation: str  # affiliation/institution of the contributor
    description: str  # brief (~1 sentence) description of the speedrun

    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig

    # by default, this is fineweb_edu_tokenized
    tokenized_dataset: InputName | LMMixtureDatasetConfig = fineweb_edu_tokenized

    @property
    def vocab_size(self) -> int:
        # TODO (Nikil): this doesn't interact well with different types (InputName, LMMixtureDatasetConfig, ExecutorStep)
        # Need to change this to automatically figure out vocab size
        # return load_tokenizer(unwrap_versioned_value(self.tokenized_dataset.tokenizer)).vocab_size
        return llama3_tokenizer_vocab_size

    def estimate_model_flops(self) -> float:
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

        # Get MoE parameters if available, otherwise use defaults for dense models
        num_experts = getattr(self.model_config, "n_routed_experts", 1)
        num_shared_experts = getattr(self.model_config, "n_shared_experts", 0)
        num_experts_per_tok = getattr(self.model_config, "num_experts_per_tok", 1)

        logger.info(
            f"num_experts: {num_experts}, num_shared_experts: {num_shared_experts}, num_experts_per_tok: {num_experts_per_tok}"
        )
        logger.info(f"total_tokens: {total_tokens}")
        logger.info(f"num_shared_experts: {num_shared_experts}")
        logger.info(f"num_experts_per_tok: {num_experts_per_tok}")

        flops_per_token = lm_flops_per_token(
            self.model_config.hidden_dim,
            self.model_config.intermediate_dim,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.model_config.num_heads,
            self.model_config.seq_len,
            self.vocab_size,
            glu=True,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
        )  # only fwd flops

        flops_per_token *= 3  # fwd + bwd

        estimated_model_flops = flops_per_token * total_tokens

        logger.info(
            f"""The rough estimated compute (calculated as model FLOPs / (Estimate of MFU))
            for your run is around {estimated_model_flops/0.5:.2e} FLOPs assuming an MFU of 0.5.
            This is calculated based on plausible MFU values and can be used as a rough estimate to
            guide your config/training setup."""
        )

        return estimated_model_flops

    @property
    def device_flops(self) -> float:
        """Get the peak FLOPs/s for the device type."""
        return self.train_config.resources.device_flops()

    @property
    def num_devices(self) -> int:
        """Get the number of devices."""
        return self.train_config.resources.total_device_count()


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

    step_times = get_step_times_from_wandb(run_id=wandb_run_id, entity=config.wandb_entity, project=config.wandb_project)
    if not step_times:
        logger.error("No step times available; analysis aborted.")
        return

    model_flops = config.speedrun_config.estimate_model_flops()
    model_size = compute_num_parameters(config.speedrun_config.model_config, config.speedrun_config.vocab_size)
    total_tokens = (
        config.speedrun_config.train_config.train_batch_size
        * config.speedrun_config.model_config.seq_len
        * config.speedrun_config.train_config.num_train_steps
    )

    training_time_seconds = sum(step_times)
    training_time_in_minutes = training_time_seconds / 60
    training_hardware_flops = training_time_seconds * config.speedrun_config.num_devices * config.speedrun_config.device_flops
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

    run_info = {
        "speedrun_config": {
            "author": config.speedrun_config.author or "marin-community",
            "affiliation": config.speedrun_config.affiliation,
            "description": config.speedrun_config.description,
            "model_size": model_size,
            "total_tokens": total_tokens,
            "model_config": dataclasses.asdict(config.speedrun_config.model_config),
            "train_config": dataclasses.asdict(config.speedrun_config.train_config),
            "tokenized_dataset": str(config.speedrun_config.tokenized_dataset),
            "resources": dataclasses.asdict(config.speedrun_config.train_config.resources),
            "run_completion_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "wandb_run_link": full_wandb_url,
            "model_flops": model_flops,
            "training_time_in_minutes": training_time_in_minutes,
            "training_hardware_flops": training_hardware_flops,
            "eval/paloma/c4_en/bpb": (
                float(wandb_metrics.get("eval/paloma/c4_en/bpb"))
                if wandb_metrics.get("eval/paloma/c4_en/bpb") is not None
                else None
            ),
        },
    }

    logger.info(f"Speedrun stats: {run_info}")

    output_data = {"runs": [run_info]}
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
            wandb_run_id = train_step  # gets converted to output path when passing it into the results step

    assert wandb_run_id is not None, "Could not extract wandb run ID from train step"

    results_step = ExecutorStep(
        name=f"speedrun/{name}-speedrun_results",
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
