# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from enum import Enum

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.processing.tokenize import add_validation_sets_to_mixture, lm_data_config
from marin.speedrun.paloma_local_download import speedrun_paloma_tokenized
from marin.training.training import TrainLmOnPodConfig
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT
from marin.utils import asdict_excluding

import wandb
from experiments.defaults import _get_tokenizer_for_train, default_train
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B

logger = logging.getLogger("ray")


def _num_accelerator_chips(resources: ResourceConfig) -> int:
    """Return the total number of accelerator chips for the provided resources."""
    return resources.chip_count()


@dataclass(frozen=True)
class Author:
    """Author information for a speedrun."""

    name: str
    """Name of the author/individual submitting the speedrun."""

    affiliation: str
    """Affiliation/institution of the contributor."""

    url: str | None = None
    """Optional URL to author's profile/website."""


@dataclass
class SpeedrunConfig:
    author: Author
    """Author information for the speedrun."""
    description: str
    """Brief (~1 sentence) description of the speedrun."""

    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig

    # by default, this is fineweb_edu_subcache_10B
    tokenized_dataset: InputName | LMMixtureDatasetConfig = fineweb_edu_subcache_10B

    @property
    def vocab_size(self) -> int:
        return llama3_tokenizer_vocab_size

    def as_json_dict(self) -> dict:
        """Convert SpeedrunConfig to a JSON-serializable dictionary."""

        def _make_serializable(obj):
            """Recursively convert non-serializable objects (like Enums) to serializable forms."""
            if isinstance(obj, Enum):
                return obj.name
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(v) for v in obj]
            return obj

        # runtimeenv is not serializable
        train_config_dict = asdict_excluding(self.train_config, exclude={"resources", "runtime_env"})
        resources_dict = asdict_excluding(self.train_config.resources, exclude={"runtime_env"})
        return {
            "author": {"name": self.author.name, "affiliation": self.author.affiliation, "url": self.author.url},
            "description": self.description,
            "model_config": _make_serializable(dataclasses.asdict(self.model_config)),
            "train_config": _make_serializable(train_config_dict),
            "tokenized_dataset": str(self.tokenized_dataset),
            "resources": _make_serializable(resources_dict),
        }

    def print_run_info(self) -> None:
        """Print information about speedrun configuration, device FLOPS, model FLOPS, and hardware configuration.
        Mainly to sanity-check runs by calling speedrun_config.print_run_info() before actually running it."""

        num_devices = self.num_devices
        num_chips = self.num_chips
        device_flops = self.device_flops
        total_peak_flops = device_flops * num_chips

        # Print simplified config info
        logger.info("----- START OF PRINT RUN INFO -----")
        logger.info("Speedrun Configuration:")
        logger.info(json.dumps(self.as_json_dict(), indent=4))

        model_flops = self.compute_model_flops()

        logger.info("Hardware and Model FLOPS Information:")
        logger.info(f"Number of devices: {num_devices}")
        logger.info(f"Number of chips: {num_chips}")
        logger.info(f"Device FLOPs: {device_flops:.2e} FLOP/s")
        logger.info(f"Total peak hardware FLOPs: {total_peak_flops:.2e} FLOP/s")
        logger.info(f"Model FLOPs: {model_flops:.2e} FLOP")

        # model size
        model_size = self.model_config.total_trainable_params(self.vocab_size)
        if model_size is None:
            logger.info("Model size: unknown (model did not report total_trainable_params).")
        else:
            logger.info(f"Model size: {model_size/1e6:.2f} million parameters")
        logger.info("----- END OF PRINT RUN INFO -----")

    def compute_model_flops(self) -> float:
        # TODO (Nikil): make this a helper and handle edge-cases
        context_length: int | None = self.train_config.train_seq_len
        if context_length is None:
            context_length = self.model_config.max_seq_len

        assert isinstance(context_length, int)

        if isinstance(self.train_config.train_batch_size, list) and len(self.train_config.train_batch_size) > 0:
            from levanter.schedule import BatchSchedule

            batch_schedule = BatchSchedule(self.train_config.train_batch_size)
            total_tokens = batch_schedule.global_data_offset_by_step(self.train_config.num_train_steps) * context_length
        else:
            # integer batch size
            total_tokens = self.train_config.train_batch_size * self.train_config.num_train_steps * context_length

        flops_per_token = self.model_config.flops_per_token(self.vocab_size, context_length)
        if flops_per_token is None:
            raise ValueError("Model config must provide flops_per_token to compute model FLOPs.")

        flops_per_token *= 3  # fwd + bwd

        estimated_model_flops = flops_per_token * total_tokens

        logger.info(
            f"""
The rough estimated compute (calculated as (total model FLOPs / Assumed MFU)) for your run is probably between:
      * {estimated_model_flops/0.5:.2e} FLOPs assuming an MFU of 0.5, and
      * {estimated_model_flops/0.2:.2e} FLOPs assuming an MFU of 0.2.

This is calculated based on assumed MFU values and can be used as a rough estimate to guide your config/training setup.
""".strip()
        )

        return estimated_model_flops

    @property
    def device_flops(self) -> float:
        """Get the peak FLOPs/s for the device type."""
        device_flops = self.train_config.resources.device_flops()
        if device_flops is None:
            raise ValueError("Resources must provide device_flops() for speedrun calculations.")
        return device_flops

    @property
    def num_devices(self) -> int:
        """Get the number of devices."""
        return self.train_config.resources.chip_count()

    @property
    def num_chips(self) -> int:
        """Get the number of accelerator chips."""

        return _num_accelerator_chips(self.train_config.resources)


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
        return [
            row["throughput/duration"]
            for row in run.scan_history(keys=["throughput/duration"])
            if "throughput/duration" in row
        ]
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

    model_flops = config.speedrun_config.compute_model_flops()
    model_size = config.speedrun_config.model_config.total_trainable_params(config.speedrun_config.vocab_size)
    if isinstance(config.speedrun_config.train_config, SimpleTrainConfig):
        context_length = config.speedrun_config.train_config.train_seq_len
    else:
        context_length = config.speedrun_config.train_config.train_config.train_seq_len

    if context_length is None:
        context_length = config.speedrun_config.model_config.max_seq_len

    total_tokens = (
        config.speedrun_config.train_config.train_batch_size
        * context_length
        * config.speedrun_config.train_config.num_train_steps
    )
    flops_per_token = config.speedrun_config.model_config.flops_per_token(
        config.speedrun_config.vocab_size, context_length
    )

    training_time = sum(step_times)
    num_devices = config.speedrun_config.num_devices
    num_chips = config.speedrun_config.num_chips
    training_hardware_flops = training_time * num_chips * config.speedrun_config.device_flops
    logger.info(f"Training time: {training_time:.2f} seconds")
    # devices
    logger.info(f"Number of devices: {num_devices}")
    logger.info(f"Number of chips: {num_chips}")
    logger.info(f"Device FLOPs: {config.speedrun_config.device_flops:.2e} FLOPs")
    logger.info(f"Training hardware FLOPs: {training_hardware_flops:.2e} FLOPs")

    # get wandb run and metrics
    run = wandb.Api().run(f"{config.wandb_entity}/{config.wandb_project}/{wandb_run_id}")
    wandb_metrics = {
        "eval/paloma/c4_en/bpb": run.summary.get("eval/paloma/c4_en/bpb", None),
    }

    try:
        run.summary.update({"model/flops_per_token": float(flops_per_token)})
    except Exception as exc:
        logger.warning(f"Failed to write flops_per_token to Weights & Biases: {exc}")

    wandb_num_devices = run.summary.get("num_devices", None)
    if wandb_num_devices is not None:
        if wandb_num_devices != config.speedrun_config.num_devices:
            logger.warning(
                f"Number of devices in wandb ({wandb_num_devices}) does not match number of devices in config"
                f"({config.speedrun_config.num_devices}). Going with config value."
            )

    wandb_device_flops = run.summary.get("throughput/theoretical_flops_per_device", None)
    if wandb_device_flops is not None:
        if wandb_device_flops != config.speedrun_config.device_flops:
            logger.warning(
                f"Device FLOPS in wandb ({wandb_device_flops}) does not match device FLOPS in config "
                f"({config.speedrun_config.device_flops})."
                "Going with config value."
            )

    # Get timestamps in UTC
    start_time = datetime.datetime.fromisoformat(run.createdAt.replace("Z", "+00:00"))  # Convert ISO string to datetime
    runtime_seconds = run.summary["_runtime"]
    end_time = start_time + datetime.timedelta(seconds=runtime_seconds)

    wandb_run_link = f"https://wandb.ai/{config.wandb_entity}/{config.wandb_project}/runs/{wandb_run_id}"

    # Start with the base config as a dictionary
    speedrun_dict = config.speedrun_config.as_json_dict()

    # Add computed values and results
    run_info = {
        **speedrun_dict,
        # Model metrics
        "model_size": model_size,
        "total_tokens": total_tokens,
        "model_flops": model_flops,
        "model_flops_per_token": flops_per_token,
        # Training metrics
        "num_devices": num_devices,
        "num_chips": num_chips,
        "device_flops": config.speedrun_config.device_flops,
        "training_time": training_time,
        "training_hardware_flops": training_hardware_flops,
        "eval/paloma/c4_en/bpb": (
            float(wandb_metrics.get("eval/paloma/c4_en/bpb"))
            if wandb_metrics.get("eval/paloma/c4_en/bpb") is not None
            else None
        ),
        # Run metadata
        "run_completion_timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "wandb_run_link": wandb_run_link,
    }

    logger.info(f"Speedrun info and results: {run_info}")

    output_data = {"runs": [{"run_info": run_info}]}
    with fsspec.open(config.output_path, "w") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)
    logger.info(f"Speedrun info and results written to {config.output_path}")


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

    logger.info(f"Running speedrun {name}")
    config.print_run_info()

    run_tags = ["speedrun"] + (tags or [])

    train_config = dataclasses.replace(config.train_config, data_seed=42)
    if isinstance(config.tokenized_dataset, InputName | ExecutorStep):
        pretraining_data = lm_data_config(
            training_set=config.tokenized_dataset,
            validation_sets=speedrun_paloma_tokenized(tokenizer=(_get_tokenizer_for_train(config.tokenized_dataset))),
            # TODO: when should we update this
            permutation_type="feistel",
        )
    else:
        pretraining_data = add_validation_sets_to_mixture(
            config.tokenized_dataset,
            speedrun_paloma_tokenized(tokenizer=(_get_tokenizer_for_train(config.tokenized_dataset))),
        )

    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=pretraining_data,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=None,
        override_output_path=override_output_path,
        use_default_validation=False,
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
