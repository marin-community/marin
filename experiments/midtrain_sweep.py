from experiments.simple_train_config import SimpleTrainConfig
from experiments.isoflop_sweep import generate_isoflop_sweep
from experiments.tootsie.exp1295_32b import nemotron_mix
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, InputName
import os
import re
import ray
from dataclasses import dataclass, replace
from levanter.data.text import LMMixtureDatasetConfig
from levanter.schedule import BatchSchedule
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from experiments.defaults import default_train
from marin.utils import fsspec_get_curr_subdirectories
from marin.training.training import TrainLmOnPodConfig
from marin.training.training import (
    run_levanter_train_lm,
)
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3


@dataclass
class DecayConfig:
    checkpoint_path: str
    train_config: SimpleTrainConfig
    train_lm_on_pod_config: TrainLmOnPodConfig


_STEP_DIR_RE = re.compile(r"^step-(\d+)$")


def _parse_step_from_path(path: str) -> int | None:
    """Extract the last `step-<n>` occurrence from a path string, if present."""
    m = re.search(r"step-(\d+)", path)
    return int(m.group(1)) if m else None


def get_latest_checkpoint_before_decay(checkpoint_path: str, num_train_steps: int, decay: float) -> str | None:
    """
    For a GCS checkpoint directory, list immediate subdirectories named `step-<n>` and
    find the latest step folder that is <= num_train_steps * (1 - decay).

    Writes a checkpoint.jsonl file to the output path containing the chosen checkpoint path.

    This does not recurse; only immediate children are considered.
    """
    # List immediate subdirectories and filter to step-* only
    subdirs = fsspec_get_curr_subdirectories(checkpoint_path)
    step_dirs: list[tuple[int, str]] = []
    for d in subdirs:
        name = os.path.basename(d.rstrip("/"))
        m = _STEP_DIR_RE.match(name)
        if m:
            step_num = int(m.group(1))
            step_dirs.append((step_num, d))

    # Target step before the decay
    target_step = int(num_train_steps * (1.0 - decay))

    # Choose the latest step not exceeding the target
    chosen_step_path: str | None = None
    chosen_step_num: int | None = None
    for s, p in step_dirs:
        if s <= target_step and (chosen_step_num is None or s > chosen_step_num):
            chosen_step_num, chosen_step_path = s, p

    return chosen_step_path


# Executed outside the Executor Context because we need to use the function step_to_lm_mixture_component
# which requires it
def generate_tokenized_dataset_config(
    num_train_steps: int,
    decay: float,
    train_batch_size: int,
    stable_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig,
    decay_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig,
) -> LMMixtureDatasetConfig:
    # Determine the stage change in training steps, aligned to mixture block boundaries.
    target_stable_steps = int(num_train_steps * (1.0 - decay))

    # Use actual batch schedule to convert steps -> example offset, then align to mixture block size.
    batch_schedule = BatchSchedule(train_batch_size)

    # We will use the stable mixture's block size since we replace it below to construct the merged mix.
    if not isinstance(stable_data_mix, LMMixtureDatasetConfig):
        raise ValueError("stable_data_mix must be an LMMixtureDatasetConfig for midtrain scheduling")

    mixture_block_size = stable_data_mix.mixture_block_size
    target_offset = batch_schedule.global_data_offset_by_step(target_stable_steps)
    aligned_offset = (target_offset // mixture_block_size) * mixture_block_size
    num_stable_train_steps = batch_schedule.find_step_containing_offset(aligned_offset)

    weights_list = []
    if isinstance(stable_data_mix, LMMixtureDatasetConfig):
        stable_data_mix_configs = stable_data_mix.configs
        stable_data_mix_weights = stable_data_mix.train_weights
    else:
        stable_data_mix_configs = {
            "stable": step_to_lm_mixture_component(stable_data_mix, include_raw_paths=True),
        }
        stable_data_mix_weights = {
            "stable": 1.0,
        }

    if isinstance(decay_data_mix, LMMixtureDatasetConfig):
        decay_data_mix_configs = decay_data_mix.configs
        decay_data_mix_weights = decay_data_mix.train_weights
    else:
        decay_data_mix_configs = {
            "decay": step_to_lm_mixture_component(decay_data_mix, include_raw_paths=True),
        }
        decay_data_mix_weights = {
            "decay": 1.0,
        }

    # TODO(chris): Support non tokenized step and inputname

    weights_list = [
        (0, {name: weight for name, weight in stable_data_mix_weights.items()}),
        (num_stable_train_steps, {name: weight for name, weight in decay_data_mix_weights.items()}),
    ]
    data_mix_configs = stable_data_mix_configs | decay_data_mix_configs
    dataset_config = replace(stable_data_mix, configs=data_mix_configs, train_weights=weights_list)

    return dataset_config


# Must be created outside of the ExecutorStep Context for some reason. If not, it will think that
# paloma/4chan is in marin-us-central2?
def generate_train_config(dataset_config, model_config, train_config, experiment_name) -> TrainLmOnPodConfig:
    train_config = default_train(
        name=f"{experiment_name}",
        tokenized=dataset_config,
        model_config=model_config,
        train_config=train_config,
        eval_harness_tasks=[],
        tags=(
            # f"FLOPs={config.train_config.flops:.1e}",
            f"d={model_config.hidden_dim}",
            f"L={model_config.num_layers}",
            f"B={train_config.train_batch_size}",
            f"steps={train_config.num_train_steps}",
            # f"tpu={tpu_type}",
        ),
        only_return_config=True,
    )

    return train_config


# Executed within the ExecutorStep context because we need the ExecutorStep to grab
# the checkpoint path
@ray.remote
def generate_midtrain_step(config: DecayConfig):
    chosen_step_path = get_latest_checkpoint_before_decay(
        config.checkpoint_path, config.train_config.num_train_steps, config.train_config.decay
    )

    train_config = replace(
        config.train_lm_on_pod_config,
        train_config=replace(
            config.train_lm_on_pod_config.train_config, initialize_from_checkpoint_path=chosen_step_path
        ),
    )
    ray.get(run_levanter_train_lm.remote(train_config))


# Test
isoflop_steps, isoflop_model_configs, isoflop_train_configs = generate_isoflop_sweep(
    nemotron_mix, experiment_name="nemo-wider-depth-adapt"
)
sweep_step = isoflop_steps[1]

decay_data_tokenized = get_dolmino_step_llama3("flan")

decay_dataset_config = generate_tokenized_dataset_config(
    num_train_steps=isoflop_train_configs[1].num_train_steps,
    decay=isoflop_train_configs[1].decay,
    train_batch_size=isoflop_train_configs[1].train_batch_size,
    stable_data_mix=nemotron_mix,
    decay_data_mix=decay_data_tokenized,
)
decay_train_config = generate_train_config(
    dataset_config=decay_dataset_config,
    model_config=isoflop_model_configs[1],
    train_config=isoflop_train_configs[1],
    experiment_name="nemo-wider-depth-adapt-flan-test",
)
decay_step = ExecutorStep(
    name="checkpoints/nemo-wider-depth-adapt-flan-test",
    fn=generate_midtrain_step,
    config=DecayConfig(
        train_lm_on_pod_config=decay_train_config,
        checkpoint_path=output_path_of(sweep_step, "checkpoints"),
        train_config=isoflop_train_configs[1],
    ),
)

if __name__ == "__main__":
    executor_main(steps=[decay_step])
