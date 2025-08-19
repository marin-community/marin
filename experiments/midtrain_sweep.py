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
class GetCheckpointConfig:
    checkpoint_path: str
    # output_path: str


@dataclass
class DecayConfig:
    # experiment_name: str
    # output_path: str
    checkpoint_path: str
    train_config: SimpleTrainConfig
    # model_config: LlamaConfig
    # dataset_config: LMMixtureDatasetConfig
    train_lm_on_pod_config: TrainLmOnPodConfig

    # Data configs
    # stable_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig
    # decay_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig


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
    if not checkpoint_path.startswith("gs://"):
        raise ValueError("checkpoint_path must be a GCS path starting with 'gs://'")

    # # If a specific step is provided (…/step-<n>/), list its parent directory instead.
    # tail = os.path.basename(checkpoint_path.rstrip("/"))
    # if re.fullmatch(r"step-\d+", tail):
    #     base_dir = os.path.dirname(checkpoint_path.rstrip("/"))
    # else:
    #     base_dir = checkpoint_path.rstrip("/")

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

    # Steps between the step encoded in the provided path (if any) and the target
    start_step = _parse_step_from_path(checkpoint_path)
    if start_step is None:
        start_step = chosen_step_num or 0
    steps_between = max(0, target_step - start_step)

    print(f"Chosen step: {chosen_step_path}")
    print(f"Steps between: {steps_between}")
    # print(f"Train config: {ctrain_config}")

    # with fsspec.open(config.output_path, "w") as f:
    #     if chosen_step_path is not None:
    #         f.write(json.dumps({
    #             "checkpoint_path": chosen_step_path,
    #         }))
    #     else:
    #         f.write(json.dumps({
    #             "checkpoint_path": "error",
    #         }))

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
    # TODO(chris): Uncomment later
    dataset_config = replace(stable_data_mix, configs=data_mix_configs, train_weights=weights_list)

    # NOTE(chris): This works still
    # dataset_config = config.stable_data_mix
    print(dataset_config)

    # lm_varying_mixture_data_config(
    #     components={"stable": config.stable_data_mix, "decay": config.decay_data_mix},
    #     weights_list=[
    #         (0, {"stable": 1.0}),
    #         (num_stable_train_steps, {"decay": 1.0}),
    #     ]
    # )

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
    # with fsspec.open(config.checkpoint_path, "r") as f:
    #     chosen_step_path = json.load(f)["checkpoint_path"]

    # if chosen_step_path == "error":
    #     raise ValueError("No checkpoint found")
    chosen_step_path = get_latest_checkpoint_before_decay(
        config.checkpoint_path, config.train_config.num_train_steps, config.train_config.decay
    )

    # Determine the stage change in training steps, aligned to mixture block boundaries.
    # target_stable_steps = int(config.train_config.num_train_steps * (1.0 - config.train_config.decay))

    # # Use actual batch schedule to convert steps -> example offset, then align to mixture block size.
    # batch_schedule = BatchSchedule(config.train_config.train_batch_size)

    # # We will use the stable mixture's block size since we replace it below to construct the merged mix.
    # if not isinstance(config.stable_data_mix, LMMixtureDatasetConfig):
    #     raise ValueError("stable_data_mix must be an LMMixtureDatasetConfig for midtrain scheduling")

    # mixture_block_size = config.stable_data_mix.mixture_block_size
    # target_offset = batch_schedule.global_data_offset_by_step(target_stable_steps)
    # aligned_offset = (target_offset // mixture_block_size) * mixture_block_size
    # num_stable_train_steps = batch_schedule.find_step_containing_offset(aligned_offset)

    # weights_list = []
    # if isinstance(config.stable_data_mix, LMMixtureDatasetConfig):
    #     stable_data_mix_configs = config.stable_data_mix.configs
    #     stable_data_mix_weights = config.stable_data_mix.train_weights
    # else:
    #     stable_data_mix_configs = {
    #         "stable": as_lm_dataset_source_config(config.stable_data_mix, include_raw_paths=True),
    #     }
    #     stable_data_mix_weights = {
    #         "stable": 1.0,
    #     }

    # if isinstance(config.decay_data_mix, LMMixtureDatasetConfig):
    #     decay_data_mix_configs = config.decay_data_mix.configs
    #     decay_data_mix_weights = config.decay_data_mix.train_weights
    # else:
    #     decay_data_mix_configs = {
    #         "decay": as_lm_dataset_source_config(config.decay_data_mix, include_raw_paths=True),
    #     }
    #     decay_data_mix_weights = {
    #         "decay": 1.0,
    #     }

    # #TODO(chris): Support non tokenized step and inputname

    # weights_list = [
    #     (0, {name: weight for name, weight in stable_data_mix_weights.items()}),
    #     (num_stable_train_steps, {name: weight for name, weight in decay_data_mix_weights.items()}),
    # ]
    # data_mix_configs = stable_data_mix_configs | decay_data_mix_configs
    # # TODO(chris): Uncomment later
    # dataset_config = replace(config.stable_data_mix, configs=data_mix_configs, train_weights=weights_list)

    # NOTE(chris): This works still
    # dataset_config = config.stable_data_mix
    # print(dataset_config)

    # lm_varying_mixture_data_config(
    #     components={"stable": config.stable_data_mix, "decay": config.decay_data_mix},
    #     weights_list=[
    #         (0, {"stable": 1.0}),
    #         (num_stable_train_steps, {"decay": 1.0}),
    #     ]
    # )

    # decay_train_config = replace(
    #     config.train_config,
    #     initialize_from_checkpoint_path=chosen_step_path,
    # )
    # print(f"New train config: {decay_train_config}")

    # train_config = default_train(
    #     name=f"{config.experiment_name}",
    #     tokenized=config.dataset_config,
    #     model_config=config.model_config,
    #     train_config=decay_train_config,
    #     eval_harness_tasks=[],
    #     tags=(
    #         # f"FLOPs={config.train_config.flops:.1e}",
    #         f"d={config.model_config.hidden_dim}",
    #         f"L={config.model_config.num_layers}",
    #         f"B={config.train_config.train_batch_size}",
    #         f"steps={config.train_config.num_train_steps}",
    #         # f"tpu={tpu_type}",
    #     ),
    #     only_return_config=True,
    # )

    # Since we are already in an Executor Context, we just set the path to the output path from our config
    # normally, it would just use this_output_path but that is assuming we ran the default_train function outside
    # of the Executor Context
    # train_config = replace(train_config, output_path=config.output_path)

    # ray.get(run_levanter_train_lm.remote(train_config))
    train_config = replace(
        config.train_lm_on_pod_config,
        train_config=replace(
            config.train_lm_on_pod_config.train_config, initialize_from_checkpoint_path=chosen_step_path
        ),
    )
    ray.get(run_levanter_train_lm.remote(train_config))
    # return train_config


# components = {"pretrain": nemotron_mix, "midtrain": nemotron_mix}
# # Note: start index is in sequences, not steps. If you want to switch at step S, use S * train_batch_size.
# weights_list = [
#   (0,   {"pretrain": 1.0, "midtrain": 0.0}),
#   (X,   {"pretrain": 0.0, "midtrain": 1.0}),
# ]
# data_cfg = lm_varying_mixture_data_config(components, weights_list)


# def decay_schedule(config: )

# stage1 = (0, {self.rare_data_name: self.rare_weight_stage1, self.common_data_name: self.common_weight_stage1})
#         stage2 = (
#             transition_idx,
#             {self.rare_data_name: self.rare_weight_stage2, self.common_data_name: self.common_weight_stage2},
#         )

#         if transition_idx == 0:
#             weights_list = [stage2]
#         else:
#             weights_list = [stage1, stage2]

# Test
isoflop_steps, isoflop_model_configs, isoflop_train_configs = generate_isoflop_sweep(
    nemotron_mix, experiment_name="nemo-wider-depth-adapt"
)
sweep_step = isoflop_steps[1]

# TODO(chris): Don't use a ExecutorStep for this.. just call the function directly
# NOTE(chris): We should relaly remove the hashing of the config because now we can't access the config.
# decay_step = ExecutorStep(
#     name="checkpoints/nemo-wider-depth-adapt-decay-test",
#     fn=generate_midtrain_step,
#     config=GetCheckpointConfig(
#         checkpoint_path=output_path_of(sweep_step, "checkpoints"),
#         train_config=isoflop_train_configs[1],
#         model_config=isoflop_model_configs[1],
#         stable_data_mix=nemotron_mix,
#         decay_data_mix=nemotron_mix,
#         experiment_name="nemo-wider-depth-adapt-decay",
#     )
# )

decay_data_tokenized = get_dolmino_step_llama3("flan")
# decay_step = generate_midtrain_step(
#     DecayConfig(
#         # checkpoint_path=output_path_of(sweep_step, "checkpoints"),
#         # TODO(chris): This needs to be executor-step'd
#         checkpoint_path="gs://marin-us-central1/checkpoints/isoflop-1e+18-d640-L7-B16-nemo-wider-depth-adapt-ca0cca/checkpoints",
#         train_config=isoflop_train_configs[1],
#         model_config=isoflop_model_configs[1],
#         stable_data_mix=nemotron_mix,
#         decay_data_mix=decay_data_tokenized,
#         experiment_name="nemo-wider-depth-adapt-flan",
#     )
# )

# checkpoint_before_decay = ExecutorStep(
#     name="raw/scratch/checkpoint-before-decay",
#     fn=get_latest_checkpoint_before_decay,
#     config=GetCheckpointConfig(
#         checkpoint_path=output_path_of(sweep_step, "checkpoints"),
#         output_path=this_output_path("checkpoint.jsonl"),
#     )
# )

# ExecutorStep(
#     name="raw/scratch/midtrain-step",
#     fn=generate_midtrain_step,
#     config=DecayConfig(
#         checkpoint_path=output_path_of(checkpoint_before_decay, "checkpoint.jsonl"),
#         train_config=isoflop_train_configs[1],
#         model_config=isoflop_model_configs[1],
#         stable_data_mix=nemotron_mix,
#         decay_data_mix=decay_data_tokenized,
#         experiment_name="nemo-wider-depth-adapt-flan",
#     )
# )

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
    # config=DecayConfig(
    #     # checkpoint_path=output_path_of(checkpoint_before_decay, "checkpoint.jsonl"),
    #     output_path=this_output_path(),
    #     checkpoint_path=output_path_of(sweep_step, "checkpoints"),
    #     train_config=isoflop_train_configs[1],
    #     model_config=isoflop_model_configs[1],
    #     dataset_config=decay_dataset_config,
    #     experiment_name="nemo-wider-depth-adapt-flan",
    # )
    config=DecayConfig(
        train_lm_on_pod_config=decay_train_config,
        checkpoint_path=output_path_of(sweep_step, "checkpoints"),
        train_config=isoflop_train_configs[1],
    ),
)

# generate_midtrain_step(
#     DecayConfig(
#         checkpoint_path=output_path_of(checkpoint_before_decay, "checkpoint.jsonl"),
#         train_config=isoflop_train_configs[1],
#         model_config=isoflop_model_configs[1],
#         stable_data_mix=nemotron_mix,
#         decay_data_mix=decay_data_tokenized,
#         experiment_name="nemo-wider-depth-adapt-flan",
#     )
# )

if __name__ == "__main__":
    executor_main(steps=[decay_step])
