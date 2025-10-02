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

from experiments.simple_train_config import SimpleTrainConfig
from experiments.isoflop_sweep import IsoFlopSweepConfig, generate_isoflop_steps
from experiments.tootsie.exp1295_32b import nemotron_mix
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, InputName
import os
import re
import ray
from dataclasses import dataclass, replace
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.schedule import BatchSchedule
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from experiments.defaults import default_train, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.utils import fsspec_get_curr_subdirectories
from marin.training.training import TrainLmOnPodConfig
from marin.training.training import (
    run_levanter_train_lm,
)
from experiments.dolmino.tokenize_dolmino import get_dolmino_step_llama3
from experiments.isoflop_sweep import pick_v4_type, pick_v5p_type, get_vocab_size_for_tokenizer
from marin.resources import TpuPodConfig
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.task_configs import EvalTaskConfig, MMLU_5_SHOT
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL


@dataclass
class DecayConfig:
    checkpoint_path: str
    train_config: SimpleTrainConfig
    train_lm_on_pod_config: TrainLmOnPodConfig


_STEP_DIR_RE = re.compile(r"^step-(\d+)$")
CHECKPOINT_BASE_REGION = "marin-us-central1"
REGION_TO_TPU_TYPE = {
    "marin-us-central1": "v5p",
    "marin-us-central2": "v4",
}


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


def get_train_job_name(sweep_step: ExecutorStep, decay_data_name: str, pt_to_mt_split: str) -> str:
    sweep_step_name_without_checkpoint_prefix = sweep_step.name.split("/")[-1]
    return f"{sweep_step_name_without_checkpoint_prefix}-mt-{decay_data_name}-{pt_to_mt_split}"


# NOTE(chris): Executed outside the Executor Context because we need to use the function step_to_lm_mixture_component
# which requires the input to be a Step instead of a string. In the executor context, the ExecutorStep or InputName
# gets converted to a string.
def generate_tokenized_dataset_config(
    num_train_steps: int,
    decay: float,
    train_batch_size: int,
    stable_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig,
    decay_data_mix: InputName | ExecutorStep | LMMixtureDatasetConfig,
    pt_weight: float,
    mt_weight: float,
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

    tokens_needed = (num_train_steps - num_stable_train_steps) * train_batch_size * 4096
    print(f"tokens needed for decay: {tokens_needed / 1e9:.2f}B")

    # Normalzie by 100 because mt_weight passed in as a percentage from 1-100
    print(f"num special tokens needed: {tokens_needed * mt_weight / (100 * 1e9):.2f}B")

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

    # Build 70/30 mixture for the decay stage (normalized over the union of dataset keys)
    union_keys = set((stable_data_mix_configs or {}).keys()) | set((decay_data_mix_configs or {}).keys())

    def _normalize_to_union(weights_dict: dict[str, float], keys: set[str]) -> dict[str, float]:
        total = sum(weights_dict.get(k, 0.0) for k in keys)
        if total <= 0:
            return {k: 0.0 for k in keys}
        return {k: weights_dict.get(k, 0.0) / total for k in keys}

    stable_norm = _normalize_to_union(stable_data_mix_weights, union_keys)
    decay_norm = _normalize_to_union(decay_data_mix_weights, union_keys)

    mixed_decay = {k: pt_weight * stable_norm[k] + mt_weight * decay_norm[k] for k in union_keys}
    mixed_total = sum(mixed_decay.values())
    if mixed_total > 0:
        mixed_decay = {k: v / mixed_total for k, v in mixed_decay.items()}

    weights_list = [
        (0, {name: weight for name, weight in stable_data_mix_weights.items()}),
        (num_stable_train_steps, mixed_decay),
    ]
    data_mix_configs = stable_data_mix_configs | decay_data_mix_configs
    dataset_config = replace(stable_data_mix, configs=data_mix_configs, train_weights=weights_list)

    return dataset_config


# NOTE(chris): Default train config must be created outside of the ExecutorStep Context for some reason.
# If not, it will think that paloma/4chan is in marin-us-central2 and lead to error about model/train region
# mismatch.
def generate_train_config(dataset_config, model_config, train_config, experiment_name, budget) -> TrainLmOnPodConfig:
    vocab_size = get_vocab_size_for_tokenizer("stanford-crfm/marin-tokenizer")
    if os.getenv("BUCKET") != CHECKPOINT_BASE_REGION:
        tpu_type = pick_v4_type(
            model_config,
            model_config.hidden_dim,
            model_config.num_layers,
            train_config.train_batch_size,
            model_config.seq_len,
            vocab_size,
        )
        tpu_config = TpuPodConfig(tpu_type=tpu_type)
        train_config = replace(train_config, resources=tpu_config)
    else:
        tpu_type = pick_v5p_type(
            model_config,
            model_config.hidden_dim,
            model_config.num_layers,
            train_config.train_batch_size,
            model_config.seq_len,
            vocab_size,
        )

    wandb_tags = (
        f"FLOPs={budget:.1e}",
        f"d={model_config.hidden_dim}",
        f"L={model_config.num_layers}",
        f"B={train_config.train_batch_size}",
        f"steps={train_config.num_train_steps}",
        f"tpu={tpu_type}",
    )
    # NOTE(chris): how to check experiment is done in a different region?
    train_config = default_train(
        name=f"{experiment_name}",
        tokenized=dataset_config,
        model_config=model_config,
        train_config=train_config,
        eval_harness_tasks=[],
        tags=wandb_tags,
        only_return_config=True,
    )

    return wandb_tags, train_config


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
decay_data_name = "flan"
decay_data_tokenized = get_dolmino_step_llama3("flan")
# NOTE(chris): Working on the low isoflop budgets for now.
sweep_cfg = IsoFlopSweepConfig(tokenized_dataset=nemotron_mix, budgets=[1e18, 3e18, 6e18, 1e19, 3e19])
isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, _ = generate_isoflop_steps(
    sweep_cfg, "nemo-wider-depth-adapt"
)

eval_tasks = (
    MMLU_5_SHOT,
    # MMLU_PRO_5_SHOT,
    # EvalTaskConfig("commonsense_qa_sl", num_fewshot=10),
    EvalTaskConfig("mmlu_sl", num_fewshot=0, task_alias="mmlu_sl_0_shot"),
    EvalTaskConfig("mmlu_sl", num_fewshot=5, task_alias="mmlu_sl_5_shot"),
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=0, task_alias="mmlu_sl_verb_0_shot"),
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
)
steps = []
# pt_to_mt_splits = [(60, 40), (70, 30), (80, 20)]
pt_to_mt_splits = [(70, 30), (80, 20)]

# BASELINE (no midtrain / decay)
for isoflop_step, isoflop_model_config, isoflop_train_config, isoflop_budget in zip(
    isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, strict=False
):
    experiment_name = isoflop_step.name.split("/")[-1]
    wandb_tags = (
        f"FLOPs={isoflop_budget:.1e}",
        f"d={isoflop_model_config.hidden_dim}",
        f"L={isoflop_model_config.num_layers}",
        f"B={isoflop_train_config.train_batch_size}",
        f"steps={isoflop_train_config.num_train_steps}",
        f"tpu={isoflop_train_config.resources.tpu_type}",
    )
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=experiment_name,
        model_path=output_path_of(isoflop_step),
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V5p_8_FULL,
        wandb_tags=wandb_tags,
    )
    steps.append(eval_step)

# Dolmino DCLM
dolmino_dclm = get_dolmino_step_llama3("dclm")
dolmino_sweep_cfg = IsoFlopSweepConfig(tokenized_dataset=dolmino_dclm, budgets=[1e18, 3e18, 6e18])
dolmino_isoflop_steps, dolmino_isoflop_model_configs, dolmino_isoflop_train_configs, dolmino_isoflop_budgets, _ = (
    generate_isoflop_steps(dolmino_sweep_cfg, "dolmino-dclm-sweep")
)

for dolmino_isoflop_step, dolmino_isoflop_model_config, dolmino_isoflop_train_config, dolmino_isoflop_budget in zip(
    dolmino_isoflop_steps,
    dolmino_isoflop_model_configs,
    dolmino_isoflop_train_configs,
    dolmino_isoflop_budgets,
    strict=False,
):
    experiment_name = dolmino_isoflop_step.name.split("/")[-1]
    wandb_tags = (
        f"FLOPs={dolmino_isoflop_budget:.1e}",
        f"d={dolmino_isoflop_model_config.hidden_dim}",
        f"L={dolmino_isoflop_model_config.num_layers}",
        f"B={dolmino_isoflop_train_config.train_batch_size}",
        f"steps={dolmino_isoflop_train_config.num_train_steps}",
        f"tpu={dolmino_isoflop_train_config.resources.tpu_type}",
    )
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=experiment_name,
        model_path=output_path_of(dolmino_isoflop_step),
        evals=eval_tasks,
        resource_config=SINGLE_TPU_V5p_8_FULL,
        wandb_tags=wandb_tags,
    )
    steps.append(eval_step)

# Midtrain mix
for pt_weight, mt_weight in pt_to_mt_splits:
    for sweep_step, model_config, train_config, budget in zip(
        isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, strict=False
    ):
        print(f"pt_weight: {pt_weight}, mt_weight: {mt_weight}")
        print(f"budget: {budget}")
        experiment_name = get_train_job_name(sweep_step, decay_data_name, f"{pt_weight}-{mt_weight}")

        decay_dataset_config = generate_tokenized_dataset_config(
            num_train_steps=train_config.num_train_steps,
            decay=train_config.decay,
            train_batch_size=train_config.train_batch_size,
            stable_data_mix=nemotron_mix,
            decay_data_mix=decay_data_tokenized,
            pt_weight=pt_weight,
            mt_weight=mt_weight,
        )
        wandb_tags, decay_train_config = generate_train_config(
            dataset_config=decay_dataset_config,
            model_config=model_config,
            train_config=train_config,
            experiment_name=experiment_name,
            budget=budget,
        )
        decay_step = ExecutorStep(
            name=f"checkpoints/{experiment_name}",
            fn=generate_midtrain_step,
            config=DecayConfig(
                train_lm_on_pod_config=decay_train_config,
                checkpoint_path=output_path_of(sweep_step, "checkpoints"),
                train_config=train_config,
            ),
        )

        eval_step = evaluate_levanter_lm_evaluation_harness(
            model_name=experiment_name,
            model_path=output_path_of(decay_step),
            evals=eval_tasks,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            wandb_tags=wandb_tags,
        )
        steps.append(eval_step)
        # steps.append(decay_step)


def generate_synthetic_ablation_steps(
    synthetic_data_name,
    synthetic_data_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    data_weight_list,
):
    synthetic_ablation_steps = []
    for pt_weight, mt_weight in data_weight_list:
        for sweep_step, model_config, train_config, budget in zip(
            isoflop_steps, isoflop_model_configs, isoflop_train_configs, isoflop_budgets, strict=False
        ):
            experiment_name = get_train_job_name(sweep_step, synthetic_data_name, f"{pt_weight}-{mt_weight}")
            wandb_tags, synthetic_ablation_train_config = generate_train_config(
                dataset_config=generate_tokenized_dataset_config(
                    num_train_steps=train_config.num_train_steps,
                    decay=train_config.decay,
                    train_batch_size=train_config.train_batch_size,
                    stable_data_mix=nemotron_mix,
                    decay_data_mix=synthetic_data_tokenized,
                    pt_weight=pt_weight,
                    mt_weight=mt_weight,
                ),
                model_config=model_config,
                train_config=train_config,
                experiment_name=experiment_name,
                budget=budget,
            )
            synthetic_ablation_step = ExecutorStep(
                name=f"checkpoints/{experiment_name}",
                fn=generate_midtrain_step,
                config=DecayConfig(
                    train_lm_on_pod_config=synthetic_ablation_train_config,
                    checkpoint_path=output_path_of(sweep_step, "checkpoints"),
                    train_config=train_config,
                ),
            )
            eval_step = evaluate_levanter_lm_evaluation_harness(
                model_name=experiment_name,
                model_path=output_path_of(synthetic_ablation_step),
                evals=eval_tasks,
                resource_config=SINGLE_TPU_V5p_8_FULL,
                wandb_tags=wandb_tags,
            )
            synthetic_ablation_steps.append(eval_step)
            # synthetic_ablation_steps.append(synthetic_ablation_step)

    return synthetic_ablation_steps


# Baseline
synthetic_data = InputName.hardcoded(
    "gs://marin-us-central1/documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa-eb7854"
)
baseline_data_tokenized = default_tokenize(
    name="dclm-shard-1-1",
    dataset=synthetic_data,
    tokenizer=llama3_tokenizer,  # tokenizing using text_key="text" which is the normal text key
)
synthetic_baseline_steps = generate_synthetic_ablation_steps(
    "dclm-shard-1-1",
    baseline_data_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

# Synthetically generated data
synthetic_data_generated_tokenized = default_tokenize(
    name="dclm-wrap-qa-1024-v2",
    dataset=synthetic_data,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
synthetic_ablation_steps = generate_synthetic_ablation_steps(
    "dclm-wrap-qa-1024-v2",
    synthetic_data_generated_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

wrapmed = InputName.hardcoded("gs://marin-us-central1/documents/synthetic-dclm-subset-chunk-qa-1024-wrapmed-3e0d3e")
wrapmed_tokenized = default_tokenize(
    name="dclm-wrap-med-1024",
    dataset=wrapmed,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
wrapmed_steps = generate_synthetic_ablation_steps(
    "dclm-wrap-med-1024",
    wrapmed_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

nemo_qa = InputName.hardcoded("gs://marin-us-central1/documents/synthetic-dclm-subset-chunk-qa-1024-nemo-qa-f36058")
nemo_qa_tokenized = default_tokenize(
    name="dclm-nemo-qa-1024",
    dataset=nemo_qa,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
nemo_qa_steps = generate_synthetic_ablation_steps(
    "dclm-nemo-qa-1024",
    nemo_qa_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

dclm_mcq = InputName.hardcoded("gs://marin-us-central1/documents/synthetic-dclm-subset-chunk-qa-1024-mcq-ad37b6")
dclm_mcq_tokenized = default_tokenize(
    name="dclm-mcq-1024",
    dataset=dclm_mcq,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
dclm_mcq_steps = generate_synthetic_ablation_steps(
    "dclm-mcq-1024",
    dclm_mcq_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

dclm_qa_1b = InputName.hardcoded("gs://marin-us-central1/documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa-1b-57383e")
dclm_qa_1b_tokenized = default_tokenize(
    name="dclm-qa-1b-1024",
    dataset=dclm_qa_1b,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
dclm_qa_1b_steps = generate_synthetic_ablation_steps(
    "dclm-qa-1b-1024",
    dclm_qa_1b_tokenized,
    isoflop_steps,
    isoflop_model_configs,
    isoflop_train_configs,
    isoflop_budgets,
    [(70, 30)],
)

if __name__ == "__main__":
    # central1
    executor_main(steps=wrapmed_steps + synthetic_baseline_steps + synthetic_ablation_steps + nemo_qa_steps)
    # executor_main(steps=nemo_qa_steps)
    # executor_main(steps=dclm_qa_1b_steps)
