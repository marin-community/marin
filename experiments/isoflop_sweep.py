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

"""Generate ISOFlop sweep steps for varying model sizes on a target dataset.

This script constructs `ExecutorStep` objects that train models of different
sizes while keeping the total training FLOPs roughly constant.
"""

from dataclasses import replace

from levanter.data.text import LMMixtureDatasetConfig

from experiments.evals.evals import default_eval
from experiments.evals.task_configs import EvalTaskConfig
from experiments.common_pile.tokenize_common_pile import comma_main_mixture
from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.simple import downloads
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.processing.tokenize import get_vocab_size_for_tokenizer, lm_mixture_data_config
from marin.scaling_laws import (
    DEFAULT_BUDGETS,
    CandidateConfig,
    ScalingRecipe,
    generate_isoflop_train_args,
    pick_v5p_type,
)

MARIN_2025_RECIPE = ScalingRecipe(name="marin-2025")
"""Default Marin scaling recipe."""


def create_isoflop_sweep_steps(
    tokenized: InputName | str | LMMixtureDatasetConfig,
    experiment_name: str,
    recipe: ScalingRecipe,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    tokenizer: str = "stanford-crfm/marin-tokenizer",
    eval_tasks: tuple[EvalTaskConfig, ...] | None = None,
    seq_len: int = 4096,
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    """Create ExecutorSteps for an ISOFlop sweep.

    This function creates ExecutorSteps directly in experiment code, using
    `generate_isoflop_train_args()` from the library to compute configs.

    Args:
        tokenized: Tokenized dataset to train on.
        experiment_name: Name suffix for the experiment (e.g., 'nemo', 'dclm').
        recipe: ScalingRecipe with hyperparameters - must be explicitly specified.
        budgets: FLOP budgets to sweep over.
        tokenizer: Tokenizer to use for vocab size.
        eval_tasks: Optional evaluation tasks to run after training.

    Returns:
        A tuple of:
        - steps: Training and evaluation ExecutorSteps for the sweep.
        - candidates: CandidateConfig for each training run with full config details.
    """
    vocab_size = get_vocab_size_for_tokenizer(tokenizer)

    # Library provides the training arguments (model configs, optimizer configs, etc.)
    train_args_list = generate_isoflop_train_args(
        budgets=budgets,
        experiment_name=experiment_name,
        vocab_size=vocab_size,
        recipe=recipe,
    )

    # Base config for training runs
    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=1,
        num_train_steps=50_000,
        learning_rate=1.0,  # Placeholder, will be overridden
        weight_decay=recipe.weight_decay,
        min_lr_ratio=recipe.min_lr_ratio,
        lr_schedule=recipe.lr_schedule,
        decay=recipe.decay,
    )

    train_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    candidates: list[CandidateConfig] = []

    # Create ExecutorSteps for each candidate configuration
    for args in train_args_list:
        candidate = args.candidate

        # Build model and optimizer configs using the recipe
        model_config = recipe.build_model_config(candidate.target_params, vocab_size, seq_len)
        optimizer_config = recipe.build_optimizer_config(candidate, vocab_size)
        tpu_type = pick_v5p_type(candidate, vocab_size, seq_len, recipe)

        train_cfg = replace(
            base_train_config,
            train_batch_size=candidate.batch_size,
            learning_rate=optimizer_config.learning_rate,
            num_train_steps=candidate.train_steps,
            resources=ResourceConfig.with_tpu(tpu_type),
            optimizer_config=optimizer_config,
        )

        # Create training step
        train_step = default_train(
            name=args.run_name,
            tokenized=tokenized,
            model_config=model_config,
            train_config=train_cfg,
            eval_harness_tasks=[],
            tags=args.tags,
        )

        # Pin to static output path for checkpoint reuse
        train_step = train_step.with_output_path(args.output_path)
        train_steps.append(train_step)
        candidates.append(candidate)

        # Create evaluation step if eval tasks specified
        if eval_tasks:
            eval_step = default_eval(
                train_step,
                resource_config=train_cfg.resources,
                evals=eval_tasks,
            )
            eval_steps.append(eval_step)

    all_steps: list[ExecutorStep] = [*train_steps, *eval_steps]
    return all_steps, candidates


# --- Tokenized Datasets ---

dclm_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=downloads["dclm_baseline"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dclm_baseline-0206f1/")

dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

dolma3_mix_tokenized = default_tokenize(
    name="dolma3_mix-150B-1025",
    dataset=downloads["dolma3_mix_150b_1025"],
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/dolma3_mix-150B-1025-15d04ee/")

dolma3_mix = lm_mixture_data_config(
    components={"dolma3_mix-150B-1025": dolma3_mix_tokenized},
    weights={"dolma3_mix-150B-1025": 1.0},
    num_validation_sequences={"dolma3_mix-150B-1025": 1024},
)


MARIN_SCALING_SUITES = {
    "nemotron": create_isoflop_sweep_steps(
        tokenized=nemotron_mix,
        experiment_name="nemo-wider-depth-adapt",
        recipe=MARIN_2025_RECIPE,
    ),
    "common_pile": create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="linear"),
        experiment_name="comma-mix",
        recipe=MARIN_2025_RECIPE,
    ),
    "common_pile_feistel": create_isoflop_sweep_steps(
        tokenized=comma_main_mixture(permutation_type="feistel"),
        experiment_name="comma-mix-feistel",
        recipe=MARIN_2025_RECIPE,
    ),
    "dclm-default": create_isoflop_sweep_steps(
        tokenized=dclm_mix,
        experiment_name="dclm-default",
        recipe=MARIN_2025_RECIPE,
    ),
    "dolma3_mix_150b": create_isoflop_sweep_steps(
        tokenized=dolma3_mix,
        experiment_name="dolma3-mix-150b-1025",
        recipe=MARIN_2025_RECIPE,
    ),
}

if __name__ == "__main__":
    steps, _ = MARIN_SCALING_SUITES["dolma3_mix_150b"]
    executor_main(steps=steps)
