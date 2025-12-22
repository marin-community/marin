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
sizes while keeping the total training FLOPs roughly constant.  It is intended
as a lightweight scaffold for ISOFlop scaling law experiments.
"""

import dataclasses
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig

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
from marin.scaling_laws.isoflop_analysis import (
    CandidateConfig,
    IsoFlopSweepConfig,
    generate_isoflop_train_args,
)


@dataclass(frozen=True)
class IsoFlopExperimentConfig:
    """Configuration for isoflop experiments with dataset and eval settings.

    Composes an IsoFlopSweepConfig for core sweep parameters and adds
    experiment-specific settings like tokenized dataset and eval tasks.
    """

    tokenized_dataset: InputName | str
    """Tokenized dataset to train on."""

    sweep_config: IsoFlopSweepConfig = dataclasses.field(default_factory=IsoFlopSweepConfig)
    """Core sweep parameters (budgets, seq_len, etc.)."""

    eval_tasks: tuple[EvalTaskConfig, ...] | None = None
    """Evaluation tasks to run after training (disabled by default)."""

    base_optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: CautiousConfig(
            learning_rate=1.0,  # Placeholder
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=0.1,
            beta1=0.95,
            beta2=0.98,
            epsilon=1e-15,
            max_grad_norm=1,
            adamc_weight_decay=True,
            lr_schedule="linear",
            decay=0.2,
        ),
    )

    base_train_config: SimpleTrainConfig = dataclasses.field(
        default_factory=lambda: SimpleTrainConfig(
            resources=ResourceConfig.with_tpu("v5p-8"),
            train_batch_size=1,
            num_train_steps=50_000,
            learning_rate=1.0,  # Placeholder
            weight_decay=0.1,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            decay=0.2,
        )
    )


def generate_isoflop_steps(
    config: IsoFlopExperimentConfig,
    experiment_name: str,
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    """Generate executor steps for an ISOFlop sweep.

    Uses generate_isoflop_train_args() from the scaling_laws library to get
    model configs, optimizer configs, and other arguments, then constructs
    ExecutorSteps using default_train().

    Returns:
        A tuple of:
        - steps: Training and evaluation ExecutorSteps for the sweep.
        - candidates: CandidateConfig for each training run (contains budget, hidden_size,
          num_layers, batch_size, train_steps, learning_rate, etc.)
    """
    vocab_size = get_vocab_size_for_tokenizer(config.sweep_config.tokenizer)

    # Get training arguments from the library
    train_args_list = generate_isoflop_train_args(
        sweep_config=config.sweep_config,
        experiment_name=experiment_name,
        vocab_size=vocab_size,
        base_optimizer_config=config.base_optimizer_config,
    )

    train_steps_list: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    candidates: list[CandidateConfig] = []

    for args in train_args_list:
        # Build SimpleTrainConfig from the library-provided arguments
        train_cfg = replace(
            config.base_train_config,
            train_batch_size=args.candidate.batch_size,
            learning_rate=args.candidate.learning_rate,
            num_train_steps=args.candidate.train_steps,
            resources=ResourceConfig.with_tpu(args.tpu_type),
            optimizer_config=args.optimizer_config,
        )

        # Create training step using default_train
        train_step = default_train(
            name=args.run_name,
            tokenized=config.tokenized_dataset,
            model_config=args.model_config,
            train_config=train_cfg,
            eval_harness_tasks=[],
            tags=args.tags,
        )

        # Pin to static output path for checkpoint reuse
        train_step = train_step.with_output_path(args.output_path)
        train_steps_list.append(train_step)
        candidates.append(args.candidate)

        # Evaluation on the latest checkpoint for each ISOFlop run
        if config.eval_tasks:
            eval_step = default_eval(
                train_step,
                resource_config=train_cfg.resources,
                evals=config.eval_tasks,
            )
            eval_steps.append(eval_step)

    all_steps: list[ExecutorStep] = [*train_steps_list, *eval_steps]
    return all_steps, candidates


def generate_isoflop_sweep(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    experiment_name: str,
    sweep_config: IsoFlopSweepConfig | None = None,
    eval_tasks: tuple[EvalTaskConfig, ...] | None = None,
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    """Generate an ISOFlop sweep for a tokenized dataset.

    Args:
        tokenized: Tokenized dataset to train on.
        experiment_name: Name suffix for the experiment (e.g., 'nemo', 'dclm').
        sweep_config: Optional custom sweep config. Uses defaults if None.
        eval_tasks: Optional evaluation tasks to run after training.

    Returns:
        A tuple of:
        - steps: Training and evaluation ExecutorSteps for the sweep.
        - candidates: CandidateConfig for each training run with full config details.
    """
    config = IsoFlopExperimentConfig(
        tokenized_dataset=tokenized,
        sweep_config=sweep_config or IsoFlopSweepConfig(),
        eval_tasks=eval_tasks,
    )
    steps, candidates = generate_isoflop_steps(config, experiment_name)

    return steps, candidates


dclm_tokenized = dataclasses.replace(
    default_tokenize(
        name="dclm_baseline",
        dataset=downloads["dclm_baseline"],
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/dclm_baseline-0206f1/"),
)


dclm_mix = lm_mixture_data_config(
    components={"dclm": dclm_tokenized},
    weights={"dclm": 1.0},
    num_validation_sequences={"dclm": 1024},
)

dolma3_mix_tokenized = dataclasses.replace(
    default_tokenize(
        name="dolma3_mix-150B-1025",
        dataset=downloads["dolma3_mix_150b_1025"],
        tokenizer=llama3_tokenizer,
    ).with_output_path("tokenized/dolma3_mix-150B-1025-15d04ee/"),
)

dolma3_mix = lm_mixture_data_config(
    components={"dolma3_mix-150B-1025": dolma3_mix_tokenized},
    weights={"dolma3_mix-150B-1025": 1.0},
    num_validation_sequences={"dolma3_mix-150B-1025": 1024},
)

MARIN_SCALING_SUITES = {
    "nemotron": generate_isoflop_sweep(nemotron_mix, experiment_name="nemo-wider-depth-adapt"),
    "common_pile": generate_isoflop_sweep(comma_main_mixture(permutation_type="linear"), experiment_name="comma-mix"),
    "common_pile_feistel": generate_isoflop_sweep(
        comma_main_mixture(permutation_type="feistel"), experiment_name="comma-mix-feistel"
    ),
    "dclm-default": generate_isoflop_sweep(dclm_mix, experiment_name="dclm-default"),
    "dolma3_mix_150b": generate_isoflop_sweep(dolma3_mix, experiment_name="dolma3-mix-150b-1025"),
}

if __name__ == "__main__":
    steps, _ = MARIN_SCALING_SUITES["dolma3_mix_150b"]
    executor_main(steps=steps)
