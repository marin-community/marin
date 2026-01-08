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

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
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
from marin.scaling_laws import (
    DEFAULT_BUDGETS,
    DEFAULT_FLOP_TOLERANCE,
    DEFAULT_SEQ_LEN,
    DEFAULT_STEPS_PER_RUN,
    CandidateConfig,
    ScalingRecipe,
    generate_isoflop_train_args,
    pick_v5p_type,
    solve_for_batch_size,
    solve_for_train_steps,
)


def _round_to_power_of_two(x: float) -> int:
    """Round x UP to the nearest power of 2."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


@dataclass(frozen=True)
class Marin2025Recipe:
    """Marin 2025 scaling recipe with all hyperparameters and formulas.

    This recipe implements all the Marin-specific decisions for scaling experiments.
    """

    name: str = "marin-2025"

    # --- Learning rate scaling ---
    # lr = lr_constant * sqrt(batch_size) / hidden_dim
    lr_constant: float = 0.33

    # --- Beta2 scaling for Adam ---
    # beta2 = beta2_base ** (batch_size / beta2_batch_divisor)
    beta2_base: float = 0.98
    beta2_batch_divisor: float = 128

    # --- Optimizer hyperparameters ---
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    beta1: float = 0.95
    epsilon: float = 1e-15
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"
    decay: float = 0.2

    # --- Architecture ratios ---
    mlp_ratio: int = 4
    hidden_head_ratio: int = 128

    # --- Architecture formula for depth-to-width scaling ---
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9

    # --- Constraints ---
    max_learning_rate: float = 0.01
    min_batch_size: int = 8

    # --- Search bounds for isoflop sweeps ---
    min_hidden_pow: int = 9
    max_hidden_pow: int = 12
    small_budget_step_size: int = 128
    large_budget_step_size: int = 256
    budget_step_threshold: float = 9e18

    def _compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        """Compute learning rate from batch size and hidden dim."""
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def _compute_beta2(self, batch_size: int) -> float:
        """Compute beta2 from batch size."""
        return self.beta2_base ** (batch_size / self.beta2_batch_divisor)

    def compute_num_layers(self, hidden_size: int) -> int:
        """Compute number of layers from hidden size using the depth-width formula."""
        hs_pow = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + (hs_pow * self.layer_scaling_factor) - self.layer_formula_offset)
        )

    def _get_step_size(self, budget: float) -> int:
        """Get hidden_size search step size based on budget."""
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def _compute_params_for_hidden_size(self, hidden_size: int, vocab_size: int) -> int:
        """Compute approximate parameter count for a given hidden size."""
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)
        head_size = hidden_size // n_heads

        embed_params = vocab_size * hidden_size * 2
        q_proj = hidden_size * head_size * n_heads
        kv_proj = 2 * hidden_size * head_size * n_heads
        o_proj = head_size * n_heads * hidden_size
        attn_params = q_proj + kv_proj + o_proj
        mlp_params = 3 * hidden_size * intermediate_dim
        norm_params = 2 * hidden_size
        layer_params = attn_params + mlp_params + norm_params
        total_layer_params = num_layers * layer_params
        final_norm = hidden_size

        return embed_params + total_layer_params + final_norm

    def hidden_size_for_params(self, target_params: int, vocab_size: int) -> int:
        """Find the hidden size that gives approximately target_params."""
        min_hidden = 2**self.min_hidden_pow
        max_hidden = 2**self.max_hidden_pow

        best_hidden = min_hidden
        best_diff = abs(self._compute_params_for_hidden_size(min_hidden, vocab_size) - target_params)

        for hidden_size in range(min_hidden, max_hidden + 1, 64):
            params = self._compute_params_for_hidden_size(hidden_size, vocab_size)
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_hidden = hidden_size

        return best_hidden

    def build_model_config(self, target_params: int, vocab_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build a Qwen3 model config for a target parameter count."""
        hidden_size = self.hidden_size_for_params(target_params, vocab_size)
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)

        return Qwen3Config(
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            max_seq_len=seq_len,
            rope=Llama3RotaryEmbeddingsConfig(),
        )

    def _build_model_config_from_hidden_size(self, hidden_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build model config from hidden_size directly."""
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)

        return Qwen3Config(
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            max_seq_len=seq_len,
            rope=Llama3RotaryEmbeddingsConfig(),
        )

    def estimate_memory_bytes(
        self,
        model_config: LlamaConfig,
        batch_size: int,
        vocab_size: int,
        optim_mult: int = 3,
        dtype_size: int = 4,
        fudge_factor: float = 2.0,
    ) -> int:
        """Estimate float32 memory usage in bytes for training."""
        param_count = self._compute_params_for_hidden_size(model_config.hidden_dim, vocab_size)
        param_bytes = param_count * optim_mult * dtype_size
        act_bytes = (batch_size * model_config.max_seq_len) * (
            (model_config.hidden_dim * model_config.num_layers) + vocab_size * fudge_factor
        )
        total_bytes = param_bytes + act_bytes
        return int(total_bytes * fudge_factor)

    def build_optimizer_config(self, candidate: CandidateConfig, vocab_size: int) -> OptimizerConfig:
        """Build optimizer config for a candidate."""
        hidden_size = self.hidden_size_for_params(candidate.target_params, vocab_size)
        learning_rate = self._compute_learning_rate(candidate.batch_size, hidden_size)
        beta2 = self._compute_beta2(candidate.batch_size)

        return CautiousConfig(
            learning_rate=learning_rate,
            weight_decay=self.weight_decay,
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=beta2,
            epsilon=self.epsilon,
            max_grad_norm=self.max_grad_norm,
            adamc_weight_decay=True,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
        )

    def candidate_configs(
        self,
        budget: float,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> Iterator[CandidateConfig]:
        """Yield candidate configurations within the FLOP budget."""
        step_size = self._get_step_size(budget)
        min_hidden = 2**self.min_hidden_pow
        max_hidden = 2**self.max_hidden_pow

        for hidden_size in range(min_hidden, max_hidden + 1, step_size):
            model_config = self._build_model_config_from_hidden_size(hidden_size, seq_len)

            batch_exact = solve_for_batch_size(model_config, vocab_size, budget, steps_per_run, seq_len)
            batch_size = _round_to_power_of_two(batch_exact)

            lr = self._compute_learning_rate(batch_size, hidden_size)
            while lr > self.max_learning_rate:
                batch_size //= 2
                lr = self._compute_learning_rate(batch_size, hidden_size)

            if batch_size < self.min_batch_size:
                continue

            train_steps = round(solve_for_train_steps(model_config, vocab_size, budget, batch_size, seq_len))

            achieved_flops = 3 * model_config.flops_per_token(vocab_size, seq_len) * batch_size * train_steps * seq_len
            if abs(achieved_flops - budget) / budget > flop_tolerance:
                continue

            tokens = batch_size * train_steps * seq_len
            target_params = self._compute_params_for_hidden_size(hidden_size, vocab_size)

            yield CandidateConfig(
                batch_size=batch_size,
                train_steps=train_steps,
                tokens=tokens,
                target_params=target_params,
                flops_budget=budget,
            )


MARIN_2025_RECIPE = Marin2025Recipe()
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

    # Base config for training runs (values overridden per-candidate via optimizer_config)
    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=1,
        num_train_steps=50_000,
        learning_rate=1.0,  # Overridden via optimizer_config
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
