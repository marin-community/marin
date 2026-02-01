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

"""C-AdamC (Cautious Adam) scaling recipe for ISOFlop sweeps.

This is the original Marin 2025 recipe using CautiousConfig optimizer.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

from levanter.data.text import LMMixtureDatasetConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig

from experiments.defaults import default_train
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import EvalTaskConfig
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName
from marin.processing.tokenize import get_vocab_size_for_tokenizer
from marin.scaling_laws import CandidateConfig, pick_v5p_type

# --- Constants ---
SEQ_LEN: int = 4096
STEPS_PER_RUN: int = 2**16


def _round_to_power_of_two(x: float) -> int:
    """Round x UP to the nearest power of 2."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _format_run_name(
    budget: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    experiment_name: str,
) -> str:
    """Format run name: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}"""
    return f"isoflop-{budget:.0e}-d{hidden_size}-L{num_layers}-B{batch_size}-{experiment_name}"


@dataclass(frozen=True)
class CAdamCRecipe:
    """C-AdamC scaling recipe using CautiousConfig optimizer."""

    name: str = "c-adamc"
    tokenizer: str = "stanford-crfm/marin-tokenizer"

    @property
    def vocab_size(self) -> int:
        return get_vocab_size_for_tokenizer(self.tokenizer)

    # --- Learning rate scaling ---
    lr_constant: float = 0.33

    # --- Adam hyperparameters ---
    weight_decay: float = 0.1
    beta1: float = 0.95
    beta2_base: float = 0.98
    beta2_batch_divisor: float = 128
    epsilon: float = 1e-15
    max_grad_norm: float = 1.0

    # --- Schedule ---
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    lr_schedule: str = "linear"
    decay: float = 0.2

    # --- Architecture ratios ---
    mlp_ratio: int = 4
    hidden_head_ratio: int = 128

    # --- Depth-to-width scaling ---
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9

    # --- Constraints ---
    max_learning_rate: float = 0.01
    min_batch_size: int = 8
    max_batch_size: int = 8192
    base_max_params: float = 12e9
    base_max_params_budget: float = 3e20
    global_max_params: float = 1e12

    # --- Search step sizes ---
    small_budget_step_size: int = 128
    large_budget_step_size: int = 256
    budget_step_threshold: float = 2e19

    def _compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def _compute_beta2(self, batch_size: int) -> float:
        return max(0.95, self.beta2_base ** (batch_size / self.beta2_batch_divisor))

    def _compute_num_layers(self, hidden_size: int) -> int:
        hs_pow = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + (hs_pow * self.layer_scaling_factor) - self.layer_formula_offset)
        )

    def _get_step_size(self, budget: float) -> int:
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def _max_params_for_budget(self, budget: float) -> float:
        scaling = self.base_max_params * math.sqrt(budget / self.base_max_params_budget)
        return min(max(self.base_max_params, scaling), self.global_max_params)

    def _build_model_config(self, hidden_size: int, seq_len: int = SEQ_LEN) -> LlamaConfig:
        if hidden_size % self.hidden_head_ratio != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by hidden_head_ratio ({self.hidden_head_ratio})."
            )
        num_layers = self._compute_num_layers(hidden_size)
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
        candidate: CandidateConfig,
        optim_mult: int = 3,
        dtype_size: int = 4,
        fudge_factor: float = 2.0,
    ) -> int:
        model_config = candidate.model_config
        batch_size = candidate.batch_size
        seq_len = model_config.max_seq_len
        hidden = model_config.hidden_dim
        intermediate = getattr(model_config, "intermediate_dim", hidden * self.mlp_ratio)
        layers = model_config.num_layers

        param_count = model_config.total_trainable_params(self.vocab_size)
        param_bytes = param_count * optim_mult * dtype_size

        hidden_act = batch_size * seq_len * hidden * 2
        attn_act = batch_size * seq_len * hidden * 4 * 2
        mlp_act = batch_size * seq_len * intermediate * 2
        per_layer_act = hidden_act + attn_act + mlp_act
        act_bytes = per_layer_act * max(layers * 3 // 4, 4)

        embed_bytes = self.vocab_size * hidden * 2

        total_bytes = param_bytes + act_bytes + embed_bytes
        return int(total_bytes * fudge_factor)

    def _build_model_configs(self, budget: float, seq_len: int = SEQ_LEN) -> Iterator[LlamaConfig]:
        step_size = self._get_step_size(budget)
        for hidden_size in range(2**9, 2**17, step_size):
            yield self._build_model_config(hidden_size, seq_len)

    def _build_candidate_config(
        self,
        model_config: LlamaConfig,
        tokens: float,
        flops_budget: float,
        seq_len: int = SEQ_LEN,
    ) -> CandidateConfig | None:
        hidden_size = model_config.hidden_dim

        batch_exact = tokens / (STEPS_PER_RUN * seq_len)
        batch_size = _round_to_power_of_two(batch_exact)

        lr = self._compute_learning_rate(batch_size, hidden_size)
        while lr > self.max_learning_rate:
            batch_size //= 2
            lr = self._compute_learning_rate(batch_size, hidden_size)

        if batch_size < self.min_batch_size or batch_size > self.max_batch_size:
            return None

        train_steps = round(tokens / (batch_size * seq_len))
        actual_tokens = batch_size * train_steps * seq_len

        beta2 = self._compute_beta2(batch_size)

        optimizer_config = CautiousConfig(
            learning_rate=lr,
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

        return CandidateConfig(
            model_config=model_config,
            optimizer_config=optimizer_config,
            batch_size=batch_size,
            train_steps=train_steps,
            tokens=actual_tokens,
            flops_budget=flops_budget,
        )

    def candidates_for_budget(self, budget: float, seq_len: int = SEQ_LEN) -> Iterator[CandidateConfig]:
        max_params = self._max_params_for_budget(budget)
        for model_config in self._build_model_configs(budget, seq_len):
            params = model_config.total_trainable_params(self.vocab_size)
            if params > max_params:
                continue
            flops_per_token = model_config.flops_per_token(self.vocab_size, seq_len)
            tokens = budget / (3 * flops_per_token)
            candidate = self._build_candidate_config(model_config, tokens, budget, seq_len)
            if candidate is not None:
                yield candidate


RECIPE = CAdamCRecipe()


def create_isoflop_sweep_steps(
    tokenized: InputName | str | LMMixtureDatasetConfig,
    experiment_name: str,
    budgets: tuple[float, ...],
    eval_tasks: tuple[EvalTaskConfig, ...] | None = None,
    seq_len: int = SEQ_LEN,
) -> tuple[list[ExecutorStep], list[CandidateConfig]]:
    """Create ExecutorSteps for an ISOFlop sweep using C-AdamC recipe."""
    candidates = [c for budget in budgets for c in RECIPE.candidates_for_budget(budget, seq_len)]

    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=1,
        num_train_steps=50_000,
        learning_rate=1.0,
    )

    train_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []

    for candidate in candidates:
        model_config = candidate.model_config
        estimated_memory = RECIPE.estimate_memory_bytes(candidate)
        tpu_type = pick_v5p_type(estimated_memory)

        run_name = _format_run_name(
            candidate.flops_budget,
            model_config.hidden_dim,
            model_config.num_layers,
            candidate.batch_size,
            experiment_name,
        )
        output_path = f"checkpoints/isoflop/{run_name}"

        params = model_config.total_trainable_params(RECIPE.vocab_size)
        tags = (
            f"FLOPs={candidate.flops_budget:.1e}",
            f"N={params:.1e}",
            f"B={candidate.batch_size}",
            f"steps={candidate.train_steps}",
            f"tokens={candidate.tokens:.1e}",
        )

        train_cfg = replace(
            base_train_config,
            train_batch_size=candidate.batch_size,
            learning_rate=candidate.optimizer_config.learning_rate,
            num_train_steps=candidate.train_steps,
            resources=ResourceConfig.with_tpu(tpu_type),
            optimizer_config=candidate.optimizer_config,
        )

        train_step = default_train(
            name=run_name,
            tokenized=tokenized,
            model_config=model_config,
            train_config=train_cfg,
            eval_harness_tasks=[],
            tags=tags,
        )
        train_step = train_step.with_output_path(output_path)
        train_steps.append(train_step)

        if eval_tasks:
            eval_step = default_eval(
                train_step,
                resource_config=train_cfg.resources,
                evals=eval_tasks,
            )
            eval_steps.append(eval_step)

    return [*train_steps, *eval_steps], candidates
