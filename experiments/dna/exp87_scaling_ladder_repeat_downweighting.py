# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 87: Scaling ladder for repeat downweighting.

Train DNA models at 3 FLOP budgets x 2 weighting strategies = 6 runs to understand
whether repeat downweighting (lowercase_weight) becomes less important with scale.

FLOPs is the single lever — model size and tokens are derived using the
Chinchilla-optimal ratio (D ~ 20N). Batch size, learning rate, and optimizer
hyperparameters are fixed across all runs.

NOTE: There are no established scaling laws for DNA language models. We borrow
the Chinchilla D/N ratio and Marin2025Recipe's depth-width formula as reasonable
starting points, but they were derived for natural-language models with very
different tokenizers and data distributions. The resulting model shapes and
token allocations should be treated as rough heuristics, not optimal points.

https://github.com/Open-Athena/bolinas-dna/issues/87
"""

import math

from fray.cluster import ResourceConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255
from experiments.simple_train_config import SimpleTrainConfig

# =============================================================================
# Constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_SEQ_LEN = 255
MODEL_SEQ_LEN = dna_effective_seq_len(DNA_SEQ_LEN, TOKENIZER)
assert MODEL_SEQ_LEN == 256, f"Expected 256, got {MODEL_SEQ_LEN}"

# DNA character-level tokenizer has a tiny vocab (~7 tokens).
# Hardcoded to avoid loading the tokenizer at import time in model_config_for_flops.
DNA_VOCAB_SIZE = 7

RESOURCES = ResourceConfig.with_tpu("v4-8")

TRAIN_DATASET = "bolinas-dna/genomes-v5-genome_set-mammals-intervals-v1_255_128"
VAL_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v1_255_255"

# Fixed training hyperparameters carried over from the nano speedrun grid search,
# which was itself a small-scale sweep — not guaranteed to be optimal at all scales.
# 2048 seqs * 256 tok/seq = ~524K tokens/step, matching the nano speedrun
# (which used 512 seqs * 1024 tok/seq).
BATCH_SIZE = 2048
LEARNING_RATE = 3e-3
LR_SCHEDULE = "linear"
WARMUP = 0.1
DECAY = 0.2
BETA1 = 0.95
BETA2 = 0.95
WEIGHT_DECAY = 0.1
EPSILON = 1e-15
MAX_GRAD_NORM = 1.0

# Ablation axis: repeat downweighting (lowercase_weight)
LOWERCASE_WEIGHTS = {
    "rw0.01": 0.01,  # strong downweighting (current default)
    "rw1.0": 1.0,  # no downweighting (uniform weighting)
}

# Evenly spaced in log-scale within the nano speedrun FLOP range
FLOP_BUDGETS = (1e17, 3e17, 1e18)


# =============================================================================
# Auto-derive model config from FLOP budget
# =============================================================================


def model_config_for_flops(
    flop_budget: float,
    seq_len: int = MODEL_SEQ_LEN,
    vocab_size: int = DNA_VOCAB_SIZE,
    chinchilla_ratio: float = 20.0,
    head_dim: int = 64,
    mlp_ratio: int = 4,
) -> tuple[Qwen3Config, int, int]:
    """Derive (model_config, tokens, num_steps) from a FLOP budget.

    Target N = sqrt(C / 120) from Chinchilla C = 6 * 20 * N^2.
    Hidden dim is searched in steps of head_dim; depth follows the Marin2025Recipe
    formula: num_layers = round(hidden / (64 + 4*log2(hidden) - 9)).
    Token count uses the exact flops_per_token from Levanter rather than the 6N
    approximation.

    Caveats: the Chinchilla ratio, depth-width formula, and MLP ratio are all
    borrowed from natural-language scaling work. Whether they transfer to
    character-level DNA models (tiny vocab, very different data distribution)
    is unknown — we use them as a principled starting point, not as known optima.
    """
    target_params = math.sqrt(flop_budget / (6 * chinchilla_ratio))

    best_hidden = head_dim
    best_diff = float("inf")
    for hidden in range(head_dim, 8192 + 1, head_dim):
        hs_pow = math.log2(hidden)
        num_layers = round(hidden / (64 + 4 * hs_pow - 9))
        if num_layers < 2:
            continue

        intermediate = hidden * mlp_ratio
        # Per-layer params: attn (Q/K/V/O) + MLP (gate + up + down for GLU)
        attn_params = 4 * hidden * hidden
        mlp_params = 3 * hidden * intermediate
        layer_params = attn_params + mlp_params
        total = num_layers * layer_params + vocab_size * hidden
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden

    hidden = best_hidden
    hs_pow = math.log2(hidden)
    num_layers = max(2, round(hidden / (64 + 4 * hs_pow - 9)))
    n_heads = max(1, hidden // head_dim)
    intermediate = hidden * mlp_ratio

    config = Qwen3Config(
        max_seq_len=seq_len,
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        num_heads=n_heads,
        num_kv_heads=n_heads,
        num_layers=num_layers,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=True,
    )

    # Exact token count via Levanter's flops_per_token (forward-pass FLOPs).
    # Total FLOPs = 3 * flops_per_token * D (forward + backward).
    fpt = config.flops_per_token(vocab_size, seq_len)
    tokens = round(flop_budget / (3 * fpt))
    num_steps = round(tokens / (BATCH_SIZE * seq_len))

    return config, tokens, num_steps


# =============================================================================
# Tokenize validation datasets: functional-only and nonfunctional-only
# =============================================================================

val_functional = default_tokenize(
    name=f"{VAL_DATASET.split('/')[-1]}-char-bos-functional",
    dataset=VAL_DATASET,
    tokenizer=TOKENIZER,
    format=DNALmDatasetFormat(uppercase_weight=1.0, lowercase_weight=0.0),
)

val_nonfunctional = default_tokenize(
    name=f"{VAL_DATASET.split('/')[-1]}-char-bos-nonfunctional",
    dataset=VAL_DATASET,
    tokenizer=TOKENIZER,
    format=DNALmDatasetFormat(uppercase_weight=0.0, lowercase_weight=1.0),
)

# =============================================================================
# Build training runs
# =============================================================================

training_steps = []
for budget in FLOP_BUDGETS:
    model_config, tokens, num_steps = model_config_for_flops(budget)
    actual_params = model_config.total_trainable_params(DNA_VOCAB_SIZE)

    train_config = SimpleTrainConfig(
        resources=RESOURCES,
        train_batch_size=BATCH_SIZE,
        num_train_steps=num_steps,
        learning_rate=LEARNING_RATE,
        lr_schedule=LR_SCHEDULE,
        warmup=WARMUP,
        decay=DECAY,
        beta1=BETA1,
        beta2=BETA2,
        weight_decay=WEIGHT_DECAY,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        steps_per_eval=num_steps,
        steps_per_task_eval=num_steps,
        steps_per_export=num_steps,
        data_seed=42,
    )

    for weight_name, lw in LOWERCASE_WEIGHTS.items():
        tokenized = default_tokenize(
            name=f"{TRAIN_DATASET.split('/')[-1]}-char-bos-{weight_name}",
            dataset=TRAIN_DATASET,
            tokenizer=TOKENIZER,
            format=DNALmDatasetFormat(lowercase_weight=lw),
        )

        data_config = lm_data_config(
            training_set=tokenized,
            validation_sets={
                "val_functional": val_functional,
                "val_nonfunctional": val_nonfunctional,
            },
        )

        run_name = f"exp87-{weight_name}-{actual_params / 1e6:.0f}M-{budget:.0e}"
        train_step = default_train(
            name=run_name,
            tokenized=data_config,
            model_config=model_config,
            train_config=train_config,
            tags=["dna", "exp87", "scaling_ladder", weight_name, f"flops={budget:.0e}"],
            eval_harness_tasks=[TRAITGYM_MENDELIAN_V2_255],
            eval_harness_max_packed_segments=1,
            use_default_validation=False,
            wandb_group="exp87-scaling-ladder-repeat-downweighting",
        )
        training_steps.append(train_step)


if __name__ == "__main__":
    executor_main(steps=training_steps)
