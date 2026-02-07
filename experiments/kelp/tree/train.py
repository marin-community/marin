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

"""Training pipeline for tree diffusion with TreeDiff supervision.

Generates training data by:
1. Corrupting clean programs via AST subtree replacement (forward process)
2. Computing TreeDiff edit paths from corrupted back to clean
3. Picking a random step along the path as the training target
4. Training the AR model to predict that edit (position + replacement tokens)
"""

import logging
import random as pyrandom
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import register_dataclass
from jaxtyping import Array

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import (
    TreeDiffusionAttentionParams,
    TreeDiffusionBlockParams,
)
from experiments.kelp.tree.edit_model import (
    EditModelParams,
    ar_loss,
    init_edit_params,
)
from experiments.kelp.tree.mutation import corrupt_program
from experiments.kelp.tree.subtree_bank import SubtreeBank
from experiments.kelp.tree.tokenizer import TreeDiffusionTokenizer
from experiments.kelp.tree.tree_diff import find_path

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class EditTrainingState:
    """Training state for the AR edit model."""

    step: int
    params: EditModelParams
    opt_state: optax.OptState
    key: jax.Array


@dataclass(frozen=True)
class EditTrainingConfig:
    """Configuration for tree diffusion training."""

    model: TreeDiffusionConfig
    """Model configuration."""

    max_seq_len: int = 512
    """Maximum sequence length for tokenized training examples."""

    learning_rate: float = 3e-4
    """Base learning rate."""

    weight_decay: float = 0.01
    """Weight decay for AdamW."""

    warmup_steps: int = 100
    """Number of warmup steps."""

    total_steps: int = 10000
    """Total training steps."""

    batch_size: int = 8
    """Global batch size."""

    log_interval: int = 10
    """Steps between logging."""

    checkpoint_interval: int = 1000
    """Steps between checkpoints."""

    seed: int = 42
    """Random seed."""

    output_dir: str = "checkpoints/kelp-edit"
    """Output directory for checkpoints."""

    max_corruption_steps: int = 5
    """Maximum number of AST mutations for corruption (paper's s_max)."""

    max_edit_stmts: int = 3
    """Maximum statement count per edit."""

    p_random: float = 0.2
    """Probability of using a random program instead of forward diffusion
    (paper's rho). Provides exposure to diverse starting points."""

    wandb_project: str | None = None
    """W&B project name. If set, enables W&B logging."""

    wandb_run_name: str | None = None
    """W&B run name."""


def _edit_weight_decay_mask(params: EditModelParams) -> EditModelParams:
    """Weight decay mask for EditModelParams.

    Excludes embeddings and normalization weights from weight decay.
    """
    masked_blocks = tuple(
        TreeDiffusionBlockParams(
            attn=TreeDiffusionAttentionParams(
                w_q=True,
                w_k=True,
                w_v=True,
                w_o=True,
            ),
            rms_attn=False,
            rms_mlp=False,
            mlp_gate=True,
            mlp_up=True,
            mlp_down=True,
        )
        for _ in params.blocks
    )
    return EditModelParams(
        token_embed=False,
        output_proj=True,
        blocks=masked_blocks,
        final_norm=False,
    )


def create_edit_optimizer(config: EditTrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup, cosine decay, and weight decay masking."""
    warmup_steps = min(config.warmup_steps, max(config.total_steps // 2, 1))
    decay_steps = max(config.total_steps, warmup_steps + 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            mask=_edit_weight_decay_mask,
        ),
    )


def make_edit_train_step(
    config: TreeDiffusionConfig,
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled training step for the AR edit model."""

    def loss_fn(params, token_ids, loss_mask):
        return ar_loss(params, token_ids, loss_mask, config)

    def train_step(
        state: EditTrainingState,
        batch: dict[str, Array],
    ) -> tuple[EditTrainingState, dict]:
        key, step_key = jax.random.split(state.key)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            batch["token_ids"],
            batch["loss_mask"],
        )

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm

        new_state = EditTrainingState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            key=key,
        )

        return new_state, metrics

    return jax.jit(train_step)


def generate_training_example(
    clean_source: str,
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: TreeDiffusionTokenizer,
    max_seq_len: int,
    config: EditTrainingConfig,
    rng: pyrandom.Random,
) -> tuple[list[int], list[int]] | None:
    """Generate a single training example from a clean program.

    Following the paper's forward_process_with_path:
    1. With probability p_random, pick a random program as the 'corrupted' version
    2. Otherwise, apply random AST mutations to corrupt the clean program
    3. Compute TreeDiff path from corrupted back to clean
    4. Pick a random step along the path
    5. Encode as (token_ids, loss_mask)

    Returns None if no valid training example could be generated.
    """
    # Step 1: Generate a corrupted version.
    if rng.random() < config.p_random and len(corpus) > 1:
        # Use a random program from the corpus.
        corrupted = rng.choice(corpus)
        while corrupted == clean_source and len(corpus) > 1:
            corrupted = rng.choice(corpus)
    else:
        # Apply random AST mutations.
        num_steps = rng.randint(1, config.max_corruption_steps)
        corrupted, _mutations = corrupt_program(
            clean_source,
            num_steps=num_steps,
            bank=bank,
            max_edit_stmts=config.max_edit_stmts,
            rng=rng,
        )

    # Step 2: Compute TreeDiff path from corrupted to clean.
    path = find_path(
        corrupted,
        clean_source,
        max_edit_stmts=config.max_edit_stmts,
    )

    if not path:
        return None

    # Step 3: Pick a random step along the path.
    step_idx = rng.randrange(len(path))

    # Apply all mutations before step_idx to get the intermediate program.
    intermediate = corrupted
    for i in range(step_idx):
        intermediate = path[i].apply(intermediate)

    # The training target is path[step_idx]: the next edit to apply.
    target_mutation = path[step_idx]

    # Step 4: Encode as training example.
    # The edit position is the character offset in the intermediate program,
    # which maps to a token index (1:1 for byte-level tokenizer).
    edit_token_idx = tokenizer.char_offset_to_token_index(
        intermediate, target_mutation.start
    )

    token_ids, loss_mask = tokenizer.encode_training_example(
        context_source=intermediate,
        edit_position_token_idx=edit_token_idx,
        replacement_source=target_mutation.replacement,
    )

    # Truncate or skip if too long.
    if len(token_ids) > max_seq_len:
        return None

    return token_ids, loss_mask


def create_edit_data_iter(
    corpus: list[str],
    bank: SubtreeBank,
    tokenizer: TreeDiffusionTokenizer,
    config: EditTrainingConfig,
    seed: int = 42,
) -> Iterator[dict[str, Array]]:
    """Create a training data iterator for tree diffusion.

    Yields batches of (token_ids, loss_mask) arrays.
    """
    rng = pyrandom.Random(seed)

    while True:
        batch_token_ids = []
        batch_loss_masks = []

        while len(batch_token_ids) < config.batch_size:
            clean_source = rng.choice(corpus)
            example = generate_training_example(
                clean_source=clean_source,
                corpus=corpus,
                bank=bank,
                tokenizer=tokenizer,
                max_seq_len=config.max_seq_len,
                config=config,
                rng=rng,
            )
            if example is None:
                continue

            token_ids, loss_mask = example
            batch_token_ids.append(token_ids)
            batch_loss_masks.append(loss_mask)

        # Pad to max_seq_len.
        padded_ids = jnp.zeros((config.batch_size, config.max_seq_len), dtype=jnp.int32)
        padded_masks = jnp.zeros((config.batch_size, config.max_seq_len), dtype=jnp.float32)

        for i in range(config.batch_size):
            seq_len = len(batch_token_ids[i])
            padded_ids = padded_ids.at[i, :seq_len].set(jnp.array(batch_token_ids[i]))
            padded_masks = padded_masks.at[i, :seq_len].set(jnp.array(batch_loss_masks[i]))

        yield {"token_ids": padded_ids, "loss_mask": padded_masks}


LogCallback = Callable[[int, dict], None]


def train_edit_model(
    config: EditTrainingConfig,
    data_iter: Iterator[dict[str, Array]],
    initial_params: EditModelParams | None = None,
    log_callback: LogCallback | None = None,
) -> EditModelParams:
    """Train a tree diffusion edit model.

    Args:
        config: Training configuration.
        data_iter: Iterator yielding batches with 'token_ids' and 'loss_mask'.
        initial_params: Optional initial parameters (for transfer learning).
        log_callback: Optional callback for logging metrics.

    Returns:
        Trained EditModelParams.
    """
    key = jax.random.PRNGKey(config.seed)
    optimizer = create_edit_optimizer(config)

    # Initialize W&B if configured.
    wandb_run = None
    if config.wandb_project is not None:
        try:
            import wandb
            from dataclasses import asdict

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )
            logger.info(f"W&B logging enabled: {wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging")

    if initial_params is None:
        key, init_key = jax.random.split(key)
        params = init_edit_params(config.model, key=init_key)
    else:
        params = initial_params

    opt_state = optimizer.init(params)
    state = EditTrainingState(step=0, params=params, opt_state=opt_state, key=key)

    train_step = make_edit_train_step(config.model, optimizer)

    logger.info(f"Starting edit model training for {config.total_steps} steps")

    for step in range(config.total_steps):
        batch = next(data_iter)
        state, metrics = train_step(state, batch)

        if step % config.log_interval == 0:
            _log_edit_metrics(step, metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {k: float(v) for k, v in metrics.items()},
                    step=step,
                )
            if log_callback is not None:
                log_callback(step, metrics)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete")
    return state.params


def _log_edit_metrics(step: int, metrics: dict) -> None:
    """Log training metrics."""
    loss = float(metrics["loss"])
    acc = float(metrics["accuracy"])
    ppl = float(metrics["perplexity"])
    grad_norm = float(metrics["grad_norm"])
    num_tokens = float(metrics["num_loss_tokens"])

    logger.info(
        f"step={step:06d} loss={loss:.4f} acc={acc:.4f} "
        f"ppl={ppl:.2f} grad_norm={grad_norm:.4f} loss_tokens={num_tokens:.0f}"
    )
