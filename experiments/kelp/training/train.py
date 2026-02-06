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

"""Training loop for Kelp tree diffusion models.

Follows the grugformer pattern of simple, explicit training code.
"""

import logging
from dataclasses import dataclass
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import register_dataclass
from jaxtyping import Array, PRNGKeyArray

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import (
    TreeDiffusionAttentionParams,
    TreeDiffusionBlockParams,
    TreeDiffusionModel,
    TreeDiffusionModelParams,
    init_parameters,
    save_model,
)
from experiments.kelp.model.noise import NoiseSchedule, get_schedule
from experiments.kelp.training.loss import tree_diffusion_loss_with_metrics

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class TrainingState:
    """State for tree diffusion training."""

    step: int
    params: TreeDiffusionModelParams
    opt_state: optax.OptState
    key: jax.Array


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for tree diffusion training."""

    model: TreeDiffusionConfig
    """Model configuration."""

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

    output_dir: str = "checkpoints/kelp"
    """Output directory for checkpoints."""

    wandb_project: str | None = None
    """W&B project name. If set, enables W&B logging."""

    wandb_run_name: str | None = None
    """W&B run name. If None, W&B auto-generates one."""


def _weight_decay_mask(params: TreeDiffusionModelParams) -> TreeDiffusionModelParams:
    """Create a mask that excludes norms and embeddings from weight decay.

    Returns a pytree of the same structure as params where True means
    "apply weight decay" and False means "skip weight decay".
    """
    masked_blocks = tuple(
        TreeDiffusionBlockParams(
            attn=TreeDiffusionAttentionParams(
                w_q=True,
                w_k=True,
                w_v=True,
                w_o=True,
            ),
            rms_attn=False,  # norm weight — no decay
            rms_mlp=False,  # norm weight — no decay
            mlp_gate=True,
            mlp_up=True,
            mlp_down=True,
        )
        for _ in params.blocks
    )
    return TreeDiffusionModelParams(
        token_embed=False,  # embedding — no decay
        timestep_embed=False,  # embedding — no decay
        output_proj=True,
        blocks=masked_blocks,
        final_norm=False,  # norm weight — no decay
    )


def create_optimizer(config: TrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup, cosine decay, and weight decay masking.

    Weight decay is not applied to normalization weights or embeddings,
    following standard practice for transformer training.
    """
    warmup_steps = min(config.warmup_steps, max(config.total_steps // 2, 1))
    decay_steps = max(config.total_steps, warmup_steps + 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=config.learning_rate * 0.1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
            mask=_weight_decay_mask,
        ),
    )

    return optimizer


def make_train_step(
    config: TreeDiffusionConfig,
    schedule: NoiseSchedule,
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled training step function.

    Args:
        config: Model configuration.
        schedule: Noise schedule.
        optimizer: Optimizer.

    Returns:
        Training step function.
    """

    def loss_fn(params, tokens, prefix_len, key):
        return tree_diffusion_loss_with_metrics(
            params=params,
            tokens=tokens,
            prefix_len=prefix_len,
            schedule=schedule,
            config=config,
            key=key,
        )

    def train_step(state: TrainingState, batch: dict[str, Array]) -> tuple[TrainingState, dict]:
        key, step_key = jax.random.split(state.key)

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            batch["tokens"],
            batch["prefix_len"],
            step_key,
        )

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm

        new_state = TrainingState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            key=key,
        )

        return new_state, metrics

    return jax.jit(train_step)


LogCallback = Callable[[int, dict], None]


def train(
    config: TrainingConfig,
    data_iter: Iterator[dict[str, Array]],
    initial_params: TreeDiffusionModelParams | None = None,
    log_callback: LogCallback | None = None,
) -> TreeDiffusionModel:
    """Train a tree diffusion model.

    Args:
        config: Training configuration.
        data_iter: Iterator yielding batches with 'tokens' and 'prefix_len'.
        initial_params: Optional initial parameters (for transfer learning).
        log_callback: Optional callback for logging metrics (e.g., to wandb).
            Called with (step, metrics_dict) at each log_interval.

    Returns:
        Trained TreeDiffusionModel.
    """
    key = jax.random.PRNGKey(config.seed)
    schedule = get_schedule(config.model.noise_schedule, config.model.num_diffusion_steps)
    optimizer = create_optimizer(config)

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
            logger.warning("wandb_project set but wandb not installed; skipping W&B logging")

    if initial_params is None:
        key, init_key = jax.random.split(key)
        params = init_parameters(config.model, key=init_key)
    else:
        params = initial_params

    opt_state = optimizer.init(params)
    state = TrainingState(step=0, params=params, opt_state=opt_state, key=key)

    train_step = make_train_step(config.model, schedule, optimizer)

    logger.info(f"Starting training for {config.total_steps} steps")

    for step in range(config.total_steps):
        batch = next(data_iter)
        state, metrics = train_step(state, batch)

        if step % config.log_interval == 0:
            log_metrics(step, metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {k: float(v) for k, v in metrics.items()},
                    step=step,
                )
            if log_callback is not None:
                log_callback(step, metrics)

        if step > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(state, config, schedule)

    save_checkpoint(state, config, schedule, final=True)

    if wandb_run is not None:
        wandb_run.finish()

    return TreeDiffusionModel(state.params, config.model, schedule)


def log_metrics(step: int, metrics: dict) -> None:
    """Log training metrics."""
    loss = float(metrics["loss"])
    acc = float(metrics["accuracy"])
    ppl = float(metrics["perplexity"])
    grad_norm = float(metrics["grad_norm"])
    mask_ratio = float(metrics["mask_ratio"])

    logger.info(
        f"step={step:06d} loss={loss:.4f} acc={acc:.4f} "
        f"ppl={ppl:.2f} grad_norm={grad_norm:.4f} mask_ratio={mask_ratio:.3f}"
    )


def save_checkpoint(
    state: TrainingState,
    config: TrainingConfig,
    schedule: NoiseSchedule,
    final: bool = False,
) -> None:
    """Save a checkpoint."""
    checkpoint_dir = f"{config.output_dir}/step-{state.step:06d}"
    if final:
        checkpoint_dir = f"{config.output_dir}/final"

    model = TreeDiffusionModel(state.params, config.model, schedule)
    save_model(model, checkpoint_dir, step=int(state.step))
    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def create_synthetic_data_iter(
    config: TrainingConfig,
    key: PRNGKeyArray,
) -> Iterator[dict[str, Array]]:
    """Create a synthetic data iterator for testing.

    Args:
        config: Training configuration.
        key: PRNG key.

    Yields:
        Batches with random tokens.
    """
    while True:
        key, data_key = jax.random.split(key)
        tokens = jax.random.randint(
            data_key,
            (config.batch_size, config.model.max_seq_len),
            0,
            config.model.vocab_size - 1,  # Reserve last token for [MASK]
        )
        prefix_len = jnp.full((config.batch_size,), config.model.prefix_max_len)

        yield {"tokens": tokens, "prefix_len": prefix_len}
