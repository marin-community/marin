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

"""Loss functions for tree diffusion training."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.model import TreeDiffusionModelParams, forward
from experiments.kelp.model.noise import NoiseSchedule, corrupt_tokens, sample_timesteps


def tree_diffusion_loss(
    params: TreeDiffusionModelParams,
    tokens: Int[Array, "B S"],
    prefix_len: Int[Array, "B"] | int,
    schedule: NoiseSchedule,
    config: TreeDiffusionConfig,
    key: jax.Array,
) -> Float[Array, ""]:
    """Compute tree diffusion training loss.

    Training procedure:
    1. Sample random timesteps
    2. Corrupt tokens by replacing with [MASK] based on timestep
    3. Predict clean tokens from corrupted input
    4. Compute cross-entropy loss at masked positions

    Args:
        params: Model parameters.
        tokens: Clean token sequences.
        prefix_len: Length of prefix to keep unmasked.
        schedule: Noise schedule.
        config: Model configuration.
        key: PRNG key.

    Returns:
        Scalar loss value.
    """
    batch_size, seq_len = tokens.shape

    key, t_key, corrupt_key = jax.random.split(key, 3)
    timesteps = sample_timesteps(batch_size, schedule.num_steps, t_key)

    corrupted = corrupt_tokens(
        tokens=tokens,
        timestep=timesteps,
        schedule=schedule,
        mask_token_id=config.effective_mask_token_id,
        key=corrupt_key,
        prefix_len=prefix_len,
    )

    logits = forward(params, corrupted, timesteps, config)

    is_masked = corrupted == config.effective_mask_token_id
    is_padding = tokens == config.pad_token_id
    loss_weight = is_masked.astype(jnp.float32) * (1 - is_padding.astype(jnp.float32))

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1).squeeze(-1)

    masked_loss = -target_log_probs * loss_weight
    num_masked = jnp.sum(loss_weight)
    loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)

    return loss


def tree_diffusion_loss_with_metrics(
    params: TreeDiffusionModelParams,
    tokens: Int[Array, "B S"],
    prefix_len: Int[Array, "B"] | int,
    schedule: NoiseSchedule,
    config: TreeDiffusionConfig,
    key: jax.Array,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    """Compute loss with additional metrics.

    Args:
        params: Model parameters.
        tokens: Clean token sequences.
        prefix_len: Length of prefix to keep unmasked.
        schedule: Noise schedule.
        config: Model configuration.
        key: PRNG key.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    batch_size, seq_len = tokens.shape

    key, t_key, corrupt_key = jax.random.split(key, 3)
    timesteps = sample_timesteps(batch_size, schedule.num_steps, t_key)

    corrupted = corrupt_tokens(
        tokens=tokens,
        timestep=timesteps,
        schedule=schedule,
        mask_token_id=config.effective_mask_token_id,
        key=corrupt_key,
        prefix_len=prefix_len,
    )

    logits = forward(params, corrupted, timesteps, config)

    is_masked = corrupted == config.effective_mask_token_id
    is_padding = tokens == config.pad_token_id
    loss_weight = is_masked.astype(jnp.float32) * (1 - is_padding.astype(jnp.float32))
    num_masked = jnp.sum(loss_weight)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1).squeeze(-1)

    masked_loss = -target_log_probs * loss_weight
    loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)

    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == tokens).astype(jnp.float32)
    accuracy = jnp.sum(correct * loss_weight) / jnp.maximum(num_masked, 1.0)

    avg_timestep = jnp.mean(timesteps.astype(jnp.float32))
    mask_ratio = num_masked / (batch_size * seq_len)

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "perplexity": jnp.exp(loss),
        "avg_timestep": avg_timestep,
        "mask_ratio": mask_ratio,
        "num_masked": num_masked,
    }

    return loss, metrics
