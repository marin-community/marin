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

"""Noise schedules and corruption/denoising for tree diffusion.

Implements the diffusion process over discrete token sequences, following
approaches like D3PM (Austin et al., 2021) and DiffuSeq (Gong et al., 2022).
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray

logger = logging.getLogger(__name__)


@dataclass
class NoiseSchedule:
    """A noise schedule for diffusion."""

    name: str
    """Schedule name."""

    num_steps: int
    """Number of diffusion steps."""

    alphas: Float[Array, "T"]
    """Alpha values (noise level at each step)."""

    alphas_cumprod: Float[Array, "T"]
    """Cumulative product of alphas."""

    betas: Float[Array, "T"]
    """Beta values (1 - alpha)."""


def cosine_schedule(num_steps: int, s: float = 0.008) -> NoiseSchedule:
    """Create a cosine noise schedule.

    Following Nichol & Dhariwal (2021), the cosine schedule provides
    more uniform noise across diffusion steps.

    Args:
        num_steps: Number of diffusion steps.
        s: Small offset to prevent singularity at t=0.

    Returns:
        NoiseSchedule with cosine-based values.
    """
    steps = jnp.arange(num_steps + 1, dtype=jnp.float32)
    f_t = jnp.cos((steps / num_steps + s) / (1 + s) * jnp.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0.0001, 0.9999)
    alphas = 1 - betas
    alphas_cumprod = alphas_cumprod[1:]

    return NoiseSchedule(
        name="cosine",
        num_steps=num_steps,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        betas=betas,
    )


def linear_schedule(num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> NoiseSchedule:
    """Create a linear noise schedule.

    Args:
        num_steps: Number of diffusion steps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        NoiseSchedule with linear beta values.
    """
    betas = jnp.linspace(beta_start, beta_end, num_steps)
    alphas = 1 - betas
    alphas_cumprod = jnp.cumprod(alphas)

    return NoiseSchedule(
        name="linear",
        num_steps=num_steps,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        betas=betas,
    )


def get_schedule(name: str, num_steps: int) -> NoiseSchedule:
    """Get a noise schedule by name.

    Args:
        name: Schedule name ('cosine' or 'linear').
        num_steps: Number of diffusion steps.

    Returns:
        NoiseSchedule instance.
    """
    if name == "cosine":
        return cosine_schedule(num_steps)
    elif name == "linear":
        return linear_schedule(num_steps)
    else:
        raise ValueError(f"Unknown schedule: {name}")


def corrupt_tokens(
    tokens: Int[Array, "B S"],
    timestep: Int[Array, "B"] | int,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
    prefix_len: Int[Array, "B"] | int | None = None,
) -> Int[Array, "B S"]:
    """Corrupt tokens by replacing some with [MASK] based on timestep.

    For discrete diffusion, we randomly mask tokens with probability
    determined by the noise schedule. Higher timesteps = more masking.

    Args:
        tokens: Input token IDs of shape (batch, seq_len).
        timestep: Diffusion timestep(s), 0 = clean, num_steps = fully noised.
        schedule: Noise schedule.
        mask_token_id: Token ID to use for masking.
        key: PRNG key.
        prefix_len: Length of prefix to keep unmasked (per-batch or single value).

    Returns:
        Corrupted tokens with some replaced by mask_token_id.
    """
    batch_size, seq_len = tokens.shape

    if isinstance(timestep, int):
        timestep = jnp.full((batch_size,), timestep)

    mask_prob = 1 - schedule.alphas_cumprod[timestep]
    mask_prob = mask_prob[:, None]

    mask = random.uniform(key, tokens.shape) < mask_prob

    if prefix_len is not None:
        if isinstance(prefix_len, int):
            prefix_len = jnp.full((batch_size,), prefix_len)
        positions = jnp.arange(seq_len)[None, :]
        prefix_mask = positions < prefix_len[:, None]
        mask = mask & ~prefix_mask

    corrupted = jnp.where(mask, mask_token_id, tokens)
    return corrupted


def sample_timesteps(
    batch_size: int,
    num_steps: int,
    key: PRNGKeyArray,
) -> Int[Array, "B"]:
    """Sample random timesteps for training.

    Args:
        batch_size: Batch size.
        num_steps: Number of diffusion steps.
        key: PRNG key.

    Returns:
        Random timesteps of shape (batch_size,).
    """
    return random.randint(key, (batch_size,), 0, num_steps)


@dataclass
class DiffusionState:
    """State during diffusion sampling."""

    tokens: Int[Array, "B S"]
    """Current token sequence."""

    timestep: int
    """Current timestep (counting down from num_steps to 0)."""


def sample_step(
    model_fn,
    state: DiffusionState,
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
    temperature: float = 1.0,
    prefix_len: int | None = None,
) -> DiffusionState:
    """Perform one step of diffusion sampling (denoising).

    Args:
        model_fn: Function that takes tokens and timestep, returns logits.
        state: Current diffusion state.
        schedule: Noise schedule.
        mask_token_id: Mask token ID.
        key: PRNG key.
        temperature: Sampling temperature.
        prefix_len: Length of prefix to keep unchanged.

    Returns:
        Updated diffusion state with less noise.
    """
    batch_size, seq_len = state.tokens.shape

    timestep_array = jnp.full((batch_size,), state.timestep)
    logits = model_fn(state.tokens, timestep_array)

    if temperature != 1.0:
        logits = logits / temperature

    k1, k2 = random.split(key)
    pred_tokens = random.categorical(k1, logits)

    is_masked = state.tokens == mask_token_id

    if state.timestep > 0:
        next_timestep = state.timestep - 1
        next_mask_prob = 1 - schedule.alphas_cumprod[next_timestep]
        should_remask = random.uniform(k2, state.tokens.shape) < next_mask_prob
        new_tokens = jnp.where(is_masked & ~should_remask, pred_tokens, state.tokens)
        new_tokens = jnp.where(is_masked & should_remask, mask_token_id, new_tokens)
    else:
        new_tokens = jnp.where(is_masked, pred_tokens, state.tokens)

    if prefix_len is not None:
        positions = jnp.arange(seq_len)[None, :]
        prefix_mask = positions < prefix_len
        new_tokens = jnp.where(prefix_mask, state.tokens, new_tokens)

    return DiffusionState(tokens=new_tokens, timestep=state.timestep - 1)


def sample_iterative(
    model_fn,
    initial_tokens: Int[Array, "B S"],
    schedule: NoiseSchedule,
    mask_token_id: int,
    key: PRNGKeyArray,
    num_steps: int | None = None,
    temperature: float = 1.0,
    prefix_len: int | None = None,
) -> Int[Array, "B S"]:
    """Run full diffusion sampling from noise to clean tokens.

    Args:
        model_fn: Function that takes tokens and timestep, returns logits.
        initial_tokens: Starting tokens (typically mostly masked except prefix).
        schedule: Noise schedule.
        mask_token_id: Mask token ID.
        key: PRNG key.
        num_steps: Number of steps (defaults to schedule.num_steps).
        temperature: Sampling temperature.
        prefix_len: Length of prefix to keep unchanged.

    Returns:
        Denoised token sequence.
    """
    if num_steps is None:
        num_steps = schedule.num_steps

    state = DiffusionState(tokens=initial_tokens, timestep=num_steps - 1)

    for t in range(num_steps - 1, -1, -1):
        key, step_key = random.split(key)
        state = sample_step(
            model_fn=model_fn,
            state=DiffusionState(tokens=state.tokens, timestep=t),
            schedule=schedule,
            mask_token_id=mask_token_id,
            key=step_key,
            temperature=temperature,
            prefix_len=prefix_len,
        )

    return state.tokens


def create_initial_tokens(
    prefix_tokens: Int[Array, "B P"],
    target_len: int,
    mask_token_id: int,
    pad_token_id: int = 0,
) -> Int[Array, "B S"]:
    """Create initial tokens for sampling: prefix + masks.

    Args:
        prefix_tokens: Prefix tokens (docstring/signature).
        target_len: Total target sequence length.
        mask_token_id: Token ID for masks.
        pad_token_id: Token ID for padding.

    Returns:
        Token sequence with prefix followed by mask tokens.
    """
    batch_size, prefix_len = prefix_tokens.shape

    if prefix_len >= target_len:
        return prefix_tokens[:, :target_len]

    mask_len = target_len - prefix_len
    masks = jnp.full((batch_size, mask_len), mask_token_id)

    return jnp.concatenate([prefix_tokens, masks], axis=1)
