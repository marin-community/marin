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

"""Tree diffusion model implementation.

A transformer-based model for discrete diffusion over token sequences.
Key differences from AR models:
1. Bidirectional attention (no causal mask)
2. Timestep conditioning
3. Predicts denoised tokens at each position
"""

import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.noise import (
    NoiseSchedule,
    corrupt_tokens,
    create_initial_tokens,
    get_schedule,
    sample_iterative,
    sample_timesteps,
)

logger = logging.getLogger(__name__)


@register_dataclass
@dataclass(frozen=True)
class TreeDiffusionAttentionParams:
    """Parameters for a single attention layer."""

    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array


@register_dataclass
@dataclass(frozen=True)
class TreeDiffusionBlockParams:
    """Parameters for a transformer block."""

    attn: TreeDiffusionAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array
    mlp_up: jax.Array
    mlp_down: jax.Array


@register_dataclass
@dataclass(frozen=True)
class TreeDiffusionModelParams:
    """All parameters for a tree diffusion model."""

    token_embed: jax.Array
    timestep_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[TreeDiffusionBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    """Initialize weights with truncated normal."""
    return std * random.truncated_normal(key, -3, 3, shape)


def init_parameters(cfg: TreeDiffusionConfig, *, key: PRNGKeyArray) -> TreeDiffusionModelParams:
    """Initialize model parameters.

    Args:
        cfg: Model configuration.
        key: PRNG key.

    Returns:
        Initialized TreeDiffusionModelParams.
    """
    head_dim = cfg.inferred_head_dim
    key, embed_key, time_key, out_key = random.split(key, 4)
    layer_keys = random.split(key, cfg.num_layers)

    token_embed = _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std)
    timestep_embed = _init_weight(time_key, (cfg.num_diffusion_steps, cfg.hidden_dim), cfg.initializer_std)
    output_proj = _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std)
    final_norm = jnp.ones((cfg.hidden_dim,), dtype=jnp.float32)

    blocks: list[TreeDiffusionBlockParams] = []
    D, N, M, H, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim

    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_gate, k_up, k_down = random.split(layer_keys[i], 7)

        attn = TreeDiffusionAttentionParams(
            w_q=_init_weight(k_q, (D, N * H), cfg.initializer_std),
            w_k=_init_weight(k_k, (D, M * H), cfg.initializer_std),
            w_v=_init_weight(k_v, (D, M * H), cfg.initializer_std),
            w_o=_init_weight(k_o, (N * H, D), cfg.initializer_std),
        )

        blocks.append(
            TreeDiffusionBlockParams(
                attn=attn,
                rms_attn=jnp.ones((D,), dtype=jnp.float32),
                rms_mlp=jnp.ones((D,), dtype=jnp.float32),
                mlp_gate=_init_weight(k_gate, (D, I), cfg.initializer_std),
                mlp_up=_init_weight(k_up, (D, I), cfg.initializer_std),
                mlp_down=_init_weight(k_down, (I, D), cfg.initializer_std),
            )
        )

    return TreeDiffusionModelParams(
        token_embed=token_embed,
        timestep_embed=timestep_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    """RMS normalization."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    out = normed * weight
    return out.astype(dtype)


def apply_rotary_embedding(
    q: Float[Array, "B S N H"],
    k: Float[Array, "B S M H"],
    seq_len: int,
    head_dim: int,
    rope_base: float = 10000.0,
) -> tuple[Float[Array, "B S N H"], Float[Array, "B S M H"]]:
    """Apply rotary positional embeddings to Q and K.

    Uses the "interleaved" rotary embedding formulation.
    """
    positions = jnp.arange(seq_len)
    half_dim = head_dim // 2
    dim_indices = jnp.arange(half_dim)
    inv_freq = 1.0 / (rope_base ** (dim_indices * 2.0 / head_dim))
    sinusoid = positions[:, None] * inv_freq[None, :]

    sin = jnp.sin(sinusoid)
    cos = jnp.cos(sinusoid)

    sin = jnp.repeat(sin, 2, axis=-1)
    cos = jnp.repeat(cos, 2, axis=-1)

    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = jnp.stack([-x2, x1], axis=-1)
        return rearrange(x_rotated, "... d two -> ... (d two)")

    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


def attention(
    q: Float[Array, "B S N H"],
    k: Float[Array, "B S M H"],
    v: Float[Array, "B S M H"],
) -> Float[Array, "B S N H"]:
    """Bidirectional scaled dot-product attention (no causal mask)."""
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[3]

    if num_heads != num_kv_heads:
        repeats = num_heads // num_kv_heads
        k = jnp.repeat(k, repeats, axis=2)
        v = jnp.repeat(v, repeats, axis=2)

    scale = 1.0 / jnp.sqrt(head_dim)
    scores = jnp.einsum("bsnh,btnh->bstn", q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bstn,btnh->bsnh", weights, v)
    return out


def mlp(block: TreeDiffusionBlockParams, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
    """SwiGLU MLP."""
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down)


def forward(
    params: TreeDiffusionModelParams,
    token_ids: Int[Array, "B S"],
    timesteps: Int[Array, "B"],
    cfg: TreeDiffusionConfig,
) -> Float[Array, "B S V"]:
    """Forward pass of the tree diffusion model.

    Args:
        params: Model parameters.
        token_ids: Input token IDs (may contain mask tokens).
        timesteps: Diffusion timesteps per batch element.
        cfg: Model configuration.

    Returns:
        Logits of shape (batch, seq, vocab).
    """
    head_dim = cfg.inferred_head_dim
    batch_size, seq_len = token_ids.shape

    hidden = params.token_embed[token_ids]

    time_emb = params.timestep_embed[timesteps]
    hidden = hidden + time_emb[:, None, :]

    for block in params.blocks:
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)

        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "b s (n d) -> b s n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "b s (m d) -> b s m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "b s (m d) -> b s m d", d=head_dim)

        if cfg.use_rope:
            q, k = apply_rotary_embedding(q, k, seq_len, head_dim, cfg.rope_base)

        attn_out = attention(q, k, v)
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o)

        hidden = hidden + attn_out

        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = mlp(block, mlp_in)
        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    logits = jnp.einsum("bsh,hv->bsv", hidden, params.output_proj)
    return logits


def loss_fn(
    params: TreeDiffusionModelParams,
    token_ids: Int[Array, "B S"],
    timesteps: Int[Array, "B"],
    corrupted_ids: Int[Array, "B S"],
    cfg: TreeDiffusionConfig,
    *,
    prefix_len: int | None = None,
) -> Float[Array, ""]:
    """Compute tree diffusion loss.

    The loss is cross-entropy between predicted logits and clean tokens,
    computed only at masked positions (optionally excluding prefix).

    Args:
        params: Model parameters.
        token_ids: Clean target token IDs.
        timesteps: Diffusion timesteps.
        corrupted_ids: Corrupted input token IDs.
        cfg: Model configuration.
        prefix_len: Length of prefix to exclude from loss.

    Returns:
        Scalar loss value.
    """
    logits = forward(params, corrupted_ids, timesteps, cfg)

    is_masked = corrupted_ids == cfg.effective_mask_token_id
    loss_weight = is_masked.astype(jnp.float32)

    if prefix_len is not None:
        seq_len = token_ids.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        prefix_mask = positions < prefix_len
        loss_weight = loss_weight * (1 - prefix_mask.astype(jnp.float32))

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1).squeeze(-1)

    masked_loss = -target_log_probs * loss_weight
    num_masked = jnp.sum(loss_weight)
    loss = jnp.sum(masked_loss) / jnp.maximum(num_masked, 1.0)

    return loss


class TreeDiffusionModel:
    """High-level interface for tree diffusion model."""

    def __init__(
        self,
        params: TreeDiffusionModelParams,
        config: TreeDiffusionConfig,
        schedule: NoiseSchedule | None = None,
    ):
        self.params = params
        self.config = config
        self.schedule = schedule or get_schedule(config.noise_schedule, config.num_diffusion_steps)

    @classmethod
    def init(cls, config: TreeDiffusionConfig, key: PRNGKeyArray) -> "TreeDiffusionModel":
        """Initialize a new model."""
        params = init_parameters(config, key=key)
        return cls(params, config)

    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        timesteps: Int[Array, "B"],
    ) -> Float[Array, "B S V"]:
        """Forward pass."""
        return forward(self.params, token_ids, timesteps, self.config)

    def sample(
        self,
        prefix: str | Int[Array, "P"],
        max_iterations: int | None = None,
        temperature: float = 1.0,
        use_grammar_constraints: bool = False,
        key: PRNGKeyArray | None = None,
        tokenizer=None,
        target_len: int | None = None,
    ) -> str:
        """Generate code from a prefix.

        Args:
            prefix: Prefix string or token IDs.
            max_iterations: Max diffusion steps.
            temperature: Sampling temperature.
            use_grammar_constraints: Whether to apply grammar masking.
            key: PRNG key.
            tokenizer: Tokenizer for string conversion.
            target_len: Target sequence length.

        Returns:
            Generated code string.
        """
        if key is None:
            key = random.PRNGKey(0)

        if isinstance(prefix, str):
            if tokenizer is None:
                raise ValueError("tokenizer required for string prefix")
            prefix_tokens = jnp.array(tokenizer.encode(prefix))[None, :]
        else:
            prefix_tokens = prefix[None, :] if prefix.ndim == 1 else prefix

        prefix_len = prefix_tokens.shape[1]
        total_len = target_len or self.config.max_seq_len

        initial_tokens = create_initial_tokens(
            prefix_tokens,
            total_len,
            self.config.effective_mask_token_id,
        )

        def model_fn(tokens, timesteps):
            return forward(self.params, tokens, timesteps, self.config)

        final_tokens = sample_iterative(
            model_fn=model_fn,
            initial_tokens=initial_tokens,
            schedule=self.schedule,
            mask_token_id=self.config.effective_mask_token_id,
            key=key,
            num_steps=max_iterations,
            temperature=temperature,
            prefix_len=prefix_len,
        )

        if tokenizer is not None:
            return tokenizer.decode(final_tokens[0].tolist())
        return final_tokens[0].tolist()


def load_model(path: str) -> TreeDiffusionModel:
    """Load a model from a checkpoint path.

    Args:
        path: Path to checkpoint directory.

    Returns:
        Loaded TreeDiffusionModel.
    """
    import json
    import pickle

    import fsspec

    fs = fsspec.filesystem(path.split(":")[0] if "://" in path else "file")

    config_path = f"{path}/config.json"
    with fs.open(config_path, "r") as f:
        config_dict = json.load(f)
    config = TreeDiffusionConfig(**config_dict)

    params_path = f"{path}/params.pkl"
    with fs.open(params_path, "rb") as f:
        params = pickle.load(f)

    return TreeDiffusionModel(params, config)


def save_model(model: TreeDiffusionModel, path: str) -> None:
    """Save a model to a checkpoint path.

    Args:
        model: Model to save.
        path: Path to checkpoint directory.
    """
    import json
    import pickle
    from dataclasses import asdict

    import fsspec

    fs = fsspec.filesystem(path.split(":")[0] if "://" in path else "file")
    fs.makedirs(path, exist_ok=True)

    config_path = f"{path}/config.json"
    with fs.open(config_path, "w") as f:
        json.dump(asdict(model.config), f, indent=2)

    params_path = f"{path}/params.pkl"
    with fs.open(params_path, "wb") as f:
        pickle.dump(model.params, f)

    logger.info(f"Saved model to {path}")
