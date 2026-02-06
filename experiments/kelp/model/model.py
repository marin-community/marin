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
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.attention import apply_rotary_embedding as grug_apply_rotary, attention as grug_attention

from experiments.kelp.model.config import TreeDiffusionConfig
from experiments.kelp.model.noise import (
    NoiseSchedule,
    create_initial_tokens,
    get_schedule,
    sample_iterative,
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
    """RMS normalization.

    Kept local rather than delegating to Grug's version because Grug's
    calls unshard(weight) which requires a JAX mesh context.
    """
    dtype = x.dtype
    x = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    out = normed * weight
    return out.astype(dtype)


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
    compute_dtype = jnp.dtype(cfg.compute_dtype)
    head_dim = cfg.inferred_head_dim
    _batch_size, seq_len = token_ids.shape

    hidden = params.token_embed[token_ids].astype(compute_dtype)

    time_emb = params.timestep_embed[timesteps]
    hidden = hidden + time_emb[:, None, :].astype(compute_dtype)

    # Padding mask: prevent padding tokens from attending or being attended to.
    # Shape (B, S) -> (B, S, S) via outer product. Grug's reference_attention
    # expects (B, Q, K) where True = allowed.
    not_pad = token_ids != cfg.pad_token_id
    padding_mask = not_pad[:, :, None] & not_pad[:, None, :]

    def _block_fn(hidden, block):
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)

        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "b s (n d) -> b s n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "b s (m d) -> b s m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "b s (m d) -> b s m d", d=head_dim)

        q, k = grug_apply_rotary(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)

        # Bidirectional attention with padding mask.
        # Passing a dense jax.Array mask uses reference_attention on all backends.
        attn_out = grug_attention(q, k, v, mask=padding_mask)
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o)

        hidden = hidden + attn_out

        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)
        mlp_out = mlp(block, mlp_in)
        hidden = hidden + mlp_out
        return hidden

    block_fn = jax.checkpoint(_block_fn) if cfg.gradient_checkpointing else _block_fn

    for block in params.blocks:
        hidden = block_fn(hidden, block)

    # Cast back to float32 for the final projection to ensure
    # numerically stable logits for loss computation.
    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    logits = jnp.einsum("bsh,hv->bsv", hidden.astype(jnp.float32), params.output_proj)
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
    is_padding = token_ids == cfg.pad_token_id
    loss_weight = is_masked.astype(jnp.float32) * (1 - is_padding.astype(jnp.float32))

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

    Uses Levanter's tensorstore-based checkpoint system.

    Args:
        path: Path to checkpoint directory containing config.json and
              tensorstore parameter data (with metadata.json).

    Returns:
        Loaded TreeDiffusionModel.
    """
    import json

    import fsspec
    from levanter.checkpoint import load_checkpoint

    from levanter.grug.attention import RotaryConfig

    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))

    config_path = f"{path}/config.json"
    with fs.open(config_path, "r") as f:
        config_dict = json.load(f)

    # Migrate legacy configs that had use_rope/rope_base instead of rope.
    if "use_rope" in config_dict or "rope_base" in config_dict:
        rope_base = config_dict.pop("rope_base", 10000.0)
        config_dict.pop("use_rope", None)
        config_dict["rope"] = {"theta": rope_base}

    # Deserialize nested RotaryConfig from dict.
    if isinstance(config_dict.get("rope"), dict):
        config_dict["rope"] = RotaryConfig(**config_dict["rope"])

    config = TreeDiffusionConfig(**config_dict)

    # Create an exemplar pytree with the correct shapes/dtypes.
    # Using eval_shape avoids allocating real arrays for large models.
    # Config must be captured as a closure since it's not a JAX type.
    exemplar = jax.eval_shape(lambda key: init_parameters(config, key=key), random.PRNGKey(0))

    # Levanter's tensorstore deserialization needs a mesh for sharding.
    # Use a single-device mesh for fully-replicated loading.
    mesh = jax.sharding.Mesh(jax.devices(), ("batch",))
    params = load_checkpoint(exemplar, path, discover_latest=False, mesh=mesh)

    return TreeDiffusionModel(params, config)


def save_model(model: TreeDiffusionModel, path: str, *, step: int = 0) -> None:
    """Save a model to a checkpoint path.

    Uses Levanter's tensorstore-based checkpoint system. Saves the config
    as JSON and the parameters via tensorstore with OCDBT format.

    Args:
        model: Model to save.
        path: Path to checkpoint directory.
        step: Training step number for checkpoint metadata.
    """
    import json
    from dataclasses import asdict

    import fsspec
    from levanter.checkpoint import save_checkpoint as levanter_save_checkpoint

    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
    fs.makedirs(path, exist_ok=True)

    config_path = f"{path}/config.json"
    with fs.open(config_path, "w") as f:
        json.dump(asdict(model.config), f, indent=2)

    levanter_save_checkpoint(model.params, step=step, checkpoint_path=path, is_temporary=False)

    logger.info(f"Saved model to {path}")
