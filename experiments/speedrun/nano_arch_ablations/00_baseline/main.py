# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Baseline Transformer.
Includes embedding, layers of (attn w/RopE, mlp) with rms prenorm, and output proj.
Muon used for training
"""

# nodryrun
from __future__ import annotations

from dataclasses import dataclass
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
from einops import rearrange
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from levanter.callbacks.profiler import ProfilerConfig
from haliax.jax_utils import named_call
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pvocab, unshard
from levanter.optim import GrugMuonConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author

from ..minimal_grug_wrapper import build_speedrun

# -----------------------------------------------------------------------------
# Model


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _rotary_cache(seq_len: int, head_dim: int, rope_theta: float) -> tuple[Float[Array, "S D"], Float[Array, "S D"]]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


@named_call
def apply_rotary_embedding(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    cos, sin = _rotary_cache(seq_len, head_dim, rope_theta)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    def _apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return _apply(q), _apply(k)


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    cfg: ModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        q_key, k_key, v_key, o_key = random.split(key, 4)
        D, N, M, H = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        w_q = reshard(_init_weight(q_key, (D, N * H), cfg.initializer_std), P("data", "model"))
        w_k = reshard(_init_weight(k_key, (D, M * H), cfg.initializer_std), P("data", "model"))
        w_v = reshard(_init_weight(v_key, (D, M * H), cfg.initializer_std), P("data", "model"))
        w_o = reshard(_init_weight(o_key, (N * H, D), cfg.initializer_std), P("model", "data"))
        return CausalSelfAttention(w_q, w_k, w_v, w_o, cfg)

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.head_dim
        seq_len = x.shape[1]
        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope_theta=self.cfg.rope_theta)
        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)
        return attn_out


class MLP(eqx.Module):
    mlp_up: jax.Array
    mlp_down: jax.Array

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        up_key, down_key = random.split(key, 2)
        D, I = cfg.hidden_dim, cfg.intermediate_dim
        mlp_up = reshard(_init_weight(up_key, (D, I), cfg.initializer_std), P("data", "model"))
        mlp_down = reshard(_init_weight(down_key, (I, D), cfg.initializer_std), P("model", "data"))
        return MLP(mlp_up, mlp_down)

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        up = jnp.einsum("bsh,hm->bsm", x, self.mlp_up)
        activated = jax.nn.relu(up)
        return jnp.einsum("bsm,mh->bsh", activated, self.mlp_down, out_sharding=Pbatch)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float):
        weight = jnp.ones((dim,), dtype=jnp.float32)
        return RMSNorm(weight, eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        out = normed * weight
        return out.astype(dtype)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MLP

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        attn_key, mlp_key = random.split(key, 2)
        rms_attn = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        attn = CausalSelfAttention.init(cfg, key=attn_key)
        rms_mlp = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        mlp = MLP.init(cfg, key=mlp_key)
        return Block(rms_attn, attn, rms_mlp, mlp)

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.rms_attn(x), mask)
        x = x + self.mlp(self.rms_mlp(x))
        return x


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: ModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        embed_key, out_key, block_key = random.split(key, 3)
        token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
        blocks = tuple(Block.init(cfg, key=block_key) for _ in range(cfg.num_layers))
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return Transformer(token_embed, output_proj, blocks, final_norm, cfg)

    @named_call
    def __call__(self, token_ids: Int[Array, "B S"], mask: AttentionMask | jax.Array | None) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        x = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        for block in self.blocks:
            x = eqx.filter_checkpoint(block)(x, mask)
        x = self.final_norm(x)
        return x


def loss_fn(
    transformer: Transformer,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    cfg: ModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    loss_dtype: jnp.dtype = jnp.float32,
):
    hidden = transformer(token_ids, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    loss_weight = loss_weight.astype(loss_dtype)

    return fused_linear_softmax_cross_entropy_loss(
        hidden,
        transformer.output_proj,
        labels,
        weight=loss_weight,
        reduction=reduction,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
    )


# -----------------------------------------------------------------------------
# Configs


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters available to the model"""

    vocab_size: int = llama3_tokenizer_vocab_size
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    max_seq_len: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope_theta: float = 10_000

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def intermediate_dim(self) -> int:
        return int(self.hidden_dim * 4)

    @property
    def total_trainable_params(self) -> int:
        token_embedding = self.vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * self.head_dim * self.num_heads
            + 2 * self.hidden_dim * self.head_dim * self.num_kv_heads
            + self.head_dim * self.num_heads * self.hidden_dim
        )
        mlp = 2 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        return int(transformer + 2 * token_embedding)

    @property
    def flops_per_token(self) -> float:
        mlp = 2 * 2 * self.hidden_dim * self.intermediate_dim
        qkv_proj = 2 * self.hidden_dim * (self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim)
        dense_proj = 2 * self.hidden_dim * self.hidden_dim
        # The following are across the whole sequence
        # assume full attention map like megatron-lm
        key_query_logits = 2 * self.max_seq_len**2 * self.num_heads * self.head_dim
        mask = 3 * self.max_seq_len**2 * self.num_heads
        mask_value = 2 * self.max_seq_len**2 * self.head_dim * self.num_heads
        seq_flops = key_query_logits + mask + mask_value
        # so we divide by the sequence length to get the per-token flops
        attn = seq_flops / self.max_seq_len
        lm_head = 2 * self.hidden_dim * self.vocab_size
        return self.num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head


def build_train_config(model_cfg: ModelConfig) -> SimpleTrainConfig:
    batch_size = 128
    num_train_steps = 1000

    muon = GrugMuonConfig(
        learning_rate=0.02,
        adam_lr=0.0064,
        weight_decay=0,
        min_lr_ratio=0.1,
        warmup=0,
        momentum=0.95,
        beta1=0.8,
        beta2=0.95,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1,
        lr_schedule="linear",
        decay=0.5,
    )

    train_cfg = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v4-8"),
        train_batch_size=batch_size,
        learning_rate=muon.learning_rate,
        explicit_mesh_axes=True,
        profiler=ProfilerConfig(enabled=True),
        train_seq_len=model_cfg.max_seq_len,
        num_train_steps=num_train_steps,
        steps_per_hf_export=-1,
        optimizer_config=muon,
    )
    return train_cfg


# -----------------------------------------------------------------------------
# Misc


def repoint_modules_for_ray(classes_in_main):
    # ensure naming compatibility if job is called from Ray workers
    import_path = getattr(__spec__, "name", __name__)
    sys.modules[import_path] = sys.modules[__name__]
    for _cls in classes_in_main:
        _cls.__module__ = import_path


# -----------------------------------------------------------------------------
# Main


def main() -> None:
    module_classes = [Transformer, ModelConfig, Block, RMSNorm, MLP, CausalSelfAttention]
    repoint_modules_for_ray(module_classes)

    model_cfg = ModelConfig()
    train_cfg = build_train_config(model_cfg)
    speedrun = build_speedrun(
        model_cls=Transformer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        loss_fn=loss_fn,
        speedrun_name="nano_ablations_00_baseline",
        speedrun_desc="Baseline Transformer with Muon",
        author=Author(
            name="Larry Dial",
            affiliation="OpenAthena",
            url="https://github.com/ClassicLarry",
        ),
    )
    steps = []
    steps.extend(speedrun)
    executor_main(steps=steps, description="Single Nano Run")


if __name__ == "__main__":
    main()
