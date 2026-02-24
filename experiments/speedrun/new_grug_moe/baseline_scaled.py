# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Scaled-up MoE Transformer (2x layers, 2x hidden_dim, 2x heads vs baseline).
Same architecture as baseline.py with SiGLU MoE experts.
Muon used for training.
"""

# nodryrun
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from jax.experimental.shard_map import shard_map
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

from einops import rearrange
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from haliax.jax_utils import named_call
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pvocab, unshard
import levanter.tracker
from haliax.partitioning import _get_mesh
from levanter.optim import GrugMuonConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author


from .helpers import build_speedrun

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
    cfg: ModelConfig

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
    eps: float

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


_ragged_dim_numbers = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)


def _ragged_moe_linear(
    x: jax.Array,
    w: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """Ragged MoE linear: (TR, In) x (E, In, Out) with groups along TR.

    Shapes:
      - x: [TR, In]
      - w: [E, In, Out]
      - group_sizes: [E] (sum == TR)
      - out: [TR, Out]
    """
    return jax.lax.ragged_dot_general(
        lhs=x,
        rhs=w,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=_ragged_dim_numbers,
    )


class MOE(eqx.Module):
    router_w: jax.Array  # [D, E]
    router_bias: jax.Array
    w1: jax.Array  # [E, D, I] (gate_proj)
    w2: jax.Array  # [E, D, I] (up_proj)
    w3: jax.Array  # [E, I, D] (down_proj)
    cfg: ModelConfig

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        k_router_w, k_w1, k_w2, k_w3 = random.split(key, 4)
        E, D, I = cfg.n_routed_experts, cfg.hidden_dim, cfg.intermediate_dim
        router_w = reshard(_init_weight(k_router_w, (D, E), cfg.initializer_std), P("data", None))
        router_bias = jnp.zeros((E,), dtype=jnp.float32)
        w1 = reshard(_init_weight(k_w1, (E, D, I), cfg.initializer_std), P(None, None, "model"))
        w2 = reshard(_init_weight(k_w2, (E, D, I), cfg.initializer_std), P(None, None, "model"))
        w3 = reshard(_init_weight(k_w3, (E, I, D), cfg.initializer_std), P(None, "model", None))
        return MOE(router_w, router_bias, w1, w2, w3, cfg)

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        B, S, D = x.shape
        K, E = self.cfg.num_experts_per_tok, self.cfg.n_routed_experts
        x_flat = jnp.reshape(x, (B * S, D))
        router_logits = jnp.einsum("td,de->te", x_flat, self.router_w)

        topk_weights, topk_idx, router_probs = self._route(router_logits)

        topk_idx_flat = jnp.reshape(topk_idx, (B * S * K,))  # [B * S, K] -> [B * S * K]

        mesh = _get_mesh()

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                Pbatch,  # x_flat: [T, D]
                Pbatch,  # topk_idx_flat: [T*K]
                Pbatch,  # topk_weights: [T, K]
                P(None, None, "model"),  # w1: [E, D, I]
                P(None, None, "model"),  # w2: [E, D, I]
                P(None, "model", None),  # w3: [E, I, D]
            ),
            out_specs=(Pbatch, P()),
            check_rep=False,
        )
        def _moe_block(x_flat, topk_idx_flat, topk_weights, w1, w2, w3):
            x_repeat_sort, group_sizes, sort_idx = self._permute(x_flat, topk_idx_flat)
            w1_out = _ragged_moe_linear(x_repeat_sort, w1, group_sizes)  # [TR, I]
            w2_out = _ragged_moe_linear(x_repeat_sort, w2, group_sizes)  # [TR, I]
            gated = jax.nn.silu(w1_out) * w2_out  # [TR, I]
            out_repeat_sort = _ragged_moe_linear(gated, w3, group_sizes)  # [TR, D]
            out_repeat_unflat = self._unpermute(out_repeat_sort, sort_idx)
            out_flat = jnp.sum(out_repeat_unflat * topk_weights[..., None], axis=1)  # [T, D]
            return out_flat, group_sizes

        out_flat, group_sizes = _moe_block(x_flat, topk_idx_flat, topk_weights, self.w1, self.w2, self.w3)
        out = jnp.reshape(out_flat, (B, S, D))

        extras = {}
        if self.cfg.lbl_coef is not None:
            # group_sizes: [E] counts assignments over token-repeat stream (TR = T*K)
            # expert_loads: [E] sums to 1

            group_sizes_f = group_sizes.astype(jnp.float32)
            expert_loads = group_sizes_f / jnp.sum(group_sizes_f)
            extras["expert_loads"] = expert_loads
            f = expert_loads * (E / K)  # [E]
            p = jnp.mean(router_probs.astype(jnp.float32), axis=0)  # [T, E] -> [E]
            extras["load_balancing_loss"] = jnp.asarray(self.cfg.lbl_coef, dtype=jnp.float32) * jnp.sum(f * p)  # scalar

        if self.cfg.rzl_coef is not None:
            z = jsp.special.logsumexp(router_logits.astype(jnp.float32), axis=-1)  # [T]
            extras["router_z_loss"] = jnp.asarray(self.cfg.rzl_coef, dtype=jnp.float32) * jnp.mean(z**2)

        return out, extras

    def _route(self, router_logits: Float[Array, "T E"]):
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        _scores, topk_idx = jax.lax.top_k(router_logits, self.cfg.num_experts_per_tok)
        topk_weights = jnp.take_along_axis(router_probs, topk_idx, axis=-1)
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        return topk_weights, topk_idx.astype(jnp.int32), router_probs

    def _permute(self, x_flat: jax.Array, topk_idx_flat: jax.Array):
        sort_idx = jnp.argsort(topk_idx_flat, axis=-1)
        x_repeat_sort = jnp.take(x_flat, sort_idx // self.cfg.num_experts_per_tok, axis=0)
        group_sizes = jnp.bincount(topk_idx_flat, length=self.cfg.n_routed_experts).astype(jnp.int32)
        return x_repeat_sort, group_sizes, sort_idx.astype(jnp.int32)

    def _unpermute(self, out_repeat_sort: jax.Array, sort_idx: jax.Array):
        inv_sort_idx = jnp.argsort(sort_idx, axis=-1)
        out_repeat = jnp.take(out_repeat_sort, inv_sort_idx, axis=0)
        return jnp.reshape(out_repeat, (-1, self.cfg.num_experts_per_tok, self.cfg.hidden_dim))


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MLP | None
    moe: MOE | None

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        attn_key, moe_key = random.split(key)
        rms_attn = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        attn = CausalSelfAttention.init(cfg, key=attn_key)
        rms_mlp = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        mlp = None
        moe = MOE.init(cfg, key=moe_key)
        return Block(rms_attn, attn, rms_mlp, mlp, moe)

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> tuple[Float[Array, "B S D"], dict]:
        x = x + self.attn(self.rms_attn(x), mask)
        residual = x
        x = self.rms_mlp(x)
        mlp_out = self.mlp(x) if self.mlp is not None else 0
        moe_out, extras = self.moe(x) if self.moe is not None else (0, 0)
        out = residual + mlp_out + moe_out
        return out, extras


class Transformer(eqx.Module):
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: ModelConfig

    @staticmethod
    def init(cfg: ModelConfig, *, key):
        embed_key, out_key, block_key = random.split(key, 3)
        token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
        block_keys = random.split(block_key, cfg.num_layers)
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return Transformer(token_embed, output_proj, blocks, final_norm, cfg)

    @named_call
    def __call__(self, token_ids: Int[Array, "B S"], mask: AttentionMask | jax.Array | None) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        x = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
        all_extras = []
        for block in self.blocks:
            x, extras = eqx.filter_checkpoint(block)(x, mask)
            all_extras.append(extras)
        x = self.final_norm(x)
        aux_loss = self.parse_aux_loss(all_extras)
        return x, aux_loss

    def parse_aux_loss(self, all_extras):
        load_balancing_loss = 0
        router_z_loss = 0
        stats = {}
        for i, extras in enumerate(all_extras):
            if "load_balancing_loss" in extras:
                stats[f"train/layer_{i}/load_balancing_loss"] = jax.lax.stop_gradient(extras["load_balancing_loss"])
                load_balancing_loss += extras["load_balancing_loss"]
            if "router_z_loss" in extras:
                stats[f"train/layer_{i}/router_z_loss"] = jax.lax.stop_gradient(extras["router_z_loss"])
                router_z_loss += extras["router_z_loss"]
            if "expert_loads" in extras:
                expert_loads = extras["expert_loads"]  # [E], sums to 1
                n_experts = self.config.n_routed_experts

                entropy = -jnp.sum(expert_loads * jnp.log(expert_loads + 1e-6))
                load_violation_max = jnp.max(expert_loads) * n_experts

                stats[f"train/layer_{i}/routing_entropy"] = jax.lax.stop_gradient(entropy)
                stats[f"train/layer_{i}/load_violation_max"] = jax.lax.stop_gradient(load_violation_max)
                for j in range(n_experts):
                    stats[f"train/layer_{i}/expert_{j}/load"] = jax.lax.stop_gradient(expert_loads[j])

        stats["train/load_balancing_loss"] = jax.lax.stop_gradient(load_balancing_loss)
        stats["train/router_z_loss"] = jax.lax.stop_gradient(router_z_loss)
        levanter.tracker.jit_log(stats)
        aux_loss = load_balancing_loss + router_z_loss
        return aux_loss


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
    hidden, aux_loss = transformer(token_ids, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    loss_weight = loss_weight.astype(loss_dtype)

    return (
        fused_linear_softmax_cross_entropy_loss(
            hidden,
            transformer.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        + aux_loss
    )


# -----------------------------------------------------------------------------
# Configs


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters available to the model"""

    vocab_size: int = llama3_tokenizer_vocab_size
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 8
    max_seq_len: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope_theta: float = 10_000

    n_routed_experts: int = 8
    lbl_coef: float = 0.01
    rzl_coef: float = 0.001
    num_experts_per_tok: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def intermediate_dim(self) -> int:
        return int(self.hidden_dim * 3)

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
        # MoE SiGLU: 3 projections (gate: D->I, up: D->I, down: I->D) per active expert
        moe_mlp = 2 * 3 * self.hidden_dim * self.intermediate_dim * self.num_experts_per_tok
        router = 2 * self.hidden_dim * self.n_routed_experts
        qkv_proj = 2 * self.hidden_dim * (self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim)
        dense_proj = 2 * self.num_heads * self.head_dim * self.hidden_dim
        # The following are across the whole sequence
        # assume full attention map like megatron-lm
        key_query_logits = 2 * self.max_seq_len**2 * self.num_heads * self.head_dim
        mask = 3 * self.max_seq_len**2 * self.num_heads
        mask_value = 2 * self.max_seq_len**2 * self.head_dim * self.num_heads
        seq_flops = key_query_logits + mask + mask_value
        # so we divide by the sequence length to get the per-token flops
        attn = seq_flops / self.max_seq_len
        lm_head = 2 * self.hidden_dim * self.vocab_size
        return self.num_layers * (moe_mlp + router + qkv_proj + dense_proj + attn) + lm_head


def build_train_config(model_cfg: ModelConfig) -> SimpleTrainConfig:
    batch_size = 128
    num_train_steps = 100

    muon = GrugMuonConfig(
        learning_rate=0.01,
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
        profiler=False,
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
        speedrun_name="grug_moe_scaled_2x_128",
        speedrun_desc="Scaled MoE Transformer (2x layers/hidden/heads) with Muon",
        author=Author(
            name="Larry Dial",
            affiliation="OpenAthena",
            url="https://github.com/ClassicLarry",
        ),
    )
    executor_main(steps=speedrun, description="Single Nano Run")


if __name__ == "__main__":
    main()
