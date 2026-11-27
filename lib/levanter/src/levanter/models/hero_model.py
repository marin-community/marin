# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first Hero inference snapshot with paged KV and causal convolution state."""

import dataclasses
import math

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict
from jax import random
from jax.sharding import get_abstract_mesh, reshard, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from levanter.grug._moe.common import _init_weight
from levanter.grug.attention import (
    AttentionMask,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
    fa4_cute_segment_bounds,
    token_validity_from_attention_mask,
    ragged_paged_attention,
)
from levanter.grug.grug_moe import MoEExpertMlp, MoEExpertMlpPspecs
from levanter.grug.sharding import unshard
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.kernels.triton.short_conv import short_conv
from levanter.layers.attention import AttentionMask as LmHeadAttentionMask
from levanter.layers.kv_cache import PageCache, KvPageCache, ListCache
from levanter.layers.paged_short_conv import ShortConvSequenceCache, paged_short_conv
from levanter.layers.stacked import ArrayStacked
from levanter.models.hero import GlobalQueryLogScaling, HeroConfig
from levanter.models.lm_model import LmHeadModel
from levanter.models.snowball import DenseMLP, GatedNorm, RMSNorm, _reshard_sequence, rms_norm
from levanter.sharding import partition_spec_of
from levanter.utils.activation import ActivationFunctionEnum

_FSDP_AXES = ("data", "expert", "context")
_EXPERT_WEIGHT_AXES = ("expert", "context")
_BATCH_AXES = ("replica_dcn", "data", "expert")
_SEQ_AXIS_NAME = "context"
_EMBED_PARTITION_SPEC = P(None, None)
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")
_ROUTING_RENORM_SUM = 2.5
_XSA_EPSILON = 1e-6


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("Hero requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, context, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"Hero requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _seq_axis(mesh: jax.sharding.AbstractMesh | None) -> str | None:
    """Return the context axis only when it partitions the sequence."""
    if mesh is None or mesh.empty:
        return None
    return _SEQ_AXIS_NAME if int(mesh.shape.get(_SEQ_AXIS_NAME, 1)) > 1 else None


def _token_axes(mesh: jax.sharding.AbstractMesh | None) -> tuple[str, ...]:
    """Return the mesh axes partitioning tokens, with batch axes before context."""
    seq = _seq_axis(mesh)
    return (*_BATCH_AXES, seq) if seq is not None else _BATCH_AXES


def _token_spec() -> P:
    """PartitionSpec for a flattened `[T = B*S, ...]` tensor, on the ambient mesh."""
    return P(_token_axes(get_abstract_mesh()))


def _activation_spec(x: Float[Array, "B S D"]) -> P:
    return partition_spec_of(x) or _batch_spec()


def _embedding_gather(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """Look up tokens locally and establish the context-sharded residual layout."""

    def _local(table: jax.Array, ids: jax.Array) -> jax.Array:
        return table[ids]

    seq_axis = _seq_axis(get_abstract_mesh())
    token_ids = reshard(token_ids, P(_BATCH_AXES, seq_axis))
    return jax.shard_map(
        _local,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, seq_axis)),
        out_specs=P(_BATCH_AXES, seq_axis, None),
    )(token_embed, token_ids)


def _sequence_axis_of(x: jax.Array) -> str | None:
    spec = partition_spec_of(x)
    return spec[1] if spec is not None and len(spec) > 1 else None


def _apply_rotary_embedding_fused(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S H D"],
    *,
    position_ids: jax.Array,
    head_dim: int,
    rotary_dim: int,
    rope,
    disable_rope: jax.Array | bool,
) -> tuple[Float[Array, "B S H D"], Float[Array, "B S H D"]]:
    # Schema-v2 fused RoPE rotates adjacent pairs; the unfused path pairs split halves.
    # Checkpoint loading and incremental decode must preserve the exported convention.
    half = rotary_dim // 2
    inv_freq = 1.0 / (rope.theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = position_ids.astype(jnp.float32)[..., None] * inv_freq
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    first_factor = jnp.repeat(cos, 2, axis=-1)
    second_factor = jnp.reshape(jnp.stack([-sin, sin], axis=-1), (*position_ids.shape, rotary_dim))
    if rotary_dim < head_dim:
        padding = head_dim - rotary_dim
        first_factor = jnp.concatenate(
            [first_factor, jnp.ones((*position_ids.shape, padding), first_factor.dtype)],
            axis=-1,
        )
        second_factor = jnp.concatenate(
            [second_factor, jnp.zeros((*position_ids.shape, padding), second_factor.dtype)],
            axis=-1,
        )
    # These factors index global positions, so they stay correct once Q/K carry a
    # context-sharded sequence axis: each shard multiplies by the rows it holds.
    first_factor = jnp.where(disable_rope, 1.0, first_factor)[:, :, None, :]
    second_factor = jnp.where(disable_rope, 0.0, second_factor)[:, :, None, :]

    def _apply(x: Float[Array, "B S H D"]) -> Float[Array, "B S H D"]:
        dtype = x.dtype
        flipped = jnp.flip(x.reshape(*x.shape[:-1], head_dim // 2, 2), axis=-1).reshape(x.shape)
        return (first_factor * x + second_factor * flipped).astype(dtype)

    return _apply(q), _apply(k)


def _apply_split_half_rope(
    q: jax.Array, k: jax.Array, rope: RotaryConfig, position_ids: jax.Array | None = None
) -> tuple[jax.Array, jax.Array]:
    half = q.shape[-1] // 2
    q_rot, k_rot = apply_rotary_embedding(
        q[..., :half], k[..., :half], seq_len=q.shape[1], head_dim=half, rope=rope, position_ids=position_ids
    )
    return (
        jnp.concatenate([q_rot, q[..., half:]], axis=-1),
        jnp.concatenate([k_rot, k[..., half:]], axis=-1),
    )


class ShortConv(eqx.Module):
    """Depthwise causal 1-D convolution over the sequence axis (Inkling-style SConv).

    A kernel of ``W`` taps mixes each channel with its own ``W-1`` causal predecessors,
    ``out[t] = sum_{lag} weight[lag] * x[t-lag]``, independently per channel. Identity-init
    (``weight[0]=1``, later taps 0) preserves the input before training.
    Context shards exchange a left halo of ``W-1`` sequence positions.

    """

    weight: Float[Array, "W C"]
    kernel_size: int = eqx.field(static=True)

    @staticmethod
    def init(channels: int, kernel_size: int) -> "ShortConv":
        weight = jnp.zeros((kernel_size, channels)).at[0].set(1.0)
        # Preserve the checkpoint's channel partitioning; forward gathers the small filter.
        return ShortConv(weight=reshard(weight, P(None, _FSDP_AXES)), kernel_size=kernel_size)

    def __call__(
        self, x: Float[Array, "B S C"], segment_ids: Int[Array, "B S"] | None = None
    ) -> Float[Array, "B S C"]:
        # With segment_ids (packed documents), a tap that reaches into a previous document is
        # zeroed so the conv never mixes across a boundary; the lag-0 (current-token) tap is
        # always kept.
        weight = reshard(self.weight, P(None, None))
        return short_conv(weight, x, segment_ids, batch_axes=_BATCH_AXES)


class HeroAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    sconv_k: "ShortConv | None"  # SConv after the K projection (cfg.sconv)
    cfg: HeroConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HeroConfig, *, key: PRNGKeyArray) -> "HeroAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        return HeroAttention(
            w_q=_init_weight(k_q, (d, n * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_k=_init_weight(k_k, (d, m * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_v=_init_weight(k_v, (d, m * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_o=_init_weight(k_o, (n * h, d), cfg.initializer_std, P("model", _FSDP_AXES)),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            sconv_k=(ShortConv.init(m * h, cfg.sconv_kernel) if cfg.uses_short_conv("k") else None),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        *,
        is_global: bool = False,
        query_log_scale: jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        # The residual's sequence layout (context-sharded under CP, or None). K/V norm and RoPE
        # stay in it, the attention output returns to it, and `w_o` writes it.
        residual_seq_axis = _sequence_axis_of(x)

        q_flat = jnp.einsum("bsh,hd->bsd", x, self.w_q)
        k_flat = jnp.einsum("bsh,hd->bsd", x, self.w_k)
        v_flat = jnp.einsum("bsh,hd->bsd", x, self.w_v)
        # SConv: depthwise causal conv after the K projection. segment_ids (packed-document
        # boundaries) come from the mask so the conv never mixes across a document boundary.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = segment_ids[0] if segment_ids is not None else None
        if self.sconv_k is not None:
            k_flat = self.sconv_k(k_flat, sconv_segment_ids)
        q = rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)
        k = rearrange(k_flat, "... (m d) -> ... m d", d=head_dim)
        v = rearrange(v_flat, "... (m d) -> ... m d", d=head_dim)

        if self.cfg.local_kv_heads is not None and self.cfg.global_kv_heads is not None:
            stored_kv_heads = self.cfg.stored_kv_heads

            # Replicate the head axis rather than pinning it to `model`: a shape can carry fewer
            # KV heads than the model axis is wide (d768 stores one), and the KV tensors are small
            # enough -- at most a dozen heads of 128 -- that replication is not worth a special case.
            #
            # Keep the projection's sequence layout: under context parallelism the K/V gather
            # happens once, right before attention, so norm and RoPE run on the local shard and
            # the gather is not trapped inside this cond.
            kv_spec = P(_BATCH_AXES, residual_seq_axis, None, None)

            def _logical_kv(projection: jax.Array, num_kv_heads: int) -> jax.Array:
                # Replicate before slicing, not after: narrowing a `model`-sharded head axis to a
                # count that does not divide the mesh axis is unsupported.
                projection = reshard(projection, kv_spec)
                if num_kv_heads == stored_kv_heads:
                    return projection
                return reshard(
                    align_kv_heads(projection[:, :, :num_kv_heads, :], num_q_heads=stored_kv_heads), kv_spec
                )

            k, v = jax.lax.cond(
                jnp.asarray(is_global, dtype=jnp.bool_),
                lambda kv: (
                    _logical_kv(kv[0], self.cfg.global_kv_heads),
                    _logical_kv(kv[1], self.cfg.global_kv_heads),
                ),
                lambda kv: (
                    _logical_kv(kv[0], self.cfg.local_kv_heads),
                    _logical_kv(kv[1], self.cfg.local_kv_heads),
                ),
                (k, v),
            )

        q = rms_norm(q)
        k = rms_norm(k)

        # Local layers rotate half of Q/K; global layers omit RoPE.
        if self.cfg.rope_fused:
            q, k = _apply_rotary_embedding_fused(
                q,
                k,
                position_ids=jnp.arange(seq_len)[None, :],
                head_dim=head_dim,
                rotary_dim=head_dim // 2,
                rope=self.cfg.rope,
                disable_rope=is_global,
            )
        elif not is_global:
            q, k = _apply_split_half_rope(q, k, self.cfg.rope)
        q = q * self.cfg.qk_mult
        if is_global and self.cfg.qk_mult_long_scale != 1.0:
            q = q * self.cfg.qk_mult_long_scale
        if query_log_scale is not None:
            q = q * query_log_scale[..., None, None].astype(q.dtype)
        # Context parallelism: shard Q's sequence over "context" and all-gather K/V, so each
        # shard attends its own queries against the whole key sequence. The backends reject a
        # sharded K/V sequence, and the output returns to the residual stream's layout, which
        # `w_o` and the residual add then keep.
        seq_axis = _seq_axis(get_abstract_mesh())
        # XSA needs v row-aligned with the local attention output, so keep the pre-gather v.
        v_local = v
        if seq_axis is not None:
            q = _reshard_sequence(q, seq_axis)
            k = _reshard_sequence(k, None)
            v = _reshard_sequence(v, None)
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        if seq_axis is not None:
            attn_out = _reshard_sequence(attn_out, residual_seq_axis)
        return _attention_output(self, x, attn_out, v_local)


def _query_log_multiplier(visible_keys: jax.Array, scaling: GlobalQueryLogScaling) -> jax.Array:
    factor = 1.0 + scaling.alpha * jnp.log(jnp.maximum(visible_keys.astype(jnp.float32) / scaling.n_floor, 1.0))
    return factor if scaling.power == 1.0 else factor**scaling.power


def _long_layer_schedule(num_layers: int, global_every: int) -> jax.Array:
    layer_indices = jnp.arange(num_layers)
    return (((layer_indices + 1) % global_every) == 0) | (layer_indices == num_layers - 1)


class HeroMoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    w_latent_down: jax.Array | None
    latent_norm: RMSNorm | None
    w_latent_up: jax.Array | None
    cfg: HeroConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HeroConfig, *, key: PRNGKeyArray) -> "HeroMoEMLP":
        k_router, k_expert, k_down, k_up = random.split(key, 4)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e = cfg.hidden_dim, cfg.num_experts
        # Routed experts live in the latent space; the router reads the full-width token, so its
        # own projection keeps `hidden_dim`.
        expert_width = cfg.latent_dim if cfg.latent_dim is not None else d
        latent = cfg.latent_dim
        return HeroMoEMLP(
            router=_init_weight(k_router, (d, e), cfg.initializer_std, P(None, None)),
            router_bias=jnp.zeros((e,)),
            w_latent_down=(
                None
                if latent is None
                else _init_weight(k_down, (d, latent), cfg.initializer_std, P(_FSDP_AXES, "model"))
            ),
            latent_norm=None if latent is None else RMSNorm.init(latent, cfg.layer_norm_eps),
            w_latent_up=(
                None
                if latent is None
                else _init_weight(k_up, (latent, d), cfg.initializer_std, P("model", _FSDP_AXES))
            ),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=expert_width,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
                pspecs=MoEExpertMlpPspecs(expert=_EXPERT_WEIGHT_AXES),
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        token_valid: Bool[Array, "B S"],
    ) -> Float[Array, "B S D"]:
        b, s, _ = x.shape
        x_flat = reshard(rearrange(x, "b s d -> (b s) d"), _token_spec())
        token_valid_flat = reshard(rearrange(token_valid, "b s -> (b s)"), _token_spec())
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + unshard(self.router_bias)
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        selected_experts = selected_experts[:, :-1]
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_fp32 = jax.nn.sigmoid(unbiased_topk)
        denom = jnp.sum(combine_weights_fp32, axis=-1, keepdims=True)
        combine_weights_fp32 = combine_weights_fp32 * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_fp32.astype(x.dtype)
        # LatentMoE: compress before dispatch so the expert-parallel all-to-all carries
        # `latent_dim`-wide rows in both directions. The router above already read the full-width
        # token, and the shared experts in the enclosing block never see this path.
        routed_input = x_flat
        if self.w_latent_down is not None and self.latent_norm is not None:
            routed_input = jnp.einsum(
                "td,dl->tl",
                x_flat,
                self.w_latent_down.astype(x_flat.dtype),
                out_sharding=_token_spec(),
            )
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = self.latent_norm(routed_input)
        moe_out = self.expert_mlp(
            routed_input,
            selected_experts.astype(jnp.int32),
            combine_weights,
            token_valid=token_valid_flat,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=False,
        )
        routed_flat = moe_out

        # Expand after the combine: `expert_mlp` already returns the weight-summed expert output,
        # which is the vector the paper's W_up acts on.
        if self.w_latent_up is not None:
            routed_flat = jnp.einsum(
                "tl,ld->td",
                routed_flat,
                self.w_latent_up.astype(routed_flat.dtype),
                out_sharding=_token_spec(),
            )

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _activation_spec(x))
        return routed


class HeroBlock(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: HeroAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: HeroMoEMLP
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: "ShortConv | None"  # SConv on the attention branch output (cfg.sconv)
    sconv_mlp: "ShortConv | None"  # SConv on the MoE branch output (cfg.sconv)

    @staticmethod
    def init(cfg: HeroConfig, *, key: PRNGKeyArray) -> "HeroBlock":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            num_shared_experts = cfg.num_shared_experts
            per_expert_dim = cfg.shared_expert_intermediate_dim
            if num_shared_experts == 1:
                shared_keys = (shared_key,)
            else:
                shared_keys = tuple(random.split(shared_key, num_shared_experts))
            shared = tuple(
                DenseMLP.init(cfg.hidden_dim, per_expert_dim, cfg.initializer_std, key=key) for key in shared_keys
            )
        return HeroBlock(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=HeroAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=HeroMoEMLP.init(cfg, key=mlp_key),
            shared=shared,
            sconv_attn=(ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.uses_short_conv("attn") else None),
            sconv_mlp=(ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.uses_short_conv("mlp") else None),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        *,
        is_global: bool = False,
        query_log_scale: jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        # segment_ids (packed-document boundaries) for the branch-output SConvs; None when unpacked.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = segment_ids[0] if segment_ids is not None else None

        attn_in = self.attn_gated_norm(self.rms_attn(x))
        attn_out = self.attn(attn_in, mask, is_global=is_global, query_log_scale=query_log_scale)
        if self.sconv_attn is not None:
            attn_out = self.sconv_attn(attn_out, sconv_segment_ids)
        x = x + attn_out
        token_valid = token_validity_from_attention_mask(mask, batch_size=x.shape[0], sequence_length=x.shape[1])
        mlp_out = self._mlp_output(x, token_valid)
        if self.sconv_mlp is not None:
            mlp_out = self.sconv_mlp(mlp_out, sconv_segment_ids)
        x = x + mlp_out
        return x

    def _mlp_output(self, x: jax.Array, token_valid: jax.Array | None) -> jax.Array:
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        output = self.mlp(mlp_in, token_valid)
        if self.shared is not None:
            for expert in self.shared:
                output = output + expert(mlp_in)
        return output

    @named_call
    def decode(
        self,
        x: jax.Array,
        cache: "HeroLayerCache",
        batch_info: PageBatchInfo,
        positions: NamedArray,
        token_valid: jax.Array,
        use_long: jax.Array,
    ) -> tuple[jax.Array, "HeroLayerCache"]:
        """Apply one layer to packed tokens and advance its KV and convolution cache."""
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        attn_out, cache = jax.lax.cond(
            use_long,
            lambda _: _decode_attention(self.attn, attn_in, cache, batch_info, positions, use_long=True),
            lambda _: _decode_attention(self.attn, attn_in, cache, batch_info, positions, use_long=False),
            operand=None,
        )
        attn_out, attn_history = _decode_convolution(
            self.sconv_attn, attn_out, cache.attn_history, batch_info, positions
        )
        x = x + attn_out
        mlp_out, mlp_history = _decode_convolution(
            self.sconv_mlp, self._mlp_output(x, token_valid), cache.mlp_history, batch_info, positions
        )
        return x + mlp_out, dataclasses.replace(cache, attn_history=attn_history, mlp_history=mlp_history)


class HeroTransformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[HeroBlock]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: HeroConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: HeroConfig, *, key: PRNGKeyArray) -> "HeroTransformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        return HeroTransformer(
            token_embed=_init_weight(
                embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std, _EMBED_PARTITION_SPEC
            ),
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=_init_weight(
                out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std, _LM_HEAD_PARTITION_SPEC
            ),
            stacked_blocks=ArrayStacked.init(cfg.num_layers, HeroBlock)(cfg, key=jnp.stack(block_keys)),
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    def __call__(self, token_ids: jax.Array, mask: AttentionMask | None = None) -> jax.Array:
        cfg = self.config
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.embed_gated_norm(self.embed_norm(_embedding_gather(self.token_embed, token_ids)))
        segment_ids = mask.segment_ids
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
        batch, seq_len = token_ids.shape
        long_lower, valid = fa4_cute_segment_bounds(long_mask, batch_size=batch, seq_len=seq_len, sliding_window=None)
        short_lower, _ = fa4_cute_segment_bounds(
            short_mask, batch_size=batch, seq_len=seq_len, sliding_window=cfg.sliding_window
        )
        bounds_spec = P(_BATCH_AXES, _seq_axis(get_abstract_mesh()))
        long_lower, short_lower, valid = (reshard(x, bounds_spec) for x in (long_lower, short_lower, valid))
        query_log_scale = None
        if cfg.global_query_log_scaling is not None:
            positions = reshard(jnp.arange(seq_len, dtype=jnp.int32)[None, :], P(None, _seq_axis(get_abstract_mesh())))
            visible_keys = jnp.where(valid, positions - long_lower + 1, 1)
            query_log_scale = _query_log_multiplier(visible_keys, cfg.global_query_log_scaling)

        def layer_step(hidden, inputs):
            block, use_long = inputs
            output = jax.lax.cond(
                use_long,
                lambda x: block(
                    x, long_mask.with_fa4_bounds(long_lower, valid), is_global=True, query_log_scale=query_log_scale
                ),
                lambda x: block(x, short_mask.with_fa4_bounds(short_lower, valid)),
                hidden,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            layer_step, hidden, (self.stacked_blocks.stacked, _long_layer_schedule(cfg.num_layers, cfg.global_every))
        )
        return self.final_gated_norm(self.final_norm(hidden))


class HeroLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[HeroConfig]):
    """Named-tensor boundary for native Hero inference."""

    transformer: HeroTransformer
    _config: HeroConfig = eqx.field(static=True)

    @property
    def config(self) -> HeroConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @classmethod
    def init(cls, Vocab: Axis, config: HeroConfig, *, key: PRNGKeyArray) -> "HeroLMHeadModel":
        config = dataclasses.replace(config, vocab_size=Vocab.size)
        return cls(HeroTransformer.init(config, key=key), config)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: LmHeadAttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        if pos_ids is not None:
            raise ValueError("Hero full forward uses contiguous positions; use decode for explicit positions")
        if attn_mask is not None and (
            not isinstance(attn_mask, LmHeadAttentionMask)
            or not attn_mask.is_causal
            or attn_mask.explicit_mask is not None
            or attn_mask.causal_offset is not None
            or attn_mask.sliding_window is not None
            or attn_mask.bidirectional_window is not None
        ):
            raise ValueError("Hero expects a causal AttentionMask with optional segment IDs")

        Pos = input_ids.resolve_axis(self.Pos.name)
        ordered = input_ids.rearrange((..., Pos))
        batch = math.prod(ordered.array.shape[:-1])
        token_spec = P(_BATCH_AXES, _seq_axis(get_abstract_mesh()))
        tokens = reshard(ordered.array.reshape(batch, Pos.size), token_spec)
        mask = AttentionMask.causal()
        if attn_mask is not None and attn_mask.segment_ids is not None:
            query, keys = attn_mask.segment_ids
            if self.KeyPos.name in keys.axis_names:
                keys = keys.rename({self.KeyPos.name: Pos.name})
            query_ids = reshard(query.broadcast_to(ordered.axes).array.reshape(batch, Pos.size), token_spec)
            key_ids = reshard(keys.broadcast_to(ordered.axes).array.reshape(batch, Pos.size), token_spec)
            mask = mask.with_segment_ids(query_ids, key_ids)
        hidden = self.transformer(tokens, mask)
        hidden = hidden.reshape((*ordered.array.shape, self.Embed.size))
        return hax.named(hidden, (*ordered.axes, self.Embed)).rearrange((*input_ids.axes, self.Embed))

    def get_lm_head(self) -> NamedArray:
        return hax.named(reshard(self.transformer.output_proj, P(None, "model")), (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key=None) -> "HeroLMHeadModel":
        """Return the model when the requested size matches the exported vocabulary."""
        if new_size != self.Vocab.size:
            raise ValueError(
                "Hero inference requires the checkpoint vocabulary; export resized weights before loading"
            )
        return self

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache["HeroLayerCache"]:
        cfg = self.config
        if _mesh_axis_size(get_abstract_mesh(), "context") != 1:
            raise ValueError("Hero paged inference requires context_axis_size=1")

        def conv_cache(site, channels):
            if not cfg.uses_short_conv(site) or cfg.sconv_kernel == 1:
                return None
            return ShortConvSequenceCache.init(spec, Axis("channel", channels), cfg.sconv_kernel, dtype)

        caches = []
        for _ in range(cfg.num_layers):
            kv = KvPageCache.init(
                spec, Axis("kv_head", cfg.stored_kv_heads), Axis("head_size", cfg.inferred_head_dim), dtype
            )
            kv = KvPageCache(hax.named(reshard(kv.kv_pages.array, P(None, None, "model", None)), kv.kv_pages.axes))
            caches.append(
                HeroLayerCache(
                    kv,
                    conv_cache("k", cfg.stored_kv_heads * cfg.inferred_head_dim),
                    conv_cache("attn", cfg.hidden_dim),
                    conv_cache("mlp", cfg.hidden_dim),
                )
            )
        return ListCache(tuple(caches))

    def decode(
        self,
        input_ids: NamedArray,
        kv_cache: ListCache["HeroLayerCache"],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        logit_positions: NamedArray | None = None,
        key=None,
    ) -> tuple[NamedArray, ListCache["HeroLayerCache"]]:
        """Advance the cache for all input rows and project the requested logits."""
        cfg = self.config
        if input_ids.ndim != 1 or pos_ids.axes != input_ids.axes:
            raise ValueError("Hero decode requires matching flat token and position arrays")
        if len(kv_cache) != cfg.num_layers:
            raise ValueError("Hero decode requires one cache per layer")
        tokens = reshard(input_ids.array[:, None], P(_BATCH_AXES, None))
        hidden = self.transformer.embed_gated_norm(
            self.transformer.embed_norm(_embedding_gather(self.transformer.token_embed, tokens))
        )
        valid = jnp.arange(tokens.shape[0])[:, None] < batch_info.num_new_tokens
        blocks = self.transformer.stacked_blocks.stacked
        expert_size = _mesh_axis_size(get_abstract_mesh(), "expert")
        if expert_size > 1:
            # Paged serving must retain every routed assignment. The ring with
            # one full token budget per source rank guarantees dropless dispatch.
            experts = dataclasses.replace(
                blocks.mlp.expert_mlp, implementation="ring", capacity_factor=float(expert_size)
            )
            blocks = eqx.tree_at(lambda block: block.mlp.expert_mlp, blocks, experts)
        stacked_cache = jax.tree.map(lambda *arrays: jnp.stack(arrays), *kv_cache.caches)

        def layer_step(x, layer_inputs):
            block, cache, use_long = layer_inputs
            return block.decode(x, cache, batch_info, pos_ids, valid, use_long)

        hidden, updated = jax.lax.scan(
            layer_step, hidden, (blocks, stacked_cache, _long_layer_schedule(cfg.num_layers, cfg.global_every))
        )
        hidden = self.transformer.final_gated_norm(self.transformer.final_norm(hidden))
        caches = ListCache(tuple(jax.tree.map(lambda array: array[i], updated) for i in range(cfg.num_layers)))
        if logit_positions is not None:
            selected = hidden[:, 0].at[logit_positions.array].get(out_sharding=P())
            logits = jnp.einsum("...d,dv->...v", selected, self.transformer.output_proj, out_sharding=P())
            return hax.named(logits, (*logit_positions.axes, self.Vocab)), caches

        logits = jnp.einsum(
            "bsd,dv->bsv", hidden, self.transformer.output_proj, out_sharding=P(_BATCH_AXES, None, "model")
        )[:, 0]
        return hax.named(logits, (input_ids.axes[0], self.Vocab)), caches

    def to_state_dict(self, prefix: str | None = None) -> StateDict:
        # Serialization imports the concrete transformer type.
        from levanter.models.hero_serialization import hero_to_state_dict  # noqa: PLC0415

        return hero_to_state_dict(self.transformer, prefix)

    def from_state_dict(self, state_dict: StateDict, prefix: str | None = None) -> "HeroLMHeadModel":
        from levanter.models.hero_serialization import hero_from_state_dict  # noqa: PLC0415

        return HeroLMHeadModel(hero_from_state_dict(self.transformer, state_dict, prefix), self.config)


class HeroLayerCache(PageCache):
    kv: KvPageCache
    k_history: ShortConvSequenceCache | None
    attn_history: ShortConvSequenceCache | None
    mlp_history: ShortConvSequenceCache | None

    def copy_page(self, src_page: int, dst_page: int) -> "HeroLayerCache":
        return dataclasses.replace(self, kv=self.kv.copy_page(src_page, dst_page))

    def copy_sequence(self, src_slot: int, dst_slot: int) -> "HeroLayerCache":
        return HeroLayerCache(
            self.kv,
            None if self.k_history is None else self.k_history.copy_sequence(src_slot, dst_slot),
            None if self.attn_history is None else self.attn_history.copy_sequence(src_slot, dst_slot),
            None if self.mlp_history is None else self.mlp_history.copy_sequence(src_slot, dst_slot),
        )

    def reset(self) -> "HeroLayerCache":
        return HeroLayerCache(
            self.kv.reset(),
            None if self.k_history is None else self.k_history.reset(),
            None if self.attn_history is None else self.attn_history.reset(),
            None if self.mlp_history is None else self.mlp_history.reset(),
        )


def _decode_convolution(
    conv: ShortConv | None,
    x: jax.Array,
    history: ShortConvSequenceCache | None,
    info: PageBatchInfo,
    positions: NamedArray,
):
    if conv is None:
        return x, history
    if conv.kernel_size == 1:
        return x * unshard(conv.weight)[0], history
    assert history is not None
    named_x = hax.named(reshard(x[:, 0], P(None, None)), (positions.axes[0], Axis("channel", x.shape[-1])))
    output, history = paged_short_conv(unshard(conv.weight), named_x, history, info, positions)
    return reshard(output.array[:, None], _activation_spec(x)), history


def _decode_attention(
    attn: HeroAttention,
    x: jax.Array,
    cache: HeroLayerCache,
    info: PageBatchInfo,
    positions: NamedArray,
    *,
    use_long: bool,
):
    cfg = attn.cfg
    width = cfg.inferred_head_dim
    q = (x @ attn.w_q).reshape(x.shape[0], 1, cfg.num_heads, width)
    k_flat, k_history = _decode_convolution(attn.sconv_k, x @ attn.w_k, cache.k_history, info, positions)
    k = k_flat.reshape(x.shape[0], 1, cfg.stored_kv_heads, width)
    v = (x @ attn.w_v).reshape(x.shape[0], 1, cfg.stored_kv_heads, width)
    logical_heads = cfg.global_kv_heads if use_long else cfg.local_kv_heads
    if logical_heads is not None and logical_heads != cfg.stored_kv_heads:
        k = align_kv_heads(
            reshard(k, P(_BATCH_AXES, None, None, None))[:, :, :logical_heads], num_q_heads=cfg.stored_kv_heads
        )
        v = align_kv_heads(
            reshard(v, P(_BATCH_AXES, None, None, None))[:, :, :logical_heads], num_q_heads=cfg.stored_kv_heads
        )
    q, k = rms_norm(q), rms_norm(k)
    if cfg.rope_fused:
        q, k = _apply_rotary_embedding_fused(
            q,
            k,
            position_ids=positions.array[:, None],
            head_dim=width,
            rotary_dim=width // 2,
            rope=cfg.rope,
            disable_rope=use_long,
        )
    elif not use_long:
        q, k = _apply_split_half_rope(q, k, cfg.rope, positions.array[:, None])
    axes = (positions.axes[0], Axis("kv_head", cfg.stored_kv_heads), Axis("head_size", width))
    kv = cache.kv.update(
        info,
        hax.named(reshard(k[:, 0], P(None, "model", None)), axes),
        hax.named(reshard(v[:, 0], P(None, "model", None)), axes),
    )
    q = q * cfg.qk_mult
    if use_long and cfg.qk_mult_long_scale != 1.0:
        q = q * cfg.qk_mult_long_scale
    if use_long and cfg.global_query_log_scaling is not None:
        scale = _query_log_multiplier(positions.array[:, None] + 1, cfg.global_query_log_scaling)
        q = q * scale[..., None, None].astype(q.dtype)
    query = q.reshape(x.shape[0], cfg.stored_kv_heads, cfg.num_heads // cfg.stored_kv_heads, width)
    query = reshard(query, P(None, "model", None, None))
    output = ragged_paged_attention(
        query,
        kv.kv_pages.array,
        info.seq_lens.array,
        info.page_indices.array,
        info.cu_q_lens.array,
        info.num_seqs,
        sm_scale=width**-0.5,
        sliding_window=None if use_long else cfg.sliding_window,
        implementation=cfg.inference_attention_implementation
        or ("reference" if cfg.attention_implementation == "reference" else None),
    )
    output = output.reshape(x.shape[0], 1, cfg.num_heads, width)
    output = _attention_output(attn, x, output, v)
    return output, dataclasses.replace(cache, kv=kv, k_history=k_history)


def _attention_output(attn: HeroAttention, x: jax.Array, output: jax.Array, v: jax.Array) -> jax.Array:
    # Exclusive Self Attention subtracts the component parallel to this token's value.
    aligned_v = reshard(
        align_kv_heads(v, num_q_heads=attn.cfg.num_heads),
        partition_spec_of(output) or P(_BATCH_AXES, None, None, "model"),
    )
    dot = jnp.sum(output * aligned_v, axis=-1, keepdims=True)
    norm = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
    output = output - (dot / (norm + _XSA_EPSILON)) * aligned_v
    gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, attn.attn_gate))[..., None]
    sequence_axis = _sequence_axis_of(x)
    output = (output * gate).reshape(
        *x.shape[:2],
        attn.cfg.num_heads * attn.cfg.inferred_head_dim,
        out_sharding=P(_BATCH_AXES, sequence_axis, "model"),
    )
    return jnp.einsum("bsh,hd->bsd", output, attn.w_o, out_sharding=P(_BATCH_AXES, sequence_axis, None))
