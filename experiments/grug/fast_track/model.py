# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel MoE grug variant model.

The default config is the fast_track baseline: sliding-window local layers and full-causal global
layers, all GQA softmax attention with half-RoPE, scanned as one layer stack. The KMA variant
(``local_mixer=KDA`` + ``mla`` + ``inkling_relpos`` + ``attn_res``) swaps the local layers for Kimi
Delta Attention, the global layers for MLA with an Inkling relative-position bias, and the residual
stream for Block Attention Residuals.
"""

import dataclasses
import functools
import itertools
import math
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.nn import ArrayStacked
from jax import core, random
from jax.sharding import NamedSharding, get_abstract_mesh, reshard
from jax.sharding import PartitionSpec as P

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
    fa4_cute_segment_bounds,
    inkling_rel_bias,
)
from levanter.grug.grug_moe import (
    MoeActivation,
    MoEExpertMlp,
)
from levanter.grug.loss import BlockSizes, fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import unshard
from levanter.kernels.pallas.short_conv import short_conv
from levanter.tracker.histogram import SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track.router_metrics import (
    local_routing_stats,
    reduce_router_stats,
    summarize_router_metrics,
)
from experiments.grug.moe.kda import chunk_kda, kda_fused

_GATED_NORM_RANK = 128
_QB_HIST_BINS = 10_000
_CE_TOKENS_PER_RANK = 65_536
# A vocab tile of 8192 in the fused lm_head + cross-entropy loop measured ~3% faster than 4096 at d512.
_CE_BLOCK_SIZES = BlockSizes(b_block_size=_CE_TOKENS_PER_RANK, v_block_size=8192)
# Axes the non-expert params FSDP-shard over.
_FSDP_AXES: tuple[str, ...] = ("data", "expert")
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
# Metrics-dict key that carries the auxiliary-loss residual stream from the forward to the loss.
_AUX_HIDDEN = "aux_lm_hidden"
# Metrics-dict keys carrying the AttnRes z-loss term (with gradient) from the forward to the loss.
_ATTN_RES_Z = "attn_res_z_term"
# Per-gate token-mean AttnRes source weights (variable length per gate), popped into logging scalars.
_ATTN_RES_W_ATTN = "attn_res_weights_attn"
_ATTN_RES_W_MLP = "attn_res_weights_mlp"

# Kimi K3's KDA layer: low-rank forget-gate width, and the per-token log-decay floor
# ``g = -KDA_MIN_LOG_DECAY * sigmoid(...)``.
_KDA_GATE_RANK = 128
KDA_MIN_LOG_DECAY = 5.0
# 16-token chunks keep the kernels' intra-chunk rescaling exact for the -5 per-token log-decay
# floor (cumulative >= -80 = -DEFLATE_EXP_CAP; see kda_prep_pallas).
KDA_CHUNK_SIZE = 16


class LocalMixer(StrEnum):
    """Token mixer of the local layers; global layers (every ``global_every``-th + last) are always
    full causal softmax attention."""

    SLIDING_WINDOW = "sliding_window"
    KDA = "kda"
    """Kimi Delta Attention (``KimiDeltaAttention``); requires ``attn_res``."""


class AttnResLayerBackward(StrEnum):
    """How a Block AttnRes layer's backward obtains its forward intermediates."""

    RECOMPUTE = "recompute"
    """Custom VJP that saves only the layer inputs and re-runs the layer forward in backward (lowest
    memory; costs a second forward of every layer)."""
    SAVE = "save"
    """Plain autodiff: the forward keeps the layer's residuals, so backward runs no forward again.
    Same math as RECOMPUTE; needs the memory to hold every layer's residuals."""


class RouterCombine(StrEnum):
    """How the MoE combine weights of the K selected experts are formed from their unbiased logits."""

    SIGMOID_RENORM = "sigmoid_renorm"
    """``sigmoid(logit)``, renormalized to sum to ``routing_renorm_sum``."""
    SOFTMAX_RENORM = "softmax_renorm"
    """Softmax over the K selected logits, times ``routing_renorm_sum``."""
    SIGMOID_RAW = "sigmoid_raw"
    """``sigmoid(logit)`` times a constant (``routing_renorm_sum / (K/2)``, so the sum matches at init),
    no renormalization: a token's total expert weight can vary."""


class ValueEmbeds(StrEnum):
    """Value embeddings on the MLA layers: ``v = lambda1 * v + w * value_embed[token]``."""

    NONE = "none"
    LAMBDA = "lambda"
    """``w = lambda2``, a learned scalar per layer (``lambda1`` init 1, ``lambda2`` init 0)."""
    GATED = "gated"
    """``w = sigmoid(x W_ve_gate)`` per head, ``W_ve_gate`` zero-initialized (0.5 at init)."""


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/fast_track requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/fast_track requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _local_gather(table: jax.Array, ids: jax.Array) -> jax.Array:
    return table[ids]


@jax.custom_vjp
def _embedding_gather(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """Look up tokens from a replicated table without a cross-rack collective.

    The backward scatter-adds each shard's row cotangents in float32 and then sums over the batch axes.
    Under the bf16 compute policy the table arrives in bf16, and the default gather transpose would
    scatter-add every token's cotangent straight into a bf16 table: contended bf16 atomics that are
    slow (~11 ms/step at d512) and round each frequent token's accumulated gradient at every add.
    """
    token_ids = reshard(token_ids, P(_BATCH_AXES, None))
    return shard_map(
        _local_gather,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(token_embed, token_ids)


def _embedding_gather_fwd(token_embed: jax.Array, token_ids: jax.Array):
    return _embedding_gather(token_embed, token_ids), (token_ids, jnp.zeros((0, *token_embed.shape), token_embed.dtype))


def _embedding_gather_bwd(residuals, g: jax.Array):
    token_ids, table_like = residuals
    vocab, dim = table_like.shape[1:]

    def _local_scatter(ids: jax.Array, cot: jax.Array) -> jax.Array:
        local = jnp.zeros((vocab, dim), jnp.float32).at[ids].add(cot.astype(jnp.float32))
        return jax.lax.psum(local.astype(table_like.dtype), _BATCH_AXES)

    d_table = shard_map(
        _local_scatter,
        mesh=get_abstract_mesh(),
        in_specs=(P(_BATCH_AXES, None), P(_BATCH_AXES, None, None)),
        out_specs=P(None, None),
    )(reshard(token_ids, P(_BATCH_AXES, None)), reshard(g, P(_BATCH_AXES, None, None)))
    return d_table, np.zeros(token_ids.shape, dtype=jax.dtypes.float0)


_embedding_gather.defvjp(_embedding_gather_fwd, _embedding_gather_bwd)


# Pair-combine multiplier and murmur3 finalizer constants for the (previous, current) bigram hash.
_BIGRAM_HASH_PAIR = 0x9E3779B1
_MURMUR_C1 = 0x85EBCA6B
_MURMUR_C2 = 0xC2B2AE35


def _bigram_hash_ids(
    token_ids: Int[Array, "B S"], segment_ids: Int[Array, "B S"] | None, num_buckets: int
) -> Int[Array, "B S"]:
    """Hash each (previous token, token) pair into ``num_buckets`` rows. At position 0 and at every
    document start the previous token is replaced by the sentinel ``num_buckets`` (never a real id)."""
    prev = jnp.pad(token_ids[:, :-1], ((0, 0), (1, 0)), constant_values=num_buckets)
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        prev = jnp.where(starts, num_buckets, prev)
    x = prev.astype(jnp.uint32) * jnp.uint32(_BIGRAM_HASH_PAIR) + token_ids.astype(jnp.uint32)
    x = (x ^ (x >> 16)) * jnp.uint32(_MURMUR_C1)
    x = (x ^ (x >> 13)) * jnp.uint32(_MURMUR_C2)
    x = x ^ (x >> 16)
    return (x % jnp.uint32(num_buckets)).astype(jnp.int32)


def _partition_spec_of(x: jax.Array) -> P | None:
    sharding = jax.typeof(x).sharding if isinstance(x, core.Tracer) else x.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    return None


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer. Defaults mirror the d512 MoE rung."""

    vocab_size: int = 16384
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 256
    num_shared_experts: int = 2
    num_experts: int = 384
    num_experts_per_token: int = 8
    # LatentMoE (arXiv 2601.18089); latent RMSNorm per issue #6822.
    latent_dim: int | None = 256
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 1
    local_kv_heads: int | None = 1
    global_kv_heads: int | None = 1
    head_dim: int | None = 128
    max_seq_len: int = 4096
    sliding_window: int = 2048
    global_every: int = 4
    global_layers: tuple[int, ...] | None = None
    """Explicit 0-indexed global (softmax / MLA) layers, overriding ``global_every``."""
    capacity_factor: float = 1.15
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    # QK-norm: non-parametric RMS norm on per-head q/k of the softmax-attention layers.
    qk_norm: bool = True
    sconv: bool = True
    sconv_kernel: int = 4
    sconv_sites: tuple[str, ...] = ("k", "attn", "mlp")
    pooled_transport_capacity_factor: float | None = 1.15
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    # Dense (no-MoE) mode: every block is a single DenseMLP(hidden, intermediate_dim) SwiGLU; the MoE fields are ignored.
    dense_mlp: bool = False
    # Inkling (Thinking Machines) relative-position bias in place of RoPE on the softmax layers: a
    # per-head, content-dependent term added to the pre-softmax logits (see ``InklingRelPos``). One
    # extent for all layers (the stacked layers share one bias bank).
    inkling_relpos: bool = False
    rel_dim: int = 16
    rel_extent: int = 1024
    # MLA (DeepSeek-V2, arXiv 2405.04434) on the softmax layers, KV compression only: k/v are
    # up-projected from a learnable-RMSNormed ``mla_kv_latent_dim`` latent shared by all heads; q is
    # full rank. No decoupled RoPE (requires ``inkling_relpos``); heads are ``head_dim`` wide.
    mla: bool = False
    mla_kv_latent_dim: int = 512
    local_mixer: LocalMixer = LocalMixer.SLIDING_WINDOW
    kda_dt_range: tuple[float, float] = (0.02, 0.5)
    """KDA ``dt_bias`` init: the per-token log-decay ``|g|`` at zero gate input is log-uniform in this range."""
    kda_save_chunk_states: bool = False
    """Keep the fused KDA state pass's per-chunk states for backward instead of re-running the pass
    (same values; ~0.7 GB per KDA layer at d512's per-GPU batch)."""
    # Block Attention Residuals (Kimi, arXiv 2603.15031): each sublayer's input is a per-token softmax
    # attention over the residual-block history (the embedding, completed block sums and the running
    # partial), scored by a learned per-sublayer pseudo-query against the RMS-normalized sources.
    attn_res: bool = False
    attn_res_num_blocks: int = 8
    attn_res_layer_backward: AttnResLayerBackward = AttnResLayerBackward.RECOMPUTE
    attn_res_remat_attention: bool = False
    """Rematerialize the attention branch inside each AttnRes layer's backward, so its residuals (incl.
    the ``[B, H, S, W]`` Inkling bias) are never alive during the MLP backward. Costs one extra
    attention forward per layer; needed for memory at d1280."""
    value_embeds: "ValueEmbeds" = dataclasses.field(default_factory=lambda: ValueEmbeds.NONE)
    """Per-MLA-layer value-embedding table added to v (see ``ValueEmbeds``)."""
    sublayer_scales: bool = False
    """A learnable scalar (init 1) on every attention and MLP sublayer output, before it enters the
    AttnRes history."""
    router_combine: "RouterCombine" = dataclasses.field(default_factory=lambda: RouterCombine.SIGMOID_RENORM)
    routing_renorm_sum: float = 2.5
    """Total combine weight of a token's K routed experts (``RouterCombine``)."""
    latent_out_norm: bool = False
    """Kimi K3 normalized LatentMoE: a learnable RMSNorm on the combined routed output before ``W_latent_up``."""
    proj_biases: tuple[str, ...] = ()
    """Zero-init learnable biases at these sites: ``qkv`` (KDA q/k/v and MLA q / KV-latent projections),
    ``attn_out`` (the attention sublayer output) and ``mlp_out`` (the MoE sublayer output)."""
    qb_freeze_step: int | None = None
    """Stop updating the QB router biases from this step on (they keep their last value)."""
    embed_gated_norm: bool = True
    """GatedNorm after the embedding RMSNorm (else the RMSNorm alone)."""
    final_gated_norm: bool = True
    """GatedNorm after the final RMSNorm, before the lm_head (else the RMSNorm alone)."""
    mtp_weight: float = 0.0
    """Weight of a depth-1 multi-token-prediction loss (0 disables it): predict token t+2 from
    ``h_t + W_mtp rms_norm(embed(token_{t+1}))`` through a parameter-free RMS norm and the shared lm_head."""
    kda_no_decay_layers: tuple[int, ...] = ()
    """KDA layers (0-indexed model layers) run without decay (g = 0: the state never forgets)."""
    kda_no_beta_layers: tuple[int, ...] = ()
    """KDA layers run without a learned write strength (beta = 1)."""
    kda_decay_conv: bool = False
    """A causal ShortConv over KDA's rank-128 decay input (``x W_a↓``), so each position's decay sees a few
    preceding tokens instead of only its own (still computed before the recurrence, so chunking holds)."""
    kda_decay_per_head: bool = False
    """Gated-DeltaNet-style decay: one log-decay per head (broadcast over its channels) instead of KDA's
    per-channel decay. ``W_a↑`` and ``dt_bias`` shrink to one column per head."""
    kda_beta_rank: int | None = None
    """KDA write strength through a low-rank MLP of this hidden width,
    ``beta = sigmoid(SiLU(x W_beta_down) W_beta_up)``, instead of the linear ``sigmoid(x W_beta)``."""
    kda_beta_up_zero_init: bool = False
    """Zero-init ``W_β↑`` (β = 0.5 everywhere at init)."""
    kda_gate_per_head: bool = False
    """KDA output gate ``sigmoid(x W_g)`` with one value per head instead of one per channel."""
    loop_passes: int = 1
    """Apply the whole layer stack this many times per forward (tied weights). Every pass keeps
    extending the AttnRes history and has its own gate queries; each extra pass starts its running
    partial from ``loop_inject_scale * embedding`` (input injection)."""
    loop_grow_step: int | None = None
    """Run one pass before this step and ``loop_passes`` from it on (looped growth); None: always loop."""
    attn_res_logit_bias: bool = False
    """A learnable zero-init bias per (gate, source) on the AttnRes logits (Adam at the query LR)."""
    attn_res_z_loss: float = 0.0
    """Weight of a z-loss (mean squared logsumexp) on every AttnRes gate's logits; needs layer backward SAVE."""
    attn_res_pull: bool = False
    """'Pull' AttnRes: each source has a static learned key (zero-init; the partial has its own), and the
    query is the gate's current residual stream (the RMS-normed plain sum of its visible sources)."""
    attn_res_pull_embed: bool = False
    """Hybrid: standard (push) AttnRes, but the embedding source's logit is a learned linear projection of
    the gate's current residual stream (one zero-init vector per gate)."""
    attn_res_mask_attn_for: tuple[int, ...] = ()
    """Full AttnRes only: layers whose attention gate cannot read any earlier attention output."""
    attn_res_final_mode: str = "attn"
    """Final AttnRes gate: ``attn`` (learned query), ``uniform`` (query fixed at 0: a plain average of all
    sources) or ``sum`` (a straight sum of all sources, like a standard residual stream)."""
    attn_res_sum_inputs: tuple[str, ...] = ()
    """Full AttnRes only: components that read the straight sum of the visible sources instead of their
    AttnRes mix: any of ``q``, ``k``, ``v`` (those attention projections) and ``mlp`` (the MoE input)."""
    attn_res_full: bool = False
    """Full (not Block) AttnRes: every attention and MoE sublayer output is its own source. Needs
    attn_res_layer_backward=SAVE."""
    mla_share_kv_latent: bool = False
    """Every MLA layer after the first reuses the first MLA layer's normed KV latent (own W_uk / W_uv, so
    absorption still works), halving the MLA KV cache at d512. Needs attn_res_layer_backward=SAVE."""
    learnable_qk_mult: bool = False
    """A learnable scalar per softmax-attention layer (init ``qk_mult``) in place of the fixed ``qk_mult``."""
    aux_lm_layer: int | None = None
    """Early auxiliary LM loss: the residual stream after this layer (the plain sum of the AttnRes
    sources) goes through a parameter-free RMS norm and the shared lm_head. None disables it."""
    aux_lm_weight: float = 1.0
    """Weight of the auxiliary loss at step 0; annealed linearly to 0 at ``aux_lm_steps``."""
    aux_lm_steps: int = 500
    second_embed: bool = False
    second_embed_bigram: bool = False
    """Index the second table by a hash of (previous token, token) instead of the token: a bigram
    embedding with ``vocab_size`` hashed rows (the previous token is a sentinel at document starts)."""
    """A second, independently initialized token-embedding table, RMS-normed, as an extra AttnRes source."""

    def __post_init__(self) -> None:
        if not self.dense_mlp and self.num_experts_per_token >= self.num_experts:
            # QB routing takes top-(k+1) and keeps the last entry as the threshold alpha, so a
            # full-bank top-k asks `jax.lax.top_k` for more entries than the router has experts.
            raise ValueError("num_experts_per_token must be < num_experts, because QB routing selects top-(k+1)")
        if self.local_mixer == LocalMixer.KDA and not self.attn_res:
            raise ValueError("local_mixer=kda requires attn_res (KDA layers run in the unrolled AttnRes loop)")

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def model_type(self) -> type["Transformer"]:
        return Transformer

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    @property
    def stored_kv_heads(self) -> int:
        if self.local_kv_heads is None or self.global_kv_heads is None:
            return self.num_kv_heads
        return max(self.local_kv_heads, self.global_kv_heads)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "Transformer":
        cfg = self if Vocab.size == self.vocab_size else dataclasses.replace(self, vocab_size=Vocab.size)
        return Transformer.init(cfg, key=key)


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Non-parametric RMS norm over the last dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class ShortConv(eqx.Module):
    """Depthwise causal 1-D convolution over the sequence axis (Inkling-style SConv).

    A kernel of ``W`` taps mixes each channel with its own ``W-1`` causal predecessors,
    ``out[t] = sum_{lag} weight[lag] * x[t-lag]``, independently per channel. Identity-init
    (``weight[0]=1``, later taps 0) makes it a pass-through at step 0. Weights are tiny (``W*C``) and
    routed to Adam. Shard-local -- no cross-channel or cross-shard dependency, so no collectives.

    The body dispatches to ``levanter.kernels.pallas.short_conv``, which selects a fused Pallas
    kernel on GPU and the pad-and-shift weighted sum everywhere else; see that module's docstring.
    """

    weight: Float[Array, "W C"]
    kernel_size: int = eqx.field(static=True)

    @staticmethod
    def init(channels: int, kernel_size: int) -> "ShortConv":
        weight = jnp.zeros((kernel_size, channels)).at[0].set(1.0)
        # FSDP-shard the channel dim so the grad reduce-scatters instead of all-reducing; the
        # forward gathers the weight back to replicated.
        return ShortConv(weight=reshard(weight, P(None, _FSDP_AXES)), kernel_size=kernel_size)

    def __call__(self, x: Float[Array, "B S C"], segment_ids: Int[Array, "B S"] | None = None) -> Float[Array, "B S C"]:
        # segment_ids zero any tap reaching into a previous document, so the conv never crosses a boundary.
        weight = reshard(self.weight, P(None, None))
        return short_conv(weight, x, segment_ids, batch_axes=_BATCH_AXES)


class InklingRelPos(eqx.Module):
    """Inkling (Thinking Machines) relative-position bias: a per-head, content-dependent term added
    to the pre-softmax attention logits, in place of RoPE. ``r_proj`` maps the residual to a per-head
    relative feature R (``rel_dim`` per head); the shared bank ``proj`` [rel_dim, rel_extent] turns it
    into one bias value per query-key distance, bias[b,h,i,j] = R[b,i,h,:]·proj[:,i-j].
    ``__call__`` returns it in the banded layout of ``levanter.grug.attention.inkling_rel_bias``."""

    r_proj: jax.Array  # [D, num_heads * rel_dim]
    proj: jax.Array  # [rel_dim, rel_extent], shared across heads
    rel_dim: int = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "InklingRelPos":
        k_r, k_p = random.split(key, 2)
        d, n, r = cfg.hidden_dim, cfg.num_heads, cfg.rel_dim
        return InklingRelPos(
            r_proj=reshard(_init_weight(k_r, (d, n * r), cfg.initializer_std), P(_FSDP_AXES, "model")),
            # Scaled so the bias is O(1) at init.
            proj=reshard(_init_weight(k_p, (r, cfg.rel_extent), 1.0 / (r**0.5)), P(None, None)),
            rel_dim=r,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B H S W"]:
        relative_states = jnp.einsum("bsd,dk->bsk", x, self.r_proj.astype(x.dtype))
        relative_states = rearrange(relative_states, "b s (h r) -> b h s r", r=self.rel_dim)
        # Banded (key-aligned) bias layout consumed directly by the fused attention kernels; heads keep
        # the model-axis sharding.
        return inkling_rel_bias(
            relative_states,
            self.proj.astype(x.dtype),
            out_sharding=P(_BATCH_AXES, "model", None, None),
        )


class CausalSelfAttention(eqx.Module):
    """Softmax attention: GQA (``w_q``/``w_k``/``w_v``), or MLA with a compressed KV latent
    (``w_q``/``w_dkv``/``w_uk``/``w_uv``); either with half-RoPE or the Inkling bias."""

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"] | None
    w_v: Float[Array, "D MH"] | None
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    sconv_k: "ShortConv | None"  # SConv after the K projection (cfg.sconv)
    sconv_q: "ShortConv | None"  # MLA only: SConv after the q projection ("q" in cfg.sconv_sites)
    rel_pos: "InklingRelPos | None"  # Inkling relative-position bias (replaces RoPE when set)
    w_dkv: Float[Array, "D L"] | None
    kv_latent_norm: "RMSNorm | None"
    w_uk: Float[Array, "L NH"] | None
    w_uv: Float[Array, "L NH"] | None
    value_embed: Float[Array, "V NH"] | None
    ve_lambda: Float[Array, " 2"] | None  # (lambda1 on v, lambda2 on the value embedding)
    ve_gate: Float[Array, "D N"] | None
    qk_mult: Float[Array, ""] | None  # learnable logit scale (cfg.learnable_qk_mult); else cfg.qk_mult
    bias_q: Float[Array, " NH"] | None
    bias_dkv: Float[Array, " L"] | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        std = cfg.initializer_std
        attn_gate = reshard(jnp.zeros((d, n)), P(None, None))
        if cfg.mla:
            k_q, k_dkv, k_uk, k_uv, k_o, k_rel, k_ve = random.split(key, 7)
            kvl = cfg.mla_kv_latent_dim
            use_ve = cfg.value_embeds != ValueEmbeds.NONE
            return CausalSelfAttention(
                w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
                w_k=None,
                w_v=None,
                w_o=reshard(_init_weight(k_o, (n * h, d), std), P("model", _FSDP_AXES)),
                attn_gate=attn_gate,
                sconv_k=(ShortConv.init(n * h, cfg.sconv_kernel) if cfg.sconv and "k" in cfg.sconv_sites else None),
                sconv_q=(ShortConv.init(n * h, cfg.sconv_kernel) if cfg.sconv and "q" in cfg.sconv_sites else None),
                # Without Inkling the MLA layers are NoPE (they are global, so RoPE is disabled there).
                rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
                w_dkv=reshard(_init_weight(k_dkv, (d, kvl), std), P(_FSDP_AXES, None)),
                kv_latent_norm=RMSNorm.init(kvl, cfg.layer_norm_eps),
                w_uk=reshard(_init_weight(k_uk, (kvl, n * h), std), P(None, "model")),
                w_uv=reshard(_init_weight(k_uv, (kvl, n * h), std), P(None, "model")),
                value_embed=(
                    reshard(_init_weight(k_ve, (cfg.vocab_size, n * h), std), P(None, None)) if use_ve else None
                ),
                ve_lambda=jnp.array([1.0, 0.0]) if use_ve else None,
                ve_gate=(reshard(jnp.zeros((d, n)), P(None, None)) if cfg.value_embeds == ValueEmbeds.GATED else None),
                qk_mult=jnp.asarray(cfg.qk_mult, jnp.float32) if cfg.learnable_qk_mult else None,
                bias_q=jnp.zeros((n * h,)) if "qkv" in cfg.proj_biases else None,
                bias_dkv=jnp.zeros((kvl,)) if "qkv" in cfg.proj_biases else None,
                cfg=cfg,
            )
        if "qkv" in cfg.proj_biases:
            raise ValueError("proj_biases 'qkv' is implemented for MLA and KDA only")
        k_q, k_k, k_v, k_o, k_rel = random.split(key, 5)
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std), P("model", _FSDP_AXES)),
            attn_gate=attn_gate,
            sconv_k=(ShortConv.init(m * h, cfg.sconv_kernel) if cfg.sconv and "k" in cfg.sconv_sites else None),
            sconv_q=None,
            rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
            w_dkv=None,
            kv_latent_norm=None,
            w_uk=None,
            w_uv=None,
            value_embed=None,
            ve_lambda=None,
            ve_gate=None,
            qk_mult=jnp.asarray(cfg.qk_mult, jnp.float32) if cfg.learnable_qk_mult else None,
            bias_q=None,
            bias_dkv=None,
            cfg=cfg,
        )

    def _mla_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        token_ids: Int[Array, "B S"] | None,
        kv_share: dict[str, jax.Array] | None = None,
        proj_inputs: dict[str, jax.Array] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """MLA with a compressed KV latent: full-rank q; k and v up-projected from one normed latent."""
        assert self.w_dkv is not None and self.kv_latent_norm is not None
        head_dim = self.cfg.inferred_head_dim
        proj_inputs = proj_inputs or {}
        q_flat = jnp.einsum("bsh,hd->bsd", proj_inputs.get("q", x), self.w_q)
        latent = jnp.einsum("bsh,hl->bsl", x, self.w_dkv)
        if self.bias_q is not None and self.bias_dkv is not None:
            q_flat = q_flat + unshard(self.bias_q).astype(x.dtype)
            latent = latent + unshard(self.bias_dkv).astype(x.dtype)
        if self.sconv_q is not None:
            q_flat = self.sconv_q(q_flat, sconv_segment_ids)
        q = rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)
        if kv_share is not None and "latent" in kv_share:
            kv_latent = kv_share["latent"]
        else:
            kv_latent = self.kv_latent_norm(latent)
            if kv_share is not None:
                kv_share["latent"] = kv_latent
        # k / v may read a different stream than the shared latent (attn_res_sum_inputs); each then gets its
        # own latent from that stream (same W_dkv and latent norm).
        k_latent = (
            kv_latent
            if "k" not in proj_inputs
            else self.kv_latent_norm(jnp.einsum("bsh,hl->bsl", proj_inputs["k"], self.w_dkv))
        )
        v_latent = (
            kv_latent
            if "v" not in proj_inputs
            else self.kv_latent_norm(jnp.einsum("bsh,hl->bsl", proj_inputs["v"], self.w_dkv))
        )
        k_flat = jnp.einsum("bsl,ld->bsd", k_latent, self.w_uk)
        if self.sconv_k is not None:
            k_flat = self.sconv_k(k_flat, sconv_segment_ids)
        k = rearrange(k_flat, "... (n d) -> ... n d", d=head_dim)
        v = rearrange(jnp.einsum("bsl,ld->bsd", v_latent, self.w_uv), "... (n d) -> ... n d", d=head_dim)
        if self.value_embed is not None:
            assert self.ve_lambda is not None and token_ids is not None
            ve = _embedding_gather(self.value_embed.astype(x.dtype), token_ids)
            ve = rearrange(ve, "... (n d) -> ... n d", d=head_dim)
            lam = self.ve_lambda.astype(x.dtype)
            if self.ve_gate is not None:
                ve_weight = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.ve_gate.astype(x.dtype)))[..., None]
            else:
                ve_weight = lam[1]
            v = lam[0] * v + ve_weight * reshard(ve, _partition_spec_of(v) or P(_BATCH_AXES, None, None, None))
        return q, k, v

    def _gqa_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        is_global: bool | jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.w_k is not None and self.w_v is not None
        head_dim = self.cfg.inferred_head_dim
        q_flat = jnp.einsum("bsh,hd->bsd", x, self.w_q)
        k_flat = jnp.einsum("bsh,hd->bsd", x, self.w_k)
        v_flat = jnp.einsum("bsh,hd->bsd", x, self.w_v)
        # SConv: depthwise causal conv after the K projection.
        if self.sconv_k is not None:
            k_flat = self.sconv_k(k_flat, sconv_segment_ids)
        q = rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)
        k = rearrange(k_flat, "... (m d) -> ... m d", d=head_dim)
        v = rearrange(v_flat, "... (m d) -> ... m d", d=head_dim)

        if self.cfg.local_kv_heads is not None and self.cfg.global_kv_heads is not None:
            stored_kv_heads = self.cfg.stored_kv_heads

            # Both lax.cond branches must match sharding, not just shape: reshard both to the same replicated head spec.
            kv_spec = P(_BATCH_AXES, None, None, None)

            def _logical_kv(projection: jax.Array, num_kv_heads: int) -> jax.Array:
                # Replicate before slicing, not after: narrowing a `model`-sharded head axis to a
                # count that does not divide the mesh axis is unsupported.
                projection = reshard(projection, kv_spec)
                if num_kv_heads == stored_kv_heads:
                    return projection
                return align_kv_heads(projection[:, :, :num_kv_heads, :], num_q_heads=stored_kv_heads)

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
        return q, k, v

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
        token_ids: Int[Array, "B S"] | None = None,
        kv_share: dict[str, jax.Array] | None = None,
        proj_inputs: dict[str, jax.Array] | None = None,
    ) -> Float[Array, "B S D"]:
        """``kv_share`` (MLA only) is a per-forward mailbox for ``mla_share_kv_latent``; ``proj_inputs``
        optionally replaces the input of the ``q`` / ``k`` / ``v`` projections."""
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()
        # segment_ids (packed-document boundaries) come from the mask so the SConv never mixes across a
        # document boundary.
        sconv_segment_ids = _sconv_segment_ids(mask)
        if self.cfg.mla:
            q, k, v = self._mla_qkv(x, sconv_segment_ids, token_ids, kv_share, proj_inputs)
        else:
            q, k, v = self._gqa_qkv(x, sconv_segment_ids, is_global)

        if self.cfg.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)

        # Half-RoPE: rotate only the first half of Q/K head_dim; disable_rope skips RoPE on long/global layers.
        def _rope(qh: jax.Array, kh: jax.Array) -> tuple[jax.Array, jax.Array]:
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                qh[..., :half], kh[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            return (
                jnp.concatenate([q_rot, qh[..., half:]], axis=-1),
                jnp.concatenate([k_rot, kh[..., half:]], axis=-1),
            )

        # The Inkling bias replaces RoPE: no rotation, a per-head content-dependent bias (from x) on the
        # pre-softmax logits instead.
        rel_bias = None
        if self.rel_pos is not None:
            rel_bias = self.rel_pos(x)
        elif isinstance(disable_rope, bool):
            if not disable_rope:
                q, k = _rope(q, k)
        else:
            q_roped, k_roped = _rope(q, k)
            keep = ~jnp.asarray(disable_rope, dtype=jnp.bool_)
            q = jnp.where(keep, q_roped, q)
            k = jnp.where(keep, k_roped, k)
        q = q * (self.cfg.qk_mult if self.qk_mult is None else self.qk_mult.astype(q.dtype))
        # The fa4-cute kernel is GPU-only; fall back to auto-select off-GPU so the model still lowers
        # on CPU (e.g. the grug variant-contract tests).
        attn_impl = "gpu_fa4_cute" if jax.default_backend() == "gpu" else None
        attn_out = attention(q, k, v, mask, implementation=attn_impl, rel_bias=rel_bias)
        # Exclusive Self Attention (XSA): subtract the component of yᵢ parallel to vᵢ, per head.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ.
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        # GPU XSA with GQA can give attn_out a backend-specific head sharding;
        # match v to that dynamic sharding before the per-head projection math.
        aligned_v = reshard(aligned_v, _partition_spec_of(attn_out) or P(_BATCH_AXES, None, None, "model"))
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: sigmoid(x @ attn_gate) produces one scalar per head.
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        # Merge heads into hidden dim while keeping model-axis sharding for w_o.
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:-2], attn_out.shape[-2] * attn_out.shape[-1]),
            out_sharding=P(_BATCH_AXES, None, "model"),
        )
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


def _kda_dt_bias_init(cfg: GrugModelConfig, key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
    """``dt_bias`` such that ``|g| = KDA_MIN_LOG_DECAY * sigmoid(dt_bias)`` (zero gate input, ``A_log = 0``)
    is log-uniform in ``kda_dt_range``."""
    lo, hi = cfg.kda_dt_range
    decay = jnp.exp(random.uniform(key, shape, minval=math.log(lo), maxval=math.log(hi)))
    return jax.scipy.special.logit(decay / KDA_MIN_LOG_DECAY)


def _kda_kernel(q, k, v, g, beta, segment_ids=None, *, save_chunk_states: bool):
    """KDA on the model layout ``(B, S, H, d)``: the fused Pallas kernels on GPU, else the XLA
    ``chunk_kda`` (heads-first layout)."""
    if jax.default_backend() == "gpu":
        return kda_fused(
            q, k, v, g, beta, segment_ids=segment_ids, chunk_size=KDA_CHUNK_SIZE, save_chunk_states=save_chunk_states
        )
    q, k, v, g, beta = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta))
    seg = None if segment_ids is None else segment_ids[:, None, :]  # same documents for every head
    return jnp.swapaxes(chunk_kda(q, k, v, g, beta, chunk_size=KDA_CHUNK_SIZE, segment_ids=seg)[0], 1, 2)


class KimiDeltaAttention(eqx.Module):
    """KDA linear-attention token mixer for the local layers, following Kimi K3's KDA layer.

    Per head (``N`` heads of width ``h``): q/k/v are bias-free ``D -> N*h`` projections (no GQA), each
    through a depthwise causal ShortConv + SiLU; q/k are L2-normalized in the kernel (q scaled by
    ``1/sqrt(h)``). ``beta = sigmoid(x W_beta)`` per head, and the per-channel log-decay
    ``g = -5 * sigmoid(exp(A_log) * (x W_a_down W_a_up + dt_bias))`` lies in ``(-5, 0)``. The state
    follows ``S_t = (I - beta k k^T) Diag(exp(g)) S_{t-1} + beta k v^T``, read as ``o_t = S_t^T q_t``,
    and is hard-reset at packed-document starts. The output gets a per-head RMSNorm with a learnable
    ``h``-dim scale shared across heads, a full-rank per-channel gate ``sigmoid(x W_g)``, and ``w_o``.
    No positional encoding and no window. The recurrence runs under ``shard_map`` (batch on the batch
    axes, heads on ``model``).
    """

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D NH"]
    w_v: Float[Array, "D NH"]
    w_o: Float[Array, "NH D"]
    w_g: Float[Array, "D NH"]  # [D, N] with cfg.kda_gate_per_head
    w_a_down: Float[Array, "D R"]
    w_a_up: Float[Array, "R NH"]
    a_log: Float[Array, " N"]
    dt_bias: Float[Array, "N H"]
    w_beta: Float[Array, "D N"] | None
    w_beta_down: Float[Array, "D R"] | None
    w_beta_up: Float[Array, "R N"] | None
    o_norm: "RMSNorm"
    bias_qkv: Float[Array, "3 NH"] | None
    sconv_q: ShortConv
    sconv_k: ShortConv
    sconv_v: ShortConv
    sconv_a: ShortConv | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KimiDeltaAttention":
        k_q, k_k, k_v, k_o, k_g, k_ad, k_au, k_b, k_dt = random.split(key, 9)
        d, n, h, r, std = cfg.hidden_dim, cfg.num_heads, cfg.inferred_head_dim, _KDA_GATE_RANK, cfg.initializer_std
        return KimiDeltaAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std), P("model", _FSDP_AXES)),
            w_g=reshard(
                _init_weight(k_g, (d, n if cfg.kda_gate_per_head else n * h), std),
                P(None, None) if cfg.kda_gate_per_head else P(_FSDP_AXES, "model"),
            ),
            w_a_down=reshard(_init_weight(k_ad, (d, r), std), P(_FSDP_AXES, None)),
            w_a_up=reshard(
                _init_weight(k_au, (r, n if cfg.kda_decay_per_head else n * h), std),
                P(None, None) if cfg.kda_decay_per_head else P(None, "model"),
            ),
            a_log=jnp.zeros((n,)),
            dt_bias=_kda_dt_bias_init(cfg, k_dt, (n, 1) if cfg.kda_decay_per_head else (n, h)),
            w_beta=None if cfg.kda_beta_rank else reshard(_init_weight(k_b, (d, n), std), P(None, None)),
            w_beta_down=(
                reshard(_init_weight(k_b, (d, cfg.kda_beta_rank), std), P(None, None)) if cfg.kda_beta_rank else None
            ),
            w_beta_up=(
                None
                if not cfg.kda_beta_rank
                else reshard(
                    (
                        jnp.zeros((cfg.kda_beta_rank, n))
                        if cfg.kda_beta_up_zero_init
                        else _init_weight(
                            random.fold_in(k_b, 1), (cfg.kda_beta_rank, n), 1.0 / math.sqrt(cfg.kda_beta_rank)
                        )
                    ),
                    P(None, None),
                )
            ),
            o_norm=RMSNorm.init(h, 1e-6),
            bias_qkv=jnp.zeros((3, n * h)) if "qkv" in cfg.proj_biases else None,
            sconv_q=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_k=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_v=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_a=ShortConv.init(r, cfg.sconv_kernel) if cfg.kda_decay_conv else None,
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        segment_ids: Int[Array, "B S"] | None = None,
        proj_inputs: dict[str, jax.Array] | None = None,
        no_decay: bool = False,
        no_beta: bool = False,
    ) -> Float[Array, "B S D"]:
        """``proj_inputs`` optionally replaces the input of the ``q`` / ``k`` / ``v`` projections;
        ``no_decay`` / ``no_beta`` (static, per layer) replace g with 0 / beta with 1."""
        cfg = self.cfg
        head_dim = cfg.inferred_head_dim
        b, s, _ = x.shape

        proj_inputs = proj_inputs or {}

        def project(w: jax.Array, conv: ShortConv, bias_row: int, name: str) -> jax.Array:
            y = jnp.einsum("bsh,hd->bsd", proj_inputs.get(name, x), w)
            if self.bias_qkv is not None:
                y = y + unshard(self.bias_qkv[bias_row]).astype(x.dtype)
            y = jax.nn.silu(conv(y, segment_ids))
            return rearrange(y, "... (n d) -> ... n d", d=head_dim)

        q = project(self.w_q, self.sconv_q, 0, "q")
        k = project(self.w_k, self.sconv_k, 1, "k")
        v = project(self.w_v, self.sconv_v, 2, "v")
        a_low = jnp.einsum("bsd,dr->bsr", x, self.w_a_down)
        if self.sconv_a is not None:
            a_low = self.sconv_a(a_low, segment_ids)
        a = jnp.einsum("bsr,re->bse", a_low, self.w_a_up)
        a = rearrange(a, "... (n d) -> ... n d", d=1 if cfg.kda_decay_per_head else head_dim).astype(jnp.float32)
        scale = jnp.exp(self.a_log.astype(jnp.float32))[:, None]
        g = -KDA_MIN_LOG_DECAY * jax.nn.sigmoid(scale * (a + self.dt_bias.astype(jnp.float32)))
        if cfg.kda_decay_per_head:
            g = jnp.broadcast_to(g, (*g.shape[:-1], head_dim))
        if self.w_beta_down is not None and self.w_beta_up is not None:
            beta_hidden = jax.nn.silu(jnp.einsum("bsd,dr->bsr", x, self.w_beta_down))
            beta_logits = jnp.einsum("bsr,rn->bsn", beta_hidden, self.w_beta_up.astype(beta_hidden.dtype))
        else:
            assert self.w_beta is not None
            beta_logits = jnp.einsum("bsd,dn->bsn", x, self.w_beta)
        beta = jax.nn.sigmoid(beta_logits.astype(jnp.float32))
        if no_decay:
            g = jnp.zeros_like(g)
        if no_beta:
            beta = jnp.ones_like(beta)

        spec4 = P(_BATCH_AXES, None, "model", None)
        spec3 = P(_BATCH_AXES, None, "model")
        q, k, v, g = (reshard(t, spec4) for t in (q, k, v, g))
        beta = reshard(beta, spec3)
        run = functools.partial(_kda_kernel, save_chunk_states=cfg.kda_save_chunk_states)
        args = (q, k, v, g, beta)
        in_specs = (spec4,) * 4 + (spec3,)
        if segment_ids is not None:
            args += (reshard(jnp.broadcast_to(segment_ids, (b, s)), P(_BATCH_AXES, None)),)
            in_specs += (P(_BATCH_AXES, None),)
        # The Pallas custom VJPs are not vma-annotated, so skip the varying-axes check.
        o = jax.shard_map(run, mesh=get_abstract_mesh(), in_specs=in_specs, out_specs=spec4, check_vma=False)(*args)
        o = self.o_norm(o.astype(x.dtype))
        o = jnp.reshape(o, (b, s, cfg.num_heads * head_dim), out_sharding=P(_BATCH_AXES, None, "model"))
        gate = jax.nn.sigmoid(jnp.einsum("bsd,de->bse", x, self.w_g))
        if cfg.kda_gate_per_head:
            gate = jnp.repeat(gate, head_dim, axis=-1, total_repeat_length=cfg.num_heads * head_dim)
        o = o * gate
        return jnp.einsum("bsh,hd->bsd", o, self.w_o, out_sharding=_batch_spec())


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating. Compensates for AdamH's bounded activation norms.
    See https://arxiv.org/abs/2601.22966v1"""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "GatedNorm":
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, _GATED_NORM_RANK), initializer_std), P(None, None)),
            w_up=reshard(_init_weight(k_up, (_GATED_NORM_RANK, hidden_dim), initializer_std), P(None, None)),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


class DenseMLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "DenseMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        return DenseMLP(
            w_gate=reshard(
                _init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P(_FSDP_AXES, "model")
            ),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P(_FSDP_AXES, "model")),
            w_down=reshard(
                _init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std), P("model", _FSDP_AXES)
            ),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        *,
        activation: MoeActivation = ActivationFunctionEnum.silu,
        moe_output_reshard: bool = True,
    ) -> Float[Array, "B S D"]:
        if isinstance(activation, ActivationFunctionEnum):
            activation_fn = activation.to_jax_fn()
        else:
            activation_fn = activation

        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        if moe_output_reshard:
            # Force the fused token sharding on the shared-expert output so the residual add matches on multi-node.
            out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
            return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))
        # Dense path: pin the token axis to the batch spec so XLA can infer w_down's output sharding.
        out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


def _bincount_upper_quantile(
    s_local: jax.Array,
    *,
    num_experts: int,
    n_bins: int,
    lo: jax.Array,
    hi: jax.Array,
    target_rank: float,
) -> jax.Array:
    """Per-expert (1-K/E) upper quantile of ``s_local`` via one fused bincount over ``[lo, hi]``.

    Runs inside a ``shard_map``: a single ``jnp.bincount`` over an expert-major flat index
    (``expert*n_bins + bin``, clip-to-edge) builds the local per-expert histogram, one integer ``psum``
    pools it globally, and beta is read from the top-cumulative counts, interpolated in the crossing bin.
    """
    bin_width = (hi - lo) / n_bins
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)[None, :]
    idx = jnp.clip(((s_local - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1)
    flat = (expert_ids * n_bins + idx).reshape(-1)
    local_counts = jnp.bincount(flat, length=num_experts * n_bins).reshape(num_experts, n_bins)
    counts = jax.lax.psum(local_counts, axis_name=_BATCH_AXES).astype(jnp.float32)
    cum_from_top = jnp.cumsum(counts[:, ::-1], axis=-1)[:, ::-1]  # #{margins in bins >= b}
    bstar = jnp.clip(jnp.sum((cum_from_top >= target_rank).astype(jnp.int32), axis=-1) - 1, 0, n_bins - 1)
    ct_b = jnp.take_along_axis(cum_from_top, bstar[:, None], axis=-1)[:, 0]
    h_b = jnp.take_along_axis(counts, bstar[:, None], axis=-1)[:, 0]
    lower_edge = lo + bstar.astype(jnp.float32) * bin_width
    return lower_edge + bin_width * (ct_b - target_rank) / jnp.maximum(h_b, 1.0)


def _qb_beta_hist(
    s_ma: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    *,
    num_experts_per_token: int,
    num_experts: int,
    n_bins: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global (1-K/E)-quantile of the logit margins over the live ``[min, max]`` grid (this step's).

    A ``pmin``/``pmax`` sets the grid to the exact current range of the margins, then
    ``_bincount_upper_quantile`` reads the per-expert threshold. Replaces the per-device ``top_k`` +
    ``pmean`` estimate with a smoother global quantile at the cost of the per-expert count reduction.

    Returns ``(beta, margin_min, margin_max)``: the per-expert threshold plus the live margin range
    (the grid ``lo``/``hi``), surfaced for logging.
    """
    target_rank = float(s_ma.shape[0]) * num_experts_per_token / num_experts  # tokens at/above beta per expert

    def _fn(s_local: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # pmin/pmax have no autodiff rule and the range is a control quantity, so detach their inputs;
        # the bincount path drops tangents at the integer bin cast, so it needs none downstream either.
        lo = jax.lax.pmin(jax.lax.stop_gradient(jnp.min(s_local)), axis_name=_BATCH_AXES)
        hi = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(s_local)), axis_name=_BATCH_AXES)
        hi_grid = jnp.maximum(hi, lo + 1e-6)  # guard a degenerate all-equal range
        beta = _bincount_upper_quantile(
            s_local, num_experts=num_experts, n_bins=n_bins, lo=lo, hi=hi_grid, target_rank=target_rank
        )
        return beta, lo, hi  # surface the live margin range for logging

    return shard_map(_fn, mesh=mesh, in_specs=(P(_BATCH_AXES, None),), out_specs=(P(), P(), P()))(s_ma)


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    w_latent_down: jax.Array | None
    latent_norm: RMSNorm | None
    w_latent_up: jax.Array | None
    latent_out_norm: RMSNorm | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
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
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            w_latent_down=(
                None
                if latent is None
                else reshard(_init_weight(k_down, (d, latent), cfg.initializer_std), P(_FSDP_AXES, "model"))
            ),
            latent_norm=None if latent is None else RMSNorm.init(latent, cfg.layer_norm_eps),
            w_latent_up=(
                None
                if latent is None
                else reshard(_init_weight(k_up, (latent, d), cfg.initializer_std), P("model", _FSDP_AXES))
            ),
            latent_out_norm=(
                RMSNorm.init(latent, cfg.layer_norm_eps) if latent is not None and cfg.latent_out_norm else None
            ),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=expert_width,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation="fixed_pooled_wave_all_to_all",
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
                pooled_transport_capacity_factor=cfg.pooled_transport_capacity_factor,
                expert_chunks=1,
                num_expert_waves=1,
            ),
            cfg=cfg,
        )

    def input_projection_weights(self, dtype: jnp.dtype) -> list[jax.Array]:
        """The ``[D, *]`` projections this MLP applies to its input: the router, then the latent down."""
        weights = [reshard(self.router, P(None, None))]
        if self.w_latent_down is not None:
            weights.append(reshard(self.w_latent_down.astype(dtype), P(None, None)))
        return weights

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        projected: list[jax.Array] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``projected`` holds ``x_flat @ w`` for each of ``input_projection_weights`` when the caller
        computed them already (fused with other projections of the same input)."""
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        if projected is None:
            projected = [jnp.einsum("td,de->te", x_flat, w) for w in self.input_projection_weights(x_flat.dtype)]
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = projected[0].astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        k = self.cfg.num_experts_per_token
        renorm_sum = self.cfg.routing_renorm_sum
        if self.cfg.router_combine == RouterCombine.SOFTMAX_RENORM:
            combine_weights_f = renorm_sum * jax.nn.softmax(unbiased_topk, axis=-1)
        elif self.cfg.router_combine == RouterCombine.SIGMOID_RAW:
            combine_weights_f = jax.nn.sigmoid(unbiased_topk) * (renorm_sum / (k / 2))
        else:
            combine_weights_f = jax.nn.sigmoid(unbiased_topk)
            denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
            combine_weights_f = combine_weights_f * (renorm_sum / (denom + 1e-9))
        combine_weights = combine_weights_f.astype(x.dtype)
        mesh = get_abstract_mesh()
        # Per-shard partials only; the cross-device reduction happens once after the layer scan.
        router_stats = local_routing_stats(
            reshard(selected_experts, P(_BATCH_AXES, None)),
            reshard(router_probs, P(_BATCH_AXES, None)),
            reshard(router_logits, P(_BATCH_AXES, None)),
            mesh,
            num_experts=self.cfg.num_experts,
            batch_axes=_BATCH_AXES,
        )
        # Sharded QB: estimate each expert's threshold beta from the margins `s - alpha` by binning
        # them into fixed bins over the live global range and reading the (1-K/E) quantile.
        s_minus_alpha = reshard(router_logits - qb_alpha, P(_BATCH_AXES, None))
        beta, margin_min, margin_max = _qb_beta_hist(
            s_minus_alpha,
            mesh,
            num_experts_per_token=self.cfg.num_experts_per_token,
            num_experts=self.cfg.num_experts,
            n_bins=_QB_HIST_BINS,
        )
        router_stats["qb_beta"] = beta
        router_stats["margin_min"] = margin_min
        router_stats["margin_max"] = margin_max

        routed_input = x_flat
        if self.w_latent_down is not None and self.latent_norm is not None:
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = self.latent_norm(reshard(projected[1], _batch_spec()))
        moe_out = self.expert_mlp(
            routed_input,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=True,
        )
        routed_flat, capacity_overflow = moe_out
        dropped_assignments = capacity_overflow.dropped
        sender_dropped_assignments = capacity_overflow.sender_dropped
        receiver_dropped_assignments = capacity_overflow.receiver_dropped
        router_stats["capacity_overflow"] = dropped_assignments
        router_stats["sender_capacity_overflow"] = sender_dropped_assignments
        router_stats["receiver_capacity_overflow"] = receiver_dropped_assignments

        # Expand after the combine: `expert_mlp` already returns the weight-summed expert output,
        # which is the vector the paper's W_up acts on.
        if self.latent_out_norm is not None:
            routed_flat = self.latent_out_norm(routed_flat)
        if self.w_latent_up is not None:
            routed_flat = jnp.einsum(
                "tl,ld->td",
                routed_flat,
                self.w_latent_up.astype(routed_flat.dtype),
                out_sharding=_batch_spec(),
            )

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


def moe_and_shared_fused(
    mlp: MoEMLP, shared: tuple[DenseMLP, ...], x: Float[Array, "B S D"]
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """Routed MoE plus the shared SwiGLU experts with every projection of ``x`` in one GEMM.

    The router, latent down-projection and each shared expert's gate/up read the same input, so they
    run as one ``[D, sum widths]`` GEMM; the shared experts' down-projections run as one GEMM over
    their concatenated hidden units (the sum over shared experts happens in its accumulator). Same
    math as ``mlp(x) + sum(expert(x) for expert in shared)`` (~3% faster at d512); parameters stay
    separate leaves.
    """
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    replicated = P(None, None)
    moe_weights = mlp.input_projection_weights(x_flat.dtype)
    shared_weights = [reshard(e.w_gate, replicated) for e in shared] + [reshard(e.w_up, replicated) for e in shared]
    weights = moe_weights + shared_weights
    fused = jnp.einsum("td,de->te", x_flat, jnp.concatenate(weights, axis=1), out_sharding=_batch_spec())
    parts = jnp.split(fused, list(itertools.accumulate(w.shape[1] for w in weights[:-1])), axis=1)
    routed, stats = mlp(x, projected=parts[: len(moe_weights)])
    gates, ups = parts[len(moe_weights) : len(moe_weights) + len(shared)], parts[len(moe_weights) + len(shared) :]
    hidden = jnp.concatenate([jax.nn.silu(g) * u for g, u in zip(gates, ups, strict=True)], axis=1)
    w_down = jnp.concatenate([reshard(e.w_down, replicated) for e in shared], axis=0)
    shared_out = jnp.einsum("tm,md->td", hidden, w_down, out_sharding=_batch_spec())
    return routed + _batch_reshard(rearrange(shared_out, "(b s) d -> b s d", b=b, s=s)), stats


def _sconv_segment_ids(mask: AttentionMask | jax.Array) -> jax.Array | None:
    """segment_ids (packed-document boundaries) for the SConvs and KDA; None when unpacked."""
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    return segment_ids[0] if segment_ids is not None else None


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention | KimiDeltaAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: "MoEMLP | DenseMLP"
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: "ShortConv | None"
    sconv_mlp: "ShortConv | None"
    # Block AttnRes pseudo-queries of the attention and MLP sublayers (None without cfg.attn_res).
    attn_res_query_attn: Float[Array, " D"] | None
    attn_res_query_mlp: Float[Array, " D"] | None
    # Learnable sublayer output scalars (None without cfg.sublayer_scales).
    attn_out_scale: Float[Array, ""] | None
    mlp_out_scale: Float[Array, ""] | None
    bias_attn_out: Float[Array, " D"] | None
    bias_mlp_out: Float[Array, " D"] | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, use_kda: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        attn = KimiDeltaAttention.init(cfg, key=attn_key) if use_kda else CausalSelfAttention.init(cfg, key=attn_key)
        # KDA blocks have no branch-output SConv (K3 has only the q/k/v convs).
        use_attn_sconv = cfg.sconv and "attn" in cfg.sconv_sites and not use_kda
        # Zero-init: every source scores 0, so each gate starts as a uniform average of its sources.
        attn_res_query = reshard(jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32), P(None)) if cfg.attn_res else None
        if cfg.dense_mlp:
            # Dense block: one SwiGLU DenseMLP(hidden, intermediate_dim), no MoE and no shared experts.
            mlp = DenseMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key)
            shared = None
        else:
            mlp = MoEMLP.init(cfg, key=mlp_key)
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
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=attn,
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=mlp,
            shared=shared,
            sconv_attn=(ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if use_attn_sconv else None),
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None
            ),
            attn_res_query_attn=attn_res_query,
            attn_res_query_mlp=attn_res_query,
            attn_out_scale=jnp.ones((), dtype=jnp.float32) if cfg.sublayer_scales else None,
            mlp_out_scale=jnp.ones((), dtype=jnp.float32) if cfg.sublayer_scales else None,
            bias_attn_out=jnp.zeros((cfg.hidden_dim,)) if "attn_out" in cfg.proj_biases else None,
            bias_mlp_out=jnp.zeros((cfg.hidden_dim,)) if "mlp_out" in cfg.proj_biases else None,
        )

    def attn_branch(
        self,
        h: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array,
        is_global: bool | jax.Array,
        token_ids: Int[Array, "B S"] | None = None,
        kv_share: dict[str, jax.Array] | None = None,
        sum_stream: Float[Array, "B S D"] | None = None,
        sum_components: tuple[str, ...] = (),
        kda_ablation: tuple[bool, bool] = (False, False),
    ) -> Float[Array, "B S D"]:
        """``sum_stream`` (with ``sum_components``) feeds those q/k/v projections from the straight-sum
        stream, through the same RMSNorm and GatedNorm, instead of the AttnRes mix ``h``."""
        attn_in = self.attn_gated_norm(self.rms_attn(h))
        proj_inputs = None
        attn_components = tuple(c for c in sum_components if c in ("q", "k", "v"))
        if attn_components:
            assert sum_stream is not None
            sum_in = self.attn_gated_norm(self.rms_attn(sum_stream))
            proj_inputs = {c: sum_in for c in attn_components}
        if isinstance(self.attn, KimiDeltaAttention):
            # KDA has no positional encoding or window; it only needs the document boundaries.
            out = self.attn(
                attn_in,
                _sconv_segment_ids(mask),
                proj_inputs=proj_inputs,
                no_decay=kda_ablation[0],
                no_beta=kda_ablation[1],
            )
        else:
            out = self.attn(
                attn_in,
                mask,
                disable_rope=disable_rope,
                is_global=is_global,
                token_ids=token_ids,
                kv_share=kv_share,
                proj_inputs=proj_inputs,
            )
        if self.bias_attn_out is not None:
            out = out + unshard(self.bias_attn_out).astype(out.dtype)
        if self.sconv_attn is not None:
            out = self.sconv_attn(out, _sconv_segment_ids(mask))
        if self.attn_out_scale is not None:
            out = out * self.attn_out_scale.astype(out.dtype)
        return out

    def mlp_branch(
        self, h: Float[Array, "B S D"], mask: AttentionMask | jax.Array
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        mlp_in = self.mlp_gated_norm(self.rms_mlp(h))
        stats: dict[str, jax.Array] = {}
        if isinstance(self.mlp, DenseMLP):
            out = self.mlp(mlp_in, moe_output_reshard=False)
        elif self.shared is not None:
            out, stats = moe_and_shared_fused(self.mlp, self.shared, mlp_in)
        else:
            out, stats = self.mlp(mlp_in)
        if self.bias_mlp_out is not None:
            out = out + unshard(self.bias_mlp_out).astype(out.dtype)
        if self.sconv_mlp is not None:
            out = self.sconv_mlp(out, _sconv_segment_ids(mask))
        if self.mlp_out_scale is not None:
            out = out * self.mlp_out_scale.astype(out.dtype)
        return out, stats

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        x = x + self.attn_branch(x, mask, disable_rope, is_global)
        mlp_out, router_stats = self.mlp_branch(x, mask)
        return x + mlp_out, router_stats


@named_call
def _attn_res_source_logits(
    source: Float[Array, "B S D"], queries: Float[Array, "G D"], eps: float
) -> Float[Array, "G B S"]:
    """Float32 AttnRes logits of one source against ``G`` queries: ``q_g . rms_norm(source)``.

    RMS normalization is a per-token scalar, so it is applied to the ``[G, B, S]`` dot products instead
    of materializing normalized keys; a completed block is read once for every gate that will ever see it.
    The key norm is parameter-free: a learnable gain would be redundant with the query.
    """
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(source.astype(jnp.float32)), axis=-1) + eps)
    dots = jnp.einsum("bsd,gd->gbs", source, queries.astype(source.dtype), preferred_element_type=jnp.float32)
    return dots * inv_rms[None]


def _block_logit(block_logits: jax.Array, queries: jax.Array, gate_index: int) -> jax.Array:
    """Gate ``gate_index``'s row of a block's logits, which cover the trailing queries of the stack."""
    return block_logits[gate_index - (queries.shape[0] - block_logits.shape[0])]


def _softmax_mix(logits: list[jax.Array], sources: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Per-token softmax weights ``[N, B, S]`` over ``sources`` and the weighted sum (in float32)."""
    weights = jax.nn.softmax(jnp.stack(logits), axis=0)
    mixed = weights[0][..., None] * sources[0].astype(jnp.float32)
    for weight, source in zip(weights[1:], sources[1:], strict=True):
        mixed = mixed + weight[..., None] * source.astype(jnp.float32)
    return weights, mixed


@named_call
def _attn_res_mix(
    blocks: tuple[jax.Array, ...],
    block_logits: tuple[jax.Array, ...],
    partial: Float[Array, "B S D"] | None,
    queries: Float[Array, "G D"],
    gate_index: int,
    eps: float,
    extras: dict[str, jax.Array | None] | None = None,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """One AttnRes gate: softmax over the completed blocks (+ the running partial) and their weighted sum.

    ``extras`` (``_gate_extras``) optionally adds per-(gate, source) biases and masks, pull keys and a
    pull embedding logit; column ``n`` is block ``n`` and the last column is the partial.
    Also returns the gate's mean squared logsumexp (the z-loss term) and its token-mean source weights
    (blocks in order, then the partial), for logging.

    ``block_logits[n]`` holds block ``n``'s precomputed logits against the queries of every gate from the
    first one that reads it to the end of the query stack (``_block_logit``); only the partial (new at
    each gate) is scored here. Only the valid sources are read -- no masked slots.
    """
    sources = list(blocks)
    logits = [_block_logit(bl, queries, gate_index) for bl in block_logits]
    if partial is not None:
        sources.append(partial)
        logits.append(_attn_res_source_logits(partial, queries[gate_index][None], eps)[0])
    logits = _bias_gate_logits(logits, extras, gate_index, sources, eps, has_partial=partial is not None)
    weights, mixed = _softmax_mix(logits, sources)
    mean_weights = jax.lax.stop_gradient(jnp.mean(weights, axis=(1, 2)))
    return reshard(mixed.astype(sources[0].dtype), _batch_spec()), _gate_z(logits), mean_weights


def _bias_gate_logits(
    logits: list[jax.Array],
    extras: dict[str, jax.Array | None] | None,
    gate_index: int,
    sources: list[jax.Array],
    eps: float,
    *,
    has_partial: bool,
) -> list[jax.Array]:
    """Apply ``extras`` to one gate's source logits (column n = block n, last column = the partial)."""
    if extras is None:
        return logits
    num_blocks = len(logits) - int(has_partial)
    columns = list(range(num_blocks)) + ([-1] if has_partial else [])
    pull_keys, embed_query = extras.get("pull_keys"), extras.get("embed_query")
    if pull_keys is not None or embed_query is not None:
        stream = sources[0].astype(jnp.float32)
        for src in sources[1:]:
            stream = stream + src.astype(jnp.float32)
        stream = rms_norm(stream, eps)
        if pull_keys is not None:
            logits = [jnp.einsum("bsd,d->bs", stream, pull_keys[c]) for c in columns]
        if embed_query is not None:
            logits = [jnp.einsum("bsd,d->bs", stream, embed_query[gate_index]), *logits[1:]]
    for name in ("bias", "mask"):
        table = extras.get(name)
        if table is not None:
            row = table[gate_index]
            logits = [logit + row[c] for logit, c in zip(logits, columns, strict=True)]
    return logits


def _gate_z(logits: list[jax.Array]) -> jax.Array:
    """Mean over tokens of ``logsumexp(logits)^2`` for one gate."""
    return jnp.mean(jnp.square(jax.nn.logsumexp(jnp.stack(logits), axis=0)))


def _attn_res_layer(
    diff_args: tuple[Block, tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array | None, jax.Array, jax.Array | None],
    mask: AttentionMask,
    token_ids: Int[Array, "B S"],
    use_long: bool,
    layer_index: int,
    eps: float,
    kv_share: dict[str, jax.Array] | None = None,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """One Block AttnRes layer on ``diff_args = (layer, blocks, block_logits, partial, queries)``.

    ``partial`` is None right after a block boundary (it was just rolled into ``blocks``), in which
    case this layer starts a fresh partial sum.
    """
    layer, blocks, block_logits, partial, queries, logit_bias = diff_args
    h, z_attn, w_attn = _attn_res_mix(blocks, block_logits, partial, queries, 2 * layer_index, eps, logit_bias)
    attn_branch = type(layer).attn_branch
    if layer.attn.cfg.attn_res_remat_attention:
        attn_branch = eqx.filter_checkpoint(attn_branch, policy=None)
    physical = layer_index % layer.attn.cfg.num_layers
    kda_ablation = (physical in layer.attn.cfg.kda_no_decay_layers, physical in layer.attn.cfg.kda_no_beta_layers)
    attn_out = attn_branch(layer, h, mask, use_long, use_long, token_ids, kv_share, None, (), kda_ablation)
    partial = attn_out if partial is None else partial + attn_out
    # The MLP re-attends over the history including this layer's attention write.
    h, z_mlp, w_mlp = _attn_res_mix(blocks, block_logits, partial, queries, 2 * layer_index + 1, eps, logit_bias)
    mlp_out, router_stats = layer.mlp_branch(h, mask)
    return partial + mlp_out, {
        **router_stats,
        _ATTN_RES_Z: z_attn + z_mlp,
        _ATTN_RES_W_ATTN: w_attn,
        _ATTN_RES_W_MLP: w_mlp,
    }


def _attn_res_layer_full(diff_args, mask, token_ids, use_long, layer_index, eps, kv_share=None):
    """One full-AttnRes layer: the attention output becomes its own source before the MoE gate, and the
    MoE output is returned as the partial, which the next layer rolls into its own source. Returns
    ``(partial, blocks, block_logits, router_stats)`` like ``_attn_res_layer_passthrough``."""
    layer, blocks, block_logits, partial, queries, logit_bias = diff_args
    assert partial is None, "full AttnRes rolls every sublayer output into its own source"
    sum_components = layer.attn.cfg.attn_res_sum_inputs
    h, z_attn, w_attn = _attn_res_mix(blocks, block_logits, None, queries, 2 * layer_index, eps, logit_bias)
    attn_out = type(layer).attn_branch(
        layer, h, mask, use_long, use_long, token_ids, kv_share, _stream_sum(blocks), sum_components
    )
    blocks = (*blocks, attn_out)
    block_logits = (*block_logits, _attn_res_source_logits(attn_out, queries[2 * layer_index + 1 :], eps))
    h, z_mlp, w_mlp = _attn_res_mix(blocks, block_logits, None, queries, 2 * layer_index + 1, eps, logit_bias)
    if "mlp" in sum_components:
        h = _stream_sum(blocks)
    mlp_out, router_stats = layer.mlp_branch(h, mask)
    stats = {**router_stats, _ATTN_RES_Z: z_attn + z_mlp, _ATTN_RES_W_ATTN: w_attn, _ATTN_RES_W_MLP: w_mlp}
    return mlp_out, blocks, block_logits, stats


def _stream_sum(sources: tuple[jax.Array, ...]) -> jax.Array:
    """The straight sum of the AttnRes sources (a standard residual stream), in the sources' dtype."""
    total = sources[0].astype(jnp.float32)
    for src in sources[1:]:
        total = total + src.astype(jnp.float32)
    return reshard(total.astype(sources[0].dtype), _batch_spec())


def _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps, kv_share=None):
    """``_attn_res_layer`` returning ``(partial, blocks, block_logits, router_stats)``: the history is
    passed through so ``_attn_res_layer_remat`` can thread each block's cotangent layer to layer."""
    _, blocks, block_logits, _, _, _ = diff_args
    partial, router_stats = _attn_res_layer(diff_args, mask, token_ids, use_long, layer_index, eps, kv_share)
    return partial, blocks, block_logits, router_stats


@eqx.filter_custom_vjp
def _attn_res_layer_remat(diff_args, mask, token_ids, use_long, layer_index, eps):
    """``_attn_res_layer_passthrough`` with a backward shaped for the unrolled layer loop.

    The history is passed through unchanged so each block's cotangent is threaded layer to layer and
    accumulated eagerly. Otherwise JAX sums a block's per-gate cotangents with one ``add_any`` at the
    end, which XLA fuses into a single late reduction that keeps every later gate's ``[B, S, D]`` input
    gradient alive. The backward recomputes the layer from its inputs (nothing but the inputs is saved)
    behind an optimization barrier with the incoming cotangents, and emits all of its cotangents through
    a second barrier so the whole layer's backward, weight gradients included, finishes before the next
    layer's starts. Without the barriers the unrolled loop's peak memory at d1024 is 3.6x the scanned
    baseline's.
    """
    return _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps)


@_attn_res_layer_remat.def_fwd
def _attn_res_layer_remat_fwd(perturbed, diff_args, mask, token_ids, use_long, layer_index, eps):
    del perturbed
    return _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps), None


@_attn_res_layer_remat.def_bwd
def _attn_res_layer_remat_bwd(residuals, grad_out, perturbed, diff_args, mask, token_ids, use_long, layer_index, eps):
    del residuals, perturbed
    # Router stats are logging-only and carry no cotangent.
    d_partial, d_blocks, d_block_logits, _d_stats = grad_out
    _, blocks, block_logits, _, _, _ = diff_args
    d_blocks = jax.tree.map(lambda d, x: jnp.zeros_like(x) if d is None else d, d_blocks, blocks, is_leaf=_is_none)
    d_block_logits = jax.tree.map(
        lambda d, x: jnp.zeros_like(x) if d is None else d, d_block_logits, block_logits, is_leaf=_is_none
    )
    with jax.named_scope(f"attn_res_bwd{layer_index}"):
        diff_args, d_partial, d_blocks, d_block_logits = jax.lax.optimization_barrier(
            (diff_args, d_partial, d_blocks, d_block_logits)
        )
        _, vjp_fn = jax.vjp(
            lambda args: _attn_res_layer(args, mask, token_ids, use_long, layer_index, eps)[0], diff_args
        )
        ((d_layer, d_blocks_own, d_block_logits_own, d_partial_in, d_queries, d_logit_bias),) = vjp_fn(d_partial)
        d_blocks = tuple(a + b for a, b in zip(d_blocks, d_blocks_own, strict=True))
        d_block_logits = tuple(a + b for a, b in zip(d_block_logits, d_block_logits_own, strict=True))
        # One barrier over every cotangent: the weight gradients are off the critical path, and without
        # it XLA defers them past later layers' backward and keeps their inputs alive.
        return jax.lax.optimization_barrier((d_layer, d_blocks, d_block_logits, d_partial_in, d_queries, d_logit_bias))


def _is_none(x) -> bool:
    return x is None


def _is_long_layer(
    layer_index: int, num_layers: int, global_every: int, global_layers: tuple[int, ...] | None = None
) -> bool:
    if global_layers is not None:
        return layer_index in global_layers
    # Every global_every-th layer is full-causal, and the last layer always is, so a depth that is
    # not a multiple of global_every still ends on a global-context layer.
    return (layer_index + 1) % global_every == 0 or layer_index == num_layers - 1


def _long_layer_schedule(num_layers: int, global_every: int, global_layers: tuple[int, ...] | None = None) -> jax.Array:
    return jnp.asarray(
        [_is_long_layer(i, num_layers, global_every, global_layers) for i in range(num_layers)], dtype=jnp.bool_
    )


def _kda_layer_indices(cfg: GrugModelConfig) -> tuple[int, ...]:
    """Layers whose mixer is KDA: the local layers when ``local_mixer`` is KDA, else none."""
    if cfg.local_mixer != LocalMixer.KDA:
        return ()
    return tuple(
        i for i in range(cfg.num_layers) if not _is_long_layer(i, cfg.num_layers, cfg.global_every, cfg.global_layers)
    )


def _unstack_layers(stacked: ArrayStacked[Block]) -> list[Block]:
    """Split the stacked layer params into per-layer modules (split, so the grad is one concatenate)."""
    num_layers = stacked.num_layers
    leaves, treedef = jax.tree.flatten(stacked.stacked)
    split_leaves = [jax.lax.split(leaf, (1,) * num_layers, axis=0) for leaf in leaves]
    return [treedef.unflatten([parts[i][0] for parts in split_leaves]) for i in range(num_layers)]


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm | None
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[Block]
    """The softmax-attention layers: every layer, or the global layers when the local layers are KDA."""
    kda_blocks: ArrayStacked[Block] | None
    """The KDA (local) layers. The AttnRes loop splits each stack whole into its layers (never slices
    it), so the hybrid keeps one stack per mixer kind: fewer, larger optimizer leaves."""
    final_norm: RMSNorm
    final_gated_norm: GatedNorm | None
    attn_res_query_final: Float[Array, " D"] | None
    """Pseudo-query of the final AttnRes gate, whose mix feeds the final norms and the lm_head."""
    token_embed2: jax.Array | None
    embed2_norm: RMSNorm | None
    attn_res_query_bias: Float[Array, "G N"] | None
    """AttnRes logit bias per (gate, source); the last column is the running partial."""
    attn_res_query_pull: Float[Array, "N D"] | None
    """Pull-AttnRes source keys (``attn_res_pull``); the last row is the partial's."""
    attn_res_query_embed: Float[Array, "G D"] | None
    """Per-gate pull projection for the embedding logit (``attn_res_pull_embed``)."""
    w_mtp: Float[Array, "D D"] | None
    """Next-token-embedding projection of the MTP head (``mtp_weight``)."""
    attn_res_query_loop: Float[Array, "P G D"] | None
    """AttnRes pseudo-queries of the extra loop passes (``loop_passes - 1`` of them), zero-init."""
    loop_inject_scale: Float[Array, " P"] | None
    """Input-injection scale per extra loop pass, init 1."""
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(
        cfg_or_vocab: GrugModelConfig | Axis,
        config: GrugModelConfig | None = None,
        *,
        key: PRNGKeyArray,
    ) -> "Transformer":
        if isinstance(cfg_or_vocab, Axis):
            if config is None:
                raise ValueError("config must be provided when initializing with a Vocab axis")
            cfg = (
                config
                if cfg_or_vocab.size == config.vocab_size
                else dataclasses.replace(config, vocab_size=cfg_or_vocab.size)
            )
        else:
            if config is not None:
                raise ValueError("config must not be provided when initializing directly from GrugModelConfig")
            cfg = cfg_or_vocab

        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        # Folded off the root key so the optional table leaves every other init unchanged.
        embed2_key = random.fold_in(key, 2)
        # The embedding is fully replicated for a local lookup.
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), P(None, None)
        )
        output_proj = reshard(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), _LM_HEAD_PARTITION_SPEC
        )

        def stack(layers: tuple[int, ...], use_kda: bool) -> ArrayStacked[Block]:
            keys = jnp.stack([block_keys[i] for i in layers])
            return ArrayStacked.init(len(layers), Block)(cfg, key=keys, use_kda=use_kda)

        softmax_layers, kda_layers = _stack_layer_indices(cfg)
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key) if cfg.embed_gated_norm else None
            ),
            output_proj=output_proj,
            stacked_blocks=stack(softmax_layers, False),
            kda_blocks=stack(kda_layers, True) if kda_layers else None,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key) if cfg.final_gated_norm else None
            ),
            attn_res_query_final=(
                reshard(jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32), P(None)) if cfg.attn_res else None
            ),
            token_embed2=(
                reshard(_init_weight(embed2_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), P(None, None))
                if cfg.second_embed
                else None
            ),
            embed2_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps) if cfg.second_embed else None,
            attn_res_query_bias=(
                jnp.zeros((2 * cfg.num_layers * cfg.loop_passes + 1, _attn_res_num_sources(cfg)), jnp.float32)
                if cfg.attn_res_logit_bias
                else None
            ),
            attn_res_query_pull=(
                jnp.zeros((_attn_res_num_sources(cfg), cfg.hidden_dim), jnp.float32) if cfg.attn_res_pull else None
            ),
            attn_res_query_embed=(
                jnp.zeros((2 * cfg.num_layers * cfg.loop_passes + 1, cfg.hidden_dim), jnp.float32)
                if cfg.attn_res_pull_embed
                else None
            ),
            w_mtp=(
                reshard(
                    _init_weight(random.fold_in(key, 3), (cfg.hidden_dim, cfg.hidden_dim), cfg.initializer_std),
                    P(_FSDP_AXES, None),
                )
                if cfg.mtp_weight > 0
                else None
            ),
            attn_res_query_loop=(
                reshard(jnp.zeros((cfg.loop_passes - 1, 2 * cfg.num_layers, cfg.hidden_dim), jnp.float32), P())
                if cfg.loop_passes > 1
                else None
            ),
            loop_inject_scale=jnp.ones((cfg.loop_passes - 1,), jnp.float32) if cfg.loop_passes > 1 else None,
            config=cfg,
        )

    def layer_stacks(self) -> list[ArrayStacked[Block]]:
        """The block stacks; stack ``k`` holds layers ``stack_layer_indices()[k]``."""
        return [self.stacked_blocks] if self.kda_blocks is None else [self.stacked_blocks, self.kda_blocks]

    def stack_layer_indices(self) -> list[tuple[int, ...]]:
        """The layer indices held by each of ``layer_stacks()``, in stack order."""
        return [indices for indices in _stack_layer_indices(self.config) if indices]

    def layers(self) -> list[Block]:
        """Every layer as its own module, in layer order (each stack split whole, never sliced)."""
        by_index: dict[int, Block] = {}
        for stack, indices in zip(self.layer_stacks(), self.stack_layer_indices(), strict=True):
            by_index.update(zip(indices, _unstack_layers(stack), strict=True))
        return [by_index[i] for i in range(self.config.num_layers)]

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        loop_active: bool | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``loop_active`` (static, with ``loop_grow_step``) selects one pass (False) or all ``loop_passes``
        (True); None runs all passes."""
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        hidden = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_norm(hidden)
        if self.embed_gated_norm is not None:
            hidden = self.embed_gated_norm(hidden)

        # Local layers use a sliding window; every global_every-th layer is full causal.
        segment_ids = None
        if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
            # Pin the [B, S] segment ids batch-sharded and reuse one array for both attention sides.
            q_segment_ids, _ = mask.segment_ids
            q_segment_ids = _batch_reshard(q_segment_ids)
            segment_ids = (q_segment_ids, q_segment_ids)
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        # Precompute FA4 per-token metadata for long/short layers outside the layer loop.
        batch_size, seq_len = hidden.shape[0], hidden.shape[1]
        long_lower_bounds, valid = fa4_cute_segment_bounds(
            long_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=None
        )
        short_lower_bounds, _ = fa4_cute_segment_bounds(
            short_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=cfg.sliding_window
        )
        long_lower_bounds = _batch_reshard(long_lower_bounds)
        short_lower_bounds = _batch_reshard(short_lower_bounds)
        valid = _batch_reshard(valid)

        final_gate_stats: dict[str, jax.Array] = {}
        if cfg.second_embed and not cfg.attn_res:
            raise ValueError("second_embed requires attn_res")
        if cfg.attn_res:
            extra_sources = ()
            if self.token_embed2 is not None:
                assert self.embed2_norm is not None
                ids2 = token_ids
                if cfg.second_embed_bigram:
                    doc_start = None if segment_ids is None else segment_ids[0]
                    ids2 = _bigram_hash_ids(token_ids, doc_start, cfg.vocab_size)
                extra_sources = (self.embed2_norm(_embedding_gather(self.token_embed2, ids2)),)
            hidden, stacked_router_stats, final_gate_stats = self._attn_res_layers(
                hidden,
                token_ids,
                extra_sources,
                loop_active,
                long_mask.with_fa4_bounds(long_lower_bounds, valid),
                long_mask.with_fa4_bounds(short_lower_bounds, valid),
            )
        else:
            # One compiled Block body scanned over the stacked layers; per-layer short/long is a
            # Bool[num_layers] scan input, and the FA4 metadata is selected per layer with jnp.where.
            mask_schedule = _long_layer_schedule(cfg.num_layers, cfg.global_every, cfg.global_layers)

            def _scan_layers(
                carry_hidden: Float[Array, "B S D"],
                scan_inputs: tuple[Block, jax.Array],
            ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
                layer, layer_use_long_mask = scan_inputs
                use_long = jnp.asarray(layer_use_long_mask, dtype=jnp.bool_)
                lower_bounds = jnp.where(use_long, long_lower_bounds, short_lower_bounds)
                layer_mask = long_mask.with_fa4_bounds(lower_bounds, valid)
                return eqx.filter_checkpoint(layer, policy=None)(
                    carry_hidden,
                    layer_mask,
                    use_long,
                    use_long,
                )

            hidden, stacked_router_stats = jax.lax.scan(
                _scan_layers, hidden, xs=(self.stacked_blocks.stacked, mask_schedule)
            )
        if cfg.dense_mlp:
            router_metrics: dict[str, jax.Array] = {}
        else:
            # One cross-device reduction for the whole layer stack, not one per layer (see router_metrics).
            reduced_router_stats = reduce_router_stats(
                stacked_router_stats,
                num_experts=cfg.num_experts,
                num_experts_per_token=cfg.num_experts_per_token,
                num_tokens=batch_size * seq_len,
            )
            router_metrics = {
                "routing_entropy_per_layer": reduced_router_stats["routing_entropy"],
                "routing_counts_per_layer": reduced_router_stats["routing_counts"],
                "load_balancing_loss_per_layer": reduced_router_stats["load_balancing_loss"],
                "router_z_loss_per_layer": reduced_router_stats["router_z_loss"],
                "qb_beta_per_layer": reduced_router_stats["qb_beta"],
                "capacity_overflow_per_layer": stacked_router_stats["capacity_overflow"],
                "sender_capacity_overflow_per_layer": stacked_router_stats["sender_capacity_overflow"],
                "receiver_capacity_overflow_per_layer": stacked_router_stats["receiver_capacity_overflow"],
                "margin_min_per_layer": stacked_router_stats["margin_min"],
                "margin_max_per_layer": stacked_router_stats["margin_max"],
            }
        router_metrics.update(final_gate_stats)
        hidden = self.final_norm(hidden)
        if self.final_gated_norm is not None:
            hidden = self.final_gated_norm(hidden)
        return hidden, router_metrics

    def _attn_res_layers(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        extra_sources: tuple[jax.Array, ...],
        loop_active: bool | None,
        long_layer_mask: AttentionMask,
        short_layer_mask: AttentionMask,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], dict[str, jax.Array]]:
        """Block AttnRes over the layers, unrolled so each gate reads only its valid sources.

        The residual history is a Python tuple of completed block sums (the token embedding is the
        first, rolled in at layer 0) plus the running partial; every ``seg_size``-th layer rolls its
        incoming partial into a new block. Completed blocks are immutable, so each block's logits
        against the queries of every gate that reads it (layer ``i``'s onwards and the final gate) are
        computed once when it is rolled, and each gate then only scores the partial. Blocks are shared
        by reference, so block memory is one copy per block. With ``AttnResLayerBackward.RECOMPUTE``
        each layer is rematerialized by ``_attn_res_layer_remat``, which saves only its inputs.

        Returns the final gate's mix, the per-layer router stats and the final gate's mean max weight
        and mean entropy.
        """
        cfg = self.config
        assert self.attn_res_query_final is not None
        eps = cfg.layer_norm_eps
        num_layers, passes = cfg.num_layers, cfg.loop_passes
        seg_size = max(1, num_layers // cfg.attn_res_num_blocks)
        block_cap = cfg.attn_res_num_blocks * passes
        layers = self.layers()
        embedding = hidden
        # Every query in gate order [pass 0: attn_0, mlp_0, ..., mlp_{L-1}; pass 1: ...; final].
        gate_queries = []
        for layer in layers:
            assert layer.attn_res_query_attn is not None and layer.attn_res_query_mlp is not None
            gate_queries += [layer.attn_res_query_attn, layer.attn_res_query_mlp]
        loop_queries = [] if self.attn_res_query_loop is None else list(self.attn_res_query_loop)
        queries = jnp.concatenate([jnp.stack(gate_queries), *loop_queries, self.attn_res_query_final[None]])
        logit_bias = _gate_extras(self, queries.shape[0])
        if cfg.attn_res_final_mode not in ("attn", "uniform", "sum"):
            raise ValueError(f"attn_res_final_mode must be attn, uniform or sum, got {cfg.attn_res_final_mode!r}")
        if cfg.attn_res_sum_inputs and not cfg.attn_res_full:
            raise ValueError("attn_res_sum_inputs needs attn_res_full")
        if set(cfg.attn_res_sum_inputs) - {"q", "k", "v", "mlp"}:
            raise ValueError(f"attn_res_sum_inputs must be a subset of q, k, v, mlp, got {cfg.attn_res_sum_inputs}")
        if cfg.attn_res_full and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("attn_res_full needs attn_res_layer_backward=SAVE")
        if cfg.mla_share_kv_latent and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("mla_share_kv_latent needs attn_res_layer_backward=SAVE (the latent crosses layers)")
        if cfg.attn_res_z_loss > 0 and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("attn_res_z_loss needs attn_res_layer_backward=SAVE (the remat VJP drops stat cotangents)")
        layer_fn = (
            _attn_res_layer_remat
            if cfg.attn_res_layer_backward == AttnResLayerBackward.RECOMPUTE
            else _attn_res_layer_passthrough
        )

        weight_logs: dict[int, tuple[jax.Array, bool]] = {}

        def run_pass(state, pass_index):
            """One pass over the physical layers, extending the history; returns the new state, the
            per-layer router stats and the pass's gate z terms."""
            blocks, block_logits, partial = state
            stats_out, z_out = [], []
            kv_share: dict[str, jax.Array] | None = {} if cfg.mla_share_kv_latent else None
            for i, layer in enumerate(layers):
                eff = pass_index * num_layers + i
                if cfg.attn_res_full or (eff % seg_size == 0 and eff // seg_size < block_cap):
                    assert partial is not None
                    blocks = (*blocks, partial)
                    # Score the new block only against the gates that can read it (this layer's onwards).
                    block_logits = (*block_logits, _attn_res_source_logits(partial, queries[2 * eff :], eps))
                    partial = None
                if pass_index > 0 and i == 0:
                    assert self.loop_inject_scale is not None
                    # Input injection: the extra pass's running partial starts from the embedding.
                    scale = self.loop_inject_scale[pass_index - 1]
                    inject = (scale * embedding.astype(jnp.float32)).astype(embedding.dtype)
                    partial = inject if partial is None else partial + inject
                use_long = _is_long_layer(i, num_layers, cfg.global_every, cfg.global_layers)
                partial_before = partial
                layer_args = (
                    (layer, blocks, block_logits, partial, queries, logit_bias),
                    long_layer_mask if use_long else short_layer_mask,
                    token_ids,
                    use_long,
                    eff,
                    eps,
                )
                if cfg.attn_res_full:
                    partial, blocks, block_logits, stats = _attn_res_layer_full(*layer_args, kv_share)
                elif kv_share is None:
                    partial, blocks, block_logits, stats = layer_fn(*layer_args)
                else:
                    partial, blocks, block_logits, stats = _attn_res_layer_passthrough(*layer_args, kv_share)
                z_out.append(stats.pop(_ATTN_RES_Z))
                has_partial_attn = partial_before is not None
                weight_logs[2 * eff] = (stats.pop(_ATTN_RES_W_ATTN), has_partial_attn)
                weight_logs[2 * eff + 1] = (stats.pop(_ATTN_RES_W_MLP), not cfg.attn_res_full)
                stats_out.append(stats)
            return (blocks, block_logits, partial), stats_out, z_out

        def final_gate(state):
            blocks, block_logits, partial = state
            final_index = queries.shape[0] - 1
            logits = [_block_logit(bl, queries, final_index) for bl in block_logits]
            logits.append(_attn_res_source_logits(partial, queries[final_index][None], eps)[0])
            logits = _bias_gate_logits(logits, logit_bias, final_index, [*blocks, partial], eps, has_partial=True)
            if cfg.attn_res_final_mode == "uniform":
                logits = [jnp.zeros_like(logit) for logit in logits]
            weights, mixed = _softmax_mix(logits, [*blocks, partial])
            if cfg.attn_res_final_mode == "sum":
                mixed = _stream_sum((*blocks, partial)).astype(jnp.float32)
            weight_logs[final_index] = (jax.lax.stop_gradient(jnp.mean(weights, axis=(1, 2))), True)
            max_weight = jax.lax.stop_gradient(jnp.mean(jnp.max(weights, axis=0)))
            entropy = jax.lax.stop_gradient(jnp.mean(-jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-30)), axis=0)))
            return mixed, _gate_z(logits), max_weight, entropy

        def merge_passes(per_pass_stats):
            """Per physical layer: sum integer stats (counts) and average the rest over passes."""
            stacked = [jax.tree.map(lambda *xs: jnp.stack(xs), *layer) for layer in zip(*per_pass_stats, strict=True)]

            def reduce(x):
                return jnp.sum(x, axis=0) if jnp.issubdtype(x.dtype, jnp.integer) else jnp.mean(x, axis=0)

            return [jax.tree.map(reduce, layer) for layer in stacked]

        def finish(state, per_pass_stats, z_terms):
            mixed, z_final, max_weight, entropy = final_gate(state)
            z_total = (sum(z_terms) + z_final) / (len(z_terms) + 1)
            return mixed, merge_passes(per_pass_stats), z_total, max_weight, entropy

        state = (extra_sources, tuple(_attn_res_source_logits(src, queries, eps) for src in extra_sources), hidden)
        state, pass0_stats, pass0_z = run_pass(state, 0)
        aux_hidden = None
        if cfg.aux_lm_layer is not None:
            raise ValueError("aux_lm_layer is not supported by this AttnRes loop")

        def all_passes(state):
            per_pass, z_terms = [pass0_stats], list(pass0_z)
            for pass_index in range(1, passes):
                state, stats, z = run_pass(state, pass_index)
                per_pass.append(stats)
                z_terms += z
            return finish(state, per_pass, z_terms)

        def one_pass(state):
            # Same structure as all_passes: the pass-0 stats stand in for every pass.
            return finish(state, [pass0_stats] * passes, list(pass0_z))

        run_all = passes == 1 or cfg.loop_grow_step is None or loop_active is None or loop_active
        mixed, layer_stats, z_total, max_weight, entropy = all_passes(state) if run_all else one_pass(state)
        final_stats = {"attn_res_final_max_weight": max_weight, "attn_res_final_entropy": entropy}
        # Source n is block n (the embedding is block len(extra_sources)); "p" is the running partial.
        for gate, (mean_weights, has_partial) in sorted(weight_logs.items()):
            num_blocks = mean_weights.shape[0] - int(has_partial)
            for n in range(num_blocks):
                final_stats[f"attn_res_w_g{gate:02d}_b{n:02d}"] = mean_weights[n]
            if has_partial:
                final_stats[f"attn_res_w_g{gate:02d}_p"] = mean_weights[-1]
        for i, layer in enumerate(layers):
            if isinstance(layer.attn, CausalSelfAttention) and layer.attn.qk_mult is not None:
                final_stats[f"attn_res_qk_mult_L{i}"] = jax.lax.stop_gradient(layer.attn.qk_mult)
            if layer.attn_out_scale is not None and layer.mlp_out_scale is not None:
                final_stats[f"attn_res_scale_attn_L{i}"] = jax.lax.stop_gradient(layer.attn_out_scale)
                final_stats[f"attn_res_scale_mlp_L{i}"] = jax.lax.stop_gradient(layer.mlp_out_scale)
        hidden = reshard(mixed.astype(hidden.dtype), _batch_spec())
        if aux_hidden is not None:
            final_stats[_AUX_HIDDEN] = aux_hidden
        final_stats["attn_res_z"] = jax.lax.stop_gradient(z_total)
        if cfg.attn_res_z_loss > 0:
            final_stats[_ATTN_RES_Z] = z_total
        query_norms = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.square(queries.astype(jnp.float32)), axis=-1)))
        final_stats["attn_res_query_norm_mean"] = jnp.mean(query_norms)
        final_stats["attn_res_query_norm_max"] = jnp.max(query_norms)
        for g in range(queries.shape[0]):
            final_stats[f"attn_res_query_norm_g{g:02d}"] = query_norms[g]
        final_stats.update(
            {f"attn_res_loop_inject_p{p + 1}": jax.lax.stop_gradient(v) for p, v in enumerate(self.loop_inject_scale)}
            if self.loop_inject_scale is not None
            else {}
        )
        return hidden, jax.tree.map(lambda *xs: jnp.stack(xs), *layer_stats), final_stats

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=batch_spec)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
        aux_loss_weight: jax.Array | None = None,
        loop_active: bool | None = None,
        train_terms: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        """``aux_loss_weight`` scales the early auxiliary LM loss (``aux_lm_layer``); it is skipped at 0.
        ``train_terms`` adds the training-only objectives (MTP, AttnRes z-loss); evals leave it off so they
        score the plain next-token loss."""
        hidden, router_metrics = self(token_ids, mask=mask, loop_active=loop_active)
        aux_hidden = router_metrics.pop(_AUX_HIDDEN, None)
        attn_res_z = router_metrics.pop(_ATTN_RES_Z, None)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        def lm_loss(h: jax.Array) -> jax.Array:
            return fused_linear_softmax_cross_entropy_loss(
                h,
                self.output_proj,
                labels,
                weight=loss_weight,
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_CE_BLOCK_SIZES,
            )

        cross_entropy_loss = lm_loss(hidden)
        # Router z-loss is logged for monitoring only; it is not added to the training loss.
        loss = cross_entropy_loss
        aux_loss = None
        if aux_hidden is not None and aux_loss_weight is not None:
            aux_in = reshard(rms_norm(aux_hidden.astype(hidden.dtype)), _batch_spec())
            aux_loss = jax.lax.cond(
                aux_loss_weight > 0,
                lambda h: lm_loss(h).astype(loss_dtype),
                lambda h: jnp.zeros((), loss_dtype),
                aux_in,
            )
            loss = loss + aux_loss_weight.astype(loss_dtype) * aux_loss
        if attn_res_z is not None and train_terms:
            loss = loss + self.config.attn_res_z_loss * attn_res_z.astype(loss_dtype)
        mtp_loss = None
        # The MTP term is a training objective only; evals score plain next-token loss.
        if self.w_mtp is not None and train_terms:
            next_ids = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1)))
            next_embed = rms_norm(_embedding_gather(self.token_embed, next_ids).astype(hidden.dtype))
            mtp_hidden = hidden + jnp.einsum("bsd,de->bse", next_embed, self.w_mtp.astype(hidden.dtype))
            mtp_hidden = reshard(rms_norm(mtp_hidden), _batch_spec())
            labels2 = jnp.pad(token_ids[:, 2:], ((0, 0), (0, 2))).astype(jnp.int32)
            # The last two positions have no t+2 target.
            weight2 = loss_weight * (jnp.arange(token_ids.shape[1]) < token_ids.shape[1] - 2)[None, :].astype(loss_dtype)
            mtp_loss = fused_linear_softmax_cross_entropy_loss(
                mtp_hidden,
                self.output_proj,
                labels2,
                weight=weight2,
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_CE_BLOCK_SIZES,
            )
            loss = loss + self.config.mtp_weight * mtp_loss
        if return_router_metrics:
            final_gate_metrics = {
                f"train/attn_res/{name.removeprefix('attn_res_')}": router_metrics.pop(name)
                for name in list(router_metrics)
                if name.startswith("attn_res_")
            }
            if not router_metrics:
                # Dense model: no router to summarize.
                return loss, {"train/cross_entropy_loss": cross_entropy_loss, **final_gate_metrics}
            summarized_metrics = summarize_router_metrics(router_metrics)
            summarized_metrics.update(final_gate_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            if aux_loss is not None:
                summarized_metrics["train/attn_res/aux_lm_loss"] = aux_loss
            if mtp_loss is not None:
                summarized_metrics["train/attn_res/mtp_loss"] = mtp_loss
            num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
            summarized_metrics["train/router/z_loss_logging_only"] = (
                jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
            )
            # Keep per-layer int32 counts; the trainer sums over layers in host int64 (jnp.sum here overflows int32).
            summarized_metrics["moe/dropped_assignments"] = router_metrics["capacity_overflow_per_layer"]
            summarized_metrics["moe/sender_dropped_assignments"] = router_metrics["sender_capacity_overflow_per_layer"]
            summarized_metrics["moe/receiver_dropped_assignments"] = router_metrics[
                "receiver_capacity_overflow_per_layer"
            ]
            return loss, summarized_metrics
        return loss


def _stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)``: the layer indices of ``Transformer.stacked_blocks`` and ``kda_blocks``."""
    kda_layers = _kda_layer_indices(cfg)
    return tuple(i for i in range(cfg.num_layers) if i not in kda_layers), kda_layers


def _gate_extras(model: "Transformer", num_gates: int) -> dict[str, jax.Array | None] | None:
    """The per-gate logit extras of ``_bias_gate_logits``, or None when the model uses none of them."""
    cfg = model.config
    if cfg.attn_res_pull_embed and cfg.second_embed:
        raise ValueError("attn_res_pull_embed assumes the embedding is source 0 (no second_embed)")
    mask = None
    if cfg.attn_res_mask_attn_for:
        if not cfg.attn_res_full:
            raise ValueError("attn_res_mask_attn_for needs attn_res_full")
        mask = np.zeros((num_gates, _attn_res_num_sources(cfg)), np.float32)
        offset = int(cfg.second_embed)
        for layer in cfg.attn_res_mask_attn_for:
            # Full AttnRes sources: embedding(s), then layer j's attention output at 1 + 2j, MoE at 2 + 2j.
            for j in range(layer):
                mask[2 * layer, offset + 1 + 2 * j] = -1e9
        mask = jnp.asarray(mask)
    extras = {
        "bias": model.attn_res_query_bias,
        "pull_keys": model.attn_res_query_pull,
        "embed_query": model.attn_res_query_embed,
        "mask": mask,
    }
    return extras if any(v is not None for v in extras.values()) else None


def _attn_res_num_sources(cfg: GrugModelConfig) -> int:
    """Most sources any AttnRes gate reads: every completed block (extra embeddings included) + the partial."""
    seg_size = max(1, cfg.num_layers // cfg.attn_res_num_blocks)
    cap = cfg.attn_res_num_blocks * cfg.loop_passes
    if cfg.attn_res_full:
        return 2 * cfg.num_layers * cfg.loop_passes + int(cfg.second_embed) + 1
    rolled = sum(1 for i in range(cfg.num_layers * cfg.loop_passes) if i % seg_size == 0 and i // seg_size < cap)
    return rolled + int(cfg.second_embed) + 1


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def debug_mesh_and_token_pspec(num_devices: int) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    expert = 2 if num_devices % 2 == 0 else 1
    data = max(1, num_devices // expert)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data, expert, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("replica_dcn", "data", "expert"), None)


__all__ = [
    "AttnResLayerBackward",
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "KimiDeltaAttention",
    "LocalMixer",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "ShortConv",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "moe_and_shared_fused",
]
