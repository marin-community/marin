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
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

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
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
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
    MoeOverlapWork,
    moe_mlp,
)
from levanter.grug.loss import BlockSizes, fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import unshard
from levanter.kernels.pallas.relu2_mlp import fused_relu2
from levanter.kernels.pallas.short_conv import short_conv
from levanter.tracker.histogram import SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track.router_metrics import (
    local_routing_stats,
    reduce_router_stats,
    summarize_router_metrics,
)
from experiments.grug.moe.kda import chunk_kda, doc_starts, kda_fused

_GATED_NORM_RANK = 128
_QB_HIST_BINS = 10_000
# Tokens (split evenly over the batch shards) whose top-K sets ``qb_bias_damping``'s churn metric re-routes.
_QB_CHURN_TOKENS = 1024
_CE_TOKENS_PER_RANK = 65_536
# A vocab tile of 8192 in the fused lm_head + cross-entropy loop measured ~3% faster than 4096 at d512.
_CE_BLOCK_SIZES = BlockSizes(b_block_size=_CE_TOKENS_PER_RANK, v_block_size=8192)
# Axes the non-expert params FSDP-shard over.
_FSDP_AXES: tuple[str, ...] = ("data", "expert")
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")
# Tokens of the first sequence whose output-bigram logit term is logged (a [64, V] matmul).
_OUTPUT_BIGRAM_STAT_TOKENS = 64
# Extra hidden columns carrying ``lm_head_bias`` into the fused CE (two used, padded to a multiple of 8).
_LM_HEAD_BIAS_COLS = 8
# Input channels the per-token MoE output gate reads (cfg.moe_out_gate).
_MOE_OUT_GATE_DIMS = 12


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
# ``newton_muon``: the per-layer expert-input second moment ``Z^T Z / N`` ([L, n, n], layer order) the
# forward returns to the trainer, and the per-shard partial sums it is reduced from.
NEWTON_GRAM_KEY = "_newton_gram"
NEWTON_GRAM_LOCAL_KEY = "newton_gram_local"
# Metrics-dict key that carries the auxiliary-loss residual stream from the forward to the loss.
_AUX_HIDDEN = "aux_lm_hidden"
# Folded into the per-step route key for the ERC proxy-token noise, so it is independent of the Gumbel noise.
_ERC_KEY_SALT = 0xE2C
_NITP_TARGET = "nitp_target"
# Metrics-dict key that carries the raw (pre-norm) token embeddings from the forward to the MTP loss.
_MTP_EMBED = "mtp_embed"
# Folded into the per-step route key for the MTP position subsample (``mtp_position_frac``).
_MTP_KEY_SALT = 0x3F7
# Per-layer product-key memory diagnostics, lifted out of the layer stats into ``train/attn_res/knob_mem_*``.
# Per-layer stats with this prefix returned by a layer are exported as ``<name>_L<layer>``.
_LAYER_KNOB_PREFIX = "attn_res_knob_"
_MEMORY_STAT_PREFIX = f"{_LAYER_KNOB_PREFIX}mem_"
_KDA_ERASE_STAT_PREFIX = f"{_LAYER_KNOB_PREFIX}kda_erase_"
# Bound on one chunk's gathered ``[tokens, rows, dim]`` memory rows in the product-key EmbeddingBag.
_MEMORY_BAG_CHUNK_ELEMS = 1 << 26
# Metrics-dict keys carrying the AttnRes z-loss term (with gradient) from the forward to the loss.
_ATTN_RES_Z = "attn_res_z_term"
# Per-gate token-mean AttnRes source weights (variable length per gate), popped into logging scalars.
_ATTN_RES_W_ATTN = "attn_res_weights_attn"
_ATTN_RES_W_MLP = "attn_res_weights_mlp"
_ATTN_RES_W_V = "attn_res_weights_v"
# Per-layer router stats carrying each token's ``[T, K]`` selected experts and combine weights to
# ``Transformer.routing_assignments``.
_ROUTING_SELECTED = "routing_selected"
_ROUTING_WEIGHTS = "routing_weights"
ROUTING_SELECTED_KEY = "routing_selected_per_layer"
ROUTING_WEIGHTS_KEY = "routing_weights_per_layer"

# Kimi K3's KDA layer: low-rank forget-gate width, and the per-token log-decay floor
# ``g = -KDA_MIN_LOG_DECAY * sigmoid(...)``.
_KDA_GATE_RANK = 128
KDA_MIN_LOG_DECAY = 5.0
# 16-token chunks keep the kernels' intra-chunk rescaling exact for the -5 per-token log-decay
# floor (cumulative >= -80 = -DEFLATE_EXP_CAP; see kda_prep_pallas).
KDA_CHUNK_SIZE = 16
# kda_dd_rope: per-pair base angular frequencies (rad/token) are log-spaced over this range, periods ~6 to
# ~800 tokens, spanning the KDA decay's memory lengths (|g| init in kda_dt_range).
_KDA_ROT_OMEGA_RANGE = (1.0 / 128, 1.0)
# KDA per-layer activation diagnostics, lifted out of the layer stats into ``train/attn_res/knob_kda_*``.
_KDA_STAT_PREFIX = "attn_res_knob_kda_"
# Folded off the root key for the KV side stream (kv_stream_dim), so it leaves every other init unchanged.
_KV_STREAM_KEY_SALT = 0x4B5


class MtpMode(StrEnum):
    """Multi-token-prediction objective (training only; evals score the plain next-token loss)."""

    OFF = "off"
    DEEPSEEK = "deepseek"
    """DeepSeek-V3 depth-1 MTP (arXiv 2412.19437 sec. 2.2) with an attention-free block (``MtpHead``)."""


class NgramStatMode(StrEnum):
    """Where the n-gram statistic reader's output enters the model."""

    SOURCE = "source"
    """Its own AttnRes source (RMS-normed, content-gated). It takes a softmax share at every gate from step 0, while
    the table is still empty; at d512 this cost +0.010 to +0.012 Paloma macro loss at equal steps."""
    BIGRAM = "bigram"
    """Added to the trained bigram table's source through a zero-initialized reader output, so step 0 is exactly the
    model without it and no gate's softmax is diluted (needs ``second_embed_bigram``)."""


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
    SQRT_SOFTPLUS_RENORM = "sqrt_softplus_renorm"
    """``sqrt(softplus(logit))`` (DeepSeek-V4's SqrtSoftplus gate), renormalized to sum to ``routing_renorm_sum``."""


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


def _embedding_gather_autodiff(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """``_embedding_gather`` with JAX's default transpose: the backward scatter-adds straight into the
    (bf16) table (``embed_grad_fp32=False``, the pre-fix behavior)."""
    token_ids = reshard(token_ids, P(_BATCH_AXES, None))
    return shard_map(
        _local_gather,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(token_embed, token_ids)


# Pair-combine multiplier and murmur3 finalizer constants for the (previous, current) bigram hash.
_BIGRAM_HASH_PAIR = 0x9E3779B1
_MURMUR_C1 = 0x85EBCA6B
_MURMUR_C2 = 0xC2B2AE35
# Per-head salt step for the multi-head n-gram hash (salt 0 is the single-head hash).
_HASH_SALT_STEP = 0x27D4EB2D
# Hash salt of the statistic table (distinct from the trained bigram table's salt 0) and its code's seed.
_NGRAM_STAT_SALT = 7
_NGRAM_STAT_CODE_SEED = 20260927


def _bigram_hash_ids(
    token_ids: Int[Array, "B S"],
    segment_ids: Int[Array, "B S"] | None,
    num_buckets: int,
    ngram: int = 2,
    salt: int = 0,
) -> Int[Array, "B S"]:
    """Hash each n-gram ending at a token (the ``ngram - 1`` previous tokens, then the token) into
    ``num_buckets`` rows. A previous token before position 0 or in an earlier document is replaced by the
    sentinel ``num_buckets`` (never a real id)."""
    x = jnp.zeros(token_ids.shape, jnp.uint32)
    for lag in range(ngram - 1, 0, -1):
        prev = jnp.pad(token_ids[:, :-lag], ((0, 0), (lag, 0)), constant_values=num_buckets)
        if segment_ids is not None:
            other_doc = jnp.pad(segment_ids[:, lag:] != segment_ids[:, :-lag], ((0, 0), (lag, 0)), constant_values=True)
            prev = jnp.where(other_doc, num_buckets, prev)
        x = (x + prev.astype(jnp.uint32)) * jnp.uint32(_BIGRAM_HASH_PAIR)
    x = x + token_ids.astype(jnp.uint32) + jnp.uint32((salt * _HASH_SALT_STEP) & 0xFFFFFFFF)
    x = (x ^ (x >> 16)) * jnp.uint32(_MURMUR_C1)
    x = (x ^ (x >> 13)) * jnp.uint32(_MURMUR_C2)
    x = x ^ (x >> 16)
    return (x % jnp.uint32(num_buckets)).astype(jnp.int32)


def _padded_spec(x: jax.Array) -> tuple:
    """``x``'s partition spec entries, padded with None to ``x.ndim``."""
    spec = tuple(_partition_spec_of(x) or ())
    return spec + (None,) * (x.ndim - len(spec))


def _partition_spec_of(x: jax.Array) -> P | None:
    sharding = jax.typeof(x).sharding if isinstance(x, core.Tracer) else x.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    return None


# Expert-parallel MoE transports the model supports: the fixed pooled-wave all-to-all, and the near-dropless
# ragged all-to-all (one XLA ragged_all_to_all per (peer, local expert), expert MLP on grouped ragged_dot).
MOE_IMPLEMENTATIONS = ("fixed_pooled_wave_all_to_all", "ragged_all_to_all")


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
    latent_out_dim: int | None = None
    """Width the routed experts write (``w_down``'s output dim); None: their input width (``latent_dim``, or
    ``hidden_dim`` without a latent). ``w_latent_up`` maps it to ``hidden_dim`` and is dropped when it equals
    ``hidden_dim`` (the experts write the stream directly). Splits LatentMoE's read and write compression."""
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
    kv_stream_dim: int = 0
    """KV side stream width ``w`` (0: off). A second residual stream, seeded by its own ``[vocab, w]`` token
    embedding and advanced by one small pre-norm block (causal document-masked softmax attention + ReLU^2
    MLP, see ``KvStreamBlock``) per main layer, is the only source of keys and values: main layer ``l``'s
    K/V projections (KDA ``w_k`` / ``w_v``, MLA ``w_dkv``, GQA ``w_k`` / ``w_v``) read the normed side stream
    leaving side block ``l`` (shape ``[w, ...]``) instead of the main stream, so "what am I" is built apart
    from "who comes next". Queries, gates, beta, decay and the Inkling bias stay on the main stream. Needs
    ``attn_res`` (the side-stream states reach the layers as AttnRes layer extras)."""
    kv_stream_heads: int = 1
    """Attention heads of the side-stream blocks (head dim ``kv_stream_dim / kv_stream_heads``; keep it 128
    for the FA4 kernel)."""
    kv_stream_mlp_mult: int = 4
    """Side-stream MLP width as a multiple of ``kv_stream_dim``."""
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
    latent_select_pattern: str = "first"
    """Which ``latent_dim`` hidden channels ``latent_select`` reads: ``first`` (channels ``[0, latent_dim)`` in
    every layer), ``random`` (a fixed random subset per layer), or ``rotating`` (a contiguous window offset by
    ``hidden_dim / num_layers`` per layer, wrapping around)."""
    latent_select: bool = False
    """With a MoE latent, form the expert input by selecting the first ``latent_dim`` channels of the hidden
    state (then the learnable ``latent_norm``) instead of projecting with ``w_latent_down``. The output side
    keeps ``w_latent_up``."""
    latent_select_layers: str = "all"
    """Layers whose MoE uses ``latent_select``: ``all``, ``kda`` (the KDA layers; the global layers project) or
    ``global`` (the global attention layers; the KDA layers project)."""
    latent_select_plus_proj: bool = False
    """In the selecting layers, add a learned ``w_latent_down`` projection to the selected channels before
    ``latent_norm`` (projection on top of selection)."""
    expert_read_subset: int = 0
    """If > 0, each routed expert reads only this many of its input channels (a fixed 0/1 mask on the rows of
    ``w_up``/``w_gate``, applied at init and in the forward, so the masked rows stay zero under MuonH): the experts
    together cover the whole input, but each one reads a slice. ``expert_read_subset_pattern`` picks the slices."""
    expert_read_subset_pattern: str = "blocks"
    """``blocks``: expert ``e`` reads contiguous block ``e mod (in_dim / expert_read_subset)``; ``random``: a fixed
    random subset per expert; ``shared``: every expert reads channels ``[0, expert_read_subset)`` (the control that
    isolates the per-expert split from the masking itself)."""
    expert_read_groups: int = 0
    """If > 0 (G), routed expert ``e`` reads only channel group ``e mod G`` of the MLP-pre-normed stream: the
    stream is split into G contiguous slices of width ``hidden_dim / G``, each re-normalized by its own learnable
    RMSNorm (``expert_read_norm``, a stacked ``[G, W]`` gain), and the experts' ``w_up`` are real ``[E, W, I]``
    matrices. Each (token, slot) assignment dispatches only its expert's slice, so the dispatch bytes shrink by
    G. The gather-based, per-slice-normed counterpart of ``expert_read_subset``; needs ``latent_dim=None``
    and an explicit ``latent_out_dim``."""
    latent_write_select: bool = False
    """With a MoE latent, drop ``w_latent_up``: the combined routed output is written into the first ``latent_dim``
    hidden channels (the rest get zero), scaled by ``initializer_std * sqrt(latent_dim)``, the gain of the
    replaced matrix at init (MuonH holds that matrix's norm fixed)."""
    latent_out_norm: bool = False
    """Kimi K3 normalized LatentMoE: a learnable RMSNorm on the combined routed output before ``W_latent_up``."""
    proj_biases: tuple[str, ...] = ()
    """Zero-init learnable biases at these sites: ``qkv`` (KDA q/k/v and MLA q / KV-latent projections),
    ``attn_out`` (the attention sublayer output) and ``mlp_out`` (the MoE sublayer output)."""
    qb_freeze_step: int | None = None
    """Stop updating the QB router biases from this step on (they keep their last value)."""
    qb_bias_damping: float | None = None
    """Damped QB bias update (StableMoE arXiv 2204.08396; phi-balancing arXiv 2605.15403): ``b <- (1 - gamma) b +
    gamma (-beta)`` instead of ``b <- -beta``. Setting it (1.0 is the undamped update) also logs
    ``knob_router_qb_churn``: the fraction of the first ``_QB_CHURN_TOKENS`` tokens whose top-K set changes
    when this step's logits are re-routed with the next step's bias. None: undamped, no churn metric."""
    router_logit_scale: bool = False
    """A learnable scalar per MoE layer (init 1, Adam) multiplying the router logits before QB and the combine.
    Selection is scale-invariant under QB (a positive scale is monotone and QB re-thresholds), so it sets only
    the combine temperature, which a norm-pinned router (``router_group=muonh``) cannot change otherwise."""
    expert_output_gain: bool = False
    """A learnable gain per expert (init 1, Adam) on each routed expert's output, folded into the combine weight
    of every (token, slot) as ``w * gain[expert]`` (so every MoE backend runs it unchanged)."""
    simbal_loss_weight: float = 0.0
    """SimBal (arXiv 2506.14038): adds ``weight * sum_l ||R_l^T R_l - I||_1`` over the real-expert router
    columns ``R_l`` [D, E] to the training loss (unnormalized, as in the paper; it uses 0.1). 0: off."""
    newton_muon: bool = False
    """Newton-Muon (arXiv 2604.01472) right-preconditioning of the routed-expert gate/up gradients:
    ``G <- G (K + gamma tr(K)/n I)^{-1}`` before momentum and Newton-Schulz, with ``K`` an EMA of the
    second moment ``Z Z^T / N`` of the expert input. All routed experts of a layer read the same (latent)
    input, so each layer keeps one ``[latent, latent]`` ``K``, which the forward emits and the trainer holds."""
    newton_muon_beta: float = 0.95
    """EMA decay of ``K`` per refresh (the paper's short-track record: 0.95)."""
    newton_muon_eps: float = 0.2
    """Ridge ``gamma``: the damping added to ``K`` is ``gamma * tr(K) / n`` (the paper: 0.2)."""
    newton_muon_every: int = 32
    """Refresh ``K`` and its inverse every this many steps (the paper: 32); the first step initializes ``K``."""
    embed_gated_norm: bool = True
    """GatedNorm after the embedding RMSNorm (else the RMSNorm alone)."""
    final_gated_norm: bool = True
    """GatedNorm after the final RMSNorm, before the lm_head (else the RMSNorm alone)."""
    mtp_mode: MtpMode = MtpMode.OFF
    """Depth-1 multi-token prediction (``MtpMode``): position t also predicts token t+2 of its document from
    ``W_proj [rms(h_t); rms(Emb(x_{t+1}))]`` through one extra block and the shared embedding and lm_head."""
    mtp_weight: float = 0.3
    """Weight of the MTP loss (DeepSeek-V3: 0.3 for the first 10T tokens, then 0.1)."""
    mtp_mlp_mult: int = 4
    """Hidden width of the MTP block's ReLU^2 MLP, as a multiple of ``hidden_dim``."""
    mtp_position_frac: float = 1.0
    """Fraction of positions (a fresh random subset each step, shared by the batch rows) that the MTP block and
    its lm_head pass run on; 0.5 halves the MTP cost. Needs the per-step route key."""
    nitp_weight: float = 0.0
    """Next Implicit Token Prediction (arXiv 2605.24956; 0 disables it): an MLP head ``P = W_2 gelu(W_1 h_t)``
    on the final hidden state predicts the stop-gradient residual stream of token t+1 after layer ``nitp_layer``,
    with loss ``1 - cos(P(h_t), z_{t+1})`` (pairs crossing a packed-document boundary are masked). Training-only."""
    nitp_layer: int = 1
    """0-indexed layer whose output stream (the plain sum of the AttnRes sources) is the NITP target (~20% depth)."""
    kda_push_buckets: int = 0
    """'Push' decay for KDA (0 disables it): M delta-rule states, each decaying at its own learned,
    reader-independent per-channel rate; the writing token splits its write across them
    (``beta * pi_m``), so a write chooses its own half-life. The read sums the M states."""
    kda_push_mode: str = "pure"
    """``pure``: the buckets' static decays replace the reader decay. ``hybrid``: each bucket decays at
    the reader decay plus its own static rate."""
    kda_push_weights: str = "softmax"
    """How a token splits its write over the buckets: ``softmax`` (sums to 1) or ``sigmoid`` (independent)."""
    kda_push_decay_range: tuple[float, float] = (0.02, 0.5)
    """Initial per-token log-decay magnitudes |g| of the slowest / fastest bucket (log-spaced between)."""
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
    kda_dd_rope: bool = False
    """Data-dependent rotation of the KDA state: complex-eigenvalue transitions (Mamba-3, arXiv 2603.15569;
    Selective RoPE, arXiv 2511.17388) via the RoPE trick. Channel pairs ``(j, j + h/2)`` of q and k are rotated
    by ``theta_t = cumsum_{s<=t} gamma_j * omega_j * softplus(x_s W_rot_down W_rot_up)`` (reset at document
    starts); ``omega_j`` is log-spaced in ``[1/128, 1]`` rad/token and the per-pair amplitude ``gamma`` is
    zero-init, so the layer is exact KDA at init. The decay is tied across each pair (``W_a↑`` / ``dt_bias``
    shrink to ``h/2`` columns per head), which makes each pair's transition ``exp(g) R(delta theta)`` a complex
    eigenvalue that commutes with the rotation; otherwise the trick is not the rotated recurrence."""
    kda_dd_rope_rank: int = 16
    """Hidden width of the low-rank angle projection ``W_rot_down W_rot_up`` (``kda_dd_rope``)."""
    kda_out_correction: bool = False
    """Comba output correction (arXiv 2506.02475, eq. 5): KDA reads its state with ``q - d_h k`` (q and k
    L2-normalized first; the kernel re-normalizes the corrected query, as in the FLA Comba reference) with a
    learnable per-head scalar ``d_h`` (Adam), init ``kda_out_correction_init``."""
    kda_out_correction_init: float = 0.02
    """Initial ``d_h`` of ``kda_out_correction`` (the paper: 0.02 for its small models, 1 from 1.3B)."""
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
    AttnRes mix: any of ``q``, ``k``, ``v`` (those attention projections), ``mlp`` (the whole MoE input),
    ``mlp_shared`` (the shared experts only), ``mlp_routed`` (router + latent-down) or ``mlp_router``."""
    attn_res_heads: int = 1
    """Multi-head AttnRes (RMT-style retrieval heads): each gate's pseudo-query is split into this many
    D/H chunks, and chunk h scores and mixes the sources on its own channel slice with its own softmax."""
    attn_res_v_gate: bool = False
    """A separate AttnRes pseudo-query per layer for the value projections (V reads its own mix). With
    ``attn_res_full`` it applies to the KDA layers only (an MLA layer's K and V share one latent)."""
    attn_res_additive: bool = False
    """Delta AttnRes (arXiv 2605.18855) additive routing: each gate's input is the plain sum of its
    sources (the standard residual stream) plus its softmax mix, instead of the mix alone."""
    attn_res_stream_source: bool = False
    """Every AttnRes gate also scores the plain sum of its visible sources (the standard residual
    stream) as one extra source, so "take the plain residual" is a selectable option (logged as ``_p``)."""
    attn_res_source_delta: bool = False
    """Full AttnRes over output deltas: source n becomes ``x_n - x_{n-1}`` for the gate's source sequence
    ``[embedding, sublayer outputs...]`` (source 0 stays the embedding); logits are scored on the deltas."""
    attn_res_head_sub: str = "none"
    """Hierarchical multi-head AttnRes (needs ``attn_res_heads``): each gate keeps one full-width query that
    sets every source's overall strength, and head h adds a zero-init sub-query correction. ``slice``: the
    sub-query is D/H wide and dots only its channel slice; ``full``: it is D wide and dots the whole key.
    Each head then mixes its own channel slice. ``none``: plain multi-head (the query split into slices)."""
    attn_res_temperature: bool = False
    """A learnable logit multiplier per gate (and per head with ``attn_res_heads``), init 1."""
    attn_res_head_norm: bool = False
    """Multi-head AttnRes: RMS-normalize each head's channel slice of a source for its logits, instead
    of one RMS over all channels."""
    attn_res_key_rank: int | None = None
    """Low-Rank AttnRes (LR-AttnRes, arXiv 2607.09694): every source's routing key is the parameter-free
    RMS norm of its last ``r`` channels, ``rms_norm(source[..., -r:])``, and every pseudo-query (per-layer,
    V-gate, loop and final) is ``r``-wide; the mix still sums the full-width sources. The paper's sliced
    variant (its learned-projection keys cost ~10x the routing FLOPs for the same loss). None: full-width."""
    embed_norm_mode: str = "rms"
    """Token-embedding norm before it enters the stack (AttnRes mixes it raw against the layer outputs):
    ``rms`` (RMSNorm with a learned gain), ``rms_nogain`` (RMSNorm, no gain) or ``raw`` (the table row as is)."""
    embed_scale: float = 1.0
    """Constant multiplier on the (normed) token embedding: its weight in the AttnRes mixes."""
    router_share_block: int = 1
    """PathMoE (arXiv 2603.18297): consecutive layers of each block stack share one router matrix, in groups of
    this many stack entries (the first layer of each group owns it). Per-layer QB biases stay separate. 1: off."""
    sublayer_out_norm: bool = False
    """Peri-LN (arXiv 2502.02732): RMSNorm with a gain (init 1) on each attention and MLP sublayer output
    before it becomes an AttnRes source, so the source mix combines unit-RMS values."""
    dyt_norm: bool = False
    """Dynamic Tanh (arXiv 2503.10622): the attention and MLP pre-norms become ``gamma * tanh(alpha * x) + beta``
    (learnable scalar ``alpha``, per-channel ``gamma``/``beta`` init 1/0). The embedding, final and internal
    q/k/latent norms stay RMSNorm."""
    dyt_alpha_attn: float = 0.8
    """``alpha`` init of the attention-input DyT (the paper's LLaMA-7B value: attention inputs get a larger one)."""
    dyt_alpha_mlp: float = 0.2
    """``alpha`` init of the MLP-input DyT (the paper's LLaMA-7B non-attention value)."""
    laurel_rank: int = 0
    """LAuReL-LR (arXiv 2411.07501): each sublayer input becomes ``h + (h A) B`` with a rank-r ``A``
    (random init) and ``B`` (zero init), both trained with Adam (0: off)."""
    mla_k_norm: bool = False
    """Weightless per-head RMSNorm on the MLA keys only, no query norm (DeepSeek-V4.1's setup)."""
    mla_q_norm: bool = False
    """Weightless per-head RMSNorm on the MLA queries (composes with ``mla_k_norm``). Queries are recomputed per
    token, so under MLA absorption this adds nothing to the cache."""
    mla_k_norm_shared: bool = False
    """With ``mla_k_norm``, normalize the keys by one RMS over all heads per token instead of per head. Under MLA
    absorption this caches 1 scalar per token instead of ``num_heads``."""
    mla_k_norm_split: bool = False
    """With ``mla_k_norm``, RMS-normalize the two key halves separately, i.e. the previous-token half from
    ``mla_key_offset`` and the current-token half, so neither dominates the logit by magnitude."""
    mla_v_filter: bool = False
    """Noise-filtering value gate (arXiv 2609.22005, projection gate) on the MLA layers: every value read is
    scaled by ``sigmoid(w_h . v_j + b_h)`` per head (``w`` [N, H] zero-init, ``b`` [N] init
    ``mla_v_filter_bias_init``; both Adam). Applied to the final values (after value embeds / residual)."""
    mla_v_filter_bias_init: float = 4.0
    """Initial ``b_h`` of ``mla_v_filter`` (the paper's +4: every gate starts at ~0.982)."""
    mla_forget_gate: bool = False
    """Forgetting Transformer forget gate (FoX, arXiv 2503.02130) on the MLA layers: per head a scalar
    ``f_t = sigmoid(w_h . x_t + b_h)`` (``w`` [D, N] zero-init, ``b`` [N] init ``mla_forget_gate_bias_init``;
    both Adam) adds ``sum_{k=j+1..i} log f_k = c_i - c_j`` to logit (i, j), ``c`` the per-document cumsum of
    ``log f``. The row term ``c_i`` cancels in the softmax; the kernel takes no per-key bias, so ``-c_j`` rides
    in extra q/k channels (q: 1, k: a bf16 hi/mid/lo split of ``-c_j``) and q/k/v are zero-padded to the
    kernel's 32-channel granule (head_dim 128 -> 160, ~1.25x attention FLOPs on these layers)."""
    mla_forget_gate_bias_init: float = 0.0
    """Initial ``b_h`` of ``mla_forget_gate``. The paper uses 0 (f = 0.5: a strongly local start); larger
    values start closer to plain attention (+4: f ~0.982)."""
    zero_centered_gains: bool = False
    """Zero-centered norm gains (Qwen3-Next; arXiv 2608.30320 sec. 2.1.1): every learned-gain RMSNorm scales by
    ``1 + gamma`` with ``gamma`` zero-init (identical at init), so the optimizer's ``gain_weight_decay`` pulls
    gains toward 1 rather than 0. DyT gains are unchanged."""
    mla_ssmax: bool = False
    """Scalable-softmax (SSMax, arXiv 2501.19399) on the MLA layers: each query is scaled by
    ``1 + s_h * log(n)``, with ``n`` the number of keys it can see in its document and ``s_h`` a
    zero-init per-head parameter, so the softmax can stay sharp as the context grows."""
    mla_key_offset: bool = False
    """modded-nanogpt partial key offset (record #49): on the MLA layers, the first half of each head's key
    channels come from the previous token (within documents), enabling one-layer induction."""
    mla_diff_attn: bool = False
    """Differential attention (DIFF Transformer, arXiv 2410.05258) on the MLA layers:
    ``(softmax(q1 k1^T) - lambda softmax(q2 k2^T)) v`` with a second q projection and key up-projection
    (``w_q2`` / ``w_uk2``, same head_dim and q/k transforms), ``lambda = exp(lq1 . lk1) - exp(lq2 . lk2) +
    lambda_init``, ``lambda_init = 0.8 - 0.6 exp(-0.3 (layer - 1))`` (1-indexed layer), and each head's output
    RMS-normalized and scaled by ``1 - lambda_init``."""
    attn_res_final_signed: bool = False
    """Backout-style signed correction on the final AttnRes gate: ``final = mix + sum_n c_n * source_n`` with
    learned ``c`` (zero-init), so the lm_head input can subtract a source (softmax weights can't)."""
    smear: bool = False
    """modded-nanogpt Smear (record #34): ``x_t += lambda * sigmoid(x_t[:12] @ w) * x_{t-1}`` on the embedding,
    within documents (``lambda`` learned, init 0)."""
    mla_head_mix: bool = False
    """Post-attention rank-1 dynamic cross-head mix on MLA (a post-kernel stand-in for DCMHA's query-wise
    post-softmax composition): ``y_h += w2_h(t) * sum_g w1_g(t) y_g`` with ``[w1, w2] = x @ W`` (``W2`` zero-init)."""
    attn_res_dynamic_rank: int = 0
    """MUDD-style dynamic AttnRes: each gate adds a per-token logit delta ``GELU(rms_norm(stream) @ W1) @ W2``
    over its sources (this hidden width, ``W2`` zero-init) to the static-query logits. 0: off."""
    xsa_mode: str = "fixed"
    """MLA Exclusive Self Attention strength: ``fixed`` subtracts the full self-value projection,
    ``learned`` scales it by a per-head scalar (init 1), ``gated`` by ``2 * sigmoid(x @ W_xsa)`` per token and
    head (``W_xsa`` zero-init, so 1 at init), ``tanh`` by ``tanh(alpha_h)`` with ``alpha`` zero-init (modded-nanogpt
    record #82: no XSA at init)."""
    logit_soft_cap: float | None = None
    """Tanh soft-cap on the lm_head logits, ``c * tanh(z / c)`` (Gemma 2); None: off."""
    logit_soft_cap_asym: tuple[float, ...] = ()
    """Asymmetric logit cap ``A * sigmoid((z + B) / C)`` as ``(A, B, C)`` (modded-nanogpt record #54);
    overrides ``logit_soft_cap`` when set."""
    output_bigram_rank: int = 0
    """Output-side bigram logit prior of this rank (0: off): the pre-cap logits gain ``rms(U[x_t]) @ W``, a
    low-rank ``P(x_{t+1} | x_t)`` table (n-gram interpolation in logit space). ``U`` [vocab, r] and ``W``
    [r, vocab] (zero-init, so the model is unchanged at init) ride the lm_head as ``r`` extra contraction
    columns of the one fused cross-entropy call, so the soft cap and the z-loss see the combined logits."""
    lm_head_unigram_bias: bool = False
    """A learnable ``[V]`` output bias (Adam) added to the logits inside the soft-cap. The trainer sets it to the
    log unigram frequencies of the first training batches before step 0 (``lm_head_unigram_batches``); a model
    built without the trainer starts at zero."""
    init_std_mult_gates: float = 1.0
    """Init-std multiplier of the MuonH-trained sigmoid gate matrices (KDA output gate ``w_g`` and write strength
    ``w_beta``). MuonH pins each matrix's Frobenius norm at its init (Hyperball II, arXiv 2606.16899), so the
    init std fixes that family's scale for the whole run. The zero-init Adam gates (``attn_gate``, ``ve_gate``)
    move freely and are unaffected."""
    init_std_mult_experts: float = 1.0
    """Init-std multiplier of the routed experts' ``w_up`` / ``w_down`` (see ``init_std_mult_gates``)."""
    init_std_mult_attn_out: float = 1.0
    """Init-std multiplier of the attention output projections ``w_o`` (MLA / GQA and KDA; see
    ``init_std_mult_gates``)."""
    kda_write_gate: bool = False
    """Gated DeltaNet-2 (arXiv 2605.22791) write gate: each KDA value channel is scaled by ``2 sigmoid(x W_w)``
    before the delta-rule update, a channel-wise write strength next to KDA's per-head beta. Runs through the
    existing kernel."""
    kda_erase_gate: bool = False
    """Gated DeltaNet-2 (arXiv 2605.22791) erase gate: the KDA delta rule reads and erases along
    ``e = b * k`` and writes along ``k``, ``S_t = (I - beta k e^T) D_t S_{t-1} + beta k v^T``, with a
    channel-wise ``b = 2 sigmoid(x W_b)`` (``W_b`` zero-init, so exactly KDA at init; ``beta * b`` spans
    the paper's (0, 2) negative-eigenvalue range). Runs through the KDA kernels' erase-key path."""
    kda_beta_negative: bool = False
    """KDA write strength ``beta = 2 * sigmoid(logit - log 3)`` in (0, 2), so the transition ``I - beta k k^T``
    can have negative eigenvalues (Grazzi et al. 2025); the shift keeps the mean beta at init at 1/2."""
    shared_expert_gate: bool = False
    """Qwen-MoE shared-expert gate: the shared experts' output is scaled per token by ``2 * sigmoid(x @ w)``,
    with ``w`` zero-init so the gate starts at 1."""
    shared_ungated_relu2: bool = False
    """Shared experts are ``relu(x @ W_up)^2 @ W_down`` with no gate projection (truly ungated: the gate
    GEMM is dropped from the fused projection). Parameter-match with 1.5x ``shared_expert_intermediate_dim``."""
    moe_expert_waves: int = 1
    """Static dispatch waves in the pooled-wave EP backend. Waves are independent, so the scheduler can overlap
    one wave's all-to-all with another's expert compute (the all-to-all was ~13% exposed at d512)."""
    moe_shared_overlap: bool = False
    """Run the shared experts inside the pooled-wave EP shard under the first wave's dispatch all-to-all
    (forward) and its reverse (backward), pinned there by optimization barriers; same math. Their gate/up
    GEMM leaves the fused router/latent projection. Without EP (one expert shard) they just run in place."""
    moe_expert_remat: bool = True
    """Recompute the pooled-wave expert MLP and combine all-to-all in the backward. Off trades activation
    memory for one fewer expert forward and combine all-to-all per wave."""
    moe_fp8_dispatch: bool = False
    """DeepSeek-V3 FP8 dispatch: the EP dispatch all-to-all sends activations as e4m3 with one fp32 scale per
    128-channel block (~0.52x the bf16 bytes); combine and every backward collective stay bf16 (STE)."""
    moe_implementation: str = "fixed_pooled_wave_all_to_all"
    """Expert-parallel transport, one of ``MOE_IMPLEMENTATIONS``. ``ragged_all_to_all`` ignores the pooled-wave
    knobs (``pooled_transport_capacity_factor``, ``moe_expert_waves``, ``moe_expert_remat``) and sizes its receiver
    buffers by ``capacity_factor``; its XLA transport kernel is ``GrugRunConfig.ragged_transport``."""
    moe_bank2_experts: int = 0
    """Heterogeneous experts: the last ``moe_bank2_experts`` of ``num_experts`` form a second expert bank with its
    own width (``moe_bank2_intermediate_dim``) and activation (``moe_bank2_activation``). One router scores all
    experts; each token takes ``num_experts_per_token - moe_bank2_topk`` experts from bank 1 and ``moe_bank2_topk``
    from bank 2 (a fixed split, so each bank's FLOPs are fixed), QB balances each bank against its own K/E, and
    the combine weights are renormalized over all selected experts. 0: one uniform bank."""
    moe_null_experts: int = 0
    """MoE++ (arXiv 2410.07348) zero experts: extra router columns past ``num_experts`` (after them come the
    ``moe_copy_experts`` and ``moe_const_experts`` columns) whose output is 0. The router takes top-K over the
    real and all zero-computation experts, so a token uses between 0 and K real experts; the combine weights
    are renormalized over all K slots, so a null slot's weight is taken from the real experts. Null slots go to
    the real-expert backend as combine-weight-0 assignments spread over the experts (the fixed-capacity EP
    dispatch still carries and computes them), so this changes the math, not the FLOPs. 0: off."""
    moe_copy_experts: int = 0
    """MoE++ copy experts (zero-computation): output = the expert input (the latent-space token)."""
    moe_const_experts: int = 0
    """MoE++ constant experts (zero-computation): output = ``a1 * x + a2 * v`` with a learned vector ``v`` and
    ``[a1, a2] = softmax(x @ W_c)`` per constant expert (``v``, ``W_c`` zero-init, Adam)."""
    moe_null_target_frac: float | None = None
    """QB treatment of the zero-computation experts. None: they are free (router bias 0) and QB only balances
    the real experts against each other at their measured load, with the mean real bias pinned at 0, so the
    real-vs-null split is left to the learned router logits. A float: one QB over all columns sends this
    fraction of the top-K slots to the zero-computation experts (split evenly), like MoE++'s tau-weighted
    balance loss."""
    moe_bank2_topk: int = 0
    moe_bank2_intermediate_dim: int = 0
    moe_bank2_scale: bool = False
    """A learnable output scale per expert bank (init 1, Adam): MuonH pins each bank's weight norm, so without it
    the banks' relative output magnitudes are fixed by their activations and widths."""
    moe_bank2_activation: str = "relu2"
    """``relu2`` (ungated, like ``moe_ungated_relu2``) or ``swiglu``."""
    expert_leaky_slope: float = 0.0
    """Ungated ReLU^2 experts (routed and shared) use ``leaky_relu(u, slope)^2`` instead (Parameter Golf #493 /
    #549: 0.5; #1948: 0.3). 0: plain ReLU^2."""
    moe_out_gate: bool = False
    """Parameter Golf #1941: per-token gate ``sigmoid(x[:, :12] w + b)`` (w = 0, b = +5, about 0.993 at init) on each
    block's whole MoE output (routed + shared)."""
    moe_fused_relu2: bool = False
    """With ``moe_ungated_kernel``, run the ungated ReLU^2 expert MLP through the fused-epilogue kernels
    (``levanter.kernels.pallas.relu2_mlp`` for pooled-wave, ``relu2_ragged_mlp`` for ragged all-to-all): ``pre``
    and ``d post`` never reach HBM."""
    moe_ungated_kernel: bool = False
    """With ``moe_ungated_relu2``, run the EP (pooled-wave or ragged) experts truly ungated (one ``W_up`` GEMM)
    instead of tying the gate to ``W_up``. Same math; skips the duplicated GEMM."""
    moe_dense_router_grad: bool = False
    """Default MoE (Panda et al. 2025): the router also gets a gradient for the experts it did not pick,
    through ``sum_{e not in top-k} (w_e - sg(w_e)) * sg(y_e)``. That term is zero in the forward pass.
    ``y_e`` is expert e applied to the mean input of the tokens routed to it in this batch, so it stands
    in for the expert's typical output. Only for the renormalized-sigmoid combine."""
    moe_ungated_relu2: bool = False
    """Routed experts are ``relu(x @ W_up)^2 @ W_down`` (no gate projection). Run through the gated kernels
    as ``relu(g) * u`` with the gate tied to ``W_up`` (``relu(u) * u = relu(u)^2``), so every MoE backend
    computes it unchanged; the gate GEMM is still paid. Parameter-match with 1.5x ``num_experts`` and
    ``num_experts_per_token``."""
    expert_activation: str = "silu"
    """Gate activation of the routed and shared GLU experts (an ``ActivationFunctionEnum`` value, e.g.
    ``relu2`` for ReLU^2-GLU as in Primer / ReMoE)."""
    moe_hash_layers: tuple[int, ...] = ()
    """Physical layers whose routed experts are picked by a fixed token-id hash (Hash Layers, Roller et
    al. 2021): each vocab id maps to K distinct random experts; combine weights still come from the router."""
    moe_drop_renorm: bool = False
    """Renormalize each token's combine weights over the slots that survived the capacity-limited EP dispatch,
    so they keep their pre-drop total (``routing_renorm_sum`` under the renormalized combines). Without it a
    dropped (token, expert) assignment just loses its weight: the token's routed output shrinks. Tokens with
    no drop are unchanged. Costs one int8 all-to-all per expert wave (the receiver keep mask back to the
    senders)."""
    router_token_bias_rank: int = 0
    """Rank ``r`` of a learned token-identity bias on the router logits, ``A[token_id] @ B_l``: ``A`` is one
    shared ``[vocab, r]`` table (random, like an embedding) and ``B_l`` a zero-init ``[r, E]`` per layer, so
    the model is unchanged at init. Added before QB, which then balances it. 0: off."""
    router_embed_tie: tuple[str, ...] = ()
    """Router-embedding ties ``"L:E:V"`` (``L`` a layer index or ``*`` for every layer): layer ``L``'s router
    column for expert ``E`` is ``alpha * token_embed[V]``, so its logit is ``alpha * token_embed[V] . x`` on
    the normed MLP input ``x``. The embedding row is the shared input-embedding parameter (the LM gradient
    reaches it through the tie); ``alpha`` is a learnable scalar per tie, initialized to
    ``||router[:, E]|| / ||token_embed[V]||`` so the tied column starts at the untied column's scale. The
    replaced router column gets no gradient. ``"L:E:V1|V2|..."`` ties to the centroid
    ``alpha * mean(token_embed[V1], token_embed[V2], ...)`` of the current rows (every row gets gradient).
    Seeds expert specialization (expert-specialization study); parsed once by ``_router_ties``."""
    router_embed_tie_release_step: int | None = None
    """Release the ``router_embed_tie`` ties at this step: the router columns take their tied value (so the
    function is continuous) and then train freely; the tied program runs only before it. None: tied throughout."""
    router_bias_seed: tuple[str, ...] = ()
    """Fixed router logit priors ``"L:E:V:b"`` (``L`` a layer index or ``*``): add ``b`` to expert ``E``'s
    router logit (before QB and the combine) wherever the current token is ``V``. Needs attn_res."""
    router_logit_soft_cap: float | None = None
    """Tanh soft-cap ``c * tanh(z / c)`` on the router logits (after ``router_token_bias_rank``), before the
    QB-biased top-K selection and the combine weights; QB estimates its thresholds on the capped logits.
    None: off."""
    attn_res_logit_soft_cap: float | None = None
    """Tanh soft-cap ``c * tanh(z / c)`` on every AttnRes gate's source logits (after the ``_gate_extras``
    biases, before the softmax and the z term); the ``attn_res_dual_query`` second mix is not capped.
    None: off."""
    moe_gumbel_tau: float = 0.0
    """Training-only Gumbel noise (scale tau) added to the biased router logits before top-K: samples K
    experts without replacement from softmax(logits / tau) instead of taking the top K. Evals use top-K."""
    embed_grad_fp32: bool = True
    """Accumulate the token-embedding gradient in float32 (``_embedding_gather``); False restores JAX's
    default gather transpose, which scatter-adds into the bf16 table."""
    attn_res_dual_query: bool = False
    """Two pseudo-queries per AttnRes gate: each mixes the sources with its own softmax, and the two mixes
    are merged per token by ``s = sigmoid(rms_norm(stream) . w + b)`` (``w``, ``b`` zero-init, so s = 1/2),
    where the stream is the plain sum of the gate's sources. The second query is ``N(0, dual_init_std)``."""
    attn_res_dual_init_std: float = 0.005
    attn_res_blend: str = "none"
    """Blend each AttnRes gate's mix with the uniform average of its sources (the standard residual
    direction): ``input = l1 * mix + l2 * mean(sources)``. ``static``: learned scalars per gate, init 1;
    ``dynamic``: plus ``rms_norm(stream) . w_k`` per token (``w`` zero-init)."""
    mlp_in_center: bool = False
    """Subtract the batch mean (over all tokens, stop-gradient) from every MoE input."""
    mlp_in_whiten_power: float = 0.0
    """Whiten every MoE input by ``C^(-p/2)``, ``C`` its batch covariance (after centering, stop-gradient,
    eigenvalues trace-normalized and shrunk by ``mlp_in_whiten_eps``); 0 is off, 1 is full ZCA whitening."""
    mlp_in_whiten_eps: float = 1e-3
    router_rank: int | None = None
    """Low-rank router: logits = f(x W_r_down) W_r_up with this inner width (None: the linear x W_r)."""
    router_rank_act: str = "none"
    """Activation on the low-rank router features: ``none``, ``norm`` (learnable RMSNorm) or ``silu``."""
    attn_res_full: bool = False
    """Full (not Block) AttnRes: every attention and MoE sublayer output is its own source. Needs
    attn_res_layer_backward=SAVE."""
    moe_shortcut: bool = False
    """Shortcut-connected MoE (ScMoE, arXiv 2404.05019; LongCat-Flash arXiv 2509.01322): layer ``l``'s MoE
    gate mixes the same AttnRes sources as layer ``l``'s attention gate (with its own query), excluding that
    attention's output, which still enters the history for later gates. The MoE dispatch then does not
    depend on the attention, so its all-to-all can overlap the attention compute. Needs ``attn_res``."""
    mla_share_kv_latent: bool = False
    """Every MLA layer after the first reuses the first MLA layer's normed KV latent (own W_uk / W_uv, so
    absorption still works), halving the MLA KV cache at d512. Needs attn_res_layer_backward=SAVE."""
    value_residual_layers: tuple[int, ...] = ()
    """ResFormer value residual learning (arXiv 2410.17897): these 0-indexed layers mix the first layer's
    attention values into their own, ``v <- l1 * v + l2 * v_first``, with learnable per-layer ``(l1, l2)``
    (KDA: after the v ShortConv + SiLU; MLA: after the up-projection). Layer 0 is the source and cannot
    be listed. Empty: off. Needs attn_res_layer_backward=SAVE (the values cross layers)."""
    value_residual_init: tuple[float, float] = (0.5, 0.5)
    """Initial ``(l1, l2)`` of ``value_residual_layers`` (modded-nanogpt's learnable 0.5 / 0.5)."""
    learnable_qk_mult: bool = False
    """A learnable scalar per softmax-attention layer (init ``qk_mult``) in place of the fixed ``qk_mult``."""
    attn_gate_elementwise: bool = False
    """Gated attention (arXiv 2505.06708, its best variant G1): the output gate is per channel,
    ``2 sigmoid(x W_g)`` with ``W_g`` [D, N*H] zero-init, instead of one scalar per head."""
    qk_mult_per_head: bool = False
    """With ``learnable_qk_mult``, one logit scale per head instead of per layer, so each head picks its own
    softmax temperature (with q and k normalized, qk_mult is the whole temperature)."""
    aux_lm_layer: int | None = None
    """Early auxiliary LM loss: the residual stream after this layer (the plain sum of the AttnRes
    sources) goes through a parameter-free RMS norm and the shared lm_head. None disables it."""
    aux_lm_weight: float = 1.0
    """Weight of the auxiliary loss at step 0; annealed linearly to 0 at ``aux_lm_steps``."""
    aux_lm_steps: int = 500
    second_embed: bool = False
    embed2_rows: int = 0
    ple_dim: int = 0
    """Per-layer embeddings (Gemma 3n PLE): a ``[vocab, num_layers * ple_dim]`` token table whose layer-``l`` slice
    (RMS-normed) is added to layer ``l``'s attention input as ``h + (gelu(rms(h) W_gate) * ple_l) W_up``, ``W_up``
    zero-init. Stored and sharded like the second table (``embed2_fsdp``, ``embed2_grad_fp32``). 0: off."""
    memory_layers: tuple[int, ...] = ()
    """Product-key memory layers (arXiv 1907.05242; memory+ of arXiv 2412.09764): each listed 0-indexed layer
    adds ``(bag(V, topk(rms(x W_q) . rms(K))) * silu(x W_gate)) W_out`` to its MoE output, ``x`` the RMS-normed
    MoE input. Each head's query is split in two halves scored against the head's two ``memory_keys``-row
    codebooks; the ``memory_topk^2`` candidate sums keep the top ``memory_topk`` of ``memory_keys^2`` slots,
    softmax-weighted. Heads share one ``[memory_keys^2, hidden_dim]`` value table per layer, stored like the
    second table (``embed2_fsdp``) and trained by the ``memory`` Adam group. ``W_out`` is zero-init. Empty: off."""
    memory_keys: int = 512
    memory_topk: int = 32
    memory_heads: int = 4
    memory_key_dim: int = 256
    """Query / key width per memory head (both halves together)."""
    byte_aux_bytes: int = 0
    """Byte-level auxiliary loss: a separate head predicts the first ``byte_aux_bytes`` UTF-8 bytes of the
    next token (256-way each, positions past the token's end masked), weighted by a schedule the trainer
    passes (``byte_aux_weight``, decayed to 0 by ``byte_aux_decay_frac`` of training). 0: off."""
    byte_aux_weight: float = 0.1
    byte_aux_mid_layer: bool = False
    """Feed the byte head the RMS-normed mid-network AttnRes stream (the embedding plus the first half of the
    completed blocks) instead of the final hidden, so the byte objective shapes mid-network features rather than
    the lm_head input."""
    byte_aux_decay_frac: float = 0.5
    erc_loss_weight: float = 0.0
    """Expert-router coupling loss (arXiv 2512.23447): each real expert's router column, perturbed by
    ``U(1 +- eps_i)`` noise (``eps_i`` = half the distance to the nearest other router column over its norm),
    is a proxy token that runs the token's path into the experts (latent down-projection and ``latent_norm``)
    and every expert's first projection (``w_gate``, or ``w_up`` when ungated); with ``M[i, j]`` the L2 norm of
    proxy ``i`` through expert ``j``, the loss is ``mean_{i != j} relu(M_ij - alpha M_ii) + relu(M_ji - alpha M_ii)``
    over all ``n^2`` entries, summed over MoE layers. Router-parameter only: token-count independent, never
    touches the QB bias. Training only. 0: off (the paper uses 1)."""
    erc_alpha: float = 1.0
    """ERC coupling margin ``alpha`` in [0, 1]; smaller is stricter (the paper's n=256 optimum is 0.5)."""
    window_embed_dim: int = 0
    """Explicit short context window as an AttnRes source: every token gets a small ``window_embed_dim``-wide
    embedding; for position t the embeddings of tokens t, t-1, ..., t-(k-1) (k = hidden_dim / window_embed_dim)
    are concatenated slot by slot (zero where a slot crosses a document start), linearly projected to
    ``hidden_dim`` and RMS-normed. 0: off."""
    window_embed_mode: str = "source"
    """``source``: the token window is its own AttnRes source. ``add_embed``: it is added to the token-embedding
    source. As its own source, the gates gave it ~0.001-0.01 weight by step 350, which starved its gradient."""
    embed3_rows: int = 0
    """Rows of a hashed *trigram* table that is its own AttnRes source next to the bigram table, so each
    gate weighs the bigram and trigram views separately (0: off; needs ``second_embed_bigram``)."""
    bigram_gate: bool = False
    """Engram-style content gate on the bigram source: ``g_t = sigmoid(sum_d w_d rms(e_t)_d rms(b_t)_d + c)`` scales
    each token's bigram row by how its token embedding and bigram row interact, which the static AttnRes queries
    can't express (w = 0 and c = +2 at init)."""
    embed2_hash_heads: int = 1
    """Engram-style multi-head hashing: each bigram row is split into this many slices of ``hidden_dim / heads``, each
    indexed by its own hash of the bigram, so a collision corrupts only the slices whose hashes collide. The table is
    stored as ``[heads * rows, hidden_dim / heads]`` (the same parameters and gathered bytes as one table)."""
    embed2_head_orders: tuple[int, ...] = ()
    """With ``embed2_hash_heads``, the n-gram order of each head (e.g. ``(2, 2, 3, 3)``: two bigram and two trigram
    slices in the same table and gather). Empty: every head uses ``embed2_ngram``."""
    trigram_gate: bool = False
    """The ``bigram_gate`` content gate (same rank) on the trigram source (``embed3_rows``), with its own parameters."""
    bigram_gate_rank: int = 0
    """With ``bigram_gate``, a per-channel gate instead of a scalar: ``g_t = sigmoid(rms(e_t) * rms(b_t) @ A @ B + c)``
    with rank-r ``A`` (random) and ``B`` (zero), so each token keeps some bigram features and drops others."""
    embed2_fsdp: bool = False
    """Store the second table (and so its optimizer / EMA state) row-sharded over the FSDP axes, all-gathering
    a replicated copy for the lookup. Same math; for rungs where the replicated table's state doesn't fit."""
    ngram_stat_rows: int = 0
    """Rows per n-gram order of a fixed-encoder *statistic* table, its own AttnRes source (0: off). Row h holds
    ``[sum of code(y), count]`` over every occurrence of an n-gram hashing to h followed by next token y, where
    ``code`` is a fixed random ``[vocab, ngram_stat_dim]`` matrix. The table is never trained: the trainer adds
    each batch after its step (``write_ngram_stats``), so a row means the same thing at every step. The model
    reads ``[sum / count, log1p(count)]`` of every order through a learned reader (see
    .agents/projects/stable-compressor-memory.md)."""
    ngram_stat_dim: int = 64
    """Width of the fixed next-token code of the statistic table (the table stores this plus a count column)."""
    ngram_stat_orders: tuple[int, ...] = (2,)
    """n-gram orders of the statistic table, ``ngram_stat_rows`` rows each (2: the (previous, current) bigram)."""
    ngram_stat_mlp_dim: int = 0
    """Width of a GELU layer in the statistic reader (0: a linear reader straight to ``hidden_dim``)."""
    ngram_stat_gate: bool = True
    """Scalar Engram content gate on the statistic source (w = 0, c = +2 at init), like ``bigram_gate`` (``SOURCE``
    mode only)."""
    ngram_stat_mode: NgramStatMode = NgramStatMode.SOURCE
    """Where the reader's output enters: its own AttnRes source, or added into the bigram source (``NgramStatMode``)."""
    embed2_grad_fp32: bool = True
    """Second table's backward through the fp32 local scatter + psum (True), or JAX's default bf16
    scatter-add (False). Hashed rows see few adds each, so the bf16 path's atomic contention and rounding
    matter less than for the token table."""
    embed2_ngram: int = 2
    """With ``second_embed_bigram``, the n-gram length hashed into the table (2: bigram, 3: trigram)."""
    embed2_dim: int | None = None
    """Low-rank second table: ``embed2_rows x embed2_dim`` rows up-projected to ``hidden_dim`` by a shared
    ``embed2_up`` matrix (None: full-width rows). Shrinks the table's per-step gradient all-reduce and
    optimizer work by ``hidden_dim / embed2_dim``."""
    """Rows of the second embedding table (0: ``vocab_size``). With ``second_embed_bigram`` the (previous,
    current) hash spreads over these rows, so a table much larger than the vocab keeps bigrams apart."""
    second_embed_mode: str = "source"
    """``source``: the second embedding is one more AttnRes source. ``input`` (modded-nanogpt style): it
    is RMS-normalized and added to every sublayer's AttnRes input as ``h + lambda_g * rms(h) * e2``, with
    a learned ``lambda_g`` per gate initialized to ``embed2_lambda_init``."""
    embed2_lambda_init: float = 0.1
    second_embed_bigram: bool = False
    """Index the second table by a hash of (previous token, token) instead of the token: a bigram
    embedding with ``vocab_size`` hashed rows (the previous token is a sentinel at document starts)."""
    """A second, independently initialized token-embedding table, RMS-normed, as an extra AttnRes source."""

    def __post_init__(self) -> None:
        if self.moe_implementation not in MOE_IMPLEMENTATIONS:
            raise ValueError(f"moe_implementation must be one of {MOE_IMPLEMENTATIONS}, got {self.moe_implementation!r}")
        if self.moe_fp8_dispatch and self.moe_implementation != "fixed_pooled_wave_all_to_all":
            raise ValueError("moe_fp8_dispatch requires moe_implementation=fixed_pooled_wave_all_to_all")
        if not self.dense_mlp and self.num_experts_per_token >= self.num_experts:
            # QB routing takes top-(k+1) and keeps the last entry as the threshold alpha, so a
            # full-bank top-k asks `jax.lax.top_k` for more entries than the router has experts.
            raise ValueError("num_experts_per_token must be < num_experts, because QB routing selects top-(k+1)")
        if self.local_mixer == LocalMixer.KDA and not self.attn_res:
            raise ValueError("local_mixer=kda requires attn_res (KDA layers run in the unrolled AttnRes loop)")
        if self.attn_res_key_rank is not None:
            if not self.attn_res or not 0 < self.attn_res_key_rank < self.hidden_dim:
                raise ValueError("attn_res_key_rank needs attn_res and 0 < attn_res_key_rank < hidden_dim")
            if self.attn_res_heads > 1 or self.attn_res_pull or self.attn_res_pull_embed or self.attn_res_dual_query:
                raise ValueError("attn_res_key_rank needs single-head push AttnRes without attn_res_dual_query")
        if self.kda_dd_rope:
            if self.local_mixer != LocalMixer.KDA:
                raise ValueError("kda_dd_rope requires local_mixer=kda")
            if self.inferred_head_dim % 2:
                raise ValueError("kda_dd_rope rotates channel pairs and needs an even head_dim")
            if self.kda_push_buckets:
                raise ValueError("kda_dd_rope needs pair-tied decays; kda_push_buckets' per-channel decays are not")
        if self.moe_drop_renorm and self.moe_implementation != "fixed_pooled_wave_all_to_all":
            raise ValueError("moe_drop_renorm needs moe_implementation=fixed_pooled_wave_all_to_all")
        if self.router_token_bias_rank and not self.attn_res:
            raise ValueError("router_token_bias_rank needs attn_res (the scanned stack has no router extras)")
        if self.router_embed_tie or self.router_bias_seed:
            if self.dense_mlp or self.router_rank:
                raise ValueError("router_embed_tie / router_bias_seed need MoE layers with the full-rank router")
            if self.router_bias_seed and not self.attn_res:
                raise ValueError("router_bias_seed needs attn_res (the scanned stack has no router extras)")
            ties = _router_ties(self)
            if len({(t.layer, t.expert) for t in ties}) != len(ties):
                raise ValueError(f"router_embed_tie ties one (layer, expert) twice: {self.router_embed_tie}")
            specs = [(t, t.tokens) for t in ties] + [(s, (s.token,)) for s in _router_seeds(self)]
            for spec, tokens in specs:
                if not (0 <= spec.expert < self.num_experts and all(0 <= v < self.vocab_size for v in tokens)):
                    raise ValueError(f"router tie/seed out of range (experts {self.num_experts}): {spec}")
        if self.router_embed_tie_release_step is not None:
            if not self.router_embed_tie:
                raise ValueError("router_embed_tie_release_step needs router_embed_tie")
            if self.router_embed_tie_release_step <= 0:
                raise ValueError(f"router_embed_tie_release_step must be positive: {self.router_embed_tie_release_step}")
        for name in ("router_logit_soft_cap", "attn_res_logit_soft_cap"):
            cap = getattr(self, name)
            if cap is not None and cap <= 0:
                raise ValueError(f"{name} must be positive, got {cap}")
        if self.num_null_experts:
            if self.moe_bank2_experts or self.moe_hash_layers or self.moe_dense_router_grad:
                raise ValueError("zero-computation experts do not support moe_bank2, hash layers or dense router grad")
            if self.moe_null_target_frac is not None and not 0.0 < self.moe_null_target_frac < 1.0:
                raise ValueError(f"moe_null_target_frac must be in (0, 1), got {self.moe_null_target_frac}")

        if self.latent_select_pattern not in ("first", "random", "rotating"):
            raise ValueError(
                f"latent_select_pattern must be first, random or rotating, got {self.latent_select_pattern!r}"
            )
        if self.expert_read_subset_pattern not in ("blocks", "random", "shared"):
            raise ValueError(
                f"expert_read_subset_pattern must be blocks, random or shared, got {self.expert_read_subset_pattern!r}"
            )
        if self.expert_read_subset and (
            self.expert_in_dim % self.expert_read_subset or self.moe_bank2_experts or self.moe_const_experts
        ):
            raise ValueError("expert_read_subset must divide the expert input width, without moe_bank2 or const experts")
        if self.expert_read_groups and (
            self.latent_dim is not None
            or self.latent_select
            or self.latent_out_dim is None
            or self.hidden_dim % self.expert_read_groups
            or self.expert_read_subset
            or self.moe_bank2_experts
            or self.moe_const_experts
            or self.num_null_experts
            or self.moe_dense_router_grad
            or self.newton_muon
            or self.erc_loss_weight > 0
        ):
            raise ValueError(
                "expert_read_groups must divide hidden_dim and needs latent_dim=None, an explicit latent_out_dim, one "
                "expert bank and no expert_read_subset, null/const experts, dense router grad, Newton-Muon or ERC"
            )
        if self.latent_select_layers not in ("all", "kda", "global"):
            raise ValueError(f"latent_select_layers must be all, kda or global, got {self.latent_select_layers!r}")
        if self.latent_select_layers != "all" and self.local_mixer != LocalMixer.KDA:
            raise ValueError("latent_select_layers kda/global needs local_mixer=KDA")
        if self.latent_write_select and (self.latent_dim is None or self.latent_dim > self.hidden_dim):
            raise ValueError("latent_write_select needs latent_dim <= hidden_dim")
        if self.latent_select and (
            self.latent_dim is None or self.latent_dim > self.hidden_dim or self.erc_loss_weight > 0
        ):
            raise ValueError(
                "latent_select needs latent_dim <= hidden_dim and no ERC loss (ERC maps router rows via w_latent_down)"
            )
        if self.qb_bias_damping is not None and not 0.0 < self.qb_bias_damping <= 1.0:
            raise ValueError(f"qb_bias_damping must be in (0, 1], got {self.qb_bias_damping}")
        if self.simbal_loss_weight > 0 and (self.dense_mlp or self.router_rank):
            raise ValueError("simbal_loss_weight needs a MoE with a full-rank router")
        if self.latent_out_dim is not None and (
            self.dense_mlp or self.latent_write_select or self.moe_bank2_experts or self.num_null_experts
        ):
            raise ValueError(
                "latent_out_dim needs routed experts and no latent_write_select, moe_bank2 or zero-computation experts"
            )
        if self.erc_loss_weight > 0 and (self.dense_mlp or self.router_rank or self.moe_bank2_experts):
            raise ValueError("erc_loss_weight needs a MoE with a full-rank router and one expert bank")

        if self.newton_muon and (self.dense_mlp or self.newton_muon_every < 1):
            raise ValueError("newton_muon needs routed experts (dense_mlp=False) and newton_muon_every >= 1")

        if self.kv_stream_dim:
            if not self.attn_res or self.kv_stream_dim % self.kv_stream_heads:
                raise ValueError("kv_stream_dim needs attn_res and a multiple of kv_stream_heads")
            if (
                self.mla_share_kv_latent
                or self.attn_res_v_gate
                or self.attn_res_source_delta
                or self.mla_forget_gate
                or {"k", "v"} & set(self.attn_res_sum_inputs)
            ):
                raise ValueError(
                    "kv_stream_dim does not combine with mla_share_kv_latent, attn_res_v_gate, attn_res_source_delta, "
                    "mla_forget_gate (a key-side main-stream bias) or k/v attn_res_sum_inputs"
                )
        if self.moe_shortcut and not self.attn_res:
            raise ValueError("moe_shortcut requires attn_res (the MoE shortcut is an AttnRes gate)")
        if self.memory_layers:
            if not self.attn_res:
                raise ValueError("memory_layers requires attn_res (the memory runs in the AttnRes loop)")
            layers = set(self.memory_layers)
            if not layers <= set(range(self.num_layers)) or len(layers) != len(self.memory_layers):
                raise ValueError(f"memory_layers must be distinct layers in 0..{self.num_layers - 1}")
            if self.memory_key_dim % 2 or self.memory_topk > self.memory_keys:
                raise ValueError("memory_key_dim must be even and memory_topk <= memory_keys")

    @property
    def expert_in_dim(self) -> int:
        """Width the routed experts read (one ``expert_read_groups`` slice when set)."""
        if self.expert_read_groups:
            return self.hidden_dim // self.expert_read_groups
        return self.latent_dim if self.latent_dim is not None else self.hidden_dim

    @property
    def expert_out_dim(self) -> int:
        """Width the routed experts write (``latent_out_dim``)."""
        return self.latent_out_dim if self.latent_out_dim is not None else self.expert_in_dim

    @property
    def has_latent_up(self) -> bool:
        """The MoE maps the experts' output to ``hidden_dim`` with ``w_latent_up``."""
        if self.latent_out_dim is not None:
            return self.latent_out_dim != self.hidden_dim
        return self.latent_dim is not None and not self.latent_write_select

    @property
    def kv_in_dim(self) -> int:
        """Input width of the attention K/V projections: the side stream (``kv_stream_dim``) or the main stream."""
        return self.kv_stream_dim or self.hidden_dim

    @property
    def num_null_experts(self) -> int:
        """Zero-computation (zero + copy + constant) experts, the router columns past ``num_experts``."""
        return self.moe_null_experts + self.moe_copy_experts + self.moe_const_experts

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
    attn_gate: Float[Array, "D G"]  # G = N heads (headwise) or N*H channels (cfg.attn_gate_elementwise)
    sconv_k: "ShortConv | None"  # SConv after the K projection (cfg.sconv)
    sconv_q: "ShortConv | None"  # MLA only: SConv after the q projection ("q" in cfg.sconv_sites)
    rel_pos: "InklingRelPos | None"  # Inkling relative-position bias (replaces RoPE when set)
    w_dkv: Float[Array, "D L"] | None
    kv_latent_norm: "LearnedRMSNorm | None"
    w_uk: Float[Array, "L NH"] | None
    w_uv: Float[Array, "L NH"] | None
    value_embed: Float[Array, "V NH"] | None
    ve_lambda: Float[Array, " 2"] | None  # (lambda1 on v, lambda2 on the value embedding)
    ve_gate: Float[Array, "D N"] | None
    qk_mult: Float[Array, "..."] | None  # learnable logit scale, [] or [N] per head (cfg.learnable_qk_mult)
    xsa_scale: Float[Array, " N"] | None  # per-head XSA strength (cfg.xsa_mode == "learned")
    xsa_gate: Float[Array, "D N"] | None  # per-token XSA gate weights (cfg.xsa_mode == "gated")
    head_mix: Float[Array, "D 2N"] | None  # [w1 | w2] projections of cfg.mla_head_mix (w2 half zero-init)
    ssmax_scale: Float[Array, " N"] | None  # per-head SSMax log-length query scale (cfg.mla_ssmax)
    w_q2: Float[Array, "D NH"] | None  # second query projection (cfg.mla_diff_attn)
    w_uk2: Float[Array, "L NH"] | None  # second key up-projection from the KV latent (cfg.mla_diff_attn)
    diff_lambda: Float[Array, "4 H"] | None  # [lq1, lk1, lq2, lk2] of the DIFF lambda reparameterization
    diff_lambda_init: Float[Array, ""] | None  # constant lambda_init of this layer (never trained)
    vres_lambda: Float[Array, " 2"] | None  # (l1 on v, l2 on the first layer's v): cfg.value_residual_layers
    bias_q: Float[Array, " NH"] | None
    bias_dkv: Float[Array, " L"] | None
    v_filter_w: Float[Array, "N H"] | None  # noise-filter value gate direction (cfg.mla_v_filter), zero-init
    v_filter_b: Float[Array, " N"] | None  # noise-filter value gate bias (cfg.mla_v_filter)
    forget_gate_w: Float[Array, "D N"] | None  # FoX forget gate direction (cfg.mla_forget_gate), zero-init
    forget_gate_b: Float[Array, " N"] | None  # FoX forget gate bias (cfg.mla_forget_gate)
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, layer_index: jax.Array) -> "CausalSelfAttention":
        """``layer_index`` (0-indexed, may be traced under the stacked init) sets the DIFF ``lambda_init``."""
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        std = cfg.initializer_std
        attn_gate = (
            reshard(jnp.zeros((d, n * h)), P(None, "model"))
            if cfg.attn_gate_elementwise
            else reshard(jnp.zeros((d, n)), P(None, None))
        )
        if cfg.mla:
            k_q, k_dkv, k_uk, k_uv, k_o, k_rel, k_ve = random.split(key, 7)
            # A separate key stream, so turning mla_diff_attn on leaves every other initial weight unchanged.
            k_q2, k_uk2, k_lam = random.split(random.fold_in(key, 1), 3)
            kvl = cfg.mla_kv_latent_dim
            use_ve = cfg.value_embeds != ValueEmbeds.NONE
            diff = cfg.mla_diff_attn
            return CausalSelfAttention(
                w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
                w_k=None,
                w_v=None,
                w_o=reshard(_init_weight(k_o, (n * h, d), std * cfg.init_std_mult_attn_out), P("model", _FSDP_AXES)),
                attn_gate=attn_gate,
                sconv_k=(ShortConv.init(n * h, cfg.sconv_kernel) if cfg.sconv and "k" in cfg.sconv_sites else None),
                sconv_q=(ShortConv.init(n * h, cfg.sconv_kernel) if cfg.sconv and "q" in cfg.sconv_sites else None),
                # Without Inkling the MLA layers are NoPE (they are global, so RoPE is disabled there).
                rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
                w_dkv=reshard(_init_weight(k_dkv, (cfg.kv_in_dim, kvl), std), P(_FSDP_AXES, None)),
                kv_latent_norm=_learned_rms_norm(cfg, kvl, cfg.layer_norm_eps),
                w_uk=reshard(_init_weight(k_uk, (kvl, n * h), std), P(None, "model")),
                w_uv=reshard(_init_weight(k_uv, (kvl, n * h), std), P(None, "model")),
                value_embed=(
                    reshard(_init_weight(k_ve, (cfg.vocab_size, n * h), std), P(None, None)) if use_ve else None
                ),
                ve_lambda=jnp.array([1.0, 0.0]) if use_ve else None,
                ve_gate=(reshard(jnp.zeros((d, n)), P(None, None)) if cfg.value_embeds == ValueEmbeds.GATED else None),
                qk_mult=_qk_mult_init(cfg, n),
                xsa_scale=(
                    jnp.full((n,), 1.0 if cfg.xsa_mode == "learned" else 0.0, jnp.float32)
                    if cfg.xsa_mode in ("learned", "tanh")
                    else None
                ),
                xsa_gate=reshard(jnp.zeros((d, n)), P(None, None)) if cfg.xsa_mode == "gated" else None,
                head_mix=(reshard(jnp.zeros((d, 2 * n)), P(None, None)) if cfg.mla_head_mix else None),
                ssmax_scale=jnp.zeros((n,), jnp.float32) if cfg.mla_ssmax else None,
                w_q2=reshard(_init_weight(k_q2, (d, n * h), std), P(_FSDP_AXES, "model")) if diff else None,
                w_uk2=reshard(_init_weight(k_uk2, (kvl, n * h), std), P(None, "model")) if diff else None,
                diff_lambda=0.1 * random.normal(k_lam, (4, h), jnp.float32) if diff else None,
                diff_lambda_init=((0.8 - 0.6 * jnp.exp(-0.3 * jnp.asarray(layer_index, jnp.float32))) if diff else None),
                vres_lambda=_vres_lambda_init(cfg),
                bias_q=jnp.zeros((n * h,)) if "qkv" in cfg.proj_biases else None,
                bias_dkv=jnp.zeros((kvl,)) if "qkv" in cfg.proj_biases else None,
                v_filter_w=jnp.zeros((n, h), jnp.float32) if cfg.mla_v_filter else None,
                v_filter_b=jnp.full((n,), cfg.mla_v_filter_bias_init, jnp.float32) if cfg.mla_v_filter else None,
                forget_gate_w=reshard(jnp.zeros((d, n)), P(None, None)) if cfg.mla_forget_gate else None,
                forget_gate_b=(
                    jnp.full((n,), cfg.mla_forget_gate_bias_init, jnp.float32) if cfg.mla_forget_gate else None
                ),
                cfg=cfg,
            )
        if "qkv" in cfg.proj_biases:
            raise ValueError("proj_biases 'qkv' is implemented for MLA and KDA only")
        if cfg.mla_diff_attn:
            raise ValueError("mla_diff_attn needs mla")
        if cfg.value_residual_layers:
            raise ValueError("value_residual_layers is implemented for MLA and KDA only")
        if cfg.mla_v_filter:
            raise ValueError("mla_v_filter needs mla")
        if cfg.mla_forget_gate:
            raise ValueError("mla_forget_gate needs mla")
        k_q, k_k, k_v, k_o, k_rel = random.split(key, 5)
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (cfg.kv_in_dim, m * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (cfg.kv_in_dim, m * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std * cfg.init_std_mult_attn_out), P("model", _FSDP_AXES)),
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
            qk_mult=_qk_mult_init(cfg, n),
            xsa_scale=(
                jnp.full((n,), 1.0 if cfg.xsa_mode == "learned" else 0.0, jnp.float32)
                if cfg.xsa_mode in ("learned", "tanh")
                else None
            ),
            xsa_gate=reshard(jnp.zeros((d, n)), P(None, None)) if cfg.xsa_mode == "gated" else None,
            head_mix=None,
            ssmax_scale=None,
            w_q2=None,
            w_uk2=None,
            diff_lambda=None,
            diff_lambda_init=None,
            vres_lambda=None,
            bias_q=None,
            bias_dkv=None,
            v_filter_w=None,
            v_filter_b=None,
            forget_gate_w=None,
            forget_gate_b=None,
            cfg=cfg,
        )

    def _mla_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        token_ids: Int[Array, "B S"] | None,
        kv_share: dict[str, jax.Array] | None = None,
        proj_inputs: dict[str, jax.Array] | None = None,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None]:
        """MLA with a compressed KV latent: full-rank q; k and v up-projected from one normed latent (of
        ``kv_input``, the KV side stream, when given). The last element is the second (q, k) pair of
        ``mla_diff_attn`` (same latent and SConvs), else None."""
        assert self.w_dkv is not None and self.kv_latent_norm is not None
        head_dim = self.cfg.inferred_head_dim
        proj_inputs = proj_inputs or {}
        q_in = proj_inputs.get("q", x)
        latent = jnp.einsum("bsh,hl->bsl", x if kv_input is None else kv_input, self.w_dkv)
        if self.bias_dkv is not None:
            latent = latent + unshard(self.bias_dkv).astype(x.dtype)

        def project_q(w_q: jax.Array) -> jax.Array:
            q_flat = jnp.einsum("bsh,hd->bsd", q_in, w_q)
            if self.bias_q is not None:
                q_flat = q_flat + unshard(self.bias_q).astype(x.dtype)
            if self.sconv_q is not None:
                q_flat = self.sconv_q(q_flat, sconv_segment_ids)
            return rearrange(q_flat, "... (n d) -> ... n d", d=head_dim)

        q = project_q(self.w_q)
        share_latent = kv_share is not None and self.cfg.mla_share_kv_latent
        if share_latent and "latent" in kv_share:
            kv_latent = kv_share["latent"]
        else:
            kv_latent = self.kv_latent_norm(latent)
            if share_latent:
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

        def project_k(w_uk: jax.Array) -> jax.Array:
            k_flat = jnp.einsum("bsl,ld->bsd", k_latent, w_uk)
            if self.sconv_k is not None:
                k_flat = self.sconv_k(k_flat, sconv_segment_ids)
            return rearrange(k_flat, "... (n d) -> ... n d", d=head_dim)

        k = project_k(self.w_uk)
        second_qk = None
        if self.w_q2 is not None and self.w_uk2 is not None:
            second_qk = (project_q(self.w_q2), project_k(self.w_uk2))
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
        return q, k, v, second_qk

    def _gqa_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        is_global: bool | jax.Array,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.w_k is not None and self.w_v is not None
        head_dim = self.cfg.inferred_head_dim
        kv_in = x if kv_input is None else kv_input
        q_flat = jnp.einsum("bsh,hd->bsd", x, self.w_q)
        k_flat = jnp.einsum("bsh,hd->bsd", kv_in, self.w_k)
        v_flat = jnp.einsum("bsh,hd->bsd", kv_in, self.w_v)
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
        value_residual: bool = False,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``kv_share`` (MLA only) is a per-forward mailbox for ``mla_share_kv_latent`` and
        ``value_residual_layers``; ``proj_inputs`` optionally replaces the input of the ``q`` / ``k`` / ``v``
        projections; ``value_residual`` (static) mixes the first layer's values into this layer's;
        ``kv_input`` (``kv_stream_dim``) is the normed KV side stream the K/V projections read instead of ``x``.
        Returns the output and logging-only stats (``mla_v_filter``)."""
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()
        # segment_ids (packed-document boundaries) come from the mask so the SConv never mixes across a
        # document boundary.
        sconv_segment_ids = _sconv_segment_ids(mask)
        second_qk = None
        if self.cfg.mla:
            q, k, v, second_qk = self._mla_qkv(x, sconv_segment_ids, token_ids, kv_share, proj_inputs, kv_input)
            if self.vres_lambda is not None:
                assert kv_share is not None
                v = _value_residual(v, self.vres_lambda, kv_share, value_residual)
        else:
            q, k, v = self._gqa_qkv(x, sconv_segment_ids, is_global, kv_input)
        stats: dict[str, jax.Array] = {}
        if self.v_filter_w is not None and self.v_filter_b is not None:
            # Noise filter: each value read is kept by sigmoid(w_h . v_j + b_h).
            head_axis = _padded_spec(v)[2]
            w = reshard(self.v_filter_w, P(head_axis, None))
            gate_logits = jnp.einsum("bsnd,nd->bsn", v.astype(jnp.float32), w) + reshard(self.v_filter_b, P(head_axis))
            v_gate = jax.nn.sigmoid(gate_logits)
            v = v * v_gate[..., None].astype(v.dtype)
            v_gate = jax.lax.stop_gradient(v_gate)
            stats[f"{_LAYER_KNOB_PREFIX}vfilter_gate_mean"] = jnp.mean(v_gate)
            stats[f"{_LAYER_KNOB_PREFIX}vfilter_frac_lt0p1"] = jnp.mean((v_gate < 0.1).astype(jnp.float32))
        fox_key_bias = None
        # The FoX key channels reach the kernel's 1/sqrt(padded head_dim) scale; q is rescaled to keep q.k exact.
        fox_q_scale = 1.0
        if self.forget_gate_w is not None and self.forget_gate_b is not None:
            # FoX: logit (i, j) += c_i - c_j, c the per-document cumsum of log f; only -c_j survives the softmax
            # (the kernel applies no soft-cap or other nonlinearity to the logits).
            gate_logits = jnp.einsum("bsd,dn->bsn", x.astype(jnp.float32), self.forget_gate_w.astype(jnp.float32))
            log_forget = jax.nn.log_sigmoid(gate_logits + self.forget_gate_b)
            fox_key_bias = -_segment_cumsum(log_forget, sconv_segment_ids)
            fox_q_scale = math.sqrt(_fox_head_dim(head_dim) / head_dim)
            log_forget = jax.lax.stop_gradient(log_forget)
            stats[f"{_LAYER_KNOB_PREFIX}fox_gate_mean"] = jnp.mean(jnp.exp(log_forget))
            stats[f"{_LAYER_KNOB_PREFIX}fox_neg_log_gate_mean"] = -jnp.mean(log_forget)
            stats[f"{_LAYER_KNOB_PREFIX}fox_gate_min_head"] = jnp.min(jnp.mean(jnp.exp(log_forget), axis=(0, 1)))

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

        def _transform_qk(q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Key offset, q/k norms, RoPE, qk_mult and SSMax: everything between the projections and the kernel."""
            if self.cfg.mla and self.cfg.mla_key_offset:
                k = _partial_key_offset(k, sconv_segment_ids)
            if self.cfg.qk_norm:
                q = rms_norm(q)
                k = rms_norm(k)
            elif self.cfg.mla_k_norm and self.cfg.mla:
                # DeepSeek-V4.1: weightless per-head RMSNorm on the keys only; queries keep their scale.
                if self.cfg.mla_k_norm_shared:
                    # One RMS over all heads' keys per token: 1 cached scalar per token under MLA absorption
                    # instead of one per head.
                    k32 = k.astype(jnp.float32)
                    k = (k32 * jax.lax.rsqrt(jnp.mean(jnp.square(k32), axis=(-2, -1), keepdims=True) + 1e-6)).astype(
                        k.dtype
                    )
                elif self.cfg.mla_k_norm_split:
                    # With the partial key offset, normalize the previous-token and current-token halves separately.
                    half = k.shape[-1] // 2
                    k = jnp.concatenate([rms_norm(k[..., :half]), rms_norm(k[..., half:])], axis=-1)
                else:
                    k = rms_norm(k)
            if self.cfg.mla_q_norm and self.cfg.mla and not self.cfg.qk_norm:
                q = rms_norm(q)
            # The Inkling bias replaces RoPE (no rotation).
            if self.rel_pos is None:
                if isinstance(disable_rope, bool):
                    if not disable_rope:
                        q, k = _rope(q, k)
                else:
                    q_roped, k_roped = _rope(q, k)
                    keep = ~jnp.asarray(disable_rope, dtype=jnp.bool_)
                    q = jnp.where(keep, q_roped, q)
                    k = jnp.where(keep, k_roped, k)
            if self.qk_mult is None:
                q = q * (self.cfg.qk_mult * fox_q_scale)
            else:
                qk_mult = (self.qk_mult * fox_q_scale).astype(q.dtype)
                q = q * (qk_mult[:, None] if qk_mult.ndim else qk_mult)
            if self.ssmax_scale is not None:
                log_keys = jnp.log1p(_positions_in_document(sconv_segment_ids, seq_len).astype(jnp.float32))
                q = q * (1.0 + self.ssmax_scale[:, None] * log_keys[..., None, None]).astype(q.dtype)
            return q, k

        # The Inkling bias: a per-head content-dependent bias (from x) on the pre-softmax logits.
        rel_bias = self.rel_pos(x) if self.rel_pos is not None else None
        q, k = _transform_qk(q, k)
        # The fa4-cute kernel is GPU-only; fall back to auto-select off-GPU so the model still lowers
        # on CPU (e.g. the grug variant-contract tests).
        attn_impl = "gpu_fa4_cute" if jax.default_backend() == "gpu" else None

        def _attend(qh: jax.Array, kh: jax.Array) -> jax.Array:
            if fox_key_bias is None:
                return attention(qh, kh, v, mask, implementation=attn_impl, rel_bias=rel_bias)
            qh, kh, vh = _fox_augment(qh, kh, v, fox_key_bias)
            return attention(qh, kh, vh, mask, implementation=attn_impl, rel_bias=rel_bias)[..., :head_dim]

        attn_out = _attend(q, k)
        if second_qk is not None:
            # Differential attention: subtract lambda times a second softmax map over the same values, then
            # RMS-normalize each head and scale by (1 - lambda_init).
            assert self.diff_lambda is not None and self.diff_lambda_init is not None
            attn_out2 = _attend(*_transform_qk(*second_qk))
            lq1, lk1, lq2, lk2 = self.diff_lambda
            lambda_init = jax.lax.stop_gradient(self.diff_lambda_init)
            lam = jnp.exp(jnp.sum(lq1 * lk1)) - jnp.exp(jnp.sum(lq2 * lk2)) + lambda_init
            diff_out = attn_out.astype(jnp.float32) - lam * attn_out2.astype(jnp.float32)
            attn_out = (rms_norm(diff_out) * (1.0 - lambda_init)).astype(attn_out.dtype)
        # Exclusive Self Attention (XSA): subtract the component of yᵢ parallel to vᵢ, per head.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ.
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        # GPU XSA with GQA can give attn_out a backend-specific head sharding;
        # match v to that dynamic sharding before the per-head projection math.
        aligned_v = reshard(aligned_v, _partition_spec_of(attn_out) or P(_BATCH_AXES, None, None, "model"))
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        xsa = (dot / (v_norm_sq + 1e-6)) * aligned_v
        if self.xsa_scale is not None:
            scale = jnp.tanh(self.xsa_scale) if self.cfg.xsa_mode == "tanh" else self.xsa_scale
            xsa = xsa * scale.astype(xsa.dtype)[:, None]
        elif self.xsa_gate is not None:
            xsa = xsa * (2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.xsa_gate)))[..., None].astype(xsa.dtype)
        attn_out = attn_out - xsa
        if self.head_mix is not None:
            n_heads = attn_out.shape[2]
            w = jnp.einsum("bsd,dm->bsm", x, self.head_mix).astype(jnp.float32)
            # w1 = 1 + x @ W_1 (a head-average at init), w2 = x @ W_2 (zero at init: no mixing yet).
            w1, w2 = 1.0 + w[..., :n_heads], w[..., n_heads:]
            mixed = jnp.einsum("bsg,bsgd->bsd", w1 / n_heads, attn_out.astype(jnp.float32))
            attn_out = attn_out + (w2[..., None] * mixed[:, :, None, :]).astype(attn_out.dtype)
        # Headwise gating: sigmoid(x @ attn_gate) produces one scalar per head (or per channel).
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))
        gate = rearrange(gate, "... (n d) -> ... n d", d=head_dim) if self.cfg.attn_gate_elementwise else gate[..., None]
        attn_out = gate * attn_out
        # Merge heads into hidden dim while keeping model-axis sharding for w_o.
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:-2], attn_out.shape[-2] * attn_out.shape[-1]),
            out_sharding=P(_BATCH_AXES, None, "model"),
        )
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec), stats


# FoX key bias channels: a bf16 hi/mid/lo split of -c_j (bf16 alone is off by ~|c| * 2^-9 logits; with three
# parts the fp32 kernel accumulator, ~|c| * 2^-24, is the floor).
_FOX_BIAS_CHANNELS = 3
# FA4/CuTe pads head_dim to a multiple of 32 internally, so padding to it costs no extra kernel FLOPs.
_FOX_HEAD_DIM_GRANULE = 32


def _fox_head_dim(head_dim: int) -> int:
    """Kernel head_dim with the FoX bias channels appended, rounded up to the kernel's 32-channel granule."""
    return -(-(head_dim + _FOX_BIAS_CHANNELS) // _FOX_HEAD_DIM_GRANULE) * _FOX_HEAD_DIM_GRANULE


def _fox_augment(
    q: Float[Array, "B S N H"], k: Float[Array, "B S N H"], v: Float[Array, "B S N H"], key_bias: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Zero-pad q/k/v to ``_fox_head_dim`` with q carrying 1s and k carrying ``key_bias * sqrt(padded)`` split into
    bf16 parts in the first ``_FOX_BIAS_CHANNELS`` extra channels, so the kernel's ``q'.k' / sqrt(padded)`` adds
    ``key_bias`` [B, S, N] per key. ``q`` must already carry the ``sqrt(padded / H)`` rescale."""
    head_dim = q.shape[-1]
    padded = _fox_head_dim(head_dim)
    extra = padded - head_dim
    head_spec = _padded_spec(k)[:3]
    rest = reshard(key_bias.astype(jnp.float32), P(*head_spec)) * math.sqrt(padded)
    parts = []
    for _ in range(_FOX_BIAS_CHANNELS):
        part = rest.astype(k.dtype)
        parts.append(part)
        rest = rest - part.astype(jnp.float32)
    k_tail = jnp.pad(jnp.stack(parts, axis=-1), ((0, 0), (0, 0), (0, 0), (0, extra - _FOX_BIAS_CHANNELS)))
    q_tail_values = jnp.asarray([1.0] * _FOX_BIAS_CHANNELS + [0.0] * (extra - _FOX_BIAS_CHANNELS), q.dtype)
    q_tail = reshard(jnp.broadcast_to(q_tail_values, (*q.shape[:-1], extra)), P(*_padded_spec(q)[:3], None))
    pad_last = ((0, 0), (0, 0), (0, 0), (0, extra))
    return jnp.concatenate([q, q_tail], axis=-1), jnp.concatenate([k, k_tail], axis=-1), jnp.pad(v, pad_last)


def _learned_knob_stats(layer: "Block", i: int) -> dict[str, jax.Array]:
    """Values of the small learned knobs of layer ``i`` (value residual mix, DIFF lambda, DyT alpha, PLE
    up-projection norm), exported as ``train/attn_res/knob_*`` to diagnose how each feature is used."""
    stats = {}
    if layer.attn.cfg.attn_res_key_rank is not None:
        # LR-AttnRes: norms of this layer's r-wide pseudo-queries (0 at init = uniform routing).
        for name, query in (("attn", layer.attn_res_query_attn), ("mlp", layer.attn_res_query_mlp)):
            stats[f"attn_res_knob_lrkey_qnorm_{name}_L{i}"] = jnp.linalg.norm(jax.lax.stop_gradient(query))
    if layer.attn.cfg.kv_stream_dim:
        # The main layer's K/V projections, which read the KV side stream.
        names = ("w_dkv",) if isinstance(layer.attn, CausalSelfAttention) and layer.attn.cfg.mla else ("w_k", "w_v")
        for name in names:
            weight = jax.lax.stop_gradient(getattr(layer.attn, name)).astype(jnp.float32)
            stats[f"attn_res_knob_kv_proj_{name}_norm_L{i}"] = jnp.linalg.norm(weight)
    router_tok_b = getattr(layer.mlp, "router_tok_b", None)
    if router_tok_b is not None:
        stats[f"attn_res_knob_router_tok_b_norm_L{i}"] = jnp.linalg.norm(
            jax.lax.stop_gradient(router_tok_b).astype(jnp.float32)
        )
    vres = getattr(layer.attn, "vres_lambda", None)
    if vres is not None:
        stats[f"attn_res_knob_vres_l1_L{i}"], stats[f"attn_res_knob_vres_l2_L{i}"] = jax.lax.stop_gradient(vres)
    if isinstance(layer.attn, CausalSelfAttention) and layer.attn.diff_lambda is not None:
        lq1, lk1, lq2, lk2 = jax.lax.stop_gradient(layer.attn.diff_lambda)
        stats[f"attn_res_knob_diff_lambda_L{i}"] = (
            jnp.exp(jnp.dot(lq1, lk1)) - jnp.exp(jnp.dot(lq2, lk2)) + layer.attn.diff_lambda_init
        )
    for name in ("rms_attn", "rms_mlp"):
        norm = getattr(layer, name)
        if isinstance(norm, DyT):
            stats[f"attn_res_knob_dyt_alpha_{name.removeprefix('rms_')}_L{i}"] = jax.lax.stop_gradient(norm.dyt_alpha)
    if isinstance(layer.attn, CausalSelfAttention) and layer.attn.forget_gate_b is not None:
        stats[f"attn_res_knob_fox_bias_gate_mean_L{i}"] = jnp.mean(
            jax.nn.sigmoid(jax.lax.stop_gradient(layer.attn.forget_gate_b))
        )
        stats[f"attn_res_knob_fox_w_norm_L{i}"] = jnp.linalg.norm(
            jax.lax.stop_gradient(layer.attn.forget_gate_w).astype(jnp.float32)
        )
    rot_scale = getattr(layer.attn, "rot_scale", None)
    if rot_scale is not None:
        freq = jnp.abs(jax.lax.stop_gradient(rot_scale).astype(jnp.float32)) * _kda_rot_omega(2 * rot_scale.shape[-1])
        stats[f"attn_res_knob_kda_rot_freq_mean_L{i}"] = jnp.mean(freq)
        stats[f"attn_res_knob_kda_rot_freq_max_L{i}"] = jnp.max(freq)
    if isinstance(layer.mlp, MoEMLP):
        stats.update(_router_knob_stats(layer.mlp, i))
    if layer.ple_up is not None:
        stats[f"attn_res_knob_ple_up_norm_L{i}"] = jnp.linalg.norm(
            jax.lax.stop_gradient(layer.ple_up).astype(jnp.float32)
        )
    if isinstance(layer.attn, KimiDeltaAttention) and layer.attn.comba_d is not None:
        comba_d = jax.lax.stop_gradient(layer.attn.comba_d)
        stats[f"attn_res_knob_comba_d_L{i}"] = jnp.mean(comba_d)
        stats[f"attn_res_knob_comba_d_min_L{i}"] = jnp.min(comba_d)
        stats[f"attn_res_knob_comba_d_max_L{i}"] = jnp.max(comba_d)
    if isinstance(layer.mlp, MoEMLP) and layer.mlp.cfg.latent_out_dim is not None:
        em = layer.mlp.expert_mlp
        read = [em.w_up] if em.w_gate is None else [em.w_gate, em.w_up]
        stats[f"attn_res_knob_expert_read_pr_L{i}"] = _participation_ratio(read, "eri,esi->rs")
        stats[f"attn_res_knob_expert_write_pr_L{i}"] = _participation_ratio([em.w_down], "eir,eis->rs")
    if isinstance(layer.mlp, MoEMLP) and layer.mlp.expert_read_norm is not None:
        norm = layer.mlp.expert_read_norm
        gain = norm.weight if isinstance(norm, RMSNorm) else 1.0 + norm.gamma
        gain = jnp.abs(jax.lax.stop_gradient(gain).astype(jnp.float32))
        for g in range(gain.shape[0]):
            stats[f"attn_res_knob_expert_read_gain_g{g}_L{i}"] = jnp.mean(gain[g])
    # Mean |gamma| of the zero-centered gains: the attention / MLP pre-norms, and every norm inside the mixer
    # and MoE (KV latent, KDA output, latent / router norms) plus the Peri-LN output norms.
    groups = {
        "attn": [layer.rms_attn],
        "mlp": [layer.rms_mlp],
        "inner": [layer.attn, layer.mlp, layer.out_norm_attn, layer.out_norm_mlp],
    }
    for name, modules in groups.items():
        gammas = [g for m in modules for g in _zero_centered_gammas(m)]
        if gammas:
            flat = jnp.concatenate([jax.lax.stop_gradient(g).reshape(-1) for g in gammas])
            stats[f"attn_res_knob_gain_abs_{name}_L{i}"] = jnp.mean(jnp.abs(flat))
    return stats


def _router_knob_stats(mlp: "MoEMLP", i: int) -> dict[str, jax.Array]:
    """Router logit scale, per-expert output gains and, with SimBal or the logit scale, the router's Frobenius
    norm and mean |cosine| between its real-expert columns."""
    sg = jax.lax.stop_gradient
    cfg = mlp.cfg
    stats = {}
    if mlp.router_logit_scale is not None:
        stats[f"attn_res_knob_router_logit_scale_L{i}"] = sg(mlp.router_logit_scale)
    if mlp.expert_output_gain is not None:
        gain = sg(mlp.expert_output_gain[: cfg.num_experts]).astype(jnp.float32)
        stats[f"attn_res_knob_expert_gain_mean_L{i}"] = jnp.mean(gain)
        stats[f"attn_res_knob_expert_gain_std_L{i}"] = jnp.std(gain)
        stats[f"attn_res_knob_expert_gain_min_L{i}"] = jnp.min(gain)
        stats[f"attn_res_knob_expert_gain_max_L{i}"] = jnp.max(gain)
    if mlp.router is not None and (cfg.simbal_loss_weight > 0 or cfg.router_logit_scale):
        r = reshard(sg(mlp.router[:, : cfg.num_experts]).astype(jnp.float32), P(None, None))
        norms = jnp.sqrt(jnp.sum(jnp.square(r), axis=0))
        cos = jnp.einsum("de,df->ef", r / norms, r / norms, out_sharding=P(None, None))
        n = cfg.num_experts
        stats[f"attn_res_knob_router_norm_L{i}"] = jnp.sqrt(jnp.sum(jnp.square(norms)))
        stats[f"attn_res_knob_router_col_cos_abs_L{i}"] = (jnp.sum(jnp.abs(cos)) - n) / (n * (n - 1))
    return stats


def _participation_ratio(weights: list[jax.Array], gram_spec: str) -> jax.Array:
    """``tr(G)^2 / tr(G^2)`` of the summed Gram ``G`` of the expert banks over one side (``gram_spec`` contracts
    the experts and the other side): the number of input (read) or output (write) channels the experts use."""
    gram = functools.reduce(
        jnp.add,
        [
            jnp.einsum(gram_spec, w, w, out_sharding=P(None, None))
            for w in (jax.lax.stop_gradient(w).astype(jnp.float32) for w in weights)
        ],
    )
    return jnp.trace(gram) ** 2 / jnp.sum(jnp.square(gram))


_LATENT_SELECT_SALT = 0x5E1EC7


def _latent_select_mask(cfg: GrugModelConfig, layer_index: jax.Array | int) -> jax.Array | None:
    """Per-layer 0/1 hidden-channel mask for ``latent_select`` with a random or rotating pattern."""
    if not cfg.latent_select or cfg.latent_select_pattern == "first":
        return None
    assert cfg.latent_dim is not None
    d, latent = cfg.hidden_dim, cfg.latent_dim
    if cfg.latent_select_pattern == "random":
        key = random.fold_in(random.PRNGKey(_LATENT_SELECT_SALT), layer_index)
        idx = random.permutation(key, d)[:latent]
    else:
        idx = (jnp.arange(latent) + jnp.asarray(layer_index) * (d // cfg.num_layers)) % d
    return reshard(jnp.zeros((d,), jnp.float32).at[idx].set(1.0), P(None))


def _qk_mult_init(cfg: GrugModelConfig, num_heads: int) -> jax.Array | None:
    if not cfg.learnable_qk_mult:
        return None
    return jnp.full((num_heads,) if cfg.qk_mult_per_head else (), cfg.qk_mult, jnp.float32)


def _vres_lambda_init(cfg: GrugModelConfig) -> jax.Array | None:
    return jnp.asarray(cfg.value_residual_init, jnp.float32) if cfg.value_residual_layers else None


def _value_residual(v: jax.Array, lam: jax.Array, kv_share: dict[str, jax.Array], mix: bool) -> jax.Array:
    """ResFormer value residual: the first layer to run stores its ``v`` in ``kv_share``; a ``mix`` layer
    returns ``l1 * v + l2 * v_first``."""
    if "v_first" not in kv_share:
        kv_share["v_first"] = v
        return v
    if not mix:
        return v
    lam = lam.astype(v.dtype)
    v_first = reshard(kv_share["v_first"], _partition_spec_of(v) or P(_BATCH_AXES, None, None, None))
    return lam[0] * v + lam[1] * v_first


def _kda_dt_bias_init(cfg: GrugModelConfig, key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
    """``dt_bias`` such that ``|g| = KDA_MIN_LOG_DECAY * sigmoid(dt_bias)`` (zero gate input, ``A_log = 0``)
    is log-uniform in ``kda_dt_range``."""
    lo, hi = cfg.kda_dt_range
    decay = jnp.exp(random.uniform(key, shape, minval=math.log(lo), maxval=math.log(hi)))
    return jax.scipy.special.logit(decay / KDA_MIN_LOG_DECAY)


def _kda_push_decay_init(cfg: GrugModelConfig, num_heads: int, head_dim: int) -> jax.Array:
    """Push-bucket decay logits: bucket m's |g| is log-spaced across ``kda_push_decay_range`` (slow first)."""
    lo, hi = cfg.kda_push_decay_range
    mags = jnp.exp(jnp.linspace(math.log(lo), math.log(hi), cfg.kda_push_buckets))
    logits = jax.scipy.special.logit(mags / KDA_MIN_LOG_DECAY)
    return jnp.broadcast_to(logits[:, None, None], (cfg.kda_push_buckets, num_heads, head_dim)).astype(jnp.float32)


def _kda_rot_omega(head_dim: int) -> jax.Array:
    """``kda_dd_rope`` base frequencies: one per channel pair, log-spaced over ``_KDA_ROT_OMEGA_RANGE`` (fast first)."""
    lo, hi = _KDA_ROT_OMEGA_RANGE
    return jnp.exp(jnp.linspace(math.log(hi), math.log(lo), head_dim // 2))


def _segment_cumsum(x: jax.Array, segment_ids: jax.Array | None) -> jax.Array:
    """Cumulative sum of ``x`` ``(B, S, ...)`` over ``S``, restarting at every packed-document start."""
    if segment_ids is None:
        return jnp.cumsum(x, axis=1)
    starts = doc_starts(segment_ids).astype(bool)
    starts = jnp.broadcast_to(starts.reshape(*starts.shape, *(1,) * (x.ndim - 2)), x.shape)

    def combine(left, right):
        (lv, lr), (rv, rr) = left, right
        return jnp.where(rr, rv, lv + rv), lr | rr

    return jax.lax.associative_scan(combine, (x, starts), axis=1)[0]


def _kda_rotate_qk(q, k, rate, segment_ids):
    """Rotate channel pairs ``(j, j + h/2)`` of q and k by ``theta = segment_cumsum(rate)``: the RoPE trick
    for the rotated recurrence ``S_t = (I - beta k k^T) exp(g) R(rate_t) S_{t-1} + beta k v^T`` (pair-tied g).
    Rotations preserve the norm, so this commutes with the kernel's q/k L2 normalization."""
    theta = _segment_cumsum(rate, segment_ids)
    cos, sin = jnp.cos(theta), jnp.sin(theta)

    def rotate(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half].astype(jnp.float32), x[..., half:].astype(jnp.float32)
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)

    return rotate(q), rotate(k)


def _kda_kernel_rotating(q, k, v, g, beta, extras: dict[str, jax.Array], *, save_chunk_states: bool):
    """``_kda_kernel`` after rotating q and k by ``extras["rot_rate"]`` (``kda_dd_rope``)."""
    extras = dict(extras)
    q, k = _kda_rotate_qk(q, k, extras.pop("rot_rate"), extras.get("segment_ids"))
    return _kda_kernel(q, k, v, g, beta, extras, save_chunk_states=save_chunk_states)


def _kda_kernel(q, k, v, g, beta, extras: dict[str, jax.Array], *, save_chunk_states: bool):
    """KDA on the model layout ``(B, S, H, d)``: the fused Pallas kernels on GPU, else the XLA
    ``chunk_kda`` (heads-first layout). ``extras`` optionally holds ``segment_ids`` and ``erase``."""
    segment_ids, erase = extras.get("segment_ids"), extras.get("erase")
    if jax.default_backend() == "gpu":
        return kda_fused(
            q,
            k,
            v,
            g,
            beta,
            segment_ids=segment_ids,
            erase=erase,
            chunk_size=KDA_CHUNK_SIZE,
            save_chunk_states=save_chunk_states,
        )
    q, k, v, g, beta = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta))
    seg = None if segment_ids is None else segment_ids[:, None, :]  # same documents for every head
    erase = None if erase is None else jnp.swapaxes(erase, 1, 2)
    out = chunk_kda(q, k, v, g, beta, chunk_size=KDA_CHUNK_SIZE, segment_ids=seg, erase=erase)[0]
    return jnp.swapaxes(out, 1, 2)


class KimiDeltaAttention(eqx.Module):
    """KDA linear-attention token mixer for the local layers, following Kimi K3's KDA layer.

    Per head (``N`` heads of width ``h``): q/k/v are bias-free ``D -> N*h`` projections (no GQA), each
    through a depthwise causal ShortConv + SiLU; q/k are L2-normalized in the kernel (q scaled by
    ``1/sqrt(h)``). ``beta = sigmoid(x W_beta)`` per head, and the per-channel log-decay
    ``g = -5 * sigmoid(exp(A_log) * (x W_a_down W_a_up + dt_bias))`` lies in ``(-5, 0)``. The state
    follows ``S_t = (I - beta k k^T) Diag(exp(g)) S_{t-1} + beta k v^T``, read as ``o_t = S_t^T q_t``,
    and is hard-reset at packed-document starts. The output gets a per-head RMSNorm with a learnable
    ``h``-dim scale shared across heads, a full-rank per-channel gate ``sigmoid(x W_g)``, and ``w_o``.
    No positional encoding (unless ``kda_dd_rope``) and no window. The recurrence runs under ``shard_map``
    (batch on the batch axes, heads on ``model``).
    """

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D NH"]
    w_v: Float[Array, "D NH"]
    w_o: Float[Array, "NH D"]
    w_g: Float[Array, "D NH"]  # [D, N] with cfg.kda_gate_per_head
    w_write: Float[Array, "D NH"] | None  # channel-wise value write gate (cfg.kda_write_gate)
    w_erase: Float[Array, "D NH"] | None  # channel-wise key erase gate, zero-init (cfg.kda_erase_gate)
    w_a_down: Float[Array, "D R"]
    w_a_up: Float[Array, "R NH"]
    a_log: Float[Array, " N"]
    dt_bias: Float[Array, "N H"]
    w_beta: Float[Array, "D N"] | None
    w_beta_down: Float[Array, "D R"] | None
    w_beta_up: Float[Array, "R N"] | None
    o_norm: "LearnedRMSNorm"
    bias_qkv: Float[Array, "3 NH"] | None
    sconv_q: ShortConv
    sconv_k: ShortConv
    sconv_v: ShortConv
    sconv_a: ShortConv | None
    vres_lambda: Float[Array, " 2"] | None  # (l1 on v, l2 on the first layer's v): cfg.value_residual_layers
    push_decay: Float[Array, "M N H"] | None
    """Per-bucket, per-channel log-decay logits of the push buckets (``kda_push_buckets``)."""
    w_push: Float[Array, "D NM"] | None
    """Writer's bucket weights, zero-init (uniform split)."""
    w_rot_down: Float[Array, "D R"] | None
    w_rot_up: Float[Array, "R NP"] | None
    rot_scale: Float[Array, "N P"] | None
    """``kda_dd_rope`` per-pair angle amplitude ``gamma`` (zero-init: no rotation at init)."""
    comba_d: Float[Array, " N"] | None
    """Per-head Comba output-correction scalar ``d`` (``kda_out_correction``)."""
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KimiDeltaAttention":
        k_q, k_k, k_v, k_o, k_g, k_ad, k_au, k_b, k_dt = random.split(key, 9)
        d, n, h, r, std = cfg.hidden_dim, cfg.num_heads, cfg.inferred_head_dim, _KDA_GATE_RANK, cfg.initializer_std
        # kda_dd_rope ties the decay across each rotated channel pair: h/2 decay columns per head.
        decay_cols = 1 if cfg.kda_decay_per_head else (h // 2 if cfg.kda_dd_rope else h)
        rot_rank = cfg.kda_dd_rope_rank
        return KimiDeltaAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (cfg.kv_in_dim, n * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (cfg.kv_in_dim, n * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std * cfg.init_std_mult_attn_out), P("model", _FSDP_AXES)),
            w_g=reshard(
                _init_weight(k_g, (d, n if cfg.kda_gate_per_head else n * h), std * cfg.init_std_mult_gates),
                P(None, None) if cfg.kda_gate_per_head else P(_FSDP_AXES, "model"),
            ),
            w_write=(
                reshard(_init_weight(random.fold_in(k_v, 7), (d, n * h), std), P(_FSDP_AXES, "model"))
                if cfg.kda_write_gate
                else None
            ),
            w_erase=reshard(jnp.zeros((d, n * h)), P(_FSDP_AXES, "model")) if cfg.kda_erase_gate else None,
            w_a_down=reshard(_init_weight(k_ad, (d, r), std), P(_FSDP_AXES, None)),
            w_a_up=reshard(
                _init_weight(k_au, (r, n * decay_cols), std),
                P(None, None) if cfg.kda_decay_per_head else P(None, "model"),
            ),
            a_log=jnp.zeros((n,)),
            dt_bias=_kda_dt_bias_init(cfg, k_dt, (n, decay_cols)),
            w_beta=(
                None
                if cfg.kda_beta_rank
                else reshard(_init_weight(k_b, (d, n), std * cfg.init_std_mult_gates), P(None, None))
            ),
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
            o_norm=_learned_rms_norm(cfg, h, 1e-6),
            bias_qkv=jnp.zeros((3, n * h)) if "qkv" in cfg.proj_biases else None,
            sconv_q=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_k=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_v=ShortConv.init(n * h, cfg.sconv_kernel),
            sconv_a=ShortConv.init(r, cfg.sconv_kernel) if cfg.kda_decay_conv else None,
            vres_lambda=_vres_lambda_init(cfg),
            push_decay=_kda_push_decay_init(cfg, n, h) if cfg.kda_push_buckets else None,
            w_push=(reshard(jnp.zeros((d, n * cfg.kda_push_buckets)), P(None, None)) if cfg.kda_push_buckets else None),
            w_rot_down=(
                reshard(_init_weight(random.fold_in(k_ad, 1), (d, rot_rank), std), P(_FSDP_AXES, None))
                if cfg.kda_dd_rope
                else None
            ),
            w_rot_up=(
                reshard(
                    _init_weight(random.fold_in(k_au, 1), (rot_rank, n * (h // 2)), 1.0 / math.sqrt(rot_rank)),
                    P(None, "model"),
                )
                if cfg.kda_dd_rope
                else None
            ),
            rot_scale=jnp.zeros((n, h // 2)) if cfg.kda_dd_rope else None,
            comba_d=jnp.full((n,), cfg.kda_out_correction_init, jnp.float32) if cfg.kda_out_correction else None,
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
        kv_share: dict[str, jax.Array] | None = None,
        value_residual: bool = False,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``kv_input`` (``kv_stream_dim``) is the normed KV side stream that the ``k`` / ``v`` projections
        (and their SConvs) read instead of ``x``; the erase / write gates, beta and decay stay on ``x``.
        ``proj_inputs`` optionally replaces the input of the ``q`` / ``k`` / ``v`` projections;
        ``no_decay`` / ``no_beta`` (static, per layer) replace g with 0 / beta with 1; ``kv_share`` /
        ``value_residual`` as in ``CausalSelfAttention``. Also returns logging stats (``kda_dd_rope`` rates;
        the erase gate's mean and mean binary entropy with ``kda_erase_gate``)."""
        cfg = self.cfg
        head_dim = cfg.inferred_head_dim
        b, s, _ = x.shape

        proj_inputs = proj_inputs or {}

        def project(w: jax.Array, conv: ShortConv, bias_row: int, name: str) -> jax.Array:
            source = kv_input if kv_input is not None and name in ("k", "v") else proj_inputs.get(name, x)
            y = jnp.einsum("bsh,hd->bsd", source, w)
            if self.bias_qkv is not None:
                y = y + unshard(self.bias_qkv[bias_row]).astype(x.dtype)
            y = jax.nn.silu(conv(y, segment_ids))
            return rearrange(y, "... (n d) -> ... n d", d=head_dim)

        q = project(self.w_q, self.sconv_q, 0, "q")
        k = project(self.w_k, self.sconv_k, 1, "k")
        v = project(self.w_v, self.sconv_v, 2, "v")
        if self.comba_d is not None:
            # Comba output correction on the L2-normalized q / k; the kernel re-normalizes q - d k.
            def unit(t: jax.Array) -> jax.Array:
                return rms_norm(t.astype(jnp.float32)) * head_dim**-0.5

            q = (unit(q) - self.comba_d[:, None] * unit(k)).astype(q.dtype)
        if self.vres_lambda is not None:
            assert kv_share is not None
            v = _value_residual(v, self.vres_lambda, kv_share, value_residual)
        if self.w_write is not None:
            write = jnp.einsum("bsh,hd->bsd", proj_inputs.get("v", x), self.w_write)
            v = v * rearrange(
                2.0 * jax.nn.sigmoid(write.astype(jnp.float32)), "... (n d) -> ... n d", d=head_dim
            ).astype(v.dtype)
        a_low = jnp.einsum("bsd,dr->bsr", x, self.w_a_down)
        if self.sconv_a is not None:
            a_low = self.sconv_a(a_low, segment_ids)
        a = jnp.einsum("bsr,re->bse", a_low, self.w_a_up)
        a = rearrange(a, "... (n d) -> ... n d", d=self.dt_bias.shape[-1]).astype(jnp.float32)
        scale = jnp.exp(self.a_log.astype(jnp.float32))[:, None]
        g = -KDA_MIN_LOG_DECAY * jax.nn.sigmoid(scale * (a + self.dt_bias.astype(jnp.float32)))
        if cfg.kda_decay_per_head:
            g = jnp.broadcast_to(g, (*g.shape[:-1], head_dim))
        elif cfg.kda_dd_rope:
            g = jnp.concatenate([g, g], axis=-1)  # channels j and j + h/2 form one rotated pair
        if self.w_beta_down is not None and self.w_beta_up is not None:
            beta_hidden = jax.nn.silu(jnp.einsum("bsd,dr->bsr", x, self.w_beta_down))
            beta_logits = jnp.einsum("bsr,rn->bsn", beta_hidden, self.w_beta_up.astype(beta_hidden.dtype))
        else:
            assert self.w_beta is not None
            beta_logits = jnp.einsum("bsd,dn->bsn", x, self.w_beta)
        if cfg.kda_beta_negative:
            beta = 2.0 * jax.nn.sigmoid(beta_logits.astype(jnp.float32) - math.log(3.0))
        else:
            beta = jax.nn.sigmoid(beta_logits.astype(jnp.float32))
        if no_decay:
            g = jnp.zeros_like(g)
        if no_beta:
            beta = jnp.ones_like(beta)

        stats: dict[str, jax.Array] = {}
        spec4 = P(_BATCH_AXES, None, "model", None)
        spec3 = P(_BATCH_AXES, None, "model")
        q, k, v, g = (reshard(t, spec4) for t in (q, k, v, g))
        beta = reshard(beta, spec3)
        extras: dict[str, jax.Array] = {}
        extra_specs: dict[str, P] = {}
        if self.w_erase is not None:
            erase_logits = jnp.einsum("bsh,hd->bsd", proj_inputs.get("k", x), self.w_erase).astype(jnp.float32)
            erase_logits = rearrange(erase_logits, "... (n d) -> ... n d", d=head_dim)
            extras["erase"], extra_specs["erase"] = reshard(2.0 * jax.nn.sigmoid(erase_logits), spec4), spec4
            # b / 2 = sigmoid(z): H = -p log p - (1 - p) log(1 - p), max log 2 (the zero-init value).
            p_erase = jax.nn.sigmoid(erase_logits)
            entropy = -(p_erase * jax.nn.log_sigmoid(erase_logits) + (1 - p_erase) * jax.nn.log_sigmoid(-erase_logits))
            stats[f"{_KDA_ERASE_STAT_PREFIX}mean"] = jax.lax.stop_gradient(jnp.mean(2.0 * p_erase))
            stats[f"{_KDA_ERASE_STAT_PREFIX}entropy"] = jax.lax.stop_gradient(jnp.mean(entropy))
        if self.w_rot_down is not None and self.w_rot_up is not None and self.rot_scale is not None:
            rot = jnp.einsum("bsr,re->bse", jnp.einsum("bsd,dr->bsr", x, self.w_rot_down), self.w_rot_up)
            rot = rearrange(rot, "... (n p) -> ... n p", p=head_dim // 2).astype(jnp.float32)
            freq = self.rot_scale.astype(jnp.float32) * _kda_rot_omega(head_dim)
            rate = reshard(jax.nn.softplus(rot) * freq, spec4)  # rad/token per channel pair
            rate_sg = jax.lax.stop_gradient(rate)
            stats[f"{_KDA_STAT_PREFIX}rot_rate_abs_mean"] = jnp.mean(jnp.abs(rate_sg))
            stats[f"{_KDA_STAT_PREFIX}rot_rate_std"] = jnp.std(rate_sg)
            stats[f"{_KDA_STAT_PREFIX}rot_rate_pair_abs_max"] = jnp.max(jnp.mean(jnp.abs(rate_sg), axis=(0, 1)))
            extras["rot_rate"], extra_specs["rot_rate"] = rate, spec4
        if segment_ids is not None:
            extras["segment_ids"] = reshard(jnp.broadcast_to(segment_ids, (b, s)), P(_BATCH_AXES, None))
            extra_specs["segment_ids"] = P(_BATCH_AXES, None)
        kernel_fn = _kda_kernel_rotating if "rot_rate" in extras else _kda_kernel
        run = functools.partial(kernel_fn, save_chunk_states=cfg.kda_save_chunk_states)
        args = (q, k, v, g, beta, extras)
        in_specs = (spec4,) * 4 + (spec3, extra_specs)
        # The Pallas custom VJPs are not vma-annotated, so skip the varying-axes check.
        kernel = jax.shard_map(run, mesh=get_abstract_mesh(), in_specs=in_specs, out_specs=spec4, check_vma=False)
        if self.push_decay is None or self.w_push is None:
            o = kernel(*args)
        else:
            # Push decay: bucket m is a delta-rule state with its own static decay; the writer's pi_m scales
            # its write strength into that bucket. The read sums the buckets.
            num_buckets = cfg.kda_push_buckets
            pi_logits = rearrange(
                jnp.einsum("bsd,de->bse", x, self.w_push).astype(jnp.float32), "b s (n m) -> b s n m", m=num_buckets
            )
            pi = jax.nn.softmax(pi_logits, axis=-1) if cfg.kda_push_weights == "softmax" else jax.nn.sigmoid(pi_logits)
            o = None
            for m in range(num_buckets):
                g_m = -KDA_MIN_LOG_DECAY * jax.nn.sigmoid(self.push_decay[m].astype(jnp.float32))
                g_m = jnp.broadcast_to(g_m, g.shape)
                if cfg.kda_push_mode == "hybrid":
                    # Keep the per-token floor the chunked kernels assume (see KDA_CHUNK_SIZE).
                    g_m = jnp.maximum(g_m + g, -KDA_MIN_LOG_DECAY)
                bucket_args = (q, k, v, reshard(g_m, spec4), reshard(beta * pi[..., m], spec3), extras)
                o_m = kernel(*bucket_args)
                o = o_m if o is None else o + o_m
        o = self.o_norm(o.astype(x.dtype))
        o = jnp.reshape(o, (b, s, cfg.num_heads * head_dim), out_sharding=P(_BATCH_AXES, None, "model"))
        gate = jax.nn.sigmoid(jnp.einsum("bsd,de->bse", x, self.w_g))
        if cfg.kda_gate_per_head:
            gate = jnp.repeat(gate, head_dim, axis=-1, total_repeat_length=cfg.num_heads * head_dim)
        o = o * gate
        return jnp.einsum("bsh,hd->bsd", o, self.w_o, out_sharding=_batch_spec()), stats


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


class ZeroCenteredRMSNorm(eqx.Module):
    """RMSNorm with a zero-centered gain ``1 + gamma`` (``cfg.zero_centered_gains``)."""

    gamma: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "ZeroCenteredRMSNorm":
        return ZeroCenteredRMSNorm(gamma=jnp.zeros((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gain = 1.0 + unshard(self.gamma)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * gain).astype(dtype)


LearnedRMSNorm = RMSNorm | ZeroCenteredRMSNorm


def _learned_rms_norm(cfg: GrugModelConfig, dim: int, eps: float) -> LearnedRMSNorm:
    """A learned-gain RMSNorm, zero-centered under ``cfg.zero_centered_gains``."""
    return ZeroCenteredRMSNorm.init(dim, eps) if cfg.zero_centered_gains else RMSNorm.init(dim, eps)


def _zero_centered_gammas(module: eqx.Module | None) -> list[jax.Array]:
    """The ``gamma`` leaves of every ZeroCenteredRMSNorm inside ``module``."""
    is_norm = lambda x: isinstance(x, ZeroCenteredRMSNorm)  # noqa: E731
    return [n.gamma for n in jax.tree.leaves(module, is_leaf=is_norm) if is_norm(n)]


class DyT(eqx.Module):
    """Dynamic Tanh (arXiv 2503.10622): ``weight * tanh(dyt_alpha * x) + dyt_beta``, a norm-free drop-in for RMSNorm."""

    dyt_alpha: jax.Array
    weight: jax.Array
    dyt_beta: jax.Array

    @staticmethod
    def init(dim: int, alpha: float) -> "DyT":
        return DyT(
            dyt_alpha=jnp.full((), alpha, dtype=jnp.float32),
            weight=jnp.ones((dim,), dtype=jnp.float32),
            dyt_beta=jnp.zeros((dim,), dtype=jnp.float32),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        dtype = x.dtype
        y = jnp.tanh(self.dyt_alpha * x.astype(jnp.float32))
        return (y * unshard(self.weight) + unshard(self.dyt_beta)).astype(dtype)


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


class MtpHead(eqx.Module):
    """Depth-1 DeepSeek-V3 MTP module (arXiv 2412.19437 sec. 2.2): ``x = W_proj [rms(h_t); rms(Emb(x_{t+1}))]``,
    one block ``x + W_down relu(W_up rms(x))^2``, then its own learned RMSNorm into the shared lm_head.

    DeepSeek's block is a full transformer layer. This one is attention-free: ``h_t`` already carries the causal
    context, so the block only has to combine it with the next token's embedding, and at vocab 16k and d512 the
    extra lm_head pass, not the block, is the cost. Being per-position, the whole module runs on just the
    ``mtp_position_frac`` subset of positions. Matrices are random-init (MuonH)."""

    w_proj: Float[Array, "C D"]
    w_up: Float[Array, "D M"]
    w_down: Float[Array, "M D"]
    out_norm: LearnedRMSNorm

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MtpHead":
        k_proj, k_up, k_down = random.split(key, 3)
        d, m, std = cfg.hidden_dim, cfg.mtp_mlp_mult * cfg.hidden_dim, cfg.initializer_std
        return MtpHead(
            w_proj=reshard(_init_weight(k_proj, (2 * d, d), std), P(_FSDP_AXES, None)),
            w_up=reshard(_init_weight(k_up, (d, m), std), P(_FSDP_AXES, "model")),
            w_down=reshard(_init_weight(k_down, (m, d), std), P("model", _FSDP_AXES)),
            out_norm=_learned_rms_norm(cfg, d, cfg.layer_norm_eps),
        )

    @named_call
    def __call__(self, hidden: Float[Array, "B S D"], next_embed: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        b, s, _ = hidden.shape
        x = jnp.concatenate([rms_norm(hidden), rms_norm(next_embed.astype(hidden.dtype))], axis=-1)
        x_flat = jnp.einsum(
            "tc,cd->td", rearrange(x, "b s c -> (b s) c"), self.w_proj.astype(hidden.dtype), out_sharding=_batch_spec()
        )
        up = jnp.einsum("td,dm->tm", rms_norm(x_flat), self.w_up.astype(hidden.dtype))
        x_flat = x_flat + jnp.einsum(
            "tm,md->td", jnp.square(jax.nn.relu(up)), self.w_down.astype(hidden.dtype), out_sharding=_batch_spec()
        )
        return self.out_norm(_batch_reshard(rearrange(x_flat, "(b s) d -> b s d", b=b, s=s)))


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
    target_rank: float | jax.Array,
) -> jax.Array:
    """Per-expert (1-K/E) upper quantile of ``s_local`` via one fused bincount over ``[lo, hi]``.

    ``target_rank`` is the number of tokens at or above each expert's threshold: a scalar, or one per expert.

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
    target_rank = jnp.broadcast_to(jnp.asarray(target_rank, jnp.float32), (num_experts,))
    bstar = jnp.clip(jnp.sum((cum_from_top >= target_rank[:, None]).astype(jnp.int32), axis=-1) - 1, 0, n_bins - 1)
    ct_b = jnp.take_along_axis(cum_from_top, bstar[:, None], axis=-1)[:, 0]
    h_b = jnp.take_along_axis(counts, bstar[:, None], axis=-1)[:, 0]
    lower_edge = lo + bstar.astype(jnp.float32) * bin_width
    return lower_edge + bin_width * (ct_b - target_rank) / jnp.maximum(h_b, 1.0)


def _qb_beta_hist(
    s_ma: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    *,
    target_share: float | jax.Array,
    num_experts: int,
    n_bins: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Global (1-K/E)-quantile of the logit margins over the live ``[min, max]`` grid (this step's).

    A ``pmin``/``pmax`` sets the grid to the exact current range of the margins, then
    ``_bincount_upper_quantile`` reads the per-expert threshold. Replaces the per-device ``top_k`` +
    ``pmean`` estimate with a smoother global quantile at the cost of the per-expert count reduction.
    ``target_share`` is the fraction of tokens each expert should take (``K/E``), or one per expert.

    Returns ``(beta, margin_min, margin_max)``: the per-expert threshold plus the live margin range
    (the grid ``lo``/``hi``), surfaced for logging.
    """
    # Tokens at/above beta per expert.
    target_rank = jnp.broadcast_to(jnp.asarray(float(s_ma.shape[0]) * target_share, jnp.float32), (num_experts,))

    def _fn(s_local: jax.Array, target_rank: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # pmin/pmax have no autodiff rule and the range is a control quantity, so detach their inputs;
        # the bincount path drops tangents at the integer bin cast, so it needs none downstream either.
        lo = jax.lax.pmin(jax.lax.stop_gradient(jnp.min(s_local)), axis_name=_BATCH_AXES)
        hi = jax.lax.pmax(jax.lax.stop_gradient(jnp.max(s_local)), axis_name=_BATCH_AXES)
        hi_grid = jnp.maximum(hi, lo + 1e-6)  # guard a degenerate all-equal range
        beta = _bincount_upper_quantile(
            s_local, num_experts=num_experts, n_bins=n_bins, lo=lo, hi=hi_grid, target_rank=target_rank
        )
        return beta, lo, hi  # surface the live margin range for logging

    return shard_map(_fn, mesh=mesh, in_specs=(P(_BATCH_AXES, None), P()), out_specs=(P(), P(), P()))(
        s_ma, reshard(target_rank, P())
    )


def _local_input_gram(z: Float[Array, "T L"]) -> Float[Array, "shards L L"]:
    """Per-shard ``Z^T Z`` of the expert input (fp32 accumulation, no gradient) for ``newton_muon``.

    Like the router partials, the cross-device sum is deferred to one reduction after the layer scan.
    """

    def _local(x: jax.Array) -> jax.Array:
        return jnp.einsum("tl,tm->lm", x, x, preferred_element_type=jnp.float32)[None]

    return shard_map(
        _local, mesh=get_abstract_mesh(), in_specs=P(_BATCH_AXES, None), out_specs=P(_BATCH_AXES, None, None)
    )(reshard(jax.lax.stop_gradient(z), P(_BATCH_AXES, None)))


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array | None
    router_down: jax.Array | None
    router_up: jax.Array | None
    router_norm: "LearnedRMSNorm | None"
    router_bias: jax.Array
    router_tok_b: Float[Array, "r E"] | None
    """Per-layer zero-init up-projection of the shared token-identity router bias (``router_token_bias_rank``)."""
    expert_mlp: MoEExpertMlp
    expert_mlp_b: MoEExpertMlp | None
    bank_scale: Float[Array, " 2"] | None
    router_logit_scale: Float[Array, ""] | None
    expert_output_gain: Float[Array, " E"] | None
    null_const_v: Float[Array, "C L"] | None
    null_const_w: Float[Array, "C L 2"] | None
    w_latent_down: jax.Array | None
    latent_norm: LearnedRMSNorm | None
    w_latent_up: jax.Array | None
    latent_out_norm: LearnedRMSNorm | None
    expert_read_norm: LearnedRMSNorm | None
    """Per-group ``[G, W]`` learnable RMSNorm of the ``expert_read_groups`` input slices."""
    latent_select_mask: Float[Array, " D"] | None
    """0/1 mask of the hidden channels read by ``latent_select`` (``latent_select_pattern`` random/rotating),
    frozen for the optimizer. A mask rather than indices because it stays exact under the bf16 compute cast."""
    cfg: GrugModelConfig = eqx.field(static=True)
    latent_selects: bool = eqx.field(static=True, default=False)
    """This layer forms its expert input by ``latent_select`` (see ``latent_select_layers``)."""

    @staticmethod
    def init(
        cfg: GrugModelConfig, *, key: PRNGKeyArray, layer_index: jax.Array | int = 0, use_kda: bool = False
    ) -> "MoEMLP":
        k_router, k_expert, k_down, k_up = random.split(key, 4)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e = cfg.hidden_dim, cfg.num_experts + cfg.num_null_experts
        # Routed experts live in the latent space; the router reads the full-width token, so its
        # own projection keeps `hidden_dim`.
        expert_width, out_width = cfg.expert_in_dim, cfg.expert_out_dim
        latent = cfg.latent_dim
        selects = cfg.latent_select and (
            cfg.latent_select_layers == "all" or (cfg.latent_select_layers == "kda") == use_kda
        )
        return MoEMLP(
            router=(
                None if cfg.router_rank else reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None))
            ),
            router_down=(
                reshard(_init_weight(k_router, (d, cfg.router_rank), cfg.initializer_std), P(None, None))
                if cfg.router_rank
                else None
            ),
            router_up=(
                reshard(
                    _init_weight(random.fold_in(k_router, 1), (cfg.router_rank, e), 1.0 / math.sqrt(cfg.router_rank)),
                    P(None, None),
                )
                if cfg.router_rank
                else None
            ),
            router_norm=(
                _learned_rms_norm(cfg, cfg.router_rank, cfg.layer_norm_eps)
                if cfg.router_rank and cfg.router_rank_act == "norm"
                else None
            ),
            router_bias=jnp.zeros((e,)),
            router_tok_b=jnp.zeros((cfg.router_token_bias_rank, e)) if cfg.router_token_bias_rank else None,
            w_latent_down=(
                None
                if latent is None or (selects and not cfg.latent_select_plus_proj)
                else reshard(_init_weight(k_down, (d, latent), cfg.initializer_std), P(_FSDP_AXES, "model"))
            ),
            latent_norm=None if latent is None else _learned_rms_norm(cfg, latent, cfg.layer_norm_eps),
            w_latent_up=(
                reshard(_init_weight(k_up, (out_width, d), cfg.initializer_std), P("model", _FSDP_AXES))
                if cfg.has_latent_up
                else None
            ),
            latent_out_norm=(
                _learned_rms_norm(cfg, out_width, cfg.layer_norm_eps)
                if cfg.latent_out_norm and (latent is not None or cfg.latent_out_dim is not None)
                else None
            ),
            expert_read_norm=(
                _grouped_rms_norm(cfg, cfg.expert_read_groups, expert_width) if cfg.expert_read_groups else None
            ),
            latent_select_mask=_latent_select_mask(cfg, layer_index) if selects else None,
            expert_mlp=_expert_mlp_init(_bank_config(cfg, 1), expert_width, out_width, k_expert),
            expert_mlp_b=(
                _expert_mlp_init(_bank_config(cfg, 2), expert_width, out_width, random.fold_in(k_expert, 2))
                if cfg.moe_bank2_experts
                else None
            ),
            bank_scale=jnp.ones((2,), jnp.float32) if cfg.moe_bank2_experts and cfg.moe_bank2_scale else None,
            router_logit_scale=jnp.ones((), jnp.float32) if cfg.router_logit_scale else None,
            expert_output_gain=jnp.ones((e,), jnp.float32) if cfg.expert_output_gain else None,
            null_const_v=(
                jnp.zeros((cfg.moe_const_experts, expert_width), jnp.float32) if cfg.moe_const_experts else None
            ),
            null_const_w=(
                jnp.zeros((cfg.moe_const_experts, expert_width, 2), jnp.float32) if cfg.moe_const_experts else None
            ),
            cfg=cfg,
            latent_selects=selects,
        )

    def input_projection_weights(self, dtype: jnp.dtype) -> list[jax.Array]:
        """The ``[D, *]`` projections this MLP applies to its input: the router, then the latent down."""
        router_in = self.router if self.router is not None else self.router_down
        assert router_in is not None
        weights = [reshard(router_in, P(None, None))]
        if self.w_latent_down is not None:
            weights.append(reshard(self.w_latent_down.astype(dtype), P(None, None)))
        return weights

    def simbal_loss(self) -> jax.Array:
        """SimBal router orthogonality loss ``||R^T R - I||_1`` over the real-expert columns (``simbal_loss_weight``)."""
        assert self.router is not None
        r = reshard(self.router[:, : self.cfg.num_experts].astype(jnp.float32), P(None, None))
        gram = jnp.einsum("de,df->ef", r, r, out_sharding=P(None, None))
        return jnp.sum(jnp.abs(gram - jnp.eye(self.cfg.num_experts, dtype=jnp.float32)))

    def _qb_churn_stats(
        self,
        router_logits: Float[Array, "T E"],
        beta: Float[Array, " E"],
        banks: list[tuple[int, int, int]],
        mesh: jax.sharding.AbstractMesh,
    ) -> dict[str, jax.Array]:
        """Bias-induced routing churn (``qb_bias_damping``): the first ``_QB_CHURN_TOKENS`` tokens are routed
        with this step's bias and with the next step's (damped) bias on the same logits; returns the fraction
        of their top-K sets and of their slots that differ."""
        gamma = self.cfg.qb_bias_damping
        assert gamma is not None
        bias_now = jax.lax.stop_gradient(self.router_bias).astype(jnp.float32)
        beta = jax.lax.stop_gradient(beta).astype(jnp.float32)
        bias_next = (1.0 - gamma) * bias_now - gamma * (beta - jnp.mean(beta))
        shards = math.prod(_mesh_axis_size(mesh, axis) for axis in _BATCH_AXES)
        per_shard = max(1, _QB_CHURN_TOKENS // shards)

        def _local(logits: jax.Array, now: jax.Array, nxt: jax.Array) -> tuple[jax.Array, jax.Array]:
            logits = jax.lax.stop_gradient(logits[:per_shard])

            def route(bias: jax.Array) -> jax.Array:
                biased = logits + bias
                picks = [jax.lax.top_k(biased[:, st : st + size], bank_k)[1] + st for st, size, bank_k in banks]
                return jnp.concatenate(picks, axis=-1)

            sel_now, sel_next = route(now), route(nxt)
            kept = jnp.any(sel_next[:, :, None] == sel_now[:, None, :], axis=-1)
            set_churn = jnp.mean(jnp.any(~kept, axis=-1).astype(jnp.float32))
            slot_churn = jnp.mean((~kept).astype(jnp.float32))
            return jax.lax.pmean(set_churn, _BATCH_AXES), jax.lax.pmean(slot_churn, _BATCH_AXES)

        set_churn, slot_churn = shard_map(
            _local, mesh=mesh, in_specs=(P(_BATCH_AXES, None), P(), P()), out_specs=(P(), P())
        )(reshard(router_logits, P(_BATCH_AXES, None)), reshard(bias_now, P()), reshard(bias_next, P()))
        return {
            f"{_LAYER_KNOB_PREFIX}router_qb_churn": set_churn,
            f"{_LAYER_KNOB_PREFIX}router_qb_slot_churn": slot_churn,
        }

    def erc_loss(self, key: PRNGKeyArray) -> tuple[jax.Array, jax.Array]:
        """Expert-router coupling loss (``erc_loss_weight``) and ``mean diag(M) / mean offdiag(M)``.

        ``M`` is ``[n, n]`` over the real experts; its ``[n, n, I]`` pre-norm activations are sharded over the
        expert axis like ``w_up``, so each device computes only its experts' columns.
        """
        cfg = self.cfg
        n = cfg.num_experts
        sg = jax.lax.stop_gradient
        assert self.router is not None
        rows = reshard(self.router[:, :n].astype(jnp.float32).T, P(None, None))
        fixed = sg(rows)
        norms = jnp.sqrt(jnp.sum(jnp.square(fixed), axis=-1))
        sq_dist = norms[:, None] ** 2 + norms[None, :] ** 2 - 2.0 * fixed @ fixed.T
        dist = jnp.sqrt(jnp.maximum(sq_dist, 0.0)) + jnp.where(jnp.eye(n, dtype=bool), jnp.inf, 0.0)
        eps = jnp.min(dist, axis=-1) / (2.0 * norms)
        delta = 1.0 + eps[:, None] * random.uniform(key, rows.shape, minval=-1.0, maxval=1.0)
        proxy = rows * delta
        if self.w_latent_down is not None and self.latent_norm is not None:
            # The proxy follows a token's path into the latent experts.
            down = reshard(self.w_latent_down.astype(jnp.float32), P(None, None))
            proxy = self.latent_norm(jnp.einsum("id,dl->il", proxy, down, out_sharding=P(None, None)))
        em = self.expert_mlp
        w_in = (em.w_up if em.w_gate is None else em.w_gate).astype(jnp.float32)
        spec = _padded_spec(w_in)
        act = jnp.einsum("il,jlh->ijh", proxy, w_in, out_sharding=P(None, spec[0], spec[2]))
        m = reshard(jnp.sqrt(jnp.sum(jnp.square(act), axis=-1)), P(None, None))
        diag = jnp.diagonal(m)
        off = 1.0 - jnp.eye(n, dtype=jnp.float32)
        alpha = cfg.erc_alpha
        loss = jnp.mean((jax.nn.relu(m - alpha * diag[:, None]) + jax.nn.relu(m - alpha * diag[None, :])) * off)
        ratio = sg(jnp.mean(diag) / (jnp.sum(m * off) / (n * (n - 1))))
        return loss, ratio

    def _default_expert_term(
        self,
        routed_input: Float[Array, "T L"],
        router_logits: Float[Array, "T E"],
        selected_experts: Int[Array, "T K"],
        unbiased_topk: Float[Array, "T K"],
    ) -> Float[Array, "T O"]:
        """Zero-valued term carrying the dense router gradient of ``moe_dense_router_grad``."""
        if self.cfg.router_combine != RouterCombine.SIGMOID_RENORM:
            raise ValueError("moe_dense_router_grad supports only the renormalized-sigmoid combine")
        sg = jax.lax.stop_gradient
        em = self.expert_mlp
        selected = jnp.sum(jax.nn.one_hot(selected_experts, self.cfg.num_experts, dtype=jnp.float32), axis=1)
        counts = jnp.sum(selected, axis=0)
        mean_in = jnp.einsum("te,tl->el", selected, sg(routed_input).astype(jnp.float32), out_sharding=P(None, None))
        mean_in = mean_in / jnp.maximum(counts, 1.0)[:, None]
        up_spec = _padded_spec(em.w_up)
        mean_in = reshard(mean_in, P(up_spec[0], up_spec[1]))
        hidden_spec = P(up_spec[0], up_spec[2])
        up = jnp.einsum("el,eli->ei", mean_in, sg(em.w_up).astype(jnp.float32), out_sharding=hidden_spec)
        gate = (
            up
            if em.w_gate is None
            else jnp.einsum("el,eli->ei", mean_in, sg(em.w_gate).astype(jnp.float32), out_sharding=hidden_spec)
        )
        hidden = em.activation.to_jax_fn()(gate) * up
        down_spec = _padded_spec(em.w_down)
        expert_out = jnp.einsum(
            "ei,eio->eo", hidden, sg(em.w_down).astype(jnp.float32), out_sharding=P(down_spec[0], down_spec[2])
        )
        expert_out = reshard(expert_out, P(None, None))
        denom = sg(jnp.sum(jax.nn.sigmoid(unbiased_topk), axis=-1, keepdims=True))
        weights = jax.nn.sigmoid(router_logits) * (self.cfg.routing_renorm_sum / (denom + 1e-9))
        delta = (weights - sg(weights)) * (1.0 - selected)
        return jnp.einsum("te,eo->to", delta, expert_out, out_sharding=_partition_spec_of(routed_input))

    def _null_qb_stats(
        self, margins: Float[Array, "T N"], selected_experts: Int[Array, "T K"], mesh: jax.sharding.AbstractMesh
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """QB ``(beta, margin_min, margin_max)`` over the real plus zero-computation router columns.

        With ``moe_null_target_frac`` every column is balanced: the real experts share ``K (1 - f)`` slots per
        token, the zero-computation experts ``K f``. Without it the zero-computation columns get beta 0 (no bias,
        free to use) and the real experts are balanced at this step's measured real load, with their mean beta
        removed so QB never moves the real-vs-null split.
        """
        num_real, num_null = self.cfg.num_experts, self.cfg.num_null_experts
        k = self.cfg.num_experts_per_token
        margins = reshard(margins, P(_BATCH_AXES, None))
        frac = self.cfg.moe_null_target_frac
        if frac is not None:
            share = jnp.concatenate(
                [jnp.full((num_real,), k * (1.0 - frac) / num_real), jnp.full((num_null,), k * frac / num_null)]
            )
            return _qb_beta_hist(
                margins, mesh, target_share=share, num_experts=num_real + num_null, n_bins=_QB_HIST_BINS
            )
        real_slots = jax.lax.stop_gradient(jnp.mean(jnp.sum(selected_experts < num_real, axis=-1, dtype=jnp.float32)))
        beta, lo, hi = _qb_beta_hist(
            margins[:, :num_real], mesh, target_share=real_slots / num_real, num_experts=num_real, n_bins=_QB_HIST_BINS
        )
        beta = jnp.concatenate([beta - jnp.mean(beta), jnp.zeros((num_null,), beta.dtype)])
        return beta, lo, hi

    def _split_null_slots(
        self,
        routed_input: Float[Array, "T L"],
        selected_experts: Int[Array, "T K"],
        combine_weights: Float[Array, "T K"],
    ) -> tuple[Int[Array, "T K"], Float[Array, "T K"], Float[Array, "T L"]]:
        """Split the top-K slots into real-expert assignments and the zero-computation experts' output.

        Returns ``(selected, weights, null_out)``: every null slot becomes a combine-weight-0 assignment to a
        real expert (spread over the experts by token and slot, so it loads the fixed-capacity dispatch like a
        balanced real slot), and ``null_out`` is the weighted sum of the copy and constant experts' outputs.
        """
        cfg = self.cfg
        num_real = cfg.num_experts
        copy_start = num_real + cfg.moe_null_experts
        const_start = copy_start + cfg.moe_copy_experts
        is_null = selected_experts >= num_real
        t, k = selected_experts.shape
        slot_ids = jax.lax.broadcasted_iota(jnp.int32, (t, k), 0) * k + jax.lax.broadcasted_iota(jnp.int32, (t, k), 1)
        spread = reshard(slot_ids % num_real, _partition_spec_of(selected_experts))
        selected = jnp.where(is_null, spread, selected_experts)
        weights = jnp.where(is_null, jnp.zeros_like(combine_weights), combine_weights)

        w = combine_weights.astype(jnp.float32)
        in_spec = _partition_spec_of(routed_input)
        x = routed_input.astype(jnp.float32)
        is_copy = (selected_experts >= copy_start) & (selected_experts < const_start)
        x_scale = jnp.sum(jnp.where(is_copy, w, 0.0), axis=-1)
        null_out = jnp.zeros_like(x)
        if self.null_const_v is not None and self.null_const_w is not None:
            # [T, C]: each constant expert's summed combine weight (0 where it was not picked).
            const_w = jnp.einsum(
                "tk,tkc->tc",
                w,
                jax.nn.one_hot(selected_experts - const_start, cfg.moe_const_experts, dtype=jnp.float32),
            )
            mix = jax.nn.softmax(jnp.einsum("tl,clm->tcm", x, self.null_const_w), axis=-1)
            x_scale = x_scale + jnp.sum(const_w * mix[..., 0], axis=-1)
            null_out = jnp.einsum("tc,cl->tl", const_w * mix[..., 1], self.null_const_v, out_sharding=in_spec)
        return selected, weights, null_out + x_scale[:, None] * x

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        projected: list[jax.Array] | None = None,
        hash_token_ids: Int[Array, "B S"] | None = None,
        noise_key: jax.Array | None = None,
        overlap: MoeOverlapWork | None = None,
        router_tok_rows: Float[Array, "B S r"] | None = None,
        router_seed_bias: Float[Array, "B S E"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``projected`` holds ``x_flat @ w`` for each of ``input_projection_weights`` when the caller
        computed them already (fused with other projections of the same input). ``hash_token_ids`` picks
        the experts by token-id hash (``moe_hash_layers``); ``noise_key`` adds the training-only Gumbel
        noise of ``moe_gumbel_tau`` to the expert selection; ``router_tok_rows`` are the tokens' rows of the
        shared ``router_token_bias_rank`` table and ``router_seed_bias`` the ``router_bias_seed`` logit bias.
        ``overlap`` is ``[T, D]`` token-local work run under the first bank's dispatch all-to-all
        (``moe_shared_overlap``) and added to the output."""
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        if projected is None:
            projected = [jnp.einsum("td,de->te", x_flat, w) for w in self.input_projection_weights(x_flat.dtype)]
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        if self.router_up is not None:
            z = projected[0]
            if self.router_norm is not None:
                z = self.router_norm(z)
            elif self.cfg.router_rank_act == "silu":
                z = jax.nn.silu(z)
            router_logits = jnp.einsum("tr,re->te", z.astype(jnp.float32), self.router_up.astype(jnp.float32))
        else:
            router_logits = projected[0].astype(jnp.float32)
        if self.router_logit_scale is not None:
            router_logits = router_logits * self.router_logit_scale.astype(jnp.float32)
        if self.router_tok_b is not None:
            assert router_tok_rows is not None
            rows = rearrange(router_tok_rows, "b s r -> (b s) r").astype(jnp.float32)
            router_logits = router_logits + jnp.einsum(
                "tr,re->te",
                reshard(rows, _partition_spec_of(router_logits)),
                self.router_tok_b.astype(jnp.float32),
                out_sharding=_partition_spec_of(router_logits),
            )
        if router_seed_bias is not None:
            seed_bias = rearrange(router_seed_bias, "b s e -> (b s) e")
            router_logits = router_logits + reshard(seed_bias, _partition_spec_of(router_logits))
        cap = self.cfg.router_logit_soft_cap
        if cap is not None:
            router_logits = cap * jnp.tanh(router_logits / cap)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        k = self.cfg.num_experts_per_token
        banks = _expert_banks(self.cfg)
        num_real = self.cfg.num_experts
        if self.cfg.num_null_experts:
            # The zero-computation experts compete in the one bank's top-K.
            banks = [(0, num_real + self.cfg.num_null_experts, k)]
        # Select top-(K+1) on biased logits per bank; the (K+1)-th is that bank's QB threshold alpha.
        bank_selected, bank_alpha = [], []
        for start, size, bank_k in banks:
            topk_logits, sel = _small_top_k(biased_logits[:, start : start + size], bank_k + 1)
            bank_alpha.append(topk_logits[:, -1:])
            bank_selected.append(sel[:, :-1] + start)
        selected_experts = jnp.concatenate(bank_selected, axis=-1) if len(banks) > 1 else bank_selected[0]
        if len(banks) > 1 and (
            self.cfg.moe_gumbel_tau > 0 or hash_token_ids is not None or self.cfg.moe_dense_router_grad
        ):
            raise ValueError("moe_bank2_experts does not support Gumbel routing, hash routing or the dense router grad")
        if noise_key is not None and self.cfg.moe_gumbel_tau > 0:
            noisy = biased_logits + self.cfg.moe_gumbel_tau * _gumbel_noise(noise_key, biased_logits.shape)
            _, selected_experts = _small_top_k(noisy, k)
        if hash_token_ids is not None:
            table = reshard(jnp.asarray(_hash_expert_table(self.cfg.vocab_size, self.cfg.num_experts, k)), P(None, None))
            flat_ids = reshard(rearrange(hash_token_ids, "b s -> (b s)"), P(_BATCH_AXES))
            selected_experts = shard_map(
                _local_gather,
                mesh=get_abstract_mesh(),
                in_specs=(P(None, None), P(_BATCH_AXES)),
                out_specs=P(_BATCH_AXES, None),
            )(table, flat_ids)
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        renorm_sum = self.cfg.routing_renorm_sum
        if self.cfg.router_combine == RouterCombine.SOFTMAX_RENORM:
            combine_weights_f = renorm_sum * jax.nn.softmax(unbiased_topk, axis=-1)
        elif self.cfg.router_combine == RouterCombine.SIGMOID_RAW:
            combine_weights_f = jax.nn.sigmoid(unbiased_topk) * (renorm_sum / (k / 2))
        else:
            if self.cfg.router_combine == RouterCombine.SQRT_SOFTPLUS_RENORM:
                # The floor keeps sqrt's gradient finite if softplus underflows to 0 (logit below about -100).
                combine_weights_f = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
            else:
                combine_weights_f = jax.nn.sigmoid(unbiased_topk)
            denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
            combine_weights_f = combine_weights_f * (renorm_sum / (denom + 1e-9))
        if self.expert_output_gain is not None:
            # Scaling expert e's output by g_e is scaling its combine weight in every (token, slot) that picked it.
            gain = shard_map(
                _local_gather,
                mesh=get_abstract_mesh(),
                in_specs=(P(None), P(_BATCH_AXES, None)),
                out_specs=P(_BATCH_AXES, None),
            )(
                reshard(self.expert_output_gain.astype(jnp.float32), P(None)),
                reshard(selected_experts, P(_BATCH_AXES, None)),
            )
            combine_weights_f = combine_weights_f * reshard(gain, _partition_spec_of(combine_weights_f))
        combine_weights = combine_weights_f.astype(x.dtype)
        mesh = get_abstract_mesh()
        # Per-token assignments for the routing dump (``Transformer.routing_assignments``); unused (and so
        # dead-code eliminated) in the training forward.
        assignments = {_ROUTING_SELECTED: selected_experts.astype(jnp.int32), _ROUTING_WEIGHTS: combine_weights_f}
        # Per-shard partials only; the cross-device reduction happens once after the layer scan.
        router_stats = local_routing_stats(
            reshard(selected_experts, P(_BATCH_AXES, None)),
            reshard(router_probs, P(_BATCH_AXES, None)),
            reshard(router_logits, P(_BATCH_AXES, None)),
            mesh,
            num_experts=num_real + self.cfg.num_null_experts,
            batch_axes=_BATCH_AXES,
        )
        # Sharded QB: estimate each expert's threshold beta from the margins `s - alpha` by binning
        # them into fixed bins over the live global range and reading the (1-K/E) quantile.
        if self.cfg.num_null_experts:
            bank_stats = [self._null_qb_stats(router_logits - bank_alpha[0], selected_experts, mesh)]
        else:
            bank_stats = [
                _qb_beta_hist(
                    reshard(router_logits[:, start : start + size] - alpha, P(_BATCH_AXES, None)),
                    mesh,
                    target_share=bank_k / size,
                    num_experts=size,
                    n_bins=_QB_HIST_BINS,
                )
                for (start, size, bank_k), alpha in zip(banks, bank_alpha, strict=True)
            ]
        beta = jnp.concatenate([b for b, _, _ in bank_stats], axis=-1)
        margin_min = functools.reduce(jnp.minimum, [lo for _, lo, _ in bank_stats])
        margin_max = functools.reduce(jnp.maximum, [hi for _, _, hi in bank_stats])
        router_stats["qb_beta"] = beta
        router_stats["margin_min"] = margin_min
        router_stats["margin_max"] = margin_max
        if self.cfg.qb_bias_damping is not None:
            router_stats.update(self._qb_churn_stats(router_logits, beta, banks, mesh))

        routed_input = x_flat
        if self.latent_selects:
            assert self.cfg.latent_dim is not None and self.latent_norm is not None
            if self.latent_select_mask is None:
                selected = x_flat[..., : self.cfg.latent_dim]
            else:
                mask = jax.lax.stop_gradient(self.latent_select_mask) > 0.5
                (idx,) = jnp.nonzero(mask, size=self.cfg.latent_dim)
                selected = jnp.take(x_flat, idx, axis=-1)
            if self.w_latent_down is not None:
                selected = selected + reshard(projected[1], _batch_spec())
            routed_input = self.latent_norm(selected)
        elif self.w_latent_down is not None and self.latent_norm is not None:
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = self.latent_norm(reshard(projected[1], _batch_spec()))
        if self.cfg.newton_muon:
            router_stats[NEWTON_GRAM_LOCAL_KEY] = _local_input_gram(routed_input)
        bank_mlps = [self.expert_mlp] if self.expert_mlp_b is None else [self.expert_mlp, self.expert_mlp_b]
        if self.cfg.expert_read_subset:
            bank_mlps = [_mask_expert_reads(em, self.cfg) for em in bank_mlps]
        real_selected, real_weights, null_out = selected_experts, combine_weights, None
        if self.cfg.num_null_experts:
            real_selected, real_weights, null_out = self._split_null_slots(
                routed_input, selected_experts, combine_weights
            )
        outputs, overflows, bank_keeps, col = [], [], [], 0
        overlap_out = None
        for (start, _, bank_k), em, bank_index in zip(banks, bank_mlps, (1, 2), strict=False):
            bank_selected = (real_selected[:, col : col + bank_k] - start).astype(jnp.int32)
            if self.expert_read_norm is not None:
                router_stats.update(_expert_read_group_shares(bank_selected, self.cfg.expert_read_groups))
                out, overflow, *bank_overlap_out = _run_grouped_read_bank(
                    em,
                    _bank_config(self.cfg, bank_index),
                    self.expert_read_norm,
                    routed_input,
                    bank_selected,
                    real_weights[:, col : col + bank_k],
                    overlap,
                )
            else:
                out, overflow, *bank_overlap_out = _run_expert_bank(
                    em,
                    _bank_config(self.cfg, bank_index),
                    routed_input,
                    bank_selected,
                    real_weights[:, col : col + bank_k],
                    overlap if bank_index == 1 else None,
                )
            if bank_overlap_out:
                overlap_out = bank_overlap_out[0]
            if overflow.assignment_keep is not None:
                bank_keeps.append(overflow.assignment_keep)
                overflow = overflow._replace(assignment_keep=None)
            if self.bank_scale is not None:
                out = out * self.bank_scale[bank_index - 1].astype(out.dtype)
            outputs.append(out)
            overflows.append(overflow)
            col += bank_k
        routed_flat = functools.reduce(jnp.add, outputs)
        if null_out is not None:
            routed_flat = routed_flat + null_out.astype(routed_flat.dtype)
        if bank_keeps:
            # moe_drop_renorm: the null slots never leave the device, so only real slots can be dropped.
            kept = reshard(jnp.concatenate(bank_keeps, axis=-1), _partition_spec_of(combine_weights_f))
            kept = kept | (selected_experts >= num_real)
            factor, drop_stats = _drop_renorm_factor(combine_weights_f, kept)
            routed_flat = routed_flat * factor[:, None].astype(routed_flat.dtype)
            router_stats.update(drop_stats)
        capacity_overflow = overflows[0]
        if len(overflows) > 1:
            capacity_overflow = jax.tree.map(lambda *xs: functools.reduce(jnp.add, xs), *overflows)
        if self.cfg.moe_dense_router_grad:
            routed_flat = routed_flat + self._default_expert_term(
                routed_input, router_logits, selected_experts, unbiased_topk
            ).astype(routed_flat.dtype)
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
        elif self.cfg.latent_write_select:
            assert self.cfg.latent_dim is not None
            gain = self.cfg.initializer_std * math.sqrt(self.cfg.latent_dim)
            pad = self.cfg.hidden_dim - self.cfg.latent_dim
            routed_flat = jnp.pad(routed_flat * gain, ((0, 0), (0, pad)))
        if overlap_out is not None:
            routed_flat = routed_flat + reshard(overlap_out, _batch_spec()).astype(routed_flat.dtype)

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, {**router_stats, **assignments}


def _shared_experts_tail(
    cfg: "GrugModelConfig",
    parts: Sequence[jax.Array],
    num_shared: int,
    gated: bool,
    w_down: jax.Array,
    x_flat: jax.Array,
    shared_gate: jax.Array | None,
    out_sharding: P | None,
) -> jax.Array:
    """Shared-expert output from each expert's gate/up projections ``parts`` (gates first when ``gated``)."""
    gates, ups = (parts[:num_shared], parts[num_shared:]) if gated else ((), parts)
    if gated:
        act = ActivationFunctionEnum(cfg.expert_activation).to_jax_fn()
        hidden = jnp.concatenate([act(g) * u for g, u in zip(gates, ups, strict=True)], axis=1)
    else:
        slope = cfg.expert_leaky_slope
        hidden = jnp.concatenate([jnp.square(jax.nn.leaky_relu(u, slope)) for u in ups], axis=1)
    shared_out = jnp.einsum("tm,md->td", hidden, w_down, out_sharding=out_sharding)
    if shared_gate is not None:
        gate_logit = jnp.einsum("td,d->t", x_flat.astype(jnp.float32), shared_gate)
        shared_out = shared_out * (2.0 * jax.nn.sigmoid(gate_logit))[:, None].astype(shared_out.dtype)
    return shared_out


def _shared_experts_local(
    tokens: tuple[jax.Array, jax.Array],
    params: tuple[jax.Array, jax.Array, jax.Array | None],
    *,
    cfg: "GrugModelConfig",
    widths: tuple[int, ...],
    gated: bool,
) -> jax.Array:
    """The shared experts on one EP shard's tokens (``moe_shared_overlap``): ``tokens`` is (shared-expert input,
    MoE input), ``params`` the concatenated gate/up weights, the concatenated down weights and the gate."""
    shared_in, x_flat = tokens
    w_gate_up, w_down, shared_gate = params
    proj = jnp.einsum("td,de->te", shared_in, w_gate_up)
    parts = jnp.split(proj, list(itertools.accumulate(widths[:-1])), axis=1)
    num_shared = len(widths) // 2 if gated else len(widths)
    return _shared_experts_tail(cfg, parts, num_shared, gated, w_down, x_flat, shared_gate, None)


def moe_and_shared_fused(
    mlp: MoEMLP,
    shared: tuple[DenseMLP, ...],
    x: Float[Array, "B S D"],
    part_inputs: dict[str, jax.Array] | None = None,
    hash_token_ids: Int[Array, "B S"] | None = None,
    noise_key: jax.Array | None = None,
    shared_gate: Float[Array, " D"] | None = None,
    router_tok_rows: Float[Array, "B S r"] | None = None,
    router_seed_bias: Float[Array, "B S E"] | None = None,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """Routed MoE plus the shared SwiGLU experts with every projection of ``x`` in one GEMM.

    The router, latent down-projection and each shared expert's gate/up read the same input, so they
    run as one ``[D, sum widths]`` GEMM; the shared experts' down-projections run as one GEMM over
    their concatenated hidden units (the sum over shared experts happens in its accumulator). Same
    math as ``mlp(x) + sum(expert(x) for expert in shared)`` (~3% faster at d512); parameters stay
    separate leaves. ``moe_shared_overlap`` instead runs the whole shared-expert MLP under the EP dispatch
    all-to-all (``MoeOverlapWork``), so only the router and latent projections stay fused.
    """
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    replicated = P(None, None)
    moe_weights = mlp.input_projection_weights(x_flat.dtype)
    gated = shared[0].w_gate is not None
    shared_weights = [reshard(e.w_gate, replicated) for e in shared if gated] + [
        reshard(e.w_up, replicated) for e in shared
    ]
    w_down = jnp.concatenate([reshard(e.w_down, replicated) for e in shared], axis=0)
    gate = None if shared_gate is None else unshard(shared_gate)
    overlap = None
    weights = moe_weights
    if mlp.cfg.moe_shared_overlap:
        shared_in = x_flat
        if part_inputs and "shared" in part_inputs:
            shared_in = rearrange(part_inputs["shared"], "b s d -> (b s) d")
        overlap = MoeOverlapWork(
            functools.partial(
                _shared_experts_local, cfg=mlp.cfg, widths=tuple(w.shape[1] for w in shared_weights), gated=gated
            ),
            (shared_in, x_flat),
            (jnp.concatenate(shared_weights, axis=1), w_down, gate),
        )
    else:
        weights = moe_weights + shared_weights
    if not part_inputs:
        fused = jnp.einsum("td,de->te", x_flat, jnp.concatenate(weights, axis=1), out_sharding=_batch_spec())
        parts = jnp.split(fused, list(itertools.accumulate(w.shape[1] for w in weights[:-1])), axis=1)
    else:
        # Some projections read another stream (attn_res_sum_inputs): one GEMM per projection.
        names = ["router", "latent"][: len(moe_weights)] + ["shared"] * (len(weights) - len(moe_weights))
        flats = {k: rearrange(v, "b s d -> (b s) d") for k, v in part_inputs.items()}
        parts = [
            jnp.einsum("td,de->te", flats.get(n, x_flat), w, out_sharding=_batch_spec())
            for n, w in zip(names, weights, strict=True)
        ]
    routed, stats = mlp(
        part_inputs.get("latent", x) if part_inputs else x,
        projected=parts[: len(moe_weights)],
        hash_token_ids=hash_token_ids,
        noise_key=noise_key,
        router_tok_rows=router_tok_rows,
        router_seed_bias=router_seed_bias,
        overlap=overlap,
    )
    if overlap is not None:
        return routed, stats
    shared_out = _shared_experts_tail(
        mlp.cfg, parts[len(moe_weights) :], len(shared), gated, w_down, x_flat, gate, _batch_spec()
    )
    return routed + _batch_reshard(rearrange(shared_out, "(b s) d -> b s d", b=b, s=s)), stats


def _sconv_segment_ids(mask: AttentionMask | jax.Array) -> jax.Array | None:
    """segment_ids (packed-document boundaries) for the SConvs and KDA; None when unpacked."""
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    return segment_ids[0] if segment_ids is not None else None


def _laurel_a(cfg: "GrugModelConfig", key: PRNGKeyArray) -> jax.Array | None:
    if not cfg.laurel_rank:
        return None
    return reshard(_init_weight(key, (cfg.hidden_dim, cfg.laurel_rank), 1.0 / math.sqrt(cfg.hidden_dim)), P(None, None))


def _laurel_b(cfg: "GrugModelConfig") -> jax.Array | None:
    if not cfg.laurel_rank:
        return None
    return reshard(jnp.zeros((cfg.laurel_rank, cfg.hidden_dim), jnp.float32), P(None, None))


def _laurel(h: Float[Array, "B S D"], a: jax.Array | None, b: jax.Array | None) -> Float[Array, "B S D"]:
    """``h + (h A) B`` (LAuReL-LR), or ``h`` when off."""
    if a is None or b is None:
        return h
    low = jnp.einsum("bsd,dr->bsr", h, a.astype(h.dtype), out_sharding=_batch_spec())
    return h + jnp.einsum("bsr,rd->bsd", low, b.astype(h.dtype), out_sharding=_batch_spec())


def _ple_inject(layer: "Block", h: Float[Array, "B S D"], extras: dict[str, jax.Array | None] | None) -> jax.Array:
    """``h + (gelu(rms(h) W_gate) * ple) W_up`` with this layer's normed PLE rows ``extras["ple"]``, or ``h``."""
    rows = None if extras is None else extras.get("ple")
    if rows is None:
        return h
    assert layer.ple_gate is not None and layer.ple_up is not None
    gate = jnp.einsum("bsd,dp->bsp", rms_norm(h), layer.ple_gate.astype(h.dtype), out_sharding=_batch_spec())
    low = jax.nn.gelu(gate) * rows.astype(h.dtype)
    return h + jnp.einsum("bsp,pd->bsd", low, layer.ple_up.astype(h.dtype), out_sharding=_batch_spec())


def _memory_bag_chunks(x: jax.Array, rows_per_token: int, dim: int) -> jax.Array:
    """``[b, s, ...]`` -> ``[chunks, c, ...]`` with ``c * rows_per_token * dim`` at most ``_MEMORY_BAG_CHUNK_ELEMS``
    (or ``c`` odd), so the gathered ``[c, rows, dim]`` rows of one chunk bound the bag's transient memory."""
    tokens = x.shape[0] * x.shape[1]
    chunk = tokens
    while chunk % 2 == 0 and chunk * rows_per_token * dim > _MEMORY_BAG_CHUNK_ELEMS:
        chunk //= 2
    return x.reshape(tokens // chunk, chunk, *x.shape[2:])


def _memory_bag_local(values: jax.Array, slots: jax.Array, weights: jax.Array) -> jax.Array:
    b, s, rows = slots.shape
    dim = values.shape[1]

    def chunk_bag(args):
        ids, w = args
        return jnp.einsum("cr,crd->cd", w, values[ids].astype(jnp.float32))

    chunks = (_memory_bag_chunks(slots, rows, dim), _memory_bag_chunks(weights, rows, dim))
    return jax.lax.map(chunk_bag, chunks).reshape(b, s, dim).astype(values.dtype)


def _memory_bag_bwd_local(values, slots, weights, g):
    b, s, rows = slots.shape
    dim = values.shape[1]

    def chunk_grad(d_table, args):
        ids, w, g_c = args
        g32 = g_c.astype(jnp.float32)
        d_w = jnp.einsum("cd,crd->cr", g32, values[ids].astype(jnp.float32))
        d_table = d_table.at[ids.reshape(-1)].add((w[..., None] * g32[:, None, :]).reshape(-1, dim))
        return d_table, d_w

    chunks = (
        _memory_bag_chunks(slots, rows, dim),
        _memory_bag_chunks(weights, rows, dim),
        _memory_bag_chunks(g, rows, dim),
    )
    # The accumulator is per-shard (varying over the batch axes) until the psum.
    d_table0 = jax.lax.pcast(jnp.zeros(values.shape, jnp.float32), _BATCH_AXES, to="varying")
    d_table, d_w = jax.lax.scan(chunk_grad, d_table0, chunks)
    return jax.lax.psum(d_table.astype(values.dtype), _BATCH_AXES), d_w.reshape(b, s, rows)


@jax.custom_vjp
def _memory_bag(values: jax.Array, slots: Int[Array, "B S R"], weights: Float[Array, "B S R"]) -> jax.Array:
    """EmbeddingBag ``out[b, s] = sum_r weights[b, s, r] * values[slots[b, s, r]]`` from a replicated table
    (``slots`` and ``weights`` batch-sharded).

    Like ``_embedding_gather``, each batch shard gathers locally and the backward scatter-adds its row
    cotangents in float32 before one psum; forward and backward walk the tokens in chunks so the gathered
    ``[tokens, R, D]`` rows are never materialized whole.
    """
    return shard_map(
        _memory_bag_local,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None, None), P(_BATCH_AXES, None, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(values, slots, weights)


def _memory_bag_fwd(values, slots, weights):
    return _memory_bag(values, slots, weights), (values, slots, weights)


def _memory_bag_bwd(residuals, g):
    values, slots, weights = residuals
    d_values, d_weights = shard_map(
        _memory_bag_bwd_local,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None, None), P(_BATCH_AXES, None, None), P(_BATCH_AXES, None, None)),
        out_specs=(P(None, None), P(_BATCH_AXES, None, None)),
    )(values, slots, weights, reshard(g, P(_BATCH_AXES, None, None)))
    return d_values, np.zeros(slots.shape, dtype=jax.dtypes.float0), d_weights


_memory_bag.defvjp(_memory_bag_fwd, _memory_bag_bwd)


def _memory_address_local(q: jax.Array, keys: jax.Array, topk: int):
    """Product-key addressing of one batch shard: ``q [b, s, H, 2, K/2]`` against ``keys [H, 2, n, K/2]``,
    both RMS-normed. Returns the top ``topk`` slots per head and their softmax weights (``[b, s, H*topk]``),
    the batch-global per-slot hit counts and the batch-global mean top-1 score and top-1 weight."""
    n, half = keys.shape[2], keys.shape[3]
    scores = jnp.einsum(
        "bshcd,hcnd->bshcn", rms_norm(q.astype(jnp.float32)), rms_norm(keys.astype(jnp.float32))
    ) / math.sqrt(half)
    s1, i1 = jax.lax.top_k(scores[..., 0, :], topk)
    s2, i2 = jax.lax.top_k(scores[..., 1, :], topk)
    candidates = (s1[..., :, None] + s2[..., None, :]).reshape(*s1.shape[:-1], topk * topk)
    best, flat = jax.lax.top_k(candidates, topk)
    slots = jnp.take_along_axis(i1, flat // topk, axis=-1) * n + jnp.take_along_axis(i2, flat % topk, axis=-1)
    weights = jax.nn.softmax(best, axis=-1)
    b, s = slots.shape[:2]
    slots = slots.reshape(b, s, -1)
    counts = jax.lax.psum(jnp.zeros((n * n,), jnp.int32).at[slots.reshape(-1)].add(1), _BATCH_AXES)
    num_heads = jax.lax.psum(float(best[..., 0].size), _BATCH_AXES)
    top1_score = jax.lax.psum(jnp.sum(jax.lax.stop_gradient(best[..., 0])), _BATCH_AXES) / num_heads
    top1_weight = jax.lax.psum(jnp.sum(jax.lax.stop_gradient(weights[..., 0])), _BATCH_AXES) / num_heads
    return slots, weights.reshape(b, s, -1), counts, top1_score, top1_weight


class ProductKeyMemory(eqx.Module):
    """Product-key memory+ sublayer (arXiv 1907.05242, 2412.09764); see ``GrugModelConfig.memory_layers``."""

    w_q: Float[Array, "D Q"]
    keys: Float[Array, "H 2 N K"]
    values: Float[Array, "M D"]
    w_gate: Float[Array, "D D"]
    w_out: Float[Array, "D D"]
    topk: int = eqx.field(static=True)
    fsdp: bool = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "ProductKeyMemory":
        q_key, k_key, v_key, g_key = random.split(key, 4)
        d, n, heads = cfg.hidden_dim, cfg.memory_keys, cfg.memory_heads
        return ProductKeyMemory(
            w_q=reshard(_init_weight(q_key, (d, heads * cfg.memory_key_dim), cfg.initializer_std), P(None, None)),
            keys=reshard(random.normal(k_key, (heads, 2, n, cfg.memory_key_dim // 2), jnp.float32), P()),
            values=reshard(
                _init_weight(v_key, (n * n, d), 1.0 / math.sqrt(d)),
                P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
            ),
            w_gate=reshard(_init_weight(g_key, (d, d), cfg.initializer_std), P(None, None)),
            w_out=reshard(jnp.zeros((d, d), jnp.float32), P(None, None)),
            topk=cfg.memory_topk,
            fsdp=cfg.embed2_fsdp,
        )

    def __call__(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        heads, _, n, half = self.keys.shape
        q = jnp.einsum("bsd,dq->bsq", x, self.w_q.astype(x.dtype), out_sharding=_batch_spec())
        q = q.reshape(*q.shape[:2], heads, 2, half)
        slots, weights, counts, top1_score, top1_weight = shard_map(
            functools.partial(_memory_address_local, topk=self.topk),
            mesh=get_abstract_mesh(),
            in_specs=(P(_BATCH_AXES), P()),
            out_specs=(P(_BATCH_AXES, None, None), P(_BATCH_AXES, None, None), P(None), P(), P()),
        )(q, self.keys)
        table = reshard(self.values, P(None, None)) if self.fsdp else self.values
        bag = _memory_bag(table, slots, weights).astype(x.dtype)
        gate = jax.nn.silu(jnp.einsum("bsd,de->bse", x, self.w_gate.astype(x.dtype), out_sharding=_batch_spec()))
        out = jnp.einsum("bsd,de->bse", bag * gate, self.w_out.astype(x.dtype), out_sharding=_batch_spec())
        usage = jax.lax.stop_gradient(counts).astype(jnp.float32)
        usage = usage / jnp.sum(usage)
        entropy = -jnp.sum(jnp.where(usage > 0, usage * jnp.log(jnp.maximum(usage, 1e-30)), 0.0))
        stats = {
            f"{_MEMORY_STAT_PREFIX}slot_frac": jnp.mean((usage > 0).astype(jnp.float32)),
            f"{_MEMORY_STAT_PREFIX}usage_entropy": entropy / math.log(n * n),
            f"{_MEMORY_STAT_PREFIX}top1_score": top1_score,
            f"{_MEMORY_STAT_PREFIX}top1_weight": top1_weight,
        }
        return out, stats


def _memory_branch(
    h: Float[Array, "B S D"], extras: dict[str, jax.Array | None] | None
) -> tuple[Float[Array, "B S D"] | None, dict[str, jax.Array]]:
    """This layer's product-key memory output on the RMS-normed MoE input, or None without ``extras["memory"]``."""
    memory = None if extras is None else extras.get("memory")
    if memory is None:
        return None, {}
    assert isinstance(memory, ProductKeyMemory)
    return memory(rms_norm(h))


class Block(eqx.Module):
    rms_attn: LearnedRMSNorm | DyT
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention | KimiDeltaAttention
    rms_mlp: LearnedRMSNorm | DyT
    mlp_gated_norm: GatedNorm
    mlp: "MoEMLP | DenseMLP"
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: "ShortConv | None"
    sconv_mlp: "ShortConv | None"
    sconv_mlp_in: "ShortConv | None"  # Canon-C conv on the normed MLP input ("mlp_in" in cfg.sconv_sites)
    shared_gate: Float[Array, " D"] | None  # cfg.shared_expert_gate
    moe_out_gate_w: Float[Array, " G"] | None  # cfg.moe_out_gate
    moe_out_gate_b: Float[Array, ""] | None
    # Block AttnRes pseudo-queries of the attention and MLP sublayers (None without cfg.attn_res).
    attn_res_query_attn: Float[Array, " D"] | None
    attn_res_query_mlp: Float[Array, " D"] | None
    attn_res_query_v: Float[Array, " D"] | None  # value-projection gate (cfg.attn_res_v_gate)
    # Learnable sublayer output scalars (None without cfg.sublayer_scales).
    attn_out_scale: Float[Array, ""] | None
    mlp_out_scale: Float[Array, ""] | None
    out_norm_attn: "LearnedRMSNorm | None"
    out_norm_mlp: "LearnedRMSNorm | None"
    laurel_a_attn: Float[Array, "D R"] | None
    laurel_b_attn: Float[Array, "R D"] | None
    laurel_a_mlp: Float[Array, "D R"] | None
    laurel_b_mlp: Float[Array, "R D"] | None
    bias_attn_out: Float[Array, " D"] | None
    bias_mlp_out: Float[Array, " D"] | None
    # Per-layer-embedding gate and zero-init up-projection (cfg.ple_dim).
    ple_gate: Float[Array, "D P"] | None
    ple_up: Float[Array, "P D"] | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, layer_index: jax.Array, use_kda: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        attn = (
            KimiDeltaAttention.init(cfg, key=attn_key)
            if use_kda
            else CausalSelfAttention.init(cfg, key=attn_key, layer_index=layer_index)
        )
        # KDA blocks have no branch-output SConv (K3 has only the q/k/v convs).
        use_attn_sconv = cfg.sconv and "attn" in cfg.sconv_sites and not use_kda
        # Zero-init: every source scores 0, so each gate starts as a uniform average of its sources.
        attn_res_query = reshard(jnp.zeros((_attn_res_key_dim(cfg),), jnp.float32), P(None)) if cfg.attn_res else None
        if cfg.dense_mlp:
            # Dense block: one SwiGLU DenseMLP(hidden, intermediate_dim), no MoE and no shared experts.
            mlp = DenseMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key)
            shared = None
        else:
            mlp = MoEMLP.init(cfg, key=mlp_key, layer_index=layer_index, use_kda=use_kda)
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
                if cfg.shared_ungated_relu2:
                    shared = tuple(eqx.tree_at(lambda m: m.w_gate, e, None, is_leaf=lambda x: x is None) for e in shared)
        return Block(
            rms_attn=(
                DyT.init(cfg.hidden_dim, cfg.dyt_alpha_attn)
                if cfg.dyt_norm
                else _learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps)
            ),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=attn,
            rms_mlp=(
                DyT.init(cfg.hidden_dim, cfg.dyt_alpha_mlp)
                if cfg.dyt_norm
                else _learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps)
            ),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=mlp,
            shared=shared,
            sconv_attn=(ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if use_attn_sconv else None),
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None
            ),
            shared_gate=jnp.zeros((cfg.hidden_dim,), jnp.float32) if cfg.shared_expert_gate else None,
            moe_out_gate_w=jnp.zeros((_MOE_OUT_GATE_DIMS,), jnp.float32) if cfg.moe_out_gate else None,
            moe_out_gate_b=jnp.full((), 5.0, jnp.float32) if cfg.moe_out_gate else None,
            sconv_mlp_in=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp_in" in cfg.sconv_sites else None
            ),
            attn_res_query_attn=attn_res_query,
            attn_res_query_mlp=attn_res_query,
            attn_res_query_v=attn_res_query if cfg.attn_res_v_gate else None,
            attn_out_scale=jnp.ones((), dtype=jnp.float32) if cfg.sublayer_scales else None,
            mlp_out_scale=jnp.ones((), dtype=jnp.float32) if cfg.sublayer_scales else None,
            out_norm_attn=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps) if cfg.sublayer_out_norm else None,
            out_norm_mlp=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps) if cfg.sublayer_out_norm else None,
            laurel_a_attn=_laurel_a(cfg, random.fold_in(key, 91)),
            laurel_b_attn=_laurel_b(cfg),
            laurel_a_mlp=_laurel_a(cfg, random.fold_in(key, 92)),
            laurel_b_mlp=_laurel_b(cfg),
            bias_attn_out=jnp.zeros((cfg.hidden_dim,)) if "attn_out" in cfg.proj_biases else None,
            bias_mlp_out=jnp.zeros((cfg.hidden_dim,)) if "mlp_out" in cfg.proj_biases else None,
            ple_gate=(
                reshard(
                    _init_weight(random.fold_in(key, 93), (cfg.hidden_dim, cfg.ple_dim), cfg.initializer_std),
                    P(None, None),
                )
                if cfg.ple_dim
                else None
            ),
            ple_up=(
                reshard(jnp.zeros((cfg.ple_dim, cfg.hidden_dim), jnp.float32), P(None, None)) if cfg.ple_dim else None
            ),
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
        value_residual: bool = False,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``sum_stream`` (with ``sum_components``) feeds those q/k/v projections from the straight-sum
        stream, through the same RMSNorm and GatedNorm, instead of the AttnRes mix ``h``; ``kv_input``
        (``kv_stream_dim``) is the normed KV side stream the K/V projections read. Returns the branch output
        and the mixer's logging stats."""
        attn_in = self.attn_gated_norm(self.rms_attn(_laurel(h, self.laurel_a_attn, self.laurel_b_attn)))
        proj_inputs = None
        attn_components = tuple(c for c in sum_components if c in ("q", "k", "v"))
        if attn_components:
            assert sum_stream is not None
            sum_in = self.attn_gated_norm(self.rms_attn(sum_stream))
            proj_inputs = {c: sum_in for c in attn_components}
        stats: dict[str, jax.Array] = {}
        if isinstance(self.attn, KimiDeltaAttention):
            # KDA has no positional encoding or window; it only needs the document boundaries.
            out, stats = self.attn(
                attn_in,
                _sconv_segment_ids(mask),
                proj_inputs=proj_inputs,
                no_decay=kda_ablation[0],
                no_beta=kda_ablation[1],
                kv_share=kv_share,
                value_residual=value_residual,
                kv_input=kv_input,
            )
        else:
            out, stats = self.attn(
                attn_in,
                mask,
                disable_rope=disable_rope,
                is_global=is_global,
                token_ids=token_ids,
                kv_share=kv_share,
                proj_inputs=proj_inputs,
                value_residual=value_residual,
                kv_input=kv_input,
            )
        if self.bias_attn_out is not None:
            out = out + unshard(self.bias_attn_out).astype(out.dtype)
        if self.sconv_attn is not None:
            out = self.sconv_attn(out, _sconv_segment_ids(mask))
        if self.out_norm_attn is not None:
            out = self.out_norm_attn(out)
        if self.attn_out_scale is not None:
            out = out * self.attn_out_scale.astype(out.dtype)
        return out, stats

    def mlp_branch(
        self,
        h: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        sum_stream: Float[Array, "B S D"] | None = None,
        sum_parts: tuple[str, ...] = (),
        hash_token_ids: Int[Array, "B S"] | None = None,
        noise_key: jax.Array | None = None,
        router_tok_rows: Float[Array, "B S r"] | None = None,
        router_seed_bias: Float[Array, "B S E"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``sum_stream`` feeds ``sum_parts`` (``router`` / ``latent`` / ``shared``) instead of ``h``;
        ``hash_token_ids`` / ``noise_key`` / ``router_tok_rows`` / ``router_seed_bias`` go to the router
        (``moe_hash_layers``, ``moe_gumbel_tau``, ``router_token_bias_rank``, ``router_bias_seed``)."""
        normed = self.mlp_gated_norm(self.rms_mlp(_laurel(h, self.laurel_a_mlp, self.laurel_b_mlp)))
        if self.sconv_mlp_in is not None:
            normed = self.sconv_mlp_in(normed, _sconv_segment_ids(mask))
        mlp_in = _spread_mlp_input(normed, self.attn.cfg)
        part_inputs = None
        if sum_parts:
            assert sum_stream is not None
            sum_in = self.mlp_gated_norm(self.rms_mlp(sum_stream))
            part_inputs = {p: sum_in for p in sum_parts}
        stats: dict[str, jax.Array] = {}
        if isinstance(self.mlp, DenseMLP):
            out = self.mlp(mlp_in, moe_output_reshard=False)
        elif self.shared is not None:
            out, stats = moe_and_shared_fused(
                self.mlp,
                self.shared,
                mlp_in,
                part_inputs,
                hash_token_ids,
                noise_key,
                self.shared_gate,
                router_tok_rows,
                router_seed_bias,
            )
        else:
            out, stats = self.mlp(
                mlp_in,
                hash_token_ids=hash_token_ids,
                noise_key=noise_key,
                router_tok_rows=router_tok_rows,
                router_seed_bias=router_seed_bias,
            )
        if self.moe_out_gate_w is not None and self.moe_out_gate_b is not None:
            gate_logit = jnp.einsum(
                "bsg,g->bs", normed[..., :_MOE_OUT_GATE_DIMS].astype(jnp.float32), self.moe_out_gate_w
            )
            out = out * jax.nn.sigmoid(gate_logit + self.moe_out_gate_b)[..., None].astype(out.dtype)
        if self.bias_mlp_out is not None:
            out = out + unshard(self.bias_mlp_out).astype(out.dtype)
        if self.sconv_mlp is not None:
            out = self.sconv_mlp(out, _sconv_segment_ids(mask))
        if self.out_norm_mlp is not None:
            out = self.out_norm_mlp(out)
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
        attn_out, _ = self.attn_branch(x, mask, disable_rope, is_global)
        x = x + attn_out
        mlp_out, router_stats = self.mlp_branch(x, mask)
        return x + mlp_out, router_stats


class RouterTie(NamedTuple):
    """One ``router_embed_tie`` entry: layer ``layer``'s router column ``expert`` is tied to the mean of
    ``token_embed[tokens]`` (a single row for one token)."""

    layer: int
    expert: int
    tokens: tuple[int, ...]

    @property
    def tag(self) -> str:
        """Metric-name suffix: ``L{layer}_e{expert}_v{token}``, or ``..._c{n}`` for an ``n``-token centroid."""
        target = f"v{self.tokens[0]}" if len(self.tokens) == 1 else f"c{len(self.tokens)}"
        return f"L{self.layer}_e{self.expert}_{target}"


class RouterSeed(NamedTuple):
    """One ``router_bias_seed`` entry: ``bias`` on expert ``expert``'s logit in layer ``layer`` at token ``token``."""

    layer: int
    expert: int
    token: int
    bias: float


def _spec_layers(text: str, num_layers: int) -> list[int]:
    if text == "*":
        return list(range(num_layers))
    layer = int(text)
    if not 0 <= layer < num_layers:
        raise ValueError(f"router tie/seed layer must be '*' or in 0..{num_layers - 1}, got {text!r}")
    return [layer]


def _router_ties(cfg: "GrugModelConfig") -> tuple[RouterTie, ...]:
    """``cfg.router_embed_tie`` parsed, ``*`` expanded to every layer, in spec order."""
    return _parse_router_ties(cfg.router_embed_tie, cfg.num_layers)


@functools.lru_cache(maxsize=16)
def _parse_router_ties(specs: tuple[str, ...], num_layers: int) -> tuple[RouterTie, ...]:
    ties = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"router_embed_tie entries are 'L:E:V' or 'L:E:V1|V2|...', got {spec!r}")
        try:
            expert = int(parts[1])
            tokens = tuple(int(v) for v in parts[2].split("|"))
        except ValueError as e:
            raise ValueError(f"router_embed_tie expert and token ids must be integers, got {spec!r}") from e
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"router_embed_tie token ids repeat in {spec!r}")
        ties += [RouterTie(layer, expert, tokens) for layer in _spec_layers(parts[0], num_layers)]
    return tuple(ties)


def _router_seeds(cfg: "GrugModelConfig") -> tuple[RouterSeed, ...]:
    """``cfg.router_bias_seed`` parsed, ``*`` expanded to every layer."""
    seeds = []
    for spec in cfg.router_bias_seed:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"router_bias_seed entries are 'L:E:V:b', got {spec!r}")
        seeds += [
            RouterSeed(layer, int(parts[1]), int(parts[2]), float(parts[3]))
            for layer in _spec_layers(parts[0], cfg.num_layers)
        ]
    return tuple(seeds)


def _router_seed_bias(cfg: "GrugModelConfig", physical_layer: int, token_ids: Int[Array, "B S"]) -> jax.Array | None:
    """``[B, S, E]`` router logit bias of this layer's ``router_bias_seed`` entries (None when it has none)."""
    seeds = [s for s in _router_seeds(cfg) if s.layer == physical_layer]
    if not seeds:
        return None
    table = np.zeros((len(seeds), cfg.num_experts + cfg.num_null_experts), np.float32)
    for j, s in enumerate(seeds):
        table[j, s.expert] = s.bias
    hits = (token_ids[..., None] == jnp.asarray([s.token for s in seeds], dtype=token_ids.dtype)).astype(jnp.float32)
    return jnp.einsum("bsn,ne->bse", hits, jnp.asarray(table), out_sharding=_batch_spec())


def _router_tie_directions(model: "Transformer") -> Float[Array, "N D"]:
    """Per tie, the mean of its current ``token_embed`` rows (each distinct token set gathered once)."""
    ties = _router_ties(model.config)
    means = {
        tokens: jnp.mean(model.token_embed[np.asarray(tokens)], axis=0)
        for tokens in dict.fromkeys(t.tokens for t in ties)
    }
    return jnp.stack([means[t.tokens] for t in ties])


def _router_tie_alpha_init(model: "Transformer") -> Float[Array, " N"]:
    """Per tie ``||router[:, E]|| / ||direction||`` at init, so a tied column starts at its untied scale."""
    ties = _router_ties(model.config)
    rows = jnp.linalg.norm(_router_tie_directions(model).astype(jnp.float32), axis=-1)
    routers = _routers_by_layer(model)
    cols = jnp.stack([jnp.linalg.norm(routers[t.layer][:, t.expert].astype(jnp.float32)) for t in ties])
    return reshard(cols / rows, P(None))


def _routers_by_layer(model: "Transformer") -> dict[int, jax.Array]:
    """Each layer's ``[D, E]`` router, by layer index."""
    routers = {}
    for stack, indices in zip(model.layer_stacks(), model.stack_layer_indices(), strict=True):
        router = stack.stacked.mlp.router
        assert router is not None
        routers.update({layer: router[j] for j, layer in enumerate(indices)})
    return routers


def tie_routers(model: "Transformer") -> "Transformer":
    """Replace each ``router_embed_tie`` column by ``alpha * mean(token_embed[V...])`` (gradients reach the
    embedding). Applied to the parameters it is the ``router_embed_tie_release_step`` rewrite: the untied
    forward of the result equals the tied forward of ``model``."""
    ties = _router_ties(model.config)
    assert model.router_tie_alpha is not None
    rows = _router_tie_directions(model)
    cols = rows * model.router_tie_alpha[:, None].astype(rows.dtype)
    tied = []
    for stack, indices in zip(model.layer_stacks(), model.stack_layer_indices(), strict=True):
        router = stack.stacked.mlp.router
        assert router is not None
        spec = _partition_spec_of(router)
        for j, t in enumerate(ties):
            if t.layer in indices:
                router = router.at[indices.index(t.layer), :, t.expert].set(cols[j].astype(router.dtype))
        tied.append(reshard(router, spec))
    return eqx.tree_at(lambda t: [s.stacked.mlp.router for s in t.layer_stacks()], model, tied)


@functools.lru_cache(maxsize=4)
def _hash_expert_table(vocab_size: int, num_experts: int, k: int) -> np.ndarray:
    """``[vocab, k]`` distinct random experts per token id (fixed seed), for ``moe_hash_layers``."""
    rng = np.random.default_rng(0)
    return np.argsort(rng.random((vocab_size, num_experts)), axis=1)[:, :k].astype(np.int32)


def _gumbel_noise(key: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Batch-sharded ``[T, E]`` standard Gumbel noise, drawn locally per shard (no global array)."""

    def local(key_data):
        shard_key = jax.random.fold_in(key_data[0], jax.lax.axis_index(_BATCH_AXES))
        return jax.random.gumbel(shard_key, (shape[0] // _batch_shards(), shape[1]), jnp.float32)

    return shard_map(local, mesh=get_abstract_mesh(), in_specs=(P(None),), out_specs=P(_BATCH_AXES, None))(key[None])


def _batch_shards() -> int:
    mesh = get_abstract_mesh()
    axes = _BATCH_AXES if isinstance(_BATCH_AXES, tuple) else (_BATCH_AXES,)
    return math.prod(mesh.shape[a] for a in axes)


_EXPERT_READ_SUBSET_SALT = 0x5B5E7


def _expert_read_mask(cfg: "GrugModelConfig", num_experts: int, in_dim: int) -> jax.Array:
    """``[E, in_dim, 1]`` 0/1 mask of the input channels each routed expert reads (``expert_read_subset``)."""
    width = cfg.expert_read_subset
    if cfg.expert_read_subset_pattern in ("blocks", "shared"):
        groups = 1 if cfg.expert_read_subset_pattern == "shared" else in_dim // width
        start = (jnp.arange(num_experts) % groups) * width
        channel = jnp.arange(in_dim)
        mask = (channel[None, :] >= start[:, None]) & (channel[None, :] < start[:, None] + width)
    else:
        keys = random.split(random.PRNGKey(_EXPERT_READ_SUBSET_SALT), num_experts)
        ranks = jax.vmap(lambda k: jnp.argsort(random.permutation(k, in_dim)))(keys)
        mask = ranks < width
    return mask.astype(jnp.float32)[:, :, None]


def _mask_expert_reads(em: MoEExpertMlp, cfg: "GrugModelConfig") -> MoEExpertMlp:
    """Zero the input rows of ``w_up`` (and ``w_gate``) outside each expert's ``expert_read_subset`` slice."""
    mask = _expert_read_mask(cfg, em.w_up.shape[0], em.w_up.shape[1])
    mask = reshard(mask, P(*_padded_spec(em.w_up)[:2], None))
    em = eqx.tree_at(lambda m: m.w_up, em, em.w_up * mask.astype(em.w_up.dtype))
    if em.w_gate is not None:
        em = eqx.tree_at(lambda m: m.w_gate, em, em.w_gate * mask.astype(em.w_gate.dtype))
    return em


def _expert_mlp_init(cfg: "GrugModelConfig", in_width: int, out_width: int, key: PRNGKeyArray) -> MoEExpertMlp:
    """The routed expert bank; ``moe_ungated_relu2`` drops the gate (``w_gate=None``) and uses ReLU."""
    mlp = MoEExpertMlp.init(
        num_experts=cfg.num_experts,
        hidden_dim=in_width,
        output_dim=out_width,
        intermediate_dim=cfg.intermediate_dim,
        initializer_std=cfg.initializer_std,
        key=key,
        implementation=cfg.moe_implementation,
        activation=(
            ActivationFunctionEnum.relu if cfg.moe_ungated_relu2 else ActivationFunctionEnum(cfg.expert_activation)
        ),
        capacity_factor=cfg.capacity_factor,
        pooled_transport_capacity_factor=cfg.pooled_transport_capacity_factor,
        expert_chunks=1,
        num_expert_waves=cfg.moe_expert_waves,
        fp8_dispatch=cfg.moe_fp8_dispatch,
        expert_remat=cfg.moe_expert_remat,
    )
    if cfg.init_std_mult_experts != 1.0:
        mult = cfg.init_std_mult_experts
        mlp = eqx.tree_at(lambda m: (m.w_up, m.w_down), mlp, (mlp.w_up * mult, mlp.w_down * mult))
    if cfg.moe_ungated_relu2:
        mlp = eqx.tree_at(lambda m: m.w_gate, mlp, None, is_leaf=lambda x: x is None)
    # Masked input rows start at zero, so they stay zero (no gradient) and take no share of the MuonH norm.
    return _mask_expert_reads(mlp, cfg) if cfg.expert_read_subset else mlp


def _positions_in_document(segment_ids: Int[Array, "B S"] | None, seq_len: int) -> jax.Array:
    """Each token's 0-based position within its packed document (``[S]`` when unpacked)."""
    idx = jnp.arange(seq_len, dtype=jnp.int32)
    if segment_ids is None:
        return idx
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return idx - jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _partial_key_offset(k: Float[Array, "B S H D"], segment_ids: Int[Array, "B S"] | None) -> jax.Array:
    """Replace the first half of each key's channels with the previous token's (zero at document starts)."""
    half = k.shape[-1] // 2
    prev = jnp.pad(k[:, :-1, :, :half], ((0, 0), (1, 0), (0, 0), (0, 0)))
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        prev = jnp.where(starts[..., None, None], 0, prev)
    return reshard(jnp.concatenate([prev.astype(k.dtype), k[..., half:]], axis=-1), _partition_spec_of(k))


def _smear(
    hidden: Float[Array, "B S D"], w: jax.Array, lam: jax.Array, segment_ids: Int[Array, "B S"] | None
) -> Float[Array, "B S D"]:
    """modded-nanogpt Smear: ``x_t + lam * sigmoid(x_t[:12] @ w) * x_{t-1}``, zeroed at document starts."""
    prev = jnp.pad(hidden[:, :-1], ((0, 0), (1, 0), (0, 0)))
    if segment_ids is not None:
        starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
        prev = jnp.where(starts[..., None], 0, prev)
    gate = jax.nn.sigmoid(jnp.einsum("bsk,k->bs", hidden[..., :12].astype(jnp.float32), w))[..., None]
    return (hidden.astype(jnp.float32) + lam * gate * prev.astype(jnp.float32)).astype(hidden.dtype)


def _expert_banks(cfg: "GrugModelConfig") -> list[tuple[int, int, int]]:
    """``(first expert, number of experts, experts per token)`` for each expert bank."""
    if not cfg.moe_bank2_experts:
        return [(0, cfg.num_experts, cfg.num_experts_per_token)]
    size_a = cfg.num_experts - cfg.moe_bank2_experts
    k_a = cfg.num_experts_per_token - cfg.moe_bank2_topk
    if size_a <= k_a or cfg.moe_bank2_experts <= cfg.moe_bank2_topk or k_a < 1 or cfg.moe_bank2_topk < 1:
        raise ValueError("each expert bank needs 1 <= top-k < its expert count (QB selects top-(k+1))")
    return [(0, size_a, k_a), (size_a, cfg.moe_bank2_experts, cfg.moe_bank2_topk)]


def _bank_config(cfg: "GrugModelConfig", bank: int) -> "GrugModelConfig":
    """The per-bank view of ``cfg`` used to build and run that bank's experts."""
    if not cfg.moe_bank2_experts:
        return cfg
    _, size, bank_k = _expert_banks(cfg)[bank - 1]
    if bank == 1:
        return dataclasses.replace(cfg, num_experts=size, num_experts_per_token=bank_k)
    if cfg.moe_bank2_activation not in ("relu2", "swiglu"):
        raise ValueError(f"moe_bank2_activation must be relu2 or swiglu, got {cfg.moe_bank2_activation!r}")
    return dataclasses.replace(
        cfg,
        num_experts=size,
        num_experts_per_token=bank_k,
        intermediate_dim=cfg.moe_bank2_intermediate_dim or cfg.intermediate_dim,
        moe_ungated_relu2=cfg.moe_bank2_activation == "relu2",
        expert_activation="silu",
        expert_leaky_slope=cfg.expert_leaky_slope if cfg.moe_bank2_activation == "relu2" else 0.0,
    )


def _run_expert_bank(
    em: MoEExpertMlp,
    cfg: "GrugModelConfig",
    routed_input: jax.Array,
    selected: jax.Array,
    combine_weights: jax.Array,
    overlap: MoeOverlapWork | None = None,
):
    """Dispatch one expert bank: ungated ReLU^2 through the EP backends' ungated path (or the tied gate on the
    dropless local backends), gated experts through ``MoEExpertMlp``. Returns ``(out, overflow)``, plus the
    ``overlap`` output when given. ``moe_drop_renorm`` also asks the backend for the per-slot keep mask
    (``MoeDispatchCounts.assignment_keep``)."""
    if em.w_gate is None:
        ungated = em.implementation in MOE_IMPLEMENTATIONS and cfg.moe_ungated_kernel
        return moe_mlp(
            routed_input,
            selected,
            combine_weights,
            em.w_up if ungated else jnp.concatenate([em.w_up, em.w_up], axis=-1),
            em.w_down,
            activation=_ungated_expert_activation(cfg) if ungated else _tied_expert_activation(cfg, em),
            implementation=em.implementation,
            mesh=get_abstract_mesh(),
            capacity_factor=em.capacity_factor,
            pooled_transport_capacity_factor=em.pooled_transport_capacity_factor,
            report_capacity_overflow=True,
            report_assignment_keep=cfg.moe_drop_renorm,
            expert_chunks=em.expert_chunks,
            num_expert_waves=em.num_expert_waves,
            fp8_dispatch=em.fp8_dispatch,
            expert_remat=em.expert_remat,
            overlap=overlap,
        )
    return em(
        routed_input,
        selected,
        combine_weights,
        mesh=get_abstract_mesh(),
        report_capacity_overflow=True,
        overlap=overlap,
        report_assignment_keep=cfg.moe_drop_renorm,
    )


def _grouped_rms_norm(cfg: "GrugModelConfig", groups: int, width: int) -> LearnedRMSNorm:
    """A learned RMSNorm with a stacked ``[groups, width]`` gain, normalizing each ``[..., groups, width]`` slice."""
    norm = _learned_rms_norm(cfg, width, cfg.layer_norm_eps)
    return jax.tree.map(lambda g: jnp.broadcast_to(g, (groups, width)), norm)


def _expert_read_group_shares(selected: Int[Array, "T K"], groups: int) -> dict[str, jax.Array]:
    """Fraction of the routed assignments that go to each ``expert_read_groups`` channel group."""
    group = (selected % groups).reshape(-1)
    return {
        f"{_LAYER_KNOB_PREFIX}expert_read_share_g{g}": jax.lax.stop_gradient(jnp.mean((group == g).astype(jnp.float32)))
        for g in range(groups)
    }


def _run_grouped_read_bank(
    em: MoEExpertMlp,
    cfg: "GrugModelConfig",
    read_norm: LearnedRMSNorm,
    x_flat: Float[Array, "T D"],
    selected: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    overlap: MoeOverlapWork | None,
):
    """Run the routed experts on per-assignment ``expert_read_groups`` slices.

    Splits ``x_flat`` into G normed ``[T, G, W]`` slices, gathers each (token, slot)'s slice ``g(expert) =
    expert mod G`` to ``[T, K, W]`` and dispatches the ``T*K`` slot rows as single-slot tokens (top-1 over the
    same experts, in the same token-major order, so capacity and drops match the ``[T, D]`` dispatch), then sums
    the K weighted slot outputs back per token.
    """
    t, k = selected.shape
    groups = cfg.expert_read_groups
    slices = jnp.reshape(x_flat, (t, groups, x_flat.shape[-1] // groups), out_sharding=P(_BATCH_AXES, None, None))
    slices = read_norm(slices)
    slot_rows = jnp.take_along_axis(slices, (selected % groups)[:, :, None], axis=1)
    flat_rows = jnp.reshape(slot_rows, (t * k, slot_rows.shape[-1]), out_sharding=P(_BATCH_AXES, None))
    flat_selected = jnp.reshape(selected, (t * k, 1), out_sharding=P(_BATCH_AXES, None))
    flat_weights = jnp.reshape(combine_weights, (t * k, 1), out_sharding=P(_BATCH_AXES, None))
    out, overflow, *overlap_out = _run_expert_bank(em, cfg, flat_rows, flat_selected, flat_weights, overlap)
    slot_out = jnp.reshape(out, (t, k, out.shape[-1]), out_sharding=P(_BATCH_AXES, None, None))
    out = jnp.sum(slot_out.astype(jnp.float32), axis=1).astype(out.dtype)
    if overflow.assignment_keep is not None:
        keep = jnp.reshape(overflow.assignment_keep, (t, k), out_sharding=P(_BATCH_AXES, None))
        overflow = overflow._replace(assignment_keep=keep)
    return out, overflow, *overlap_out


def _drop_renorm_factor(
    combine_weights: Float[Array, "T K"], kept: Bool[Array, "T K"]
) -> tuple[Float[Array, " T"], dict[str, jax.Array]]:
    """Per-token scale that renormalizes the combine weights over the surviving slots (``moe_drop_renorm``).

    The routed output is linear in the combine weights, so rescaling it by ``sum(w) / sum(w * kept)`` is the
    same as renormalizing the kept weights to the pre-drop total. Exactly 1 for tokens with no drop (and for
    tokens that lost every slot, whose output is zero anyway). Also returns the fraction of tokens with a
    drop and their mean factor, for logging.
    """
    w = combine_weights.astype(jnp.float32)
    kept_total = jnp.sum(jnp.where(kept, w, 0.0), axis=-1)
    any_drop = jnp.any(~kept, axis=-1)
    dropped = any_drop & (kept_total > 0)
    factor = jnp.where(dropped, jnp.sum(w, axis=-1) / jnp.where(dropped, kept_total, 1.0), 1.0)
    num_dropped = jnp.sum(dropped.astype(jnp.float32))
    stats = {
        f"{_LAYER_KNOB_PREFIX}moe_drop_token_frac": jax.lax.stop_gradient(jnp.mean(any_drop.astype(jnp.float32))),
        f"{_LAYER_KNOB_PREFIX}moe_drop_renorm_factor": jax.lax.stop_gradient(
            jnp.sum(jnp.where(dropped, factor, 0.0)) / jnp.maximum(num_dropped, 1.0)
        ),
    }
    return factor, stats


def _ungated_expert_activation(cfg: "GrugModelConfig"):
    """Activation of the ungated experts: ``leaky_relu(u, slope)^2`` (plain ReLU^2 at slope 0)."""
    if cfg.expert_leaky_slope:
        if cfg.moe_fused_relu2:
            raise ValueError("moe_fused_relu2 implements plain ReLU^2 only; set expert_leaky_slope=0")
        slope = cfg.expert_leaky_slope
        return lambda u: jnp.square(jax.nn.leaky_relu(u, slope))
    return fused_relu2 if cfg.moe_fused_relu2 else ActivationFunctionEnum.relu2


def _tied_expert_activation(cfg: "GrugModelConfig", em: MoEExpertMlp):
    """``act`` with ``act(u) * u == leaky_relu(u, slope)^2`` for backends that take the gate tied to ``W_up``."""
    if cfg.moe_ungated_relu2 and cfg.expert_leaky_slope:
        slope_sq = cfg.expert_leaky_slope**2
        return lambda u: jax.nn.leaky_relu(u, slope_sq)
    return em.activation


def _logit_cap(cfg: "GrugModelConfig") -> float | tuple[float, float, float] | None:
    if cfg.logit_soft_cap_asym:
        a, b, c = cfg.logit_soft_cap_asym
        return (float(a), float(b), float(c))
    return cfg.logit_soft_cap


def _small_top_k(x: Float[Array, "T E"], k: int) -> tuple[Float[Array, "T k"], Int[Array, "T k"]]:
    """``jax.lax.top_k`` over the last axis as ``k`` unrolled max/argmax passes (values stop-gradient).

    On GPU, XLA lowers ``lax.top_k`` to a full sort: for the router (k = 9 of 384, 65k tokens per H100)
    that is 1.12 ms against 0.35 ms here. Ties resolve to the lowest index, as in ``lax.top_k``.
    """
    x = jax.lax.stop_gradient(x)
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
    values, indices = [], []
    for _ in range(k):
        index = jnp.argmax(x, axis=-1).astype(jnp.int32)
        values.append(jnp.max(x, axis=-1))
        indices.append(index)
        x = jnp.where(iota == index[..., None], -jnp.inf, x)
    return jnp.stack(values, axis=-1), jnp.stack(indices, axis=-1)


def _spread_mlp_input(x: Float[Array, "B S D"], cfg: GrugModelConfig) -> Float[Array, "B S D"]:
    """Center and/or whiten an MoE input with its batch statistics (``mlp_in_center``,
    ``mlp_in_whiten_power``). The statistics are stop-gradient, so the transform acts as a fixed
    per-step reparameterization of the input projections."""
    if not cfg.mlp_in_center and cfg.mlp_in_whiten_power == 0:
        return x
    flat = rearrange(x.astype(jnp.float32), "b s d -> (b s) d")
    mean = jax.lax.stop_gradient(jnp.mean(flat, axis=0))
    flat = flat - mean
    if cfg.mlp_in_whiten_power > 0:
        cov = jnp.einsum("td,te->de", flat, flat, out_sharding=P(None, None)) / flat.shape[0]
        evals, evecs = jnp.linalg.eigh(jax.lax.stop_gradient(cov))
        evals = jnp.maximum(evals, 0) / jnp.mean(evals) + cfg.mlp_in_whiten_eps
        whiten = (evecs * evals ** (-cfg.mlp_in_whiten_power / 2)) @ evecs.T
        flat = jnp.einsum("td,de->te", flat, jax.lax.stop_gradient(whiten), out_sharding=_batch_spec())
    return reshard(rearrange(flat, "(b s) d -> b s d", b=x.shape[0]).astype(x.dtype), _batch_spec())


def _attn_res_key_dim(cfg: GrugModelConfig) -> int:
    """Width of every AttnRes pseudo-query: ``attn_res_key_rank`` if set, else ``hidden_dim``."""
    return cfg.hidden_dim if cfg.attn_res_key_rank is None else cfg.attn_res_key_rank


@named_call
def _attn_res_source_logits(
    source: Float[Array, "B S D"], queries: Float[Array, "G D"], eps: float, head_norm: bool = False
) -> Float[Array, "G B S"]:
    """Float32 AttnRes logits of one source against ``G`` queries: ``q_g . rms_norm(source)``.

    RMS normalization is a per-token scalar, so it is applied to the ``[G, B, S]`` dot products instead
    of materializing normalized keys; a completed block is read once for every gate that will ever see it.
    The key norm is parameter-free: a learnable gain would be redundant with the query.
    Narrower single-head queries (``attn_res_key_rank``) score only the source's last ``r`` channels.
    """
    if queries.ndim == 2 and queries.shape[-1] < source.shape[-1]:
        source = source[..., -queries.shape[-1] :]
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(source.astype(jnp.float32)), axis=-1) + eps)
    if queries.ndim == 3 and queries.shape[-1] == source.shape[-1]:
        # Hierarchical multi-head AttnRes: full-width per-head queries [G, H, D] against the whole key.
        dots = jnp.einsum("bsd,ghd->gbsh", source, queries.astype(source.dtype), preferred_element_type=jnp.float32)
        return dots * inv_rms[None, ..., None]
    if queries.ndim == 3:
        # Multi-head AttnRes: queries are [G, H, D/H]; one logit per (gate, head) on that channel slice.
        heads = queries.shape[1]
        chunks = rearrange(source, "b s (h d) -> b s h d", h=heads)
        dots = jnp.einsum("bshd,ghd->gbsh", chunks, queries.astype(source.dtype), preferred_element_type=jnp.float32)
        if head_norm:
            head_inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(chunks.astype(jnp.float32)), axis=-1) + eps)
            return dots * head_inv_rms[None]
        return dots * inv_rms[None, ..., None]
    dots = jnp.einsum("bsd,gd->gbs", source, queries.astype(source.dtype), preferred_element_type=jnp.float32)
    return dots * inv_rms[None]


def _block_logit(block_logits: jax.Array, queries: jax.Array, gate_index: int) -> jax.Array:
    """Gate ``gate_index``'s row of a block's logits, which cover the trailing queries of the stack."""
    return block_logits[gate_index - (queries.shape[0] - block_logits.shape[0])]


def _softmax_mix(logits: list[jax.Array], sources: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Per-token softmax weights ``[N, B, S]`` over ``sources`` and the weighted sum (in float32)."""
    weights = jax.nn.softmax(jnp.stack(logits), axis=0)
    if weights.ndim == 4:
        # Multi-head: weights [N, B, S, H] mix each D/H channel slice separately.
        heads = weights.shape[-1]

        def chunked(x):
            return rearrange(x.astype(jnp.float32), "b s (h d) -> b s h d", h=heads)

        mixed = weights[0][..., None] * chunked(sources[0])
        for weight, source in zip(weights[1:], sources[1:], strict=True):
            mixed = mixed + weight[..., None] * chunked(source)
        return weights, rearrange(mixed, "b s h d -> b s (h d)")
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
    *,
    head_norm: bool = False,
    additive: bool = False,
    stream_source: bool = False,
    source_delta: bool = False,
    soft_cap: float | None = None,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """One AttnRes gate: softmax over the completed blocks (+ the running partial) and their weighted sum.

    ``additive`` (Delta AttnRes) adds the plain sum of the sources to the mix; ``head_norm`` scores
    each head's channel slice with its own RMS (``attn_res_head_norm``); ``soft_cap`` tanh-caps the final
    logits (``attn_res_logit_soft_cap``).

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
    if source_delta:
        # Output deltas: the precomputed block logits score raw sources, so score the deltas here.
        assert partial is None and extras is None, "attn_res_source_delta runs in full AttnRes without extras"
        sources = [sources[0]] + [cur - prev for prev, cur in itertools.pairwise(sources)]
        logits = [_attn_res_source_logits(src, queries[gate_index][None], eps, head_norm)[0] for src in sources]
    if partial is not None:
        sources.append(partial)
        logits.append(_attn_res_source_logits(partial, queries[gate_index][None], eps, head_norm)[0])
    logits = _bias_gate_logits(logits, extras, gate_index, sources, eps, has_partial=partial is not None)
    if stream_source:
        stream = _stream_sum(tuple(sources))
        sources.append(stream)
        logits.append(_attn_res_source_logits(stream, queries[gate_index][None], eps, head_norm)[0])
    logits = _soft_cap_logits(logits, soft_cap)
    weights, mixed = _softmax_mix(logits, sources)
    mixed = _gate_variants(mixed, sources, extras, gate_index, eps, has_partial=partial is not None)
    if extras is not None and extras.get("embed2") is not None:
        # nanogpt-style per-layer embedding input, scaled to the mix's RMS so lambda is a relative weight.
        scale = jnp.sqrt(jnp.mean(jnp.square(mixed), axis=-1, keepdims=True))
        mixed = mixed + extras["embed2_lambda"][gate_index] * scale * extras["embed2"].astype(jnp.float32)
    if additive:
        mixed = mixed + _stream_sum(tuple(sources)).astype(jnp.float32)
    mean_weights = jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim))))
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
    temperature = extras.get("temperature")
    if temperature is not None:
        logits = [logit * temperature[gate_index] for logit in logits]
    dyn_w1 = extras.get("dyn_w1")
    if dyn_w1 is not None:
        # MUDD-style per-token logit delta from the gate's current residual stream (W2 zero-init).
        total = sources[0].astype(jnp.float32)
        for src in sources[1:]:
            total = total + src.astype(jnp.float32)
        hid = jax.nn.gelu(jnp.einsum("bsd,dr->bsr", rms_norm(total, eps), dyn_w1[gate_index]))
        delta = jnp.einsum("bsr,rn->bsn", hid, extras["dyn_w2"][gate_index])
        logits = [logit + delta[..., c] for logit, c in zip(logits, columns, strict=True)]
    for name in ("bias", "mask"):
        table = extras.get(name)
        if table is not None:
            row = table[gate_index]
            logits = [logit + row[c] for logit, c in zip(logits, columns, strict=True)]
    return logits


def _soft_cap_logits(logits: list[jax.Array], cap: float | None) -> list[jax.Array]:
    """``cap * tanh(logit / cap)`` on each source logit (``attn_res_logit_soft_cap``); unchanged when None."""
    if cap is None:
        return logits
    return [cap * jnp.tanh(logit / cap) for logit in logits]


def _gate_variants(
    mixed: jax.Array,
    sources: list[jax.Array],
    extras: dict[str, jax.Array | None] | None,
    gate_index: int,
    eps: float,
    *,
    has_partial: bool,
) -> jax.Array:
    """Apply ``attn_res_dual_query`` and ``attn_res_blend`` to one gate's float32 mix of ``sources``."""
    if extras is None or (extras.get("dual") is None and extras.get("blend") is None):
        return mixed
    total = sources[0].astype(jnp.float32)
    for src in sources[1:]:
        total = total + src.astype(jnp.float32)
    stream = rms_norm(total, eps)
    dual = extras.get("dual")
    if dual is not None:
        logits = [_attn_res_source_logits(src, dual[gate_index][None], eps)[0] for src in sources]
        logits = _bias_gate_logits(logits, extras, gate_index, sources, eps, has_partial=has_partial)
        _, mixed2 = _softmax_mix(logits, sources)
        sel_logit = jnp.einsum("bsd,d->bs", stream, extras["dual_sel"][gate_index]) + extras["dual_sel_bias"][gate_index]
        sel = jax.nn.sigmoid(sel_logit)[..., None]
        mixed = sel * mixed + (1 - sel) * mixed2
    blend = extras.get("blend")
    if blend is not None:
        lam = jnp.broadcast_to(blend[gate_index], (*mixed.shape[:-1], 2))
        if extras.get("blend_proj") is not None:
            lam = lam + jnp.einsum("bsd,kd->bsk", stream, extras["blend_proj"][gate_index])
        mixed = lam[..., :1] * mixed + lam[..., 1:] * (total / len(sources))
    return mixed


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
    noise_key: jax.Array | None = None,
    kv_share: dict[str, jax.Array] | None = None,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """One Block AttnRes layer on ``diff_args = (layer, blocks, block_logits, partial, queries)``.

    ``partial`` is None right after a block boundary (it was just rolled into ``blocks``), in which
    case this layer starts a fresh partial sum.
    """
    layer, blocks, block_logits, partial, queries, logit_bias = diff_args
    cfg = layer.attn.cfg
    opts = {
        "head_norm": cfg.attn_res_head_norm,
        "additive": cfg.attn_res_additive,
        "stream_source": cfg.attn_res_stream_source,
        "source_delta": cfg.attn_res_source_delta,
        "soft_cap": cfg.attn_res_logit_soft_cap,
    }
    h, z_attn, w_attn = _attn_res_mix(blocks, block_logits, partial, queries, 2 * layer_index, eps, logit_bias, **opts)
    h = _ple_inject(layer, h, logit_bias)
    attn_branch = type(layer).attn_branch
    if cfg.attn_res_remat_attention:
        attn_branch = eqx.filter_checkpoint(attn_branch, policy=None)
    physical = layer_index % cfg.num_layers
    kda_ablation = (physical in cfg.kda_no_decay_layers, physical in cfg.kda_no_beta_layers)
    v_stream = None
    v_stats: dict[str, jax.Array] = {}
    if cfg.attn_res_v_gate:
        v_gate = 2 * cfg.num_layers + layer_index
        v_stream, _, w_v = _attn_res_mix(blocks, block_logits, partial, queries, v_gate, eps, logit_bias, **opts)
        v_stats[_ATTN_RES_W_V] = w_v
    attn_out, attn_stats = attn_branch(
        layer,
        h,
        mask,
        use_long,
        use_long,
        token_ids,
        kv_share,
        v_stream,
        ("v",) if v_stream is not None else (),
        kda_ablation,
        physical in cfg.value_residual_layers,
        _kv_stream_input(logit_bias),
    )
    # The MLP re-attends over the history including this layer's attention write, or without it
    # (moe_shortcut) so the MoE does not wait on the attention.
    shortcut_partial = partial
    partial = attn_out if partial is None else partial + attn_out
    mlp_partial = shortcut_partial if cfg.moe_shortcut else partial
    h, z_mlp, w_mlp = _attn_res_mix(
        blocks, block_logits, mlp_partial, queries, 2 * layer_index + 1, eps, logit_bias, **opts
    )
    mlp_out, router_stats = layer.mlp_branch(h, mask, **_route_kwargs(cfg, physical, token_ids, noise_key, logit_bias))
    mem_out, mem_stats = _memory_branch(h, logit_bias)
    if mem_out is not None:
        mlp_out = mlp_out + mem_out
    return partial + mlp_out, {
        **router_stats,
        **attn_stats,
        **v_stats,
        **mem_stats,
        **attn_stats,
        _ATTN_RES_Z: z_attn + z_mlp,
        _ATTN_RES_W_ATTN: w_attn,
        _ATTN_RES_W_MLP: w_mlp,
    }


def _attn_res_layer_full(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key=None, kv_share=None):
    """One full-AttnRes layer: the attention output becomes its own source before the MoE gate, and the
    MoE output is returned as the partial, which the next layer rolls into its own source. Returns
    ``(partial, blocks, block_logits, router_stats)`` like ``_attn_res_layer_passthrough``."""
    layer, blocks, block_logits, partial, queries, logit_bias = diff_args
    assert partial is None, "full AttnRes rolls every sublayer output into its own source"
    cfg = layer.attn.cfg
    opts = {
        "head_norm": cfg.attn_res_head_norm,
        "additive": cfg.attn_res_additive,
        "stream_source": cfg.attn_res_stream_source,
        "source_delta": cfg.attn_res_source_delta,
        "soft_cap": cfg.attn_res_logit_soft_cap,
    }
    sum_components = cfg.attn_res_sum_inputs
    h, z_attn, w_attn = _attn_res_mix(blocks, block_logits, None, queries, 2 * layer_index, eps, logit_bias, **opts)
    h = _ple_inject(layer, h, logit_bias)
    attn_side_stream = _stream_sum(blocks)
    v_stats: dict[str, jax.Array] = {}
    # V gate on KDA layers only: an MLA layer's K and V share one KV latent.
    if cfg.attn_res_v_gate and isinstance(layer.attn, KimiDeltaAttention):
        if sum_components:
            raise ValueError("attn_res_v_gate with attn_res_full does not combine with attn_res_sum_inputs")
        v_gate = 2 * cfg.num_layers + layer_index
        attn_side_stream, _, v_stats[_ATTN_RES_W_V] = _attn_res_mix(
            blocks, block_logits, None, queries, v_gate, eps, logit_bias, **opts
        )
        sum_components = ("v",)
    attn_out, attn_stats = type(layer).attn_branch(
        layer,
        h,
        mask,
        use_long,
        use_long,
        token_ids,
        kv_share,
        attn_side_stream,
        sum_components,
        value_residual=layer_index % cfg.num_layers in cfg.value_residual_layers,
        kv_input=_kv_stream_input(logit_bias),
    )
    sum_components = cfg.attn_res_sum_inputs
    # moe_shortcut: the MoE gate reads the history before this layer's attention write.
    shortcut_history = (blocks, block_logits)
    first_reader = 2 * layer_index + 1 + int(cfg.moe_shortcut)
    blocks = (*blocks, attn_out)
    block_logits = (
        *block_logits,
        _attn_res_source_logits(attn_out, queries[first_reader:], eps, cfg.attn_res_head_norm),
    )
    mlp_blocks, mlp_block_logits = shortcut_history if cfg.moe_shortcut else (blocks, block_logits)
    h, z_mlp, w_mlp = _attn_res_mix(
        mlp_blocks, mlp_block_logits, None, queries, 2 * layer_index + 1, eps, logit_bias, **opts
    )
    if "mlp" in sum_components:
        h = _stream_sum(mlp_blocks)
    sum_parts: tuple[str, ...] = ()
    if "mlp_shared" in sum_components:
        sum_parts += ("shared",)
    if "mlp_routed" in sum_components:
        sum_parts += ("router", "latent")
    if "mlp_router" in sum_components:
        sum_parts += ("router",)
    sum_parts = tuple(dict.fromkeys(sum_parts))
    mlp_out, router_stats = layer.mlp_branch(
        h,
        mask,
        _stream_sum(mlp_blocks) if sum_parts else None,
        sum_parts,
        **_route_kwargs(cfg, layer_index % cfg.num_layers, token_ids, noise_key, logit_bias),
    )
    mem_out, mem_stats = _memory_branch(h, logit_bias)
    if mem_out is not None:
        mlp_out = mlp_out + mem_out
    stats = {
        **router_stats,
        **attn_stats,
        **v_stats,
        **mem_stats,
        **attn_stats,
        _ATTN_RES_Z: z_attn + z_mlp,
        _ATTN_RES_W_ATTN: w_attn,
        _ATTN_RES_W_MLP: w_mlp,
    }
    return mlp_out, blocks, block_logits, stats


def _kv_stream_input(extras: dict[str, jax.Array | None] | None) -> jax.Array | None:
    """This layer's normed KV side-stream state ``extras["kv_stream"]`` (``kv_stream_dim``), or None."""
    return None if extras is None else extras.get("kv_stream")


def _route_kwargs(
    cfg: GrugModelConfig,
    physical_layer: int,
    token_ids: jax.Array,
    noise_key: jax.Array | None,
    extras: dict[str, jax.Array | None] | None,
) -> dict[str, jax.Array | None]:
    """Router extras for one layer: token ids on ``moe_hash_layers``, the per-layer noise key otherwise, the
    tokens' ``router_token_bias_rank`` rows (``extras["router_tok"]``) and the ``router_bias_seed`` bias."""
    return {
        "hash_token_ids": token_ids if physical_layer in cfg.moe_hash_layers else None,
        "noise_key": noise_key,
        "router_tok_rows": None if extras is None else extras.get("router_tok"),
        "router_seed_bias": _router_seed_bias(cfg, physical_layer, token_ids) if cfg.router_bias_seed else None,
    }


def _stream_sum(sources: tuple[jax.Array, ...]) -> jax.Array:
    """The straight sum of the AttnRes sources (a standard residual stream), in the sources' dtype."""
    total = sources[0].astype(jnp.float32)
    for src in sources[1:]:
        total = total + src.astype(jnp.float32)
    return reshard(total.astype(sources[0].dtype), _batch_spec())


def _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key=None, kv_share=None):
    """``_attn_res_layer`` returning ``(partial, blocks, block_logits, router_stats)``: the history is
    passed through so ``_attn_res_layer_remat`` can thread each block's cotangent layer to layer."""
    _, blocks, block_logits, _, _, _ = diff_args
    partial, router_stats = _attn_res_layer(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key, kv_share)
    return partial, blocks, block_logits, router_stats


@eqx.filter_custom_vjp
def _attn_res_layer_remat(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key):
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
    return _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key)


@_attn_res_layer_remat.def_fwd
def _attn_res_layer_remat_fwd(perturbed, diff_args, mask, token_ids, use_long, layer_index, eps, noise_key):
    del perturbed
    return _attn_res_layer_passthrough(diff_args, mask, token_ids, use_long, layer_index, eps, noise_key), None


@_attn_res_layer_remat.def_bwd
def _attn_res_layer_remat_bwd(
    residuals, grad_out, perturbed, diff_args, mask, token_ids, use_long, layer_index, eps, noise_key
):
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
            lambda args: _attn_res_layer(args, mask, token_ids, use_long, layer_index, eps, noise_key)[0], diff_args
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


class KvStreamBlock(eqx.Module):
    """One pre-norm block of the KV side stream (``kv_stream_dim`` = ``w``): ``s += attn(rms(s))`` then
    ``s += mlp(rms(s))``. The attention is plain causal softmax attention within the side stream, masked
    by the main model's full-causal document mask, with weightless per-head QK RMSNorm and full RoPE
    (``cfg.rope``); the MLP is ungated ReLU^2 of width ``kv_stream_mlp_mult * w``. ``kv_norm`` is the learned
    RMSNorm of the block's output that the paired main layer's K/V projections read. All matrices are
    random-init at ``cfg.initializer_std`` (MuonH)."""

    rms_attn: LearnedRMSNorm
    w_q: Float[Array, "W W"]
    w_k: Float[Array, "W W"]
    w_v: Float[Array, "W W"]
    w_o: Float[Array, "W W"]
    rms_mlp: LearnedRMSNorm
    w_up: Float[Array, "W M"]
    w_down: Float[Array, "M W"]
    kv_norm: LearnedRMSNorm
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KvStreamBlock":
        w, std, eps = cfg.kv_stream_dim, cfg.initializer_std, cfg.layer_norm_eps
        m = cfg.kv_stream_mlp_mult * w
        k_q, k_k, k_v, k_o, k_up, k_down = random.split(key, 6)
        return KvStreamBlock(
            rms_attn=_learned_rms_norm(cfg, w, eps),
            w_q=reshard(_init_weight(k_q, (w, w), std), P(_FSDP_AXES, None)),
            w_k=reshard(_init_weight(k_k, (w, w), std), P(_FSDP_AXES, None)),
            w_v=reshard(_init_weight(k_v, (w, w), std), P(_FSDP_AXES, None)),
            w_o=reshard(_init_weight(k_o, (w, w), std), P(None, _FSDP_AXES)),
            rms_mlp=_learned_rms_norm(cfg, w, eps),
            w_up=reshard(_init_weight(k_up, (w, m), std), P(_FSDP_AXES, None)),
            w_down=reshard(_init_weight(k_down, (m, w), std), P(None, _FSDP_AXES)),
            kv_norm=_learned_rms_norm(cfg, w, eps),
            cfg=cfg,
        )

    @named_call
    def __call__(self, s: Float[Array, "B S W"], mask: AttentionMask) -> Float[Array, "B S W"]:
        head_dim = self.cfg.kv_stream_dim // self.cfg.kv_stream_heads
        x = self.rms_attn(s)

        def heads(w: jax.Array) -> jax.Array:
            return rearrange(jnp.einsum("bsw,wd->bsd", x, w), "... (n d) -> ... n d", d=head_dim)

        q, k = apply_rotary_embedding(
            rms_norm(heads(self.w_q)),
            rms_norm(heads(self.w_k)),
            seq_len=s.shape[1],
            head_dim=head_dim,
            rope=self.cfg.rope,
        )
        attn_impl = "gpu_fa4_cute" if jax.default_backend() == "gpu" else None
        o = attention(q, k, heads(self.w_v), mask, implementation=attn_impl)
        o = jnp.reshape(o, (*o.shape[:-2], o.shape[-2] * o.shape[-1]), out_sharding=P(_BATCH_AXES, None, None))
        s = s + jnp.einsum("bsd,dw->bsw", o, self.w_o, out_sharding=_batch_spec())
        hidden = jnp.square(jax.nn.relu(jnp.einsum("bsw,wm->bsm", self.rms_mlp(s), self.w_up)))
        return s + jnp.einsum("bsm,mw->bsw", hidden, self.w_down, out_sharding=_batch_spec())


class KvStream(eqx.Module):
    """The KV side stream of ``kv_stream_dim``: its own token embedding (RMS-normed, no gain) advanced by
    one ``KvStreamBlock`` per main layer. It reads only the token ids, so it runs once before the main
    layers; main layer ``l`` reads ``blocks[l].kv_norm`` of the state leaving side block ``l`` (so layer 0's
    K/V already had one side block). Under ``AttnResLayerBackward.RECOMPUTE`` each side block is
    rematerialized in the backward (only its input is saved), like the main layers."""

    token_embed: Float[Array, "V W"]
    blocks: ArrayStacked[KvStreamBlock]

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KvStream":
        k_embed, k_blocks = random.split(key)
        return KvStream(
            token_embed=reshard(
                _init_weight(k_embed, (cfg.vocab_size, cfg.kv_stream_dim), cfg.initializer_std), P(None, None)
            ),
            blocks=ArrayStacked.init(cfg.num_layers, KvStreamBlock)(cfg, key=random.split(k_blocks, cfg.num_layers)),
        )

    def __call__(
        self, token_ids: Int[Array, "B S"], mask: AttentionMask, gather
    ) -> tuple[list[Float[Array, "B S W"]], dict[str, jax.Array]]:
        """Per main layer, the normed side-stream state it reads; and each state's RMS for logging."""
        s = rms_norm(gather(self.token_embed, token_ids))
        kv_inputs, stats = [], {}
        for i, block in enumerate(_unstack_layers(self.blocks)):
            recompute = block.cfg.attn_res_layer_backward == AttnResLayerBackward.RECOMPUTE
            s = (eqx.filter_checkpoint(block, policy=None) if recompute else block)(s, mask)
            kv_inputs.append(block.kv_norm(s))
            stats[f"{_LAYER_KNOB_PREFIX}kv_stream_rms_L{i}"] = jnp.sqrt(
                jnp.mean(jnp.square(jax.lax.stop_gradient(s).astype(jnp.float32)))
            )
            for name in ("w_q", "w_k", "w_v", "w_o", "w_up", "w_down"):
                weight = jax.lax.stop_gradient(getattr(block, name)).astype(jnp.float32)
                stats[f"{_LAYER_KNOB_PREFIX}kv_stream_{name}_norm_L{i}"] = jnp.linalg.norm(weight)
            gain = jnp.concatenate([jax.lax.stop_gradient(g).reshape(-1) for g in jax.tree.leaves(block.kv_norm)])
            stats[f"{_LAYER_KNOB_PREFIX}kv_stream_kv_norm_gain_mean_L{i}"] = jnp.mean(gain)
        return kv_inputs, stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: LearnedRMSNorm
    embed_gated_norm: GatedNorm | None
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[Block]
    """The softmax-attention layers: every layer, or the global layers when the local layers are KDA."""
    kda_blocks: ArrayStacked[Block] | None
    """The KDA (local) layers. The AttnRes loop splits each stack whole into its layers (never slices
    it), so the hybrid keeps one stack per mixer kind: fewer, larger optimizer leaves."""
    final_norm: LearnedRMSNorm
    final_gated_norm: GatedNorm | None
    attn_res_query_final: Float[Array, " D"] | None
    """Pseudo-query of the final AttnRes gate, whose mix feeds the final norms and the lm_head."""
    token_embed2: jax.Array | None
    embed2_norm: LearnedRMSNorm | None
    bigram_gate_w: Float[Array, " D"] | None
    bigram_gate_b: Float[Array, ""] | None
    bigram_gate_a_lr: Float[Array, "D R"] | None
    bigram_gate_b_lr: Float[Array, "R D"] | None
    trigram_gate_w: Float[Array, " D"] | None
    trigram_gate_b: Float[Array, ""] | None
    trigram_gate_a_lr: Float[Array, "D R"] | None
    trigram_gate_b_lr: Float[Array, "R D"] | None
    token_embed3: jax.Array | None
    embed3_norm: LearnedRMSNorm | None
    ngram_stat_table: jax.Array | None
    """``[len(ngram_stat_orders) * ngram_stat_rows, ngram_stat_dim + 1]`` fp32 sums and counts, one block of rows
    per order; frozen for the optimizer."""
    ngram_stat_code: jax.Array | None
    """``[vocab, ngram_stat_dim]`` fixed random next-token code; frozen for the optimizer."""
    ngram_stat_hidden: jax.Array | None
    ngram_stat_up: jax.Array | None
    ngram_stat_norm: RMSNorm | None
    ngram_stat_gate_w: Float[Array, " D"] | None
    ngram_stat_gate_b: Float[Array, ""] | None
    token_embed_window: jax.Array | None
    byte_head: jax.Array | None
    window_proj: jax.Array | None
    window_norm: LearnedRMSNorm | None
    embed2_up: jax.Array | None
    token_embed_ple: jax.Array | None
    """Per-layer-embedding table ``[vocab, num_layers * ple_dim]`` (``ple_dim``)."""
    router_tok_a: Float[Array, "V r"] | None
    """Shared token table of the router token-identity bias (``router_token_bias_rank``), ``N(0, 1/r)``."""
    router_tie_alpha: Float[Array, " N"] | None
    """Per-tie scale of the ``router_embed_tie`` router columns ``alpha * token_embed[V]``."""
    memory: tuple[ProductKeyMemory, ...] | None
    """Product-key memories of ``memory_layers``, in that order."""
    embed2_lambda: Float[Array, " G"] | None
    """Per-gate weight of the second embedding on each sublayer input (``second_embed_mode="input"``)."""
    attn_res_query_bias: Float[Array, "G N"] | None
    """AttnRes logit bias per (gate, source); the last column is the running partial."""
    attn_res_query_pull: Float[Array, "N D"] | None
    """Pull-AttnRes source keys (``attn_res_pull``); the last row is the partial's."""
    attn_res_query_embed: Float[Array, "G D"] | None
    """Per-gate pull projection for the embedding logit (``attn_res_pull_embed``)."""
    attn_res_query_dyn1: Float[Array, "G D R"] | None
    attn_res_query_dyn2: Float[Array, "G R N"] | None
    """``attn_res_dynamic_rank`` MLP per gate, in query-stack order (``dyn2`` zero-init)."""
    attn_res_query_backout: Float[Array, " N"] | None
    """Signed per-source correction of the final AttnRes gate (``attn_res_final_signed``), zero-init."""
    smear_w: Float[Array, " 12"] | None
    smear_lambda: Float[Array, ""] | None
    attn_res_query_sub: Float[Array, "G H S"] | None
    """Hierarchical multi-head AttnRes sub-queries (``attn_res_head_sub``), zero-init, in query-stack order."""
    attn_res_query_temp: Float[Array, "G H"] | None
    """Per-gate (and per-head) logit multiplier (``attn_res_temperature``), init 1."""
    attn_res_query_dual: Float[Array, "G D"] | None
    """Second pseudo-query per gate (``attn_res_dual_query``)."""
    attn_res_query_dual_sel: Float[Array, "G D"] | None
    attn_res_query_dual_sel_bias: Float[Array, " G"] | None
    attn_res_query_blend: Float[Array, "G 2"] | None
    """Per-gate ``(l1, l2)`` of ``attn_res_blend``."""
    attn_res_query_blend_proj: Float[Array, "G 2 D"] | None
    mtp: "MtpHead | None"
    """The DeepSeek-V3 MTP module (``mtp_mode``)."""
    nitp_w1: Float[Array, "D D"] | None
    nitp_w2: Float[Array, "D D"] | None
    """The NITP predictor head's two matrices (``nitp_weight``), random init (MuonH)."""
    attn_res_query_loop: Float[Array, "P G D"] | None
    """AttnRes pseudo-queries of the extra loop passes (``loop_passes - 1`` of them), zero-init."""
    loop_inject_scale: Float[Array, " P"] | None
    """Input-injection scale per extra loop pass, init 1."""
    output_bigram_u: Float[Array, "V R"] | None
    output_bigram_w: Float[Array, "R V"] | None
    """Output bigram prior (``output_bigram_rank``): current-token code table ``U`` and zero-init read-out ``W``."""
    lm_head_bias: Float[Array, " V"] | None
    """Output logit bias (``lm_head_unigram_bias``), float32 even in the compute copy."""
    kv_stream: KvStream | None
    """The KV side stream (``kv_stream_dim``), the only source of the layers' keys and values."""
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
            layer_index = jnp.asarray(layers, dtype=jnp.int32)
            return ArrayStacked.init(len(layers), Block)(cfg, key=keys, layer_index=layer_index, use_kda=use_kda)

        softmax_layers, kda_layers = _stack_layer_indices(cfg)
        model = Transformer(
            token_embed=token_embed,
            embed_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key) if cfg.embed_gated_norm else None
            ),
            output_proj=output_proj,
            stacked_blocks=stack(softmax_layers, False),
            kda_blocks=stack(kda_layers, True) if kda_layers else None,
            final_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key) if cfg.final_gated_norm else None
            ),
            attn_res_query_final=(
                reshard(jnp.zeros((_attn_res_key_dim(cfg),), jnp.float32), P(None)) if cfg.attn_res else None
            ),
            token_embed2=(
                reshard(
                    _init_weight(
                        embed2_key,
                        (
                            (cfg.embed2_rows or cfg.vocab_size) * cfg.embed2_hash_heads,
                            (cfg.embed2_dim or cfg.hidden_dim) // cfg.embed2_hash_heads,
                        ),
                        cfg.initializer_std,
                    ),
                    P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
                )
                if cfg.second_embed
                else None
            ),
            token_embed_ple=(
                reshard(
                    _init_weight(
                        random.fold_in(embed2_key, 17),
                        (cfg.vocab_size, cfg.num_layers * cfg.ple_dim),
                        cfg.initializer_std,
                    ),
                    P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
                )
                if cfg.ple_dim
                else None
            ),
            router_tok_a=(
                reshard(
                    _init_weight(
                        random.fold_in(embed2_key, 19),
                        (cfg.vocab_size, cfg.router_token_bias_rank),
                        1.0 / math.sqrt(cfg.router_token_bias_rank),
                    ),
                    P(None, None),
                )
                if cfg.router_token_bias_rank
                else None
            ),
            memory=(
                tuple(
                    ProductKeyMemory.init(cfg, key=random.fold_in(embed2_key, 1000 + layer))
                    for layer in cfg.memory_layers
                )
                if cfg.memory_layers
                else None
            ),
            embed2_lambda=(
                jnp.full(
                    (_attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate),),
                    cfg.embed2_lambda_init,
                    jnp.float32,
                )
                if cfg.second_embed and cfg.second_embed_mode == "input"
                else None
            ),
            embed2_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps) if cfg.second_embed else None,
            bigram_gate_w=jnp.zeros((cfg.hidden_dim,), jnp.float32) if cfg.bigram_gate else None,
            bigram_gate_b=jnp.full((), 2.0, jnp.float32) if cfg.bigram_gate else None,
            trigram_gate_w=jnp.zeros((cfg.hidden_dim,), jnp.float32) if cfg.trigram_gate else None,
            trigram_gate_b=jnp.full((), 2.0, jnp.float32) if cfg.trigram_gate else None,
            trigram_gate_a_lr=(
                _init_weight(
                    random.fold_in(embed2_key, 13),
                    (cfg.hidden_dim, cfg.bigram_gate_rank),
                    1.0 / math.sqrt(cfg.hidden_dim),
                )
                if cfg.trigram_gate and cfg.bigram_gate_rank
                else None
            ),
            trigram_gate_b_lr=(
                jnp.zeros((cfg.bigram_gate_rank, cfg.hidden_dim), jnp.float32)
                if cfg.trigram_gate and cfg.bigram_gate_rank
                else None
            ),
            bigram_gate_a_lr=(
                _init_weight(
                    random.fold_in(embed2_key, 9),
                    (cfg.hidden_dim, cfg.bigram_gate_rank),
                    1.0 / math.sqrt(cfg.hidden_dim),
                )
                if cfg.bigram_gate and cfg.bigram_gate_rank
                else None
            ),
            bigram_gate_b_lr=(
                jnp.zeros((cfg.bigram_gate_rank, cfg.hidden_dim), jnp.float32)
                if cfg.bigram_gate and cfg.bigram_gate_rank
                else None
            ),
            token_embed3=(
                reshard(
                    _init_weight(random.fold_in(embed2_key, 3), (cfg.embed3_rows, cfg.hidden_dim), cfg.initializer_std),
                    P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
                )
                if cfg.embed3_rows
                else None
            ),
            embed3_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps) if cfg.embed3_rows else None,
            ngram_stat_table=(
                reshard(
                    jnp.zeros((len(cfg.ngram_stat_orders) * cfg.ngram_stat_rows, cfg.ngram_stat_dim + 1), jnp.float32),
                    P(None, None),
                )
                if cfg.ngram_stat_rows
                else None
            ),
            ngram_stat_code=(
                reshard(
                    random.normal(
                        random.PRNGKey(_NGRAM_STAT_CODE_SEED), (cfg.vocab_size, cfg.ngram_stat_dim), jnp.float32
                    )
                    / math.sqrt(cfg.ngram_stat_dim),
                    P(None, None),
                )
                if cfg.ngram_stat_rows
                else None
            ),
            ngram_stat_hidden=(
                reshard(
                    _init_weight(
                        random.fold_in(embed2_key, 22),
                        (_ngram_stat_feature_dim(cfg), cfg.ngram_stat_mlp_dim),
                        1.0 / math.sqrt(_ngram_stat_feature_dim(cfg)),
                    ),
                    P(None, None),
                )
                if cfg.ngram_stat_rows and cfg.ngram_stat_mlp_dim
                else None
            ),
            ngram_stat_up=(
                reshard(
                    _init_weight(
                        random.fold_in(embed2_key, 21),
                        (cfg.ngram_stat_mlp_dim or _ngram_stat_feature_dim(cfg), cfg.hidden_dim),
                        1.0 / math.sqrt(cfg.ngram_stat_mlp_dim or _ngram_stat_feature_dim(cfg)),
                    )
                    * (cfg.ngram_stat_mode is NgramStatMode.SOURCE),  # zero-init (no-harm) in BIGRAM mode
                    P(None, None),
                )
                if cfg.ngram_stat_rows
                else None
            ),
            ngram_stat_norm=(RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps) if _ngram_stat_source_mode(cfg) else None),
            ngram_stat_gate_w=(
                jnp.zeros((cfg.hidden_dim,), jnp.float32)
                if _ngram_stat_source_mode(cfg) and cfg.ngram_stat_gate
                else None
            ),
            ngram_stat_gate_b=(
                jnp.full((), 2.0, jnp.float32) if _ngram_stat_source_mode(cfg) and cfg.ngram_stat_gate else None
            ),
            token_embed_window=(
                reshard(
                    _init_weight(random.fold_in(embed2_key, 5), (cfg.vocab_size, cfg.window_embed_dim), 1.0),
                    P(None, None),
                )
                if cfg.window_embed_dim
                else None
            ),
            window_proj=(
                reshard(
                    _init_weight(random.fold_in(embed2_key, 6), (cfg.hidden_dim, cfg.hidden_dim), cfg.initializer_std),
                    P(_FSDP_AXES, "model"),
                )
                if cfg.window_embed_dim
                else None
            ),
            window_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps) if cfg.window_embed_dim else None,
            byte_head=(
                reshard(
                    _init_weight(
                        random.fold_in(out_key, 11),
                        (cfg.hidden_dim, cfg.byte_aux_bytes * _BYTE_CLASSES),
                        cfg.initializer_std,
                    ),
                    P(None, None),
                )
                if cfg.byte_aux_bytes
                else None
            ),
            embed2_up=(
                reshard(
                    _init_weight(
                        random.fold_in(embed2_key, 1),
                        (cfg.embed2_dim, cfg.hidden_dim),
                        1.0 / math.sqrt(cfg.embed2_dim),
                    ),
                    P(None, None),
                )
                if cfg.second_embed and cfg.embed2_dim
                else None
            ),
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
            attn_res_query_dyn1=(
                (1.0 / math.sqrt(cfg.hidden_dim))
                * random.normal(
                    random.fold_in(key, 13),
                    (
                        _attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate),
                        cfg.hidden_dim,
                        cfg.attn_res_dynamic_rank,
                    ),
                    jnp.float32,
                )
                if cfg.attn_res_dynamic_rank > 0
                else None
            ),
            attn_res_query_dyn2=(
                jnp.zeros(
                    (
                        _attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate),
                        cfg.attn_res_dynamic_rank,
                        _attn_res_num_sources(cfg),
                    ),
                    jnp.float32,
                )
                if cfg.attn_res_dynamic_rank > 0
                else None
            ),
            attn_res_query_backout=(
                jnp.zeros((_attn_res_num_sources(cfg),), jnp.float32) if cfg.attn_res_final_signed else None
            ),
            smear_w=jnp.zeros((12,), jnp.float32) if cfg.smear else None,
            smear_lambda=jnp.zeros((), jnp.float32) if cfg.smear else None,
            attn_res_query_sub=(
                jnp.zeros(
                    (
                        _attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate),
                        cfg.attn_res_heads,
                        cfg.hidden_dim if cfg.attn_res_head_sub == "full" else cfg.hidden_dim // cfg.attn_res_heads,
                    ),
                    jnp.float32,
                )
                if cfg.attn_res_head_sub != "none"
                else None
            ),
            attn_res_query_temp=(
                jnp.ones((_attn_res_num_gates(cfg) + cfg.num_layers * int(cfg.attn_res_v_gate), cfg.attn_res_heads))
                if cfg.attn_res_temperature
                else None
            ),
            attn_res_query_dual=(
                cfg.attn_res_dual_init_std
                * random.normal(random.fold_in(key, 11), (_attn_res_num_gates(cfg), cfg.hidden_dim), jnp.float32)
                if cfg.attn_res_dual_query
                else None
            ),
            attn_res_query_dual_sel=(
                jnp.zeros((_attn_res_num_gates(cfg), cfg.hidden_dim), jnp.float32) if cfg.attn_res_dual_query else None
            ),
            attn_res_query_dual_sel_bias=(
                jnp.zeros((_attn_res_num_gates(cfg),), jnp.float32) if cfg.attn_res_dual_query else None
            ),
            attn_res_query_blend=(
                jnp.ones((_attn_res_num_gates(cfg), 2), jnp.float32) if cfg.attn_res_blend != "none" else None
            ),
            attn_res_query_blend_proj=(
                jnp.zeros((_attn_res_num_gates(cfg), 2, cfg.hidden_dim), jnp.float32)
                if cfg.attn_res_blend == "dynamic"
                else None
            ),
            mtp=MtpHead.init(cfg, key=random.fold_in(key, 3)) if cfg.mtp_mode == MtpMode.DEEPSEEK else None,
            nitp_w1=(
                reshard(
                    _init_weight(random.fold_in(key, 4), (cfg.hidden_dim, cfg.hidden_dim), cfg.initializer_std),
                    P(_FSDP_AXES, None),
                )
                if cfg.nitp_weight > 0
                else None
            ),
            nitp_w2=(
                reshard(
                    _init_weight(random.fold_in(key, 5), (cfg.hidden_dim, cfg.hidden_dim), cfg.initializer_std),
                    P(_FSDP_AXES, None),
                )
                if cfg.nitp_weight > 0
                else None
            ),
            attn_res_query_loop=(
                reshard(jnp.zeros((cfg.loop_passes - 1, 2 * cfg.num_layers, _attn_res_key_dim(cfg)), jnp.float32), P())
                if cfg.loop_passes > 1
                else None
            ),
            loop_inject_scale=jnp.ones((cfg.loop_passes - 1,), jnp.float32) if cfg.loop_passes > 1 else None,
            output_bigram_u=(
                reshard(
                    _init_weight(random.fold_in(key, 17), (cfg.vocab_size, cfg.output_bigram_rank), cfg.initializer_std),
                    P(None, None),
                )
                if cfg.output_bigram_rank
                else None
            ),
            output_bigram_w=(
                reshard(jnp.zeros((cfg.output_bigram_rank, cfg.vocab_size), jnp.float32), P(None, "model"))
                if cfg.output_bigram_rank
                else None
            ),
            lm_head_bias=(
                reshard(jnp.zeros((cfg.vocab_size,), jnp.float32), P(None)) if cfg.lm_head_unigram_bias else None
            ),
            kv_stream=KvStream.init(cfg, key=random.fold_in(key, _KV_STREAM_KEY_SALT)) if cfg.kv_stream_dim else None,
            router_tie_alpha=None,
            config=cfg,
        )
        if cfg.router_embed_tie:
            model = eqx.tree_at(lambda m: m.router_tie_alpha, model, _router_tie_alpha_init(model), is_leaf=_is_none)
        return model

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
        route_key: jax.Array | None = None,
        return_routing: bool = False,
        router_tie_active: bool | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``loop_active`` (static, with ``loop_grow_step``) selects one pass (False) or all ``loop_passes``
        (True); None runs all passes. ``route_key`` (training only) seeds ``moe_gumbel_tau``.
        ``router_tie_active`` (static) applies the ``router_embed_tie`` ties; None applies them unless
        ``router_embed_tie_release_step`` is set (a released model's router columns already hold them).
        ``return_routing`` adds every layer's ``[L, T, K]`` selected experts and combine weights to the
        metrics (``ROUTING_SELECTED_KEY`` / ``ROUTING_WEIGHTS_KEY``)."""
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        if cfg.router_share_block > 1:
            self = _share_routers(self, cfg.router_share_block)
        if cfg.router_embed_tie and (
            cfg.router_embed_tie_release_step is None if router_tie_active is None else router_tie_active
        ):
            self = tie_routers(self)
        gather = _embedding_gather if cfg.embed_grad_fp32 else _embedding_gather_autodiff
        hidden = raw_embed = gather(self.token_embed, token_ids)
        if cfg.embed_norm_mode == "rms":
            hidden = self.embed_norm(hidden)
        elif cfg.embed_norm_mode == "rms_nogain":
            hidden = rms_norm(hidden.astype(jnp.float32), cfg.layer_norm_eps).astype(hidden.dtype)
        elif cfg.embed_norm_mode != "raw":
            raise ValueError(f"embed_norm_mode must be rms, rms_nogain or raw, got {cfg.embed_norm_mode!r}")
        if cfg.embed_scale != 1.0:
            hidden = (hidden * cfg.embed_scale).astype(hidden.dtype)
        if self.embed_gated_norm is not None:
            hidden = self.embed_gated_norm(hidden)

        # Local layers use a sliding window; every global_every-th layer is full causal.
        segment_ids = None
        if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
            # Pin the [B, S] segment ids batch-sharded and reuse one array for both attention sides.
            q_segment_ids, _ = mask.segment_ids
            q_segment_ids = _batch_reshard(q_segment_ids)
            segment_ids = (q_segment_ids, q_segment_ids)
        if self.smear_w is not None and self.smear_lambda is not None:
            seg = None if segment_ids is None else segment_ids[0]
            hidden = _smear(hidden, self.smear_w, self.smear_lambda, seg)
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
        bigram_gate_stats: dict[str, jax.Array] = {}
        if (cfg.second_embed or cfg.ple_dim) and not cfg.attn_res:
            raise ValueError("second_embed and ple_dim require attn_res")
        if cfg.value_residual_layers and not cfg.attn_res:
            raise ValueError("value_residual_layers requires attn_res")
        if cfg.attn_res:
            extra_sources = ()
            input_embed2 = None
            if cfg.second_embed_mode not in ("source", "input"):
                raise ValueError(f"second_embed_mode must be source or input, got {cfg.second_embed_mode!r}")
            if self.token_embed2 is not None:
                assert self.embed2_norm is not None
                ids2 = token_ids
                gather2 = _embedding_gather if cfg.embed2_grad_fp32 else _embedding_gather_autodiff
                table2 = reshard(self.token_embed2, P(None, None)) if cfg.embed2_fsdp else self.token_embed2
                if cfg.second_embed_bigram and cfg.embed2_hash_heads > 1:
                    doc_start = None if segment_ids is None else segment_ids[0]
                    rows_per_head = cfg.embed2_rows or cfg.vocab_size
                    heads = cfg.embed2_hash_heads
                    if cfg.embed2_head_orders and len(cfg.embed2_head_orders) != heads:
                        raise ValueError(f"embed2_head_orders needs {heads} entries, got {cfg.embed2_head_orders}")
                    head_ids = jnp.stack(
                        [
                            _bigram_hash_ids(token_ids, doc_start, rows_per_head, order, salt=h) + h * rows_per_head
                            for h, order in enumerate(cfg.embed2_head_orders or (cfg.embed2_ngram,) * heads)
                        ],
                        axis=-1,
                    )
                    b_, s_ = token_ids.shape
                    flat_ids = jax.lax.reshape(head_ids, (b_, s_ * heads), out_sharding=P(_BATCH_AXES, None))
                    head_rows = gather2(table2, flat_ids)
                    rows2 = jax.lax.reshape(
                        head_rows, (b_, s_, head_rows.shape[-1] * heads), out_sharding=P(_BATCH_AXES, None, None)
                    )
                else:
                    if cfg.second_embed_bigram:
                        doc_start = None if segment_ids is None else segment_ids[0]
                        ids2 = _bigram_hash_ids(
                            token_ids, doc_start, cfg.embed2_rows or cfg.vocab_size, cfg.embed2_ngram
                        )
                    rows2 = gather2(table2, ids2)
                if self.embed2_up is not None:
                    rows2 = jnp.einsum(
                        "bsr,rd->bsd", rows2, self.embed2_up.astype(rows2.dtype), out_sharding=_batch_spec()
                    )
                embed2 = self.embed2_norm(rows2)
                if self.bigram_gate_w is not None and self.bigram_gate_b is not None:
                    embed2, gate = _content_gate(
                        hidden,
                        embed2,
                        self.bigram_gate_w,
                        self.bigram_gate_b,
                        self.bigram_gate_a_lr,
                        self.bigram_gate_b_lr,
                    )
                    bigram_gate_stats = {
                        "attn_res_bigram_gate_mean": jax.lax.stop_gradient(jnp.mean(gate)),
                        "attn_res_bigram_gate_std": jax.lax.stop_gradient(jnp.std(gate)),
                    }
                if self.ngram_stat_table is not None and cfg.ngram_stat_mode is NgramStatMode.BIGRAM:
                    if not cfg.second_embed_bigram:
                        raise ValueError("ngram_stat_mode=bigram needs second_embed_bigram")
                    doc_start = None if segment_ids is None else segment_ids[0]
                    embed2 = embed2 + self._ngram_stat_read(hidden, token_ids, doc_start)
                if cfg.second_embed_mode == "input":
                    input_embed2 = embed2
                else:
                    extra_sources = (embed2,)
            if self.token_embed_window is not None:
                assert self.window_proj is not None and self.window_norm is not None
                doc_start = None if segment_ids is None else segment_ids[0]
                window = _token_window(gather(self.token_embed_window, token_ids), doc_start, cfg.hidden_dim)
                projected = jnp.einsum(
                    "bsd,de->bse", window, self.window_proj.astype(window.dtype), out_sharding=_batch_spec()
                )
                if cfg.window_embed_mode == "add_embed":
                    # Summed into the token-embedding source, so it can't be gated away before it is useful.
                    hidden = (hidden + self.window_norm(projected)).astype(hidden.dtype)
                elif cfg.window_embed_mode == "source":
                    extra_sources = (*extra_sources, self.window_norm(projected))
                else:
                    raise ValueError(f"window_embed_mode must be source or add_embed, got {cfg.window_embed_mode!r}")
            if self.token_embed3 is not None:
                assert self.embed3_norm is not None and cfg.second_embed_bigram and cfg.second_embed_mode == "source"
                doc_start = None if segment_ids is None else segment_ids[0]
                ids3 = _bigram_hash_ids(token_ids, doc_start, cfg.embed3_rows, 3)
                table3 = reshard(self.token_embed3, P(None, None)) if cfg.embed2_fsdp else self.token_embed3
                gather3 = _embedding_gather if cfg.embed2_grad_fp32 else _embedding_gather_autodiff
                embed3 = self.embed3_norm(gather3(table3, ids3))
                if self.trigram_gate_w is not None and self.trigram_gate_b is not None:
                    embed3, gate3 = _content_gate(
                        hidden,
                        embed3,
                        self.trigram_gate_w,
                        self.trigram_gate_b,
                        self.trigram_gate_a_lr,
                        self.trigram_gate_b_lr,
                    )
                    bigram_gate_stats["attn_res_trigram_gate_mean"] = jax.lax.stop_gradient(jnp.mean(gate3))
                    bigram_gate_stats["attn_res_trigram_gate_std"] = jax.lax.stop_gradient(jnp.std(gate3))
                extra_sources = (*extra_sources, embed3)
            if self.ngram_stat_table is not None and cfg.ngram_stat_mode is NgramStatMode.SOURCE:
                doc_start = None if segment_ids is None else segment_ids[0]
                extra_sources = (*extra_sources, self._ngram_stat_source(hidden, token_ids, doc_start))
            ple_rows = None
            if self.token_embed_ple is not None:
                gather_ple = _embedding_gather if cfg.embed2_grad_fp32 else _embedding_gather_autodiff
                table_ple = reshard(self.token_embed_ple, P(None, None)) if cfg.embed2_fsdp else self.token_embed_ple
                ple_rows = gather_ple(table_ple, token_ids)
            hidden, stacked_router_stats, final_gate_stats = self._attn_res_layers(
                hidden,
                token_ids,
                extra_sources,
                loop_active,
                long_mask.with_fa4_bounds(long_lower_bounds, valid),
                long_mask.with_fa4_bounds(short_lower_bounds, valid),
                route_key,
                input_embed2,
                ple_rows,
            )
        else:
            if cfg.nitp_weight > 0:
                raise ValueError("nitp_weight needs attn_res (the NITP target is read from the AttnRes sources)")
            if cfg.moe_hash_layers or cfg.moe_gumbel_tau > 0:
                raise ValueError(
                    "moe_hash_layers / moe_gumbel_tau need attn_res (the scanned stack has no router extras)"
                )
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
                num_experts=cfg.num_experts + cfg.num_null_experts,
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
            if return_routing:
                if cfg.loop_passes != 1:
                    raise ValueError("return_routing needs loop_passes=1 (passes merge the per-layer stats)")
                router_metrics[ROUTING_SELECTED_KEY] = stacked_router_stats[_ROUTING_SELECTED]
                router_metrics[ROUTING_WEIGHTS_KEY] = stacked_router_stats[_ROUTING_WEIGHTS]
            if cfg.newton_muon:
                gram_sum = jnp.sum(stacked_router_stats[NEWTON_GRAM_LOCAL_KEY], axis=1)
                router_metrics[NEWTON_GRAM_KEY] = reshard(gram_sum / (batch_size * seq_len), P(None, None, None))
        router_metrics.update(final_gate_stats)
        router_metrics.update(bigram_gate_stats)
        hidden = self.final_norm(hidden)
        if self.final_gated_norm is not None:
            hidden = self.final_gated_norm(hidden)
        if self.mtp is not None:
            router_metrics[_MTP_EMBED] = raw_embed
        return hidden, router_metrics

    def _attn_res_layers(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        extra_sources: tuple[jax.Array, ...],
        loop_active: bool | None,
        long_layer_mask: AttentionMask,
        short_layer_mask: AttentionMask,
        route_key: jax.Array | None = None,
        input_embed2: jax.Array | None = None,
        ple_rows: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], dict[str, jax.Array]]:
        """Block AttnRes over the layers, unrolled so each gate reads only its valid sources.

        The residual history is a Python tuple of completed block sums (the token embedding is the
        first, rolled in at layer 0) plus the running partial; every ``seg_size``-th layer rolls its
        incoming partial into a new block. Completed blocks are immutable, so each block's logits
        against the queries of every gate that reads it (layer ``i``'s onwards and the final gate) are
        computed once when it is rolled, and each gate then only scores the partial. Blocks are shared
        by reference, so block memory is one copy per block. With ``AttnResLayerBackward.RECOMPUTE``
        each layer is rematerialized by ``_attn_res_layer_remat``, which saves only its inputs.

        ``ple_rows`` (``ple_dim``) holds every layer's per-layer-embedding rows side by side; layer ``i``
        gets its RMS-normed slice through its ``extras["ple"]``.

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
        # V-gate queries sit between the per-layer gates and the final gate, so every block's precomputed
        # logits (scored against queries[2i:]) cover them; V gate of layer i is row 2 * L * passes + i.
        v_queries = []
        if cfg.attn_res_v_gate:
            if passes != 1:
                raise ValueError("attn_res_v_gate supports loop_passes=1 only")
            v_queries = [jnp.stack([layer.attn_res_query_v for layer in layers])]
        queries_flat = jnp.concatenate(
            [jnp.stack(gate_queries), *loop_queries, *v_queries, self.attn_res_query_final[None]]
        )
        queries = queries_flat
        if cfg.attn_res_heads > 1:
            if cfg.hidden_dim % cfg.attn_res_heads:
                raise ValueError("attn_res_heads must divide hidden_dim")
            if cfg.attn_res_pull or cfg.attn_res_pull_embed:
                raise ValueError("attn_res_heads > 1 is not implemented for pull AttnRes")
            if cfg.attn_res_head_sub == "none":
                queries = queries_flat.reshape(queries_flat.shape[0], cfg.attn_res_heads, -1)
            else:
                assert self.attn_res_query_sub is not None
                queries = queries_flat[:, None, :] + _full_width_sub_queries(self.attn_res_query_sub, cfg)
        logit_bias = _gate_extras(self, queries.shape[0])
        if self.router_tok_a is not None:
            # router_token_bias_rank: every MoE layer reads the tokens' rows of the shared table.
            logit_bias = {**(logit_bias or {}), "router_tok": _embedding_gather(self.router_tok_a, token_ids)}
        if input_embed2 is not None:
            assert self.embed2_lambda is not None
            logit_bias = {**(logit_bias or {}), "embed2": input_embed2, "embed2_lambda": self.embed2_lambda}
        if cfg.attn_res_head_sub not in ("none", "slice", "full"):
            raise ValueError(f"attn_res_head_sub must be none, slice or full, got {cfg.attn_res_head_sub!r}")
        if cfg.attn_res_head_sub != "none" and (cfg.attn_res_heads < 2 or cfg.attn_res_head_norm):
            raise ValueError("attn_res_head_sub needs attn_res_heads > 1 and no attn_res_head_norm")
        if cfg.attn_res_source_delta and not cfg.attn_res_full:
            raise ValueError("attn_res_source_delta needs attn_res_full")
        if cfg.attn_res_final_mode not in ("attn", "uniform", "sum"):
            raise ValueError(f"attn_res_final_mode must be attn, uniform or sum, got {cfg.attn_res_final_mode!r}")
        if cfg.attn_res_blend not in ("none", "static", "dynamic"):
            raise ValueError(f"attn_res_blend must be none, static or dynamic, got {cfg.attn_res_blend!r}")
        if (cfg.attn_res_dual_query or cfg.attn_res_blend != "none") and (
            cfg.attn_res_v_gate or cfg.attn_res_heads > 1 or cfg.attn_res_pull or cfg.attn_res_pull_embed
        ):
            raise ValueError("attn_res_dual_query / attn_res_blend need single-head push AttnRes without V gates")
        if cfg.attn_res_stream_source and not cfg.attn_res_full:
            raise ValueError("attn_res_stream_source needs attn_res_full (the last column is logged as the stream)")
        if cfg.attn_res_stream_source and logit_bias is not None:
            raise ValueError("attn_res_stream_source does not combine with per-source logit extras")
        if cfg.attn_res_sum_inputs and not cfg.attn_res_full:
            raise ValueError("attn_res_sum_inputs needs attn_res_full")
        allowed = {"q", "k", "v", "mlp", "mlp_shared", "mlp_routed", "mlp_router"}
        if set(cfg.attn_res_sum_inputs) - allowed:
            raise ValueError(f"attn_res_sum_inputs must be a subset of {sorted(allowed)}, got {cfg.attn_res_sum_inputs}")
        if cfg.attn_res_full and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("attn_res_full needs attn_res_layer_backward=SAVE")
        if cfg.mla_share_kv_latent and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("mla_share_kv_latent needs attn_res_layer_backward=SAVE (the latent crosses layers)")
        if cfg.value_residual_layers and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("value_residual_layers needs attn_res_layer_backward=SAVE (the values cross layers)")
        if not set(cfg.value_residual_layers) <= set(range(1, num_layers)):
            raise ValueError(f"value_residual_layers must be in 1..{num_layers - 1}, got {cfg.value_residual_layers}")
        if cfg.attn_res_z_loss > 0 and cfg.attn_res_layer_backward != AttnResLayerBackward.SAVE:
            raise ValueError("attn_res_z_loss needs attn_res_layer_backward=SAVE (the remat VJP drops stat cotangents)")
        layer_fn = (
            _attn_res_layer_remat
            if cfg.attn_res_layer_backward == AttnResLayerBackward.RECOMPUTE
            else _attn_res_layer_passthrough
        )

        weight_logs: dict[int, tuple[jax.Array, bool]] = {}
        layer_logs: dict[str, jax.Array] = {}
        kv_inputs = None
        if self.kv_stream is not None:
            gather = _embedding_gather if cfg.embed_grad_fp32 else _embedding_gather_autodiff
            kv_inputs, kv_stream_logs = self.kv_stream(token_ids, long_layer_mask, gather)
            layer_logs.update(kv_stream_logs)
        nitp_logs: dict[str, jax.Array] = {}
        if cfg.nitp_weight > 0 and not 0 <= cfg.nitp_layer < num_layers:
            raise ValueError(f"nitp_layer must be in 0..{num_layers - 1}, got {cfg.nitp_layer}")

        def run_pass(state, pass_index):
            """One pass over the physical layers, extending the history; returns the new state, the
            per-layer router stats and the pass's gate z terms."""
            blocks, block_logits, partial = state
            stats_out, z_out = [], []
            kv_share: dict[str, jax.Array] | None = {} if cfg.mla_share_kv_latent or cfg.value_residual_layers else None
            for i, layer in enumerate(layers):
                eff = pass_index * num_layers + i
                if cfg.attn_res_full or (eff % seg_size == 0 and eff // seg_size < block_cap):
                    assert partial is not None
                    blocks = (*blocks, partial)
                    # Score the new block only against the gates that can read it (this layer's onwards).
                    block_logits = (
                        *block_logits,
                        _attn_res_source_logits(partial, queries[2 * eff :], eps, cfg.attn_res_head_norm),
                    )
                    partial = None
                if pass_index > 0 and i == 0:
                    assert self.loop_inject_scale is not None
                    # Input injection: the extra pass's running partial starts from the embedding.
                    scale = self.loop_inject_scale[pass_index - 1]
                    inject = (scale * embedding.astype(jnp.float32)).astype(embedding.dtype)
                    partial = inject if partial is None else partial + inject
                use_long = _is_long_layer(i, num_layers, cfg.global_every, cfg.global_layers)
                partial_before = partial
                layer_extras = logit_bias
                if ple_rows is not None:
                    ple_i = rms_norm(ple_rows[..., i * cfg.ple_dim : (i + 1) * cfg.ple_dim], eps)
                    layer_extras = {**(logit_bias or {}), "ple": ple_i}
                if kv_inputs is not None:
                    layer_extras = {**(layer_extras or {}), "kv_stream": kv_inputs[i]}
                if i in cfg.memory_layers:
                    assert self.memory is not None
                    layer_extras = {**(layer_extras or {}), "memory": self.memory[cfg.memory_layers.index(i)]}
                layer_args = (
                    (layer, blocks, block_logits, partial, queries, layer_extras),
                    long_layer_mask if use_long else short_layer_mask,
                    token_ids,
                    use_long,
                    eff,
                    eps,
                    None if route_key is None else jax.random.fold_in(route_key, eff),
                )
                if cfg.attn_res_full:
                    partial, blocks, block_logits, stats = _attn_res_layer_full(*layer_args, kv_share)
                elif kv_share is None:
                    partial, blocks, block_logits, stats = layer_fn(*layer_args)
                else:
                    partial, blocks, block_logits, stats = _attn_res_layer_passthrough(*layer_args, kv_share)
                z_out.append(stats.pop(_ATTN_RES_Z))
                if cfg.nitp_weight > 0 and pass_index == 0 and i == cfg.nitp_layer:
                    # The plain residual stream after this layer: embedding plus every sublayer output so far.
                    nitp_logs[_NITP_TARGET] = _stream_sum((*blocks[len(extra_sources) :], partial))
                for name in [k for k in stats if k.startswith(_LAYER_KNOB_PREFIX)]:
                    layer_logs[f"{name}_L{eff}"] = stats.pop(name)
                has_partial_attn = partial_before is not None
                stream_col = cfg.attn_res_stream_source
                weight_logs[2 * eff] = (stats.pop(_ATTN_RES_W_ATTN), has_partial_attn or stream_col)
                has_partial_mlp = has_partial_attn if cfg.moe_shortcut else not cfg.attn_res_full
                weight_logs[2 * eff + 1] = (stats.pop(_ATTN_RES_W_MLP), has_partial_mlp or stream_col)
                if _ATTN_RES_W_V in stats:
                    # V-gate query rows follow the per-layer gates (loop_passes == 1).
                    weight_logs[2 * num_layers + i] = (stats.pop(_ATTN_RES_W_V), has_partial_attn)
                stats_out.append(stats)
            return (blocks, block_logits, partial), stats_out, z_out

        def final_gate(state):
            blocks, block_logits, partial = state
            final_index = queries.shape[0] - 1
            logits = [_block_logit(bl, queries, final_index) for bl in block_logits]
            logits.append(_attn_res_source_logits(partial, queries[final_index][None], eps, cfg.attn_res_head_norm)[0])
            logits = _bias_gate_logits(logits, logit_bias, final_index, [*blocks, partial], eps, has_partial=True)
            if cfg.attn_res_final_mode == "uniform":
                logits = [jnp.zeros_like(logit) for logit in logits]
            logits = _soft_cap_logits(logits, cfg.attn_res_logit_soft_cap)
            weights, mixed = _softmax_mix(logits, [*blocks, partial])
            if cfg.attn_res_final_mode == "attn":
                mixed = _gate_variants(mixed, [*blocks, partial], logit_bias, final_index, eps, has_partial=True)
            if cfg.attn_res_final_mode == "sum":
                mixed = _stream_sum((*blocks, partial)).astype(jnp.float32)
            elif cfg.attn_res_additive:
                mixed = mixed + _stream_sum((*blocks, partial)).astype(jnp.float32)
            if self.attn_res_query_backout is not None:
                srcs = [*blocks, partial]
                cols = [*range(len(blocks)), -1]
                for src, c in zip(srcs, cols, strict=True):
                    mixed = mixed + self.attn_res_query_backout[c] * src.astype(jnp.float32)
            weight_logs[final_index] = (
                jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim)))),
                True,
            )
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

        state = (
            extra_sources,
            tuple(_attn_res_source_logits(src, queries, eps, cfg.attn_res_head_norm) for src in extra_sources),
            hidden,
        )
        state, pass0_stats, pass0_z = run_pass(state, 0)
        aux_hidden = None
        if cfg.aux_lm_layer is not None:
            raise ValueError("aux_lm_layer is not supported by this AttnRes loop")
        if cfg.byte_aux_mid_layer:
            # Mid-network stream for the byte head: the embedding plus the first half of the completed blocks.
            stream = state[0][len(extra_sources) :]
            aux_hidden = functools.reduce(jnp.add, stream[: max(1, len(stream) // 2)])

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
                qk_mult = jax.lax.stop_gradient(layer.attn.qk_mult)
                final_stats[f"attn_res_qk_mult_L{i}"] = jnp.mean(qk_mult)
                if qk_mult.ndim:
                    final_stats[f"attn_res_qk_mult_min_L{i}"] = jnp.min(qk_mult)
                    final_stats[f"attn_res_qk_mult_max_L{i}"] = jnp.max(qk_mult)
            if layer.attn_out_scale is not None and layer.mlp_out_scale is not None:
                final_stats[f"attn_res_scale_attn_L{i}"] = jax.lax.stop_gradient(layer.attn_out_scale)
                final_stats[f"attn_res_scale_mlp_L{i}"] = jax.lax.stop_gradient(layer.mlp_out_scale)
            final_stats.update(_learned_knob_stats(layer, i))
        final_stats.update(layer_logs)
        if self.router_tok_a is not None:
            final_stats["attn_res_knob_router_tok_a_rms"] = jnp.sqrt(
                jnp.mean(jnp.square(jax.lax.stop_gradient(self.router_tok_a).astype(jnp.float32)))
            )
        if self.router_tie_alpha is not None:
            alpha = jax.lax.stop_gradient(self.router_tie_alpha).astype(jnp.float32)
            directions = jax.lax.stop_gradient(_router_tie_directions(self)).astype(jnp.float32)
            routers = _routers_by_layer(self)
            for j, tie in enumerate(_router_ties(cfg)):
                final_stats[f"attn_res_knob_router_tie_alpha_{tie.tag}"] = alpha[j]
                # 1 while tied; after the release, how far the freed column has turned from the tie direction.
                column = jax.lax.stop_gradient(routers[tie.layer][:, tie.expert]).astype(jnp.float32)
                final_stats[f"attn_res_knob_router_tie_cos_{tie.tag}"] = jnp.dot(column, directions[j]) / (
                    jnp.linalg.norm(column) * jnp.linalg.norm(directions[j])
                )
        if cfg.attn_res_key_rank is not None:
            final_stats["attn_res_knob_lrkey_qnorm_final"] = jnp.linalg.norm(
                jax.lax.stop_gradient(self.attn_res_query_final)
            )
        if self.lm_head_bias is not None:
            bias = jax.lax.stop_gradient(self.lm_head_bias).astype(jnp.float32)
            final_stats["attn_res_knob_lm_head_bias_mean"] = jnp.mean(bias)
            final_stats["attn_res_knob_lm_head_bias_std"] = jnp.std(bias)
            final_stats["attn_res_knob_lm_head_bias_max"] = jnp.max(bias)
        for name, norm in (("embed", self.embed_norm), ("final", self.final_norm)):
            if isinstance(norm, ZeroCenteredRMSNorm):
                final_stats[f"attn_res_knob_gain_abs_{name}"] = jnp.mean(jnp.abs(jax.lax.stop_gradient(norm.gamma)))
        for i, memory in zip(cfg.memory_layers, self.memory or (), strict=True):
            w_out = jax.lax.stop_gradient(memory.w_out).astype(jnp.float32)
            final_stats[f"{_MEMORY_STAT_PREFIX}out_norm_L{i}"] = jnp.linalg.norm(w_out)
        hidden = reshard(mixed.astype(hidden.dtype), _batch_spec())
        if aux_hidden is not None:
            final_stats[_AUX_HIDDEN] = aux_hidden
        final_stats.update(nitp_logs)
        final_stats["attn_res_z"] = jax.lax.stop_gradient(z_total)
        if cfg.attn_res_z_loss > 0:
            final_stats[_ATTN_RES_Z] = z_total
        query_norms = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.square(queries_flat.astype(jnp.float32)), axis=-1)))
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

    def routing_assignments(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Int[Array, "L B S K"], Float[Array, "L B S K"]]:
        """Every MoE layer's selected experts and their combine weights per token (a debug forward; the
        training path never materializes them). Null-expert slots keep their ids ``>= num_experts``."""
        if self.config.dense_mlp:
            raise ValueError("routing_assignments needs MoE layers")
        _, metrics = self(token_ids, mask=mask, return_routing=True)
        b, s = token_ids.shape
        selected, weights = metrics[ROUTING_SELECTED_KEY], metrics[ROUTING_WEIGHTS_KEY]
        spec = P(None, _BATCH_AXES, None, None)
        return (
            jax.lax.reshape(selected, (selected.shape[0], b, s, selected.shape[-1]), out_sharding=spec),
            jax.lax.reshape(weights, (weights.shape[0], b, s, weights.shape[-1]), out_sharding=spec),
        )

    def layer_kinds(self) -> tuple[str, ...]:
        """Per layer, its mixer and context, e.g. ``kda_local`` or ``mla_global``."""
        cfg = self.config
        return tuple(
            ("kda" if isinstance(layer.attn, KimiDeltaAttention) else "mla" if cfg.mla else "gqa")
            + ("_global" if _is_long_layer(i, cfg.num_layers, cfg.global_every, cfg.global_layers) else "_local")
            for i, layer in enumerate(self.layers())
        )

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        hidden, lm_head = self._lm_head_operands(hidden, token_ids)
        return jnp.einsum("bsh,hd->bsd", hidden, lm_head, out_sharding=batch_spec)

    def _output_bigram_features(self, token_ids: Int[Array, "B S"], dtype: jnp.dtype) -> Float[Array, "B S R"]:
        assert self.output_bigram_u is not None
        return rms_norm(_embedding_gather(self.output_bigram_u, token_ids)).astype(dtype)

    def _lm_head_operands(
        self, hidden: Float[Array, "... D"], bigram_ids: Int[Array, "..."] | None
    ) -> tuple[Float[Array, "... E"], Float[Array, "E V"]]:
        """The lm_head's ``(input, weight)`` for the fused CE kernel, extended along the contraction:

        - with ``output_bigram_rank`` and ``bigram_ids`` (the current tokens), the bigram prior's features and
          read-out (``[h, rms(U[x])] @ [[W_out], [W]]``); ``bigram_ids=None`` leaves the prior out;
        - with ``lm_head_bias``, extra rows read by constant-1 hidden columns, so the bias lands inside the
          soft-cap. The float32 bias is split into a compute-dtype high part and its residual (two rows), padded
          to ``_LM_HEAD_BIAS_COLS`` columns.
        """
        use_bigram = self.output_bigram_w is not None and bigram_ids is not None
        if not use_bigram and self.lm_head_bias is None:
            return hidden, self.output_proj
        dtype = self.output_proj.dtype
        # The CE kernel replicates the head anyway; gathering it here keeps the concatenation's operands alike.
        head_rows = [reshard(self.output_proj, P(None, None))]
        hidden_cols = [hidden]
        if use_bigram:
            assert self.output_bigram_w is not None and bigram_ids is not None
            hidden_cols.append(self._output_bigram_features(bigram_ids, hidden.dtype))
            head_rows.append(reshard(self.output_bigram_w.astype(dtype), P(None, None)))
        if self.lm_head_bias is not None:
            high = self.lm_head_bias.astype(dtype)
            low = (self.lm_head_bias - high.astype(jnp.float32)).astype(dtype)
            pad = _LM_HEAD_BIAS_COLS - 2
            rows = jnp.concatenate([high[None], low[None], jnp.zeros((pad, high.shape[0]), dtype)], axis=0)
            head_rows.append(reshard(rows, P(None, None)))
            # Sliced from ``hidden`` so the constant columns inherit its batch sharding.
            hidden_cols.extend([jnp.ones_like(hidden[..., :2]), jnp.zeros_like(hidden[..., :pad])])
        hidden = jnp.concatenate(hidden_cols, axis=-1)
        if use_bigram:
            hidden = reshard(hidden, _batch_spec())
        return hidden, jnp.concatenate(head_rows, axis=0)

    def _output_bigram_stats(self) -> dict[str, jax.Array]:
        """Norms of ``U`` and ``W`` and the RMS of the prior's logit term over a fixed, evenly spaced set of probe
        token ids (a slice of the batch-sharded tokens can't be taken under explicit sharding)."""
        assert self.output_bigram_u is not None and self.output_bigram_w is not None
        u = jax.lax.stop_gradient(self.output_bigram_u).astype(jnp.float32)
        w = jax.lax.stop_gradient(self.output_bigram_w).astype(jnp.float32)
        probe = jnp.arange(_OUTPUT_BIGRAM_STAT_TOKENS) * (u.shape[0] // _OUTPUT_BIGRAM_STAT_TOKENS)
        sample = rms_norm(u[probe])
        return {
            "train/attn_res/knob_output_bigram_u_norm": jnp.linalg.norm(u),
            "train/attn_res/knob_output_bigram_w_norm": jnp.linalg.norm(w),
            "train/attn_res/knob_output_bigram_logit_rms": jnp.sqrt(jnp.mean(jnp.square(sample @ w))),
        }

    def _ngram_stat_read(
        self, hidden: Float[Array, "B S D"], token_ids: Int[Array, "B S"], segment_ids: Int[Array, "B S"] | None
    ) -> Float[Array, "B S D"]:
        """Gather each position's row of every order (no gradient into the table) and read ``[mean code, log count]``
        of all orders through the reader to ``hidden_dim``."""
        cfg = self.config
        assert self.ngram_stat_table is not None and self.ngram_stat_up is not None
        ids = _ngram_stat_ids(cfg, token_ids, segment_ids)
        b, s, k = ids.shape
        flat_ids = jax.lax.reshape(ids, (b, s * k), out_sharding=P(_BATCH_AXES, None))
        rows = _embedding_gather_autodiff(jax.lax.stop_gradient(self.ngram_stat_table), flat_ids).astype(jnp.float32)
        features = _ngram_stat_features(rows)
        features = jax.lax.reshape(
            features, (b, s, k * features.shape[-1]), out_sharding=P(_BATCH_AXES, None, None)
        ).astype(hidden.dtype)
        if self.ngram_stat_hidden is not None:
            features = jax.nn.gelu(
                jnp.einsum(
                    "bsf,fm->bsm", features, self.ngram_stat_hidden.astype(hidden.dtype), out_sharding=_batch_spec()
                )
            )
        return jnp.einsum("bsf,fd->bsd", features, self.ngram_stat_up.astype(hidden.dtype), out_sharding=_batch_spec())

    def _ngram_stat_source(
        self, hidden: Float[Array, "B S D"], token_ids: Int[Array, "B S"], segment_ids: Int[Array, "B S"] | None
    ) -> Float[Array, "B S D"]:
        """``NgramStatMode.SOURCE``: the reader's output, RMS-normed and content-gated, as its own AttnRes source."""
        assert self.ngram_stat_norm is not None
        source = self.ngram_stat_norm(self._ngram_stat_read(hidden, token_ids, segment_ids))
        if self.ngram_stat_gate_w is not None and self.ngram_stat_gate_b is not None:
            source, _ = _content_gate(hidden, source, self.ngram_stat_gate_w, self.ngram_stat_gate_b, None, None)
        return source

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
        route_key: jax.Array | None = None,
        head_replay: "HeadReplay | None" = None,
        byte_table: Int[Array, "V N"] | None = None,
        byte_aux_weight: jax.Array | None = None,
        router_tie_active: bool | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        """``aux_loss_weight`` scales the early auxiliary LM loss (``aux_lm_layer``); it is skipped at 0.
        ``train_terms`` adds the training-only objectives (MTP, AttnRes z-loss); evals leave it off so they
        score the plain next-token loss."""
        hidden, router_metrics = self(
            token_ids, mask=mask, loop_active=loop_active, route_key=route_key, router_tie_active=router_tie_active
        )
        aux_hidden = router_metrics.pop(_AUX_HIDDEN, None)
        nitp_target = router_metrics.pop(_NITP_TARGET, None)
        attn_res_z = router_metrics.pop(_ATTN_RES_Z, None)
        mtp_embed = router_metrics.pop(_MTP_EMBED, None)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)
        if head_replay is not None and self.output_bigram_w is not None:
            raise ValueError("head_replay stores the final hidden only; it cannot replay output_bigram_rank's prior")

        def lm_loss(h: jax.Array) -> jax.Array:
            head_in, lm_head = self._lm_head_operands(h, token_ids)
            return fused_linear_softmax_cross_entropy_loss(
                head_in,
                lm_head,
                labels,
                weight=loss_weight,
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_CE_BLOCK_SIZES,
                logit_soft_cap=_logit_cap(self.config),
            )

        cross_entropy_loss = lm_loss(hidden)
        replay_loss = None
        if head_replay is not None:
            # Replay a stored (final hidden, label) batch through the lm_head only: the head's gradient is
            # exact at the current W_head; the stored hidden is a stale view of that data (stop-gradient).
            replay_hidden, replay_head = self._lm_head_operands(
                jax.lax.stop_gradient(head_replay.hidden).astype(hidden.dtype), None
            )
            replay_loss = fused_linear_softmax_cross_entropy_loss(
                replay_hidden,
                replay_head,
                head_replay.labels,
                weight=head_replay.weight.astype(loss_dtype),
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_CE_BLOCK_SIZES,
                logit_soft_cap=_logit_cap(self.config),
            )
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
        if self.mtp is not None and mtp_embed is not None and train_terms:
            mtp_key = None if route_key is None else jax.random.fold_in(route_key, _MTP_KEY_SALT)
            mtp_hidden, mtp_labels, mtp_weight = _mtp_inputs(
                self.config,
                self.mtp,
                hidden,
                mtp_embed,
                token_ids,
                loss_weight,
                _sconv_segment_ids(mask),
                mtp_key,
            )
            mtp_hidden, mtp_head = self._lm_head_operands(mtp_hidden, None)
            mtp_loss = fused_linear_softmax_cross_entropy_loss(
                mtp_hidden,
                mtp_head,
                mtp_labels,
                weight=mtp_weight,
                reduction=reduction,
                # The final-logit z-loss regularizes the shared lm_head once, through the main head.
                logsumexp_weight=None,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_CE_BLOCK_SIZES,
                logit_soft_cap=_logit_cap(self.config),
            )
            loss = loss + self.config.mtp_weight * mtp_loss
        nitp_loss = nitp_cos = None
        if self.nitp_w1 is not None and self.nitp_w2 is not None and nitp_target is not None and train_terms:
            nitp_loss, nitp_cos = _nitp_loss(
                hidden, nitp_target, self.nitp_w1, self.nitp_w2, loss_weight, _sconv_segment_ids(mask)
            )
            loss = loss + self.config.nitp_weight * nitp_loss.astype(loss_dtype)
        if replay_loss is not None and head_replay is not None:
            loss = loss + head_replay.scale.astype(loss_dtype) * replay_loss
        byte_loss = None
        if self.byte_head is not None and byte_table is not None and byte_aux_weight is not None and train_terms:
            targets = shard_map(
                _local_gather,
                mesh=get_abstract_mesh(),
                in_specs=(P(None, None), P(_BATCH_AXES, None)),
                out_specs=P(_BATCH_AXES, None, None),
            )(byte_table, reshard(labels, P(_BATCH_AXES, None)))
            head = self.byte_head
            byte_in = hidden
            if self.config.byte_aux_mid_layer:
                if aux_hidden is None:
                    raise ValueError("byte_aux_mid_layer needs the AttnRes mid-network stream (attn_res=True)")
                byte_in = reshard(rms_norm(aux_hidden.astype(hidden.dtype)), _batch_spec())
            byte_loss = jax.lax.cond(
                byte_aux_weight > 0,
                lambda h: _byte_aux_loss(h, head, targets, loss_weight),
                lambda h: jnp.zeros((), jnp.float32),
                byte_in,
            )
            loss = loss + byte_aux_weight.astype(loss_dtype) * byte_loss.astype(loss_dtype)
        erc_loss, erc_ratios = None, {}
        if self.config.erc_loss_weight > 0 and train_terms:
            if route_key is None:
                raise ValueError("erc_loss_weight needs route_key (the per-step proxy-token noise)")
            erc_key = jax.random.fold_in(route_key, _ERC_KEY_SALT)
            erc_terms = []
            for i, layer in enumerate(self.layers()):
                if not isinstance(layer.mlp, MoEMLP):
                    continue
                layer_loss, erc_ratios[f"train/aux/erc_diag_offdiag_L{i}"] = layer.mlp.erc_loss(
                    jax.random.fold_in(erc_key, i)
                )
                erc_terms.append(layer_loss)
            erc_loss = functools.reduce(jnp.add, erc_terms)
            loss = loss + self.config.erc_loss_weight * erc_loss.astype(loss_dtype)
        simbal_loss = None
        if self.config.simbal_loss_weight > 0 and train_terms:
            simbal_loss = functools.reduce(
                jnp.add, [layer.mlp.simbal_loss() for layer in self.layers() if isinstance(layer.mlp, MoEMLP)]
            )
            loss = loss + self.config.simbal_loss_weight * simbal_loss.astype(loss_dtype)
        if return_router_metrics:
            final_gate_metrics = {
                f"train/attn_res/{name.removeprefix('attn_res_')}": router_metrics.pop(name)
                for name in list(router_metrics)
                if name.startswith("attn_res_")
            }
            if self.output_bigram_w is not None:
                final_gate_metrics.update(self._output_bigram_stats())
            if not router_metrics:
                # Dense model: no router to summarize.
                return loss, {"train/cross_entropy_loss": cross_entropy_loss, **final_gate_metrics}
            summarized_metrics = summarize_router_metrics(router_metrics)
            summarized_metrics.update(final_gate_metrics)
            if NEWTON_GRAM_KEY in router_metrics:
                summarized_metrics[NEWTON_GRAM_KEY] = router_metrics[NEWTON_GRAM_KEY]
            if self.config.num_null_experts:
                summarized_metrics.update(_null_slot_metrics(self.config, router_metrics["routing_counts_per_layer"]))
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            if replay_loss is not None:
                summarized_metrics["train/head_replay_loss"] = replay_loss
            if head_replay is not None:
                summarized_metrics[FINAL_HIDDEN_KEY] = jax.lax.stop_gradient(hidden)
            if aux_loss is not None:
                summarized_metrics["train/attn_res/aux_lm_loss"] = aux_loss
            if mtp_loss is not None and self.mtp is not None:
                summarized_metrics["train/aux/mtp_loss"] = mtp_loss
                summarized_metrics.update(_mtp_knob_stats(self.mtp))
            if byte_loss is not None:
                summarized_metrics["train/aux/byte_loss"] = byte_loss
            if simbal_loss is not None:
                summarized_metrics["train/aux/simbal_loss"] = simbal_loss
            if erc_loss is not None:
                summarized_metrics["train/aux/erc_loss"] = erc_loss
                summarized_metrics.update(erc_ratios)
            if nitp_loss is not None:
                summarized_metrics["train/aux/nitp_loss"] = nitp_loss
                summarized_metrics["train/aux/nitp_cos"] = nitp_cos
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


FINAL_HIDDEN_KEY = "_final_hidden"


def _mtp_knob_stats(head: MtpHead) -> dict[str, jax.Array]:
    """How the MTP projection splits between ``rms(h_t)`` and the next token's embedding (Frobenius norms of the
    two halves of ``W_proj``; MuonH fixes only their sum of squares), and the MTP output-norm gain."""
    w_proj = jax.lax.stop_gradient(head.w_proj).astype(jnp.float32)
    d = w_proj.shape[1]
    norm = head.out_norm
    gain = norm.weight if isinstance(norm, RMSNorm) else 1.0 + norm.gamma
    return {
        "train/attn_res/knob_mtp_proj_h_norm": jnp.linalg.norm(w_proj[:d]),
        "train/attn_res/knob_mtp_proj_e_norm": jnp.linalg.norm(w_proj[d:]),
        "train/attn_res/knob_mtp_out_gain": jnp.mean(jax.lax.stop_gradient(gain)),
    }


def _mtp_targets(
    token_ids: Int[Array, "B S"], loss_weight: Float[Array, "B S"], segment_ids: Int[Array, "B S"] | None
) -> tuple[Int[Array, "B S"], Float[Array, "B S"]]:
    """Depth-1 MTP targets: position t predicts token t+2. Its weight is the main loss's weight on token t+2
    (``loss_weight[t+1]``), zeroed at the last two positions and where token t+2 is in another packed document."""
    seq_len = token_ids.shape[1]
    labels = jnp.pad(token_ids[:, 2:], ((0, 0), (0, 2))).astype(jnp.int32)
    valid = jnp.broadcast_to(jnp.arange(seq_len) < seq_len - 2, loss_weight.shape)
    if segment_ids is not None:
        valid = valid & jnp.pad(segment_ids[:, 2:] == segment_ids[:, :-2], ((0, 0), (0, 2)))
    weight = jnp.pad(loss_weight[:, 1:], ((0, 0), (0, 1))) * valid.astype(loss_weight.dtype)
    return labels, weight


def _mtp_inputs(
    cfg: GrugModelConfig,
    head: MtpHead,
    hidden: Float[Array, "B S D"],
    embed: Float[Array, "B S D"],
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    segment_ids: Int[Array, "B S"] | None,
    key: jax.Array | None,
) -> tuple[Float[Array, "B K D"], Int[Array, "B K"], Float[Array, "B K"]]:
    """The MTP head's lm_head inputs, labels and weights on the ``mtp_position_frac`` position subset (K of S)."""
    labels, weight = _mtp_targets(token_ids, loss_weight, segment_ids)
    next_embed = jnp.pad(embed[:, 1:], ((0, 0), (0, 1), (0, 0)))
    seq_len = token_ids.shape[1]
    if not 0.0 < cfg.mtp_position_frac <= 1.0:
        raise ValueError(f"mtp_position_frac must be in (0, 1], got {cfg.mtp_position_frac}")
    if cfg.mtp_position_frac < 1.0:
        if key is None:
            raise ValueError("mtp_position_frac < 1 needs route_key (the per-step position subsample)")
        num_kept = max(1, round(seq_len * cfg.mtp_position_frac))
        positions = jax.random.permutation(key, seq_len)[:num_kept]
        hidden = _batch_reshard(jnp.take(hidden, positions, axis=1))
        next_embed = _batch_reshard(jnp.take(next_embed, positions, axis=1))
        labels = jnp.take(labels, positions, axis=1)
        weight = jnp.take(weight, positions, axis=1)
    return reshard(head(hidden, next_embed), _batch_spec()), labels, weight


def _nitp_loss(
    hidden: Float[Array, "B S D"],
    target: Float[Array, "B S D"],
    w1: Float[Array, "D D"],
    w2: Float[Array, "D D"],
    loss_weight: Float[Array, "B S"],
    segment_ids: Int[Array, "B S"] | None,
) -> tuple[jax.Array, jax.Array]:
    """NITP (arXiv 2605.24956): ``1 - cos(P(h_t), sg(z_{t+1}))`` averaged over positions whose next token is
    trained and in the same document. Returns ``(loss, mean cos over the next-token pairs)``."""
    pred = jnp.einsum(
        "bsd,de->bse", jax.nn.gelu(jnp.einsum("bsd,de->bse", hidden, w1.astype(hidden.dtype))), w2.astype(hidden.dtype)
    )
    next_target = jax.lax.stop_gradient(jnp.pad(target[:, 1:], ((0, 0), (0, 1), (0, 0))))
    next_target = reshard(next_target, _batch_spec())
    pred32, next32 = pred.astype(jnp.float32), next_target.astype(jnp.float32)
    eps = 1e-6
    cos = jnp.sum(pred32 * next32, axis=-1) * jax.lax.rsqrt(
        jnp.maximum(jnp.sum(jnp.square(pred32), axis=-1) * jnp.sum(jnp.square(next32), axis=-1), eps)
    )
    seq_len = hidden.shape[1]
    valid = jnp.broadcast_to(jnp.arange(seq_len) < seq_len - 1, loss_weight.shape)
    if segment_ids is not None:
        valid = valid & jnp.pad(segment_ids[:, 1:] == segment_ids[:, :-1], ((0, 0), (0, 1)))
    weight = loss_weight.astype(jnp.float32) * valid.astype(jnp.float32)
    mean_cos = jnp.sum(cos * weight) / jnp.maximum(jnp.sum(weight), 1.0)
    return 1.0 - mean_cos, mean_cos


def _null_slot_metrics(cfg: GrugModelConfig, routing_counts: Float[Array, "L N"]) -> dict[str, jax.Array]:
    """Fraction of the top-K slots that went to the zero-computation experts: per layer, and per type."""
    counts = routing_counts.astype(jnp.float32)
    total = jnp.maximum(jnp.sum(counts, axis=-1), 1.0)
    bounds = list(
        itertools.accumulate((cfg.num_experts, cfg.moe_null_experts, cfg.moe_copy_experts, cfg.moe_const_experts))
    )
    null_frac = jnp.sum(counts[:, bounds[0] :], axis=-1) / total
    out = {"train/router/null_frac_mean": jnp.mean(null_frac)}
    for name, lo, hi in zip(("zero", "copy", "const"), bounds[:-1], bounds[1:], strict=True):
        if hi > lo:
            out[f"train/router/null_frac_{name}_mean"] = jnp.mean(jnp.sum(counts[:, lo:hi], axis=-1) / total)
    for i in range(counts.shape[0]):
        out[f"train/router/layer_{i}/null_frac"] = null_frac[i]
    return out


@dataclass(frozen=True)
class HeadReplay:
    """A stored batch of final hidden states and labels to replay through the lm_head (``scale`` x CE)."""

    hidden: jax.Array
    labels: jax.Array
    weight: jax.Array
    scale: jax.Array


jax.tree_util.register_dataclass(HeadReplay, data_fields=["hidden", "labels", "weight", "scale"], meta_fields=[])


def _stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)``: the layer indices of ``Transformer.stacked_blocks`` and ``kda_blocks``."""
    kda_layers = _kda_layer_indices(cfg)
    return tuple(i for i in range(cfg.num_layers) if i not in kda_layers), kda_layers


def upper_softmax_slice_mask(cfg: GrugModelConfig) -> tuple[bool, ...]:
    """Per ``Transformer.stacked_blocks`` slice, whether its layer is in the upper half (``>= num_layers // 2``)."""
    softmax_layers, _ = _stack_layer_indices(cfg)
    return tuple(i >= cfg.num_layers // 2 for i in softmax_layers)


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
        offset = _num_extra_embeds(cfg)
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
        "dual": model.attn_res_query_dual,
        "dual_sel": model.attn_res_query_dual_sel,
        "dual_sel_bias": model.attn_res_query_dual_sel_bias,
        "blend": model.attn_res_query_blend,
        "blend_proj": model.attn_res_query_blend_proj,
        "temperature": _temperature_rows(model),
        "dyn_w1": model.attn_res_query_dyn1,
        "dyn_w2": model.attn_res_query_dyn2,
    }
    return extras if any(v is not None for v in extras.values()) else None


def _full_width_sub_queries(sub: jax.Array, cfg: GrugModelConfig) -> jax.Array:
    """Per-head sub-query corrections as full-width [G, H, D] rows: a ``slice`` sub-query [G, H, D/H]
    lands in its own channel slice (zeros elsewhere); a ``full`` one is already [G, H, D]."""
    if cfg.attn_res_head_sub == "full":
        return sub
    heads = cfg.attn_res_heads
    placed = jnp.eye(heads, dtype=sub.dtype)[None, :, :, None] * sub[:, :, None, :]
    return placed.reshape(sub.shape[0], heads, heads * sub.shape[-1])


def _temperature_rows(model: "Transformer") -> jax.Array | None:
    """``attn_res_query_temp`` in the query stack's row order: per-layer gates, V gates, then the final
    gate (the parameter stores the final gate's row before the V gates). Single-head rows are [G, 1]."""
    temp = model.attn_res_query_temp
    if temp is None:
        return None
    cfg = model.config
    gates = _attn_res_num_gates(cfg)
    rows = jnp.concatenate([temp[: gates - 1], temp[gates:], temp[gates - 1 : gates]])
    return rows if cfg.attn_res_heads > 1 else rows[:, 0:1]


def _attn_res_num_gates(cfg: GrugModelConfig) -> int:
    """AttnRes gates without V gates: two per layer per pass, plus the final gate."""
    return 2 * cfg.num_layers * cfg.loop_passes + 1


def _attn_res_num_sources(cfg: GrugModelConfig) -> int:
    """Most sources any AttnRes gate reads: every completed block (extra embeddings included) + the partial."""
    seg_size = max(1, cfg.num_layers // cfg.attn_res_num_blocks)
    cap = cfg.attn_res_num_blocks * cfg.loop_passes
    if cfg.attn_res_full:
        return 2 * cfg.num_layers * cfg.loop_passes + _num_extra_embeds(cfg) + 1
    rolled = sum(1 for i in range(cfg.num_layers * cfg.loop_passes) if i % seg_size == 0 and i // seg_size < cap)
    return rolled + _num_extra_embeds(cfg) + 1


def _share_routers(model: "Transformer", block: int) -> "Transformer":
    """Replace each stack's per-layer router with its group leader's (``block`` consecutive entries per group)."""
    stacks = model.layer_stacks()
    shared = []
    for stack in stacks:
        router = stack.stacked.mlp.router
        if router is None:
            raise ValueError("router_share_block needs the full-rank router (router_rank=None)")
        leaders = (jnp.arange(router.shape[0]) // block) * block
        shared.append(reshard(jnp.take(router, leaders, axis=0), _partition_spec_of(router)))
    return eqx.tree_at(lambda t: [s.stacked.mlp.router for s in t.layer_stacks()], model, shared)


_BYTE_CLASSES = 256


def _byte_aux_loss(
    hidden: Float[Array, "B S D"], head: jax.Array, targets: Int[Array, "B S N"], loss_weight: Float[Array, "B S"]
) -> jax.Array:
    """Mean 256-way cross-entropy over the target token's first N bytes (``targets`` is -1 past its end)."""
    b, s, n = targets.shape
    logits = jnp.einsum("bsd,dk->bsk", hidden, head.astype(hidden.dtype), out_sharding=_batch_spec())
    logits = logits.reshape(b, s, n, _BYTE_CLASSES).astype(jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, jnp.maximum(targets, 0)[..., None], axis=-1)[..., 0]
    valid = (targets >= 0).astype(jnp.float32) * loss_weight[..., None].astype(jnp.float32)
    return jnp.sum((lse - picked) * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def _content_gate(
    hidden: Float[Array, "B S D"],
    source: Float[Array, "B S D"],
    w: jax.Array,
    bias: jax.Array,
    a_lr: jax.Array | None,
    b_lr: jax.Array | None,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """Engram-style content gate on an n-gram source: ``sigmoid(f(hidden * source) + bias)``, with ``f`` a rank-r
    projection to per-channel logits (``a_lr``, ``b_lr``) or a ``w`` dot to one logit per token. Both inputs are
    RMS-normed already, so it stays in the compute dtype."""
    dtype = source.dtype
    interaction = hidden.astype(dtype) * source
    if a_lr is not None and b_lr is not None:
        low = jnp.einsum("bsd,dr->bsr", interaction, a_lr.astype(dtype))
        logits = jnp.einsum("bsr,rd->bsd", low, b_lr.astype(dtype), out_sharding=_batch_spec())
        gate = jax.nn.sigmoid(logits + bias.astype(dtype))
        return source * gate, gate
    gate = jax.nn.sigmoid(jnp.einsum("bsd,d->bs", interaction, w.astype(dtype)) + bias.astype(dtype))
    return source * gate[..., None], gate


def _ngram_stat_source_mode(cfg: GrugModelConfig) -> bool:
    return cfg.ngram_stat_rows > 0 and cfg.ngram_stat_mode is NgramStatMode.SOURCE


def _ngram_stat_feature_dim(cfg: GrugModelConfig) -> int:
    return len(cfg.ngram_stat_orders) * (cfg.ngram_stat_dim + 1)


def _ngram_stat_ids(
    cfg: GrugModelConfig, token_ids: Int[Array, "B S"], segment_ids: Int[Array, "B S"] | None
) -> Int[Array, "B S K"]:
    """Each position's statistic-table row for every order k, offset into that order's block of rows."""
    rows = cfg.ngram_stat_rows
    return jnp.stack(
        [
            _bigram_hash_ids(token_ids, segment_ids, rows, order, salt=_NGRAM_STAT_SALT + k) + k * rows
            for k, order in enumerate(cfg.ngram_stat_orders)
        ],
        axis=-1,
    )


def _ngram_stat_features(rows: jax.Array) -> jax.Array:
    """``[sum, count]`` rows to the reader's input ``[sum / max(count, 1), log1p(count) / 4]``."""
    sums, counts = rows[..., :-1], rows[..., -1:]
    return jnp.concatenate([sums / jnp.maximum(counts, 1.0), jnp.log1p(counts) / 4.0], axis=-1)


def write_ngram_stats(
    model: "Transformer",
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    segment_ids: Int[Array, "B S"] | None,
) -> "Transformer":
    """Add a batch to the fixed-encoder statistic table: each position's n-gram row gains ``[code(next), 1]``
    times its loss weight (0 on the last position and on masked targets). Every device all-gathers the batch's
    row ids, next tokens and weights (a few MB) and scatter-adds the whole batch into its replicated table, so
    no table-sized collective runs. Called after the optimizer step, so a batch never reads its own targets."""
    assert model.ngram_stat_table is not None and model.ngram_stat_code is not None
    table = ngram_stat_table_add(
        model.config, model.ngram_stat_table, model.ngram_stat_code, token_ids, loss_weight, segment_ids
    )
    return eqx.tree_at(lambda m: m.ngram_stat_table, model, table)


def ngram_stat_table_add(
    cfg: GrugModelConfig,
    table: jax.Array,
    code: jax.Array,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    segment_ids: Int[Array, "B S"] | None,
) -> jax.Array:
    """``write_ngram_stats`` on the bare table (the pre-fill jits this alone, donating only the table)."""
    ids = _ngram_stat_ids(cfg, token_ids, segment_ids)
    next_ids = jnp.roll(token_ids, -1, axis=1)
    orders = len(cfg.ngram_stat_orders)

    def _local_write(table, code, ids, next_ids, weight):
        ids, next_ids, weight = (jax.lax.all_gather(x, _BATCH_AXES, tiled=True) for x in (ids, next_ids, weight))
        weight = weight.astype(jnp.float32).reshape(-1, 1)
        values = jnp.concatenate([code[next_ids.reshape(-1)] * weight, weight], axis=-1)
        # Every order's row of a position gets the same value.
        values = jnp.broadcast_to(values[:, None, :], (values.shape[0], orders, values.shape[1]))
        return table.at[ids.reshape(-1)].add(values.reshape(-1, values.shape[-1]))

    # The gathered batch is identical on every device, so the written table is replicated (check_vma can't see it).
    return jax.shard_map(
        _local_write,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(None, None), P(_BATCH_AXES, None, None), P(_BATCH_AXES, None), P(_BATCH_AXES, None)),
        out_specs=P(None, None),
        check_vma=False,
    )(
        table,
        code,
        reshard(ids, P(_BATCH_AXES, None, None)),
        reshard(next_ids, P(_BATCH_AXES, None)),
        reshard(loss_weight, P(_BATCH_AXES, None)),
    )


def _num_extra_embeds(cfg: GrugModelConfig) -> int:
    """Extra embedding tables ahead of the token embedding in the AttnRes source list."""
    window_source = cfg.window_embed_dim > 0 and cfg.window_embed_mode == "source"
    return int(cfg.second_embed) + int(cfg.embed3_rows > 0) + int(window_source) + int(_ngram_stat_source_mode(cfg))


def _token_window(
    emb: Float[Array, "B S N"], segment_ids: Int[Array, "B S"] | None, width: int
) -> Float[Array, "B S W"]:
    """Concatenate each position's last ``width // N`` token embeddings (slot j = token t - j), zeroing slots that
    reach before position 0 or into an earlier document."""
    n = emb.shape[-1]
    if width % n:
        raise ValueError(f"window_embed_dim={n} must divide hidden_dim={width}")
    slots = []
    for lag in range(width // n):
        shifted = jnp.pad(emb[:, : emb.shape[1] - lag], ((0, 0), (lag, 0), (0, 0))) if lag else emb
        if segment_ids is not None and lag:
            same_doc = jnp.pad(segment_ids[:, lag:] == segment_ids[:, :-lag], ((0, 0), (lag, 0)))
            shifted = jnp.where(same_doc[..., None], shifted, 0)
        slots.append(shifted)
    return reshard(jnp.concatenate(slots, axis=-1), _batch_spec())


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
    "ZeroCenteredRMSNorm",
    "debug_mesh_and_token_pspec",
    "moe_and_shared_fused",
]
