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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.nn import ArrayStacked
from jax import core, random
from jax.ad_checkpoint import checkpoint_name
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
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, rel_extent_of_band
from levanter.grug.grug_moe import (
    MoeActivation,
    MoEExpertMlp,
    MoeOverlapWork,
    OverlapParamGather,
    moe_mlp,
)
from levanter.grug.loss import BlockSizes, fused_linear_softmax_cross_entropy_loss
from levanter.grug.mxfp8 import ExpertMxfp8, mx_dense
from levanter.grug.sharding import unshard
from levanter.kernels.pallas.short_conv import short_conv
from levanter.tracker.histogram import SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track import attention_rows_probe, router_soft_select, trace_capture
from experiments.grug.fast_track.attn_res_block_mix import block_mix_local, block_mix_weight_grads_local
from experiments.grug.fast_track.attn_res_source_grads import accumulate_source_grads_local
from experiments.grug.fast_track.gating import _load_error_grad, mixture_weights
from experiments.grug.fast_track.key_offset import (
    apply_lag_groups,
    parse_lag_groups,
    partial_key_offset,
    shifted_channels,
)
from experiments.grug.fast_track.qb_balancing import qb_beta_hist
from experiments.grug.fast_track.router_metrics import (
    local_routing_stats,
    reduce_router_stats,
    summarize_router_metrics,
)
from experiments.grug.fast_track.sparse_attention import (
    mla_sparse_attention,
)
from experiments.grug.moe.kda import chunk_kda, doc_starts, kda_fused
from experiments.grug.moe.kda_prep_pallas import EraseInput, GateFn, GateKind, gate_log_decay
from experiments.grug.moe.kda_state_pallas import StateGrads

_GATED_NORM_RANK = 128
_QB_HIST_BINS = 10_000
_CE_TOKENS_PER_RANK = 65_536
# A vocab tile of 8192 in the fused lm_head + cross-entropy loop measured ~3% faster than 4096 at d512 on H100. On
# GB200 the kernel's own defaults win (44 vs 53 ms forward+backward at 65k tokens x 6144 x 16k vocab).
_CE_BLOCK_SIZES_H100 = BlockSizes(b_block_size=_CE_TOKENS_PER_RANK, v_block_size=8192)


def _ce_block_sizes() -> BlockSizes | None:
    """Fused CE block sizes for the current accelerator (None: the kernel's defaults)."""
    return None if "GB200" in jax.devices()[0].device_kind else _CE_BLOCK_SIZES_H100


# Axes the non-expert params FSDP-shard over.
_FSDP_AXES: tuple[str, ...] = ("data", "expert")
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")
# Extra hidden columns carrying ``lm_head_bias`` into the fused CE (two used, padded to a multiple of 8).
_LM_HEAD_BIAS_COLS = 8


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
# Batch axes a ``P(_FSDP_AXES, None)`` table is replicated over: their tokens all add to the same rows.
_TABLE_REPLICA_AXES = tuple(axis for axis in _BATCH_AXES if axis not in _FSDP_AXES)
# ``newton_muon``: the per-layer expert-input second moment ``Z^T Z / N`` ([L, n, n], layer order) the
# forward returns to the trainer, and the per-shard partial sums it is reduced from.
NEWTON_GRAM_KEY = "_newton_gram"
# Per-layer detached router distributions [B, S, E] for ``router_history``; popped by the layer loop, never summarized.
ROUTER_PROBS_KEY = "router_probs"
# Metrics-dict key that carries the auxiliary-loss residual stream from the forward to the loss.
# The ``skip_probe_layer`` probes' logits ``(f0, f1, g)``, [3, B, S].
_SKIP_PROBE_LOGITS = "skip_probe_logits"
# Per-layer logits of the ``final_conf_layers`` probes, [n, B, S].
_FINAL_CONF_LOGITS = "final_conf_logits"
# The stream entering ``mid_head_layer``, for the mid lm_head and its confidence head.
_MID_HIDDEN = "mid_head_hidden"
_NITP_TARGET = "nitp_target"
_LM_HEAD_EXTRA = "lm_head_extra"
# Metrics-dict key that carries the raw (pre-norm) token embeddings from the forward to the MTP loss.
_MTP_EMBED = "mtp_embed"
# Per-layer product-key memory diagnostics, lifted out of the layer stats into ``train/attn_res/knob_mem_*``.
# Per-layer stats with this prefix returned by a layer are exported as ``<name>_L<layer>``.
_LAYER_KNOB_PREFIX = "attn_res_knob_"
# Diagnostic forward probe (``forward_probe``). Every recorded value is a stat named with ``PROBE_STAT`` (per-layer ones
# end in ``_L<layer>``): MLA layers report each attention query's softmax probabilities over its ``ATTN_PROBE_KEYS``
# most recent keys; at the one probe ``spot`` every layer reports its AttnRes inputs and sublayer outputs, the final
# AttnRes its sources, gate and mix, and the head its input and the lm_head columns of ``spot_token_ids``.
PROBE_STAT = f"{_LAYER_KNOB_PREFIX}probe_"
ATTN_PROBE_STAT = f"{PROBE_STAT}attn"
ATTN_PROBE_KEYS = 2048


@dataclass(frozen=True)
class ForwardProbe:
    attn_queries: tuple[tuple[int, int], ...]
    spot: tuple[int, int] | None
    spot_token_ids: tuple[int, ...]


_FORWARD_PROBE: ForwardProbe | None = None
# ``attention_rows_probe``: traced (rows, positions) [Q] of the queries whose full softmax rows the MLA layers record.
ATTN_ROWS_STAT = f"{_LAYER_KNOB_PREFIX}probe_attn_rows"
# Under ``attention_rows_probe``: each query's Inkling bias by distance back, ``[Q, H, rel_extent]``.
RELPOS_ROWS_STAT = f"{_LAYER_KNOB_PREFIX}probe_relpos_rows"

# Per-token, per-query-head attention statistics (``head_probe``), ``[B, S, H, len(HEAD_PROBE_FIELDS)]`` per layer:
# softmax mass on the document's first token and on the query's own position, attention entropy (nats), the head's
# output gate, its gated output norm, and the norm of its contribution to the residual (through its ``w_o`` rows).
HEAD_PROBE_STAT = f"{_LAYER_KNOB_PREFIX}head_probe"
HEAD_PROBE_FIELDS = ("p_doc_start", "p_self", "entropy", "gate", "out_norm", "contrib_norm")
# Per layer, the token-mean cosine similarity ``[H, H]`` between query heads' (QK-normed, RoPE'd) query vectors, and
# its token-mean absolute value ``[H, H]`` (``head_probe``): under MQA every head's query lives in the one shared key
# space, and RoPE rotates all heads alike at a position, so the cosine compares what the heads ask.
HEAD_QCOS_STAT = f"{_LAYER_KNOB_PREFIX}head_qcos"
_HEAD_PROBE_QUERY_BLOCK = 512
_HEAD_PROBE = False
_MEMORY_STAT_PREFIX = f"{_LAYER_KNOB_PREFIX}mem_"
_KDA_ERASE_STAT_PREFIX = f"{_LAYER_KNOB_PREFIX}kda_erase_"
# Bound on one chunk's gathered ``[tokens, rows, dim]`` memory rows in the product-key EmbeddingBag.
_MEMORY_BAG_CHUNK_ELEMS = 1 << 26
# Bound on one chunk's FSDP-group-gathered ``[tokens, dim]`` rows in ``_row_sharded_embedding_gather``: 256 MiB
# in float32 (the backward's cotangent), so d1024's 2^30-element group batch walks in 16 chunks.
# 2**30 elements (2 GiB bf16) per chunk: at hero width a 2**26 bound split each lookup into 512 latency-bound
# all-gather / reduce-scatter pairs.
_ROW_GATHER_CHUNK_ELEMS = 1 << 30
# Metrics-dict keys carrying the AttnRes z-loss term (with gradient) from the forward to the loss.
_ATTN_RES_Z = "attn_res_z_term"
# Per-gate token-mean AttnRes source weights (variable length per gate), popped into logging scalars.
_ATTN_RES_W_ATTN = "attn_res_weights_attn"
_ATTN_RES_W_MLP = "attn_res_weights_mlp"
_ATTN_RES_W_V = "attn_res_weights_v"
# Per-layer extras key carrying ``LayerSkip.keep`` into a skipped layer.
_SKIP_KEEP = "layer_skip_keep"
# Per-layer extras key carrying the extra-MLP token mask (``extra_mlp_layers``), and the entropy head's prediction.
_EXTRA_KEEP = "extra_mlp_keep"
_ENTROPY_PRED = "entropy_pred"
_ENTROPY_CHUNK = 1024
# Per-layer router stats carrying each token's ``[T, K]`` selected experts and combine weights to
# ``Transformer.routing_assignments``.
_ROUTING_SELECTED = "routing_selected"
_ROUTING_WEIGHTS = "routing_weights"
ROUTING_SELECTED_KEY = "routing_selected_per_layer"
# Per-layer routed-expert inputs and shared-expert pre-activations for ``Transformer.neuron_inputs`` (the
# neuron-health dump); passthroughs, so the training forward drops them.
_ROUTED_INPUT = "routed_input"
# ``scan_save_routed_output``: a layer's combined routed-expert output ``[T, L]`` (a stat the scan collects and
# strips), and the extras key that hands the saved value back to the backward's recompute.
_ROUTED_COMBINE = "routed_combine"
_SAVED_ROUTED = "saved_routed"
# Per-layer moe_compress / moe_shadow_width loss and metrics, stacked over layers like the router stats.
_COMPRESS_LOSS = "compress_loss"
_DIFFERENTIABLE_STATS = (_COMPRESS_LOSS, _LM_HEAD_EXTRA)
_COMPRESS_PREFIX = "compress/"
# Per-layer branch-output max |x| and RMS (``layer_output_stats``), stacked over layers like the router stats.
_OUTSTAT_PREFIX = "outstat/"
_SHARED_PRE = "shared_pre"
ROUTED_INPUT_KEY = "routed_input_per_layer"
SHARED_PRE_KEY = "shared_pre_per_layer"
ROUTING_WEIGHTS_KEY = "routing_weights_per_layer"

# Kimi K3's KDA layer: low-rank forget-gate width, and the per-token log-decay floor
# ``g = -KDA_MIN_LOG_DECAY * sigmoid(...)``.
_KDA_GATE_RANK = 128
KDA_MIN_LOG_DECAY = 5.0
# 16-token chunks keep the kernels' intra-chunk rescaling exact for the -5 per-token log-decay
# floor (cumulative >= -80 = -DEFLATE_EXP_CAP; see kda_prep_pallas). The CPU (XLA) path always uses it.
KDA_CHUNK_SIZE = 16
# GPU chunk size (``kda_chunk_size``) -> decay tile of the fused kernels: longer chunks stay exact at the
# -5 floor with tiles referenced at their middle token, <= 32 tokens (``kda_fused``'s ``sub_chunk_size``).
KDA_CHUNK_TILES: dict[int, int | None] = {16: None, 32: 32, 64: 32}
# kda_dd_rope: per-pair base angular frequencies (rad/token) are log-spaced over this range, periods ~6 to
# ~800 tokens, spanning the KDA decay's memory lengths (|g| init in kda_dt_range).
_KDA_ROT_OMEGA_RANGE = (1.0 / 128, 1.0)


class SkipProbeTarget(StrEnum):
    """What the ``skip_probe_layer`` probes predict."""

    CORRECT = "correct"  # the final head's top token is the next token (BCE on a logit)
    LOSS = "loss"  # the final next-token loss (squared error)


class ExtraMlpSelect(StrEnum):
    """Which tokens ``extra_mlp_layers`` serve, by the entropy head's prediction (``extra_mlp_frac`` of the batch)."""

    HIGH = "high"  # the highest predicted entropy
    LOW = "low"  # the lowest
    RANDOM = "random"  # a fixed pseudo-random subset (the control)


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


class ContentGateActivation(StrEnum):
    """The nonlinearity between the low-rank content gate's two projections (``bigram_gate_rank``)."""

    NONE = "none"
    SILU = "silu"
    GELU = "gelu"


class UngatedExpertActivation(StrEnum):
    """The activation of the ungated (single ``W_up``) experts."""

    RELU2 = "relu2"
    """``leaky_relu(u, expert_leaky_slope)^2``."""
    POLYNORM = "polynorm"
    """PolyNorm (Motif 2.6B / Motif 3, arXiv 2608.09119): ``sum_{n=1..3} (1/3) u^n / RMS(u^n)``, each power
    RMS-normalized per token over the expert's hidden units. Fixed coefficients (Motif learns them per expert)."""


class SharedExpertPlacement(StrEnum):
    """Where the shared experts run relative to the routed experts (same math in every placement)."""

    FUSED = "fused"
    """Gate/up projections share the router/latent input GEMM."""
    UNFUSED = "unfused"
    """A separate GEMM, so the scheduler may (but need not) run the shared MLP under the EP transports."""
    OVERLAP = "overlap"
    """The whole shared MLP is pinned under the first EP dispatch all-to-all by optimization barriers
    (``MoeOverlapWork``), forward and backward."""


class MoeDispatch(StrEnum):
    """How the MoE block lays out its projections and the routed experts' EP transport (same math)."""

    FAST_TRACK = "fast_track"
    """fast_track's block: ``moe_shared_placement`` places the shared experts, and the router, latent down and
    (fused) shared gate/up projections run on replicated weights."""
    HERO_LEVANTER = "hero_levanter"
    """``moe_hero_ep``'s block: the router alone in high precision, the latent down and the shared experts as their
    own ``_proj`` GEMMs on the FSDP-sharded weights (under a real MXFP8 recipe ``mx_linear`` gathers each shard's
    MXFP8 bytes and reduce-scatters the gradient with the rest of the layer), the routed experts on levanter's
    plain ragged EP path with no pinned overlap work, and the shared experts after it, where the scheduler places
    them. Needs ``moe_shared_placement=fused`` (the placement it replaces)."""


class Fp8Recipe(StrEnum):
    """Low-precision recipe for the projection and expert GEMMs (embeddings, router, lm head, norms, attention
    core and the optimizer stay in their usual precision)."""

    NONE = "none"
    MXFP8_SIM = "mxfp8_sim"
    """Simulated MXFP8 (``levanter.grug.mxfp8``): e4m3 operands with a just-in-time power-of-two scale per 32
    elements along each GEMM's contraction axis, gradients included. Numerics only; GEMMs still run in bf16."""
    MXFP8_SIM_DENSE = "mxfp8_sim_dense"
    """``MXFP8_SIM`` on the dense projection GEMMs only; the routed experts stay bf16 (the ragged EP backend has no
    simulated-MXFP8 expert path)."""
    MXFP8_EXPERTS = "mxfp8_experts"
    """Real MXFP8 routed experts (ungated ReLU^2 or gated SwiGLU) on the ``ragged_all_to_all`` backend (QuACK
    block-scaled GEMMs, SM100): rows are dispatched as e4m3 with e8m0 scales and the four activation-path expert
    GEMMs run block-scaled; the expert weight gradients and every dense projection stay bf16."""
    MXFP8_DENSE = "mxfp8_dense"
    """Real MXFP8 dense projections (every ``_proj``: attention/KDA, the fused router-latent-shared projection with
    router columns excluded, the latent and shared-expert down projections) on QuACK block-scaled GEMMs, all three
    GEMMs of each; routed experts stay bf16."""
    MXFP8 = "mxfp8"
    """``MXFP8_EXPERTS`` and ``MXFP8_DENSE`` together."""


_DENSE_FP8_SIM = (Fp8Recipe.MXFP8_SIM, Fp8Recipe.MXFP8_SIM_DENSE)
_DENSE_FP8_REAL = (Fp8Recipe.MXFP8_DENSE, Fp8Recipe.MXFP8)


class NormGainFn(StrEnum):
    """How a learned positive scale is read from its raw parameter ``w``. Every option is 1 at init."""

    LINEAR = "linear"
    """``w``, init 1: Adam moves the scale additively and it can cross zero."""
    EXP = "exp"
    """``e^w``, init 0: steps are relative (the same fraction at any size) and the scale stays positive."""
    SOFTPLUS = "softplus"
    """``softplus(w)``, init ``log(e - 1)``: positive, ~linear above 1, exponential-like near 0; 0.63x step at init."""
    SIGMOID2 = "sigmoid2"
    """``2 sigmoid(w)``, init 0: bounded in (0, 2); 0.5x step at init, slowing toward the bounds."""
    SIGMOID = "sigmoid"
    """``sigmoid(w)``, init 0 (a gain of 0.5, not 1): bounded in (0, 1), for norms whose gains settle below 1;
    0.25x step at init. Only through ``norm_gain_overrides``, since it changes the model at init."""


_GAIN_INIT = {
    NormGainFn.LINEAR: 1.0,
    NormGainFn.EXP: 0.0,
    NormGainFn.SOFTPLUS: math.log(math.e - 1.0),
    NormGainFn.SIGMOID2: 0.0,
    NormGainFn.SIGMOID: 0.0,
}

# Norms ``norm_gain_overrides`` can name.
NORM_GAIN_ROLES = ("embed", "embed2", "kv_latent", "kda_o_norm", "final", "rms_attn", "rms_mlp", "moe_latent")


def apply_gain_fn(fn: NormGainFn, w: jax.Array) -> jax.Array:
    """The scale a ``NormGainFn`` reads from its raw parameter ``w``."""
    if fn == NormGainFn.LINEAR:
        return w
    if fn == NormGainFn.EXP:
        return jnp.exp(w)
    if fn == NormGainFn.SOFTPLUS:
        return jax.nn.softplus(w)
    if fn == NormGainFn.SIGMOID2:
        return 2.0 * jax.nn.sigmoid(w)
    if fn == NormGainFn.SIGMOID:
        return jax.nn.sigmoid(w)
    raise ValueError(f"unknown NormGainFn {fn!r}")


class MoeDynamicK(StrEnum):
    """How ``moe_dynamic_layers`` choose each token's routed-expert count, out of the top
    ``num_experts_per_token`` it routes to. Dropped slots get combine weight 0 (they still run, so this measures
    the quality of the rule, not its speed); renormalizing combines renormalize over the kept slots."""

    NONE = "none"
    POSITION = "position"
    """``num_experts_per_token`` for tokens at document position >= ``moe_dynamic_position``, else ``moe_k_low``."""
    ATTN_NORM = "attn_norm"
    """``num_experts_per_token`` for the ``moe_dynamic_frac`` of tokens whose attention output has the largest
    norm in the batch, else ``moe_k_low``."""
    RELU = "relu"
    """Combine weight ``relu(logit + moe_relu_offset)`` (unnormalized, scaled so the sum is about
    ``routing_renorm_sum`` at init), so a token keeps zero to ``num_experts_per_token`` experts; an L1 penalty
    ``moe_relu_l1 * mean_t sum_e relu(logit + offset)`` on the router logits pushes the count down."""
    EPS = "eps"
    """Skip the routed slots whose combine weight is below ``moe_eps`` times the routed total (no renormalization)."""
    TOP_P = "top_p"
    """Keep each token's highest-weight routed slots until they cover ``moe_top_p`` of its combine weight (Huang et
    al. 2024, "Harder Tasks Need More Experts"), then renormalize."""
    THRESHOLD = "threshold"
    """Keep the ``moe_dynamic_frac`` of all (token, routed slot) pairs in the batch with the largest router logits
    (a batch-level null-routing budget: a token keeps 0 to ``num_experts_per_token`` experts)."""


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
    SQRT_SOFTPLUS_RAW = "sqrt_softplus_raw"
    """``sqrt(softplus(logit))`` times a constant (``routing_renorm_sum / (K sqrt(log 2))``, so the sum matches at
    init), no renormalization: a token's total expert weight can vary."""


class ValueEmbeds(StrEnum):
    """Value embeddings on the MLA layers: ``v = lambda1 * v + w * value_embed[token]``."""

    NONE = "none"
    LAMBDA = "lambda"
    """``w = lambda2``, a learned scalar per layer (``lambda1`` init 1, ``lambda2`` init 0)."""
    GATED = "gated"
    """``w = sigmoid(x W_ve_gate)`` per head, ``W_ve_gate`` zero-initialized (0.5 at init)."""
    GATED_LAMBDA = "gated_lambda"
    """``w = lambda2 * sigmoid(x W_ve_gate)``: the gate times a learned scalar per layer (``lambda2`` init 1)."""


_GATED_VALUE_EMBEDS = (ValueEmbeds.GATED, ValueEmbeds.GATED_LAMBDA)


class LatentMixBalance(StrEnum):
    """Load balancing of a mixture-of-latents gate."""

    NONE = "none"
    BIAS = "bias"
    """Auxiliary-loss-free (DeepSeek-V3): a per-block bias on the selection logits only (the weights stay
    ``sigmoid(logit)``), moved each step by ``-latent_mix_bias_rate * sign(load - mean load)``."""
    ENTROPY = "entropy"
    """sigma-MoE: the gradient of ``latent_mix_entropy_weight * -H(mean_t softmax(logits_t))`` is added to the
    gate logits' gradient, pushing the token-averaged block distribution toward uniform."""


class KvMixMode(StrEnum):
    """How the MLA KV-latent mixture combines the chosen experts' down-projections."""

    CONCAT = "concat"
    """``mla_kv_latent_dim`` splits into ``latent_mix_experts`` blocks; each chosen block keeps its own slot of the
    latent (the rest are zero), so each expert has its own rows of ``w_uk`` / ``w_uv`` and the cache needs the
    block indices."""
    SUM = "sum"
    """Each expert down-projects to the full ``mla_kv_latent_dim``; the chosen ones are summed (weighted) into one
    shared latent before the RMSNorm, so ``w_uk`` / ``w_uv`` are shared and MLA absorption is unchanged."""


LATENT_MIX_SITES = ("q", "k", "v", "o", "kv", "moe_in", "moe_out")


def _ve_lambda_init(mode: "ValueEmbeds") -> jax.Array | None:
    if mode == ValueEmbeds.NONE:
        return None
    return jnp.array([1.0, 1.0 if mode == ValueEmbeds.GATED_LAMBDA else 0.0])


def _value_embed_weight(mode: "ValueEmbeds", x: jax.Array, gate: jax.Array | None, lam: jax.Array) -> jax.Array:
    """The value-embedding weight ``w`` of ``v = lambda1 * v + w * value_embed[token]`` (see ``ValueEmbeds``)."""
    if gate is None:
        return lam[1]
    weight = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, gate.astype(x.dtype)))[..., None]
    return lam[1] * weight if mode == ValueEmbeds.GATED_LAMBDA else weight


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


def _fsdp_group_size() -> int:
    mesh = get_abstract_mesh()
    return math.prod(_mesh_axis_size(mesh, axis) for axis in _FSDP_AXES)


def _row_gather_chunks(x: jax.Array, group: int, token_elems: int) -> jax.Array:
    """Flat local ``[t, ...]`` -> ``[chunks, c, ...]`` with ``group * c * token_elems`` at most
    ``_ROW_GATHER_CHUNK_ELEMS`` (or ``c`` odd): the FSDP group's gathered rows of one chunk bound the transients."""
    chunk = x.shape[0]
    while chunk % 2 == 0 and group * chunk * token_elems > _ROW_GATHER_CHUNK_ELEMS:
        chunk //= 2
    return x.reshape(-1, chunk, *x.shape[1:])


def _row_shard_local_ids(table_rows: int, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """FSDP-group-gathered ids relative to this shard's rows, clipped, and whether this shard owns them."""
    local = ids - jax.lax.axis_index(_FSDP_AXES) * table_rows
    valid = (local >= 0) & (local < table_rows)
    return jnp.clip(local, 0, table_rows - 1), valid


def _row_sharded_gather_local(table: jax.Array, ids: jax.Array) -> jax.Array:
    b, s = ids.shape
    rows, dim = table.shape
    group = _fsdp_group_size()

    def chunk_rows(chunk_ids):
        local, valid = _row_shard_local_ids(rows, jax.lax.all_gather(chunk_ids, _FSDP_AXES, tiled=True))
        partial = jnp.where(valid[:, None], table[local], jnp.zeros((), table.dtype))
        # Exactly one shard contributes each row, so the sum is exact in the table dtype.
        return jax.lax.psum_scatter(partial, _FSDP_AXES, scatter_dimension=0, tiled=True)

    return jax.lax.map(chunk_rows, _row_gather_chunks(ids.reshape(-1), group, dim)).reshape(b, s, dim)


def _row_sharded_scatter_local(rows: int, dtype: jnp.dtype, ids: jax.Array, g: jax.Array) -> jax.Array:
    dim = g.shape[-1]
    group = _fsdp_group_size()
    chunk_ids = _row_gather_chunks(ids.reshape(-1), group, dim)
    chunk_g = _row_gather_chunks(g.reshape(-1, dim), group, dim)

    def chunk_grad(d_table, args):
        c_ids, c_g = args
        local, valid = _row_shard_local_ids(rows, jax.lax.all_gather(c_ids, _FSDP_AXES, tiled=True))
        cot = jax.lax.all_gather(c_g, _FSDP_AXES, tiled=True).astype(jnp.float32)
        return d_table.at[local].add(jnp.where(valid[:, None], cot, 0.0)), None

    d_table0 = jax.lax.pcast(jnp.zeros((rows, dim), jnp.float32), _BATCH_AXES, to="varying")
    d_table, _ = jax.lax.scan(chunk_grad, d_table0, (chunk_ids, chunk_g))
    return jax.lax.psum(d_table, _TABLE_REPLICA_AXES).astype(dtype)


@jax.custom_vjp
def _row_sharded_embedding_gather(table: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """``_embedding_gather`` from a table row-sharded ``P(_FSDP_AXES, None)``, without all-gathering the table.

    Each FSDP shard all-gathers its group's token ids, looks up the rows it owns (zeros elsewhere) and
    reduce-scatters them back to the owning tokens; the backward all-gathers the cotangent and scatter-adds the
    rows it owns in float32, so neither the table nor a dense table-sized gradient crosses the network. Both
    walk the tokens in chunks (``_ROW_GATHER_CHUNK_ELEMS``) to bound the gathered ``[tokens, D]`` temporaries.
    """
    return shard_map(
        _row_sharded_gather_local,
        mesh=get_abstract_mesh(),
        in_specs=(P(_FSDP_AXES, None), P(_BATCH_AXES, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(table, reshard(token_ids, P(_BATCH_AXES, None)))


def _row_sharded_embedding_gather_fwd(table: jax.Array, token_ids: jax.Array):
    residual_like = jnp.zeros((0, *table.shape), table.dtype)
    return _row_sharded_embedding_gather(table, token_ids), (token_ids, residual_like)


def _row_sharded_embedding_gather_bwd(residuals, g: jax.Array):
    token_ids, table_like = residuals
    d_table = shard_map(
        functools.partial(_row_sharded_scatter_local, table_like.shape[1] // _fsdp_group_size(), table_like.dtype),
        mesh=get_abstract_mesh(),
        in_specs=(P(_BATCH_AXES, None), P(_BATCH_AXES, None, None)),
        out_specs=P(_FSDP_AXES, None),
    )(reshard(token_ids, P(_BATCH_AXES, None)), reshard(g, P(_BATCH_AXES, None, None)))
    return d_table, np.zeros(token_ids.shape, dtype=jax.dtypes.float0)


_row_sharded_embedding_gather.defvjp(_row_sharded_embedding_gather_fwd, _row_sharded_embedding_gather_bwd)


def _table2_gather(cfg: "GrugModelConfig"):
    """Lookup for the tables stored like the second one (``embed2_fsdp``, ``embed2_grad_fp32``)."""
    if not cfg.embed2_fsdp:
        return _embedding_gather
    if cfg.embed2_row_sharded_gather:
        return _row_sharded_embedding_gather
    return _replicated_fp32_gather


def _replicated_fp32_gather(table: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """All-gather the row-sharded table and look up through the fp32-backward gather. Its backward builds a dense
    full-size gradient and all-reduces it; cheaper than the row-sharded path while the table is small (d512: +6.5%
    step time for the row-sharded path), but it does not fit at d1024 (a 2M x 1024 table)."""
    return _embedding_gather(reshard(table, P(None, None)), token_ids)


# Pair-combine multiplier and murmur3 finalizer constants for the (previous, current) bigram hash.
_BIGRAM_HASH_PAIR = 0x9E3779B1
_MURMUR_C1 = 0x85EBCA6B
_MURMUR_C2 = 0xC2B2AE35
# Per-head salt step for the multi-head n-gram hash (salt 0 is the single-head hash).
_HASH_SALT_STEP = 0x27D4EB2D


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
    moe_dynamic: "MoeDynamicK" = dataclasses.field(default_factory=lambda: MoeDynamicK.NONE)
    """Per-token routed-expert counts on ``moe_dynamic_layers`` (``MoeDynamicK``); ``num_experts_per_token`` is the
    most a token gets."""
    moe_dynamic_layers: tuple[int, ...] = ()
    moe_k_static: int | None = None
    """Routed-expert count of every token on the other MoE layers when set (they route the top
    ``num_experts_per_token`` and keep the first this many, i.e. top-``moe_k_static``); None: all of them."""
    moe_k_low: int = 5
    attn_head_skip: float = 0.0
    """Zero each attention head whose output gate, normalized to (0, 1) (MLA's ``2 sigmoid`` halved, KDA's
    per-channel sigmoid averaged over the head), is below this; 0: off. Every layer logs the fraction of heads
    under 0.1 (``head_gate_off``) and the fraction skipped."""
    moe_small_expert_width: float = 0.0
    """When > 0, the first half of the routed experts use only this fraction of ``intermediate_dim`` hidden units
    (the rest are masked to zero, so the function is a mixed-width bank; compute is not saved). E.g. 1/3 with
    ``intermediate_dim`` 1.5x the base width gives half 0.5x and half 1.5x experts at the base average."""
    moe_dynamic_position: int = 512
    moe_dynamic_frac: float = 0.5
    moe_relu_l1: float = 0.0
    moe_relu_offset: float = 0.0
    """RELU mode keeps the slots with ``logit + moe_relu_offset > 0`` (and the L1 acts on that shifted logit), so a
    positive offset starts training with every routed slot on."""
    moe_eps: float = 0.0
    moe_top_p: float = 0.0
    moe_k_layers: tuple[int, ...] = ()
    """Static per-layer routed counts (one per layer, each <= ``num_experts_per_token``): every layer routes top
    ``num_experts_per_token`` and keeps its first ``moe_k_layers[layer]`` slots. Overrides ``moe_k_static``."""
    moe_null_layers: tuple[int, ...] = ()
    """Null routing on these layers: the router also scores ``num_experts`` zero-computation experts and each
    token takes the top ``2 * num_experts_per_token`` over all ``2 * num_experts`` columns, so a token can spend
    some of its slots on "nothing". A null slot adds zero and has no weights to train. The other layers keep
    top-``num_experts_per_token`` over the real experts (their null router columns are never read)."""
    moe_null_renorm_scale: float = 1.0
    """``moe_null_layers`` combine with ``routing_renorm_sum`` times this, so the real slots can keep the routed
    output's magnitude when part of the top-2K weight goes to null (2.0: the full sum on half the slots)."""
    moe_null_learned_gain: bool = False
    """A learnable scalar (init 1) on each ``moe_null_layers`` layer's routed output."""
    moe_null_target_frac: float | None = None
    """QB treatment of the null experts on ``moe_null_layers``. None: free (bias 0, the router picks the split);
    a float ``f``: QB sends ``f`` of the slots to them on average (``2K f`` null and ``2K (1 - f)`` real slots per
    token), so the router decides which tokens use fewer real experts, not how many on average."""
    # LatentMoE (arXiv 2601.18089); latent RMSNorm per issue #6822.
    latent_dim: int | None = 256
    latent_out_dim: int | None = None
    latent_out_full_layers: tuple[int, ...] = ()
    """Layers whose routed experts write the full residual stream (``latent_out_dim = hidden_dim``, no
    ``w_latent_up``) while the others keep the latent write. These layers live in their own stacks
    (``Transformer.stacked_blocks_tail`` / ``kda_blocks_tail``); needs Block AttnRes, whose loop runs layer by layer."""
    num_layers: int = 6
    num_heads: int = 4
    mla_num_heads: int | None = None
    """Query/KV heads of the global (MLA / full-attention) layers when they differ from ``num_heads`` (the KDA
    layers keep ``num_heads``); the head width stays ``inferred_head_dim``."""
    num_kv_heads: int = 1
    local_kv_heads: int | None = 1
    global_kv_heads: int | None = 1
    head_dim: int | None = 128
    gpu_attention: str = "gpu_fa4_cute"  # "gpu_pallas_fa4": Pallas fwd, FA4 bwd
    max_seq_len: int = 4096
    sliding_window: int = 2048
    global_every: int = 4
    capacity_factor: float = 1.05
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    # QK-norm: non-parametric RMS norm on per-head q/k of the softmax-attention layers.
    qk_norm: bool = True
    rope_global_layers: bool = False
    """Apply RoPE on the long/global softmax layers too (otherwise they use no position encoding)."""
    rope_full_head: bool = False
    """Rotate the whole head dim (otherwise only its first half)."""
    attn_xsa: bool = True
    """Exclusive Self Attention on the softmax layers: subtract each head output's component along its own value."""
    attn_output_gate: bool = True
    """Per-head sigmoid output gate on the softmax layers."""
    sconv: bool = True
    sconv_kernel: int = 4
    sconv_impl: str | None = None
    """``short_conv`` implementation for every SConv: None (the fused Pallas kernel on GPU) or ``triton_gpu``
    (streams the sequence with the taps in registers, kernel size 4 only; from #9708)."""
    sconv_sites: tuple[str, ...] = ("k", "attn", "mlp")
    pooled_transport_capacity_factor: float | None = 1.05
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
    kda_chunk_size: int = 16
    """Chunk length of the fused (GPU) KDA kernels, one of ``KDA_CHUNK_TILES``: 32 runs the same exact
    math ~20% faster per layer on H100 (fwd+bwd 4.25 vs 5.36 ms at d512) via mid-referenced decay tiles."""
    kda_onchip_decay_gate: bool = False
    """Form the KDA log-decay ``-5 sigmoid(exp(A_log) (a + dt_bias))`` inside the fused kernels from the bf16
    projection ``a`` instead of materializing it in fp32 (same math)."""
    kda_onchip_erase_gate: bool = False
    """Form the erase gate ``2 sigmoid(z)`` (``kda_erase_gate``) inside the fused kernels from its bf16 logits."""
    kda_state_grads: StateGrads = StateGrads.XLA
    """Where the fused KDA state pass's backward forms its per-chunk cotangents (``StateGrads``; same math)."""
    kda_state_block_v: int = 32
    """d_v tile of the fused KDA state pass: 64 with ``kda_state_grads=xla`` is 8% faster on GB200 (same math)."""
    moe_shared_placement: SharedExpertPlacement = SharedExpertPlacement.FUSED
    moe_dispatch: MoeDispatch = MoeDispatch.FAST_TRACK
    # Block Attention Residuals (Kimi, arXiv 2603.15031): each sublayer's input is a learned per-sublayer
    # softmax over the RMS-normalized residual-block history (embedding, block sums, running partial).
    attn_res: bool = False
    attn_res_num_blocks: int = 8
    attn_res_two_phase: bool = True
    """``scan_layers``: each layer mixes the history for its attention and MLP gates in one pass over the sources
    (Kimi's two-phase Block AttnRes: inter-block mix + log-sum-exp per gate, then an online-softmax merge with the
    partial), instead of one pass per gate: ~1.5% faster at the GB200 hero gate (lc1-gb200-tp5). Same math; the
    inter-block mix is rounded to the sources' dtype."""
    attn_res_plain_sum: bool = False
    """Every AttnRes gate (and the final gate) returns the plain sum of its sources instead of a softmax mix: the
    standard residual stream (embedding + every earlier sublayer output), keeping the AttnRes loop's features
    (``attn_res_kv_layers`` then reads the residual stream after the leading layers)."""
    attn_res_shared_query: bool = False
    """A third zero-init pseudo-query per layer whose AttnRes mix feeds the shared experts only; the router and
    routed experts keep the MLP gate's mix."""
    attn_res_kv_layers: tuple[int, ...] = ()
    """Layers whose attention K/V projections read their own AttnRes mix: a separate zero-init pseudo-query per
    layer that mixes only the sources the first ``attn_res_kv_source_layers`` layers produced (the extra sources,
    the embedding and the blocks completed by that layer), normed like the attention input. Q and everything
    else keep the attention gate's mix."""
    extra_mlp_layers: tuple[int, ...] = ()
    """Layers with an extra SwiGLU MLP (width ``extra_mlp_dim``, on the MLP gate's mix) whose output is scaled by a
    learned per-layer gain (init 0) and reaches only ``extra_mlp_frac`` of the tokens, chosen by ``extra_mlp_select``
    from an entropy head: a linear read of the stream entering the first of these layers, trained (on stop-gradient
    inputs) to predict the model's own next-token entropy. The trainer turns the extra MLPs on part-way through
    (``extra_mlp_active``)."""
    extra_mlp_dim: int = 0
    entropy_skip_layers: tuple[int, ...] = ()
    """Layers the entropy head's chosen tokens (``extra_mlp_select``, ``extra_mlp_frac``) skip, as in ``LayerSkip``,
    from when ``extra_mlp_active`` turns on. The head reads the stream entering the first of these and
    ``extra_mlp_layers`` unless ``entropy_head_input_layer`` is set."""
    skip_adapter_dim: int = 0
    """With a ``LayerSkip``, tokens that skip its layers pass the final residual stream through one extra SwiGLU MLP
    of this width (``h + MLP(rms_norm(h))``) before the final norm and lm_head, so they can restructure their
    representation for the readout. Tokens that keep the layers never use it. 0: off."""
    skip_router_layers: tuple[int, ...] = ()
    """Layers a learned router lets some tokens skip (mixture-of-depths): a linear score of the rms-normed stream
    entering the first of them; the ``skip_router_frac`` of the batch with the lowest score skips them (as in
    ``LayerSkip``), and the kept tokens' sublayer outputs are scaled by ``2 sigmoid(score)``, which trains the
    router. The threshold is the batch quantile of the scores."""
    skip_router_frac: float = 0.2
    mid_head_layer: int | None = None
    """A second lm_head (own weights, initialized like the lm_head) reads the rms-normed stream entering this layer
    and adds its next-token loss (``mid_head_loss_weight``) in training. A confidence head (``mid_confidence_dim``
    hidden ReLU MLP) reads the stop-gradient stream and the mid head's answer (its top token's embedding, top
    log-prob and entropy) and predicts, with a BCE loss, the probability that the mid head's top token is the
    next token."""
    mid_head_loss_weight: float = 1.0
    mid_head_trunk_grad: bool = True
    """False: the mid head trains on the stop-gradient stream, a probe that leaves the trunk unchanged."""
    mid_confidence_dim: int = 256
    final_conf_layers: tuple[int, ...] = ()
    """Probes (``mid_confidence_dim`` hidden ReLU MLPs, one per layer) on the stop-gradient rms-normed stream
    entering each of these layers (0: the embedding sources), trained with a BCE loss to predict whether the final
    head's top token is the next token. They leave the model unchanged."""
    skip_probe_layer: int | None = None
    """Two stop-gradient probes (``mid_confidence_dim`` hidden ReLU MLPs) on the rms-normed stream entering this
    layer predict, with BCE, whether the final head's top token is the next token: an aware probe, told whether
    this token skips the later layers (``LayerSkip``; ``d`` = 1 if it does), with logit ``f0(x) + d f1(x)``, so
    ``f1`` estimates each token's effect of skipping; and a blind probe ``g(x)``. ``skip_probe_target`` picks
    the predicted quantity."""
    skip_probe_target: SkipProbeTarget = SkipProbeTarget.CORRECT
    mid_skip_layers: tuple[int, ...] = ()
    """Layers that tokens skip (as in ``LayerSkip``) when the confidence head is above ``mid_skip_threshold``, from
    when ``extra_mlp_active`` turns on (the confidence head trains first)."""
    mid_skip_threshold: float = 0.9
    entropy_head_input_layer: int | None = None
    """The layer whose input stream the entropy head reads (0: the embedding sources); at most the first of
    ``entropy_head_layers``. None: the first of ``entropy_head_layers``."""
    extra_mlp_select: ExtraMlpSelect = ExtraMlpSelect.HIGH
    extra_mlp_frac: float = 0.25
    attn_res_kv_source_layers: int | None = None
    """How many leading layers' outputs ``attn_res_kv_layers`` may read (None: ``num_layers // 2``)."""
    attn_res_partial_scale: bool = False
    """Learned scalars ``(a_attn, a_mlp)`` per layer, init 1: each sublayer adds ``a * out`` to its block's partial,
    so a block (and the MLP gate's in-layer partial) weights its attention and MLP outputs."""
    sublayer_gated_norm: bool = True
    """GatedNorm after the RMSNorm on every attention and MLP input (else the RMSNorm alone)."""
    attn_res_layer_backward: AttnResLayerBackward = AttnResLayerBackward.RECOMPUTE
    scan_offload_partials: bool = False
    """With ``scan_layers``, keep the scan's backward residuals in pinned host memory: the per-layer incoming AttnRes
    partials (one ``[B, S, D]`` per scanned layer, 33 GiB per GPU at hero width, brought back one unit at a time in
    the reverse scan) and the final history sources. As the GB200 hero offloads its layer carry. Same math."""
    scan_prefetch_partials: bool = False
    """With ``scan_offload_partials``, the reverse scan copies unit ``u - 1``'s offloaded residuals back to HBM while
    unit ``u``'s backward runs, instead of at the start of unit ``u - 1`` (one more unit's residuals stay in HBM).
    Same math."""
    scan_device_sources: bool = False
    """With ``scan_offload_partials``, the final history sources stay in HBM between the forward and the backward
    instead of a pinned-host round trip at the forward/backward turn, where no compute hides the copies. Same math."""
    scan_release_weight_grads: bool = False
    """``scan_layers`` backward: each layer's end barrier holds the previous layer's weight gradients instead of its
    own, so a layer's weight gradients and their FSDP reduce-scatters can finish under the next layer's backward
    instead of before it starts (at most two layers' gradients in flight). Same math."""
    scan_save_routed_output: bool = False
    """``scan_layers``: keep each layer's combined routed-expert output from the forward (offloaded with the
    partials under ``scan_offload_partials``) and hand it to the backward's per-layer recompute, so the recompute
    skips the expert down projection's consumers: the return all-to-all and the combine (the hero's saved
    ``grug_moe_routed_output``; needs the expert-side routing-weight gradient). Same math."""
    attn_res_fused_source_grads: bool = False
    """``scan_layers`` backward: update the AttnRes source cotangents with one Pallas kernel per layer instead of
    XLA's fusion, which re-reads every token's gate cotangent once per source (``attn_res_source_grads``); the
    kernel skips the history slots no block has filled yet (zero weight)."""
    attn_res_fused_block_mix: bool = False
    """``scan_layers`` with ``attn_res_two_phase``: the attention and MLP gates' phase-1 mixes of the completed
    blocks (and their weight gradients) as Pallas kernels that read only the filled history slots, once for both
    gates (``attn_res_block_mix``), instead of XLA's fusions over every slot. Same math."""
    layer_carry_offload: bool = False
    """Without ``attn_res`` (the plain layer scan, every layer rematerialized): keep each layer's input, the
    residual-stream carry, in pinned host memory for the backward instead of HBM, as the GB200 hero does
    (``moe_hero_ep`` ``offload_carry`` remat mode; ~39 GiB per GPU at hero width). Same math."""
    scan_layers: bool = False
    """Run the AttnRes layer stack as a ``jax.lax.scan`` over ``global_every``-layer units on a fixed-shape
    history (``_AttnResHistory``), each layer rematerialized in the backward, instead of unrolling every layer.
    Same math; the layers after the leading run of identical units (e.g. ``latent_out_full_layers``) stay
    unrolled. Needed to compile deep models (48 layers)."""
    attn_res_remat_attention: bool = False
    """Rematerialize the attention branch inside each AttnRes layer's backward, so its residuals (incl.
    the ``[B, H, S, W]`` Inkling bias) are never alive during the MLP backward. Costs one extra
    attention forward per layer; needed for memory at d1280."""
    value_embeds: "ValueEmbeds" = dataclasses.field(default_factory=lambda: ValueEmbeds.NONE)
    value_embed_fsdp: bool = False
    """Store each MLA layer's value-embedding table row-sharded over the FSDP axes (all-gathered once per use)
    instead of replicating it. Same math; at hero width the replicated tables, their gradients and casts take
    ~30 GiB per GPU."""
    router_combine: "RouterCombine" = dataclasses.field(default_factory=lambda: RouterCombine.SIGMOID_RENORM)
    routing_renorm_sum: float = 2.5
    """Total combine weight of a token's K routed experts (``RouterCombine``)."""
    embed_gated_norm: bool = True
    """GatedNorm after the embedding RMSNorm (else the RMSNorm alone)."""
    final_gated_norm: bool = True
    """GatedNorm after the final RMSNorm, before the lm_head (else the RMSNorm alone)."""
    mla_k_norm: bool = False
    """Weightless per-head RMSNorm on the MLA keys only, no query norm (DeepSeek-V4.1's setup)."""
    mla_q_norm: bool = False
    """Weightless per-head RMSNorm on the MLA queries (composes with ``mla_k_norm``). Queries are recomputed per
    token, so under MLA absorption this adds nothing to the cache."""
    mla_key_offset: bool = False
    """modded-nanogpt partial key offset (record #49): on the MLA layers, the first half of each head's key
    channels come from the previous token (within documents), enabling one-layer induction."""
    logit_soft_cap: float | None = None
    """Tanh soft-cap on the lm_head logits, ``c * tanh(z / c)`` (Gemma 2); None: off."""
    init_std_mult_gates: float = 1.0
    """Init-std multiplier of the MuonH-trained sigmoid gate matrices (KDA ``w_g`` / ``w_beta``, MoE gates)."""
    kda_write_gate: bool = False
    """Gated DeltaNet-2 (arXiv 2605.22791) write gate: each KDA value channel is scaled by ``2 sigmoid(x W_w)``
    before the delta-rule update, a channel-wise write strength next to KDA's per-head beta. Runs through the
    existing kernel."""
    kda_erase_gate: bool = False
    """Gated DeltaNet-2 erase gate on the KDA delta rule (zero-init)."""
    init_std_mult_beta: float | None = None
    """Init-std multiplier of the KDA write-strength projection ``w_beta`` alone (None: ``init_std_mult_gates``).
    Under MuonH it fixes ``w_beta``'s norm for the whole run."""
    shared_ungated_relu2: bool = False
    """Shared experts are ``relu(x @ W_up)^2 @ W_down`` with no gate projection (truly ungated: the gate
    GEMM is dropped from the fused projection). Parameter-match with 1.5x ``shared_expert_intermediate_dim``."""
    moe_expert_waves: int = 1
    """Static dispatch waves in the pooled-wave EP backend. Waves are independent, so the scheduler can overlap
    one wave's all-to-all with another's expert compute (the all-to-all was ~13% exposed at d512)."""
    moe_expert_remat: bool = True
    """Recompute the pooled-wave expert MLP and combine all-to-all in the backward. Off trades activation
    memory for one fewer expert forward and combine all-to-all per wave."""
    fp8_recipe: Fp8Recipe = Fp8Recipe.NONE
    """Low-precision recipe for attention/KDA projections, the fused router-latent-shared projection (router
    columns excluded), the latent up-projection, the shared expert and the routed experts; see ``Fp8Recipe``."""
    mxfp8_fused_grad_quant: bool = False
    """Real-MXFP8 dense projections quantize the output gradient along rows and columns in one Pallas pass."""
    moe_ragged_chunks: int = 2
    moe_overlap_param_gather: OverlapParamGather = OverlapParamGather.IN_WORK
    """``moe_shared_placement=overlap``: where the EP shard all-gathers the shared experts' FSDP-sharded weights,
    which places their gradients' reduce-scatter (``OverlapParamGather``). Same math."""
    moe_overlap_weight_grads: bool = False
    """``ragged_all_to_all``: run the routed experts' weight gradients while their input cotangent's return
    all-to-all is in flight (``ragged_overlap_weight_grads``). Same math."""
    kda_fused_projections: bool = False
    """Under a real MXFP8 dense ``fp8_recipe``, project KDA's q/k/v/gate (and write/erase) from ``x`` as one
    block-scaled GEMM over their concatenated weights, so ``x`` is quantized once. Off: 1% slower on the GB200
    recipe with the Pallas quantizers (lc1-gb200-pq4 15.06 vs pu2 14.91 s/step)."""
    moe_mxfp8_dispatch: bool = False
    """With MXFP8 experts (``fp8_recipe``), send the dispatched rows as MXFP8 bytes instead of bf16 rows quantized on
    arrival. Off by default: the extra uint8 transport buffer OOMs NCCL's symmetric memory on the GB200 recipe."""
    mxfp8_expert_wgrad: bool = False
    """With MXFP8 experts (``fp8_recipe``), run the routed experts' weight-gradient GEMMs block-scaled too (operands
    quantized along the token axis per expert) instead of bf16 (``ExpertMxfp8.ALL``)."""
    mxfp8_shared_input_quant: bool = False
    """Under a real MXFP8 dense ``fp8_recipe``, quantize the attention input once for the q, k and v projections
    (and MLA's q and KV latent down-projection), each direction, instead of once per projection
    (``mx_linear_shared_input``; the GEMMs stay separate)."""
    mxfp8_epilogue_tokens: bool = False
    """With ``mxfp8_expert_wgrad`` on ungated ReLU^2 experts, quantize ``post`` and ``d pre`` along the token axis
    in the ReLU^2 backward GEMM's epilogue instead of two passes after it (``ExpertMxfp8.ALL_EPILOGUE_TOKENS``)."""
    moe_serial_pair: bool = False  # ragged experts: neuron halves in series, y1 + mlp_b(x + y1)
    """Sequential local-expert chunks of the ``ragged_all_to_all`` backend (each a dispatch, expert MLP and
    return); fewer chunks mean fewer, larger collectives at more transport-buffer memory."""
    moe_implementation: str = "fixed_pooled_wave_all_to_all"
    """Expert-parallel transport, one of ``MOE_IMPLEMENTATIONS``. ``ragged_all_to_all`` ignores the pooled-wave
    knobs (``pooled_transport_capacity_factor``, ``moe_expert_waves``, ``moe_expert_remat``) and sizes its receiver
    buffers by ``capacity_factor``; its XLA transport kernel is ``GrugRunConfig.ragged_transport``."""
    moe_ungated_kernel: bool = False
    """With ``moe_ungated_relu2``, run the EP (pooled-wave or ragged) experts truly ungated (one ``W_up`` GEMM)
    instead of tying the gate to ``W_up``. Same math; skips the duplicated GEMM."""
    moe_ungated_relu2: bool = False
    """Routed experts are ``relu(x @ W_up)^2 @ W_down`` (no gate projection). Run through the gated kernels
    as ``relu(g) * u`` with the gate tied to ``W_up`` (``relu(u) * u = relu(u)^2``), so every MoE backend
    computes it unchanged; the gate GEMM is still paid. Parameter-match with 1.5x ``num_experts`` and
    ``num_experts_per_token``."""
    expert_activation: str = "silu"
    """Gate activation of the routed and shared GLU experts (an ``ActivationFunctionEnum`` value, e.g.
    ``relu2`` for ReLU^2-GLU as in Primer / ReMoE)."""
    moe_drop_renorm: bool = False
    """Renormalize each token's combine weights over the slots that survived the capacity-limited EP dispatch
    (one extra int8 all-to-all per expert wave). Without it a dropped assignment just loses its weight."""
    learnable_qk_mult: bool = False
    """A learnable scalar per softmax-attention layer (init ``qk_mult``) in place of the fixed ``qk_mult``."""
    second_embed: bool = False
    embed2_rows: int = 0
    bigram_gate: bool = False
    """Engram-style content gate on the bigram source: ``g_t = sigmoid(sum_d w_d rms(e_t)_d rms(b_t)_d + c)`` scales
    each token's bigram row by how its token embedding and bigram row interact, which the static AttnRes queries
    can't express (w = 0 and c = +2 at init)."""
    bigram_gate_rank: int = 0
    """With ``bigram_gate``, a per-channel gate instead of a scalar: ``g_t = sigmoid(rms(e_t) * rms(b_t) @ A @ B + c)``
    with rank-r ``A`` (random) and ``B`` (zero), so each token keeps some bigram features and drops others."""
    embed2_fsdp: bool = False
    """Store the second table (and so its optimizer / EMA state) row-sharded over the FSDP axes. The lookup either
    all-gathers it (default) or reads rows in place (``embed2_row_sharded_gather``). Same math."""
    embed2_row_sharded_gather: bool = False
    """With ``embed2_fsdp``, look rows up where they live (all-gather the ids, reduce-scatter the rows; backward
    scatter-adds into each shard) instead of all-gathering the table and all-reducing a dense full-size gradient.
    Needed at d1024+ (the dense path's ~12 GiB of temporaries OOMs there); slower at d512 (+6.5% step)."""
    embed2_ngram: int = 2
    """With ``second_embed_bigram``, the n-gram length hashed into the table (2: bigram, 3: trigram)."""
    embed2_dim: int | None = None
    """Low-rank second table: ``embed2_rows x embed2_dim`` rows up-projected to ``hidden_dim`` by a shared
    ``embed2_up`` matrix (None: full-width rows). Shrinks the table's per-step gradient all-reduce and
    optimizer work by ``hidden_dim / embed2_dim``."""
    """Rows of the second embedding table (0: ``vocab_size``). With ``second_embed_bigram`` the (previous,
    current) hash spreads over these rows, so a table much larger than the vocab keeps bigrams apart."""
    embed2_lambda_init: float = 0.1
    second_embed_bigram: bool = False
    """Index the second table by a hash of (previous token, token) instead of the token: a bigram
    embedding with ``vocab_size`` hashed rows (the previous token is a sentinel at document starts)."""
    """A second, independently initialized token-embedding table, RMS-normed, as an extra AttnRes source."""

    def __post_init__(self) -> None:
        if self.mxfp8_expert_wgrad and self.fp8_recipe not in (Fp8Recipe.MXFP8_EXPERTS, Fp8Recipe.MXFP8):
            raise ValueError("mxfp8_expert_wgrad needs MXFP8 experts (fp8_recipe=mxfp8 or mxfp8_experts)")
        if self.mxfp8_shared_input_quant and self.fp8_recipe not in _DENSE_FP8_REAL:
            raise ValueError("mxfp8_shared_input_quant needs a real MXFP8 dense fp8_recipe (mxfp8 or mxfp8_dense)")
        if self.mxfp8_epilogue_tokens and not self.mxfp8_expert_wgrad:
            raise ValueError("mxfp8_epilogue_tokens needs mxfp8_expert_wgrad")
        if self.moe_k_layers and (
            len(self.moe_k_layers) != self.num_layers or max(self.moe_k_layers) > self.num_experts_per_token
        ):
            raise ValueError("moe_k_layers needs one count per layer, each <= num_experts_per_token")
        if self.scan_device_sources and not self.scan_offload_partials:
            raise ValueError("scan_device_sources keeps on device the sources scan_offload_partials offloads")
        if self.scan_prefetch_partials and not self.scan_offload_partials:
            raise ValueError("scan_prefetch_partials prefetches the residuals scan_offload_partials offloads")
        if self.layer_carry_offload and self.attn_res:
            raise ValueError("layer_carry_offload offloads the plain layer scan's carry (attn_res=false)")
        if self.scan_save_routed_output and not self.scan_layers:
            raise ValueError("scan_save_routed_output saves the routed output across the scan_layers backward")
        if self.moe_dispatch == MoeDispatch.HERO_LEVANTER and self.moe_shared_placement != SharedExpertPlacement.FUSED:
            raise ValueError(
                "moe_dispatch=hero_levanter places the shared experts itself: keep moe_shared_placement=fused"
            )
        if self.moe_shared_placement != SharedExpertPlacement.FUSED and self.fp8_recipe in _DENSE_FP8_SIM:
            raise ValueError("fp8_recipe covers only moe_shared_placement=fused shared experts")
        if unknown := set(()) - set(LATENT_MIX_SITES):
            raise ValueError(f"latent_mix_sites: unknown sites {sorted(unknown)}")
        attn_latents = {
            "q": 0,
            "k": 0,
            "v": 0,
            "o": 0,
        }
        if any(attn_latents.values()) and (self.mla or (0)):
            raise ValueError("attn_latent_* is implemented for plain GQA layers only")
        LatentMixBalance(LatentMixBalance.NONE)
        KvMixMode(KvMixMode.CONCAT)
        if self.moe_implementation not in MOE_IMPLEMENTATIONS:
            raise ValueError(f"moe_implementation must be one of {MOE_IMPLEMENTATIONS}, got {self.moe_implementation!r}")
        if not self.dense_mlp and self.num_experts_per_token >= self.num_experts:
            # QB routing takes top-(k+1) and keeps the last entry as the threshold alpha, so a
            # full-bank top-k asks `jax.lax.top_k` for more entries than the router has experts.
            raise ValueError("num_experts_per_token must be < num_experts, because QB routing selects top-(k+1)")
        if self.local_mixer == LocalMixer.KDA and not self.attn_res:
            raise ValueError("local_mixer=kda requires attn_res (KDA layers run in the unrolled AttnRes loop)")
        if self.scan_layers and (not self.attn_res or self.attn_res_layer_backward != AttnResLayerBackward.RECOMPUTE):
            raise ValueError("scan_layers needs attn_res with attn_res_layer_backward=recompute")
        if self.kda_chunk_size not in KDA_CHUNK_TILES:
            raise ValueError(f"kda_chunk_size must be one of {list(KDA_CHUNK_TILES)}, got {self.kda_chunk_size}")
        if self.moe_drop_renorm and self.moe_implementation != "fixed_pooled_wave_all_to_all":
            raise ValueError("moe_drop_renorm needs moe_implementation=fixed_pooled_wave_all_to_all")
        if self.extra_mlp_layers and self.extra_mlp_dim <= 0:
            raise ValueError("extra_mlp_layers needs extra_mlp_dim > 0")
        if self.entropy_head_layers:
            if not self.attn_res or self.scan_layers:
                raise ValueError("extra_mlp_layers and entropy_skip_layers need attn_res and the unrolled layer loop")
            if self.entropy_head_input_layer is not None and not (
                0 <= self.entropy_head_input_layer <= min(self.entropy_head_layers)
            ):
                raise ValueError("entropy_head_input_layer must be in [0, min(entropy_head_layers)]")
            if not 0.0 < self.extra_mlp_frac < 1.0:
                raise ValueError(f"extra_mlp_frac must be in (0, 1), got {self.extra_mlp_frac}")
        if self.skip_adapter_dim and not self.attn_res:
            raise ValueError("skip_adapter_dim needs attn_res")
        if self.skip_router_layers:
            if not self.attn_res or self.scan_layers or not 0.0 < self.skip_router_frac < 1.0:
                raise ValueError("skip_router_layers needs attn_res, the unrolled layer loop and 0 < frac < 1")
        if self.final_conf_layers and (
            not self.attn_res or self.scan_layers or not all(0 <= i < self.num_layers for i in self.final_conf_layers)
        ):
            raise ValueError("final_conf_layers needs attn_res, the unrolled layer loop and layers in [0, num_layers)")
        if self.skip_probe_layer is not None and (
            not self.attn_res or self.scan_layers or not 0 < self.skip_probe_layer < self.num_layers
        ):
            raise ValueError("skip_probe_layer needs attn_res, the unrolled layer loop and 0 < layer < num_layers")
        if self.mid_skip_layers and (self.mid_head_layer is None or min(self.mid_skip_layers) < self.mid_head_layer):
            raise ValueError("mid_skip_layers needs mid_head_layer at or before them")
        if self.mid_head_layer is not None:
            if not self.attn_res or self.scan_layers or not 0 < self.mid_head_layer < self.num_layers:
                raise ValueError("mid_head_layer needs attn_res, the unrolled layer loop and 0 < layer < num_layers")
            if not isinstance(self.logit_soft_cap, float | None):
                raise ValueError("mid_head_layer needs a scalar logit_soft_cap")
        if self.attn_res_kv_layers:
            if not self.attn_res or self.scan_layers:
                raise ValueError("attn_res_kv_layers needs attn_res and the unrolled layer loop (scan_layers=false)")
            cutoff = self.kv_source_layers
            if not 0 < cutoff < self.num_layers:
                raise ValueError(f"attn_res_kv_source_layers must be in 1..{self.num_layers - 1}, got {cutoff}")
            if bad := [i for i in self.attn_res_kv_layers if not cutoff <= i < self.num_layers]:
                raise ValueError(f"attn_res_kv_layers {bad} must read only earlier layers (>= {cutoff})")
        if (self.moe_null_renorm_scale != 1.0 or self.moe_null_learned_gain) and not self.moe_null_layers:
            raise ValueError("moe_null_renorm_scale / moe_null_learned_gain need moe_null_layers")
        if self.moe_null_target_frac is not None:
            if not self.moe_null_layers or not 0.0 < self.moe_null_target_frac < 1.0:
                raise ValueError(
                    f"moe_null_target_frac needs moe_null_layers and to be in (0, 1), got {self.moe_null_target_frac}"
                )
        if self.moe_null_layers:
            if self.scan_layers:
                raise ValueError("moe_null_layers needs the unrolled layer loop (scan_layers=false)")
            if bad := [i for i in self.moe_null_layers if not 0 <= i < self.num_layers]:
                raise ValueError(f"moe_null_layers {bad} out of range for {self.num_layers} layers")
        tail_modes = (self.latent_out_full_layers, (), (), ())
        if sum(bool(x) for x in tail_modes) > 1:
            raise ValueError(
                "latent_out_full_layers, latent_free_layers, wide_expert_layers and topk_layers share the tail stacks"
            )
        tail = _tail_layers(self)
        if tail:
            if not self.attn_res or self.latent_dim is None or self.latent_out_dim is not None or self.dense_mlp:
                raise ValueError(
                    "latent_out_full_layers / latent_free_layers need attn_res, a MoE latent (latent_dim) and no "
                    "global latent_out_dim"
                )
            if len(set(tail)) != len(tail) or not all(0 <= i < self.num_layers for i in tail):
                raise ValueError(f"tail layers must be distinct layer indices, got {tail}")
        if self.latent_out_dim is not None and (self.dense_mlp or (0) or self.num_null_experts):
            raise ValueError(
                "latent_out_dim needs routed experts and no latent_write_select, moe_bank2 or zero-computation experts"
            )
        overrides = _norm_gain_overrides(self)
        if overrides.keys() - set(NORM_GAIN_ROLES):
            raise ValueError(f"norm_gain_overrides: unknown norms {sorted(overrides.keys() - set(NORM_GAIN_ROLES))}")

    @property
    def expert_in_dim(self) -> int:
        """Width the routed experts read (one ``expert_read_groups`` slice when set, plus ``expert_private_dim``)."""
        shared = self.latent_dim if self.latent_dim is not None else self.hidden_dim
        return shared + (0)

    @property
    def expert_out_dim(self) -> int:
        """Width the routed experts write (``latent_out_dim``)."""
        return self.latent_out_dim if self.latent_out_dim is not None else self.expert_in_dim - (0)

    @property
    def qb_num_experts(self) -> int:
        """Width of the per-layer QB betas: the widest layer's expert count, null experts included."""
        return max(self.num_experts, (0)) + self.num_null_experts

    @property
    def routed_write_extra(self) -> int:
        """The lm_head-only slice the routed experts write (``final_write_extra`` with the routed source)."""
        return 0

    @property
    def has_latent_up(self) -> bool:
        """The MoE maps the experts' output to ``hidden_dim`` with ``w_latent_up``."""
        if self.latent_out_dim is not None:
            return self.latent_out_dim != self.hidden_dim + self.routed_write_extra
        return self.latent_dim is not None

    @property
    def kv_in_dim(self) -> int:
        """Input width of the attention K/V projections: the side stream (``kv_stream_dim``) or the main stream."""
        return self.hidden_dim

    @property
    def entropy_head_layers(self) -> tuple[int, ...]:
        """The layers the entropy head's selection serves; it reads the stream entering the first."""
        return (*self.extra_mlp_layers, *self.entropy_skip_layers)

    @property
    def entropy_head_read_layer(self) -> int:
        if self.entropy_head_input_layer is not None:
            return self.entropy_head_input_layer
        return min(self.entropy_head_layers)

    @property
    def kv_source_layers(self) -> int:
        """The leading layers whose outputs ``attn_res_kv_layers`` read (``attn_res_kv_source_layers``)."""
        return self.num_layers // 2 if self.attn_res_kv_source_layers is None else self.attn_res_kv_source_layers

    @property
    def num_null_experts(self) -> int:
        """Zero-computation experts, the router columns past ``num_experts`` (``moe_null_layers``)."""
        return self.num_experts if self.moe_null_layers else 0

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


def _proj(cfg: "GrugModelConfig", x: jax.Array, w: jax.Array, out_sharding: P | None = None) -> jax.Array:
    """``x[..., k] @ w[k, n]`` under ``cfg.fp8_recipe``."""
    if cfg.fp8_recipe in _DENSE_FP8_SIM:
        return mx_dense(x, w, out_sharding)
    if cfg.fp8_recipe in _DENSE_FP8_REAL:
        # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
        from levanter.grug._moe.quack_mxfp8 import mx_linear  # noqa: PLC0415

        return mx_linear(x, w, out_sharding, cfg.mxfp8_fused_grad_quant)
    return jnp.einsum("...k,kn->...n", x, w, out_sharding=out_sharding)


def _proj_shared_input(cfg: "GrugModelConfig", x: jax.Array, weights: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """``_proj(cfg, x, w)`` for every ``w`` of ``weights`` with ``x`` quantized once (``mxfp8_shared_input_quant``)."""
    from levanter.grug._moe.quack_mxfp8 import mx_linear_shared_input  # noqa: PLC0415

    return mx_linear_shared_input(x, weights, cfg.mxfp8_fused_grad_quant)


def _proj_parts(cfg: "GrugModelConfig", x: jax.Array, weights: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """``_proj(cfg, x, w)`` for every ``w`` of ``weights``, as one projection onto their concatenation."""
    fused = _proj(cfg, x, jnp.concatenate(weights, axis=1))
    return tuple(jnp.split(fused, list(itertools.accumulate(w.shape[1] for w in weights[:-1])), axis=-1))


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
    implementation: str | None = eqx.field(static=True, default=None)
    """``short_conv`` kernel (``sconv_impl``); None is its default (the Pallas kernel on GPU)."""

    @staticmethod
    def init(channels: int, kernel_size: int, implementation: str | None = None) -> "ShortConv":
        weight = jnp.zeros((kernel_size, channels)).at[0].set(1.0)
        # FSDP-shard the channel dim so the grad reduce-scatters instead of all-reducing; the
        # forward gathers the weight back to replicated.
        return ShortConv(
            weight=reshard(weight, P(None, _FSDP_AXES)), kernel_size=kernel_size, implementation=implementation
        )

    def __call__(self, x: Float[Array, "B S C"], segment_ids: Int[Array, "B S"] | None = None) -> Float[Array, "B S C"]:
        # segment_ids zero any tap reaching into a previous document, so the conv never crosses a boundary.
        weight = reshard(self.weight, P(None, None))
        return short_conv(weight, x, segment_ids, implementation=self.implementation, batch_axes=_BATCH_AXES)


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


class LatentProj(eqx.Module):
    """``x -> RMSNorm(x W_down) W_up`` with a learnable gain on the latent, so the factored map keeps a free
    scale under the hyperball's fixed matrix norms. As a mixture of latents (``mix_gate`` set) the latent is
    ``E`` blocks, each normed alone and weighted by SwitchHead's sigmoid top-k gate."""

    down: Float[Array, "I R"]
    up: Float[Array, "R O"]
    norm: "LearnedRMSNorm"  # [E, R / E] gain
    mix_gate: Float[Array, "I E"] | None
    topk: int = eqx.field(static=True)

    @staticmethod
    def init(
        cfg: GrugModelConfig, key: PRNGKeyArray, in_dim: int, width: int, out_dim: int, mix: bool, out_std: float
    ) -> "LatentProj":
        k_down, k_up, k_gate = random.split(key, 3)
        blocks = (4) if mix else 1
        return LatentProj(
            down=reshard(_init_weight(k_down, (in_dim, width), cfg.initializer_std), P(None, None)),
            up=reshard(_init_weight(k_up, (width, out_dim), out_std), P(None, None)),
            norm=_grouped_rms_norm(cfg, blocks, width // blocks),
            mix_gate=(
                reshard(_init_weight(k_gate, (in_dim, blocks), cfg.initializer_std), P(None, None)) if mix else None
            ),
            topk=(2) if mix else 1,
        )

    def __call__(self, cfg: GrugModelConfig, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
        blocks = self.norm.weight.shape[0]
        x = reshard(x, P(_BATCH_AXES, *([None] * (x.ndim - 1))))
        latent = rearrange(_proj(cfg, x, self.down), "... (e r) -> ... e r", e=blocks)
        latent = self.norm(latent)
        if self.mix_gate is not None:
            latent = latent * mixture_weights(x, self.mix_gate, blocks, self.topk)[..., None].astype(latent.dtype)
        return _proj(cfg, rearrange(latent, "... e r -> ... (e r)"), self.up)


class CausalSelfAttention(eqx.Module):
    """Softmax attention: GQA (``w_q``/``w_k``/``w_v``), or MLA with a compressed KV latent
    (``w_q``/``w_dkv``/``w_uk``/``w_uv``); either with half-RoPE or the Inkling bias."""

    w_q: Float[Array, "D NH"] | None  # None: factored (latent_q)
    w_k: Float[Array, "D MH"] | None
    w_v: Float[Array, "D MH"] | None
    w_o: Float[Array, "NH D"] | None  # None: factored (latent_o)
    attn_gate: (
        Float[Array, "D G"] | None
    )  # G = N heads, N*H channels (attn_gate_elementwise) or the rank (attn_gate_rank)
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
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, layer_index: jax.Array) -> "CausalSelfAttention":
        """``layer_index`` (0-indexed, may be traced under the stacked init) sets the DIFF ``lambda_init``."""
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        std = cfg.initializer_std
        attn_gate = reshard(jnp.zeros((d, n)), P(None, None)) if cfg.attn_output_gate else None
        if cfg.mla:
            k_q, k_dkv, k_uk, k_uv, k_o, k_rel, k_ve = random.split(key, 7)
            # A separate key stream, so turning mla_diff_attn on leaves every other initial weight unchanged.
            _k_q2, _k_uk2, _k_lam = random.split(random.fold_in(key, 1), 3)
            kvl = cfg.mla_kv_latent_dim + (0)
            use_ve = cfg.value_embeds != ValueEmbeds.NONE
            _k_qn, _k_ukn = random.split(random.fold_in(key, 2))
            # GDLA: one value head per group and only the signal heads reach the output.
            n_v = n
            n_out = n - (0)
            return CausalSelfAttention(
                w_q=reshard(_init_weight(k_q, (d, n * h * 1), std), P(_FSDP_AXES, "model")),
                w_k=None,
                w_v=None,
                w_o=reshard(_init_weight(k_o, (n_out * h, d), std * (1.0)), P("model", _FSDP_AXES)),
                attn_gate=attn_gate,
                sconv_k=(
                    ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl)
                    if cfg.sconv and "k" in cfg.sconv_sites
                    else None
                ),
                sconv_q=(
                    ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl)
                    if cfg.sconv and "q" in cfg.sconv_sites
                    else None
                ),
                # Without Inkling the MLA layers are NoPE (they are global, so RoPE is disabled there).
                rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
                w_dkv=reshard(
                    _init_weight(
                        k_dkv,
                        (
                            cfg.kv_in_dim,
                            cfg.mla_kv_latent_dim,
                        ),
                        std,
                    ),
                    P(_FSDP_AXES, None),
                ),
                kv_latent_norm=_learned_rms_norm(cfg, cfg.mla_kv_latent_dim, cfg.layer_norm_eps, role="kv_latent"),
                w_uk=reshard(_init_weight(k_uk, (kvl, n * h), std), P(None, "model")),
                w_uv=reshard(_init_weight(k_uv, (kvl, n_v * h), std), P(None, "model")),
                value_embed=(
                    reshard(
                        _init_weight(k_ve, (cfg.vocab_size, n_v * h), std),
                        P(_FSDP_AXES, None) if cfg.value_embed_fsdp else P(None, None),
                    )
                    if use_ve
                    else None
                ),
                ve_lambda=_ve_lambda_init(cfg.value_embeds),
                ve_gate=(
                    reshard(jnp.zeros((d, n_v)), P(None, None)) if cfg.value_embeds in _GATED_VALUE_EMBEDS else None
                ),
                qk_mult=_qk_mult_init(cfg, n),
                cfg=cfg,
            )
        k_q, k_k, k_v, k_o, k_rel, _k_sv, _k_so, k_lat = random.split(key, 8)
        lat_keys = dict(zip("qkvo", random.split(k_lat, 4), strict=True))
        o_std = std * (1.0)

        def _latent(site: str, width: int, in_dim: int, out_dim: int, out_std: float) -> LatentProj | None:
            if not width:
                return None
            mix = site in ()
            return LatentProj.init(cfg, lat_keys[site], in_dim, width, out_dim, mix, out_std)

        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (cfg.kv_in_dim, m * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (cfg.kv_in_dim, m * 1 * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * 1 * h, d), o_std), P("model", _FSDP_AXES)),
            attn_gate=attn_gate,
            sconv_k=(
                ShortConv.init(m * h, cfg.sconv_kernel, cfg.sconv_impl) if cfg.sconv and "k" in cfg.sconv_sites else None
            ),
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
        head_dim = self.cfg.inferred_head_dim
        proj_inputs = proj_inputs or {}
        q_in = proj_inputs.get("q", x)
        direct = self.w_dkv is None  # mla_kv_direct: k / v read the normed attention input itself
        if not direct:
            assert self.kv_latent_norm is not None
        latent = x if kv_input is None else kv_input
        assert self.w_q is not None
        q_projected = None
        if self.cfg.mxfp8_shared_input_quant and not direct and kv_input is None and "q" not in proj_inputs:
            assert self.w_dkv is not None
            q_projected, latent = _proj_shared_input(self.cfg, x, (self.w_q, self.w_dkv))
        elif not direct:
            latent = _proj(self.cfg, latent, self.w_dkv)

        def project_q(w_q: jax.Array) -> jax.Array:
            q_flat = _proj(self.cfg, q_in, w_q) if q_projected is None else q_projected
            if self.sconv_q is not None:
                q_flat = self.sconv_q(q_flat, sconv_segment_ids)
            return rearrange(q_flat, "... (n d) -> ... n d", d=head_dim * 1)

        q = project_q(self.w_q)
        if direct:
            kv_latent = latent
        else:
            kv_latent = self.kv_latent_norm(latent)

        # k / v may read a different stream than the shared latent (attn_res_sum_inputs); each then gets its
        # own latent from that stream (same W_dkv and latent norm).
        def stream_latent(name: str) -> jax.Array:
            if name not in proj_inputs:
                return kv_latent
            if direct:
                return proj_inputs[name]
            assert self.kv_latent_norm is not None and self.w_dkv is not None
            return self.kv_latent_norm(_proj(self.cfg, proj_inputs[name], self.w_dkv))

        k_latent, v_latent = stream_latent("k"), stream_latent("v")

        def project_k(w_uk: jax.Array) -> jax.Array:
            k_flat = _proj(self.cfg, k_latent, w_uk)
            if self.sconv_k is not None:
                k_flat = self.sconv_k(k_flat, sconv_segment_ids)
            return rearrange(k_flat, "... (n d) -> ... n d", d=head_dim)

        k = project_k(self.w_uk)
        if self.w_uv is None:  # mla_shared_kv_tied: the value is the cached key vector itself
            v = k
        else:
            v = rearrange(_proj(self.cfg, v_latent, self.w_uv), "... (n d) -> ... n d", d=head_dim)
        if self.value_embed is not None:
            assert self.ve_lambda is not None and token_ids is not None
            table = self.value_embed.astype(x.dtype)
            # A sharded table is all-gathered once per use (16k rows: ~200 MB bf16 per MLA layer at hero width) and
            # its gradient reduce-scattered back; the chunked row lookup issued ~500 tiny collectives per use.
            ve = _embedding_gather(reshard(table, P(None, None)) if self.cfg.value_embed_fsdp else table, token_ids)
            ve = rearrange(ve, "... (n d) -> ... n d", d=head_dim)
            lam = self.ve_lambda.astype(x.dtype)
            ve_weight = _value_embed_weight(self.cfg.value_embeds, x, self.ve_gate, lam)
            v = lam[0] * v + ve_weight * reshard(ve, _partition_spec_of(v) or P(_BATCH_AXES, None, None, None))
        return q, k, v, (None), (None)

    def _gqa_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        is_global: bool | jax.Array,
        kv_input: Float[Array, "B S W"] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        head_dim = self.cfg.inferred_head_dim
        kv_in = x if kv_input is None else kv_input

        def _project(inp: jax.Array, w: jax.Array | None, latent: LatentProj | None) -> jax.Array:
            if latent is not None:
                return latent(self.cfg, inp)
            assert w is not None
            return _proj(self.cfg, inp, w)

        if self.cfg.mxfp8_shared_input_quant and kv_input is None:
            q_flat, k_flat, v_flat = _proj_shared_input(self.cfg, x, (self.w_q, self.w_k, self.w_v))
        else:
            q_flat = _project(x, self.w_q, None)
            k_flat = _project(kv_in, self.w_k, None)
            v_flat = _project(kv_in, self.w_v, None)
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
            q, k, v, second_qk, _match = self._mla_qkv(x, sconv_segment_ids, token_ids, kv_share, proj_inputs, kv_input)
        else:
            q, k, v = self._gqa_qkv(x, sconv_segment_ids, is_global, kv_input)
        stats: dict[str, jax.Array] = {}
        fox_key_bias = None
        # The FoX key channels reach the kernel's 1/sqrt(padded head_dim) scale; q is rescaled to keep q.k exact.
        fox_q_scale = 1.0

        # Half-RoPE: rotate only the first half of Q/K head_dim; disable_rope skips RoPE on long/global layers.
        def _rope(qh: jax.Array, kh: jax.Array) -> tuple[jax.Array, jax.Array]:
            half = head_dim if self.cfg.rope_full_head else head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                qh[..., :half], kh[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            return (
                jnp.concatenate([q_rot, qh[..., half:]], axis=-1),
                jnp.concatenate([k_rot, kh[..., half:]], axis=-1),
            )

        def _transform_qk(q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Key offset, q/k norms, RoPE, qk_mult and SSMax: everything between the projections and the kernel."""
            if self.cfg.mla and ():
                q, k = apply_lag_groups(q, k, sconv_segment_ids, parse_lag_groups((), head_dim))
            if self.cfg.mla and self.cfg.mla_key_offset:
                if k.shape[2] < 0:  # shared KV: per-query-head offsets
                    k = reshard(jnp.broadcast_to(k, q.shape[:3] + k.shape[3:]), _partition_spec_of(q))
                k = partial_key_offset(k, sconv_segment_ids, _partition_spec_of(k), ())
            if self.cfg.qk_norm:
                q = rms_norm(q)
                k = rms_norm(k)
            elif self.cfg.mla_k_norm and self.cfg.mla:
                k = rms_norm(k)
            if self.cfg.mla_q_norm and self.cfg.mla and not self.cfg.qk_norm:
                q = rms_norm(q)
            # The Inkling bias replaces RoPE (no rotation).
            if self.rel_pos is None:
                if self.cfg.rope_global_layers:
                    q, k = _rope(q, k)
                elif isinstance(disable_rope, bool):
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
            return q, k

        # The Inkling bias: a per-head content-dependent bias (from x) on the pre-softmax logits.
        rel_bias = self.rel_pos(x) if self.rel_pos is not None else None
        q, k = _transform_qk(q, k)
        head_probe_attn = _head_attention_stats(q, k, mask, rel_bias, sconv_segment_ids) if _HEAD_PROBE else None
        if _HEAD_PROBE:
            stats[HEAD_QCOS_STAT] = _query_head_cosines(q, sconv_segment_ids)
        if _FORWARD_PROBE is not None and _FORWARD_PROBE.attn_queries and self.cfg.mla:
            if fox_key_bias is not None or second_qk is not None:
                raise ValueError("forward_probe attention supports MLA without FoX or differential attention")
            stats[ATTN_PROBE_STAT] = _probe_attention(q, k, mask, rel_bias, _FORWARD_PROBE.attn_queries)
        if trace_capture.TRACE and self.cfg.mla:
            seg = sconv_segment_ids if sconv_segment_ids is not None else jnp.zeros(q.shape[:2], jnp.int32)
            offset = self.cfg.mla_key_offset
            shifted = shifted_channels(q.shape[2], q.shape[3], ()) if offset else 0.0
            top_i, top_f = trace_capture.attention_topk(q, k, rel_bias, seg, shifted, _BATCH_AXES)
            stats[f"{_LAYER_KNOB_PREFIX}trace_topk_idx"] = top_i
            stats[f"{_LAYER_KNOB_PREFIX}trace_topk_fields"] = top_f
        if attention_rows_probe.ROWS_PROBE is not None and self.cfg.mla:
            rows, positions = attention_rows_probe.ROWS_PROBE
            stats[ATTN_ROWS_STAT] = attention_rows_probe.probe_attention_rows(q, k, mask, rel_bias, rows, positions)
            if rel_bias is not None:
                stats[RELPOS_ROWS_STAT] = attention_rows_probe.relpos_rows(rel_bias, rows, positions)
        # GPU-only kernels; auto-select off-GPU so the model still lowers on CPU.
        attn_impl = self.cfg.gpu_attention if jax.default_backend() == "gpu" else None
        # GDLA: query heads are laid out group by group ([g signal heads, 1 noise head] per group), so each
        # group's value head expands to its g + 1 query heads.
        kernel_v = v

        def _attend(qh: jax.Array, kh: jax.Array) -> jax.Array:
            if fox_key_bias is None:
                return attention(qh, kh, kernel_v, mask, implementation=attn_impl, rel_bias=rel_bias)
            qh, kh, vh = _fox_augment(qh, kh, kernel_v, fox_key_bias)
            return attention(qh, kh, vh, mask, implementation=attn_impl, rel_bias=rel_bias)[..., :head_dim]

        if self.cfg.mla and (0):
            attn_out = mla_sparse_attention(self, x, q, k, kernel_v, rel_bias, sconv_segment_ids, rms_norm, _proj)
        else:
            attn_out = _attend(q, k)[..., :head_dim]
        if self.cfg.attn_xsa:
            # Exclusive Self Attention (XSA): subtract the component of yᵢ parallel to vᵢ, per head.
            # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ.
            aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
            # GPU XSA with GQA can give attn_out a backend-specific head sharding;
            # match v to that dynamic sharding before the per-head projection math.
            aligned_v = reshard(aligned_v, _partition_spec_of(attn_out) or P(_BATCH_AXES, None, None, "model"))
            dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
            v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
            xsa = (dot / (v_norm_sq + 1e-6)) * aligned_v
            attn_out = attn_out - xsa
        gate = None
        if self.attn_gate is not None:
            gate_logit = jnp.einsum("bsd,dn->bsn", x, self.attn_gate).astype(jnp.float32)
            head_keep = _head_gate_keep(self.cfg, jax.nn.sigmoid(gate_logit), stats)
            gate = (2 * jax.nn.sigmoid(gate_logit) * head_keep).astype(attn_out.dtype)
            gate = gate[..., None]
            attn_out = gate * attn_out
        if head_probe_attn is not None:
            if gate is None:
                raise ValueError("head_probe needs attn_output_gate")
            if self.w_o is None:
                raise ValueError("head_probe does not support switchhead_experts or attn_latent_o")
            stats[HEAD_PROBE_STAT] = _head_output_stats(head_probe_attn, gate, attn_out, self.w_o)
        # Merge heads (and SwitchHead experts) into hidden dim while keeping model-axis sharding for w_o.
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:2], math.prod(attn_out.shape[2:])),
            out_sharding=P(_BATCH_AXES, None, "model"),
        )
        assert self.w_o is not None
        # Under an MXFP8 ``fp8_recipe`` the output projection is MXFP8 like q/k/v (as in moe_hero_ep).
        return _proj(self.cfg, attn_out, self.w_o, out_sharding=batch_spec), stats


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
    if isinstance(layer.attn, KimiDeltaAttention) and layer.attn.w_beta is not None:
        # The write-strength projection's magnitude: pinned under MuonH, free under Adam.
        w_beta = jax.lax.stop_gradient(layer.attn.w_beta).astype(jnp.float32)
        stats[f"attn_res_knob_kda_w_beta_norm_L{i}"] = jnp.linalg.norm(w_beta)
    gauss = getattr(layer.attn, "gauss", None)
    if gauss is not None:  # per-head shift and width of the value kernel
        sigma = jax.nn.softplus(gauss.rho.astype(jnp.float32)) + 0.05
        weight = jax.nn.softmax(gauss.mix.astype(jnp.float32), axis=-1)
        for (h, c), _ in np.ndenumerate(np.zeros(gauss.mu.shape)):
            tag = f"L{i}_H{h}" + (f"_C{c}" if gauss.mu.shape[1] > 1 else "")
            stats[f"attn_res_knob_gauss_mu_{tag}"] = jax.lax.stop_gradient(gauss.mu[h, c]).astype(jnp.float32)
            stats[f"attn_res_knob_gauss_sigma_{tag}"] = jax.lax.stop_gradient(sigma[h, c])
            if gauss.mu.shape[1] > 1:
                stats[f"attn_res_knob_gauss_weight_{tag}"] = jax.lax.stop_gradient(weight[h, c])
    router_tok_b = getattr(layer.mlp, "router_tok_b", None)
    if router_tok_b is not None:
        stats[f"attn_res_knob_router_tok_b_norm_L{i}"] = jnp.linalg.norm(
            jax.lax.stop_gradient(router_tok_b).astype(jnp.float32)
        )
    vres = getattr(layer.attn, "vres_lambda", None)
    if vres is not None:
        stats[f"attn_res_knob_vres_l1_L{i}"], stats[f"attn_res_knob_vres_l2_L{i}"] = jax.lax.stop_gradient(vres)
    for name in ("rms_attn", "rms_mlp"):
        norm = getattr(layer, name)
        if isinstance(norm, DyT):
            stats[f"attn_res_knob_dyt_alpha_{name.removeprefix('rms_')}_L{i}"] = jax.lax.stop_gradient(norm.dyt_alpha)
    rot_scale = getattr(layer.attn, "rot_scale", None)
    if rot_scale is not None:
        freq = jnp.abs(jax.lax.stop_gradient(rot_scale).astype(jnp.float32)) * _kda_rot_omega(2 * rot_scale.shape[-1])
        stats[f"attn_res_knob_kda_rot_freq_mean_L{i}"] = jnp.mean(freq)
        stats[f"attn_res_knob_kda_rot_freq_max_L{i}"] = jnp.max(freq)
    if isinstance(layer.mlp, MoEMLP):
        stats.update(_router_knob_stats(layer.mlp, i))
    if isinstance(layer.mlp, MoEMLP) and layer.mlp.cfg.latent_out_dim is not None:
        em = layer.mlp.expert_mlp
        read = [em.w_up] if em.w_gate is None else [em.w_gate, em.w_up]
        stats[f"attn_res_knob_expert_read_pr_L{i}"] = _participation_ratio(read, "eri,esi->rs")
        stats[f"attn_res_knob_expert_write_pr_L{i}"] = _participation_ratio([em.w_down], "eir,eis->rs")
    # Mean |gamma| of the zero-centered gains: the attention / MLP pre-norms, and every norm inside the mixer
    # and MoE (KV latent, KDA output, latent / router norms) plus the Peri-LN output norms.
    groups = {
        "attn": [layer.rms_attn],
        "mlp": [layer.rms_mlp],
        "inner": [layer.attn, layer.mlp, None, None],
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
    stats = {}
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


def _qk_mult_init(cfg: GrugModelConfig, num_heads: int) -> jax.Array | None:
    if not cfg.learnable_qk_mult:
        return None
    return jnp.full((), cfg.qk_mult, jnp.float32)


def _kda_dt_bias_init(cfg: GrugModelConfig, key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
    """``dt_bias`` such that ``|g| = KDA_MIN_LOG_DECAY * sigmoid(dt_bias)`` (zero gate input, ``A_log = 0``)
    is log-uniform in ``kda_dt_range``."""
    lo, hi = cfg.kda_dt_range
    decay = jnp.exp(random.uniform(key, shape, minval=math.log(lo), maxval=math.log(hi)))
    return jax.scipy.special.logit(decay / KDA_MIN_LOG_DECAY)


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


class KdaKernelOptions(NamedTuple):
    """Static options of the KDA kernel call (``_kda_kernel``)."""

    chunk_size: int  # fused (GPU) kernels' chunk length, see KDA_CHUNK_TILES
    save_chunk_states: bool
    state_grads: StateGrads
    state_block_v: int


def _kda_kernel_rotating(q, k, v, g, beta, extras: dict[str, jax.Array], *, options: KdaKernelOptions):
    """``_kda_kernel`` after rotating q and k by ``extras["rot_rate"]`` (``kda_dd_rope``)."""
    extras = dict(extras)
    q, k = _kda_rotate_qk(q, k, extras.pop("rot_rate"), extras.get("segment_ids"))
    return _kda_kernel(q, k, v, g, beta, extras, options=options)


def _kda_kernel(q, k, v, g, beta, extras: dict[str, jax.Array], *, options: KdaKernelOptions):
    """KDA on the model layout ``(B, S, H, d)``: the fused Pallas kernels on GPU, else the XLA ``chunk_kda``
    (heads-first layout, ``KDA_CHUNK_SIZE``). ``extras`` optionally holds ``segment_ids``, the erase gate
    ``erase`` or its logits ``erase_logits``, and ``gate_p`` / ``gate_bias`` (``kda_onchip_decay_gate``:
    ``g`` is then the gate pre-activation of ``-KDA_MIN_LOG_DECAY * sigmoid(gate_p * (g + gate_bias))``)."""
    segment_ids = extras.get("segment_ids")
    gate = (extras["gate_p"], extras["gate_bias"]) if "gate_p" in extras else None
    gate_fn = GateFn(GateKind.SIGMOID, KDA_MIN_LOG_DECAY)
    erase, erase_input = extras.get("erase"), EraseInput.GATE
    if "erase_logits" in extras:
        erase, erase_input = extras["erase_logits"], EraseInput.LOGITS
    if jax.default_backend() == "gpu":
        return kda_fused(
            q,
            k,
            v,
            g,
            beta,
            gate=gate,
            gate_fn=gate_fn,
            segment_ids=segment_ids,
            erase=erase,
            erase_input=erase_input,
            chunk_size=options.chunk_size,
            sub_chunk_size=KDA_CHUNK_TILES[options.chunk_size],
            save_chunk_states=options.save_chunk_states,
            state_grads=options.state_grads,
            state_block_v=options.state_block_v,
        )
    if gate is not None:
        g = gate_log_decay(gate_fn, g.astype(jnp.float32), gate[0], gate[1])
    if erase is not None and erase_input == EraseInput.LOGITS:
        erase = 2.0 * jax.nn.sigmoid(erase.astype(jnp.float32))
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
    No positional encoding (unless ``kda_dd_rope``) and no window.
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
    o_norm: "LearnedRMSNorm"
    sconv_q: ShortConv
    sconv_k: ShortConv
    sconv_v: ShortConv
    ve_lambda: Float[Array, " 2"] | None
    ve_gate: Float[Array, "D N"] | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KimiDeltaAttention":
        k_q, k_k, k_v, k_o, k_g, k_ad, k_au, k_b, k_dt = random.split(key, 9)
        d, n, h, r, std = cfg.hidden_dim, cfg.num_heads, cfg.inferred_head_dim, _KDA_GATE_RANK, cfg.initializer_std
        # kda_dd_rope ties the decay across each rotated channel pair: h/2 decay columns per head.
        decay_cols = h
        return KimiDeltaAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (cfg.kv_in_dim, n * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (cfg.kv_in_dim, n * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std * (1.0)), P("model", _FSDP_AXES)),
            w_g=reshard(
                _init_weight(k_g, (d, (n * h)), std * cfg.init_std_mult_gates),
                P(_FSDP_AXES, "model"),
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
                P(None, "model"),
            ),
            a_log=jnp.zeros((n,)),
            dt_bias=_kda_dt_bias_init(cfg, k_dt, (n, decay_cols)),
            w_beta=reshard(
                _init_weight(
                    k_b,
                    (d, n),
                    std * (cfg.init_std_mult_gates if cfg.init_std_mult_beta is None else cfg.init_std_mult_beta),
                ),
                P(None, None),
            ),
            o_norm=_learned_rms_norm(cfg, h, 1e-6, role="kda_o_norm"),
            sconv_q=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            sconv_k=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            sconv_v=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            ve_lambda=_ve_lambda_init(ValueEmbeds.NONE),
            ve_gate=(reshard(jnp.zeros((d, n)), P(None, None)) if (ValueEmbeds.NONE) in _GATED_VALUE_EMBEDS else None),
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
        token_ids: Int[Array, "B S"] | None = None,
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
        # Every projection of x as one block-scaled GEMM: x is quantized once and the GEMM is 4-6x wider. Only under
        # the real MXFP8 dense recipes, whose per-projection quantization passes dominate at these widths.
        fused: dict[str, jax.Array] = {}
        one_quantization = cfg.mxfp8_shared_input_quant or (
            cfg.kda_fused_projections and cfg.fp8_recipe in _DENSE_FP8_REAL
        )
        if one_quantization and not proj_inputs and kv_input is None:
            named = {
                "q": self.w_q,
                "k": self.w_k,
                "v": self.w_v,
                "g": self.w_g,
                "write": self.w_write,
                "erase": self.w_erase,
            }
            named = {name: w for name, w in named.items() if w is not None}
            # kda_fused_projections: one GEMM over the concatenated weights; mxfp8_shared_input_quant: x quantized
            # once, one GEMM per projection.
            project_all = _proj_parts if cfg.kda_fused_projections else _proj_shared_input
            fused = dict(zip(named, project_all(cfg, x, tuple(named.values())), strict=True))

        def project(w: jax.Array, conv: ShortConv, bias_row: int, name: str) -> jax.Array:
            source = kv_input if kv_input is not None and name in ("k", "v") else proj_inputs.get(name, x)
            y = fused[name] if fused else _proj(cfg, source, w)
            y = jax.nn.silu(conv(y, segment_ids))
            return rearrange(y, "... (n d) -> ... n d", d=head_dim)

        q = project(self.w_q, self.sconv_q, 0, "q")
        k = project(self.w_k, self.sconv_k, 1, "k")
        v = project(self.w_v, self.sconv_v, 2, "v")
        if self.w_write is not None:
            write = fused["write"] if fused else _proj(cfg, proj_inputs.get("v", x), self.w_write)
            v = v * rearrange(
                2.0 * jax.nn.sigmoid(write.astype(jnp.float32)), "... (n d) -> ... n d", d=head_dim
            ).astype(v.dtype)
        a_low = jnp.einsum("bsd,dr->bsr", x, self.w_a_down)
        a = jnp.einsum("bsr,re->bse", a_low, self.w_a_up)
        a = rearrange(a, "... (n d) -> ... n d", d=self.dt_bias.shape[-1])
        scale = jnp.exp(self.a_log.astype(jnp.float32))[:, None]
        onchip_gate = cfg.kda_onchip_decay_gate and not no_decay
        if onchip_gate:  # the kernels form -5 sigmoid(scale (a + dt_bias)) from the bf16 projection
            g = a
        else:
            g = -KDA_MIN_LOG_DECAY * jax.nn.sigmoid(scale * (a.astype(jnp.float32) + self.dt_bias.astype(jnp.float32)))
        assert self.w_beta is not None
        beta_logits = jnp.einsum("bsd,dn->bsn", x, self.w_beta)
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
        if onchip_gate:
            head_spec = P("model", None)
            extras["gate_p"] = reshard(jnp.broadcast_to(scale, self.dt_bias.shape), head_spec)
            extras["gate_bias"] = reshard(self.dt_bias.astype(jnp.float32), head_spec)
            extra_specs["gate_p"] = extra_specs["gate_bias"] = head_spec
        if self.w_erase is not None:
            z = fused["erase"] if fused else _proj(cfg, proj_inputs.get("k", x), self.w_erase)
            z = rearrange(z, "... (n d) -> ... n d", d=head_dim)
            erase_logits = z.astype(jnp.float32)
            if cfg.kda_onchip_erase_gate:
                extras["erase_logits"], extra_specs["erase_logits"] = reshard(z, spec4), spec4
            else:
                extras["erase"], extra_specs["erase"] = reshard(2.0 * jax.nn.sigmoid(erase_logits), spec4), spec4
            # b / 2 = sigmoid(z): H = -p log p - (1 - p) log(1 - p), max log 2 (the zero-init value).
            p_erase = jax.nn.sigmoid(erase_logits)
            entropy = -(p_erase * jax.nn.log_sigmoid(erase_logits) + (1 - p_erase) * jax.nn.log_sigmoid(-erase_logits))
            stats[f"{_KDA_ERASE_STAT_PREFIX}mean"] = jax.lax.stop_gradient(jnp.mean(2.0 * p_erase))
            stats[f"{_KDA_ERASE_STAT_PREFIX}entropy"] = jax.lax.stop_gradient(jnp.mean(entropy))
        if segment_ids is not None:
            extras["segment_ids"] = reshard(jnp.broadcast_to(segment_ids, (b, s)), P(_BATCH_AXES, None))
            extra_specs["segment_ids"] = P(_BATCH_AXES, None)
        if "rot_rate" in extras:
            kernel_fn = _kda_kernel_rotating
        else:
            kernel_fn = _kda_kernel
        options = KdaKernelOptions(
            cfg.kda_chunk_size, cfg.kda_save_chunk_states, cfg.kda_state_grads, cfg.kda_state_block_v
        )
        run = functools.partial(kernel_fn, options=options)
        args = (q, k, v, g, beta, extras)
        in_specs = (spec4,) * 4 + (spec3, extra_specs)
        # The Pallas custom VJPs are not vma-annotated, so skip the varying-axes check.
        kernel = jax.shard_map(run, mesh=get_abstract_mesh(), in_specs=in_specs, out_specs=spec4, check_vma=False)
        o = kernel(*args)
        o = self.o_norm(o.astype(x.dtype))
        o = jnp.reshape(o, (b, s, cfg.num_heads * head_dim), out_sharding=P(_BATCH_AXES, None, "model"))
        gate = jax.nn.sigmoid(fused["g"] if fused else _proj(cfg, x, self.w_g))
        gate_heads = rearrange(gate, "... (n d) -> ... n d", d=head_dim)
        head_keep = _head_gate_keep(cfg, jnp.mean(gate_heads.astype(jnp.float32), axis=-1), stats)
        if cfg.attn_head_skip:
            gate_heads = gate_heads * head_keep[..., None].astype(gate.dtype)
            gate = rearrange(gate_heads, "... n d -> ... (n d)")
        o = o * gate
        out = _proj(cfg, o, self.w_o, out_sharding=_batch_spec())
        return out, stats


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)
    gain_fn: NormGainFn = eqx.field(static=True, default=NormGainFn.LINEAR)
    """How the gain is read from ``weight`` (``norm_gain_fn``)."""

    @staticmethod
    def init(dim: int, eps: float, gain_fn: NormGainFn = NormGainFn.LINEAR) -> "RMSNorm":
        return RMSNorm(weight=jnp.full((dim,), _GAIN_INIT[gain_fn], dtype=jnp.float32), eps=eps, gain_fn=gain_fn)

    def gain(self) -> jax.Array:
        return apply_gain_fn(self.gain_fn, unshard(self.weight))

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = self.gain()
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


def _learned_rms_norm(cfg: GrugModelConfig, dim: int, eps: float, role: str | None = None) -> LearnedRMSNorm:
    """A learned-gain RMSNorm, zero-centered under ``cfg.zero_centered_gains``."""
    return RMSNorm.init(dim, eps, _norm_gain_overrides(cfg).get(role, (NormGainFn.LINEAR)))


def _norm_gain_overrides(cfg: "GrugModelConfig") -> dict[str, NormGainFn]:
    """``norm_gain_overrides`` as ``{role: NormGainFn}``; raises on a malformed entry or an unknown function."""
    out = {}
    return out


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


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array | None
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    w_latent_down: jax.Array | None
    latent_norm: LearnedRMSNorm | None
    w_latent_up: jax.Array | None
    latent_out_norm: LearnedRMSNorm | None
    latent_mix_in_gate: Float[Array, "D E"] | None  # mixture of latents on the input latent (latent_mix_sites)
    expert_private_proj: Float[Array, "D GR"] | None  # cfg.expert_private_dim: one [D, r] projection per group
    expert_private_norm: "LearnedRMSNorm | None"
    latent_width_gate: Float[Array, "D B"] | None  # cfg.latent_matryoshka_blocks: per-token latent width gate
    latent_width_bias: Float[Array, " B"] | None  # sign-SGD selection bias balancing the widths
    latent_mix_out_gate: Float[Array, "D E"] | None  # mixture of latents on the experts' output latent
    null_gain: Float[Array, ""] | None  # cfg.moe_null_learned_gain (every layer carries it; null layers read it)
    cfg: GrugModelConfig = eqx.field(static=True)
    latent_selects: bool = eqx.field(static=True, default=False)
    """This layer forms its expert input by ``latent_select`` (see ``latent_select_layers``)."""

    def _mix_latent(
        self, latent: Float[Array, "T R"], x_flat: Float[Array, "T D"], norm: LearnedRMSNorm, gate: jax.Array
    ) -> Float[Array, "T R"]:
        """Mixture of latents: norm each of the ``E`` latent blocks and weight it by its top-k gate from ``x``."""
        weights = mixture_weights(x_flat, gate, (4), (2))
        normed = norm(rearrange(latent, "t (e r) -> t e r", e=(4)))
        return rearrange(normed * weights[..., None].astype(normed.dtype), "t e r -> t (e r)")

    def _private_rows(self, x_flat: Float[Array, "T D"], selected: Int[Array, "T K"]) -> jax.Array | None:
        """Each assignment's private input slice ``RMSNorm(x P_g)``, ``g = expert mod expert_private_groups``."""
        if self.expert_private_proj is None:
            return None
        assert self.expert_private_norm is not None
        groups, r = (64), (0)
        proj = jnp.einsum("td,dr->tr", x_flat, self.expert_private_proj.astype(x_flat.dtype))
        proj = jnp.reshape(proj, (proj.shape[0], groups, r), out_sharding=P(_BATCH_AXES, None, None))
        rows = jnp.take_along_axis(proj, (selected % groups)[:, :, None], axis=1)
        return self.expert_private_norm(rows)

    def _matryoshka_latent(
        self, latent: Float[Array, "T R"], x_flat: Float[Array, "T D"], stats: dict[str, jax.Array]
    ) -> Float[Array, "T R"]:
        """Keep each token's gate-chosen number of leading latent blocks. The chosen gate enters as
        ``exp(log_sigmoid(l) - stop_gradient(log_sigmoid(l)))`` (value 1), so the gate learns from the loss without
        rescaling the latent and without the 0/0 a saturated sigmoid would give."""
        assert self.latent_width_gate is not None and self.latent_width_bias is not None
        logits = jnp.einsum("td,db->tb", x_flat, self.latent_width_gate).astype(jnp.float32)
        width = jnp.argmax(logits + jax.lax.stop_gradient(self.latent_width_bias), axis=-1)  # keep blocks 0..width
        chosen = jax.nn.one_hot(width, (0), dtype=jnp.float32)
        bias = _load_error_grad(self.latent_width_bias, jax.lax.stop_gradient(jnp.mean(chosen, axis=0)))
        log_w = jnp.sum(jax.nn.log_sigmoid(logits) * chosen, axis=-1, keepdims=True) + 0.0 * jnp.sum(bias)
        scale = jnp.exp(log_w - jax.lax.stop_gradient(log_w)).astype(latent.dtype)
        kept = (jnp.arange(0)[None, :] <= width[:, None]).astype(latent.dtype)
        mask = jnp.repeat(kept, latent.shape[-1] // (0), axis=-1)
        stats[f"{_LAYER_KNOB_PREFIX}latent_kept_frac"] = jax.lax.stop_gradient(jnp.mean(kept.astype(jnp.float32)))
        return latent * mask * scale

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
        mix_in, mix_out = False, False
        k_priv, k_width = random.split(random.fold_in(key, 11))
        r_priv, g_priv, b_width = (0), (64), (0)
        return MoEMLP(
            router=reshard(
                _init_weight(k_router, (d, e), cfg.initializer_std),
                P(None, None),
            ),
            router_bias=jnp.zeros((e,)),
            w_latent_down=(
                None
                if latent is None
                else reshard(_latent_proj_init(cfg, k_down, (d, latent * 1)), P(_FSDP_AXES, "model"))
            ),
            latent_norm=(
                None
                if latent is None
                else (
                    _grouped_rms_norm(cfg, (4), latent // (4))
                    if mix_in
                    else _learned_rms_norm(cfg, latent, cfg.layer_norm_eps, role="moe_latent")
                )
            ),
            w_latent_up=(
                reshard(_latent_proj_init(cfg, k_up, (out_width, d)), P("model", _FSDP_AXES))
                if cfg.has_latent_up
                else None
            ),
            latent_out_norm=(_grouped_rms_norm(cfg, (4), out_width // (4)) if mix_out else None),
            expert_private_proj=(
                reshard(_init_weight(k_priv, (d, g_priv * r_priv), 1.0 / math.sqrt(d)), P(None, None))
                if r_priv
                else None
            ),
            expert_private_norm=_learned_rms_norm(cfg, r_priv, cfg.layer_norm_eps) if r_priv else None,
            latent_width_gate=(
                reshard(_init_weight(k_width, (d, b_width), cfg.initializer_std), P(None, None)) if b_width else None
            ),
            latent_width_bias=jnp.zeros((b_width,), jnp.float32) if b_width else None,
            latent_mix_in_gate=(
                reshard(_init_weight(random.fold_in(k_down, 1), (d, (4)), cfg.initializer_std), P(None, None))
                if mix_in
                else None
            ),
            latent_mix_out_gate=(
                reshard(_init_weight(random.fold_in(k_up, 1), (d, (4)), cfg.initializer_std), P(None, None))
                if mix_out
                else None
            ),
            null_gain=jnp.ones((), jnp.float32) if cfg.moe_null_learned_gain else None,
            expert_mlp=_expert_mlp_init(
                _bank_config(cfg, 1),
                expert_width,
                out_width,
                k_expert,
            ),
            cfg=cfg,
            latent_selects=(False),
        )

    def input_projection_weights(self, dtype: jnp.dtype) -> list[jax.Array]:
        """The ``[D, *]`` projections this MLP applies to its input: the router (unless ``router_on_latent``), then
        the latent down."""
        weights = []
        router_in = self.router if self.router is not None else None
        assert router_in is not None
        weights.append(reshard(router_in, P(None, None)))
        if self.w_latent_down is not None:
            weights.append(reshard(self.w_latent_down.astype(dtype), P(None, None)))
        return weights

    def simbal_loss(self) -> jax.Array:
        """SimBal router orthogonality loss ``||R^T R - I||_1`` over the real-expert columns (``simbal_loss_weight``)."""
        assert self.router is not None
        r = reshard(self.router[:, : self.cfg.num_experts].astype(jnp.float32), P(None, None))
        gram = jnp.einsum("de,df->ef", r, r, out_sharding=P(None, None))
        return jnp.sum(jnp.abs(gram - jnp.eye(self.cfg.num_experts, dtype=jnp.float32)))

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
        loss = jnp.mean((jax.nn.relu(m - (1.0) * diag[:, None]) + jax.nn.relu(m - (1.0) * diag[None, :])) * off)
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

        With ``moe_null_target_frac`` every column is balanced: the real experts share ``2K (1 - f)`` slots per
        token, the zero-computation experts ``2K f``. Without it the zero-computation columns get beta 0 (no bias,
        free to use) and the real experts are balanced at this step's measured real load, with their mean beta
        removed so QB never moves the real-vs-null split.
        """
        num_real, num_null = self.cfg.num_experts, self.cfg.num_null_experts
        margins = reshard(margins, P(_BATCH_AXES, None))
        frac = self.cfg.moe_null_target_frac
        if frac is not None:
            k = selected_experts.shape[-1]
            share = jnp.concatenate(
                [jnp.full((num_real,), k * (1.0 - frac) / num_real), jnp.full((num_null,), k * frac / num_null)]
            )
            return qb_beta_hist(
                margins,
                mesh,
                target_share=share,
                num_experts=num_real + num_null,
                n_bins=_QB_HIST_BINS,
                batch_axes=_BATCH_AXES,
            )
        real_slots = jax.lax.stop_gradient(jnp.mean(jnp.sum(selected_experts < num_real, axis=-1, dtype=jnp.float32)))
        beta, lo, hi = qb_beta_hist(
            margins[:, :num_real],
            mesh,
            target_share=real_slots / num_real,
            num_experts=num_real,
            n_bins=_QB_HIST_BINS,
            batch_axes=_BATCH_AXES,
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
        copy_start = num_real + (0)
        const_start = copy_start + (0)
        is_null = selected_experts >= num_real
        t, k = selected_experts.shape
        slot_ids = jax.lax.broadcasted_iota(jnp.int32, (t, k), 0) * k + jax.lax.broadcasted_iota(jnp.int32, (t, k), 1)
        spread = reshard(slot_ids % num_real, _partition_spec_of(selected_experts))
        selected = jnp.where(is_null, spread, selected_experts)
        weights = jnp.where(is_null, jnp.zeros_like(combine_weights), combine_weights)

        w = combine_weights.astype(jnp.float32)
        _partition_spec_of(routed_input)
        x = routed_input.astype(jnp.float32)
        is_copy = (selected_experts >= copy_start) & (selected_experts < const_start)
        x_scale = jnp.sum(jnp.where(is_copy, w, 0.0), axis=-1)
        null_out = jnp.zeros_like(x)
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
        router_embed: Float[Array, "B S D"] | None = None,
        null_routing: bool = False,
        saved_routed: Float[Array, "T L"] | None = None,
        moe_k_keep: Int[Array, "B S"] | None = None,
        moe_dynamic_layer: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``router_embed`` is the token embedding the router reads under ``router_on_embed``.
        ``moe_k_keep`` (a ``moe_dynamic_layers`` layer) is each token's routed-expert count (``MoeDynamicK``).
        ``saved_routed`` is this call's combined expert output from the forward (``scan_save_routed_output``):
        the recompute returns it, with the gradient still flowing through the experts.
        ``null_routing`` (a ``moe_null_layers`` layer) routes top-2K over the real and null columns.
        ``projected`` holds ``x_flat @ w`` for each of ``input_projection_weights`` when the caller
        computed them already (fused with other projections of the same input). ``hash_token_ids`` picks
        the experts by token-id hash (``moe_hash_layers``); ``noise_key`` adds the training-only Gumbel
        noise of ``moe_gumbel_tau`` to the expert selection; ``router_tok_rows`` are the tokens' rows of the
        shared ``router_token_bias_rank`` table and ``router_seed_bias`` the ``router_bias_seed`` logit bias.
        ``overlap`` is ``[T, D]`` token-local work run under the first bank's dispatch all-to-all
        (``SharedExpertPlacement.OVERLAP``) and added to the output."""
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        if projected is None:
            projected = [jnp.einsum("td,de->te", x_flat, w) for w in self.input_projection_weights(x_flat.dtype)]
        router_logits = projected[0].astype(jnp.float32)
        if router_seed_bias is not None:
            seed_bias = rearrange(router_seed_bias, "b s e -> (b s) e")
            router_logits = router_logits + reshard(seed_bias, _partition_spec_of(router_logits))
        relu_layer = moe_dynamic_layer and self.cfg.moe_dynamic == MoeDynamicK.RELU
        if relu_layer and self.cfg.moe_relu_l1:
            router_logits = _relu_l1_grad(router_logits, self.cfg.moe_relu_l1, self.cfg.moe_relu_offset)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        k = self.cfg.num_experts_per_token
        banks = _expert_banks(self.cfg)
        num_real = self.cfg.num_experts
        null_layer = bool(self.cfg.num_null_experts) and null_routing
        if null_layer:
            # The zero-computation experts compete in the one bank's top-2K.
            k = 2 * k
            banks = [(0, num_real + self.cfg.num_null_experts, k)]
        # Select top-(K+1) on biased logits per bank; the (K+1)-th is that bank's QB threshold alpha.
        bank_selected, bank_alpha = [], []
        for start, size, bank_k in banks:
            topk_logits, sel = _small_top_k(biased_logits[:, start : start + size], bank_k + 1)
            bank_alpha.append(topk_logits[:, -1:])
            bank_selected.append(sel[:, :-1] + start)
        selected_experts = jnp.concatenate(bank_selected, axis=-1) if len(banks) > 1 else bank_selected[0]
        if len(banks) > 1 and (hash_token_ids is not None):
            raise ValueError("moe_bank2_experts does not support Gumbel routing, hash routing or the dense router grad")
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
        renorm_sum = self.cfg.routing_renorm_sum * (self.cfg.moe_null_renorm_scale if null_layer else 1.0)
        if self.cfg.router_combine == RouterCombine.SOFTMAX_RENORM:
            combine_weights_f = renorm_sum * jax.nn.softmax(unbiased_topk, axis=-1)
        elif self.cfg.router_combine == RouterCombine.SIGMOID_RAW:
            combine_weights_f = jax.nn.sigmoid(unbiased_topk) * (renorm_sum / (k / 2))
        elif self.cfg.router_combine == RouterCombine.SQRT_SOFTPLUS_RAW:
            gate = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
            combine_weights_f = gate * (renorm_sum / (k * math.sqrt(math.log(2.0))))
        else:
            if self.cfg.router_combine == RouterCombine.SQRT_SOFTPLUS_RENORM:
                # The floor keeps sqrt's gradient finite if softplus underflows to 0 (logit below about -100).
                combine_weights_f = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
            else:
                combine_weights_f = jax.nn.sigmoid(unbiased_topk)
            denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
            combine_weights_f = combine_weights_f * (renorm_sum / (denom + 1e-9))
        active_k = jnp.asarray(float(k), jnp.float32)
        if moe_k_keep is not None:
            eps_layer = moe_dynamic_layer and self.cfg.moe_dynamic == MoeDynamicK.EPS
            if relu_layer:
                # ReMoE-style: the ReLU itself is the (unnormalized) combine weight, so pushing logits down costs loss.
                shifted = unbiased_topk + self.cfg.moe_relu_offset
                keep = shifted > 0
                combine_weights_f = jax.nn.relu(shifted) * (renorm_sum / (k * max(self.cfg.moe_relu_offset, 1.0)))
            elif moe_dynamic_layer and self.cfg.moe_dynamic == MoeDynamicK.THRESHOLD:
                flat = reshard(jax.lax.stop_gradient(unbiased_topk), P(None, None))
                keep = unbiased_topk >= jnp.quantile(flat, 1.0 - self.cfg.moe_dynamic_frac)
            elif eps_layer:
                keep = combine_weights_f >= self.cfg.moe_eps * renorm_sum
            elif moe_dynamic_layer and self.cfg.moe_dynamic == MoeDynamicK.TOP_P:
                w = jax.lax.stop_gradient(combine_weights_f)
                # Weight of the slots ranked above each slot (ties broken by slot index).
                slot = jax.lax.broadcasted_iota(jnp.int32, w.shape, 1)
                above = (w[:, None, :] > w[:, :, None]) | (
                    (w[:, None, :] == w[:, :, None]) & (slot[:, None, :] < slot[:, :, None])
                )
                mass_above = jnp.sum(jnp.where(above, w[:, None, :], 0.0), axis=-1)
                keep = mass_above < self.cfg.moe_top_p * jnp.sum(w, axis=-1, keepdims=True)
            else:
                slot = jax.lax.broadcasted_iota(jnp.int32, unbiased_topk.shape, 1)
                keep = slot < rearrange(moe_k_keep, "b s -> (b s)")[:, None]
            combine_weights_f = jnp.where(keep, combine_weights_f, 0.0)
            renorms = (RouterCombine.SOFTMAX_RENORM, RouterCombine.SIGMOID_RENORM, RouterCombine.SQRT_SOFTPLUS_RENORM)
            if self.cfg.router_combine in renorms and not (eps_layer or relu_layer):
                denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
                combine_weights_f = combine_weights_f * (renorm_sum / (denom + 1e-9))
            active_k = jnp.mean(jnp.sum(keep.astype(jnp.float32), axis=-1))
        if router_soft_select.SOFT_SELECT is not None:
            biased_sel = jnp.take_along_axis(router_logits + self.router_bias, selected_experts, axis=-1)
            combine_weights_f = router_soft_select.soft_combine(combine_weights_f, biased_sel)
        combine_weights = combine_weights_f.astype(x.dtype)
        mesh = get_abstract_mesh()
        # Per-token assignments for the routing dump (``Transformer.routing_assignments``); unused (and so
        # dead-code eliminated) in the training forward.
        assignments = {_ROUTING_SELECTED: selected_experts.astype(jnp.int32), _ROUTING_WEIGHTS: combine_weights_f}
        if self.cfg.num_null_experts and not null_layer:
            # moe_null_layers route 2K slots; pad this layer's K to 2K (expert -1, weight 0) so layers stack.
            pad = ((0, 0), (0, 2 * self.cfg.num_experts_per_token - k))
            assignments = {
                _ROUTING_SELECTED: jnp.pad(assignments[_ROUTING_SELECTED], pad, constant_values=-1),
                _ROUTING_WEIGHTS: jnp.pad(assignments[_ROUTING_WEIGHTS], pad),
            }
        # Per-shard partials only; the cross-device reduction happens once after the layer scan.
        router_stats = local_routing_stats(
            reshard(selected_experts, P(_BATCH_AXES, None)),
            reshard(router_logits, P(_BATCH_AXES, None)),
            mesh,
            num_experts=num_real + self.cfg.num_null_experts,
            batch_axes=_BATCH_AXES,
            token_stride=(1),
        )
        # Sharded QB: estimate each expert's threshold beta from the margins `s - alpha` by binning
        # them into fixed bins over the live global range and reading the (1-K/E) quantile.
        if null_layer:
            bank_stats = [self._null_qb_stats(router_logits - bank_alpha[0], selected_experts, mesh)]
        else:
            bank_stats = [
                qb_beta_hist(
                    reshard(router_logits[:, start : start + size] - alpha, P(_BATCH_AXES, None)),
                    mesh,
                    target_share=bank_k / size,
                    num_experts=size,
                    n_bins=_QB_HIST_BINS,
                    batch_axes=_BATCH_AXES,
                )
                for (start, size, bank_k), alpha in zip(banks, bank_alpha, strict=True)
            ]
        beta = jnp.concatenate([b for b, _, _ in bank_stats], axis=-1)
        if self.cfg.num_null_experts and not null_layer:
            # A real-only layer leaves its (unread) null columns' bias at zero.
            beta = jnp.concatenate([beta, jnp.zeros((self.cfg.num_null_experts,), beta.dtype)])
        margin_min = functools.reduce(jnp.minimum, [lo for _, lo, _ in bank_stats])
        margin_max = functools.reduce(jnp.maximum, [hi for _, _, hi in bank_stats])
        router_stats["qb_beta"] = beta
        router_stats[f"{_OUTSTAT_PREFIX}moe_active_k"] = active_k
        router_stats["margin_min"] = margin_min
        router_stats["margin_max"] = margin_max

        routed_input = x_flat
        if self.latent_selects:
            assert self.cfg.latent_dim is not None and self.latent_norm is not None
            selected = x_flat[..., : self.cfg.latent_dim]
            if self.w_latent_down is not None:
                selected = selected + reshard(projected[1], _batch_spec())
            routed_input = self.latent_norm(selected)
        elif self.w_latent_down is not None and self.latent_norm is not None:
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = reshard(projected[1], _batch_spec())
            if self.latent_mix_in_gate is None:
                routed_input = self.latent_norm(routed_input)
            else:
                routed_input = self._mix_latent(routed_input, x_flat, self.latent_norm, self.latent_mix_in_gate)
        if self.latent_width_gate is not None:
            routed_input = self._matryoshka_latent(routed_input, x_flat, router_stats)
        bank_mlps = [self.expert_mlp]
        real_selected, real_weights, null_out = selected_experts, combine_weights, None
        if null_layer:
            real_selected, real_weights, null_out = self._split_null_slots(
                routed_input, selected_experts, combine_weights
            )
        outputs, overflows, bank_keeps, col = [], [], [], 0
        overlap_out = None
        for (start, _, bank_k), em, bank_index in zip(banks, bank_mlps, (1, 2), strict=False):
            bank_selected = (real_selected[:, col : col + bank_k] - start).astype(jnp.int32)
            if self.expert_private_proj is not None:
                out, overflow, *bank_overlap_out = _run_grouped_read_bank(
                    em,
                    _bank_config(self.cfg, bank_index),
                    None,
                    routed_input,
                    bank_selected,
                    real_weights[:, col : col + bank_k],
                    overlap,
                    self._private_rows(x_flat, bank_selected),
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
            outputs.append(out)
            overflows.append(overflow)
            col += bank_k
        routed_flat = functools.reduce(jnp.add, outputs)
        if saved_routed is not None:
            routed_flat = _saved_value(routed_flat, saved_routed.astype(routed_flat.dtype))
        if self.cfg.scan_save_routed_output:
            router_stats[_ROUTED_COMBINE] = routed_flat
        if null_layer and self.null_gain is not None:
            routed_flat = routed_flat * self.null_gain.astype(routed_flat.dtype)
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
        dropped_assignments = capacity_overflow.dropped
        sender_dropped_assignments = capacity_overflow.sender_dropped
        receiver_dropped_assignments = capacity_overflow.receiver_dropped
        router_stats["capacity_overflow"] = dropped_assignments
        router_stats["sender_capacity_overflow"] = sender_dropped_assignments
        router_stats["receiver_capacity_overflow"] = receiver_dropped_assignments

        # Expand after the combine: `expert_mlp` already returns the weight-summed expert output,
        # which is the vector the paper's W_up acts on.
        if self.cfg.routed_write_extra:
            # The lm_head-only slice: it never enters the residual stream.
            hidden_dim = self.cfg.hidden_dim
            router_stats[_LM_HEAD_EXTRA] = rearrange(routed_flat[:, hidden_dim:], "(b s) d -> b s d", b=b, s=s)
            routed_flat = routed_flat[:, :hidden_dim]
        if self.latent_mix_out_gate is not None:
            assert self.latent_out_norm is not None
            routed_flat = self._mix_latent(routed_flat, x_flat, self.latent_out_norm, self.latent_mix_out_gate)
        elif self.latent_out_norm is not None:
            routed_flat = self.latent_out_norm(routed_flat)
        if self.w_latent_up is not None:
            routed_flat = _proj(
                self.cfg, routed_flat, self.w_latent_up.astype(routed_flat.dtype), out_sharding=_batch_spec()
            )
        if overlap_out is not None:
            routed_flat = routed_flat + reshard(overlap_out, _batch_spec()).astype(routed_flat.dtype)

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, {**router_stats, **assignments, _ROUTED_INPUT: routed_input}


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
        slope = 0.0
        hidden = jnp.concatenate([jnp.square(jax.nn.leaky_relu(u, slope)) for u in ups], axis=1)
    shared_out = _proj(cfg, hidden, w_down, out_sharding=out_sharding)
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
    """The shared experts on one EP shard's tokens (``SharedExpertPlacement.OVERLAP``): ``tokens`` is
    (shared-expert input, MoE input), ``params`` the concatenated gate/up weights, the concatenated down weights
    and the gate."""
    shared_in, x_flat = tokens
    w_gate_up, w_down, shared_gate = params
    proj = _proj(cfg, shared_in, w_gate_up)
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
    shadow: DenseMLP | None = None,
    router_embed: Float[Array, "B S D"] | None = None,
    null_routing: bool = False,
    saved_routed: Float[Array, "T L"] | None = None,
    moe_k_keep: Int[Array, "B S"] | None = None,
    moe_dynamic_layer: bool = False,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """Routed MoE plus the shared SwiGLU experts with every projection of ``x`` in one GEMM.

    The router, latent down-projection and each shared expert's gate/up read the same input, so they
    run as one ``[D, sum widths]`` GEMM; the shared experts' down-projections run as one GEMM over
    their concatenated hidden units (the sum over shared experts happens in its accumulator). Same
    math as ``mlp(x) + sum(expert(x) for expert in shared)`` (~3% faster at d512); parameters stay
    separate leaves. ``moe_shared_placement`` can move the shared experts out of the fused GEMM
    (``SharedExpertPlacement``).
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
    if mlp.cfg.moe_dispatch == MoeDispatch.HERO_LEVANTER:
        if part_inputs or shadow is not None:
            raise ValueError("moe_dispatch=hero_levanter reads one MLP input and has no moe_shadow_width")
        return _moe_and_shared_hero(
            mlp,
            shared,
            x,
            x_flat,
            gate,
            dict(
                hash_token_ids=hash_token_ids,
                noise_key=noise_key,
                router_tok_rows=router_tok_rows,
                router_seed_bias=router_seed_bias,
                router_embed=router_embed,
                null_routing=null_routing,
                saved_routed=saved_routed,
                moe_k_keep=moe_k_keep,
                moe_dynamic_layer=moe_dynamic_layer,
            ),
        )
    placement = mlp.cfg.moe_shared_placement
    overlap = None
    if placement == SharedExpertPlacement.OVERLAP:
        shared_in = (
            rearrange(part_inputs["shared"], "b s d -> (b s) d") if part_inputs and "shared" in part_inputs else x_flat
        )
        # FSDP-sharded weights: the EP shard_map all-gathers them, so their gradients leave it reduce-scattered.
        rows, cols = P(_FSDP_AXES, None), P(None, _FSDP_AXES)
        overlap_up = [reshard(e.w_gate, rows) for e in shared if gated] + [reshard(e.w_up, rows) for e in shared]
        overlap = MoeOverlapWork(
            functools.partial(
                _shared_experts_local, cfg=mlp.cfg, widths=tuple(w.shape[1] for w in shared_weights), gated=gated
            ),
            (shared_in, x_flat),
            (
                jnp.concatenate(overlap_up, axis=1),
                jnp.concatenate([reshard(e.w_down, cols) for e in shared], axis=0),
                gate,
            ),
        )
    unfused = placement == SharedExpertPlacement.UNFUSED and not part_inputs
    weights = moe_weights if unfused or overlap is not None else moe_weights + shared_weights
    router_in_fused = bool(moe_weights)
    if not part_inputs and mlp.cfg.fp8_recipe in _DENSE_FP8_SIM + _DENSE_FP8_REAL:
        # The router stays in high precision; the rest of the fused projection takes the low-precision recipe.
        low = weights[1:] if router_in_fused else weights
        fused = _proj(mlp.cfg, x_flat, jnp.concatenate(low, axis=1), out_sharding=_batch_spec())
        parts = jnp.split(fused, list(itertools.accumulate(w.shape[1] for w in low[:-1])), axis=1)
        if router_in_fused:
            router = jnp.einsum("td,de->te", x_flat, weights[0], out_sharding=_batch_spec())
            parts = [router, *parts]
    elif not part_inputs:
        parts = list(_fused_projection(x_flat, tuple(weights)))
    else:
        # Some projections read another stream (attn_res_sum_inputs): one GEMM per projection.
        moe_names = ["router", "latent"]
        names = moe_names[: len(moe_weights)] + ["shared"] * (len(weights) - len(moe_weights))
        flats = {k: rearrange(v, "b s d -> (b s) d") for k, v in part_inputs.items()}
        parts = [
            jnp.einsum("td,de->te", flats.get(n, x_flat), w, out_sharding=_batch_spec())
            for n, w in zip(names, weights, strict=True)
        ]

    def routed_mlp():
        return mlp(
            part_inputs.get("latent", x) if part_inputs else x,
            projected=parts[: len(moe_weights)],
            hash_token_ids=hash_token_ids,
            noise_key=noise_key,
            router_tok_rows=router_tok_rows,
            router_seed_bias=router_seed_bias,
            overlap=overlap,
            router_embed=router_embed,
            null_routing=null_routing,
            saved_routed=saved_routed,
            moe_k_keep=moe_k_keep,
            moe_dynamic_layer=moe_dynamic_layer,
        )

    routed, stats = routed_mlp()
    if overlap is not None:
        if shadow is not None:
            raise ValueError("moe_shadow_width / moe_compress need moe_shared_placement other than overlap")
        return routed, stats
    if unfused:
        parts = [*parts, *(jnp.einsum("td,de->te", x_flat, w, out_sharding=_batch_spec()) for w in shared_weights)]
    if not gated:
        # Ungated shared experts: these parts are their pre-activations (``Transformer.neuron_inputs``).
        stats = {**stats, _SHARED_PRE: jnp.concatenate(parts[len(moe_weights) :], axis=-1)}
    shared_out = _shared_experts_tail(
        mlp.cfg, parts[len(moe_weights) :], len(shared), gated, w_down, x_flat, gate, _batch_spec()
    )
    shared_out = _batch_reshard(rearrange(shared_out, "(b s) d -> b s d", b=b, s=s))
    if shadow is not None:
        shared_in = part_inputs["shared"] if part_inputs and "shared" in part_inputs else x
        # The transfer and gap terms read the shared expert on a stop-gradient input so they move only it.
        detached = rearrange(jax.lax.stop_gradient(shared_in), "b s d -> (b s) d")
        detached_parts = [jnp.einsum("td,de->te", detached, w, out_sharding=_batch_spec()) for w in shared_weights]
        shared_detached = _shared_experts_tail(
            mlp.cfg, detached_parts, len(shared), gated, w_down, detached, gate, _batch_spec()
        )
        shared_detached = _batch_reshard(rearrange(shared_detached, "(b s) d -> b s d", b=b, s=s))
        stats = {**stats, **_compress_terms(mlp.cfg, shared_out, shared_detached, routed, shadow, shared_in)}
    return routed + shared_out, stats


def _moe_and_shared_hero(
    mlp: MoEMLP,
    shared: tuple[DenseMLP, ...],
    x: Float[Array, "B S D"],
    x_flat: Float[Array, "T D"],
    shared_gate: Float[Array, " D"] | None,
    routed_kwargs: dict[str, Any],
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """``moe_and_shared_fused`` under ``MoeDispatch.HERO_LEVANTER``, as ``moe_hero_ep``'s ``MoEMLP`` plus
    ``_shared_experts_fused``: separate projections on the FSDP-sharded weights and no pinned overlap work."""
    cfg = mlp.cfg
    b, s, _ = x.shape
    router_weights = mlp.input_projection_weights(x_flat.dtype)[0]
    projected = [jnp.einsum("td,de->te", x_flat, router_weights, out_sharding=_batch_spec())]
    if mlp.w_latent_down is not None:
        projected.append(_proj(cfg, x_flat, mlp.w_latent_down.astype(x_flat.dtype), out_sharding=_batch_spec()))
    routed, stats = mlp(x, projected=projected, **routed_kwargs)
    gated = shared[0].w_gate is not None
    up_weights = [e.w_gate for e in shared if gated] + [e.w_up for e in shared]
    # Token-sharded like ``x_flat``: left to the projection, the output takes the weights' column sharding and
    # the down projection reshards it back with an all-to-all.
    up = _proj(cfg, x_flat, jnp.concatenate(up_weights, axis=1), out_sharding=_batch_spec())
    parts = jnp.split(up, list(itertools.accumulate(w.shape[1] for w in up_weights[:-1])), axis=-1)
    if not gated:
        stats = {**stats, _SHARED_PRE: jnp.concatenate(parts, axis=-1)}
    w_down = jnp.concatenate([e.w_down for e in shared], axis=0)
    shared_out = _shared_experts_tail(cfg, parts, len(shared), gated, w_down, x_flat, shared_gate, _batch_spec())
    shared_out = _batch_reshard(rearrange(shared_out, "(b s) d -> b s d", b=b, s=s))
    return routed + shared_out, stats


def _compress_terms(
    cfg: "GrugModelConfig",
    shared_out: Float[Array, "B S D"],
    shared_detached: Float[Array, "B S D"],
    routed: Float[Array, "B S D"],
    shadow: DenseMLP | None,
    shared_in: Float[Array, "B S D"],
) -> dict[str, jax.Array]:
    """Per-layer ``moe_compress`` / ``moe_shadow_width`` loss (``_COMPRESS_LOSS``) and their metrics.
    ``shared_detached`` is the shared expert on a stop-gradient input. The routed penalty reaches the routed
    output's inputs too, but only in direction: the routed path RMS-normalizes its input."""
    s32, r32 = shared_out.astype(jnp.float32), routed.astype(jnp.float32)
    y_energy = jax.lax.stop_gradient(jnp.mean(jnp.sum((s32 + r32) ** 2, axis=-1)))
    r_energy = jnp.mean(jnp.sum(r32**2, axis=-1))
    sr = jnp.mean(jnp.sum(s32 * r32, axis=-1))
    s_energy = jnp.mean(jnp.sum(s32**2, axis=-1))
    loss = jnp.zeros((), jnp.float32)
    terms = {
        _COMPRESS_LOSS: loss,
        "r_share": jax.lax.stop_gradient(r_energy / y_energy),
        "cos_sr": jax.lax.stop_gradient(sr / jnp.sqrt(jnp.maximum(s_energy * r_energy, 1e-30))),
        # Absolute per-token RMS lengths: a shrinking r_share alone cannot tell a growing s from a silent r.
        "s_rms": jax.lax.stop_gradient(jnp.sqrt(s_energy)),
        "r_rms": jax.lax.stop_gradient(jnp.sqrt(r_energy)),
    }
    if shadow is not None:
        x_flat = rearrange(jax.lax.stop_gradient(shared_in), "b s d -> (b s) d")
        hidden = jnp.square(jax.nn.relu(jnp.einsum("td,dm->tm", x_flat, shadow.w_up.astype(x_flat.dtype))))
        pred = jnp.einsum("tm,md->td", hidden, shadow.w_down.astype(x_flat.dtype), out_sharding=_batch_spec())
        pred = _batch_reshard(rearrange(pred, "(b s) d -> b s d", b=routed.shape[0])).astype(jnp.float32)
        target = jax.lax.stop_gradient(r32)
        shadow_loss = jnp.mean(jnp.sum((pred - target) ** 2, axis=-1)) / jax.lax.stop_gradient(r_energy)
        terms[_COMPRESS_LOSS] = loss + shadow_loss
        terms["shadow_r2"] = 1.0 - jax.lax.stop_gradient(shadow_loss)
    return {f"{_COMPRESS_PREFIX}{k}" if k != _COMPRESS_LOSS else k: v for k, v in terms.items()}


_ATTN_DROPOUT_SALT, _MLP_DROPOUT_SALT = 0xD1, 0xD2


def _sublayer_dropout(cfg: "GrugModelConfig", x: jax.Array, noise_key: jax.Array | None, salt: int) -> jax.Array:
    """``sublayer_dropout`` on one sublayer output; identity in evaluation (no key) or at p = 0."""
    return x


def _sconv_segment_ids(mask: AttentionMask | jax.Array) -> jax.Array | None:
    """segment_ids (packed-document boundaries) for the SConvs and KDA; None when unpacked."""
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    return segment_ids[0] if segment_ids is not None else None


def _laurel(h: Float[Array, "B S D"], a: jax.Array | None, b: jax.Array | None) -> Float[Array, "B S D"]:
    """``h + (h A) B`` (LAuReL-LR), or ``h`` when off."""
    if a is None or b is None:
        return h
    low = jnp.einsum("bsd,dr->bsr", h, a.astype(h.dtype), out_sharding=_batch_spec())
    return h + jnp.einsum("bsr,rd->bsd", low, b.astype(h.dtype), out_sharding=_batch_spec())


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


def _row_sharded_memory_bag_local(values: jax.Array, slots: jax.Array, weights: jax.Array) -> jax.Array:
    b, s, r = slots.shape
    rows, dim = values.shape
    group = _fsdp_group_size()

    def chunk_bag(args):
        c_slots, c_w = (jax.lax.all_gather(a, _FSDP_AXES, tiled=True) for a in args)
        local, valid = _row_shard_local_ids(rows, c_slots)
        partial = jnp.einsum("cr,crd->cd", jnp.where(valid, c_w, 0.0), values[local].astype(jnp.float32))
        return jax.lax.psum_scatter(partial, _FSDP_AXES, scatter_dimension=0, tiled=True)

    chunks = tuple(_row_gather_chunks(a.reshape(b * s, r), group, r * dim) for a in (slots, weights))
    return jax.lax.map(chunk_bag, chunks).reshape(b, s, dim).astype(values.dtype)


def _row_sharded_memory_bag_bwd_local(values, slots, weights, g):
    b, s, r = slots.shape
    rows, dim = values.shape
    group = _fsdp_group_size()

    def chunk_grad(d_table, args):
        c_slots, c_w, c_g = (jax.lax.all_gather(a, _FSDP_AXES, tiled=True) for a in args)
        local, valid = _row_shard_local_ids(rows, c_slots)
        g32 = c_g.astype(jnp.float32)
        d_w = jnp.einsum("cd,crd->cr", g32, jnp.where(valid[..., None], values[local].astype(jnp.float32), 0.0))
        d_w = jax.lax.psum_scatter(d_w, _FSDP_AXES, scatter_dimension=0, tiled=True)
        d_rows = jnp.where(valid, c_w, 0.0)[..., None] * g32[:, None, :]
        return d_table.at[local.reshape(-1)].add(d_rows.reshape(-1, dim)), d_w

    chunks = tuple(_row_gather_chunks(a.reshape(b * s, *a.shape[2:]), group, r * dim) for a in (slots, weights, g))
    d_table0 = jax.lax.pcast(jnp.zeros(values.shape, jnp.float32), _BATCH_AXES, to="varying")
    d_table, d_w = jax.lax.scan(chunk_grad, d_table0, chunks)
    return jax.lax.psum(d_table, _TABLE_REPLICA_AXES).astype(values.dtype), d_w.reshape(b, s, r)


@jax.custom_vjp
def _row_sharded_memory_bag(values: jax.Array, slots: Int[Array, "B S R"], weights: Float[Array, "B S R"]):
    """``_memory_bag`` from a table row-sharded ``P(_FSDP_AXES, None)``, the way ``_row_sharded_embedding_gather``
    looks up rows: slots, weights and cotangents cross the network, never the table or its gradient."""
    return shard_map(
        _row_sharded_memory_bag_local,
        mesh=get_abstract_mesh(),
        in_specs=(P(_FSDP_AXES, None), P(_BATCH_AXES, None, None), P(_BATCH_AXES, None, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(values, slots, weights)


def _row_sharded_memory_bag_fwd(values, slots, weights):
    return _row_sharded_memory_bag(values, slots, weights), (values, slots, weights)


def _row_sharded_memory_bag_bwd(residuals, g):
    values, slots, weights = residuals
    d_values, d_weights = shard_map(
        _row_sharded_memory_bag_bwd_local,
        mesh=get_abstract_mesh(),
        in_specs=(
            P(_FSDP_AXES, None),
            P(_BATCH_AXES, None, None),
            P(_BATCH_AXES, None, None),
            P(_BATCH_AXES, None, None),
        ),
        out_specs=(P(_FSDP_AXES, None), P(_BATCH_AXES, None, None)),
    )(values, slots, weights, reshard(g, P(_BATCH_AXES, None, None)))
    return d_values, np.zeros(slots.shape, dtype=jax.dtypes.float0), d_weights


_row_sharded_memory_bag.defvjp(_row_sharded_memory_bag_fwd, _row_sharded_memory_bag_bwd)


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
        d, n, heads = cfg.hidden_dim, (512), (4)
        return ProductKeyMemory(
            w_q=reshard(_init_weight(q_key, (d, heads * (256)), cfg.initializer_std), P(None, None)),
            keys=reshard(random.normal(k_key, (heads, 2, n, (256) // 2), jnp.float32), P()),
            values=reshard(
                _init_weight(v_key, (n * n, d), 1.0 / math.sqrt(d)),
                P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
            ),
            w_gate=reshard(_init_weight(g_key, (d, d), cfg.initializer_std), P(None, None)),
            w_out=reshard(jnp.zeros((d, d), jnp.float32), P(None, None)),
            topk=(32),
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
        memory_bag = _row_sharded_memory_bag if self.fsdp else _memory_bag
        bag = memory_bag(self.values, slots, weights).astype(x.dtype)
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
    attn_gated_norm: GatedNorm | None
    attn: CausalSelfAttention | KimiDeltaAttention
    rms_mlp: LearnedRMSNorm | DyT
    mlp_gated_norm: GatedNorm | None
    mlp: "MoEMLP | DenseMLP"
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: "ShortConv | None"
    sconv_mlp: "ShortConv | None"
    sconv_mlp_in: "ShortConv | None"  # Canon-C conv on the normed MLP input ("mlp_in" in cfg.sconv_sites)
    # Block AttnRes pseudo-queries of the attention and MLP sublayers (None without cfg.attn_res).
    attn_res_query_attn: Float[Array, " D"] | None
    attn_res_query_mlp: Float[Array, " D"] | None
    attn_res_query_shared: Float[Array, " D"] | None  # cfg.attn_res_shared_query
    attn_res_query_kv: Float[Array, " D"] | None  # cfg.attn_res_kv_layers (every layer carries one; those read it)
    extra_mlp: DenseMLP | None  # cfg.extra_mlp_layers (every layer carries one; those use it)
    extra_mlp_gain: Float[Array, ""] | None
    attn_res_partial_scale: Float[Array, " 2"] | None  # cfg.attn_res_partial_scale: (a_attn, a_mlp)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, layer_index: jax.Array, use_kda: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        attn_cfg = cfg if cfg.mla_num_heads is None else dataclasses.replace(cfg, num_heads=cfg.mla_num_heads)
        attn = (
            KimiDeltaAttention.init(cfg, key=attn_key)
            if use_kda
            else CausalSelfAttention.init(attn_cfg, key=attn_key, layer_index=layer_index)
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
            rms_attn=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps, role="rms_attn"),
            attn_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key) if cfg.sublayer_gated_norm else None
            ),
            attn=attn,
            rms_mlp=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps, role="rms_mlp"),
            mlp_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key) if cfg.sublayer_gated_norm else None
            ),
            mlp=mlp,
            shared=shared,
            sconv_attn=(ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel, cfg.sconv_impl) if use_attn_sconv else None),
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel, cfg.sconv_impl)
                if cfg.sconv and "mlp" in cfg.sconv_sites
                else None
            ),
            sconv_mlp_in=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel, cfg.sconv_impl)
                if cfg.sconv and "mlp_in" in cfg.sconv_sites
                else None
            ),
            attn_res_query_attn=attn_res_query,
            attn_res_query_mlp=attn_res_query,
            attn_res_query_shared=attn_res_query if cfg.attn_res_shared_query else None,
            attn_res_query_kv=attn_res_query if cfg.attn_res_kv_layers else None,
            extra_mlp=(
                DenseMLP.init(cfg.hidden_dim, cfg.extra_mlp_dim, cfg.initializer_std, key=random.fold_in(key, 77))
                if cfg.extra_mlp_layers
                else None
            ),
            extra_mlp_gain=jnp.zeros((), jnp.float32) if cfg.extra_mlp_layers else None,
            attn_res_partial_scale=(
                reshard(jnp.ones((2,), jnp.float32), P(None)) if cfg.attn_res and cfg.attn_res_partial_scale else None
            ),
        )

    def _attn_input(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = self.rms_attn(x)
        return x if self.attn_gated_norm is None else self.attn_gated_norm(x)

    def _mlp_input(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        x = self.rms_mlp(x)
        return x if self.mlp_gated_norm is None else self.mlp_gated_norm(x)

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
        attn_in = self._attn_input(_laurel(h, None, None))
        proj_inputs = None
        attn_components = tuple(c for c in sum_components if c in ("q", "k", "v"))
        if attn_components:
            assert sum_stream is not None
            sum_in = self._attn_input(sum_stream)
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
                token_ids=token_ids,
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
        if self.sconv_attn is not None:
            out = self.sconv_attn(out, _sconv_segment_ids(mask))
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
        router_embed: Float[Array, "B S D"] | None = None,
        null_routing: bool = False,
        saved_routed: Float[Array, "T L"] | None = None,
        moe_k_keep: Int[Array, "B S"] | None = None,
        moe_dynamic_layer: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``sum_stream`` feeds ``sum_parts`` (``router`` / ``latent`` / ``shared``) instead of ``h``;
        ``hash_token_ids`` / ``noise_key`` / ``router_tok_rows`` / ``router_seed_bias`` go to the router
        (``moe_hash_layers``, ``moe_gumbel_tau``, ``router_token_bias_rank``, ``router_bias_seed``)."""
        normed = self._mlp_input(_laurel(h, None, None))
        if self.sconv_mlp_in is not None:
            normed = self.sconv_mlp_in(normed, _sconv_segment_ids(mask))
        mlp_in = _spread_mlp_input(normed, self.attn.cfg)
        part_inputs = None
        if sum_parts:
            assert sum_stream is not None
            sum_in = self._mlp_input(sum_stream)
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
                None,
                router_tok_rows,
                router_seed_bias,
                None,
                router_embed,
                null_routing,
                saved_routed,
                moe_k_keep=moe_k_keep,
                moe_dynamic_layer=moe_dynamic_layer,
            )
        else:
            out, stats = self.mlp(
                mlp_in,
                hash_token_ids=hash_token_ids,
                noise_key=noise_key,
                router_tok_rows=router_tok_rows,
                router_seed_bias=router_seed_bias,
                router_embed=router_embed,
                null_routing=null_routing,
                saved_routed=saved_routed,
                moe_k_keep=moe_k_keep,
                moe_dynamic_layer=moe_dynamic_layer,
            )
        if self.sconv_mlp is not None:
            out = self.sconv_mlp(out, _sconv_segment_ids(mask))
        return out, stats

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_out, attn_stats = self.attn_branch(x, mask, disable_rope, is_global)
        x = x + attn_out
        mlp_out, router_stats = self.mlp_branch(x, mask)
        probed = {k: attn_stats[k] for k in (HEAD_PROBE_STAT, HEAD_QCOS_STAT) if k in attn_stats}
        if probed:
            router_stats = {**router_stats, **probed}
        return x + mlp_out, router_stats


@functools.lru_cache(maxsize=4)
def _hash_expert_table(vocab_size: int, num_experts: int, k: int) -> np.ndarray:
    """``[vocab, k]`` distinct random experts per token id (fixed seed), for ``moe_hash_layers``."""
    rng = np.random.default_rng(0)
    return np.argsort(rng.random((vocab_size, num_experts)), axis=1)[:, :k].astype(np.int32)


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
        fp8_dispatch=(False),
        expert_remat=cfg.moe_expert_remat,
    )
    if cfg.moe_ungated_relu2:
        mlp = eqx.tree_at(lambda m: m.w_gate, mlp, None, is_leaf=lambda x: x is None)
    return mlp


@contextmanager
def head_probe() -> Iterator[None]:
    """Forwards traced inside this context record ``HEAD_PROBE_STAT`` from every softmax-attention layer (dense
    reference attention over query blocks, so it is for analysis forwards, not training)."""
    global _HEAD_PROBE
    previous, _HEAD_PROBE = _HEAD_PROBE, True
    try:
        yield
    finally:
        _HEAD_PROBE = previous


def _head_attention_stats(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S K D"],
    mask: AttentionMask | jax.Array,
    rel_bias: jax.Array | None,
    segment_ids: Int[Array, "B S"] | None,
) -> Float[Array, "B S H 3"]:
    """Per query and head: softmax mass on its document's first token, on itself, and the attention entropy,
    from the kernel's inputs (scale ``1/sqrt(head_dim)``, causal, document and window masks)."""
    if rel_bias is not None:
        raise ValueError("head_probe supports attention without the Inkling relative-position bias")
    if not isinstance(mask, AttentionMask):
        raise ValueError("head_probe needs an AttentionMask")
    b, s, h, d = q.shape
    # Batch-sharded only: q's heads may be model-sharded while the shared MQA key head is not.
    q = reshard(q, P(_BATCH_AXES, None, None, None))
    k = reshard(align_kv_heads(k, num_q_heads=h), P(_BATCH_AXES, None, None, None))
    starts = jnp.arange(s)[None, :] - _positions_in_document(segment_ids, s).reshape(-1, s)  # [B or 1, S]
    if mask.fa4_bounds is not None:
        lower, valid = mask.fa4_bounds
    else:
        lower, valid = jnp.broadcast_to(starts, (b, s)), jnp.ones((b, s), jnp.bool_)
        if mask.sliding_window is not None:
            lower = jnp.maximum(lower, jnp.arange(s)[None, :] - (mask.sliding_window - 1))
    block = min(_HEAD_PROBE_QUERY_BLOCK, s)
    if s % block:
        raise ValueError(f"head_probe needs the sequence length divisible by {block}, got {s}")
    keys = jnp.arange(s)

    def one_block(i):
        rows = i * block + jnp.arange(block)
        qb = jax.lax.dynamic_slice_in_dim(q, i * block, block, axis=1).astype(jnp.float32)
        logits = jnp.einsum("bqhd,bkhd->bhqk", qb, k.astype(jnp.float32)) / math.sqrt(d)
        lb = jax.lax.dynamic_slice_in_dim(lower, i * block, block, axis=1)  # [B, Q]
        ok = (keys[None, None, :] <= rows[None, :, None]) & (keys[None, None, :] >= lb[:, :, None])
        ok = ok & jax.lax.dynamic_slice_in_dim(valid, i * block, block, axis=1)[:, :, None]
        logits = jnp.where(ok[:, None], logits, -jnp.inf)
        p = jax.nn.softmax(logits, axis=-1)
        p = jnp.where(ok[:, None], p, 0.0)
        start = jax.lax.dynamic_slice_in_dim(jnp.broadcast_to(starts, (b, s)), i * block, block, axis=1)
        p_start = jnp.take_along_axis(p, start[:, None, :, None], axis=-1)[..., 0]
        p_self = jnp.take_along_axis(p, jnp.broadcast_to(rows, (b, block))[:, None, :, None], axis=-1)[..., 0]
        entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-30)), 0.0), axis=-1)
        return jnp.stack([p_start, p_self, entropy], axis=-1)  # [B, H, Q, 3]

    out = jax.lax.map(one_block, jnp.arange(s // block))  # [blocks, B, H, Q, 3]
    return jax.lax.stop_gradient(rearrange(out, "n b h q f -> b (n q) h f"))


def _query_head_cosines(q: Float[Array, "B S H D"], segment_ids: Int[Array, "B S"] | None) -> jax.Array:
    """``[2, H, H]``: token-mean cosine and token-mean absolute cosine between every pair of query heads' vectors,
    over non-padding tokens."""
    unit = reshard(q, P(_BATCH_AXES, None, None, None)).astype(jnp.float32)
    unit = unit * jax.lax.rsqrt(jnp.sum(jnp.square(unit), axis=-1, keepdims=True) + 1e-12)
    cos = jnp.einsum("bshd,bsgd->bshg", unit, unit)
    valid = jnp.ones(cos.shape[:2], jnp.float32) if segment_ids is None else (segment_ids >= 0).astype(jnp.float32)
    count = jnp.maximum(jnp.sum(valid), 1.0)
    mean = jnp.einsum("bshg,bs->hg", cos, valid) / count
    mean_abs = jnp.einsum("bshg,bs->hg", jnp.abs(cos), valid) / count
    return jax.lax.stop_gradient(reshard(jnp.stack([mean, mean_abs]), P(None, None, None)))


def _head_output_stats(
    attn_stats: Float[Array, "B S H 3"],
    gate: jax.Array,
    gated_out: Float[Array, "B S H D"],
    w_o: Float[Array, "HD E"],
) -> Float[Array, "B S H 6"]:
    """``attn_stats`` plus each head's mean output gate, gated output norm and residual-contribution norm."""
    h, d = gated_out.shape[-2:]
    gated = reshard(gated_out, P(_BATCH_AXES, None, None, None)).astype(jnp.float32)
    head_gate = jnp.mean(jnp.broadcast_to(gate, gated_out.shape).astype(jnp.float32), axis=-1)
    head_gate = reshard(head_gate, P(_BATCH_AXES, None, None))
    per_head_wo = reshard(w_o, P(None, None)).astype(jnp.float32).reshape(h, d, -1)
    contrib = jnp.einsum("bshd,hde->bshe", gated, per_head_wo)
    extra = jnp.stack([head_gate, jnp.linalg.norm(gated, axis=-1), jnp.linalg.norm(contrib, axis=-1)], axis=-1)
    return jax.lax.stop_gradient(jnp.concatenate([attn_stats, extra], axis=-1))


def _positions_in_document(segment_ids: Int[Array, "B S"] | None, seq_len: int) -> jax.Array:
    """Each token's 0-based position within its packed document (``[S]`` when unpacked)."""
    idx = jnp.arange(seq_len, dtype=jnp.int32)
    if segment_ids is None:
        return idx
    starts = jnp.pad(segment_ids[:, 1:] != segment_ids[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    return idx - jax.lax.cummax(jnp.where(starts, idx, 0), axis=1)


def _expert_banks(cfg: "GrugModelConfig") -> list[tuple[int, int, int]]:
    """``(first expert, number of experts, experts per token)`` for each expert bank."""
    return [(0, cfg.num_experts, cfg.num_experts_per_token)]


def _bank_config(cfg: "GrugModelConfig", bank: int) -> "GrugModelConfig":
    """The per-bank view of ``cfg`` used to build and run that bank's experts."""
    return cfg


def _mixed_width_weights(cfg: "GrugModelConfig", w_up: jax.Array, w_down: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``moe_small_expert_width``: zero the first half of the experts' hidden units past that fraction."""
    if not cfg.moe_small_expert_width:
        return w_up, w_down
    experts, hidden = w_down.shape[0], w_down.shape[1]
    small = jnp.arange(experts)[:, None] < experts // 2
    units = jnp.arange(hidden)[None, :] < round(cfg.moe_small_expert_width * hidden)
    mask = (~small | units).astype(w_up.dtype)  # [E, I]
    width = w_up.shape[-1] // hidden  # gated banks concatenate gate and up along the last axis
    up_mask = jnp.tile(mask, (1, width))[:, None, :]
    return w_up * up_mask, w_down * mask[:, :, None]


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
            *_mixed_width_weights(cfg, em.w_up if ungated else jnp.concatenate([em.w_up, em.w_up], axis=-1), em.w_down),
            activation=(ActivationFunctionEnum.relu2) if ungated else _tied_expert_activation(cfg, em),
            implementation=em.implementation,
            mesh=get_abstract_mesh(),
            capacity_factor=em.capacity_factor,
            pooled_transport_capacity_factor=em.pooled_transport_capacity_factor,
            report_capacity_overflow=True,
            report_assignment_keep=cfg.moe_drop_renorm,
            expert_chunks=em.expert_chunks,
            num_expert_waves=em.num_expert_waves,
            fp8_dispatch=em.fp8_dispatch,
            expert_mxfp8_sim=cfg.fp8_recipe == Fp8Recipe.MXFP8_SIM,
            ragged_expert_mxfp8=_ragged_expert_mxfp8(cfg),
            ragged_expert_mxfp8_dispatch=cfg.moe_mxfp8_dispatch,
            expert_remat=em.expert_remat,
            overlap=overlap,
            overlap_param_gather=cfg.moe_overlap_param_gather,
            ragged_expert_chunks=cfg.moe_ragged_chunks,
            expert_serial_pair=cfg.moe_serial_pair,
            ragged_overlap_weight_grads=cfg.moe_overlap_weight_grads,
        )
    if cfg.fp8_recipe == Fp8Recipe.MXFP8_SIM:
        raise ValueError("fp8_recipe=mxfp8_sim simulates ungated ReLU^2 expert banks only")
    return em(
        routed_input,
        selected,
        combine_weights,
        mesh=get_abstract_mesh(),
        report_capacity_overflow=True,
        overlap=overlap,
        overlap_param_gather=cfg.moe_overlap_param_gather,
        report_assignment_keep=cfg.moe_drop_renorm,
        ragged_expert_mxfp8=_ragged_expert_mxfp8(cfg),
        ragged_expert_mxfp8_dispatch=cfg.moe_mxfp8_dispatch,
    )


def _grouped_rms_norm(cfg: "GrugModelConfig", groups: int, width: int) -> LearnedRMSNorm:
    """A learned RMSNorm with a stacked ``[groups, width]`` gain, normalizing each ``[..., groups, width]`` slice."""
    norm = _learned_rms_norm(cfg, width, cfg.layer_norm_eps)
    return jax.tree.map(lambda g: jnp.broadcast_to(g, (groups, width)), norm)


def _run_grouped_read_bank(
    em: MoEExpertMlp,
    cfg: "GrugModelConfig",
    read_norm: LearnedRMSNorm | None,
    x_flat: Float[Array, "T D"],
    selected: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    overlap: MoeOverlapWork | None,
    private_rows: Float[Array, "T K R"] | None = None,
):
    """Run the routed experts on per-assignment inputs: ``expert_read_groups`` slices (or the whole input when
    ``read_norm`` is None), with each assignment's ``expert_private_dim`` slice appended when given.

    Splits ``x_flat`` into G normed ``[T, G, W]`` slices, gathers each (token, slot)'s slice ``g(expert) =
    expert mod G`` to ``[T, K, W]`` and dispatches the ``T*K`` slot rows as single-slot tokens (top-1 over the
    same experts, in the same token-major order, so capacity and drops match the ``[T, D]`` dispatch), then sums
    the K weighted slot outputs back per token.
    """
    t, k = selected.shape
    if read_norm is None:
        slot_rows = jnp.broadcast_to(x_flat[:, None, :], (t, k, x_flat.shape[-1]))
    else:
        slices = jnp.reshape(x_flat, (t, (0), x_flat.shape[-1] // (0)), out_sharding=P(_BATCH_AXES, None, None))
        slices = read_norm(slices)
        slot_rows = jnp.take_along_axis(slices, (selected % (0))[:, :, None], axis=1)
    if private_rows is not None:
        slot_rows = jnp.concatenate([slot_rows, private_rows.astype(slot_rows.dtype)], axis=-1)
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


def _ragged_expert_mxfp8(cfg: "GrugModelConfig") -> ExpertMxfp8:
    if cfg.fp8_recipe not in (Fp8Recipe.MXFP8_EXPERTS, Fp8Recipe.MXFP8):
        return ExpertMxfp8.NONE
    if cfg.mxfp8_epilogue_tokens:
        return ExpertMxfp8.ALL_EPILOGUE_TOKENS
    return ExpertMxfp8.ALL if cfg.mxfp8_expert_wgrad else ExpertMxfp8.ACTIVATIONS


def _tied_expert_activation(cfg: "GrugModelConfig", em: MoEExpertMlp):
    """``act`` with ``act(u) * u == leaky_relu(u, slope)^2`` for backends that take the gate tied to ``W_up``."""
    if cfg.moe_ungated_relu2 and (0.0):
        slope_sq = (0.0) ** 2
        return lambda u: jax.nn.leaky_relu(u, slope_sq)
    return em.activation


class VocabStats(NamedTuple):
    """Per-position statistics of a next-token distribution (``_vocab_stats``)."""

    entropy: Float[Array, "B S"]
    logsumexp: Float[Array, "B S"]
    top_logit: Float[Array, "B S"]
    top_id: Int[Array, "B S"]
    label_logit: Float[Array, "B S"]


def _vocab_stats(
    head_in: Float[Array, "B S E"], lm_head: Float[Array, "E V"], cap: float | None, labels: Int[Array, "B S"]
) -> VocabStats:
    """Stop-gradient statistics of each position's next-token distribution, one online pass over vocabulary chunks
    (``H = log s + m - t / s`` with running max ``m``, ``s = sum e^(z - m)``, ``t = sum e^(z - m) z``), so the
    ``[B, S, V]`` logits are never materialized. ``cap`` applies the logit soft cap first."""
    head_in = jax.lax.stop_gradient(head_in)
    vocab = lm_head.shape[1]
    chunk = math.gcd(vocab, _ENTROPY_CHUNK)
    # Scanned chunks must be unsharded along the scan axis: gather the (FSDP / model-sharded) head first.
    weights = rearrange(reshard(jax.lax.stop_gradient(lm_head), P(None, None)), "e (n c) -> n e c", c=chunk)
    b, s_len, _ = head_in.shape
    spec = P(_BATCH_AXES, None)
    init = (
        *(reshard(jnp.full((b, s_len), v, jnp.float32), spec) for v in (-jnp.inf, 0.0, 0.0, 0.0)),
        reshard(jnp.zeros((b, s_len), jnp.int32), spec),
    )
    labels = reshard(labels.astype(jnp.int32), spec)

    def step(carry, chunk_in):
        m, total, weighted, label_logit, top_id = carry
        w, start = chunk_in
        z = jnp.einsum("bse,ec->bsc", head_in, w.astype(head_in.dtype), preferred_element_type=jnp.float32)
        z = reshard(z, P(_BATCH_AXES, None, None))
        if cap is not None:
            z = cap * jnp.tanh(z / cap)
        chunk_max = jnp.max(z, axis=-1)
        top_id = jnp.where(chunk_max > m, start + jnp.argmax(z, axis=-1).astype(jnp.int32), top_id)
        m_new = jnp.maximum(m, chunk_max)
        scale = jnp.exp(m - m_new)
        e = jnp.exp(z - m_new[..., None])
        offset = labels - start
        in_chunk = (offset >= 0) & (offset < chunk)
        at_label = jnp.take_along_axis(z, jnp.clip(offset, 0, chunk - 1)[..., None], axis=-1)[..., 0]
        label_logit = jnp.where(in_chunk, at_label, label_logit)
        carry = (m_new, total * scale + jnp.sum(e, -1), weighted * scale + jnp.sum(e * z, -1), label_logit, top_id)
        return carry, None

    starts = jnp.arange(weights.shape[0], dtype=jnp.int32) * chunk
    (m, total, weighted, label_logit, top_id), _ = jax.lax.scan(step, init, (weights, starts))
    lse = jnp.log(total) + m
    return VocabStats(lse - weighted / total, lse, m, top_id, label_logit)


def _predictive_entropy(
    head_in: Float[Array, "B S E"], lm_head: Float[Array, "E V"], cap: float | None
) -> Float[Array, "B S"]:
    labels = jnp.zeros(head_in.shape[:2], jnp.int32)
    return _vocab_stats(head_in, lm_head, cap, labels).entropy


def _probe_mlp_params(prefix: str, n: int, cfg: "GrugModelConfig", key: jax.Array) -> dict[str, jax.Array | None]:
    """``n`` stacked probe MLPs (D -> ``mid_confidence_dim`` -> 1) as ``{prefix}_{w1,b1,w2,b2}``; None for n = 0."""
    names = [f"{prefix}_{name}" for name in ("w1", "b1", "w2", "b2")]
    if not n:
        return dict.fromkeys(names, None)
    d, hidden = cfg.hidden_dim, cfg.mid_confidence_dim
    values = (
        reshard(random.normal(key, (n, d, hidden), jnp.float32) / math.sqrt(d), P(None, None, None)),
        jnp.zeros((n, hidden), jnp.float32),
        jnp.zeros((n, hidden), jnp.float32),
        jnp.zeros((n,), jnp.float32),
    )
    return dict(zip(names, values, strict=True))


def _final_conf_params(cfg: "GrugModelConfig", key: jax.Array) -> dict[str, jax.Array | None]:
    return _probe_mlp_params("final_conf", len(cfg.final_conf_layers), cfg, key)


def _probe_mlp_logits(
    w1: jax.Array, b1: jax.Array, w2: jax.Array, b2: jax.Array, stream: Float[Array, "B S D"], eps: float
) -> Float[Array, "N B S"]:
    x = rms_norm(jax.lax.stop_gradient(stream).astype(jnp.float32), eps)
    h = jax.nn.relu(jnp.einsum("bsd,ndh->nbsh", x, w1) + b1[:, None, None, :])
    return jnp.einsum("nbsh,nh->nbs", h, w2) + b2[:, None, None]


def _mid_head_params(cfg: "GrugModelConfig", head_shape: tuple[int, int], key: jax.Array) -> dict[str, jax.Array | None]:
    names = ("mid_output_proj", "mid_conf_w1", "mid_conf_b1", "mid_conf_w2", "mid_conf_b2")
    if cfg.mid_head_layer is None:
        return dict.fromkeys(names, None)
    head_key, conf_key = random.split(key)
    features, hidden = 2 * cfg.hidden_dim + 2, cfg.mid_confidence_dim
    return {
        "mid_output_proj": reshard(_init_weight(head_key, head_shape, cfg.initializer_std), _LM_HEAD_PARTITION_SPEC),
        "mid_conf_w1": reshard(
            random.normal(conf_key, (features, hidden), jnp.float32) / math.sqrt(features), P(None, None)
        ),
        "mid_conf_b1": jnp.zeros((hidden,), jnp.float32),
        "mid_conf_w2": jnp.zeros((hidden,), jnp.float32),
        "mid_conf_b2": jnp.zeros((), jnp.float32),
    }


def _logit_cap(cfg: "GrugModelConfig") -> float | tuple[float, float, float] | None:
    return cfg.logit_soft_cap


@contextmanager
def forward_probe(probe: ForwardProbe) -> Iterator[None]:
    """Forwards traced inside this context record ``probe`` (see ``PROBE_STAT``). Read at trace time, so a jitted
    function must be first traced inside it."""
    global _FORWARD_PROBE
    previous, _FORWARD_PROBE = _FORWARD_PROBE, probe
    try:
        yield
    finally:
        _FORWARD_PROBE = previous


def _probe_spot() -> tuple[int, int] | None:
    return None if _FORWARD_PROBE is None else _FORWARD_PROBE.spot


def _select_row(x: jax.Array, row: int) -> jax.Array:
    """Row ``row`` of a batch-sharded ``[B, ...]`` array, replicated, as a one-hot contraction (moves one row)."""
    one_hot = jax.nn.one_hot(row, x.shape[0], dtype=jnp.float32)
    out = jnp.einsum("b...,b->...", x.astype(jnp.float32), one_hot, out_sharding=P(*([None] * (x.ndim - 1))))
    return out


def _at_spot(x: Float[Array, "B S ..."], spot: tuple[int, int]) -> jax.Array:
    return jax.lax.stop_gradient(_select_row(x, spot[0])[spot[1]])


def _probe_attention(
    q: Float[Array, "B S H D"],
    k: Float[Array, "B S Hk D"],
    mask: AttentionMask | jax.Array,
    rel_bias: Float[Array, "B H S W"] | None,
    queries: tuple[tuple[int, int], ...],
) -> Float[Array, "Q H K"]:
    """Each query's softmax probabilities over keys ``position - ATTN_PROBE_KEYS + 1 .. position`` (zero outside its
    document), from the kernel's inputs: scale ``1/sqrt(head_dim)``, the banded Inkling bias, causal, document-masked."""
    if not isinstance(mask, AttentionMask):
        raise ValueError("forward_probe attention needs an AttentionMask")
    k = align_kv_heads(k, num_q_heads=q.shape[2])
    out = []
    for row, position in queries:
        q_row, k_row = _select_row(q, row), _select_row(k, row)  # [S, H, D]
        key_positions = position - (ATTN_PROBE_KEYS - 1) + jnp.arange(ATTN_PROBE_KEYS)
        valid = key_positions >= 0
        key_positions = jnp.maximum(key_positions, 0)
        logits = jnp.einsum("hd,khd->hk", q_row[position], k_row[key_positions]) / math.sqrt(q.shape[-1])
        if rel_bias is not None:
            band = _select_row(rel_bias, row)[:, position]  # [H, W]
            column = key_positions - (position // REL_BIAS_BLOCK) * REL_BIAS_BLOCK + rel_extent_of_band(rel_bias)
            in_band = (column >= 0) & (column < band.shape[-1])
            gathered = jnp.take(band, jnp.clip(column, 0, band.shape[-1] - 1), axis=-1)
            logits = logits + jnp.where(in_band[None], gathered, 0.0)
        if mask.segment_ids is not None:
            segments = _select_row(mask.segment_ids[1], row)
            valid &= segments[key_positions] == segments[position]
        out.append(jax.nn.softmax(jnp.where(valid[None], logits, -jnp.inf), axis=-1))
    return jax.lax.stop_gradient(jnp.stack(out))


def _soft_capped(logits: jax.Array, cap: float | tuple[float, float, float] | None) -> jax.Array:
    """The fused CE kernel's logit cap: ``c * tanh(z / c)``, or ``A * sigmoid((z + B) / C)`` for ``(A, B, C)``."""
    if cap is None:
        return logits
    if isinstance(cap, tuple):
        a, b, c = cap
        return a * jax.nn.sigmoid((logits + b) / c)
    return cap * jnp.tanh(logits / cap)


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
    return x


def _attn_res_key_dim(cfg: GrugModelConfig) -> int:
    """Width of every AttnRes pseudo-query: ``attn_res_key_rank`` if set, else ``hidden_dim``."""
    return cfg.hidden_dim


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


def _token_query_logit(source: Float[Array, "B S D"], rows: Float[Array, "B S D"], eps: float) -> Float[Array, "B S"]:
    """``rows . rms_norm(source)`` per token, in float32 (``attn_res_token_query``)."""
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(source.astype(jnp.float32)), axis=-1) + eps)
    dots = jnp.einsum("bsd,bsd->bs", source, rows.astype(source.dtype), preferred_element_type=jnp.float32)
    return dots * inv_rms


@jax.custom_vjp
def _logit_rows(logits: jax.Array) -> tuple[jax.Array, ...]:
    """A block's ``[G, ...]`` gate logits as a tuple of per-gate rows.

    Each later gate reads one row. Indexing the array instead makes every read's backward a pad to the
    full ``[G, ...]`` shape plus an add, quadratic in the gate count; here the backward is one stack."""
    return tuple(logits[i] for i in range(logits.shape[0]))


def _logit_rows_fwd(logits):
    return _logit_rows(logits), None


def _logit_rows_bwd(_, row_cotangents):
    return (jnp.stack(row_cotangents),)


_logit_rows.defvjp(_logit_rows_fwd, _logit_rows_bwd)


def _block_logit(block_logits: tuple[jax.Array, ...], queries: jax.Array, gate_index: int) -> jax.Array:
    """Gate ``gate_index``'s row of a block's logits (``_logit_rows``), which cover the trailing queries."""
    return block_logits[gate_index - (queries.shape[0] - len(block_logits))]


def _softmax_mix(logits: list[jax.Array], sources: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Per-token softmax weights ``[N, B, S]`` over ``sources`` and the weighted sum (in float32)."""
    weights = jax.nn.softmax(jnp.stack(logits), axis=0)
    if weights.ndim == 4:
        # Multi-head: weights [N, B, S, H] mix each D/H channel slice separately.
        heads = weights.shape[-1]

        def chunked(x):
            return rearrange(x.astype(jnp.float32), "b s (h d) -> b s h d", h=heads)

        rows = _logit_rows(weights)
        mixed = rows[0][..., None] * chunked(sources[0])
        for weight, source in zip(rows[1:], sources[1:], strict=True):
            mixed = mixed + weight[..., None] * chunked(source)
        return weights, rearrange(mixed, "b s h d -> b s (h d)")
    # Per-source rows through ``_logit_rows``: slicing ``weights`` would pad every row's gradient to [N, B, S].
    rows = _logit_rows(weights)
    mixed = rows[0][..., None] * sources[0].astype(jnp.float32)
    for weight, source in zip(rows[1:], sources[1:], strict=True):
        mixed = mixed + weight[..., None] * source.astype(jnp.float32)
    return weights, mixed


def _weighted_sources(weights: Float[Array, "N B S"], sources: Float[Array, "N B S D"]) -> Float[Array, "B S D"]:
    """``sum_n weights[n] * sources[n]`` in float32 over a stacked source buffer, summed in source order so it
    matches ``_softmax_mix`` bit for bit."""
    mixed = weights[0][..., None] * sources[0].astype(jnp.float32)
    for n in range(1, sources.shape[0]):
        mixed = mixed + weights[n][..., None] * sources[n].astype(jnp.float32)
    return mixed


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
    plain_sum: bool = False,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """One AttnRes gate: softmax over the completed blocks (+ the running partial) and their weighted sum
    (``plain_sum``: their plain sum, ``attn_res_plain_sum``).

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
    if plain_sum:
        sources = [*sources, *([] if partial is None else [partial])]
        uniform = jnp.full((len(sources),), 1.0 / len(sources), jnp.float32)
        return reshard(_stream_sum(tuple(sources)), _batch_spec()), jnp.zeros((), jnp.float32), uniform
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
    token_query = extras.get("token_query")
    if token_query is not None:
        # The token's query row against each source's normed key; the static query's part is already in logits.
        row = _embedding_gather(token_query[gate_index % token_query.shape[0]], extras["token_ids"])
        logits = [logit + _token_query_logit(src, row, eps) for logit, src in zip(logits, sources, strict=True)]
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


GateMix = Callable[..., tuple[jax.Array, jax.Array, jax.Array]]
"""``mix(partial, gate_index, attn_gate, leading_sources_only=False) -> (h, z, mean_weights)``: one AttnRes gate over a
layer's history (``leading_sources_only``: only the ``attn_res_kv_source_layers`` sources, no partial)."""


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

    def mix(p: jax.Array | None, gate_index: int, attn_gate: bool, leading_sources_only: bool = False):
        del attn_gate
        cfg = layer.attn.cfg
        if not leading_sources_only:
            return _attn_res_mix(
                blocks, block_logits, p, queries, gate_index, eps, logit_bias, plain_sum=cfg.attn_res_plain_sum
            )
        # The leading layers' sources: the extra sources, then the blocks rolled at layers 0, seg, ... <= the
        # cutoff (each holds only earlier layers); the partial and later blocks are later layers' work.
        seg = max(1, cfg.num_layers // cfg.attn_res_num_blocks)
        cap = cfg.attn_res_num_blocks
        num_extra = len(blocks) - min(layer_index // seg + 1, cap)
        keep = num_extra + min(cfg.kv_source_layers // seg + 1, cap)
        return _attn_res_mix(
            blocks[:keep],
            block_logits[:keep],
            None,
            queries,
            gate_index,
            eps,
            logit_bias,
            plain_sum=cfg.attn_res_plain_sum,
        )

    return _attn_res_sublayers(
        layer, partial, mix, mask, token_ids, use_long, layer_index, noise_key, kv_share, logit_bias
    )


def _attn_res_sublayers(
    layer: Block,
    partial: jax.Array | None,
    mix: GateMix,
    mask: AttentionMask,
    token_ids: Int[Array, "B S"],
    use_long: bool,
    layer_index: int | jax.Array,
    noise_key: jax.Array | None,
    kv_share: dict[str, jax.Array] | None,
    logit_bias: dict[str, jax.Array | None] | None,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """The attention and MLP sublayers of one AttnRes layer, each reading its gate's ``mix`` of the history.

    ``layer_index`` is a Python int in the unrolled loop and a traced int under ``scan_layers``."""
    cfg = layer.attn.cfg
    h, z_attn, w_attn = mix(partial, 2 * layer_index, True)
    attn_in = h
    attn_branch = type(layer).attn_branch
    if cfg.attn_res_remat_attention:
        attn_branch = eqx.filter_checkpoint(attn_branch, policy=None)
    physical = layer_index % cfg.num_layers
    kda_ablation = (physical in (), physical in ())
    v_stats: dict[str, jax.Array] = {}
    kv_input = _kv_stream_input(logit_bias)
    if physical in cfg.attn_res_kv_layers:
        kv_gate = 2 * cfg.num_layers + (cfg.num_layers if cfg.attn_res_shared_query else 0) + layer_index
        h_kv, _, _ = mix(None, kv_gate, False, leading_sources_only=True)
        kv_input = layer._attn_input(h_kv)
    attn_out, attn_stats = attn_branch(
        layer,
        h,
        mask,
        use_long,
        use_long,
        token_ids,
        kv_share,
        (None),
        (),
        kda_ablation,
        physical in (),
        kv_input,
    )
    attn_out = _sublayer_dropout(cfg, attn_out, noise_key, _ATTN_DROPOUT_SALT)
    skip_keep = None if logit_bias is None else logit_bias.get(_SKIP_KEEP)
    if skip_keep is not None:
        attn_out = attn_out * skip_keep[..., None].astype(attn_out.dtype)
    scale_attn = scale_mlp = None
    if layer.attn_res_partial_scale is not None:
        scale_attn, scale_mlp = (layer.attn_res_partial_scale[j].astype(attn_out.dtype) for j in range(2))
    partial_attn = attn_out if scale_attn is None else scale_attn * attn_out
    partial = partial_attn if partial is None else partial + partial_attn
    mlp_partial = partial
    h, z_mlp, w_mlp = mix(mlp_partial, 2 * layer_index + 1, False)
    shared_kwargs: dict[str, Any] = {}
    if layer.attn_res_query_shared is not None:
        # Shared-gate rows follow the per-layer gates, ahead of the final gate.
        h_shared, z_shared, _ = mix(mlp_partial, 2 * cfg.num_layers + layer_index, False)
        z_mlp = z_mlp + z_shared
        shared_kwargs = {"sum_stream": h_shared, "sum_parts": ("shared",)}
    route = _route_kwargs(cfg, physical, token_ids, noise_key, logit_bias)
    route["moe_k_keep"] = _moe_k_keep(cfg, physical, mask, attn_out)
    route["moe_dynamic_layer"] = physical in cfg.moe_dynamic_layers
    mlp_out, router_stats = layer.mlp_branch(h, mask, **shared_kwargs, **route)
    mem_out, mem_stats = _memory_branch(h, logit_bias)
    if mem_out is not None:
        mlp_out = mlp_out + mem_out
    mlp_out = _sublayer_dropout(cfg, mlp_out, noise_key, _MLP_DROPOUT_SALT)
    extra_keep = None if logit_bias is None else logit_bias.get(_EXTRA_KEEP)
    if extra_keep is not None:
        assert layer.extra_mlp is not None and layer.extra_mlp_gain is not None
        extra = layer.extra_mlp(layer._mlp_input(h), moe_output_reshard=False)
        mlp_out = mlp_out + (layer.extra_mlp_gain.astype(extra.dtype) * extra) * extra_keep[..., None].astype(
            extra.dtype
        )
    if skip_keep is not None:
        mlp_out = mlp_out * skip_keep[..., None].astype(mlp_out.dtype)
    if trace_capture.TRACE:
        router_stats = {
            **router_stats,
            f"{_LAYER_KNOB_PREFIX}trace_attn_out": attn_out,
            f"{_LAYER_KNOB_PREFIX}trace_mlp_out": mlp_out,
        }
    spot = _probe_spot()
    if spot is not None:
        router_stats = {
            **router_stats,
            f"{PROBE_STAT}attn_in": _at_spot(attn_in, spot),
            f"{PROBE_STAT}attn_out": _at_spot(attn_out, spot),
            f"{PROBE_STAT}mlp_in": _at_spot(h, spot),
            f"{PROBE_STAT}mlp_out": _at_spot(mlp_out, spot),
        }
    return partial + (mlp_out if scale_mlp is None else scale_mlp * mlp_out), {
        **router_stats,
        **attn_stats,
        **v_stats,
        **mem_stats,
        **attn_stats,
        _ATTN_RES_Z: z_attn + z_mlp,
        _ATTN_RES_W_ATTN: w_attn,
        _ATTN_RES_W_MLP: w_mlp,
    }


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
        "hash_token_ids": token_ids if physical_layer in () else None,
        "noise_key": noise_key,
        "router_tok_rows": None if extras is None else extras.get("router_tok"),
        "router_embed": None if extras is None else extras.get("router_embed"),
        "router_seed_bias": _merged_router_bias(
            None,
            None if extras is None else extras.get("router_hist"),
        ),
        "null_routing": physical_layer in cfg.moe_null_layers,
        "saved_routed": None if extras is None else extras.get(_SAVED_ROUTED),
    }


_HEAD_GATE_OFF = 0.1


def _head_gate_keep(cfg: GrugModelConfig, head_gate: Float[Array, "B S N"], stats: dict[str, jax.Array]) -> jax.Array:
    """``attn_head_skip``'s per-head keep mask (1 / 0) from gates normalized to (0, 1); logs the near-off and
    skipped head fractions into ``stats``."""
    g = jax.lax.stop_gradient(head_gate.astype(jnp.float32))
    stats[f"{_OUTSTAT_PREFIX}head_gate_off"] = jnp.mean((g < _HEAD_GATE_OFF).astype(jnp.float32))
    if not cfg.attn_head_skip:
        stats[f"{_OUTSTAT_PREFIX}head_skip_frac"] = jnp.zeros((), jnp.float32)
        return jnp.ones_like(g)
    keep = (g >= cfg.attn_head_skip).astype(jnp.float32)
    stats[f"{_OUTSTAT_PREFIX}head_skip_frac"] = 1.0 - jnp.mean(keep)
    return keep


def _moe_k_keep(
    cfg: GrugModelConfig, layer: int, mask: AttentionMask | jax.Array, attn_out: Float[Array, "B S D"]
) -> Int[Array, "B S"] | None:
    """Each token's routed-expert count on ``moe_dynamic_layers`` (``MoeDynamicK``); None elsewhere. RELU / EPS
    decide inside the router, so their layers get ``num_experts_per_token`` everywhere (it marks the layer)."""
    norm = jnp.sum(jnp.square(jax.lax.stop_gradient(attn_out).astype(jnp.float32)), axis=-1)  # [B, S]
    if cfg.moe_k_layers:
        return jnp.full_like(norm, cfg.moe_k_layers[layer], dtype=jnp.int32)
    if layer not in cfg.moe_dynamic_layers or cfg.moe_dynamic == MoeDynamicK.NONE:
        return None if cfg.moe_k_static is None else jnp.full_like(norm, cfg.moe_k_static, dtype=jnp.int32)
    k_high, k_low = cfg.num_experts_per_token, cfg.moe_k_low
    if cfg.moe_dynamic in (MoeDynamicK.RELU, MoeDynamicK.EPS, MoeDynamicK.THRESHOLD, MoeDynamicK.TOP_P):
        return jnp.full_like(norm, k_high, dtype=jnp.int32)
    if cfg.moe_dynamic == MoeDynamicK.POSITION:
        seg = _sconv_segment_ids(mask)
        idx = jnp.cumsum(jnp.ones_like(norm, dtype=jnp.int32), axis=1) - 1
        if seg is None:
            pos = idx
        else:
            start = jnp.pad(seg[:, 1:] != seg[:, :-1], ((0, 0), (1, 0)), constant_values=True)
            pos = idx - jax.lax.cummax(jnp.where(start, idx, 0), axis=1)
        high = pos >= cfg.moe_dynamic_position
    else:
        high = norm >= jnp.quantile(reshard(norm, P(None, None)), 1.0 - cfg.moe_dynamic_frac)
    return jnp.where(high, k_high, k_low).astype(jnp.int32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _relu_l1_grad(logits: jax.Array, coef: float, offset: float) -> jax.Array:
    """Identity whose backward adds the gradient of ``coef * mean_t sum_e relu(logits + offset)``."""
    return logits


def _relu_l1_grad_fwd(logits, coef, offset):
    return logits, logits + offset > 0


def _relu_l1_grad_bwd(coef, offset, positive, g):
    return (g + (coef / positive.shape[0]) * positive.astype(g.dtype),)


_relu_l1_grad.defvjp(_relu_l1_grad_fwd, _relu_l1_grad_bwd)


def _merged_router_bias(seed: jax.Array | None, history: jax.Array | None) -> jax.Array | None:
    if seed is None or history is None:
        return seed if history is None else history
    return seed + history


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
    # Router stats are logging-only and carry no cotangent, except the moe_compress / shadow loss and the
    # lm_head-only slice (lm_head_extra_dim).
    d_partial, d_blocks, d_block_logits, d_stats = grad_out
    diff_keys = [k for k in _DIFFERENTIABLE_STATS if isinstance(d_stats, dict) and d_stats.get(k) is not None]
    _, blocks, block_logits, _, _, _ = diff_args
    d_blocks = jax.tree.map(lambda d, x: jnp.zeros_like(x) if d is None else d, d_blocks, blocks, is_leaf=_is_none)
    d_block_logits = jax.tree.map(
        lambda d, x: jnp.zeros_like(x) if d is None else d, d_block_logits, block_logits, is_leaf=_is_none
    )
    with jax.named_scope(f"attn_res_bwd{layer_index}"):
        diff_args, d_partial, d_blocks, d_block_logits = jax.lax.optimization_barrier(
            (diff_args, d_partial, d_blocks, d_block_logits)
        )
        if not diff_keys:
            _, vjp_fn = jax.vjp(
                lambda args: _attn_res_layer(args, mask, token_ids, use_long, layer_index, eps, noise_key)[0], diff_args
            )
            cotangent = d_partial
        else:

            def with_stats(args):
                partial, stats = _attn_res_layer(args, mask, token_ids, use_long, layer_index, eps, noise_key)
                return partial, tuple(stats[k] for k in diff_keys)

            _, vjp_fn = jax.vjp(with_stats, diff_args)
            cotangent = (d_partial, tuple(d_stats[k] for k in diff_keys))
        ((d_layer, d_blocks_own, d_block_logits_own, d_partial_in, d_queries, d_logit_bias),) = vjp_fn(cotangent)
        d_blocks = tuple(a + b for a, b in zip(d_blocks, d_blocks_own, strict=True))
        d_block_logits = jax.tree.map(jnp.add, d_block_logits, d_block_logits_own)
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
    return tuple(i for i in range(cfg.num_layers) if not _is_long_layer(i, cfg.num_layers, cfg.global_every, (None)))


def _unstack_layers(stacked: ArrayStacked[Block]) -> list[Block]:
    """Split the stacked layer params into per-layer modules (split, so the grad is one concatenate)."""
    return _split_leading(stacked.stacked, (1,) * stacked.num_layers, squeeze=True)


def _split_leading(tree, sizes: tuple[int, ...], *, squeeze: bool = False) -> list:
    """Split every leaf of ``tree`` along axis 0 into ``sizes`` (``squeeze``: size-1 parts lose the axis)."""
    leaves, treedef = jax.tree.flatten(tree)
    split_leaves = [jax.lax.split(leaf, sizes, axis=0) for leaf in leaves]
    return [
        treedef.unflatten([parts[i][0] if squeeze else parts[i] for parts in split_leaves]) for i in range(len(sizes))
    ]


class _BlockSchedule(NamedTuple):
    """Where Block AttnRes rolls the running partial into a new block: at every ``seg_size``-th layer, for at
    most ``block_cap`` blocks, stored after the ``num_extra`` extra sources."""

    num_extra: int
    seg_size: int
    block_cap: int

    def num_sources(self, layer_index: int) -> int:
        """History sources (extra sources, embedding, completed blocks) that layer ``layer_index`` reads."""
        return self.num_extra + min(layer_index // self.seg_size + 1, self.block_cap)

    def is_boundary(self, layer_index: int) -> bool:
        return layer_index % self.seg_size == 0 and layer_index // self.seg_size < self.block_cap


class _AttnResHistory(NamedTuple):
    """The fixed-shape Block AttnRes history threaded through the ``scan_layers`` scan.

    ``sources[n]`` is source ``n`` in the unrolled loop's order (extra sources, the embedding, then the
    completed blocks) and stays zero until it is rolled in; ``logits[n]`` holds its float32 logits against
    every gate's query; ``partial`` is the running partial sum (zero right after a block boundary).
    """

    sources: Float[Array, "N B S D"]
    logits: Float[Array, "N G B S"]
    partial: Float[Array, "B S D"]


class _UnitLayer(NamedTuple):
    """Layer ``j`` of the scanned layer unit: slice ``index`` of its mixer kind's stack (``is_kda``)."""

    is_kda: bool
    index: int
    use_long: bool


class _LayerScan(NamedTuple):
    """``scan_layers``: the first ``num_units`` units of ``len(unit)`` layers run as one ``jax.lax.scan``."""

    unit: tuple[_UnitLayer, ...]
    num_units: int
    schedule: _BlockSchedule

    @property
    def num_layers(self) -> int:
        return len(self.unit) * self.num_units

    def kind_count(self, is_kda: bool) -> int:
        return sum(layer.is_kda == is_kda for layer in self.unit)


def _layer_scan(cfg: GrugModelConfig, schedule: _BlockSchedule) -> _LayerScan:
    """The scanned prefix of ``scan_layers``: the leading run of ``global_every``-layer units whose layers
    match unit 0's mixer kinds and masks and are not tail layers (whose stacks differ in shape)."""
    unit, num_units = _scan_units(cfg)
    return _LayerScan(unit, num_units, schedule)


def _scan_units(cfg: GrugModelConfig) -> tuple[tuple[_UnitLayer, ...], int]:
    """``_layer_scan``'s unit and unit count, without the AttnRes schedule."""
    unit_size = cfg.global_every
    kda, tail = set(_kda_layer_indices(cfg)), set(_tail_layers(cfg))

    def signature(i: int) -> tuple[bool, bool, bool]:
        return i in kda, _is_long_layer(i, cfg.num_layers, cfg.global_every, None), i in tail

    unit0 = [signature(j) for j in range(min(unit_size, cfg.num_layers))]
    num_units = 0
    while unit_size * (num_units + 1) <= cfg.num_layers and all(
        signature(unit_size * num_units + j) == unit0[j] and not unit0[j][2] for j in range(unit_size)
    ):
        num_units += 1
    unit = tuple(
        _UnitLayer(is_kda, sum(other[0] == is_kda for other in unit0[:j]), use_long)
        for j, (is_kda, use_long, _) in enumerate(unit0)
    )
    return unit, num_units


def _scan_history_layer(
    layer: Block,
    history: _AttnResHistory,
    queries: Float[Array, "G D"],
    layer_index: jax.Array,
    use_long: bool,
    schedule: _BlockSchedule,
    mask: AttentionMask,
    token_ids: Int[Array, "B S"],
    eps: float,
    route_key: jax.Array | None,
    probes: tuple[jax.Array, ...] | None = None,
    saved_routed: jax.Array | None = None,
) -> tuple[_AttnResHistory, dict[str, jax.Array], tuple[jax.Array, ...]]:
    """``_attn_res_layer`` on the fixed-shape history, at a traced ``layer_index``.

    A block boundary writes the partial into its slot with its logits against every query and resets the
    partial to zero; off a boundary the slot write is a no-op. Unfilled sources get ``-inf`` logits (zero
    weight) and a fresh partial is masked out of the attention gate, so every gate mixes exactly the
    unrolled loop's sources.

    With ``probes`` (the backward's recompute) ``history.sources`` is the final history, whose slot already
    holds this partial, so the slot write is skipped; the source stack is held constant and each gate adds
    its zero probe to its mix and returns its source weights, so the caller rebuilds the source cotangent
    as ``sum_gates weights * d_probe`` in one pass (``_accumulate_source_grads``)."""
    num_sources = history.sources.shape[0]
    block = layer_index // schedule.seg_size
    boundary, slot = _roll_slot(schedule, layer_index)

    def roll(buffer: jax.Array, new: jax.Array) -> jax.Array:
        old = jax.lax.dynamic_index_in_dim(buffer, slot, axis=0, keepdims=False)
        return jax.lax.dynamic_update_index_in_dim(buffer, jnp.where(boundary, new, old), slot, axis=0)

    sources = roll(history.sources, history.partial) if probes is None else history.sources
    plain_sum = layer.attn.cfg.attn_res_plain_sum
    if plain_sum:
        logits = history.logits
    else:
        # Only a boundary writes the new block's logits (off one the slot write is a no-op), so only a boundary
        # computes them: 8 of the hero's 48 layers.
        logits = jax.lax.cond(
            boundary,
            lambda: jax.lax.dynamic_update_index_in_dim(
                history.logits, _attn_res_source_logits(history.partial, queries, eps), slot, axis=0
            ),
            lambda: history.logits,
        )
    partial = jnp.where(boundary, jnp.zeros_like(history.partial), history.partial)
    num_valid = schedule.num_extra + jnp.minimum(block + 1, schedule.block_cap)
    valid = jnp.arange(num_sources) < num_valid
    gate_weights: list[jax.Array] = []
    completed_sum = None
    if plain_sum:
        # attn_res_plain_sum: both gates read the completed sources' plain sum (computed once) plus the partial.
        unit_weights = jnp.broadcast_to(valid[:, None, None].astype(jnp.float32), sources.shape[:3])
        src = sources if probes is None else jax.lax.stop_gradient(sources)
        completed_sum = _weighted_sources(unit_weights, src)
    two_phase = layer.attn.cfg.attn_res_two_phase
    # Phase 1 (two_phase): the attention and MLP gates' mixes of the completed sources in one pass over them.
    inter = None
    if two_phase and not plain_sum:
        gate_logits_2 = jax.lax.dynamic_slice_in_dim(logits, 2 * layer_index, 2, axis=1)  # [N, 2, B, S]
        gate_logits_2 = jnp.where(valid[:, None, None, None], gate_logits_2, -jnp.inf)
        lse = jax.nn.logsumexp(gate_logits_2, axis=0)  # [2, B, S]
        inter_weights = jnp.exp(gate_logits_2 - lse[None])  # [N, 2, B, S]
        src = sources if probes is None else jax.lax.stop_gradient(sources)
        if layer.attn.cfg.attn_res_fused_block_mix:
            normed = _fused_block_mix(src, inter_weights, num_valid)
        else:
            normed = inter_weights[0][..., None] * src[0][None].astype(jnp.float32)
            for n in range(1, num_sources):
                normed = normed + inter_weights[n][..., None] * src[n][None].astype(jnp.float32)
            normed = normed.astype(sources.dtype)
        inter = (normed, lse, inter_weights)
    calls = [0]

    def probed(mixed: jax.Array, source_weights: jax.Array) -> jax.Array:
        """The gate's float32 mix in the stream dtype; in the recompute plus its zero probe (whose cotangent is
        the mix's: the stream-dtype cotangent of the cast) and the gate's source weights recorded."""
        out = mixed.astype(sources.dtype)
        if probes is not None:
            out = out + probes[len(gate_weights)]
            gate_weights.append(jax.lax.stop_gradient(source_weights))
        return reshard(out, _batch_spec())

    def two_phase_mix(p: jax.Array, gate_index: jax.Array, attn_gate: bool, j: int):
        """Phase 2: merge phase-1 mix ``j`` (0: attention gate, 1: MLP gate) with the partial (online softmax)."""
        assert inter is not None and attn_gate == (j == 0)
        normed, lse, inter_weights = inter
        query = jax.lax.dynamic_index_in_dim(queries, gate_index, axis=0, keepdims=True)
        partial_logit = _attn_res_source_logits(p, query, eps)[0]  # [B, S]
        if attn_gate:
            partial_logit = jnp.where(boundary, -jnp.inf, partial_logit)
        total = jnp.logaddexp(lse[j], partial_logit)
        alpha = jnp.exp(lse[j] - total)  # the completed sources' share
        source_weights = inter_weights[:, j] * alpha[None]
        mixed = alpha[..., None] * normed[j].astype(jnp.float32) + (1.0 - alpha)[..., None] * p.astype(jnp.float32)
        out = probed(mixed, source_weights)
        weights = jnp.concatenate([source_weights, (1.0 - alpha)[None]])
        mean_weights = jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim))))
        return out, jnp.mean(jnp.square(total)), mean_weights

    def plain_sum_mix(p: jax.Array):
        """The completed sources' plain sum plus the partial (zero right after a block boundary)."""
        assert completed_sum is not None
        out = probed(completed_sum + p.astype(jnp.float32), unit_weights)
        counted = jnp.concatenate([valid, jnp.ones((1,), bool)]).astype(jnp.float32)
        mean_weights = counted / jnp.sum(counted)
        return out, jnp.zeros((), jnp.float32), mean_weights

    def mix(p: jax.Array | None, gate_index: jax.Array, attn_gate: bool):
        assert p is not None
        if plain_sum:
            return plain_sum_mix(p)
        j = calls[0]
        calls[0] += 1
        if two_phase and j < 2:  # the attention gate, then the MLP gate (``_attn_res_sublayers``' order)
            return two_phase_mix(p, gate_index, attn_gate, j)
        gate = jax.lax.dynamic_index_in_dim(logits, gate_index, axis=1, keepdims=False)  # [N, B, S]
        gate = jnp.where(valid[:, None, None], gate, -jnp.inf)
        query = jax.lax.dynamic_index_in_dim(queries, gate_index, axis=0, keepdims=True)
        partial_logit = _attn_res_source_logits(p, query, eps)  # [1, B, S]
        if attn_gate:
            partial_logit = jnp.where(boundary, -jnp.inf, partial_logit)
        gate_logits = jnp.concatenate([gate, partial_logit])
        weights = jax.nn.softmax(gate_logits, axis=0)
        src = sources if probes is None else jax.lax.stop_gradient(sources)
        mixed = _weighted_sources(weights[:num_sources], src) + weights[num_sources][..., None] * p.astype(jnp.float32)
        out = probed(mixed, weights[:num_sources])
        mean_weights = jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim))))
        z = jnp.mean(jnp.square(jax.nn.logsumexp(gate_logits, axis=0)))
        return out, z, mean_weights

    noise_key = None if route_key is None else jax.random.fold_in(route_key, layer_index)
    extras = None if saved_routed is None else {_SAVED_ROUTED: saved_routed}
    partial, stats = _attn_res_sublayers(
        layer, partial, mix, mask, token_ids, use_long, layer_index, noise_key, None, extras
    )
    if unsupported := set(stats) & set(_DIFFERENTIABLE_STATS):
        raise ValueError(f"scan_layers does not carry the differentiable layer stats {sorted(unsupported)}")
    return _AttnResHistory(sources, logits, partial), stats, tuple(gate_weights)


def _roll_slot(schedule: _BlockSchedule, layer_index: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Whether layer ``layer_index`` rolls the partial into a new block, and that block's history slot."""
    block = layer_index // schedule.seg_size
    boundary = (layer_index % schedule.seg_size == 0) & (block < schedule.block_cap)
    return boundary, schedule.num_extra + jnp.minimum(block, schedule.block_cap - 1)


def _accumulate_source_grads(
    d_sources: jax.Array,
    d_partial: jax.Array,
    gate_weights: tuple[jax.Array, ...],
    d_mixes: tuple[jax.Array, ...],
    schedule: _BlockSchedule,
    layer_index: jax.Array,
    fused: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Add one layer's gate cotangents to the history's source cotangent: ``weights[n] * d_mix`` per gate;
    ``fused`` (``attn_res_fused_source_grads``) runs the stack update as one Pallas kernel.

    At a block boundary the slot was written from the incoming partial, so its cotangent moves onto
    ``d_partial`` and the slot's (overwritten) previous value gets none."""
    boundary, slot = _roll_slot(schedule, layer_index)
    if fused:
        block = layer_index // schedule.seg_size
        num_valid = schedule.num_extra + jnp.minimum(block + 1, schedule.block_cap)
        updated = _fused_source_grads(d_sources, gate_weights, d_mixes, num_valid)

        def roll_back(updated: jax.Array, d_partial: jax.Array) -> tuple[jax.Array, jax.Array]:
            # The kernel already summed the slot's cotangent (one rounding, as below); it moves onto the partial
            # and the slot's overwritten previous value gets none. Off a boundary nothing moves.
            d_slot = jax.lax.dynamic_index_in_dim(updated, slot, axis=0, keepdims=False)
            cleared = jax.lax.dynamic_update_index_in_dim(updated, jnp.zeros_like(d_slot), slot, axis=0)
            return cleared, d_partial + d_slot.astype(d_partial.dtype)

        return jax.lax.cond(boundary, roll_back, lambda u, p: (u, p), updated, d_partial)
    d_slot = jax.lax.dynamic_index_in_dim(d_sources, slot, axis=0, keepdims=False).astype(jnp.float32)
    for weights, d_mix in zip(gate_weights, d_mixes, strict=True):
        d_slot = d_slot + jax.lax.dynamic_index_in_dim(weights, slot, axis=0, keepdims=False)[..., None] * d_mix
    d_partial = d_partial + jnp.where(boundary, d_slot, 0).astype(d_partial.dtype)
    acc = d_sources.astype(jnp.float32)
    for weights, d_mix in zip(gate_weights, d_mixes, strict=True):
        acc = acc + weights[..., None] * d_mix[None]
    rolled = boundary & (jnp.arange(d_sources.shape[0]) == slot)
    return jnp.where(rolled[:, None, None, None], 0, acc).astype(d_sources.dtype), d_partial


def _fused_source_grads(
    d_sources: Float[Array, "N B S D"],
    gate_weights: tuple[jax.Array, ...],
    d_mixes: tuple[jax.Array, ...],
    num_valid: jax.Array,
) -> Float[Array, "N B S D"]:
    """``d_sources + sum_g weights_g[..., None] * d_mix_g[None]`` in the Pallas kernel, per batch shard. Slots
    ``>= num_valid`` have zero weight in every gate and are left as they are."""
    n, _, s, d = d_sources.shape
    stack, rows, tokens = P(None, _BATCH_AXES, None, None), P(None, _BATCH_AXES, None), P(_BATCH_AXES, None, None)

    def local(ds, ws, dms, nv):
        lb = ds.shape[1]
        out = accumulate_source_grads_local(
            ds.reshape(n, lb * s, d),
            tuple(w.reshape(n, lb * s).astype(jnp.float32) for w in ws),
            tuple(m.reshape(lb * s, d) for m in dms),
            nv,
            interpret=_pallas_interpret(),
        )
        return out.reshape(n, lb, s, d)

    kernel = jax.shard_map(
        local,
        mesh=get_abstract_mesh(),
        in_specs=(stack, tuple(rows for _ in gate_weights), tuple(tokens for _ in d_mixes), P()),
        out_specs=stack,
        check_vma=False,
    )
    return kernel(
        reshard(d_sources, stack),
        tuple(reshard(w, rows) for w in gate_weights),
        tuple(reshard(m, tokens) for m in d_mixes),
        num_valid.astype(jnp.int32),
    )


def _pallas_interpret() -> bool:
    """The AttnRes history kernels run in Pallas interpret mode on CPU (the CPU tests of the scan path)."""
    return jax.default_backend() == "cpu"


_HISTORY_SPEC = P(None, _BATCH_AXES, None, None)
_GATE_WEIGHTS_SPEC = P(None, None, _BATCH_AXES, None)


def _history_kernel(local_fn, other_spec: P, out_spec: P):
    """``local_fn(sources, other, num_valid)`` per batch shard of the ``[N, B, S, D]`` history."""
    return jax.shard_map(
        local_fn,
        mesh=get_abstract_mesh(),
        in_specs=(_HISTORY_SPEC, other_spec, P()),
        out_specs=out_spec,
        check_vma=False,
    )


def _block_mix_shard(src: jax.Array, weights: jax.Array, num_valid: jax.Array) -> jax.Array:
    n, lb, s, d = src.shape
    g = weights.shape[1]
    mixed = block_mix_local(
        src.reshape(n, lb * s, d), weights.reshape(n, g, lb * s), num_valid, interpret=_pallas_interpret()
    )
    return mixed.reshape(g, lb, s, d)


def _block_mix_weight_grads_shard(src: jax.Array, d_mixed: jax.Array, num_valid: jax.Array) -> jax.Array:
    n, lb, s, d = src.shape
    g = d_mixed.shape[0]
    d_weights = block_mix_weight_grads_local(
        src.reshape(n, lb * s, d), d_mixed.reshape(g, lb * s, d), num_valid, interpret=_pallas_interpret()
    )
    return d_weights.reshape(n, g, lb, s)


@jax.custom_vjp
def _fused_block_mix(
    sources: Float[Array, "N B S D"], weights: Float[Array, "N G B S"], num_valid: jax.Array
) -> Float[Array, "G B S D"]:
    """Phase-1 AttnRes mixes ``sum_n weights[n, g] * sources[n]`` (float32 sum, sources' dtype out) through
    ``attn_res_block_mix``; ``weights`` must be zero on slots ``>= num_valid``."""
    kernel = _history_kernel(_block_mix_shard, _GATE_WEIGHTS_SPEC, _HISTORY_SPEC)
    return kernel(reshard(sources, _HISTORY_SPEC), reshard(weights, _GATE_WEIGHTS_SPEC), num_valid.astype(jnp.int32))


def _fused_block_mix_fwd(sources, weights, num_valid):
    return _fused_block_mix(sources, weights, num_valid), (sources, weights, num_valid)


def _fused_block_mix_bwd(residuals, d_mixed):
    sources, weights, num_valid = residuals
    kernel = _history_kernel(_block_mix_weight_grads_shard, _HISTORY_SPEC, _GATE_WEIGHTS_SPEC)
    d_weights = kernel(reshard(sources, _HISTORY_SPEC), reshard(d_mixed, _HISTORY_SPEC), num_valid.astype(jnp.int32))
    # Unused (and dropped by XLA) when the sources are held constant, as in the scan's per-layer recompute.
    d_sources = jnp.einsum("ngbs,gbsd->nbsd", weights, d_mixed.astype(jnp.float32)).astype(sources.dtype)
    return d_sources, d_weights, np.zeros(jnp.shape(num_valid), dtype=jax.dtypes.float0)


_fused_block_mix.defvjp(_fused_block_mix_fwd, _fused_block_mix_bwd)


class LayerSkip(NamedTuple):
    """An eval ablation: on ``layers``, tokens with ``keep`` False contribute no attention or MLP output. Their
    K/V are still computed (other tokens attend to them as before), so with ``attn_res_kv_layers`` on those
    layers every other token's outputs are unchanged."""

    layers: tuple[int, ...]
    keep: Bool[Array, "B S"]


class _ScanArgs(NamedTuple):
    """The non-differentiable inputs of ``_scan_layers``."""

    plan: _LayerScan
    units_static: dict[bool, Block]
    long_mask: AttentionMask
    short_mask: AttentionMask
    token_ids: jax.Array
    eps: float
    route_key: jax.Array | None
    offload_partials: bool
    prefetch_partials: bool
    offload_sources: bool
    release_weight_grads: bool


@jax.custom_vjp
def _fused_projection(x: jax.Array, weights: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    """``x @ w`` for every ``w`` of ``weights`` as one GEMM over their concatenation. The backward feeds each
    part's cotangent to its own GEMMs (``dx = sum_i d_i w_i^T``, ``dw_i = x^T d_i``) instead of differentiating
    the split, whose transpose would materialize the concatenated ``[T, sum widths]`` cotangent. Same math."""
    fused = jnp.einsum("td,de->te", x, jnp.concatenate(weights, axis=1), out_sharding=_batch_spec())
    return tuple(jnp.split(fused, list(itertools.accumulate(w.shape[1] for w in weights[:-1])), axis=1))


def _fused_projection_fwd(x: jax.Array, weights: tuple[jax.Array, ...]):
    return _fused_projection(x, weights), (x, weights)


def _fused_projection_bwd(residuals, d_parts: tuple[jax.Array, ...]):
    x, weights = residuals
    dx = functools.reduce(
        jnp.add,
        (
            jnp.einsum(
                "te,de->td",
                d.astype(x.dtype),
                w.astype(x.dtype),
                preferred_element_type=jnp.float32,
                out_sharding=_batch_spec(),
            )
            for d, w in zip(d_parts, weights, strict=True)
        ),
    )
    dws = tuple(
        jnp.einsum("td,te->de", x, d.astype(x.dtype), out_sharding=_partition_spec_of(w) or P(None, None)).astype(
            w.dtype
        )
        for d, w in zip(d_parts, weights, strict=True)
    )
    return dx.astype(x.dtype), dws


_fused_projection.defvjp(_fused_projection_fwd, _fused_projection_bwd)


@jax.custom_vjp
def _saved_value(computed: jax.Array, saved: jax.Array) -> jax.Array:
    """``saved`` as the value and ``computed``'s gradient path: the recompute's output takes the forward's saved
    copy, so nothing reads ``computed`` and XLA drops the ops that only produced it, while the cotangent still
    reaches the ops that produced ``computed`` (their VJP needs no value from it)."""
    del computed
    return saved


def _saved_value_fwd(computed: jax.Array, saved: jax.Array) -> tuple[jax.Array, None]:
    del computed
    return saved, None


def _saved_value_bwd(_, g: jax.Array) -> tuple[jax.Array, jax.Array]:
    return g, jnp.zeros_like(g)


_saved_value.defvjp(_saved_value_fwd, _saved_value_bwd)


def _to_memory_kind(x, memory_kind: str):
    """Every array of the pytree ``x`` moved to ``memory_kind`` (``pinned_host`` / ``device``), keeping its sharding."""
    return jax.tree.map(lambda a: jax.device_put(a, jax.typeof(a).sharding.with_memory_kind(memory_kind)), x)


def _unit_layer(units: dict[bool, Block], layer: _UnitLayer) -> Block:
    return jax.tree.map(lambda x: x[layer.index], units[layer.is_kda])


def _run_unit_layer(
    layer: Block,
    history: _AttnResHistory,
    queries: jax.Array,
    layer_index: jax.Array,
    unit_layer: _UnitLayer,
    args: _ScanArgs,
    probes: tuple[jax.Array, ...] | None = None,
    saved_routed: jax.Array | None = None,
) -> tuple[_AttnResHistory, dict[str, jax.Array], tuple[jax.Array, ...]]:
    return _scan_history_layer(
        layer,
        history,
        queries,
        layer_index,
        unit_layer.use_long,
        args.plan.schedule,
        args.long_mask if unit_layer.use_long else args.short_mask,
        args.token_ids,
        args.eps,
        args.route_key,
        probes,
        saved_routed,
    )


def _layer_history(
    layer_diff, logits, partial, queries, probes, *, sources, layer_static, layer_index, unit_layer, args, saved_routed
):
    """``_run_unit_layer``'s ``(logits, partial)`` as a function of the differentiable inputs (the backward's
    ``jax.vjp``), with the gates' source weights as aux. ``sources`` is the final history, held constant."""
    layer = eqx.combine(_cast_floats(layer_diff, sources.dtype), layer_static)
    history, _, gate_weights = _run_unit_layer(
        layer, _AttnResHistory(sources, logits, partial), queries, layer_index, unit_layer, args, probes, saved_routed
    )
    return (history.logits, history.partial), gate_weights


def _scan_forward(
    units_diff: dict[bool, Block], queries: jax.Array, history: _AttnResHistory, args: _ScanArgs, save_partials: bool
):
    """The scanned layers' forward; with ``save_partials`` also every layer's incoming partial ``[U, g, B, S, D]``."""
    plan = args.plan

    def unit_step(history: _AttnResHistory, xs):
        unit_diff, unit_static, u = xs
        dtype = history.sources.dtype
        units = {kind: eqx.combine(_cast_floats(unit_diff[kind], dtype), unit_static[kind]) for kind in unit_diff}
        partials, stats, routed = [], [], []
        for j, unit_layer in enumerate(plan.unit):
            partials.append(history.partial)
            history, layer_stats, _ = _run_unit_layer(
                _unit_layer(units, unit_layer), history, queries, len(plan.unit) * u + j, unit_layer, args
            )
            if _ROUTED_COMBINE in layer_stats:
                layer_stats = dict(layer_stats)
                routed.append(layer_stats.pop(_ROUTED_COMBINE))
            stats.append(layer_stats)
        if not save_partials:
            return history, (tuple(stats), None)
        saved = (jnp.stack(partials), tuple(routed))
        # Offloaded per unit as the scan produces it, so the stacked residuals live in pinned host memory.
        return history, (tuple(stats), _to_memory_kind(saved, "pinned_host") if args.offload_partials else saved)

    xs = (units_diff, args.units_static, jnp.arange(plan.num_units, dtype=jnp.int32))
    return jax.lax.scan(unit_step, history, xs)


@eqx.filter_custom_vjp
def _scan_layers(diff, args: _ScanArgs):
    """The ``scan_layers`` prefix: ``diff = (units_diff, queries, history)`` -> ``(history, per-unit stats)``.

    A custom VJP saves only the final history and each layer's incoming partial. Completed blocks never
    change, so the final history (masked to the blocks filled so far) is every layer's history input;
    the backward is a reverse scan that rematerializes one layer at a time (``AttnResLayerBackward.RECOMPUTE``)
    and threads the history's cotangents from layer to layer. Layer stats carry no cotangent."""
    units_diff, queries, history = diff
    history, (stats, _) = _scan_forward(units_diff, queries, history, args, save_partials=False)
    return history, stats


@_scan_layers.def_fwd
def _scan_layers_fwd(perturbed, diff, args: _ScanArgs):
    del perturbed
    units_diff, queries, history = diff
    history, (stats, partials) = _scan_forward(units_diff, queries, history, args, save_partials=True)
    sources = _to_memory_kind(history.sources, "pinned_host") if args.offload_sources else history.sources
    return (history, stats), (sources, history.logits, partials)


@_scan_layers.def_bwd
def _scan_layers_bwd(residuals, grad_out, perturbed, diff, args: _ScanArgs):
    del perturbed
    sources, logits, partials = residuals
    if args.offload_sources:
        sources = _to_memory_kind(sources, "device")
    units_diff, queries, _ = diff
    plan = args.plan
    d_history, _ = grad_out
    d_history = _AttnResHistory(
        *(jnp.zeros_like(x) if d is None else d for d, x in zip(d_history, (sources, logits, sources[0]), strict=True))
    )

    def unit_residuals(u):
        return jax.tree.map(lambda a: jax.lax.dynamic_index_in_dim(a, u, axis=0, keepdims=False), partials)

    def unit_step(carry, xs):
        if args.prefetch_partials:
            # Unit u's residuals arrived during unit u + 1's step; start unit u - 1's copy now, so it overlaps
            # this unit's backward instead of stalling the next one.
            d_history, d_queries, (unit_partials, unit_routed) = carry
            unit_diff, unit_static, u = xs
            next_saved = _to_memory_kind(unit_residuals(jnp.maximum(u - 1, 0)), "device")
        else:
            d_history, d_queries = carry
            unit_diff, unit_static, (unit_partials, unit_routed), u = xs
            if args.offload_partials:
                unit_partials, unit_routed = _to_memory_kind((unit_partials, unit_routed), "device")
        d_unit: dict[bool, list[Block | None]] = {kind: [None] * plan.kind_count(kind) for kind in unit_diff}
        pending: tuple[_UnitLayer, Block] | None = None
        for j in reversed(range(len(plan.unit))):
            unit_layer = plan.unit[j]
            layer_diff = _unit_layer(unit_diff, unit_layer)
            layer_static = _unit_layer(unit_static, unit_layer)
            layer_index = len(plan.unit) * u + j

            with jax.named_scope(f"attn_res_scan_bwd_j{j}"):
                # Barriers as in _attn_res_layer_remat_bwd: one layer's backward, weight gradients included,
                # finishes before the next one's starts.
                layer_diff, partial_in, d_history = jax.lax.optimization_barrier(
                    (layer_diff, unit_partials[j], d_history)
                )
                layer_fn = functools.partial(
                    _layer_history,
                    sources=sources,
                    layer_static=layer_static,
                    layer_index=layer_index,
                    unit_layer=unit_layer,
                    args=args,
                    saved_routed=unit_routed[j] if unit_routed else None,
                )
                num_gates = 2 + (layer_diff.attn_res_query_shared is not None)
                probes = tuple(jnp.zeros_like(partial_in) for _ in range(num_gates))
                _, vjp_fn, gate_weights = jax.vjp(
                    layer_fn, layer_diff, logits, partial_in, queries, probes, has_aux=True
                )
                d_layer, d_logits, d_partial, d_q, d_mixes = vjp_fn((d_history.logits, d_history.partial))
                d_sources, d_partial = _accumulate_source_grads(
                    d_history.sources,
                    d_partial,
                    gate_weights,
                    d_mixes,
                    plan.schedule,
                    layer_index,
                    fused=layer_static.attn.cfg.attn_res_fused_source_grads,
                )
                d_history = _AttnResHistory(d_sources, d_logits, d_partial)
                if args.release_weight_grads and pending is None:
                    d_history, d_q = jax.lax.optimization_barrier((d_history, d_q))
                elif args.release_weight_grads:
                    # The previous layer's weight gradients (and their FSDP reduce-scatters) finish under this
                    # layer's backward: one layer late, so at most two layers' gradients are in flight.
                    prev_layer, prev_grads = pending
                    prev_grads, d_history, d_q = jax.lax.optimization_barrier((prev_grads, d_history, d_q))
                    d_unit[prev_layer.is_kda][prev_layer.index] = prev_grads
                else:
                    d_layer, d_history, d_q = jax.lax.optimization_barrier((d_layer, d_history, d_q))
            d_queries = d_queries + d_q
            if args.release_weight_grads:
                pending = (unit_layer, d_layer)
            else:
                d_unit[unit_layer.is_kda][unit_layer.index] = d_layer
        if pending is not None:
            last_layer, last_grads = pending
            d_unit[last_layer.is_kda][last_layer.index] = last_grads
        stacked = {kind: jax.tree.map(lambda *xs: jnp.stack(xs), *layers) for kind, layers in d_unit.items()}
        if args.prefetch_partials:
            return (d_history, d_queries, next_saved), stacked
        return (d_history, d_queries), stacked

    units = jnp.arange(plan.num_units, dtype=jnp.int32)
    if args.prefetch_partials:
        last = _to_memory_kind(unit_residuals(plan.num_units - 1), "device")
        xs = (units_diff, args.units_static, units)
        (d_history, d_queries, _), d_units = jax.lax.scan(
            unit_step, (d_history, jnp.zeros_like(queries), last), xs, reverse=True
        )
        return d_units, d_queries, d_history
    xs = (units_diff, args.units_static, partials, units)
    (d_history, d_queries), d_units = jax.lax.scan(unit_step, (d_history, jnp.zeros_like(queries)), xs, reverse=True)
    return d_units, d_queries, d_history


def _cast_floats(tree, dtype: jnp.dtype):
    """Cast the floating leaves of ``tree`` to ``dtype`` (``scan_layers`` casts each layer's fp32 weights at use)."""
    return jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) and x.dtype != dtype else x, tree)


def _unit_queries(scan: _LayerScan | None, units: dict[bool, Block], *names: str) -> jax.Array:
    """The scanned layers' pseudo-queries ``names`` in gate order (layer-major): ``[U * g * len(names), D]``."""
    assert scan is not None
    per_layer = [
        jnp.stack([getattr(units[layer.is_kda], name)[:, layer.index] for name in names], axis=1) for layer in scan.unit
    ]
    stacked = jnp.stack(per_layer, axis=1)  # [U, g, len(names), D]
    return stacked.reshape(-1, stacked.shape[-1])


def _index_leading(x: jax.Array, index: int) -> jax.Array:
    """``x[index]`` keeping the trailing dims' sharding (plain indexing drops it on size-1 sharded dims)."""
    return reshard(x[index], P(*jax.typeof(x).sharding.spec[1:]))


def _strip_unread_weights(stats: dict[str, jax.Array], schedule: _BlockSchedule, eff: int, num_sources: int) -> None:
    """Drop a scanned layer's zero gate weights on sources it cannot read yet (and on the partial right after a
    boundary), so its weight logs match the unrolled loop's."""
    filled = schedule.num_sources(eff)
    for name in (_ATTN_RES_W_ATTN, _ATTN_RES_W_MLP, _ATTN_RES_W_V):
        if name not in stats:
            continue
        has_partial = name == _ATTN_RES_W_MLP or not schedule.is_boundary(eff)
        weights = stats[name]
        stats[name] = jnp.concatenate([weights[:filled], weights[num_sources:]]) if has_partial else weights[:filled]


# The plain layer scan's per-layer residual-stream input, named so a remat policy can offload it
# (``layer_carry_offload``).
LAYER_CARRY_REMAT_NAME = "grug_layer_carry"


def _carry_named_block(
    layer: "Block", x: Float[Array, "B S D"], mask: AttentionMask, use_long: jax.Array
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """One plain-scan layer whose input carries ``LAYER_CARRY_REMAT_NAME``: a remat policy acts on named
    intermediates, not on the checkpoint's arguments, so every use of the input reads it through the name."""
    return layer(checkpoint_name(x, LAYER_CARRY_REMAT_NAME), mask, use_long, use_long)


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
    stacked_blocks_tail: ArrayStacked[Block] | None
    """Softmax-attention layers of ``latent_out_full_layers``, whose routed experts write the full stream."""
    kda_blocks_tail: ArrayStacked[Block] | None
    """KDA layers of ``latent_out_full_layers``."""
    stacked_blocks_rest: ArrayStacked[Block] | None
    """Softmax-attention layers of ``_rest_layers`` (``scan_layers`` only)."""
    kda_blocks_rest: ArrayStacked[Block] | None
    """KDA layers of ``_rest_layers`` (``scan_layers`` only)."""
    final_norm: LearnedRMSNorm
    final_gated_norm: GatedNorm | None
    attn_res_query_final: Float[Array, " D"] | None
    """Pseudo-query of the final AttnRes gate, whose mix feeds the final norms and the lm_head."""
    entropy_head: Float[Array, " D"] | None  # cfg.entropy_head_layers: predicts the next-token entropy
    entropy_head_bias: Float[Array, ""] | None
    skip_adapter: DenseMLP | None  # cfg.skip_adapter_dim: the skipped tokens' readout MLP
    skip_router: Float[Array, " D"] | None  # cfg.skip_router_layers: the skip router's score
    skip_router_bias: Float[Array, ""] | None
    skip_probe_w1: Float[Array, "3 D H"] | None  # cfg.skip_probe_layer probes (f0, f1, g)
    skip_probe_b1: Float[Array, "3 H"] | None
    skip_probe_w2: Float[Array, "3 H"] | None
    skip_probe_b2: Float[Array, " 3"] | None
    final_conf_w1: Float[Array, "N D H"] | None  # cfg.final_conf_layers probes
    final_conf_b1: Float[Array, "N H"] | None
    final_conf_w2: Float[Array, "N H"] | None
    final_conf_b2: Float[Array, " N"] | None
    mid_output_proj: jax.Array | None  # cfg.mid_head_layer: the mid lm_head
    mid_conf_w1: Float[Array, "F H"] | None  # the confidence head: features (2D + 2) -> H -> 1
    mid_conf_b1: Float[Array, " H"] | None
    mid_conf_w2: Float[Array, " H"] | None
    mid_conf_b2: Float[Array, ""] | None
    token_embed2: jax.Array | None
    embed2_norm: LearnedRMSNorm | None
    bigram_gate_w: Float[Array, " D"] | None
    bigram_gate_b: Float[Array, ""] | None
    bigram_gate_a_lr: Float[Array, "D R"] | None
    bigram_gate_b_lr: Float[Array, "R D"] | None
    embed2_up: jax.Array | None
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
            _init_weight(
                out_key,
                (cfg.hidden_dim + (0), cfg.vocab_size * (1)),
                cfg.initializer_std,
            ),
            _LM_HEAD_PARTITION_SPEC,
        )

        # With a wider final layer every layer pads its per-expert stats to the widest layer's expert count.
        layer_base = cfg

        def stack(
            layers: tuple[int, ...], use_kda: bool, layer_cfg: GrugModelConfig = layer_base
        ) -> ArrayStacked[Block]:
            keys = jnp.stack([block_keys[i] for i in layers])
            layer_index = jnp.asarray(layers, dtype=jnp.int32)
            return ArrayStacked.init(len(layers), Block)(layer_cfg, key=keys, layer_index=layer_index, use_kda=use_kda)

        softmax_layers, kda_layers = _stack_layer_indices(cfg)
        softmax_tail, kda_tail = _tail_stack_layer_indices(cfg)
        softmax_rest, kda_rest = _rest_stack_layer_indices(cfg)
        # The tail layers' config override is static per layer (MoEMLP.cfg); without tail layers there is none
        # (and dense models could not even build one).
        tail_cfg = _tail_layer_config(layer_base) if softmax_tail or kda_tail else layer_base
        body_cfg = layer_base
        model = Transformer(
            token_embed=token_embed,
            embed_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps, role="embed"),
            embed_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key) if cfg.embed_gated_norm else None
            ),
            output_proj=output_proj,
            stacked_blocks=stack(softmax_layers, False, body_cfg),
            kda_blocks=stack(kda_layers, True, body_cfg) if kda_layers else None,
            stacked_blocks_tail=(
                stack(softmax_tail, False, _final_layer_config(cfg, tail_cfg)) if softmax_tail else None
            ),
            kda_blocks_tail=stack(kda_tail, True, tail_cfg) if kda_tail else None,
            stacked_blocks_rest=stack(softmax_rest, False, body_cfg) if softmax_rest else None,
            kda_blocks_rest=stack(kda_rest, True, body_cfg) if kda_rest else None,
            final_norm=_learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps, role="final"),
            final_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key) if cfg.final_gated_norm else None
            ),
            attn_res_query_final=(
                reshard(jnp.zeros((_attn_res_key_dim(cfg),), jnp.float32), P(None)) if cfg.attn_res else None
            ),
            entropy_head=(
                reshard(jnp.zeros((cfg.hidden_dim,), jnp.float32), P(None)) if cfg.entropy_head_layers else None
            ),
            entropy_head_bias=jnp.zeros((), jnp.float32) if cfg.entropy_head_layers else None,
            skip_adapter=(
                DenseMLP.init(cfg.hidden_dim, cfg.skip_adapter_dim, cfg.initializer_std, key=random.fold_in(key, 9))
                if cfg.skip_adapter_dim
                else None
            ),
            skip_router=(
                reshard(
                    random.normal(random.fold_in(key, 6), (cfg.hidden_dim,), jnp.float32) * cfg.initializer_std,
                    P(None),
                )
                if cfg.skip_router_layers
                else None
            ),
            skip_router_bias=jnp.zeros((), jnp.float32) if cfg.skip_router_layers else None,
            **_mid_head_params(cfg, output_proj.shape, random.fold_in(key, 5)),
            **_final_conf_params(cfg, random.fold_in(key, 7)),
            **_probe_mlp_params("skip_probe", 3 if cfg.skip_probe_layer is not None else 0, cfg, random.fold_in(key, 8)),
            token_embed2=(
                reshard(
                    _init_weight(
                        embed2_key,
                        (
                            (cfg.embed2_rows or cfg.vocab_size) * (1),
                            (cfg.embed2_dim or cfg.hidden_dim) // (1),
                        ),
                        cfg.initializer_std,
                    ),
                    P(_FSDP_AXES, None) if cfg.embed2_fsdp else P(None, None),
                )
                if cfg.second_embed
                else None
            ),
            embed2_norm=(
                _learned_rms_norm(cfg, cfg.hidden_dim, cfg.layer_norm_eps, role="embed2") if cfg.second_embed else None
            ),
            bigram_gate_w=jnp.zeros((cfg.hidden_dim,), jnp.float32) if cfg.bigram_gate else None,
            bigram_gate_b=jnp.full((), 2.0, jnp.float32) if cfg.bigram_gate else None,
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
            config=cfg,
        )
        return model

    def layer_stacks(self) -> list[ArrayStacked[Block]]:
        """The block stacks; stack ``k`` holds layers ``stack_layer_indices()[k]``."""
        stacks = (
            self.stacked_blocks,
            self.kda_blocks,
            self.stacked_blocks_rest,
            self.kda_blocks_rest,
            self.stacked_blocks_tail,
            self.kda_blocks_tail,
        )
        return [s for s in stacks if s is not None]

    def stack_layer_indices(self) -> list[tuple[int, ...]]:
        """The layer indices held by each of ``layer_stacks()``, in stack order."""
        groups = (
            *_stack_layer_indices(self.config),
            *_rest_stack_layer_indices(self.config),
            *_tail_stack_layer_indices(self.config),
        )
        return [indices for indices in groups if indices]

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
        layer_skip: "LayerSkip | None" = None,
        extra_mlp_active: jax.Array | bool = True,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``extra_mlp_active`` (traced bool) turns on the ``extra_mlp_layers`` MLPs (off: no token reaches them).
        ``layer_skip`` (an eval ablation, AttnRes only) drops the listed layers' attention and MLP outputs for
        the tokens it marks.
        ``loop_active`` (static, with ``loop_grow_step``) selects one pass (False) or all ``loop_passes``
        (True); None runs all passes. ``route_key`` (training only) seeds ``moe_gumbel_tau``.
        ``router_tie_active`` (static) applies the ``router_embed_tie`` ties; None applies them unless
        ``router_embed_tie_release_step`` is set (a released model's router columns already hold them).
        ``return_routing`` adds every layer's ``[L, T, K]`` selected experts and combine weights to the
        metrics (``ROUTING_SELECTED_KEY`` / ``ROUTING_WEIGHTS_KEY``)."""
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        gather = _embedding_gather
        hidden = gather(self.token_embed, token_ids)
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
        bigram_gate_stats: dict[str, jax.Array] = {}
        if (cfg.second_embed or (0)) and not cfg.attn_res:
            raise ValueError("second_embed and ple_dim require attn_res")
        if cfg.attn_res:
            extra_sources = ()
            if self.token_embed2 is not None:
                assert self.embed2_norm is not None
                ids2 = token_ids
                gather2 = _table2_gather(cfg)
                table2 = self.token_embed2
                if cfg.second_embed_bigram:
                    doc_start = None if segment_ids is None else segment_ids[0]
                    ids2 = _bigram_hash_ids(token_ids, doc_start, cfg.embed2_rows or cfg.vocab_size, cfg.embed2_ngram)
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
                        act=(ContentGateActivation.NONE),
                        logit_scale=(1.0),
                    )
                    bigram_gate_stats = {
                        "attn_res_bigram_gate_mean": jax.lax.stop_gradient(jnp.mean(gate)),
                        "attn_res_bigram_gate_std": jax.lax.stop_gradient(jnp.std(gate)),
                    }
                extra_sources = (embed2,)
            hidden, stacked_router_stats, final_gate_stats = self._attn_res_layers(
                hidden,
                token_ids,
                extra_sources,
                loop_active,
                long_mask.with_fa4_bounds(long_lower_bounds, valid),
                long_mask.with_fa4_bounds(short_lower_bounds, valid),
                route_key,
                (None),
                (None),
                layer_skip,
                extra_mlp_active,
            )
        else:
            if layer_skip is not None:
                raise ValueError("layer_skip needs attn_res")
            # One compiled Block body scanned over the stacked layers; per-layer short/long is a
            # Bool[num_layers] scan input, and the FA4 metadata is selected per layer with jnp.where.
            mask_schedule = _long_layer_schedule(cfg.num_layers, cfg.global_every, (None))

            def _scan_layers(
                carry_hidden: Float[Array, "B S D"],
                scan_inputs: tuple[Block, jax.Array],
            ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
                layer, layer_use_long_mask = scan_inputs
                use_long = jnp.asarray(layer_use_long_mask, dtype=jnp.bool_)
                lower_bounds = jnp.where(use_long, long_lower_bounds, short_lower_bounds)
                layer_mask = long_mask.with_fa4_bounds(lower_bounds, valid)
                return eqx.filter_checkpoint(_carry_named_block, policy=carry_policy)(
                    layer, carry_hidden, layer_mask, use_long
                )

            carry_policy = (
                jax.checkpoint_policies.save_and_offload_only_these_names(
                    names_which_can_be_saved=[],
                    names_which_can_be_offloaded=[LAYER_CARRY_REMAT_NAME],
                    offload_src="device",
                    offload_dst="pinned_host",
                )
                if cfg.layer_carry_offload
                else None
            )

            hidden, stacked_router_stats = jax.lax.scan(
                _scan_layers, hidden, xs=(self.stacked_blocks.stacked, mask_schedule)
            )
        if cfg.dense_mlp:
            router_metrics: dict[str, jax.Array] = {}
            for key in (HEAD_PROBE_STAT, HEAD_QCOS_STAT):
                if not cfg.attn_res and key in stacked_router_stats:
                    router_metrics[key] = stacked_router_stats[key]  # [L, B, S, H, F] / [L, 2, H, H]
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
            router_metrics.update(
                {
                    k: v
                    for k, v in stacked_router_stats.items()
                    if k == _COMPRESS_LOSS or k.startswith((_COMPRESS_PREFIX, _OUTSTAT_PREFIX))
                }
            )
            for key in (HEAD_PROBE_STAT, HEAD_QCOS_STAT):
                if key in stacked_router_stats:
                    router_metrics[key] = stacked_router_stats[key]  # [L, B, S, H, F] / [L, 2, H, H]
            if return_routing:
                router_metrics[ROUTING_SELECTED_KEY] = stacked_router_stats[_ROUTING_SELECTED]
                router_metrics[ROUTING_WEIGHTS_KEY] = stacked_router_stats[_ROUTING_WEIGHTS]
                router_metrics[ROUTED_INPUT_KEY] = stacked_router_stats[_ROUTED_INPUT]
                if _SHARED_PRE in stacked_router_stats:
                    router_metrics[SHARED_PRE_KEY] = stacked_router_stats[_SHARED_PRE]
        router_metrics.update(final_gate_stats)
        router_metrics.update(bigram_gate_stats)
        router_metrics.pop(_LM_HEAD_EXTRA, None)
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
        route_key: jax.Array | None = None,
        input_embed2: jax.Array | None = None,
        ple_rows: jax.Array | None = None,
        layer_skip: "LayerSkip | None" = None,
        extra_mlp_active: jax.Array | bool = True,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], dict[str, jax.Array]]:
        """Block AttnRes over the layers, unrolled so each gate reads only its valid sources.

        The residual history is a Python tuple of completed block sums (the token embedding is the
        first, rolled in at layer 0) plus the running partial; every ``seg_size``-th layer rolls its
        incoming partial into a new block. Completed blocks are immutable, so each block's logits
        against the queries of every gate that reads it (layer ``i``'s onwards and the final gate) are
        computed once when it is rolled, and each gate then only scores the partial. Blocks are shared
        by reference, so block memory is one copy per block. With ``AttnResLayerBackward.RECOMPUTE``
        each layer is rematerialized by ``_attn_res_layer_remat``, which saves only its inputs.

        With ``scan_layers`` the leading run of identical ``global_every``-layer units runs as one scan on the
        fixed-shape history instead (``_scan_layers``, same math), and the remaining layers continue unrolled.

        ``ple_rows`` (``ple_dim``) holds every layer's per-layer-embedding rows side by side; layer ``i``
        gets its RMS-normed slice through its ``extras["ple"]``.

        Returns the final gate's mix, the per-layer router stats and the final gate's mean max weight
        and mean entropy.
        """
        cfg = self.config
        assert self.attn_res_query_final is not None
        eps = cfg.layer_norm_eps
        num_layers, passes = cfg.num_layers, (1)
        seg_size = max(1, num_layers // cfg.attn_res_num_blocks)
        block_cap = cfg.attn_res_num_blocks * passes
        schedule = _BlockSchedule(len(extra_sources), seg_size, block_cap)
        scan = _layer_scan(cfg, schedule) if cfg.scan_layers else None
        dtype = self.token_embed.dtype
        if scan is not None and scan.num_units:
            units, layers = self._split_scanned_units(scan)
        else:
            units, layers = {}, self.layers()
        # Every query in gate order [pass 0: attn_0, mlp_0, ..., mlp_{L-1}; pass 1: ...; final].
        gate_queries = (
            [_cast_floats(_unit_queries(scan, units, "attn_res_query_attn", "attn_res_query_mlp"), dtype)]
            if units
            else []
        )
        for layer in layers[scan.num_layers if units else 0 :]:
            assert layer.attn_res_query_attn is not None and layer.attn_res_query_mlp is not None
            gate_queries.append(jnp.stack([layer.attn_res_query_attn, layer.attn_res_query_mlp]))
        # V-gate queries sit between the per-layer gates and the final gate, so every block's precomputed
        # logits (scored against queries[2i:]) cover them; V gate of layer i is row 2 * L * passes + i.
        v_queries = []
        if cfg.attn_res_shared_query:
            v_queries = [_cast_floats(_unit_queries(scan, units, "attn_res_query_shared"), dtype)] if units else []
            v_queries += [layer.attn_res_query_shared[None] for layer in layers[scan.num_layers if units else 0 :]]
        # KV-gate queries (attn_res_kv_layers) follow the V gates: KV gate of layer i is row
        # 2 * L + (L if attn_res_shared_query) + i.
        kv_queries = []
        if cfg.attn_res_kv_layers:
            kv_queries = [layer.attn_res_query_kv[None] for layer in layers]
        queries_flat = jnp.concatenate([*gate_queries, *v_queries, *kv_queries, self.attn_res_query_final[None]])
        queries = queries_flat
        logit_bias = _gate_extras(self, queries.shape[0])
        allowed = {"q", "k", "v", "mlp", "mlp_shared", "mlp_routed", "mlp_router"}
        if set(()) - allowed:
            raise ValueError(f"attn_res_sum_inputs must be a subset of {sorted(allowed)}, got {()}")
        if not set(()) <= set(range(1, num_layers)):
            raise ValueError(f"value_residual_layers must be in 1..{num_layers - 1}, got {()}")
        layer_fn = (
            _attn_res_layer_remat
            if cfg.attn_res_layer_backward == AttnResLayerBackward.RECOMPUTE
            else _attn_res_layer_passthrough
        )

        weight_logs: dict[int, tuple[jax.Array, bool]] = {}
        layer_logs: dict[str, jax.Array] = {}
        probe_logs: dict[str, jax.Array] = {}
        kv_inputs = None
        nitp_logs: dict[str, jax.Array] = {}

        def absorb(eff: int, stats: dict[str, jax.Array], has_partial_attn: bool) -> None:
            """Move layer ``eff``'s z term, gate weights and per-layer logs out of its stats."""
            z_out.append(stats.pop(_ATTN_RES_Z))
            stats.pop(ROUTER_PROBS_KEY, None)
            if _LM_HEAD_EXTRA in stats:
                nitp_logs[_LM_HEAD_EXTRA] = stats.pop(_LM_HEAD_EXTRA)
            for name in [k for k in stats if k.startswith(_LAYER_KNOB_PREFIX)]:
                layer_logs[f"{name}_L{eff}"] = stats.pop(name)
            weight_logs[2 * eff] = (stats.pop(_ATTN_RES_W_ATTN), has_partial_attn)
            weight_logs[2 * eff + 1] = (stats.pop(_ATTN_RES_W_MLP), True)
            if _ATTN_RES_W_V in stats:
                # V-gate query rows follow the per-layer gates (loop_passes == 1).
                weight_logs[2 * num_layers + eff] = (stats.pop(_ATTN_RES_W_V), has_partial_attn)
            stats_out.append(stats)

        stats_out: list[dict[str, jax.Array]] = []
        z_out: list[jax.Array] = []
        extra_state: dict[str, jax.Array] = {}

        def run_pass(state, pass_index, start=0):
            """One pass over the physical layers from ``start``, extending the history; returns the new state."""
            blocks, block_logits, partial = state
            kv_share: dict[str, jax.Array] | None = None
            for i in range(start, num_layers):
                layer = layers[i]
                eff = pass_index * num_layers + i
                if schedule.is_boundary(eff):
                    assert partial is not None
                    blocks = (*blocks, partial)
                    # Score the new block only against the gates that can read it (this layer's onwards).
                    block_logits = (
                        *block_logits,
                        _logit_rows(_attn_res_source_logits(partial, queries[2 * eff :], eps, (False))),
                    )
                    partial = None
                use_long = _is_long_layer(i, num_layers, cfg.global_every, (None))
                partial_before = partial
                layer_extras = logit_bias
                if ple_rows is not None:
                    ple_i = rms_norm(ple_rows[..., i * (0) : (i + 1) * (0)], eps)
                    layer_extras = {**(logit_bias or {}), "ple": ple_i}
                if kv_inputs is not None:
                    layer_extras = {**(layer_extras or {}), "kv_stream": kv_inputs[i]}
                if i == cfg.skip_probe_layer and pass_index == 0:
                    assert self.skip_probe_w1 is not None and self.skip_probe_b1 is not None
                    assert self.skip_probe_w2 is not None and self.skip_probe_b2 is not None
                    stream = functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])
                    extra_state["skip_probe"] = _probe_mlp_logits(
                        self.skip_probe_w1, self.skip_probe_b1, self.skip_probe_w2, self.skip_probe_b2, stream, eps
                    )
                    skipped = (
                        jnp.zeros(token_ids.shape, jnp.float32)
                        if layer_skip is None or i not in layer_skip.layers
                        else 1.0 - layer_skip.keep.astype(jnp.float32)
                    )
                    extra_state["skip_probe_d"] = skipped
                if i in cfg.final_conf_layers and pass_index == 0:
                    stream = functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])
                    extra_state.setdefault("final_conf", []).append(
                        self._final_conf_logit(cfg.final_conf_layers.index(i), stream)
                    )
                if cfg.skip_router_layers and i == min(cfg.skip_router_layers) and pass_index == 0:
                    stream = functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])
                    extra_state["router_keep"], extra_state["router_frac"] = self._skip_router_keep(stream)
                if i == cfg.mid_head_layer and pass_index == 0:
                    extra_state["mid"] = functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])
                    if cfg.mid_skip_layers:
                        _, logit = self._mid_confidence(jax.lax.stop_gradient(extra_state["mid"]))
                        skip = (jax.nn.sigmoid(jax.lax.stop_gradient(logit)) > cfg.mid_skip_threshold) & jnp.asarray(
                            extra_mlp_active
                        )
                        extra_state["mid_skip_keep"] = (~skip).astype(jnp.float32)
                        extra_state["mid_skip_frac"] = jnp.mean(skip.astype(jnp.float32))
                if cfg.entropy_head_layers and i == cfg.entropy_head_read_layer and pass_index == 0:
                    stream = functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])
                    extra_state["pred"], extra_state["keep"] = self._extra_mlp_selection(
                        stream, token_ids, extra_mlp_active
                    )
                if i in cfg.extra_mlp_layers:
                    layer_extras = {**(layer_extras or {}), _EXTRA_KEEP: extra_state["keep"]}
                keeps = []
                if layer_skip is not None and i in layer_skip.layers:
                    keeps.append(layer_skip.keep.astype(jnp.float32))
                if i in cfg.entropy_skip_layers:
                    keeps.append((~extra_state["keep"]).astype(jnp.float32))
                if i in cfg.skip_router_layers:
                    keeps.append(extra_state["router_keep"])
                if i in cfg.mid_skip_layers:
                    keeps.append(extra_state["mid_skip_keep"])
                skip_keep = functools.reduce(jnp.multiply, keeps) if keeps else None
                if skip_keep is not None:
                    layer_extras = {**(layer_extras or {}), _SKIP_KEEP: skip_keep}
                layer_args = (
                    (layer, blocks, block_logits, partial, queries, layer_extras),
                    long_layer_mask if use_long else short_layer_mask,
                    token_ids,
                    use_long,
                    eff,
                    eps,
                    None if route_key is None else jax.random.fold_in(route_key, eff),
                )
                if kv_share is None:
                    partial, blocks, block_logits, stats = layer_fn(*layer_args)
                else:
                    partial, blocks, block_logits, stats = _attn_res_layer_passthrough(*layer_args, kv_share)
                absorb(eff, stats, partial_before is not None)
            return blocks, block_logits, partial

        def final_gate(state):
            blocks, block_logits, partial = state
            final_index = queries.shape[0] - 1
            logits = [_block_logit(bl, queries, final_index) for bl in block_logits]
            logits.append(_attn_res_source_logits(partial, queries[final_index][None], eps, (False))[0])
            logits = _bias_gate_logits(logits, logit_bias, final_index, [*blocks, partial], eps, has_partial=True)
            logits = _soft_cap_logits(logits, (None))
            weights, mixed = _softmax_mix(logits, [*blocks, partial])
            if cfg.attn_res_plain_sum:
                mixed = _stream_sum((*blocks, partial))
                weights = jnp.full_like(weights, 1.0 / weights.shape[0])
            spot = _probe_spot()
            if spot is not None:
                probe_logs[f"{PROBE_STAT}final_sources"] = jnp.stack([_at_spot(src, spot) for src in [*blocks, partial]])
                probe_logs[f"{PROBE_STAT}final_gate_logits"] = jnp.stack([_at_spot(lg, spot) for lg in logits])
                probe_logs[f"{PROBE_STAT}final_gate_weights"] = jnp.stack([_at_spot(w, spot) for w in weights])
                probe_logs[f"{PROBE_STAT}final_query"] = jax.lax.stop_gradient(queries[final_index])
            mixed = _gate_variants(mixed, [*blocks, partial], logit_bias, final_index, eps, has_partial=True)
            weight_logs[final_index] = (
                jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim)))),
                True,
            )
            max_weight = jax.lax.stop_gradient(jnp.mean(jnp.max(weights, axis=0)))
            entropy = jax.lax.stop_gradient(jnp.mean(-jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-30)), axis=0)))
            if spot is not None:
                probe_logs[f"{PROBE_STAT}final_mixed"] = _at_spot(mixed, spot)
            if trace_capture.TRACE:
                probe_logs[f"{PROBE_STAT}trace_final_sources"] = jnp.stack([*blocks, partial])
                probe_logs[f"{PROBE_STAT}trace_final_weights"] = jnp.asarray(weights)
                probe_logs[f"{PROBE_STAT}trace_final_mixed"] = mixed
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

        if units:
            assert scan is not None
            history, unit_stats = self._run_scanned_units(
                scan, units, queries, hidden, extra_sources, long_layer_mask, short_layer_mask, token_ids, route_key
            )
            for u in range(scan.num_units):
                for j in range(len(scan.unit)):
                    eff = len(scan.unit) * u + j
                    stats = jax.tree.map(lambda x, u=u: _index_leading(x, u), unit_stats[j])
                    _strip_unread_weights(stats, schedule, eff, history.sources.shape[0])
                    absorb(eff, stats, not schedule.is_boundary(eff))
            # The rest of the layers run unrolled on the history as tuples (every block's logits cover all gates).
            state = (
                _logit_rows(history.sources),
                tuple(_logit_rows(rows) for rows in _logit_rows(history.logits)),
                history.partial,
            )
            state = run_pass(state, 0, start=scan.num_layers)
        else:
            state = (
                extra_sources,
                tuple(_attn_res_source_logits(src, queries, eps, (False)) for src in extra_sources),
                hidden,
            )
            state = run_pass(state, 0)
        mixed, layer_stats, z_total, max_weight, entropy = finish(state, [stats_out], z_out)
        if self.skip_adapter is not None and layer_skip is not None:
            adapted = self.skip_adapter(rms_norm(mixed, eps), moe_output_reshard=False)
            skipped = (~layer_skip.keep)[..., None].astype(mixed.dtype)
            mixed = mixed + skipped * adapted.astype(mixed.dtype)
        final_stats = {"attn_res_final_max_weight": max_weight, "attn_res_final_entropy": entropy}
        if "pred" in extra_state:
            final_stats[_ENTROPY_PRED] = extra_state["pred"]
        if "mid" in extra_state:
            final_stats[_MID_HIDDEN] = extra_state["mid"]
        if "skip_probe" in extra_state:
            f0, f1, g = extra_state["skip_probe"]
            final_stats[_SKIP_PROBE_LOGITS] = jnp.stack([f0 + extra_state["skip_probe_d"] * f1, g, f0, f0 + f1])
        if "final_conf" in extra_state:
            final_stats[_FINAL_CONF_LOGITS] = jnp.stack(extra_state["final_conf"])
        if "mid_skip_frac" in extra_state:
            final_stats["attn_res_mid_skip_frac"] = extra_state["mid_skip_frac"]
        if "router_frac" in extra_state:
            final_stats["attn_res_skip_router_frac"] = extra_state["router_frac"]
        # Source n is block n (the embedding is block len(extra_sources)); "p" is the running partial.
        for gate, (mean_weights, has_partial) in sorted(weight_logs.items()):
            num_blocks = mean_weights.shape[0] - int(has_partial)
            for n in range(num_blocks):
                final_stats[f"attn_res_w_g{gate:02d}_b{n:02d}"] = mean_weights[n]
            if has_partial:
                final_stats[f"attn_res_w_g{gate:02d}_p"] = mean_weights[-1]
        for i, layer in enumerate(self.layers() if units else layers):
            if isinstance(layer.attn, CausalSelfAttention) and layer.attn.qk_mult is not None:
                qk_mult = jax.lax.stop_gradient(layer.attn.qk_mult)
                final_stats[f"attn_res_qk_mult_L{i}"] = jnp.mean(qk_mult)
                if qk_mult.ndim:
                    final_stats[f"attn_res_qk_mult_min_L{i}"] = jnp.min(qk_mult)
                    final_stats[f"attn_res_qk_mult_max_L{i}"] = jnp.max(qk_mult)
            final_stats.update(_learned_knob_stats(layer, i))
        final_stats.update(layer_logs)
        final_stats.update(probe_logs)
        for name, norm in (("embed", self.embed_norm), ("final", self.final_norm)):
            if isinstance(norm, ZeroCenteredRMSNorm):
                final_stats[f"attn_res_knob_gain_abs_{name}"] = jnp.mean(jnp.abs(jax.lax.stop_gradient(norm.gamma)))
        for i, memory in zip((), (), strict=True):
            w_out = jax.lax.stop_gradient(memory.w_out).astype(jnp.float32)
            final_stats[f"{_MEMORY_STAT_PREFIX}out_norm_L{i}"] = jnp.linalg.norm(w_out)
        hidden = reshard(mixed.astype(hidden.dtype), _batch_spec())
        final_stats.update(nitp_logs)
        final_stats["attn_res_z"] = jax.lax.stop_gradient(z_total)
        query_norms = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.square(queries_flat.astype(jnp.float32)), axis=-1)))
        final_stats["attn_res_query_norm_mean"] = jnp.mean(query_norms)
        final_stats["attn_res_query_norm_max"] = jnp.max(query_norms)
        for g in range(queries.shape[0]):
            final_stats[f"attn_res_query_norm_g{g:02d}"] = query_norms[g]
        final_stats.update({})
        # Stats whose shape differs by layer (the routed input under ``latent_free_layers``, the top-K assignments under
        # ``topk_layers``) stay per-layer tuples.
        widths = {
            k: [stats[k] for stats in layer_stats]
            for k in layer_stats[0]
            if len({jnp.shape(stats[k]) for stats in layer_stats}) > 1
        }
        stacked = jax.tree.map(
            lambda *xs: jnp.stack(xs), *[{k: v for k, v in stats.items() if k not in widths} for stats in layer_stats]
        )
        return hidden, {**stacked, **{k: tuple(v) for k, v in widths.items()}}, final_stats

    def _split_scanned_units(self, scan: _LayerScan) -> tuple[dict[bool, Block], list[Block | None]]:
        """Split each mixer kind's stack into the scanned units (leaves ``[U, per-unit count, ...]``) and the
        remaining layers; returns the units and every layer in order, None where scanned."""
        units: dict[bool, Block] = {}
        layers: list[Block | None] = [None] * self.config.num_layers
        for stack, indices in zip(self.layer_stacks(), self.stack_layer_indices(), strict=True):
            is_kda = isinstance(stack.stacked.attn, KimiDeltaAttention)
            scanned = sum(i < scan.num_layers for i in indices)
            remaining = indices[scanned:]
            assert all(i >= scan.num_layers for i in remaining), indices
            # One split per stack (never slices), so each stack's gradient is one concatenate.
            parts = _split_leading(stack.stacked, ((scanned,) if scanned else ()) + (1,) * len(remaining))
            if scanned:
                count = scan.kind_count(is_kda)
                assert scanned == count * scan.num_units and is_kda not in units, (indices, scan)
                prefix, *parts = parts
                units[is_kda] = jax.tree.map(lambda x, c=count: x.reshape(scan.num_units, c, *x.shape[1:]), prefix)
            for i, part in zip(remaining, parts, strict=True):
                # The stacks stay fp32 under scan_layers (see _cast_to_compute); unrolled layers cast here.
                layers[i] = _cast_floats(jax.tree.map(lambda x: x[0], part), self.token_embed.dtype)
        return units, layers

    def _run_scanned_units(
        self,
        scan: _LayerScan,
        units: dict[bool, Block],
        queries: Float[Array, "G D"],
        hidden: Float[Array, "B S D"],
        extra_sources: tuple[jax.Array, ...],
        long_mask: AttentionMask,
        short_mask: AttentionMask,
        token_ids: Int[Array, "B S"],
        route_key: jax.Array | None,
    ) -> tuple[_AttnResHistory, tuple[dict[str, jax.Array], ...]]:
        """Run the scanned layer units from the embedding; returns the history and each unit layer's stats
        stacked over units."""
        eps = self.config.layer_norm_eps
        if any(src.dtype != hidden.dtype for src in extra_sources):
            raise ValueError("scan_layers stacks the AttnRes sources, so they must share the embedding's dtype")
        num_sources = scan.schedule.num_sources(scan.num_layers - 1)
        zeros = [jnp.zeros_like(hidden)] * (num_sources - len(extra_sources))
        zero_logits = jnp.zeros((queries.shape[0], *hidden.shape[:2]), jnp.float32)
        batch_spec = _batch_spec()
        history = _AttnResHistory(
            sources=jnp.stack([reshard(src, batch_spec) for src in (*extra_sources, *zeros)]),
            logits=jnp.stack(
                [
                    reshard(src_logits, P(None, *batch_spec))
                    for src_logits in (
                        *(_attn_res_source_logits(src, queries, eps) for src in extra_sources),
                        *[zero_logits] * len(zeros),
                    )
                ]
            ),
            partial=hidden,
        )
        units_diff, units_static = eqx.partition(units, eqx.is_inexact_array)
        args = _ScanArgs(
            scan,
            units_static,
            long_mask,
            short_mask,
            token_ids,
            eps,
            route_key,
            self.config.scan_offload_partials,
            self.config.scan_prefetch_partials,
            self.config.scan_offload_partials and not self.config.scan_device_sources,
            self.config.scan_release_weight_grads,
        )
        return _scan_layers((units_diff, queries, history), args)

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

    def neuron_inputs(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Int[Array, "L T K"], Float[Array, "L T I"], Float[Array, "L T J"]]:
        """Every MoE layer's selected experts, the routed experts' input (after the latent projection and norm)
        and the ungated shared experts' pre-activations, with ``T = B * S`` (a debug forward for the neuron-health
        dump)."""
        if self.config.dense_mlp:
            raise ValueError("neuron_inputs needs MoE layers")
        _, metrics = self(token_ids, mask=mask, return_routing=True)
        if SHARED_PRE_KEY not in metrics:
            raise ValueError("neuron_inputs needs ungated shared experts outside moe_shared_placement=overlap")
        return metrics[ROUTING_SELECTED_KEY], metrics[ROUTED_INPUT_KEY], metrics[SHARED_PRE_KEY]

    def routing_neuron_dump(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array,
        top_neurons: int,
        chunk: int,
    ) -> dict[str, jax.Array]:
        """Per MoE layer, each token's routing and the hidden units its routed experts fire (a debug forward for
        the routing dump; ungated ReLU^2 experts, ``h = relu(x @ w_up[e])^2``), with ``T = B * S``:

        ``selected`` / ``weights`` ``[L, T, K]``; ``top_ids`` / ``top_vals`` ``[L, T, K, top_neurons]``, each slot's
        most active hidden units; ``frac_active`` ``[L, T, K]``, the share of the slot's hidden units with a positive
        pre-activation; ``expert_mean_act`` / ``expert_active_rate`` ``[L, E, I]``, each expert's mean activation and
        firing rate per unit over the tokens routed to it; ``expert_tokens`` ``[L, E]``."""
        cfg = self.config
        if cfg.dense_mlp or cfg.num_null_experts or not cfg.moe_ungated_relu2:
            raise ValueError("routing_neuron_dump needs ungated ReLU^2 routed experts without null experts")
        _, metrics = self(token_ids, mask=mask, return_routing=True)

        def replicated(x):
            return reshard(x, P(*([None] * x.ndim)))

        selected = replicated(metrics[ROUTING_SELECTED_KEY])  # [L, T, K]
        weights = replicated(metrics[ROUTING_WEIGHTS_KEY]).astype(jnp.float32)
        routed = replicated(metrics[ROUTED_INPUT_KEY])  # [L, T, R]
        _, num_tokens, k = selected.shape
        e = cfg.num_experts
        out = {
            name: []
            for name in ("top_ids", "top_vals", "frac_active", "expert_mean_act", "expert_active_rate", "expert_tokens")
        }
        for layer, block in enumerate(self.layers()):
            w_up = replicated(block.mlp.expert_mlp.w_up)  # [E, R, I]
            inter = w_up.shape[-1]

            def body(carry, xs, w_up=w_up, inter=inter):
                act_sum, fire_sum = carry
                x, sel = xs  # [C, R], [C, K]
                pre = jnp.einsum("cr,ckri->cki", x, w_up[sel], preferred_element_type=jnp.float32)
                h = jnp.square(jax.nn.relu(pre))
                vals, ids = jax.lax.top_k(h, top_neurons)
                fire = (pre > 0).astype(jnp.float32)
                flat_sel = sel.reshape(-1)
                act_sum = act_sum + jax.ops.segment_sum(h.reshape(-1, inter), flat_sel, num_segments=e)
                fire_sum = fire_sum + jax.ops.segment_sum(fire.reshape(-1, inter), flat_sel, num_segments=e)
                return (act_sum, fire_sum), (ids.astype(jnp.int16), vals.astype(jnp.float16), jnp.mean(fire, axis=-1))

            xs = (
                routed[layer].reshape(num_tokens // chunk, chunk, -1),
                selected[layer].reshape(num_tokens // chunk, chunk, k),
            )
            zeros = jnp.zeros((e, inter), jnp.float32)
            (act_sum, fire_sum), (ids, vals, frac) = jax.lax.scan(body, (zeros, zeros), xs)
            counts = jnp.zeros((e,), jnp.float32).at[selected[layer].reshape(-1)].add(1.0)
            denom = jnp.maximum(counts, 1.0)[:, None]
            out["top_ids"].append(ids.reshape(num_tokens, k, top_neurons))
            out["top_vals"].append(vals.reshape(num_tokens, k, top_neurons))
            out["frac_active"].append(frac.reshape(num_tokens, k).astype(jnp.float16))
            out["expert_mean_act"].append(act_sum / denom)
            out["expert_active_rate"].append(fire_sum / denom)
            out["expert_tokens"].append(counts)
        dump = {name: jnp.stack(parts) for name, parts in out.items()}
        dump["selected"] = selected.astype(jnp.int16)
        dump["weights"] = weights.astype(jnp.float16)
        return dump

    def layer_kinds(self) -> tuple[str, ...]:
        """Per layer, its mixer and context, e.g. ``kda_local`` or ``mla_global``."""
        cfg = self.config
        return tuple(
            ("kda" if isinstance(layer.attn, KimiDeltaAttention) else "mla" if cfg.mla else "gqa")
            + ("_global" if _is_long_layer(i, cfg.num_layers, cfg.global_every, (None)) else "_local")
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
        logits = jnp.einsum("bsh,hd->bsd", hidden, lm_head, out_sharding=batch_spec)
        return logits

    def position_predictions(
        self,
        token_ids: Int[Array, "B S"],
        positions: Int[Array, "T 2"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        k: int,
    ) -> tuple[Float[Array, " T"], Int[Array, "T k"], Float[Array, "T k"], dict[str, Float[Array, "Q H K"]]]:
        """At each ``(row, position)``: the loss of the actual next token and the top-``k`` next-token ids and
        probabilities, through the training head (bigram prior, bias and soft-cap included). The dict holds the
        ``forward_probe`` recordings when traced inside one (else empty)."""
        hidden, metrics = self(token_ids, mask=mask)
        probe = {name: value for name, value in metrics.items() if name.startswith(PROBE_STAT)}
        head_in, lm_head = self._lm_head_operands(hidden, token_ids)
        spot = _probe_spot()
        if spot is not None:
            assert _FORWARD_PROBE is not None
            probe[f"{PROBE_STAT}head_in"] = _at_spot(head_in, spot)
            ids = jnp.asarray(_FORWARD_PROBE.spot_token_ids, jnp.int32)
            probe[f"{PROBE_STAT}lm_head_cols"] = jax.lax.stop_gradient(
                reshard(lm_head, P(None, None))[:, ids].astype(jnp.float32)
            )
        head_in = reshard(head_in, P(None, None, None))
        picked = head_in[positions[:, 0], positions[:, 1]].astype(jnp.float32)
        logits = jnp.einsum(
            "te,ev->tv", picked, reshard(lm_head, P(None, None)).astype(jnp.float32), out_sharding=P(None, None)
        )
        logp = jax.nn.log_softmax(_soft_capped(logits, _logit_cap(self.config)), axis=-1)
        targets = reshard(token_ids, P(None, None))[positions[:, 0], positions[:, 1] + 1]
        loss = -jnp.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
        top_logp, top_ids = jax.lax.top_k(logp, k)
        return loss, top_ids, jnp.exp(top_logp), probe

    def _final_conf_logit(self, n: int, stream: Float[Array, "B S D"]) -> Float[Array, "B S"]:
        assert self.final_conf_w1 is not None and self.final_conf_b1 is not None
        assert self.final_conf_w2 is not None and self.final_conf_b2 is not None
        x = rms_norm(jax.lax.stop_gradient(stream).astype(jnp.float32), self.config.layer_norm_eps)
        h = jax.nn.relu(jnp.einsum("bsd,dh->bsh", x, self.final_conf_w1[n]) + self.final_conf_b1[n])
        return jnp.einsum("bsh,h->bs", h, self.final_conf_w2[n]) + self.final_conf_b2[n]

    def _skip_router_keep(self, stream: Float[Array, "B S D"]) -> tuple[Float[Array, "B S"], jax.Array]:
        """Per-token output scale for ``skip_router_layers`` (0: skip) and the skipped fraction."""
        cfg = self.config
        assert self.skip_router is not None and self.skip_router_bias is not None
        score = jnp.einsum("bsd,d->bs", rms_norm(stream.astype(jnp.float32), cfg.layer_norm_eps), self.skip_router)
        score = score + self.skip_router_bias
        chosen = jax.lax.stop_gradient(score)
        threshold = jnp.quantile(reshard(chosen, P(None, None)), cfg.skip_router_frac)
        keep = chosen > threshold
        return keep * 2.0 * jax.nn.sigmoid(score), 1.0 - jnp.mean(keep.astype(jnp.float32))

    def _extra_mlp_selection(
        self, stream: Float[Array, "B S D"], token_ids: Int[Array, "B S"], active: jax.Array | bool
    ) -> tuple[Float[Array, "B S"], Bool[Array, "B S"]]:
        """The entropy head's prediction (from the stop-gradient stream) and the ``extra_mlp_layers`` token mask: the
        ``extra_mlp_frac`` of the batch with the highest / lowest prediction, or a fixed pseudo-random subset."""
        cfg = self.config
        assert self.entropy_head is not None and self.entropy_head_bias is not None
        x = rms_norm(jax.lax.stop_gradient(stream).astype(jnp.float32), cfg.layer_norm_eps)
        pred = jnp.einsum("bsd,d->bs", x, self.entropy_head) + self.entropy_head_bias
        frac, chosen = cfg.extra_mlp_frac, jax.lax.stop_gradient(pred)
        # The batch quantile sorts the whole [B, S] prediction (a few MB), gathered off the batch sharding.
        everywhere = reshard(chosen, P(None, None))
        if cfg.extra_mlp_select == ExtraMlpSelect.HIGH:
            keep = chosen >= jnp.quantile(everywhere, 1.0 - frac)
        elif cfg.extra_mlp_select == ExtraMlpSelect.LOW:
            keep = chosen <= jnp.quantile(everywhere, frac)
        else:
            pos = jnp.broadcast_to(jnp.arange(token_ids.shape[1], dtype=jnp.uint32), token_ids.shape)
            h = pos * jnp.uint32(2654435761) + token_ids.astype(jnp.uint32) * jnp.uint32(2246822519)
            h = (h ^ (h >> 15)) * jnp.uint32(2246822519)
            keep = ((h ^ (h >> 13)).astype(jnp.float32) / jnp.float32(1 << 32)) < frac
        return pred, keep & jnp.asarray(active)

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
        return hidden, self.output_proj

    def _mid_head_terms(
        self, mid_hidden: Float[Array, "B S D"], labels: Int[Array, "B S"], loss_weight: jax.Array, loss_dtype
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """The mid head's mean next-token loss, the confidence head's mean BCE, and per-position ``loss``,
        ``correct`` (the mid head's top token is the label), ``confidence`` (the predicted probability of that)
        and ``entropy``."""
        cfg = self.config
        assert self.mid_output_proj is not None and self.mid_conf_w1 is not None
        assert self.mid_conf_b1 is not None and self.mid_conf_w2 is not None and self.mid_conf_b2 is not None
        cap = _logit_cap(cfg)
        assert cap is None or isinstance(cap, float)
        mid_in = reshard(rms_norm(mid_hidden), _batch_spec())
        head_in = mid_in if cfg.mid_head_trunk_grad else jax.lax.stop_gradient(mid_in)
        per_token = fused_linear_softmax_cross_entropy_loss(
            head_in,
            self.mid_output_proj,
            labels,
            weight=loss_weight,
            reduction="none",
            dtype=loss_dtype,
            implementation="xla_fast_bwd",
            block_sizes=_ce_block_sizes(),
            logit_soft_cap=cap,
        )
        weight = loss_weight.astype(jnp.float32)
        count = jnp.maximum(jnp.sum(weight), 1.0)
        mid_loss = jnp.sum(per_token.astype(jnp.float32)) / count
        stats, logit = self._mid_confidence(mid_hidden, labels)
        correct = (stats.top_id == labels).astype(jnp.float32)
        bce = jnp.sum(weight * (jax.nn.softplus(logit) - correct * logit)) / count
        out = {
            "loss": stats.logsumexp - stats.label_logit,
            "correct": correct,
            "confidence": jax.nn.sigmoid(logit),
            "entropy": stats.entropy,
        }
        return mid_loss, bce, out

    def _mid_confidence(
        self, mid_hidden: Float[Array, "B S D"], labels: Int[Array, "B S"] | None = None
    ) -> tuple[VocabStats, Float[Array, "B S"]]:
        """The mid head's stop-gradient ``VocabStats`` (``label_logit`` is meaningless without ``labels``) and the
        confidence head's logit that its top token is right."""
        cap = _logit_cap(self.config)
        assert cap is None or isinstance(cap, float)
        assert self.mid_output_proj is not None and self.mid_conf_w1 is not None
        assert self.mid_conf_b1 is not None and self.mid_conf_w2 is not None and self.mid_conf_b2 is not None
        mid_in = reshard(rms_norm(mid_hidden), _batch_spec())
        if labels is None:
            labels = jnp.zeros(mid_hidden.shape[:2], jnp.int32)
        stats = _vocab_stats(mid_in, self.mid_output_proj, cap, labels)
        top_logp = stats.top_logit - stats.logsumexp
        answer = rms_norm(_embedding_gather(jax.lax.stop_gradient(self.token_embed), stats.top_id).astype(jnp.float32))
        features = jnp.concatenate(
            [
                rms_norm(jax.lax.stop_gradient(mid_hidden).astype(jnp.float32)),
                answer,
                top_logp[..., None],
                stats.entropy[..., None],
            ],
            axis=-1,
        )
        h = jax.nn.relu(jnp.einsum("bsf,fh->bsh", features, self.mid_conf_w1) + self.mid_conf_b1)
        return stats, jnp.einsum("bsh,h->bs", h, self.mid_conf_w2) + self.mid_conf_b2

    def skip_probe_diagnostics(
        self, token_ids: Int[Array, "B S"], mask: AttentionMask, skip_layers: tuple[int, ...]
    ) -> dict[str, Float[Array, "B S"]]:
        """Per position: the probes' ``probe_kept`` (aware, not skipping), ``probe_skipped`` (aware, skipping) and
        ``probe_blind`` predictions (probabilities, or losses for ``SkipProbeTarget.LOSS``), and the final head's
        ``correct_*``, ``top_prob_*`` and ``entropy_*`` with every token keeping (``kept``) / skipping (``skipped``)
        ``skip_layers``."""
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        out = {}
        for name, keep in (("kept", True), ("skipped", False)):
            skip = LayerSkip(skip_layers, jnp.full(token_ids.shape, keep))
            hidden, stats = self(token_ids, mask=mask, layer_skip=skip)
            head_in, lm_head = self._lm_head_operands(hidden, token_ids)
            final = _vocab_stats(head_in, lm_head, _logit_cap(self.config), labels)
            out[f"correct_{name}"] = (final.top_id == labels).astype(jnp.float32)
            # The label-free confidence of each path, what an exit rule at the midpoint can read.
            out[f"top_prob_{name}"] = jnp.exp(final.top_logit - final.logsumexp)
            out[f"entropy_{name}"] = final.entropy
            if keep:
                logits = stats[_SKIP_PROBE_LOGITS]
                if self.config.skip_probe_target == SkipProbeTarget.CORRECT:
                    logits = jax.nn.sigmoid(logits)
                out.update(probe_blind=logits[1], probe_kept=logits[2], probe_skipped=logits[3])
        return out

    def final_conf_diagnostics(
        self, token_ids: Int[Array, "B S"], mask: AttentionMask
    ) -> dict[str, Float[Array, "B S"]]:
        """Per-position ``final_correct`` and each ``final_conf_layers`` probe's ``final_conf_L{layer}``."""
        hidden, stats = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        head_in, lm_head = self._lm_head_operands(hidden, token_ids)
        final = _vocab_stats(head_in, lm_head, _logit_cap(self.config), labels)
        out = {"final_correct": (final.top_id == labels).astype(jnp.float32), "final_entropy": final.entropy}
        for n, layer in enumerate(self.config.final_conf_layers):
            out[f"final_conf_L{layer}"] = jax.nn.sigmoid(stats[_FINAL_CONF_LOGITS][n])
        return out

    def mid_head_diagnostics(self, token_ids: Int[Array, "B S"], mask: AttentionMask) -> dict[str, Float[Array, "B S"]]:
        """Per-position mid-head outputs (``_mid_head_terms``) and the final head's ``final_correct``."""
        hidden, stats = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        weight = jnp.ones(token_ids.shape, jnp.float32)
        _, _, mid = self._mid_head_terms(stats[_MID_HIDDEN].astype(hidden.dtype), labels, weight, jnp.float32)
        head_in, lm_head = self._lm_head_operands(hidden, token_ids)
        final = _vocab_stats(head_in, lm_head, _logit_cap(self.config), labels)
        return {
            **{f"mid_{k}": v for k, v in mid.items()},
            "final_correct": (final.top_id == labels).astype(jnp.float32),
            "final_entropy": final.entropy,
        }

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
        loop_active: bool | None = None,
        train_terms: bool = False,
        route_key: jax.Array | None = None,
        head_replay: "HeadReplay | None" = None,
        byte_table: Int[Array, "V N"] | None = None,
        byte_aux_weight: jax.Array | None = None,
        router_tie_active: bool | None = None,
        future_aux_weight: jax.Array | None = None,
        layer_skip: "LayerSkip | None" = None,
        extra_mlp_active: jax.Array | bool = True,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        """``train_terms`` adds the training-only objectives (MTP, AttnRes z-loss, the mid head); evals leave it off
        so they score the plain next-token loss."""
        hidden, router_metrics = self(
            token_ids,
            mask=mask,
            loop_active=loop_active,
            route_key=route_key,
            router_tie_active=router_tie_active,
            layer_skip=layer_skip,
            extra_mlp_active=extra_mlp_active,
        )
        mid_hidden = router_metrics.pop(_MID_HIDDEN, None)
        final_conf_logits = router_metrics.pop(_FINAL_CONF_LOGITS, None)
        skip_probe_logits = router_metrics.pop(_SKIP_PROBE_LOGITS, None)
        entropy_pred = router_metrics.pop(_ENTROPY_PRED, None)
        router_metrics.pop(_NITP_TARGET, None)
        attn_res_z = router_metrics.pop(_ATTN_RES_Z, None)
        router_metrics.pop(_MTP_EMBED, None)
        compress_loss = router_metrics.pop(_COMPRESS_LOSS, None)
        compress_stats = {
            k: router_metrics.pop(k) for k in list(router_metrics) if k.startswith((_COMPRESS_PREFIX, _OUTSTAT_PREFIX))
        }
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        def lm_loss(h: jax.Array) -> jax.Array:
            head_in, lm_head = self._lm_head_operands(h, token_ids)
            ce = functools.partial(
                fused_linear_softmax_cross_entropy_loss,
                head_in,
                lm_head,
                labels,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
                implementation="xla_fast_bwd",
                block_sizes=_ce_block_sizes(),
                logit_soft_cap=_logit_cap(self.config),
            )
            return ce(weight=loss_weight, reduction=reduction)

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
                block_sizes=_ce_block_sizes(),
                logit_soft_cap=_logit_cap(self.config),
            )
        # Router z-loss is logged for monitoring only; it is not added to the training loss.
        loss = cross_entropy_loss
        mid_metrics: dict[str, jax.Array] = {}
        if skip_probe_logits is not None and train_terms:
            head_in, lm_head = self._lm_head_operands(hidden, token_ids)
            final = _vocab_stats(head_in, lm_head, _logit_cap(self.config), labels)
            correct = (final.top_id == labels).astype(jnp.float32)
            weight = loss_weight.astype(jnp.float32)
            count = jnp.maximum(jnp.sum(weight), 1.0)
            # The aware probe (f0 + d f1) and the blind probe (g) train; the counterfactual rows only log.
            trained = skip_probe_logits[:2]
            if self.config.skip_probe_target == SkipProbeTarget.LOSS:
                target = jax.lax.stop_gradient(final.logsumexp - final.label_logit)
                bce = jnp.sum(weight * jnp.square(trained - target), axis=(1, 2)) / count
                effect = skip_probe_logits[3] - skip_probe_logits[2]
            else:
                bce = jnp.sum(weight * (jax.nn.softplus(trained) - correct * trained), axis=(1, 2)) / count
                effect = jax.nn.sigmoid(skip_probe_logits[3]) - jax.nn.sigmoid(skip_probe_logits[2])
            loss = loss + jnp.sum(bce).astype(loss_dtype)
            mid_metrics.update(
                {
                    "train/aux/skip_probe_err_aware": jax.lax.stop_gradient(bce[0]),
                    "train/aux/skip_probe_err_blind": jax.lax.stop_gradient(bce[1]),
                    "train/aux/skip_probe_effect_mean": jax.lax.stop_gradient(jnp.sum(weight * effect) / count),
                    "train/aux/final_top1_acc": jnp.sum(weight * correct) / count,
                }
            )
        if final_conf_logits is not None and train_terms:
            head_in, lm_head = self._lm_head_operands(hidden, token_ids)
            final = _vocab_stats(head_in, lm_head, _logit_cap(self.config), labels)
            correct = (final.top_id == labels).astype(jnp.float32)
            weight = loss_weight.astype(jnp.float32)
            count = jnp.maximum(jnp.sum(weight), 1.0)
            bce = jnp.sum(weight * (jax.nn.softplus(final_conf_logits) - correct * final_conf_logits), axis=(1, 2))
            bce = bce / count
            loss = loss + jnp.sum(bce).astype(loss_dtype)
            mid_metrics["train/aux/final_top1_acc"] = jnp.sum(weight * correct) / count
            for n, layer in enumerate(self.config.final_conf_layers):
                mid_metrics[f"train/aux/final_conf_bce_L{layer}"] = jax.lax.stop_gradient(bce[n])
        if mid_hidden is not None and train_terms:
            mid_loss, bce, mid = self._mid_head_terms(mid_hidden.astype(hidden.dtype), labels, loss_weight, loss_dtype)
            loss = loss + (self.config.mid_head_loss_weight * mid_loss + bce).astype(loss_dtype)
            weight = loss_weight.astype(jnp.float32)
            count = jnp.maximum(jnp.sum(weight), 1.0)
            mid_metrics = {
                "train/aux/mid_lm_loss": jax.lax.stop_gradient(mid_loss),
                "train/aux/mid_conf_bce": jax.lax.stop_gradient(bce),
                "train/aux/mid_top1_acc": jnp.sum(weight * mid["correct"]) / count,
                "train/aux/mid_conf_mean": jnp.sum(weight * mid["confidence"]) / count,
                "train/aux/mid_conf_brier": jnp.sum(weight * jnp.square(mid["confidence"] - mid["correct"])) / count,
            }
        if compress_loss is not None and train_terms:
            loss = loss + jnp.sum(compress_loss).astype(loss_dtype)
        entropy_metrics: dict[str, jax.Array] = {}
        if entropy_pred is not None and train_terms:
            head_in, lm_head = self._lm_head_operands(hidden, token_ids)
            cap = _logit_cap(self.config)
            assert cap is None or isinstance(cap, float), "the entropy head needs a scalar logit_soft_cap"
            target = _predictive_entropy(head_in, lm_head, cap)
            weight = loss_weight.astype(jnp.float32)
            count = jnp.maximum(jnp.sum(weight), 1.0)
            err = entropy_pred - target
            mse = jnp.sum(weight * jnp.square(err)) / count
            loss = loss + mse.astype(loss_dtype)
            centered = [x - jnp.sum(weight * x) / count for x in (entropy_pred, target)]
            corr = jnp.sum(weight * centered[0] * centered[1]) / jnp.sqrt(
                jnp.sum(weight * jnp.square(centered[0])) * jnp.sum(weight * jnp.square(centered[1])) + 1e-12
            )
            entropy_metrics = {
                "train/aux/entropy_head_mse": jax.lax.stop_gradient(mse),
                "train/aux/entropy_head_corr": jax.lax.stop_gradient(corr),
                "train/aux/entropy_target_mean": jnp.sum(weight * target) / count,
            }
        if attn_res_z is not None and train_terms:
            loss = loss + (0.0) * attn_res_z.astype(loss_dtype)
        nitp_loss = nitp_cos = None
        if replay_loss is not None and head_replay is not None:
            loss = loss + head_replay.scale.astype(loss_dtype) * replay_loss
        erc_loss, erc_ratios = None, {}
        if return_router_metrics:
            final_gate_metrics = {
                f"train/attn_res/{name.removeprefix('attn_res_')}": router_metrics.pop(name)
                for name in list(router_metrics)
                if name.startswith("attn_res_")
            }
            if not router_metrics:
                # Dense model: no router to summarize.
                return loss, {
                    "train/cross_entropy_loss": cross_entropy_loss,
                    **final_gate_metrics,
                    **entropy_metrics,
                    **mid_metrics,
                }
            summarized_metrics = summarize_router_metrics(router_metrics)
            summarized_metrics.update(final_gate_metrics)
            summarized_metrics.update(entropy_metrics)
            summarized_metrics.update(mid_metrics)
            if NEWTON_GRAM_KEY in router_metrics:
                summarized_metrics[NEWTON_GRAM_KEY] = router_metrics[NEWTON_GRAM_KEY]
            if self.config.num_null_experts:
                summarized_metrics.update(_null_slot_metrics(self.config, router_metrics["routing_counts_per_layer"]))
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            if replay_loss is not None:
                summarized_metrics["train/head_replay_loss"] = replay_loss
            if head_replay is not None:
                summarized_metrics[FINAL_HIDDEN_KEY] = jax.lax.stop_gradient(hidden)
            if erc_loss is not None:
                summarized_metrics["train/aux/erc_loss"] = erc_loss
                summarized_metrics.update(erc_ratios)
            if compress_loss is not None:
                summarized_metrics["train/aux/compress/loss"] = jnp.sum(compress_loss)
            for name, per_layer in compress_stats.items():
                for i in range(per_layer.shape[0]):
                    summarized_metrics[f"train/aux/{name}_L{i}"] = per_layer[i]
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


def _null_slot_metrics(cfg: GrugModelConfig, routing_counts: Float[Array, "L N"]) -> dict[str, jax.Array]:
    """Fraction of the top-K slots that went to the zero-computation experts: per layer, and per type."""
    counts = routing_counts.astype(jnp.float32)
    total = jnp.maximum(jnp.sum(counts, axis=-1), 1.0)
    bounds = list(itertools.accumulate((cfg.num_experts, (0), (0), (0))))
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


def _rest_layers(cfg: GrugModelConfig) -> tuple[int, ...]:
    """Under ``scan_layers``, the non-tail layers past the scanned units. They get their own ``*_blocks_rest``
    stacks so the main stacks hold exactly the scanned layers: the scan's gradients then reshape into the
    parameters in place instead of being copied back into a longer stack (~30 GiB per GPU at hero width)."""
    if not cfg.scan_layers:
        return ()
    unit, num_units = _scan_units(cfg)
    tail = set(_tail_layers(cfg))
    return tuple(i for i in range(len(unit) * num_units, cfg.num_layers) if i not in tail)


def _rest_stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)`` of the ``*_blocks_rest`` stacks (``_rest_layers``)."""
    kda_layers = set(_kda_layer_indices(cfg))
    rest = _rest_layers(cfg)
    return tuple(i for i in rest if i not in kda_layers), tuple(i for i in rest if i in kda_layers)


def _stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)``: the layer indices of ``Transformer.stacked_blocks`` and ``kda_blocks``
    (``latent_out_full_layers`` and ``_rest_layers`` excluded: they live in the tail and rest stacks)."""
    kda_layers = _kda_layer_indices(cfg)
    tail = set(_tail_layers(cfg)) | set(_rest_layers(cfg))
    softmax = tuple(i for i in range(cfg.num_layers) if i not in kda_layers and i not in tail)
    return softmax, tuple(i for i in kda_layers if i not in tail)


def _tail_layers(cfg: GrugModelConfig) -> tuple[int, ...]:
    """The layers held by the tail stacks (``latent_out_full_layers`` or ``latent_free_layers``)."""
    return cfg.latent_out_full_layers or () or () or ()


def _final_layer_config(cfg: GrugModelConfig, tail_cfg: GrugModelConfig) -> GrugModelConfig:
    """The softmax tail stack's config when it holds only the final layer: ``lm_head_extra_dim`` and
    ``final_shared_only`` apply to it."""
    out = tail_cfg
    return out


def _tail_layer_config(cfg: GrugModelConfig) -> GrugModelConfig:
    """The tail stacks' layer config: a full-width expert write, or no latent at half the expert width."""
    return dataclasses.replace(cfg, latent_out_dim=cfg.hidden_dim, latent_out_full_layers=())


def _tail_stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)`` of ``latent_out_full_layers``: the ``*_blocks_tail`` stacks' layers."""
    kda_layers = set(_kda_layer_indices(cfg))
    tail = sorted(_tail_layers(cfg))
    return tuple(i for i in tail if i not in kda_layers), tuple(i for i in tail if i in kda_layers)


def upper_softmax_slice_mask(cfg: GrugModelConfig) -> tuple[bool, ...]:
    """Per ``Transformer.stacked_blocks`` slice, whether its layer is in the upper half (``>= num_layers // 2``)."""
    softmax_layers, _ = _stack_layer_indices(cfg)
    return tuple(i >= cfg.num_layers // 2 for i in softmax_layers)


def _gate_extras(model: "Transformer", num_gates: int) -> dict[str, jax.Array | None] | None:
    """The per-gate logit extras of ``_bias_gate_logits``, or None when the model uses none of them."""
    extras = {
        "bias": None,
        "pull_keys": None,
        "embed_query": None,
        "mask": None,
        "dual": None,
        "dual_sel": None,
        "dual_sel_bias": None,
        "blend": None,
        "blend_proj": None,
        "temperature": None,
        "dyn_w1": None,
        "dyn_w2": None,
        "token_query": None,
    }
    return extras if any(v is not None for v in extras.values()) else None


def _content_gate(
    hidden: Float[Array, "B S D"],
    source: Float[Array, "B S D"],
    w: jax.Array,
    bias: jax.Array,
    a_lr: jax.Array | None,
    b_lr: jax.Array | None,
    *,
    act: ContentGateActivation = ContentGateActivation.NONE,
    logit_scale: float = 1.0,
) -> tuple[Float[Array, "B S D"], jax.Array]:
    """Engram-style content gate on an n-gram source: ``sigmoid(f(hidden * source) + bias)``, with ``f`` a rank-r
    projection to per-channel logits (``a_lr``, ``b_lr``) or a ``w`` dot to one logit per token. Both inputs are
    RMS-normed already, so it stays in the compute dtype."""
    dtype = source.dtype
    interaction = hidden.astype(dtype) * source
    if a_lr is not None and b_lr is not None:
        low = jnp.einsum("bsd,dr->bsr", interaction, a_lr.astype(dtype))
        if act == ContentGateActivation.SILU:
            low = jax.nn.silu(low)
        elif act == ContentGateActivation.GELU:
            low = jax.nn.gelu(low)
        logits = jnp.einsum("bsr,rd->bsd", low, b_lr.astype(dtype), out_sharding=_batch_spec())
        gate = jax.nn.sigmoid(logit_scale * logits + bias.astype(dtype))
        return source * gate, gate
    if act != ContentGateActivation.NONE:
        raise ValueError("bigram_gate_act needs the low-rank gate (bigram_gate_rank > 0)")
    logit = jnp.einsum("bsd,d->bs", interaction, w.astype(dtype))
    gate = jax.nn.sigmoid(logit_scale * logit + bias.astype(dtype))
    return source * gate[..., None], gate


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _latent_proj_init(cfg: "GrugModelConfig", key: PRNGKeyArray, shape: tuple[int, int]) -> Float[Array, "..."]:
    """A LatentMoE projection: truncated normal, or with ``latent_orthogonal_init`` semi-orthogonal (orthonormal
    columns, or rows when wide) with every singular value ``std * sqrt(max(shape))``, which matches the normal
    init's Frobenius norm."""
    std = cfg.initializer_std
    return _init_weight(key, shape, std)


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


# Lookup tables: a token reads one row, so they cost no matmul FLOPs.
_EMBEDDING_TABLES = frozenset({"token_embed", "token_embed2", "value_embed"})
# Per-head or per-expert vectors stored as 2-D arrays that are added, not multiplied.
_NON_MATMUL_PARAMS = frozenset({"dt_bias", "router_bias", "a_log"})


def forward_flops_per_token(model: "Transformer") -> float:
    """Analytic forward FLOPs per token of ``model``'s architecture.

    Matmuls are read off the parameter tree: 2 x every weight a token multiplies, with the routed experts
    scaled by ``num_experts_per_token / num_experts`` (nominal routing, drops not discounted). Embedding
    tables, norms and other vectors are excluded. The attention cores have no weights and are added
    explicitly: causal softmax attention over the keys each query actually attends (a sliding window caps
    the span), ``QK^T`` plus ``PV``; and KDA's chunked delta rule, ``6 d_k d_v + 4 C (d_k + d_v)`` per head
    (state read, state update, and the intra-chunk products at chunk size ``C``)."""
    cfg = model.config
    matmul = 0.0
    q_widths: dict[str, int] = {}
    kda_heads: list[tuple[int, int, int]] = []
    # Shapes only, so abstract models (``eqx.filter_eval_shape``) work too.
    leaves = jax.tree_util.tree_flatten_with_path(
        eqx.filter(
            model, lambda x: isinstance(x, jax.Array | jax.ShapeDtypeStruct) and jnp.issubdtype(x.dtype, jnp.inexact)
        )
    )[0]
    for path, leaf in leaves:
        names = [k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)]
        stacked = "stacked" in names
        per_layer_ndim = leaf.ndim - (1 if stacked else 0)
        if names[-1] in _EMBEDDING_TABLES or names[-1] in _NON_MATMUL_PARAMS or per_layer_ndim < 2:
            continue
        size = float(math.prod(leaf.shape))
        if "expert_mlp" in names:
            size *= cfg.num_experts_per_token / cfg.num_experts
        matmul += 2 * size
        if names[-2:] == ["attn", "w_q"]:
            q_widths[names[0]] = leaf.shape[-1]
    for name in ("kda_blocks", "kda_blocks_rest", "kda_blocks_tail"):
        stack = getattr(model, name)
        if stack is None:
            continue
        attn = stack.stacked.attn
        heads = attn.a_log.shape[-1]
        kda_heads.append((stack.num_layers, heads, attn.w_q.shape[-1] // heads, attn.w_v.shape[-1] // heads))

    seq, window = cfg.max_seq_len, cfg.sliding_window
    kda_layers = set(_kda_layer_indices(cfg))
    softmax_widths = {w for name, w in q_widths.items() if not name.startswith("kda")}
    if len(softmax_widths) > 1:
        raise ValueError(f"softmax attention stacks disagree on the query width: {q_widths}")
    if not softmax_widths and len(kda_layers) < cfg.num_layers:
        raise ValueError("softmax attention layers have no w_q to size their attention core")
    core = 0.0
    if softmax_widths:
        (q_width,) = softmax_widths
        for i in range(cfg.num_layers):
            if i in kda_layers:
                continue
            long = _is_long_layer(i, cfg.num_layers, cfg.global_every, None)
            span = seq if long or window is None else min(window, seq)
            # Mean keys per query under a causal mask capped at ``span``: sum_p min(p + 1, span) / seq.
            keys = (span * (span + 1) / 2 + (seq - span) * span) / seq
            core += 2 * keys * q_width * 2  # QK^T over d_qk and PV over d_v, both q_width = heads * head_dim
    chunk = cfg.kda_chunk_size
    for num_layers, heads, d_k, d_v in kda_heads:
        core += num_layers * heads * (6 * d_k * d_v + 4 * chunk * (d_k + d_v))
    return matmul + core


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
    "forward_flops_per_token",
    "moe_and_shared_fused",
]
