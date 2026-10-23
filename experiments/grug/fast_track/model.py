# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel MoE grug variant model.

The default config is the fast_track baseline: sliding-window local layers and full-causal global
layers, all GQA softmax attention with half-RoPE, scanned as one layer stack. The KMA variant
(``local_mixer=KDA`` + ``mla`` + ``inkling_relpos`` + ``attn_res``) swaps the local layers for Kimi
Delta Attention, the global layers for MLA with an Inkling relative-position bias, and the residual
stream for Block Attention Residuals, unrolled layer by layer.
"""

import dataclasses
import functools
import itertools
import math
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
from levanter.grug.grug_moe import MoeActivation, MoEExpertMlp, moe_mlp
from levanter.grug.loss import BlockSizes, fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import unshard
from levanter.kernels.pallas.short_conv import short_conv
from levanter.tracker.histogram import SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.fast_track.key_offset import partial_key_offset
from experiments.grug.fast_track.qb_balancing import qb_beta_hist
from experiments.grug.fast_track.router_metrics import (
    local_routing_stats,
    reduce_router_stats,
    summarize_router_metrics,
)
from experiments.grug.moe.kda import chunk_kda, kda_fused
from experiments.grug.moe.kda_prep_pallas import GateFn, GateKind, gate_log_decay
from experiments.grug.moe.kda_state_pallas import StateGrads

_GATED_NORM_RANK = 128
_QB_HIST_BINS = 10_000
_CE_TOKENS_PER_RANK = 65_536
# A vocab tile of 8192 in the fused lm_head + cross-entropy loop measured ~3% faster than 4096 at d512 on H100.
_CE_BLOCK_SIZES = BlockSizes(b_block_size=_CE_TOKENS_PER_RANK, v_block_size=8192)

# Axes the non-expert params FSDP-shard over.
_FSDP_AXES: tuple[str, ...] = ("data", "expert")
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")
_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")

# Metrics-dict keys of the per-layer AttnRes stats: the z-loss term and the per-gate token-mean source weights
# (variable length per gate), popped into logging scalars.
_ATTN_RES_Z = "attn_res_z_term"
_ATTN_RES_W_ATTN = "attn_res_weights_attn"
_ATTN_RES_W_MLP = "attn_res_weights_mlp"
# Per-layer extras key carrying the token mask of a skipped layer (``LayerSkip``, entropy skip, skip router).
_SKIP_KEEP = "layer_skip_keep"
# The entropy head's prediction (``entropy_skip_layers``), from the forward to the loss.
_ENTROPY_PRED = "entropy_pred"
_ENTROPY_CHUNK = 1024
# Per-layer branch-output stats, stacked over layers like the router stats.
_OUTSTAT_PREFIX = "outstat/"

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
# Softmax-attention layers whose attention output gate is below this count as off (``head_gate_off``).
_HEAD_GATE_OFF = 0.1
# Under a pre-norm probe: each token's RMS of the residual entering final_norm, [B, S] float32 (stop-gradient).
FINAL_NORM_INPUT_RMS_KEY = "_final_norm_input_rms"
# Branch output SConv sites (``sconv_sites``).
SCONV_SITES = ("k", "attn", "mlp")


class GradProbeSite(StrEnum):
    """Where ``lm_head_grad_probe`` reads per-token gradient norms."""

    POST_NORM = "post_norm"  # the lm_head input, after final_norm (and final_gated_norm)
    PRE_NORM = "pre_norm"  # the residual stream entering final_norm
    BOTH = "both"


class ExtraMlpSelect(StrEnum):
    """Which tokens the entropy head picks for ``entropy_skip_layers`` (``extra_mlp_frac`` of the batch)."""

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


def _ve_lambda_init(mode: ValueEmbeds) -> jax.Array | None:
    if mode == ValueEmbeds.NONE:
        return None
    return jnp.array([1.0, 1.0 if mode == ValueEmbeds.GATED_LAMBDA else 0.0])


def _value_embed_weight(mode: ValueEmbeds, x: jax.Array, gate: jax.Array | None, lam: jax.Array) -> jax.Array:
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
    sconv_sites: tuple[str, ...] = SCONV_SITES
    """Where the SConvs sit (a subset of ``SCONV_SITES``): after the softmax layers' K projection, and on the
    attention and MLP branch outputs (KDA layers keep their own q/k/v convs and get no attention-output SConv)."""
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
    kda_state_grads: StateGrads = StateGrads.XLA
    """Where the fused KDA state pass's backward forms its per-chunk cotangents (``StateGrads``; same math)."""
    kda_state_block_v: int = 32
    """d_v tile of the fused KDA state pass (same math)."""
    # Block Attention Residuals (Kimi, arXiv 2603.15031): each sublayer's input is a learned per-sublayer
    # softmax over the RMS-normalized residual-block history (embedding, block sums, running partial).
    attn_res: bool = False
    attn_res_num_blocks: int = 8
    attn_res_block_starts: tuple[int, ...] = ()
    """Layers at which Block AttnRes closes the running partial into a new history source (layer 0 rolls in
    the embedding). Empty: every ``num_layers // attn_res_num_blocks``-th layer, at most ``attn_res_num_blocks``
    sources. Otherwise exactly ``attn_res_num_blocks`` strictly increasing layers starting at 0, e.g.
    (0, 2, 4, 6, 8, 10, 12, 13) at 14 layers: two-layer blocks with the last two layers kept apart."""
    attn_res_plain_sum: bool = False
    """Every AttnRes gate (and the final gate) returns the plain sum of its sources instead of a softmax mix: the
    standard residual stream (embedding + every earlier sublayer output), keeping the AttnRes loop's features
    (``attn_res_kv_layers`` then reads the residual stream after the leading layers)."""
    attn_res_kv_layers: tuple[int, ...] = ()
    """Layers whose attention K/V projections read their own AttnRes mix: a separate zero-init pseudo-query per
    layer that mixes only the sources the first ``attn_res_kv_source_layers`` layers produced (the blocks
    completed by that layer), normed like the attention input. Q and everything else keep the attention gate's
    mix."""
    attn_res_kv_source_layers: int | None = None
    """How many leading layers' outputs ``attn_res_kv_layers`` may read (None: ``num_layers // 2``)."""
    entropy_skip_layers: tuple[int, ...] = ()
    """Layers the entropy head's chosen tokens (``extra_mlp_select``, ``extra_mlp_frac``) skip, as in ``LayerSkip``,
    from when ``extra_mlp_active`` turns on. The entropy head is a linear read of the stop-gradient stream entering
    the first of these layers, trained to predict the model's own next-token entropy."""
    extra_mlp_select: ExtraMlpSelect = ExtraMlpSelect.HIGH
    extra_mlp_frac: float = 0.25
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
    sublayer_gated_norm: bool = True
    """GatedNorm after the RMSNorm on every attention and MLP input (else the RMSNorm alone)."""
    attn_res_layer_backward: AttnResLayerBackward = AttnResLayerBackward.RECOMPUTE
    attn_res_remat_attention: bool = False
    """Rematerialize the attention branch inside each AttnRes layer's backward, so its residuals (incl.
    the ``[B, H, S, W]`` Inkling bias) are never alive during the MLP backward. Costs one extra
    attention forward per layer; needed for memory at d1280."""
    value_embeds: ValueEmbeds = ValueEmbeds.NONE
    router_combine: RouterCombine = RouterCombine.SIGMOID_RENORM
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
    lm_head_grad_probe: bool = False
    """Log each token's gradient norm w.r.t. the lm_head input (the final normed hidden state) every step
    (``lm_head_grad_probe``; train-step stats under ``train/lm_head_grad/``). Without ``lm_head_grad_clip`` the
    training math is unchanged."""
    lm_head_grad_probe_site: GradProbeSite = GradProbeSite.POST_NORM
    """Where ``lm_head_grad_probe`` reads the norms (``GradProbeSite``); the pre-norm site logs under
    ``train/lm_head_grad_pre/`` with each token's input RMS under ``train/final_norm_input_rms/``."""
    lm_head_grad_dump_every: int = 0
    """With ``lm_head_grad_probe``: also write the whole batch's per-token norms and token ids every this many steps
    (and densely early on, see ``train.lm_head_grad_dump_due``); 0: no dumps."""
    token_embed_stats: bool = False
    """Log ``token_embed``'s Frobenius norm, row-norm min / p1 / p50 / max and relative update every train step
    (``train/embed/``). Logging only."""
    lm_head_grad_clip: float | None = None
    """With ``lm_head_grad_probe``: clip each training token's gradient w.r.t. the lm_head input to this norm, in
    the probe's per-token units (the gradient of the token's own loss term), ``g * min(1, c / ||g||)``. The probe
    still reports the pre-clip norms. None: no clipping."""
    init_std_mult_gates: float = 1.0
    """Init-std multiplier of the MuonH-trained sigmoid gate matrices (KDA ``w_g`` / ``w_beta``, MoE gates)."""
    init_std_mult_beta: float | None = None
    """Init-std multiplier of the KDA write-strength projection ``w_beta`` alone (None: ``init_std_mult_gates``).
    Under MuonH it fixes ``w_beta``'s norm for the whole run."""
    shared_ungated_relu2: bool = False
    """Shared experts are ``relu(x @ W_up)^2 @ W_down`` with no gate projection (truly ungated: the gate
    GEMM is dropped from the fused projection). Parameter-match with 1.5x ``shared_expert_intermediate_dim``."""
    moe_ragged_chunks: int = 2
    """Sequential local-expert chunks of the ``ragged_all_to_all`` backend (each a dispatch, expert MLP and
    return); fewer chunks mean fewer, larger collectives at more transport-buffer memory."""
    moe_implementation: str = "fixed_pooled_wave_all_to_all"
    """Expert-parallel transport, one of ``MOE_IMPLEMENTATIONS``. ``ragged_all_to_all`` ignores
    ``pooled_transport_capacity_factor`` and sizes its receiver buffers by ``capacity_factor``; its XLA transport
    kernel is ``GrugRunConfig.ragged_transport``."""
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
    learnable_qk_mult: bool = False
    """A learnable scalar per softmax-attention layer (init ``qk_mult``) in place of the fixed ``qk_mult``."""

    def __post_init__(self) -> None:
        if self.lm_head_grad_dump_every and not self.lm_head_grad_probe:
            raise ValueError("lm_head_grad_dump_every needs lm_head_grad_probe")
        if self.lm_head_grad_clip is not None and (not self.lm_head_grad_probe or self.lm_head_grad_clip <= 0):
            raise ValueError("lm_head_grad_clip needs lm_head_grad_probe and a positive threshold")
        if self.lm_head_grad_clip is not None and self.lm_head_grad_probe_site == GradProbeSite.PRE_NORM:
            raise ValueError("lm_head_grad_clip clips at the post-norm probe site")
        if unknown := set(self.sconv_sites) - set(SCONV_SITES):
            raise ValueError(f"sconv_sites must be a subset of {SCONV_SITES}, got {sorted(unknown)}")
        if self.moe_implementation not in MOE_IMPLEMENTATIONS:
            raise ValueError(f"moe_implementation must be one of {MOE_IMPLEMENTATIONS}, got {self.moe_implementation!r}")
        if not self.dense_mlp and self.num_experts_per_token >= self.num_experts:
            # QB routing takes top-(k+1) and keeps the last entry as the threshold alpha, so a
            # full-bank top-k asks `jax.lax.top_k` for more entries than the router has experts.
            raise ValueError("num_experts_per_token must be < num_experts, because QB routing selects top-(k+1)")
        if self.local_mixer == LocalMixer.KDA and not self.attn_res:
            raise ValueError("local_mixer=kda requires attn_res (KDA layers run in the unrolled AttnRes loop)")
        if self.kda_chunk_size not in KDA_CHUNK_TILES:
            raise ValueError(f"kda_chunk_size must be one of {list(KDA_CHUNK_TILES)}, got {self.kda_chunk_size}")
        if self.entropy_skip_layers:
            if not self.attn_res:
                raise ValueError("entropy_skip_layers needs attn_res")
            if not 0.0 < self.extra_mlp_frac < 1.0:
                raise ValueError(f"extra_mlp_frac must be in (0, 1), got {self.extra_mlp_frac}")
        if self.skip_adapter_dim and not self.attn_res:
            raise ValueError("skip_adapter_dim needs attn_res")
        if self.skip_router_layers and (not self.attn_res or not 0.0 < self.skip_router_frac < 1.0):
            raise ValueError("skip_router_layers needs attn_res and 0 < skip_router_frac < 1")
        if self.attn_res_kv_layers:
            if not self.attn_res:
                raise ValueError("attn_res_kv_layers needs attn_res")
            cutoff = self.kv_source_layers
            if not 0 < cutoff < self.num_layers:
                raise ValueError(f"attn_res_kv_source_layers must be in 1..{self.num_layers - 1}, got {cutoff}")
            if bad := [i for i in self.attn_res_kv_layers if not cutoff <= i < self.num_layers]:
                raise ValueError(f"attn_res_kv_layers {bad} must read only earlier layers (>= {cutoff})")

    @property
    def expert_in_dim(self) -> int:
        """Width the routed experts read and write: the MoE latent, or the hidden dim without one."""
        return self.latent_dim if self.latent_dim is not None else self.hidden_dim

    @property
    def kv_source_layers(self) -> int:
        """The leading layers whose outputs ``attn_res_kv_layers`` read (``attn_res_kv_source_layers``)."""
        return self.num_layers // 2 if self.attn_res_kv_source_layers is None else self.attn_res_kv_source_layers

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


def _proj(x: jax.Array, w: jax.Array, out_sharding: P | None = None) -> jax.Array:
    """``x[..., k] @ w[k, n]``."""
    return jnp.einsum("...k,kn->...n", x, w, out_sharding=out_sharding)


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


def _log_head_gate_off(head_gate: Float[Array, "B S N"], stats: dict[str, jax.Array]) -> None:
    """Log into ``stats`` the fraction of heads whose output gate, normalized to (0, 1), is near off."""
    g = jax.lax.stop_gradient(head_gate.astype(jnp.float32))
    stats[f"{_OUTSTAT_PREFIX}head_gate_off"] = jnp.mean((g < _HEAD_GATE_OFF).astype(jnp.float32))


class CausalSelfAttention(eqx.Module):
    """Softmax attention: GQA (``w_q``/``w_k``/``w_v``), or MLA with a compressed KV latent
    (``w_q``/``w_dkv``/``w_uk``/``w_uv``); either with half-RoPE or the Inkling bias."""

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"] | None  # GQA only
    w_v: Float[Array, "D MH"] | None  # GQA only
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"] | None
    sconv_k: ShortConv | None  # SConv after the K projection ("k" in cfg.sconv_sites)
    rel_pos: InklingRelPos | None  # Inkling relative-position bias (replaces RoPE when set)
    w_dkv: Float[Array, "D L"] | None  # MLA only
    kv_latent_norm: "RMSNorm | None"
    w_uk: Float[Array, "L NH"] | None
    w_uv: Float[Array, "L NH"] | None
    value_embed: Float[Array, "V NH"] | None
    ve_lambda: Float[Array, " 2"] | None  # (lambda1 on v, lambda2 on the value embedding)
    ve_gate: Float[Array, "D N"] | None
    qk_mult: Float[Array, ""] | None  # learnable logit scale (cfg.learnable_qk_mult)
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        std = cfg.initializer_std
        attn_gate = reshard(jnp.zeros((d, n)), P(None, None)) if cfg.attn_output_gate else None
        sconv_k_channels = n * h if cfg.mla else m * h
        sconv_k = (
            ShortConv.init(sconv_k_channels, cfg.sconv_kernel, cfg.sconv_impl)
            if cfg.sconv and "k" in cfg.sconv_sites
            else None
        )
        # Without Inkling the MLA layers are NoPE (they are global, so RoPE is disabled there).
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
                sconv_k=sconv_k,
                rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
                w_dkv=reshard(_init_weight(k_dkv, (d, kvl), std), P(_FSDP_AXES, None)),
                kv_latent_norm=RMSNorm.init(kvl, cfg.layer_norm_eps),
                w_uk=reshard(_init_weight(k_uk, (kvl, n * h), std), P(None, "model")),
                w_uv=reshard(_init_weight(k_uv, (kvl, n * h), std), P(None, "model")),
                value_embed=(
                    reshard(_init_weight(k_ve, (cfg.vocab_size, n * h), std), P(None, None)) if use_ve else None
                ),
                ve_lambda=_ve_lambda_init(cfg.value_embeds),
                ve_gate=reshard(jnp.zeros((d, n)), P(None, None)) if cfg.value_embeds in _GATED_VALUE_EMBEDS else None,
                qk_mult=_qk_mult_init(cfg),
                cfg=cfg,
            )
        # Eight keys (three unused) so every initial weight matches the runs before the cleanup.
        k_q, k_k, k_v, k_o, k_rel, *_ = random.split(key, 8)
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std), P("model", _FSDP_AXES)),
            attn_gate=attn_gate,
            sconv_k=sconv_k,
            rel_pos=InklingRelPos.init(cfg, key=k_rel) if cfg.inkling_relpos else None,
            w_dkv=None,
            kv_latent_norm=None,
            w_uk=None,
            w_uv=None,
            value_embed=None,
            ve_lambda=None,
            ve_gate=None,
            qk_mult=_qk_mult_init(cfg),
            cfg=cfg,
        )

    def _mla_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        token_ids: Int[Array, "B S"] | None,
        kv_input: Float[Array, "B S D"] | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """MLA with a compressed KV latent: full-rank q; k and v up-projected from one normed latent (of
        ``kv_input`` when given)."""
        assert self.w_dkv is not None and self.kv_latent_norm is not None
        assert self.w_uk is not None and self.w_uv is not None
        head_dim = self.cfg.inferred_head_dim
        latent = _proj(x if kv_input is None else kv_input, self.w_dkv)
        q = rearrange(_proj(x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        kv_latent = self.kv_latent_norm(latent)
        k_flat = _proj(kv_latent, self.w_uk)
        if self.sconv_k is not None:
            k_flat = self.sconv_k(k_flat, sconv_segment_ids)
        k = rearrange(k_flat, "... (n d) -> ... n d", d=head_dim)
        v = rearrange(_proj(kv_latent, self.w_uv), "... (n d) -> ... n d", d=head_dim)
        if self.value_embed is not None:
            assert self.ve_lambda is not None and token_ids is not None
            ve = _embedding_gather(self.value_embed.astype(x.dtype), token_ids)
            ve = rearrange(ve, "... (n d) -> ... n d", d=head_dim)
            lam = self.ve_lambda.astype(x.dtype)
            ve_weight = _value_embed_weight(self.cfg.value_embeds, x, self.ve_gate, lam)
            v = lam[0] * v + ve_weight * reshard(ve, _partition_spec_of(v) or P(_BATCH_AXES, None, None, None))
        return q, k, v

    def _gqa_qkv(
        self,
        x: Float[Array, "B S D"],
        sconv_segment_ids: Int[Array, "B S"] | None,
        is_global: bool | jax.Array,
        kv_input: Float[Array, "B S D"] | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        assert self.w_k is not None and self.w_v is not None
        head_dim = self.cfg.inferred_head_dim
        kv_in = x if kv_input is None else kv_input
        q_flat = _proj(x, self.w_q)
        k_flat = _proj(kv_in, self.w_k)
        v_flat = _proj(kv_in, self.w_v)
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

    def _rope(self, q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        """RoPE on the first half of the Q/K head dim (the whole head with ``rope_full_head``)."""
        head_dim = self.cfg.inferred_head_dim
        half = head_dim if self.cfg.rope_full_head else head_dim // 2
        q_rot, k_rot = apply_rotary_embedding(
            q[..., :half], k[..., :half], seq_len=q.shape[1], head_dim=half, rope=self.cfg.rope
        )
        return jnp.concatenate([q_rot, q[..., half:]], axis=-1), jnp.concatenate([k_rot, k[..., half:]], axis=-1)

    def _transform_qk(
        self, q: jax.Array, k: jax.Array, segment_ids: Int[Array, "B S"] | None, disable_rope: bool | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Key offset, q/k norms, RoPE and qk_mult: everything between the projections and the kernel."""
        cfg = self.cfg
        if cfg.mla and cfg.mla_key_offset:
            k = partial_key_offset(k, segment_ids, _partition_spec_of(k))
        if cfg.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)
        elif cfg.mla_k_norm and cfg.mla:
            k = rms_norm(k)
        if cfg.mla_q_norm and cfg.mla and not cfg.qk_norm:
            q = rms_norm(q)
        # The Inkling bias replaces RoPE (no rotation).
        if self.rel_pos is None:
            if cfg.rope_global_layers:
                q, k = self._rope(q, k)
            elif isinstance(disable_rope, bool):
                if not disable_rope:
                    q, k = self._rope(q, k)
            else:
                q_roped, k_roped = self._rope(q, k)
                keep = ~jnp.asarray(disable_rope, dtype=jnp.bool_)
                q = jnp.where(keep, q_roped, q)
                k = jnp.where(keep, k_roped, k)
        if self.qk_mult is None:
            q = q * cfg.qk_mult
        else:
            q = q * self.qk_mult.astype(q.dtype)
        return q, k

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
        token_ids: Int[Array, "B S"] | None = None,
        kv_input: Float[Array, "B S D"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``kv_input`` (``attn_res_kv_layers``) is the normed stream the K/V projections read instead of ``x``.
        Returns the output and logging-only stats."""
        head_dim = self.cfg.inferred_head_dim
        # segment_ids (packed-document boundaries) come from the mask so the SConv never mixes across a
        # document boundary.
        sconv_segment_ids = _sconv_segment_ids(mask)
        if self.cfg.mla:
            q, k, v = self._mla_qkv(x, sconv_segment_ids, token_ids, kv_input)
        else:
            q, k, v = self._gqa_qkv(x, sconv_segment_ids, is_global, kv_input)
        stats: dict[str, jax.Array] = {}
        # The Inkling bias: a per-head content-dependent bias (from x) on the pre-softmax logits.
        rel_bias = self.rel_pos(x) if self.rel_pos is not None else None
        q, k = self._transform_qk(q, k, sconv_segment_ids, disable_rope)
        # GPU-only kernels; auto-select off-GPU so the model still lowers on CPU.
        attn_impl = self.cfg.gpu_attention if jax.default_backend() == "gpu" else None
        attn_out = attention(q, k, v, mask, implementation=attn_impl, rel_bias=rel_bias)[..., :head_dim]
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
        if self.attn_gate is not None:
            gate_logit = jnp.einsum("bsd,dn->bsn", x, self.attn_gate).astype(jnp.float32)
            _log_head_gate_off(jax.nn.sigmoid(gate_logit), stats)
            gate = (2 * jax.nn.sigmoid(gate_logit)).astype(attn_out.dtype)
            attn_out = gate[..., None] * attn_out
        # Merge heads into the hidden dim while keeping model-axis sharding for w_o.
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:2], math.prod(attn_out.shape[2:])),
            out_sharding=P(_BATCH_AXES, None, "model"),
        )
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_batch_spec()), stats


def _qk_mult_init(cfg: GrugModelConfig) -> jax.Array | None:
    if not cfg.learnable_qk_mult:
        return None
    return jnp.full((), cfg.qk_mult, jnp.float32)


def _kda_dt_bias_init(cfg: GrugModelConfig, key: PRNGKeyArray, shape: tuple[int, int]) -> jax.Array:
    """``dt_bias`` such that ``|g| = KDA_MIN_LOG_DECAY * sigmoid(dt_bias)`` (zero gate input, ``A_log = 0``)
    is log-uniform in ``kda_dt_range``."""
    lo, hi = cfg.kda_dt_range
    decay = jnp.exp(random.uniform(key, shape, minval=math.log(lo), maxval=math.log(hi)))
    return jax.scipy.special.logit(decay / KDA_MIN_LOG_DECAY)


class KdaKernelOptions(NamedTuple):
    """Static options of the KDA kernel call (``_kda_kernel``)."""

    chunk_size: int  # fused (GPU) kernels' chunk length, see KDA_CHUNK_TILES
    save_chunk_states: bool
    state_grads: StateGrads
    state_block_v: int


def _kda_kernel(q, k, v, g, beta, extras: dict[str, jax.Array], *, options: KdaKernelOptions):
    """KDA on the model layout ``(B, S, H, d)``: the fused Pallas kernels on GPU, else the XLA ``chunk_kda``
    (heads-first layout, ``KDA_CHUNK_SIZE``). ``extras`` optionally holds ``segment_ids`` and ``gate_p`` /
    ``gate_bias`` (``kda_onchip_decay_gate``: ``g`` is then the gate pre-activation of
    ``-KDA_MIN_LOG_DECAY * sigmoid(gate_p * (g + gate_bias))``)."""
    segment_ids = extras.get("segment_ids")
    gate = (extras["gate_p"], extras["gate_bias"]) if "gate_p" in extras else None
    gate_fn = GateFn(GateKind.SIGMOID, KDA_MIN_LOG_DECAY)
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
            chunk_size=options.chunk_size,
            sub_chunk_size=KDA_CHUNK_TILES[options.chunk_size],
            save_chunk_states=options.save_chunk_states,
            state_grads=options.state_grads,
            state_block_v=options.state_block_v,
        )
    if gate is not None:
        g = gate_log_decay(gate_fn, g.astype(jnp.float32), gate[0], gate[1])
    q, k, v, g, beta = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, g, beta))
    seg = None if segment_ids is None else segment_ids[:, None, :]  # same documents for every head
    out = chunk_kda(q, k, v, g, beta, chunk_size=KDA_CHUNK_SIZE, segment_ids=seg)[0]
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
    No positional encoding and no window.
    """

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D NH"]
    w_v: Float[Array, "D NH"]
    w_o: Float[Array, "NH D"]
    w_g: Float[Array, "D NH"]
    w_a_down: Float[Array, "D R"]
    w_a_up: Float[Array, "R NH"]
    a_log: Float[Array, " N"]
    dt_bias: Float[Array, "N H"]
    w_beta: Float[Array, "D N"]
    o_norm: "RMSNorm"
    sconv_q: ShortConv
    sconv_k: ShortConv
    sconv_v: ShortConv
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "KimiDeltaAttention":
        k_q, k_k, k_v, k_o, k_g, k_ad, k_au, k_b, k_dt = random.split(key, 9)
        d, n, h, r, std = cfg.hidden_dim, cfg.num_heads, cfg.inferred_head_dim, _KDA_GATE_RANK, cfg.initializer_std
        beta_std = std * (cfg.init_std_mult_gates if cfg.init_std_mult_beta is None else cfg.init_std_mult_beta)
        return KimiDeltaAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (d, n * h), std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), std), P("model", _FSDP_AXES)),
            w_g=reshard(_init_weight(k_g, (d, n * h), std * cfg.init_std_mult_gates), P(_FSDP_AXES, "model")),
            w_a_down=reshard(_init_weight(k_ad, (d, r), std), P(_FSDP_AXES, None)),
            w_a_up=reshard(_init_weight(k_au, (r, n * h), std), P(None, "model")),
            a_log=jnp.zeros((n,)),
            dt_bias=_kda_dt_bias_init(cfg, k_dt, (n, h)),
            w_beta=reshard(_init_weight(k_b, (d, n), beta_std), P(None, None)),
            o_norm=RMSNorm.init(h, 1e-6),
            sconv_q=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            sconv_k=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            sconv_v=ShortConv.init(n * h, cfg.sconv_kernel, cfg.sconv_impl),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        segment_ids: Int[Array, "B S"] | None = None,
        kv_input: Float[Array, "B S D"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``kv_input`` (``attn_res_kv_layers``) is the normed stream that the ``k`` / ``v`` projections (and their
        SConvs) read instead of ``x``; beta and the decay stay on ``x``. Also returns logging stats."""
        cfg = self.cfg
        head_dim = cfg.inferred_head_dim
        b, s, _ = x.shape

        def project(w: jax.Array, conv: ShortConv, source: jax.Array) -> jax.Array:
            y = jax.nn.silu(conv(_proj(source, w), segment_ids))
            return rearrange(y, "... (n d) -> ... n d", d=head_dim)

        kv_source = x if kv_input is None else kv_input
        q = project(self.w_q, self.sconv_q, x)
        k = project(self.w_k, self.sconv_k, kv_source)
        v = project(self.w_v, self.sconv_v, kv_source)
        a_low = jnp.einsum("bsd,dr->bsr", x, self.w_a_down)
        a = jnp.einsum("bsr,re->bse", a_low, self.w_a_up)
        a = rearrange(a, "... (n d) -> ... n d", d=self.dt_bias.shape[-1])
        scale = jnp.exp(self.a_log.astype(jnp.float32))[:, None]
        if cfg.kda_onchip_decay_gate:  # the kernels form -5 sigmoid(scale (a + dt_bias)) from the bf16 projection
            g = a
        else:
            g = -KDA_MIN_LOG_DECAY * jax.nn.sigmoid(scale * (a.astype(jnp.float32) + self.dt_bias.astype(jnp.float32)))
        beta = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.w_beta).astype(jnp.float32))

        stats: dict[str, jax.Array] = {}
        spec4 = P(_BATCH_AXES, None, "model", None)
        spec3 = P(_BATCH_AXES, None, "model")
        q, k, v, g = (reshard(t, spec4) for t in (q, k, v, g))
        beta = reshard(beta, spec3)
        extras: dict[str, jax.Array] = {}
        extra_specs: dict[str, P] = {}
        if cfg.kda_onchip_decay_gate:
            head_spec = P("model", None)
            extras["gate_p"] = reshard(jnp.broadcast_to(scale, self.dt_bias.shape), head_spec)
            extras["gate_bias"] = reshard(self.dt_bias.astype(jnp.float32), head_spec)
            extra_specs["gate_p"] = extra_specs["gate_bias"] = head_spec
        if segment_ids is not None:
            extras["segment_ids"] = reshard(jnp.broadcast_to(segment_ids, (b, s)), P(_BATCH_AXES, None))
            extra_specs["segment_ids"] = P(_BATCH_AXES, None)
        options = KdaKernelOptions(
            cfg.kda_chunk_size, cfg.kda_save_chunk_states, cfg.kda_state_grads, cfg.kda_state_block_v
        )
        in_specs = (spec4,) * 4 + (spec3, extra_specs)
        # The Pallas custom VJPs are not vma-annotated, so skip the varying-axes check.
        kernel = jax.shard_map(
            functools.partial(_kda_kernel, options=options),
            mesh=get_abstract_mesh(),
            in_specs=in_specs,
            out_specs=spec4,
            check_vma=False,
        )
        o = kernel(q, k, v, g, beta, extras)
        o = self.o_norm(o.astype(x.dtype))
        o = jnp.reshape(o, (b, s, cfg.num_heads * head_dim), out_sharding=P(_BATCH_AXES, None, "model"))
        gate = jax.nn.sigmoid(_proj(x, self.w_g))
        gate_heads = rearrange(gate, "... (n d) -> ... n d", d=head_dim)
        _log_head_gate_off(jnp.mean(gate_heads.astype(jnp.float32), axis=-1), stats)
        out = _proj(o * gate, self.w_o, out_sharding=_batch_spec())
        return out, stats


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.full((dim,), 1.0, dtype=jnp.float32), eps=eps)

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
    w_gate: jax.Array | None  # None: an ungated ReLU^2 shared expert (cfg.shared_ungated_relu2)
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
        self, x: Float[Array, "B S D"], *, activation: MoeActivation = ActivationFunctionEnum.silu
    ) -> Float[Array, "B S D"]:
        assert self.w_gate is not None
        activation_fn = activation.to_jax_fn() if isinstance(activation, ActivationFunctionEnum) else activation
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        # Pin the token axis to the batch spec so XLA can infer w_down's output sharding, and so the residual
        # add matches on multi-node.
        out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights, on the LatentMoE latent when ``latent_dim`` is set."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    w_latent_down: jax.Array | None
    latent_norm: RMSNorm | None
    w_latent_up: jax.Array | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert, k_down, k_up = random.split(key, 4)
        expert_axis_size = _mesh_axis_size(get_abstract_mesh(), "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, latent = cfg.hidden_dim, cfg.latent_dim
        # Routed experts live in the latent space; the router reads the full-width token, so its
        # own projection keeps `hidden_dim`.
        width = cfg.expert_in_dim
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, cfg.num_experts), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((cfg.num_experts,)),
            w_latent_down=(
                None
                if latent is None
                else reshard(_init_weight(k_down, (d, latent), cfg.initializer_std), P(_FSDP_AXES, "model"))
            ),
            latent_norm=None if latent is None else RMSNorm.init(latent, cfg.layer_norm_eps),
            w_latent_up=(
                None
                if latent is None
                else reshard(_init_weight(k_up, (width, d), cfg.initializer_std), P("model", _FSDP_AXES))
            ),
            expert_mlp=_expert_mlp_init(cfg, width, k_expert),
            cfg=cfg,
        )

    def input_projection_weights(self, dtype: jnp.dtype) -> list[jax.Array]:
        """The ``[D, *]`` projections this MLP applies to its input: the router, then the latent down."""
        weights = [reshard(self.router, P(None, None))]
        if self.w_latent_down is not None:
            weights.append(reshard(self.w_latent_down.astype(dtype), P(None, None)))
        return weights

    def _combine_weights(self, unbiased_topk: Float[Array, "T K"]) -> Float[Array, "T K"]:
        """The K selected experts' float32 combine weights from their unbiased logits (``RouterCombine``)."""
        k = unbiased_topk.shape[-1]
        renorm_sum = self.cfg.routing_renorm_sum
        combine = self.cfg.router_combine
        if combine == RouterCombine.SOFTMAX_RENORM:
            return renorm_sum * jax.nn.softmax(unbiased_topk, axis=-1)
        if combine == RouterCombine.SIGMOID_RAW:
            return jax.nn.sigmoid(unbiased_topk) * (renorm_sum / (k / 2))
        if combine == RouterCombine.SQRT_SOFTPLUS_RAW:
            gate = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
            return gate * (renorm_sum / (k * math.sqrt(math.log(2.0))))
        if combine == RouterCombine.SQRT_SOFTPLUS_RENORM:
            # The floor keeps sqrt's gradient finite if softplus underflows to 0 (logit below about -100).
            weights = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
        else:
            weights = jax.nn.sigmoid(unbiased_topk)
        denom = jnp.sum(weights, axis=-1, keepdims=True)
        return weights * (renorm_sum / (denom + 1e-9))

    @named_call
    def __call__(
        self, x: Float[Array, "B S D"], projected: list[jax.Array] | None = None
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``projected`` holds ``x_flat @ w`` for each of ``input_projection_weights`` when the caller computed them
        already (fused with other projections of the same input)."""
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        if projected is None:
            projected = [jnp.einsum("td,de->te", x_flat, w) for w in self.input_projection_weights(x_flat.dtype)]
        router_logits = projected[0].astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        k = self.cfg.num_experts_per_token
        num_experts = self.cfg.num_experts
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        topk_logits, selected_experts = _small_top_k(biased_logits, k + 1)
        alpha = topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        # Combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = self._combine_weights(unbiased_topk)
        combine_weights = combine_weights_f.astype(x.dtype)
        mesh = get_abstract_mesh()
        # Per-shard partials only; the cross-device reduction happens once after the layer loop.
        router_stats = local_routing_stats(
            reshard(selected_experts, P(_BATCH_AXES, None)),
            reshard(router_logits, P(_BATCH_AXES, None)),
            mesh,
            num_experts=num_experts,
            batch_axes=_BATCH_AXES,
        )
        # Sharded QB: estimate each expert's threshold beta from the margins `s - alpha` by binning
        # them into fixed bins over the live global range and reading the (1-K/E) quantile.
        beta, margin_min, margin_max = qb_beta_hist(
            reshard(router_logits - alpha, P(_BATCH_AXES, None)),
            mesh,
            target_share=k / num_experts,
            num_experts=num_experts,
            n_bins=_QB_HIST_BINS,
            batch_axes=_BATCH_AXES,
        )
        router_stats["qb_beta"] = beta
        router_stats[f"{_OUTSTAT_PREFIX}moe_active_k"] = jnp.asarray(float(k), jnp.float32)
        router_stats["margin_min"] = margin_min
        router_stats["margin_max"] = margin_max

        routed_input = x_flat
        if self.w_latent_down is not None and self.latent_norm is not None:
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = self.latent_norm(reshard(projected[1], _batch_spec()))
        routed_flat, overflow = _run_expert_bank(
            self.expert_mlp, self.cfg, routed_input, selected_experts, combine_weights
        )
        router_stats["capacity_overflow"] = overflow.dropped
        router_stats["sender_capacity_overflow"] = overflow.sender_dropped
        router_stats["receiver_capacity_overflow"] = overflow.receiver_dropped

        # Expand after the combine: `expert_mlp` already returns the weight-summed expert output,
        # which is the vector the paper's W_up acts on.
        if self.w_latent_up is not None:
            routed_flat = _proj(routed_flat, self.w_latent_up.astype(routed_flat.dtype), out_sharding=_batch_spec())

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


def _shared_experts_tail(
    cfg: GrugModelConfig,
    parts: list[jax.Array],
    num_shared: int,
    gated: bool,
    w_down: jax.Array,
    out_sharding: P | None,
) -> jax.Array:
    """Shared-expert output from each expert's gate/up projections ``parts`` (gates first when ``gated``)."""
    if gated:
        gates, ups = parts[:num_shared], parts[num_shared:]
        act = ActivationFunctionEnum(cfg.expert_activation).to_jax_fn()
        hidden = jnp.concatenate([act(g) * u for g, u in zip(gates, ups, strict=True)], axis=1)
    else:
        hidden = jnp.concatenate([jnp.square(jax.nn.relu(u)) for u in parts], axis=1)
    return _proj(hidden, w_down, out_sharding=out_sharding)


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
    gated = shared[0].w_gate is not None
    shared_weights = [reshard(e.w_gate, replicated) for e in shared if gated] + [
        reshard(e.w_up, replicated) for e in shared
    ]
    w_down = jnp.concatenate([reshard(e.w_down, replicated) for e in shared], axis=0)
    parts = list(_fused_projection(x_flat, tuple(moe_weights + shared_weights)))
    routed, stats = mlp(x, projected=parts[: len(moe_weights)])
    shared_out = _shared_experts_tail(mlp.cfg, parts[len(moe_weights) :], len(shared), gated, w_down, _batch_spec())
    shared_out = _batch_reshard(rearrange(shared_out, "(b s) d -> b s d", b=b, s=s))
    return routed + shared_out, stats


def _sconv_segment_ids(mask: AttentionMask | jax.Array) -> jax.Array | None:
    """segment_ids (packed-document boundaries) for the SConvs and KDA; None when unpacked."""
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    return segment_ids[0] if segment_ids is not None else None


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm | None
    attn: CausalSelfAttention | KimiDeltaAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm | None
    mlp: MoEMLP | DenseMLP
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: ShortConv | None
    sconv_mlp: ShortConv | None
    # Block AttnRes pseudo-queries of the attention and MLP sublayers (None without cfg.attn_res).
    attn_res_query_attn: Float[Array, " D"] | None
    attn_res_query_mlp: Float[Array, " D"] | None
    attn_res_query_kv: Float[Array, " D"] | None  # cfg.attn_res_kv_layers (every layer carries one; those read it)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, use_kda: bool = False) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        attn_cfg = cfg if cfg.mla_num_heads is None else dataclasses.replace(cfg, num_heads=cfg.mla_num_heads)
        attn = (
            KimiDeltaAttention.init(cfg, key=attn_key) if use_kda else CausalSelfAttention.init(attn_cfg, key=attn_key)
        )
        # KDA blocks have no branch-output SConv (K3 has only the q/k/v convs).
        use_attn_sconv = cfg.sconv and "attn" in cfg.sconv_sites and not use_kda
        # Zero-init: every source scores 0, so each gate starts as a uniform average of its sources.
        attn_res_query = reshard(jnp.zeros((cfg.hidden_dim,), jnp.float32), P(None)) if cfg.attn_res else None
        shared = None
        if cfg.dense_mlp:
            # Dense block: one SwiGLU DenseMLP(hidden, intermediate_dim), no MoE and no shared experts.
            mlp = DenseMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key)
        else:
            mlp = MoEMLP.init(cfg, key=mlp_key)
            if cfg.shared_expert_intermediate_dim > 0:
                num_shared = cfg.num_shared_experts
                shared_keys = (shared_key,) if num_shared == 1 else tuple(random.split(shared_key, num_shared))
                shared = tuple(
                    DenseMLP.init(cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=key)
                    for key in shared_keys
                )
                if cfg.shared_ungated_relu2:
                    shared = tuple(eqx.tree_at(lambda m: m.w_gate, e, None, is_leaf=lambda x: x is None) for e in shared)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key) if cfg.sublayer_gated_norm else None
            ),
            attn=attn,
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=(
                GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key) if cfg.sublayer_gated_norm else None
            ),
            mlp=mlp,
            shared=shared,
            sconv_attn=ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel, cfg.sconv_impl) if use_attn_sconv else None,
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel, cfg.sconv_impl)
                if cfg.sconv and "mlp" in cfg.sconv_sites
                else None
            ),
            attn_res_query_attn=attn_res_query,
            attn_res_query_mlp=attn_res_query,
            attn_res_query_kv=attn_res_query if cfg.attn_res_kv_layers else None,
        )

    def attn_input(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
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
        kv_input: Float[Array, "B S D"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """The attention sublayer on its input ``h``; ``kv_input`` (``attn_res_kv_layers``) is the normed stream
        the K/V projections read. Returns the branch output and the mixer's logging stats."""
        attn_in = self.attn_input(h)
        if isinstance(self.attn, KimiDeltaAttention):
            # KDA has no positional encoding or window; it only needs the document boundaries.
            out, stats = self.attn(attn_in, _sconv_segment_ids(mask), kv_input=kv_input)
        else:
            out, stats = self.attn(
                attn_in, mask, disable_rope=disable_rope, is_global=is_global, token_ids=token_ids, kv_input=kv_input
            )
        if self.sconv_attn is not None:
            out = self.sconv_attn(out, _sconv_segment_ids(mask))
        return out, stats

    def mlp_branch(
        self, h: Float[Array, "B S D"], mask: AttentionMask | jax.Array
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        mlp_in = self._mlp_input(h)
        stats: dict[str, jax.Array] = {}
        if isinstance(self.mlp, DenseMLP):
            out = self.mlp(mlp_in)
        elif self.shared is not None:
            out, stats = moe_and_shared_fused(self.mlp, self.shared, mlp_in)
        else:
            out, stats = self.mlp(mlp_in)
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
        attn_out, _ = self.attn_branch(x, mask, disable_rope, is_global)
        x = x + attn_out
        mlp_out, router_stats = self.mlp_branch(x, mask)
        return x + mlp_out, router_stats


def _expert_mlp_init(cfg: GrugModelConfig, width: int, key: PRNGKeyArray) -> MoEExpertMlp:
    """The routed expert bank on ``width``-wide inputs and outputs; ``moe_ungated_relu2`` drops the gate
    (``w_gate=None``) and uses ReLU."""
    mlp = MoEExpertMlp.init(
        num_experts=cfg.num_experts,
        hidden_dim=width,
        output_dim=width,
        intermediate_dim=cfg.intermediate_dim,
        initializer_std=cfg.initializer_std,
        key=key,
        implementation=cfg.moe_implementation,
        activation=(
            ActivationFunctionEnum.relu if cfg.moe_ungated_relu2 else ActivationFunctionEnum(cfg.expert_activation)
        ),
        capacity_factor=cfg.capacity_factor,
        pooled_transport_capacity_factor=cfg.pooled_transport_capacity_factor,
    )
    if cfg.moe_ungated_relu2:
        mlp = eqx.tree_at(lambda m: m.w_gate, mlp, None, is_leaf=lambda x: x is None)
    return mlp


def _run_expert_bank(
    em: MoEExpertMlp,
    cfg: GrugModelConfig,
    routed_input: jax.Array,
    selected: jax.Array,
    combine_weights: jax.Array,
):
    """Dispatch the expert bank: ungated ReLU^2 through the EP backends' ungated path (or the tied gate on the
    dropless local backends), gated experts through ``MoEExpertMlp``. Returns ``(out, overflow)``."""
    if em.w_gate is None:
        ungated = em.implementation in MOE_IMPLEMENTATIONS and cfg.moe_ungated_kernel
        return moe_mlp(
            routed_input,
            selected,
            combine_weights,
            em.w_up if ungated else jnp.concatenate([em.w_up, em.w_up], axis=-1),
            em.w_down,
            # relu(u) * u = relu(u)^2 for the backends that take the gate tied to W_up.
            activation=ActivationFunctionEnum.relu2 if ungated else em.activation,
            implementation=em.implementation,
            mesh=get_abstract_mesh(),
            capacity_factor=em.capacity_factor,
            pooled_transport_capacity_factor=em.pooled_transport_capacity_factor,
            report_capacity_overflow=True,
            expert_chunks=em.expert_chunks,
            ragged_expert_chunks=cfg.moe_ragged_chunks,
        )
    return em(routed_input, selected, combine_weights, mesh=get_abstract_mesh(), report_capacity_overflow=True)


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


def _vocab_entropy(
    head_in: Float[Array, "B S E"], lm_head: Float[Array, "E V"], cap: float | None
) -> Float[Array, "B S"]:
    """Stop-gradient entropy of each position's next-token distribution, one online pass over vocabulary chunks
    (``H = log s + m - t / s`` with running max ``m``, ``s = sum e^(z - m)``, ``t = sum e^(z - m) z``), so the
    ``[B, S, V]`` logits are never materialized. ``cap`` applies the logit soft cap first."""
    head_in = jax.lax.stop_gradient(head_in)
    vocab = lm_head.shape[1]
    chunk = math.gcd(vocab, _ENTROPY_CHUNK)
    # Scanned chunks must be unsharded along the scan axis: gather the (FSDP / model-sharded) head first.
    weights = rearrange(reshard(jax.lax.stop_gradient(lm_head), P(None, None)), "e (n c) -> n e c", c=chunk)
    b, s_len, _ = head_in.shape
    init = tuple(reshard(jnp.full((b, s_len), v, jnp.float32), P(_BATCH_AXES, None)) for v in (-jnp.inf, 0.0, 0.0))

    def step(carry, w):
        m, total, weighted = carry
        z = jnp.einsum("bse,ec->bsc", head_in, w.astype(head_in.dtype), preferred_element_type=jnp.float32)
        z = reshard(z, P(_BATCH_AXES, None, None))
        if cap is not None:
            z = cap * jnp.tanh(z / cap)
        m_new = jnp.maximum(m, jnp.max(z, axis=-1))
        scale = jnp.exp(m - m_new)
        e = jnp.exp(z - m_new[..., None])
        return (m_new, total * scale + jnp.sum(e, -1), weighted * scale + jnp.sum(e * z, -1)), None

    (m, total, weighted), _ = jax.lax.scan(step, init, weights)
    lse = jnp.log(total) + m
    return lse - weighted / total


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
    # Per-source rows through ``_logit_rows``: slicing ``weights`` would pad every row's gradient to [N, B, S].
    rows = _logit_rows(weights)
    mixed = rows[0][..., None] * sources[0].astype(jnp.float32)
    for weight, source in zip(rows[1:], sources[1:], strict=True):
        mixed = mixed + weight[..., None] * source.astype(jnp.float32)
    return weights, mixed


def _stream_sum(sources: tuple[jax.Array, ...]) -> jax.Array:
    """The straight sum of the AttnRes sources (a standard residual stream), in the sources' dtype."""
    total = sources[0].astype(jnp.float32)
    for src in sources[1:]:
        total = total + src.astype(jnp.float32)
    return reshard(total.astype(sources[0].dtype), _batch_spec())


def _residual_stream(blocks: tuple[jax.Array, ...], partial: jax.Array | None) -> jax.Array:
    """The standard residual stream entering a layer: the sum of the AttnRes history (blocks and partial)."""
    return functools.reduce(jnp.add, [*blocks, *([] if partial is None else [partial])])


def _gate_z(logits: list[jax.Array]) -> jax.Array:
    """Mean over tokens of ``logsumexp(logits)^2`` for one gate."""
    return jnp.mean(jnp.square(jax.nn.logsumexp(jnp.stack(logits), axis=0)))


@named_call
def _attn_res_mix(
    blocks: tuple[jax.Array, ...],
    block_logits: tuple[jax.Array, ...],
    partial: Float[Array, "B S D"] | None,
    queries: Float[Array, "G D"],
    gate_index: int,
    eps: float,
    *,
    plain_sum: bool,
) -> tuple[Float[Array, "B S D"], jax.Array, jax.Array]:
    """One AttnRes gate: softmax over the completed blocks (+ the running partial) and their weighted sum
    (``plain_sum``: their plain sum, ``attn_res_plain_sum``).

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
    if partial is not None:
        sources.append(partial)
        logits.append(_attn_res_source_logits(partial, queries[gate_index][None], eps)[0])
    weights, mixed = _softmax_mix(logits, sources)
    mean_weights = jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim))))
    return reshard(mixed.astype(sources[0].dtype), _batch_spec()), _gate_z(logits), mean_weights


# ``(layer, blocks, block_logits, partial, queries, extras)``: the differentiable inputs of one AttnRes layer.
_LayerArgs = tuple[Block, tuple[jax.Array, ...], tuple[jax.Array, ...], jax.Array | None, jax.Array, dict | None]


def _attn_res_layer(
    diff_args: _LayerArgs,
    mask: AttentionMask,
    token_ids: Int[Array, "B S"],
    use_long: bool,
    layer_index: int,
    eps: float,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """One Block AttnRes layer on ``diff_args``: the attention and MLP sublayers, each reading its gate's mix of
    the history.

    ``partial`` is None right after a block boundary (it was just rolled into ``blocks``), in which case this
    layer starts a fresh partial sum. ``extras`` optionally holds ``_SKIP_KEEP``, the token mask of a skipped
    layer.
    """
    layer, blocks, block_logits, partial, queries, extras = diff_args
    cfg = layer.attn.cfg

    def mix(p: jax.Array | None, gate_index: int, leading_sources_only: bool = False):
        if not leading_sources_only:
            return _attn_res_mix(blocks, block_logits, p, queries, gate_index, eps, plain_sum=cfg.attn_res_plain_sum)
        # The leading layers' sources: the blocks rolled at layers 0, seg, ... <= the cutoff (each holds only
        # earlier layers); the partial and later blocks are later layers' work.
        keep = sum(start <= cfg.kv_source_layers for start in attn_res_block_starts(cfg))
        return _attn_res_mix(
            blocks[:keep], block_logits[:keep], None, queries, gate_index, eps, plain_sum=cfg.attn_res_plain_sum
        )

    h, z_attn, w_attn = mix(partial, 2 * layer_index)
    attn_branch = type(layer).attn_branch
    if cfg.attn_res_remat_attention:
        attn_branch = eqx.filter_checkpoint(attn_branch, policy=None)
    kv_input = None
    if layer_index in cfg.attn_res_kv_layers:
        # KV-gate queries follow the per-layer gates: the KV gate of layer i is row 2 L + i.
        h_kv, _, _ = mix(None, 2 * cfg.num_layers + layer_index, leading_sources_only=True)
        kv_input = layer.attn_input(h_kv)
    attn_out, attn_stats = attn_branch(layer, h, mask, use_long, use_long, token_ids, kv_input)
    skip_keep = None if extras is None else extras.get(_SKIP_KEEP)
    if skip_keep is not None:
        attn_out = attn_out * skip_keep[..., None].astype(attn_out.dtype)
    partial = attn_out if partial is None else partial + attn_out
    h, z_mlp, w_mlp = mix(partial, 2 * layer_index + 1)
    mlp_out, router_stats = layer.mlp_branch(h, mask)
    if skip_keep is not None:
        mlp_out = mlp_out * skip_keep[..., None].astype(mlp_out.dtype)
    return partial + mlp_out, {
        **router_stats,
        **attn_stats,
        _ATTN_RES_Z: z_attn + z_mlp,
        _ATTN_RES_W_ATTN: w_attn,
        _ATTN_RES_W_MLP: w_mlp,
    }


def _attn_res_layer_passthrough(diff_args: _LayerArgs, mask, token_ids, use_long, layer_index, eps):
    """``_attn_res_layer`` returning ``(partial, blocks, block_logits, router_stats)``: the history is
    passed through so ``_attn_res_layer_remat`` can thread each block's cotangent layer to layer."""
    _, blocks, block_logits, _, _, _ = diff_args
    partial, router_stats = _attn_res_layer(diff_args, mask, token_ids, use_long, layer_index, eps)
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
    layer's starts. Without the barriers the unrolled loop's peak memory at d1024 is 3.6x a scanned
    layer stack's.
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
    d_partial, d_blocks, d_block_logits, _ = grad_out
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
        ((d_layer, d_blocks_own, d_block_logits_own, d_partial_in, d_queries, d_extras),) = vjp_fn(d_partial)
        d_blocks = tuple(a + b for a, b in zip(d_blocks, d_blocks_own, strict=True))
        d_block_logits = jax.tree.map(jnp.add, d_block_logits, d_block_logits_own)
        # One barrier over every cotangent: the weight gradients are off the critical path, and without
        # it XLA defers them past later layers' backward and keeps their inputs alive.
        return jax.lax.optimization_barrier((d_layer, d_blocks, d_block_logits, d_partial_in, d_queries, d_extras))


def _is_none(x) -> bool:
    return x is None


def _is_long_layer(layer_index: int, num_layers: int, global_every: int) -> bool:
    # Every global_every-th layer is full-causal, and the last layer always is, so a depth that is
    # not a multiple of global_every still ends on a global-context layer.
    return (layer_index + 1) % global_every == 0 or layer_index == num_layers - 1


def _long_layer_schedule(num_layers: int, global_every: int) -> jax.Array:
    return jnp.asarray([_is_long_layer(i, num_layers, global_every) for i in range(num_layers)], dtype=jnp.bool_)


def _kda_layer_indices(cfg: GrugModelConfig) -> tuple[int, ...]:
    """Layers whose mixer is KDA: the local layers when ``local_mixer`` is KDA, else none."""
    if cfg.local_mixer != LocalMixer.KDA:
        return ()
    return tuple(i for i in range(cfg.num_layers) if not _is_long_layer(i, cfg.num_layers, cfg.global_every))


def _stack_layer_indices(cfg: GrugModelConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(softmax_layers, kda_layers)``: the layer indices of ``Transformer.stacked_blocks`` and ``kda_blocks``."""
    kda_layers = _kda_layer_indices(cfg)
    return tuple(i for i in range(cfg.num_layers) if i not in kda_layers), kda_layers


def _unstack_layers(stacked: ArrayStacked[Block]) -> list[Block]:
    """Split the stacked layer params into per-layer modules (split, so the grad is one concatenate)."""
    leaves, treedef = jax.tree.flatten(stacked.stacked)
    split_leaves = [jax.lax.split(leaf, (1,) * stacked.num_layers, axis=0) for leaf in leaves]
    return [treedef.unflatten([parts[i][0] for parts in split_leaves]) for i in range(stacked.num_layers)]


def attn_res_block_starts(cfg: GrugModelConfig) -> tuple[int, ...]:
    """The layers at which Block AttnRes rolls the partial into a new source (``attn_res_block_starts``)."""
    num_blocks = cfg.attn_res_num_blocks
    if not cfg.attn_res_block_starts:
        seg = max(1, cfg.num_layers // num_blocks)
        return tuple(k * seg for k in range(num_blocks) if k * seg < cfg.num_layers)
    starts = tuple(cfg.attn_res_block_starts)
    if len(starts) != num_blocks or starts[0] != 0 or starts != tuple(sorted(set(starts))):
        raise ValueError(f"attn_res_block_starts must be {num_blocks} increasing layers from 0, got {starts}")
    if starts[-1] >= cfg.num_layers:
        raise ValueError(f"attn_res_block_starts {starts} must be < num_layers={cfg.num_layers}")
    return starts


class LayerSkip(NamedTuple):
    """An eval ablation: on ``layers``, tokens with ``keep`` False contribute no attention or MLP output. Their
    K/V are still computed (other tokens attend to them as before), so with ``attn_res_kv_layers`` on those
    layers every other token's outputs are unchanged."""

    layers: tuple[int, ...]
    keep: Bool[Array, "B S"]


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
    entropy_head: Float[Array, " D"] | None  # cfg.entropy_skip_layers: predicts the next-token entropy
    entropy_head_bias: Float[Array, ""] | None
    skip_adapter: DenseMLP | None  # cfg.skip_adapter_dim: the skipped tokens' readout MLP
    skip_router: Float[Array, " D"] | None  # cfg.skip_router_layers: the skip router's score
    skip_router_bias: Float[Array, ""] | None
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
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
            attn_res_query_final=reshard(jnp.zeros((cfg.hidden_dim,), jnp.float32), P(None)) if cfg.attn_res else None,
            entropy_head=(
                reshard(jnp.zeros((cfg.hidden_dim,), jnp.float32), P(None)) if cfg.entropy_skip_layers else None
            ),
            entropy_head_bias=jnp.zeros((), jnp.float32) if cfg.entropy_skip_layers else None,
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
            config=cfg,
        )

    def layer_stacks(self) -> list[ArrayStacked[Block]]:
        """The block stacks; stack ``k`` holds layers ``stack_layer_indices()[k]``."""
        return [s for s in (self.stacked_blocks, self.kda_blocks) if s is not None]

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
        layer_skip: LayerSkip | None = None,
        extra_mlp_active: jax.Array | bool = True,
        pre_norm_grad_probe: Float[Array, "B S"] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """``extra_mlp_active`` (traced bool) turns on the entropy-head skip of ``entropy_skip_layers``.
        ``pre_norm_grad_probe`` (zeros ``[B, S]``) passes the stream entering ``final_norm`` through
        ``lm_head_grad_probe`` and adds each token's RMS of it as ``FINAL_NORM_INPUT_RMS_KEY``.
        ``layer_skip`` (an eval ablation, AttnRes only) drops the listed layers' attention and MLP outputs for
        the tokens it marks."""
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
        if cfg.attn_res:
            hidden, stacked_router_stats, final_gate_stats = self._attn_res_layers(
                hidden,
                token_ids,
                long_mask.with_fa4_bounds(long_lower_bounds, valid),
                long_mask.with_fa4_bounds(short_lower_bounds, valid),
                layer_skip,
                extra_mlp_active,
            )
        else:
            if layer_skip is not None:
                raise ValueError("layer_skip needs attn_res")
            # One compiled Block body scanned over the stacked layers; per-layer short/long is a
            # Bool[num_layers] scan input, and the FA4 metadata is selected per layer with jnp.where.
            mask_schedule = _long_layer_schedule(cfg.num_layers, cfg.global_every)

            def _scan_layer(
                carry_hidden: Float[Array, "B S D"], scan_inputs: tuple[Block, jax.Array]
            ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
                layer, layer_use_long_mask = scan_inputs
                use_long = jnp.asarray(layer_use_long_mask, dtype=jnp.bool_)
                lower_bounds = jnp.where(use_long, long_lower_bounds, short_lower_bounds)
                layer_mask = long_mask.with_fa4_bounds(lower_bounds, valid)
                return eqx.filter_checkpoint(layer, policy=None)(carry_hidden, layer_mask, use_long, use_long)

            hidden, stacked_router_stats = jax.lax.scan(
                _scan_layer, hidden, xs=(self.stacked_blocks.stacked, mask_schedule)
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
            router_metrics.update({k: v for k, v in stacked_router_stats.items() if k.startswith(_OUTSTAT_PREFIX)})
        router_metrics.update(final_gate_stats)
        if pre_norm_grad_probe is not None:
            rms = jnp.sqrt(jnp.mean(jnp.square(jax.lax.stop_gradient(hidden).astype(jnp.float32)), axis=-1))
            router_metrics[FINAL_NORM_INPUT_RMS_KEY] = rms
            hidden = lm_head_grad_probe(hidden, pre_norm_grad_probe, jnp.asarray(jnp.inf, jnp.float32))
        hidden = self.final_norm(hidden)
        if self.final_gated_norm is not None:
            hidden = self.final_gated_norm(hidden)
        return hidden, router_metrics

    def _attn_res_layers(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        long_layer_mask: AttentionMask,
        short_layer_mask: AttentionMask,
        layer_skip: LayerSkip | None,
        extra_mlp_active: jax.Array | bool,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], dict[str, jax.Array]]:
        """Block AttnRes over the layers, unrolled so each gate reads only its valid sources.

        The residual history is a Python tuple of completed block sums (the token embedding is the
        first, rolled in at layer 0) plus the running partial; each layer in ``attn_res_block_starts`` rolls its
        incoming partial into a new block. Completed blocks are immutable, so each block's logits
        against the queries of every gate that reads it (layer ``i``'s onwards and the final gate) are
        computed once when it is rolled, and each gate then only scores the partial. Blocks are shared
        by reference, so block memory is one copy per block. With ``AttnResLayerBackward.RECOMPUTE``
        each layer is rematerialized by ``_attn_res_layer_remat``, which saves only its inputs.

        Returns the final gate's mix, the per-layer router stats and the final gate's logs.
        """
        cfg = self.config
        assert self.attn_res_query_final is not None
        eps = cfg.layer_norm_eps
        num_layers = cfg.num_layers
        starts = attn_res_block_starts(cfg)
        layers = self.layers()
        # Every query in gate order: attn_0, mlp_0, ..., mlp_{L-1}; the KV gates (attn_res_kv_layers); final.
        gate_queries = []
        for layer in layers:
            assert layer.attn_res_query_attn is not None and layer.attn_res_query_mlp is not None
            gate_queries.append(jnp.stack([layer.attn_res_query_attn, layer.attn_res_query_mlp]))
        kv_queries = [layer.attn_res_query_kv[None] for layer in layers] if cfg.attn_res_kv_layers else []
        queries = jnp.concatenate([*gate_queries, *kv_queries, self.attn_res_query_final[None]])
        layer_fn = (
            _attn_res_layer_remat
            if cfg.attn_res_layer_backward == AttnResLayerBackward.RECOMPUTE
            else _attn_res_layer_passthrough
        )

        weight_logs: dict[int, tuple[jax.Array, bool]] = {}
        stats_out: list[dict[str, jax.Array]] = []
        z_out: list[jax.Array] = []
        extra_state: dict[str, jax.Array] = {}
        blocks: tuple[jax.Array, ...] = ()
        block_logits: tuple[tuple[jax.Array, ...], ...] = ()
        partial: jax.Array | None = hidden
        for i, layer in enumerate(layers):
            if i in starts:
                assert partial is not None
                blocks = (*blocks, partial)
                # Score the new block only against the gates that can read it (this layer's onwards).
                block_logits = (*block_logits, _logit_rows(_attn_res_source_logits(partial, queries[2 * i :], eps)))
                partial = None
            use_long = _is_long_layer(i, num_layers, cfg.global_every)
            partial_before = partial
            if cfg.skip_router_layers and i == min(cfg.skip_router_layers):
                extra_state["router_keep"], extra_state["router_frac"] = self._skip_router_keep(
                    _residual_stream(blocks, partial)
                )
            if cfg.entropy_skip_layers and i == min(cfg.entropy_skip_layers):
                extra_state["pred"], extra_state["keep"] = self._entropy_selection(
                    _residual_stream(blocks, partial), token_ids, extra_mlp_active
                )
            keeps = []
            if layer_skip is not None and i in layer_skip.layers:
                keeps.append(layer_skip.keep.astype(jnp.float32))
            if i in cfg.entropy_skip_layers:
                keeps.append((~extra_state["keep"]).astype(jnp.float32))
            if i in cfg.skip_router_layers:
                keeps.append(extra_state["router_keep"])
            layer_extras = {_SKIP_KEEP: functools.reduce(jnp.multiply, keeps)} if keeps else None
            partial, blocks, block_logits, stats = layer_fn(
                (layer, blocks, block_logits, partial, queries, layer_extras),
                long_layer_mask if use_long else short_layer_mask,
                token_ids,
                use_long,
                i,
                eps,
            )
            z_out.append(stats.pop(_ATTN_RES_Z))
            weight_logs[2 * i] = (stats.pop(_ATTN_RES_W_ATTN), partial_before is not None)
            weight_logs[2 * i + 1] = (stats.pop(_ATTN_RES_W_MLP), True)
            stats_out.append(stats)

        assert partial is not None
        final_index = queries.shape[0] - 1
        logits = [_block_logit(bl, queries, final_index) for bl in block_logits]
        logits.append(_attn_res_source_logits(partial, queries[final_index][None], eps)[0])
        weights, mixed = _softmax_mix(logits, [*blocks, partial])
        if cfg.attn_res_plain_sum:
            mixed = _stream_sum((*blocks, partial))
            weights = jnp.full_like(weights, 1.0 / weights.shape[0])
        weight_logs[final_index] = (jax.lax.stop_gradient(jnp.mean(weights, axis=tuple(range(1, weights.ndim)))), True)
        max_weight = jax.lax.stop_gradient(jnp.mean(jnp.max(weights, axis=0)))
        entropy = jax.lax.stop_gradient(jnp.mean(-jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-30)), axis=0)))
        z_final = _gate_z(logits)
        z_total = (sum(z_out) + z_final) / (len(z_out) + 1)

        if self.skip_adapter is not None and layer_skip is not None:
            adapted = self.skip_adapter(rms_norm(mixed, eps))
            skipped = (~layer_skip.keep)[..., None].astype(mixed.dtype)
            mixed = mixed + skipped * adapted.astype(mixed.dtype)
        final_stats = {"attn_res_final_max_weight": max_weight, "attn_res_final_entropy": entropy}
        if "pred" in extra_state:
            final_stats[_ENTROPY_PRED] = extra_state["pred"]
        if "router_frac" in extra_state:
            final_stats["attn_res_skip_router_frac"] = extra_state["router_frac"]
        # Source n is block n (the embedding is block 0); "p" is the running partial.
        for gate, (mean_weights, has_partial) in sorted(weight_logs.items()):
            num_blocks = mean_weights.shape[0] - int(has_partial)
            for n in range(num_blocks):
                final_stats[f"attn_res_w_g{gate:02d}_b{n:02d}"] = mean_weights[n]
            if has_partial:
                final_stats[f"attn_res_w_g{gate:02d}_p"] = mean_weights[-1]
        for i, layer in enumerate(layers):
            if isinstance(layer.attn, CausalSelfAttention) and layer.attn.qk_mult is not None:
                final_stats[f"attn_res_qk_mult_L{i}"] = jnp.mean(jax.lax.stop_gradient(layer.attn.qk_mult))
            if isinstance(layer.attn, KimiDeltaAttention):
                # The write-strength projection's magnitude: pinned under MuonH, free under Adam.
                w_beta = jax.lax.stop_gradient(layer.attn.w_beta).astype(jnp.float32)
                final_stats[f"attn_res_knob_kda_w_beta_norm_L{i}"] = jnp.linalg.norm(w_beta)
        hidden = reshard(mixed.astype(hidden.dtype), _batch_spec())
        final_stats["attn_res_z"] = jax.lax.stop_gradient(z_total)
        query_norms = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.square(queries.astype(jnp.float32)), axis=-1)))
        final_stats["attn_res_query_norm_mean"] = jnp.mean(query_norms)
        final_stats["attn_res_query_norm_max"] = jnp.max(query_norms)
        for g in range(queries.shape[0]):
            final_stats[f"attn_res_query_norm_g{g:02d}"] = query_norms[g]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *stats_out)
        return hidden, stacked, final_stats

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

    def _entropy_selection(
        self, stream: Float[Array, "B S D"], token_ids: Int[Array, "B S"], active: jax.Array | bool
    ) -> tuple[Float[Array, "B S"], Bool[Array, "B S"]]:
        """The entropy head's prediction (from the stop-gradient stream) and the token mask of ``entropy_skip_layers``:
        the ``extra_mlp_frac`` of the batch with the highest / lowest prediction, or a fixed pseudo-random subset."""
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
        train_terms: bool = False,
        layer_skip: LayerSkip | None = None,
        extra_mlp_active: jax.Array | bool = True,
        head_grad_probe: Float[Array, "B S"] | None = None,
        pre_norm_grad_probe: Float[Array, "B S"] | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        """``train_terms`` adds the training-only objectives (the entropy head's loss); evals leave it off so they
        score the plain next-token loss. ``head_grad_probe`` (zeros ``[B, S]``) passes the lm_head cross-entropy's
        input through ``lm_head_grad_probe``: its gradient is each token's input-gradient norm, and
        ``lm_head_grad_clip`` clips that gradient per token. ``pre_norm_grad_probe`` does the same at the stream
        entering ``final_norm`` (``Transformer.__call__``)."""
        hidden, router_metrics = self(
            token_ids,
            mask=mask,
            layer_skip=layer_skip,
            extra_mlp_active=extra_mlp_active,
            pre_norm_grad_probe=pre_norm_grad_probe,
        )
        final_norm_rms = router_metrics.pop(FINAL_NORM_INPUT_RMS_KEY, None)
        probe_metrics = {} if final_norm_rms is None else {FINAL_NORM_INPUT_RMS_KEY: final_norm_rms}
        entropy_pred = router_metrics.pop(_ENTROPY_PRED, None)
        outstats = {k: router_metrics.pop(k) for k in list(router_metrics) if k.startswith(_OUTSTAT_PREFIX)}
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        head_in = hidden
        if head_grad_probe is not None:
            clip = self.config.lm_head_grad_clip
            # The loss is a mean over loss tokens, so the raw per-token gradient is the per-token one / N.
            raw_clip = jnp.inf if clip is None else clip / jnp.maximum(jnp.sum(loss_weight.astype(jnp.float32)), 1.0)
            head_in = lm_head_grad_probe(hidden, head_grad_probe, jnp.asarray(raw_clip, jnp.float32))
        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            head_in,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            implementation="xla_fast_bwd",
            block_sizes=_CE_BLOCK_SIZES,
            logit_soft_cap=self.config.logit_soft_cap,
        )
        # Router z-loss is logged for monitoring only; it is not added to the training loss.
        loss = cross_entropy_loss
        entropy_metrics: dict[str, jax.Array] = {}
        if entropy_pred is not None and train_terms:
            target = _vocab_entropy(hidden, self.output_proj, self.config.logit_soft_cap)
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
        if not return_router_metrics:
            return loss
        final_gate_metrics = {
            f"train/attn_res/{name.removeprefix('attn_res_')}": router_metrics.pop(name)
            for name in list(router_metrics)
            if name.startswith("attn_res_")
        }
        if not router_metrics:
            # Dense model: no router to summarize.
            return loss, {
                "train/cross_entropy_loss": cross_entropy_loss,
                **probe_metrics,
                **final_gate_metrics,
                **entropy_metrics,
            }
        summarized_metrics = summarize_router_metrics(router_metrics)
        summarized_metrics.update(final_gate_metrics)
        summarized_metrics.update(entropy_metrics)
        summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
        summarized_metrics.update(probe_metrics)
        for name, per_layer in outstats.items():
            for i in range(per_layer.shape[0]):
                summarized_metrics[f"train/aux/{name}_L{i}"] = per_layer[i]
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        summarized_metrics["train/router/z_loss_logging_only"] = (
            jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        )
        # Keep per-layer int32 counts; the trainer sums over layers in host int64 (jnp.sum here overflows int32).
        summarized_metrics["moe/dropped_assignments"] = router_metrics["capacity_overflow_per_layer"]
        summarized_metrics["moe/sender_dropped_assignments"] = router_metrics["sender_capacity_overflow_per_layer"]
        summarized_metrics["moe/receiver_dropped_assignments"] = router_metrics["receiver_capacity_overflow_per_layer"]
        return loss, summarized_metrics


@jax.custom_vjp
def lm_head_grad_probe(
    h: Float[Array, "B S D"], probe: Float[Array, "B S"], raw_clip: Float[Array, ""]
) -> Float[Array, "B S D"]:
    """Identity on ``h``. Backward returns each token's cotangent L2 norm (float32, over the hidden dim, before
    clipping) as ``probe``'s cotangent, so differentiating w.r.t. a zeros ``probe`` reads the per-token gradient
    norms entering the lm_head with no host callback, and passes ``h``'s cotangent on clipped per token to norm
    ``raw_clip`` (``g * min(1, raw_clip / ||g||)``; inf: unchanged)."""
    del probe, raw_clip
    return h


def _lm_head_grad_probe_fwd(h, probe, raw_clip):
    del probe
    return h, raw_clip


def _lm_head_grad_probe_bwd(raw_clip, g):
    norm = jnp.sqrt(jnp.sum(jnp.square(g.astype(jnp.float32)), axis=-1))
    scale = jnp.minimum(1.0, raw_clip / norm)  # inf / 0 and c / 0 are inf: zero-gradient tokens keep scale 1
    return (g * scale[..., None]).astype(g.dtype), norm, jnp.zeros_like(raw_clip)


lm_head_grad_probe.defvjp(_lm_head_grad_probe_fwd, _lm_head_grad_probe_bwd)


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


# Lookup tables: a token reads one row, so they cost no matmul FLOPs.
_EMBEDDING_TABLES = frozenset({"token_embed", "value_embed"})
# Per-head or per-expert vectors stored as 2-D arrays that are added, not multiplied.
_NON_MATMUL_PARAMS = frozenset({"dt_bias", "router_bias", "a_log"})


def forward_flops_per_token(model: Transformer) -> float:
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
            long = _is_long_layer(i, cfg.num_layers, cfg.global_every)
            span = seq if long or window is None else min(window, seq)
            # Mean keys per query under a causal mask capped at ``span``: sum_p min(p + 1, span) / seq.
            keys = (span * (span + 1) / 2 + (seq - span) * span) / seq
            core += 2 * keys * q_width * 2  # QK^T over d_qk and PV over d_v, both q_width = heads * head_dim
    if model.kda_blocks is not None:
        attn = model.kda_blocks.stacked.attn
        heads = attn.a_log.shape[-1]
        d_k, d_v = attn.w_q.shape[-1] // heads, attn.w_v.shape[-1] // heads
        core += model.kda_blocks.num_layers * heads * (6 * d_k * d_v + 4 * cfg.kda_chunk_size * (d_k + d_v))
    return matmul + core
