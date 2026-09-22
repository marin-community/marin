# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-parallel MoE grug variant model."""

import dataclasses
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
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

_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5
_QB_HIST_BINS = 10_000
_CE_TOKENS_PER_RANK = 65_536
_CE_BLOCK_SIZES = BlockSizes(b_block_size=_CE_TOKENS_PER_RANK, v_block_size=4096)
# Axes the non-expert params FSDP-shard over.
_FSDP_AXES: tuple[str, ...] = ("data", "expert")
_LM_HEAD_PARTITION_SPEC = P(_FSDP_AXES, "model")


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


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


def _embedding_gather(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """Look up tokens from a replicated table without a cross-rack collective."""

    def _local(table: jax.Array, ids: jax.Array) -> jax.Array:
        return table[ids]

    token_ids = reshard(token_ids, P(_BATCH_AXES, None))
    return shard_map(
        _local,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(_BATCH_AXES, None)),
        out_specs=P(_BATCH_AXES, None, None),
    )(token_embed, token_ids)


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
    capacity_factor: float = 1.15
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    sconv: bool = True
    sconv_kernel: int = 4
    sconv_sites: tuple[str, ...] = ("k", "attn", "mlp")
    pooled_transport_capacity_factor: float | None = 1.15
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    # Dense (no-MoE) mode: every block is a single DenseMLP(hidden, intermediate_dim) SwiGLU; the MoE fields are ignored.
    dense_mlp: bool = False
    # Omni-neurons (dense only): each layer's MLP reads the concatenation of every prior sublayer snapshot
    # (embed, and per prior layer: attn_out / resid_post_attn / mlp_out / resid_post_mlp) plus this layer's
    # attn_out / resid_post_attn, so its input width grows with depth (d*(3+4*layer)). Attention is unchanged.
    omni_mlp: bool = False
    # Omni per-component norm: give each concatenated component its OWN learnable RMS gain (a (C, d) matrix
    # per layer) instead of one shared width-d gain, so the model can up/down-weight whole components. The
    # rsqrt normalization is per-component either way; this only splits the learnable gain.
    omni_component_norm: bool = False

    def __post_init__(self) -> None:
        if self.omni_mlp and not self.dense_mlp:
            raise ValueError("omni_mlp requires dense_mlp=True (omni-neurons is a dense-model variant)")
        if self.omni_component_norm and not self.omni_mlp:
            raise ValueError("omni_component_norm requires omni_mlp=True")
        if not self.dense_mlp and self.num_experts_per_token >= self.num_experts:
            # QB routing takes top-(k+1) and keeps the last entry as the threshold alpha, so a
            # full-bank top-k asks `jax.lax.top_k` for more entries than the router has experts.
            raise ValueError("num_experts_per_token must be < num_experts, because QB routing selects top-(k+1)")

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


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    sconv_k: "ShortConv | None"  # SConv after the K projection (cfg.sconv)
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.stored_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P(_FSDP_AXES, "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P(_FSDP_AXES, "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P(_FSDP_AXES, "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", _FSDP_AXES)),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            sconv_k=(ShortConv.init(m * h, cfg.sconv_kernel) if cfg.sconv and "k" in cfg.sconv_sites else None),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q_flat = jnp.einsum("bsh,hd->bsd", x, self.w_q)
        k_flat = jnp.einsum("bsh,hd->bsd", x, self.w_k)
        v_flat = jnp.einsum("bsh,hd->bsd", x, self.w_v)
        # SConv: depthwise causal conv after the K projection. segment_ids (packed-document
        # boundaries) come from the mask so the conv never mixes across a document boundary.
        _seg = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = _seg[0] if _seg is not None else None
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

        if isinstance(disable_rope, bool):
            if not disable_rope:
                q, k = _rope(q, k)
        else:
            q_roped, k_roped = _rope(q, k)
            keep = ~jnp.asarray(disable_rope, dtype=jnp.bool_)
            q = jnp.where(keep, q_roped, q)
            k = jnp.where(keep, k_roped, k)
        q = q * self.cfg.qk_mult
        # The fa4-cute kernel is GPU-only; fall back to auto-select off-GPU so the model still lowers
        # on CPU (e.g. the grug variant-contract tests).
        attn_impl = "gpu_fa4_cute" if jax.default_backend() == "gpu" else None
        attn_out = attention(q, k, v, mask, implementation=attn_impl)
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
    def init(
        hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray, out_dim: int | None = None
    ) -> "DenseMLP":
        # ``hidden_dim`` is the input width; ``out_dim`` (default ``hidden_dim``) the output width. They differ
        # only for omni-neurons, where the MLP reads a wide concatenation but still writes a d-wide residual delta.
        out_dim = hidden_dim if out_dim is None else out_dim
        k_gate, k_up, k_down = random.split(key, 3)
        return DenseMLP(
            w_gate=reshard(
                _init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P(_FSDP_AXES, "model")
            ),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P(_FSDP_AXES, "model")),
            w_down=reshard(_init_weight(k_down, (intermediate_dim, out_dim), initializer_std), P("model", _FSDP_AXES)),
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

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = jax.nn.sigmoid(unbiased_topk)
        # Renormalize K combine weights to sum to ``_ROUTING_RENORM_SUM`` (baked in).
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
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
            routed_input = jnp.einsum(
                "td,dl->tl",
                x_flat,
                self.w_latent_down.astype(x_flat.dtype),
                out_sharding=_batch_spec(),
            )
            # Keep the expert input scale independent of the down-projection initialization.
            routed_input = self.latent_norm(routed_input)
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


def _omni_mlp_in_dim(hidden_dim: int, layer_idx: int) -> int:
    """Concatenated snapshot width feeding layer ``layer_idx``'s MLP under omni-neurons.

    Before layer n's MLP the snapshot list is: embed (1) + {attn_out, resid_post_attn, mlp_out,
    resid_post_mlp} for each of the n prior layers (4n) + this layer's attn_out and resid_post_attn (2),
    each ``hidden_dim`` wide -- so ``hidden_dim * (4 * layer_idx + 3)``.
    """
    return hidden_dim * (4 * layer_idx + 3)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: "MoEMLP | DenseMLP"
    shared: tuple[DenseMLP, ...] | None
    sconv_attn: "ShortConv | None"
    sconv_mlp: "ShortConv | None"
    # Omni per-component RMS gain (C, d): one learnable gain row per concatenated component. None unless
    # omni_mlp + omni_component_norm (then the shared rms_mlp gain is bypassed for the MLP input).
    omni_component_gain: jax.Array | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray, mlp_in_dim: int | None = None) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        if cfg.dense_mlp:
            # Dense block: one SwiGLU DenseMLP(hidden, intermediate_dim), no MoE and no shared experts.
            # Omni-neurons widens only the MLP input: mlp_in_dim (the concatenated snapshot width) sizes the
            # MLP GatedNorm and the DenseMLP; rms_mlp stays width d (applied to each d-dim snapshot before concat).
            mlp_dim = mlp_in_dim if mlp_in_dim is not None else cfg.hidden_dim
            # One learnable RMS gain row per concatenated component (C = mlp_dim // d), when requested.
            omni_component_gain = None
            if mlp_in_dim is not None and cfg.omni_component_norm:
                omni_component_gain = reshard(
                    jnp.ones((mlp_dim // cfg.hidden_dim, cfg.hidden_dim), dtype=jnp.float32), P(None, None)
                )
            return Block(
                rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
                attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
                attn=CausalSelfAttention.init(cfg, key=attn_key),
                rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
                mlp_gated_norm=GatedNorm.init(mlp_dim, cfg.initializer_std, key=gn_mlp_key),
                mlp=DenseMLP.init(
                    mlp_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key, out_dim=cfg.hidden_dim
                ),
                shared=None,
                sconv_attn=(
                    ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "attn" in cfg.sconv_sites else None
                ),
                sconv_mlp=(
                    ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None
                ),
                omni_component_gain=omni_component_gain,
            )
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
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
            sconv_attn=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "attn" in cfg.sconv_sites else None
            ),
            sconv_mlp=(
                ShortConv.init(cfg.hidden_dim, cfg.sconv_kernel) if cfg.sconv and "mlp" in cfg.sconv_sites else None
            ),
            omni_component_gain=None,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool | jax.Array = False,
        is_global: bool | jax.Array = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        # segment_ids (packed-document boundaries) for the branch-output SConvs; None when unpacked.
        _seg = mask.segment_ids if isinstance(mask, AttentionMask) else None
        sconv_segment_ids = _seg[0] if _seg is not None else None

        attn_in = self.attn_gated_norm(self.rms_attn(x))
        attn_out = self.attn(attn_in, mask, disable_rope=disable_rope, is_global=is_global)
        if self.sconv_attn is not None:
            attn_out = self.sconv_attn(attn_out, sconv_segment_ids)
        x = x + attn_out
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        if isinstance(self.mlp, DenseMLP):
            mlp_out = self.mlp(mlp_in, moe_output_reshard=False)
            router_stats: dict[str, jax.Array] = {}
        else:
            mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            for shared_expert in self.shared:
                mlp_out = mlp_out + shared_expert(mlp_in, activation=ActivationFunctionEnum.silu)
        if self.sconv_mlp is not None:
            mlp_out = self.sconv_mlp(mlp_out, sconv_segment_ids)
        x = x + mlp_out
        return x, router_stats


def _long_layer_schedule(num_layers: int, global_every: int) -> jax.Array:
    # Every global_every-th layer is full-causal, and the last layer always is, so a depth that is
    # not a multiple of global_every still ends on a global-context layer.
    layer_indices = jnp.arange(num_layers)
    return (((layer_indices + 1) % global_every) == 0) | (layer_indices == num_layers - 1)


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[Block] | None
    # Omni-neurons: heterogeneous per-layer blocks (MLP input width grows with depth) can't be scanned,
    # so they are held unrolled here; exactly one of stacked_blocks / omni_blocks is set.
    omni_blocks: tuple[Block, ...] | None
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
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
        # The embedding is fully replicated for a local lookup.
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), P(None, None)
        )
        output_proj = reshard(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), _LM_HEAD_PARTITION_SPEC
        )
        if cfg.omni_mlp:
            # Per-layer MLP input widths differ, so blocks can't be stacked/scanned -- build them unrolled.
            omni_blocks = tuple(
                Block.init(cfg, key=k, mlp_in_dim=_omni_mlp_in_dim(cfg.hidden_dim, i)) for i, k in enumerate(block_keys)
            )
            stacked_blocks = None
        else:
            omni_blocks = None
            stacked_blocks = ArrayStacked.init(cfg.num_layers, Block)(cfg, key=jnp.stack(block_keys))
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            stacked_blocks=stacked_blocks,
            omni_blocks=omni_blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        hidden = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Local layers use a sliding window; every global_every-th layer is full causal.
        segment_ids = None
        if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
            # Pin the [B, S] segment ids batch-sharded and reuse one array for both attention sides.
            q_segment_ids, _ = mask.segment_ids
            q_segment_ids = _batch_reshard(q_segment_ids)
            segment_ids = (q_segment_ids, q_segment_ids)
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        # One compiled Block body scanned over the stacked layers; per-layer short/long is a Bool[num_layers] scan input.
        mask_schedule = _long_layer_schedule(cfg.num_layers, cfg.global_every)
        # Precompute FA4 per-token metadata for long/short layers outside the scan; select per layer with jnp.where.
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

        if cfg.omni_mlp:
            # Omni-neurons: unrolled layers, each MLP reading the concatenation of every prior snapshot.
            router_metrics: dict[str, jax.Array] = {}
            assert self.omni_blocks is not None
            sconv_seg = segment_ids[0] if segment_ids is not None else None
            schedule = [
                bool(((i + 1) % cfg.global_every == 0) or (i == cfg.num_layers - 1)) for i in range(cfg.num_layers)
            ]
            x = hidden
            snapshots: list[jax.Array] = [x]  # "embed": the initial (post-embed-norm) residual stream
            for layer_idx, block in enumerate(self.omni_blocks):
                use_long = schedule[layer_idx]
                lower_bounds = long_lower_bounds if use_long else short_lower_bounds
                layer_mask = long_mask.with_fa4_bounds(lower_bounds, valid)

                def _omni_layer(
                    blk: Block,
                    x_in: jax.Array,
                    snaps: tuple[jax.Array, ...],
                    _mask: AttentionMask = layer_mask,
                    _long: bool = use_long,
                ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                    attn_in = blk.attn_gated_norm(blk.rms_attn(x_in))
                    attn_out = blk.attn(attn_in, _mask, disable_rope=_long, is_global=_long)
                    if blk.sconv_attn is not None:
                        attn_out = blk.sconv_attn(attn_out, sconv_seg)
                    x_post_attn = x_in + attn_out
                    # Per-snapshot RMSNorm keeps snapshots magnitude-comparable, then concat; the wider
                    # mlp_gated_norm / DenseMLP then read every snapshot at once. With omni_component_gain,
                    # each component gets its own learnable gain row (rsqrt norm then per-component gain);
                    # otherwise the shared rms_mlp gain applies to every component.
                    components = (*snaps, attn_out, x_post_attn)
                    if blk.omni_component_gain is not None:
                        normed = [
                            rms_norm(s, blk.rms_mlp.eps) * blk.omni_component_gain[c] for c, s in enumerate(components)
                        ]
                    else:
                        normed = [blk.rms_mlp(s) for s in components]
                    mlp_in = blk.mlp_gated_norm(jnp.concatenate(normed, axis=-1))
                    assert isinstance(blk.mlp, DenseMLP)  # omni is dense-only
                    mlp_out = blk.mlp(mlp_in, moe_output_reshard=False)
                    if blk.sconv_mlp is not None:
                        mlp_out = blk.sconv_mlp(mlp_out, sconv_seg)
                    x_post_mlp = x_post_attn + mlp_out
                    return x_post_mlp, attn_out, x_post_attn, mlp_out

                x, attn_out, x_post_attn, mlp_out = eqx.filter_checkpoint(_omni_layer)(block, x, tuple(snapshots))
                # Append this layer's four snapshots for the next layers to read.
                snapshots.extend([attn_out, x_post_attn, mlp_out, x])
            hidden = x
            hidden = self.final_gated_norm(self.final_norm(hidden))
            return hidden, router_metrics

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

        assert self.stacked_blocks is not None
        hidden, stacked_router_stats = jax.lax.scan(
            _scan_layers, hidden, xs=(self.stacked_blocks.stacked, mask_schedule)
        )
        if cfg.dense_mlp:
            router_metrics = {}
        else:
            # One cross-device reduction for the whole layer stack, not one per scan iteration (see router_metrics).
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
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

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
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            implementation="xla_fast_bwd",
            block_sizes=_CE_BLOCK_SIZES,
        )
        # Router z-loss is logged for monitoring only; it is not added to the training loss.
        loss = cross_entropy_loss
        if return_router_metrics:
            if not router_metrics:
                # Dense model: no router to summarize.
                return loss, {"train/cross_entropy_loss": cross_entropy_loss}
            summarized_metrics = summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
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
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "ShortConv",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
