# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
)
from levanter.grug.grug_moe import (
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/moe requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


RematMode = Literal["recompute_all", "save_moe"]


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window // 2), mask.with_sliding_window(sliding_window)


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer.

    Architecture choices (GatedNorm, XSA, QB routing) are hardcoded.
    Only shape/size knobs live here. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    router_z_loss_coef: float = 0.001
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    num_prelude_layers: int = 0
    """Re-entrant structure: leading unique (non-looped) layers. 0 = no prelude."""
    num_coda_layers: int = 0
    """Re-entrant structure: trailing unique (non-looped) layers. 0 = no coda."""
    recurrence_steps: int = 1
    """Times the shared core stack (layers between prelude and coda) is applied per
    forward. 1 = plain stack (no recurrence); >1 = weight-tied re-entrant looping.
    Effective depth = num_prelude_layers + num_core_layers * recurrence_steps + num_coda_layers."""
    iteration_film: bool = False
    """When True and recurrence_steps > 1, modulate the shared core block per loop
    iteration with a learned FiLM (adaLN-style per-feature scale+shift) indexed by
    the iteration step and core-layer index. Gives one weight-tied block a
    coarse-to-fine schedule at ~free parameter cost. Initialized to identity, so at
    step 0 the model is numerically identical to the iteration_film=False variant."""
    randomize_recurrence: bool = False
    """E3: when True, training samples the core loop count per step from
    recurrence_choices (the model forward is called with a per-step recurrence
    override). Trains one weight-tied core to be correct at many depths, enabling
    test-time depth scaling. recurrence_steps stays the default/eval loop count."""
    recurrence_choices: tuple[int, ...] = ()
    """E3: the set of loop counts to sample among when randomize_recurrence is True
    (e.g. (2, 4, 8)). Empty unless randomize_recurrence."""
    core_consistency_weight: float = 0.0
    """E5: weight on the core-consistency penalty (mean normalized squared delta
    between consecutive core-loop hidden states). 0 disables it (E0-E3 unchanged).
    >0 pulls the weight-tied core toward a contractive/fixed-point map so extra
    test-time loops stop drifting."""
    depth_conditioned_routing: bool = False
    """E6: when True and the core is looped, the MoE router gets a learned
    per-(iteration, core-layer) additive logit bias, so each core traversal
    activates a DIFFERENT expert mixture (depth-conditioned routing). This makes
    the weight-tied loop a genuinely depth-varying map (f_t != f) instead of one
    fixed residual map re-applied in place -- the structural failure E0-E5 hit.
    Sized to max_trained_recurrence; eval at deeper R reuses the last entry.
    Zero-init, so step 0 is numerically identical to the routing=False variant."""
    anytime_supervision: bool = False
    """E4: when True and the core is looped, training adds a deep-supervision CE
    term read off the shared output head (final_norm + output_proj) after EACH
    core iteration, averaged over iterations and weighted by
    anytime_supervision_weight. Rewards monotonically-USEFUL refinement at every
    depth (the fix for E5's freeze, which suppressed refinement) and makes every
    depth anytime-decodable. Training-only: the eval (reduction='none') path and
    the parameter tree are unchanged (no new params)."""
    anytime_supervision_weight: float = 1.0
    """E4: relative weight on the averaged per-iteration deep-supervision CE term.
    Only used when anytime_supervision is True."""
    ponder_halting: bool = False
    """E7 (PonderNet): when True and the core is looped, a learned per-token halting
    head reads each per-iteration readout to predict a halting probability; training
    adds an expected-over-halting CE term plus a KL-to-geometric-prior regularizer.
    This is ADDED to the standard final CE (not a replacement), so the eval path and
    the comparable paloma metric are unchanged. Training-only; adds one (hidden_dim,)
    halt-head vector (routes to plain Adam). Builds on the same per-iteration readout
    plumbing as anytime_supervision (E4)."""
    ponder_loss_weight: float = 1.0
    """E7: weight on the expected-over-halting reconstruction CE term."""
    ponder_kl_weight: float = 0.01
    """E7: weight (beta) on the KL(halting-dist || geometric-prior) regularizer."""
    ponder_prior_lambda: float = 0.2
    """E7: geometric-prior halting rate (expected halting step ~ 1/lambda)."""
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_prelude_layers < 0 or self.num_coda_layers < 0:
            raise ValueError("num_prelude_layers and num_coda_layers must be non-negative")
        if self.num_prelude_layers + self.num_coda_layers > self.num_layers:
            raise ValueError("num_prelude_layers + num_coda_layers must be <= num_layers")
        if self.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        if self.recurrence_steps > 1 and self.num_core_layers <= 0:
            raise ValueError("recurrence_steps > 1 requires at least one core layer")
        if self.randomize_recurrence:
            if not self.recurrence_choices:
                raise ValueError("randomize_recurrence requires a non-empty recurrence_choices")
            if any(choice < 1 for choice in self.recurrence_choices):
                raise ValueError("recurrence_choices must all be >= 1")
            if self.num_core_layers < 1:
                raise ValueError("randomize_recurrence requires at least one core layer")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        resolve_moe_implementation(self.moe_implementation)

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
    def num_core_layers(self) -> int:
        """Unique layers in the shared, looped core (everything between prelude and coda)."""
        return self.num_layers - self.num_prelude_layers - self.num_coda_layers

    @property
    def effective_depth(self) -> int:
        """Number of block applications per forward (compute depth)."""
        return self.num_prelude_layers + self.num_core_layers * self.recurrence_steps + self.num_coda_layers

    @property
    def max_trained_recurrence(self) -> int:
        """Deepest core-loop count the model is trained at.

        Sizes the per-depth tables (FiLM, depth-conditioned router bias). With
        randomized recurrence this is max(recurrence_choices); otherwise the
        static recurrence_steps. Eval at a deeper R reuses the last table entry.
        """
        if self.randomize_recurrence and self.recurrence_choices:
            return max(self.recurrence_choices)
        return self.recurrence_steps


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Non-parametric RMS norm over the last dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q = rms_norm(q)
        k = rms_norm(k)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = reshard(aligned_v, P(_BATCH_AXES, None, "model", None))
        # Exclusive Self Attention: subtract the component of yᵢ parallel to vᵢ.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: sigmoid(x @ attn_gate) produces one scalar per head.
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
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
        # TODO: silu activation here isn't explored, just cargo-culted from Qwen. Likely low-hanging ablation fruit
        # (e.g. compare no activation, relu, etc.).
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
            w_gate=reshard(_init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_down=reshard(_init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std), P("model", "data")),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        *,
        activation: MoeActivation = ActivationFunctionEnum.silu,
    ) -> Float[Array, "B S D"]:
        if isinstance(activation, ActivationFunctionEnum):
            activation_fn = activation.to_jax_fn()
        else:
            activation_fn = activation

        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
        # Reshard after the reshape so the shared-expert output carries the same
        # canonical batch sharding as the routed MoE output (MoEMLP reshards its
        # routed result identically). Splitting the fused
        # ("replica_dcn", "data", "expert") token axis back into (b, s) otherwise
        # leaks the `expert` mesh axis onto the seq dim, so the shared+routed
        # residual add fails with a ShardingTypeError on a multi-node mesh.
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


def _routing_stats(
    selected_experts: Int[Array, "T K"],
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> dict[str, jax.Array]:
    router_probs_f = router_probs.astype(jnp.float32)
    router_logits_f = router_logits.astype(jnp.float32)
    expert_counts = jnp.sum(jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32), axis=(0, 1))
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))
    token_fraction = assignment_fraction * num_experts_per_token
    p = jnp.mean(router_probs_f, axis=0)
    load_balancing_loss = num_experts * jnp.sum(token_fraction * p)
    z = jsp.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.mean(z**2)

    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "qb_beta_per_layer": router_metrics.get("qb_beta_per_layer"),
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
    return out


def _histogram_from_expert_counts(expert_counts: jax.Array) -> SummaryStats:
    counts = jnp.asarray(expert_counts, dtype=jnp.float32)
    num_experts = counts.shape[0]
    expert_ids = jnp.arange(num_experts, dtype=jnp.float32)
    num = jnp.sum(counts)
    sum_values = jnp.sum(counts * expert_ids)
    sum_squares = jnp.sum(counts * expert_ids * expert_ids)
    nonzero = counts > 0
    min_value = jnp.where(nonzero, expert_ids, jnp.inf).min()
    max_value = jnp.where(nonzero, expert_ids, -jnp.inf).max()
    min_value = jnp.where(num > 0, min_value, 0.0)
    max_value = jnp.where(num > 0, max_value, 0.0)
    bucket_limits = jnp.arange(num_experts + 1, dtype=jnp.float32)
    histogram = Histogram(bucket_limits=bucket_limits, bucket_counts=counts)
    return SummaryStats.from_reduced_values(
        min=min_value,
        max=max_value,
        num=num,
        nonzero_count=jnp.sum(nonzero),
        sum=sum_values,
        sum_squares=sum_squares,
        histogram=histogram,
    )


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert_mlp = random.split(key, 2)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim

        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=e,
                hidden_dim=d,
                intermediate_dim=i,
                initializer_std=cfg.initializer_std,
                key=k_expert_mlp,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        router_logit_bias: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        # `router_logit_bias` (E6) is an optional learned per-expert (E,) additive bias,
        # added to the router logits with gradient so it shapes BOTH expert
        # selection and the sigmoid combine weights. Distinct from `router_bias`,
        # the stop-gradient QB load-balancing threshold. None reproduces the
        # depth_conditioned_routing=False path exactly (E0-E5 unchanged).
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        if router_logit_bias is not None:
            router_logits = router_logits + router_logit_bias.astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )
        # Sharded QB: compute beta locally per device, then average.
        mesh = get_abstract_mesh()
        s_minus_alpha = reshard(router_logits - qb_alpha, P(_BATCH_AXES, None))
        num_devices = 1
        for a in _BATCH_AXES:
            num_devices *= mesh.shape[a]
        local_tokens = s_minus_alpha.shape[0] // num_devices
        qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

        def _local_qb_beta(s_ma):
            topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
            beta = topk_vals[:, -1]
            return jax.lax.pmean(beta, axis_name=_BATCH_AXES)

        router_stats["qb_beta"] = shard_map(
            _local_qb_beta,
            mesh=mesh,
            in_specs=(P(_BATCH_AXES, None),),
            out_specs=P(),
        )(s_minus_alpha)

        routed_flat = self.expert_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
        )

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        film: tuple[jax.Array, jax.Array] | None = None,
        router_logit_bias: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        # `film` is an optional (scale, shift) pair, each shape (D,), broadcast over
        # (B, S, D). The same modulation is applied after both gated-norms so a
        # weight-tied core block can be told which loop iteration it is on (E2).
        # `router_logit_bias` (E6) is an optional per-expert (E,) additive bias on
        # the MoE router logits for depth-conditioned expert selection.
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        if film is not None:
            scale, shift = film
            attn_in = attn_in * (1.0 + scale) + shift
        x = _batch_reshard(x + self.attn(attn_in, mask))
        mlp_in = _batch_reshard(self.mlp_gated_norm(self.rms_mlp(x)))
        if film is not None:
            mlp_in = mlp_in * (1.0 + scale) + shift
        mlp_out, router_stats = self.mlp(mlp_in, router_logit_bias=router_logit_bias)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


def _mean_router_stats(stats_per_iter: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    """Mean-aggregate a shared core block's per-iteration router stats into one entry.

    A weight-tied core block is applied ``recurrence_steps`` times per forward but has a
    single set of router params; collapsing its per-iteration stats to one keeps
    ``qb_beta_per_layer`` (and the rest of ``router_metrics``) 1:1 with the unique blocks
    that ``_apply_qb_betas`` nudges. With a single iteration this is the identity.
    """
    if len(stats_per_iter) == 1:
        return stats_per_iter[0]
    return {k: jnp.mean(jnp.stack([s[k] for s in stats_per_iter], axis=0), axis=0) for k in stats_per_iter[0]}


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    core_film_scale: jax.Array | None
    core_film_shift: jax.Array | None
    core_router_bias: jax.Array | None
    halt_head: jax.Array | None
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        # Per-iteration FiLM tables for the shared core block. Zeros => identity
        # (Block applies x*(1+scale)+shift), so consuming no PRNG key keeps every
        # other param bit-identical to the iteration_film=False model. Replicated
        # like RMSNorm weights: tiny per-feature params, no sharding.
        core_film_scale = None
        core_film_shift = None
        if cfg.iteration_film and cfg.recurrence_steps > 1:
            film_shape = (cfg.recurrence_steps, cfg.num_core_layers, cfg.hidden_dim)
            core_film_scale = reshard(jnp.zeros(film_shape, dtype=jnp.float32), P(None, None, None))
            core_film_shift = reshard(jnp.zeros(film_shape, dtype=jnp.float32), P(None, None, None))
        # E6: per-(iteration, core-layer, expert) additive router-logit bias for
        # depth-conditioned routing. Zeros => identity at init and consumes no PRNG
        # key, so every other param stays bit-identical to the routing=False model.
        # Sized to the deepest trained core depth; the matching name substring
        # "router_bias" routes it to plain Adam (not AdamH), so the zero init can't
        # trigger the AdamH 0/0 NaN that bit E2's FiLM.
        core_router_bias = None
        if cfg.depth_conditioned_routing and cfg.max_trained_recurrence > 1:
            bias_shape = (cfg.max_trained_recurrence, cfg.num_core_layers, cfg.num_experts)
            core_router_bias = reshard(jnp.zeros(bias_shape, dtype=jnp.float32), P(None, None, None))
        # E7 (PonderNet): per-token halting head read off each per-iteration readout.
        # Zero-init and consumes no PRNG key, so every other param stays bit-identical
        # to the ponder_halting=False model. Shape (hidden_dim,) so ndim==1 routes it
        # to plain Adam (AdamH divides by the param norm -> 0/0 NaN on a zero init).
        halt_head = None
        if cfg.ponder_halting and cfg.max_trained_recurrence > 1:
            halt_head = reshard(jnp.zeros((cfg.hidden_dim,), dtype=jnp.float32), P(None))
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            core_film_scale=core_film_scale,
            core_film_shift=core_film_shift,
            core_router_bias=core_router_bias,
            halt_head=halt_head,
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        recurrence_steps: int | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        if not isinstance(mask, AttentionMask):
            mask = AttentionMask.causal()
        short_mask, long_mask = _layer_attention_masks(mask, sliding_window=cfg.sliding_window)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        # Re-entrant structure: prelude blocks (once) -> shared core stack (looped
        # recurrence_steps times, weight-tied) -> coda blocks (once). With the
        # default config (prelude=coda=0, recurrence_steps=1) this is exactly the
        # plain stack over self.blocks. `self.blocks` stays a flat tuple of the
        # unique blocks, so per-block router stats (and the QB bias update in
        # _apply_qb_betas) remain 1:1 with the unique blocks: a looped core block's
        # per-iteration stats are mean-aggregated back to a single entry below.
        prelude = cfg.num_prelude_layers
        core = cfg.num_core_layers
        prelude_blocks = self.blocks[:prelude]
        core_blocks = self.blocks[prelude : prelude + core]
        coda_blocks = self.blocks[prelude + core :]

        # `eff_idx` counts block applications (effective depth) so the every-4th-layer
        # long-attention pattern is preserved across the unrolled core.
        eff_idx = 0

        def apply_block(
            block: "Block",
            h: jax.Array,
            idx: int,
            film: tuple[jax.Array, jax.Array] | None = None,
            router_logit_bias: jax.Array | None = None,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            layer_mask = long_mask if idx % 4 == 3 else short_mask
            return eqx.filter_checkpoint(block, policy=remat_policy)(h, layer_mask, film, router_logit_bias)

        per_block_stats: list[dict[str, jax.Array]] = []
        for block in prelude_blocks:
            hidden, stats = apply_block(block, hidden, eff_idx)
            per_block_stats.append(stats)
            eff_idx += 1

        # E3: a per-call override of the core loop count (randomized depth during
        # training; deeper-than-trained eval at test time). None reproduces the
        # config default exactly, so E0/E1/E2 are unchanged.
        n_recurrence = recurrence_steps if recurrence_steps is not None else cfg.recurrence_steps
        core_iter_stats: list[list[dict[str, jax.Array]]] = [[] for _ in core_blocks]
        # E5: accumulate the per-iteration normalized squared delta between
        # consecutive core-loop states. Gated on the static config so E0-E3 trace
        # to the identical jaxpr (no penalty op, no extra metric key).
        track_consistency = cfg.core_consistency_weight > 0
        consistency_accum = jnp.zeros((), dtype=jnp.float32)
        x_prev = hidden
        # E4: collect a readout (shared head: final_norm + final_gated_norm) after
        # each core iteration so next_token_loss can deep-supervise every depth.
        # Gated on the static config so E0/E1/E2/E3/E5/E6 trace to the identical
        # jaxpr (no readout op, no extra metric key). The intermediates skip the
        # coda (cheap), so they regularize the core to be decodable at every depth;
        # the final output still flows through the coda below.
        track_anytime = cfg.anytime_supervision and cfg.max_trained_recurrence > 1
        # E7 (PonderNet): collect a per-iteration halting logit (readout . halt_head)
        # alongside the readouts. Needs the same per-iteration readout plumbing as E4.
        track_ponder = cfg.ponder_halting and cfg.max_trained_recurrence > 1
        collect_readouts = track_anytime or track_ponder
        anytime_readouts: list[jax.Array] = []
        ponder_halt_logits: list[jax.Array] = []
        for t in range(n_recurrence):
            for c, block in enumerate(core_blocks):
                film = None
                if self.core_film_scale is not None:
                    # The FiLM tables are sized for cfg.recurrence_steps; for an
                    # override deeper than trained, reuse the last iteration's FiLM.
                    film_t = min(t, cfg.recurrence_steps - 1)
                    film = (self.core_film_scale[film_t, c], self.core_film_shift[film_t, c])
                router_logit_bias = None
                if self.core_router_bias is not None:
                    # E6: depth-conditioned router bias. Sized to max_trained_recurrence;
                    # eval deeper than trained reuses the last iteration's bias.
                    bias_t = min(t, cfg.max_trained_recurrence - 1)
                    router_logit_bias = self.core_router_bias[bias_t, c]
                hidden, stats = apply_block(block, hidden, eff_idx, film, router_logit_bias)
                core_iter_stats[c].append(stats)
                eff_idx += 1
            if track_consistency:
                delta = hidden - x_prev
                num = jnp.sum(jnp.square(delta.astype(jnp.float32)), axis=-1)
                den = jnp.sum(jnp.square(x_prev.astype(jnp.float32)), axis=-1) + cfg.layer_norm_eps
                consistency_accum = consistency_accum + jnp.mean(num / den)
                x_prev = hidden
            if collect_readouts:
                readout = self.final_gated_norm(self.final_norm(hidden))
                anytime_readouts.append(readout)
                if track_ponder:
                    ponder_halt_logits.append(jnp.einsum("bsd,d->bs", readout, self.halt_head))
        per_block_stats.extend(_mean_router_stats(s) for s in core_iter_stats)

        for block in coda_blocks:
            hidden, stats = apply_block(block, hidden, eff_idx)
            per_block_stats.append(stats)
            eff_idx += 1

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in per_block_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in per_block_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in per_block_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in per_block_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in per_block_stats], axis=0),
        }
        if track_consistency:
            router_metrics["core_consistency"] = consistency_accum / n_recurrence
        if collect_readouts:
            # (n_recurrence, B, S, D). Consumed by next_token_loss (training only)
            # and dropped before _summarize_router_metrics, so it never escapes the
            # train_step metrics dict. Present whenever ponder is on too, since the
            # PonderNet recon term reads each per-iteration readout's CE.
            router_metrics["anytime_readouts"] = jnp.stack(anytime_readouts, axis=0)
        if track_ponder:
            # (n_recurrence, B, S). Per-iteration halting logits for the PonderNet loss.
            router_metrics["ponder_halt_logits"] = jnp.stack(ponder_halt_logits, axis=0)
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        recurrence_steps: int | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask, recurrence_steps=recurrence_steps)
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
        recurrence_steps: int | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(token_ids, mask=mask, recurrence_steps=recurrence_steps)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        # No load-balancing loss; router z-loss only.
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        aux_loss = self.config.router_z_loss_coef * rzl
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
        # E5: training-only core-consistency penalty. Gated on the static config so
        # weight==0 (E0-E3) leaves the eval (reduction="none") path untouched.
        consistency_weighted = None
        if reduction != "none" and self.config.core_consistency_weight > 0:
            consistency = router_metrics["core_consistency"]
            consistency_weighted = self.config.core_consistency_weight * consistency
            loss = loss + consistency_weighted
        # E4: training-only anytime deep-supervision. For each per-iteration readout
        # collected in the forward, compute CE against the same labels through the
        # shared output head, average over iterations, and add a weighted term.
        # Gated on reduction so the eval (reduction="none") path is byte-identical.
        anytime_ce = None
        anytime_ce_weighted = None
        if reduction != "none" and self.config.anytime_supervision and "anytime_readouts" in router_metrics:
            readouts = router_metrics["anytime_readouts"]
            n_iter = readouts.shape[0]
            ce_sum = jnp.zeros((), dtype=loss_dtype)
            for t in range(n_iter):
                ce_sum = ce_sum + fused_linear_softmax_cross_entropy_loss(
                    readouts[t],
                    self.output_proj,
                    labels,
                    weight=loss_weight,
                    reduction=reduction,
                    logsumexp_weight=logsumexp_weight,
                    dtype=loss_dtype,
                )
            anytime_ce = ce_sum / n_iter
            anytime_ce_weighted = self.config.anytime_supervision_weight * anytime_ce
            loss = loss + anytime_ce_weighted
        # E7 (PonderNet): training-only halting loss. Builds a per-token halting
        # distribution p_t over the R core iterations from the learned halting head,
        # then adds an expected-over-halting CE (recon) plus a KL-to-geometric-prior
        # regularizer. ADDED to the standard final CE (not a replacement), so the eval
        # (reduction="none") path and the comparable metric stay byte-identical.
        ponder_recon = None
        ponder_kl = None
        expected_halt_step = None
        if reduction != "none" and self.config.ponder_halting and "ponder_halt_logits" in router_metrics:
            readouts = router_metrics["anytime_readouts"]  # (R, B, S, D)
            halt_logits = router_metrics["ponder_halt_logits"]  # (R, B, S)
            n_recurrence = readouts.shape[0]
            # Halting distribution p_t (R, B, S), Sum_t p_t == 1 exactly: the last
            # step absorbs all remaining mass (forced halt). remain_t is the exclusive
            # cumulative product of (1 - lam) so remain_0 == 1.
            lam = jax.nn.sigmoid(halt_logits.astype(loss_dtype))
            one_minus = 1.0 - lam
            incl = jnp.cumprod(one_minus, axis=0)
            # remain_t = exclusive cumprod of (1 - lam) (remain_0 = 1). Build it by
            # shifting `incl` down one step and seeding step 0 with ones; `ones_like`
            # inherits incl's batch sharding so the concatenate stays well-sharded.
            ones = jnp.ones_like(incl[:1])
            remain = jnp.concatenate([ones, incl[:-1]], axis=0)
            p = lam * remain
            p = p.at[-1].set(remain[-1])
            # Raw (unweighted) per-position CE for each per-iteration readout (R, B, S).
            ce_list = [
                fused_linear_softmax_cross_entropy_loss(
                    readouts[t],
                    self.output_proj,
                    labels,
                    weight=None,
                    reduction="none",
                    dtype=loss_dtype,
                )
                for t in range(n_recurrence)
            ]
            ce = jnp.stack(ce_list, axis=0)  # (R, B, S)
            recon_per_pos = jnp.sum(p * ce, axis=0)  # (B, S)
            denom = jnp.sum(loss_weight)
            ponder_recon = jnp.sum(loss_weight * recon_per_pos) / denom
            # Geometric prior g_t = lp*(1-lp)^t for t<R-1, g_{R-1}=(1-lp)^{R-1}; Sum_t g_t == 1.
            lp = jnp.asarray(self.config.ponder_prior_lambda, dtype=loss_dtype)
            t_idx = jnp.arange(n_recurrence, dtype=loss_dtype)
            prior = lp * (1.0 - lp) ** t_idx
            prior = prior.at[-1].set((1.0 - lp) ** (n_recurrence - 1))
            eps = 1e-8
            g = prior.reshape((n_recurrence, 1, 1))
            kl_per_pos = jnp.sum(p * (jnp.log(p + eps) - jnp.log(g + eps)), axis=0)  # (B, S)
            ponder_kl = jnp.sum(loss_weight * kl_per_pos) / denom
            # Diagnostic: loss_weight-weighted mean of E_t[t] = Sum_t p_t * t over positions.
            steps = jnp.arange(n_recurrence, dtype=p.dtype).reshape((n_recurrence, 1, 1))
            expected_halt_per_pos = jnp.sum(p * steps, axis=0)  # (B, S)
            expected_halt_step = jnp.sum(loss_weight * expected_halt_per_pos) / denom
            loss = loss + self.config.ponder_loss_weight * ponder_recon + self.config.ponder_kl_weight * ponder_kl
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            if "core_consistency" in router_metrics:
                summarized_metrics["train/core_consistency"] = router_metrics["core_consistency"]
                summarized_metrics["train/core_consistency_weighted"] = consistency_weighted
            if anytime_ce is not None:
                summarized_metrics["train/anytime_ce"] = anytime_ce
                summarized_metrics["train/anytime_ce_weighted"] = anytime_ce_weighted
            if ponder_recon is not None:
                summarized_metrics["train/ponder_recon"] = ponder_recon
                summarized_metrics["train/ponder_kl"] = ponder_kl
                summarized_metrics["train/ponder_expected_halt_step"] = expected_halt_step
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
    "Transformer",
    "debug_mesh_and_token_pspec",
]
