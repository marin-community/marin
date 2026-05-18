# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass

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
from levanter.grug.attention import AttentionMask, RotaryConfig, align_kv_heads, apply_rotary_embedding, attention
from levanter.grug.grug_moe import MoeActivation, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard
from levanter.tracker.histogram import Histogram
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty or axis_name not in mesh.shape:
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


def _batch_spec() -> P:
    return P(("data", "expert"))


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
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    # Attention gate mode: "full" (default), "none", "truncated", "lora".
    attn_gate_mode: str = "full"
    # Fraction of hidden_dim for truncated gate_dim or LoRA low_rank.
    # Only used when attn_gate_mode is "truncated" or "lora".
    attn_gate_fraction: float = 1.0
    # Partial key offset: "none" (default), "every_4th", "every_layer".
    # Applies partial RoPE (first half of head_dim) and shifts stationary
    # key dims forward by one position to enable 1-layer induction.
    partial_key_offset: str = "none"
    # Partial rope on non-PKO layers: only rotate half the head dims.
    use_partial_rope: bool = False
    # Force the last layer to use long sliding window + PKO.
    last_layer_pko: bool = False
    # Per-token rescaling of the rotating vs stationary halves of ``q`` in
    # PKO layers only. A learned per-block ``q_split_rescale_weight`` of
    # shape ``(hidden_dim, num_kv_heads)`` (zero-init) produces a scalar
    # per (token, kv-head); broadcast across the q-heads in each GQA group.
    #
    #   s = 2 * sigmoid(x @ w)                    # stationary weight in [0, 2]
    #   r = 2 - s                                 # rotating weight in [0, 2]
    #   q_rot = q[..., :half]    * r
    #   q_stat = q[..., half:]   * s
    #
    # Init s = r = 1.0 (no-op vs baseline). Only active in PKO layers.
    #
    # ``"none"``      — no rescale (default).
    # ``"pre_norm"``  — apply BEFORE ``rms_norm(q)``; rms_norm then
    #                   normalises but the per-half ratio is preserved.
    # ``"post_norm"`` — apply AFTER ``rms_norm(q)`` and before
    #                   partial-RoPE; post-norm half magnitudes are
    #                   directly scaled.
    pko_q_split_rescale_mode: str = "none"
    # Granularity of the q-half rescale weight. ``"kv"`` -> shape
    # ``(hidden_dim, num_kv_heads)``, broadcast across GQA groups.
    # ``"q"`` -> shape ``(hidden_dim, num_heads)``, one scalar per q-head
    # (each q-head can independently choose its rotating/stationary mix).
    pko_q_split_rescale_heads: str = "kv"
    # V2 variant: TWO independent per-q-head weight matrices for the
    # stationary and rotating halves; each produces a per-token, per-q-head
    # ``sigmoid`` in [0, 1] (NOT 2*sigmoid). At init both halves are scaled
    # by 0.5 (q magnitude halved). Only active in PKO layers.
    #
    #   stat_w = sigmoid(x @ W_stat)          # (B, S, num_heads), values in [0, 1]
    #   rot_w  = sigmoid(x @ W_rot)           # (B, S, num_heads), values in [0, 1]
    #   q_stat = q[..., half:] * stat_w
    #   q_rot  = q[..., :half] * rot_w
    #
    # ``"none"``      — disabled (default).
    # ``"pre_norm"``  — apply BEFORE rms_norm(q).
    # ``"post_norm"`` — apply AFTER rms_norm(q), before partial-RoPE.
    #
    # Weight matrices are routed to a dedicated optimizer group with
    # Adam(b1=0.95, b2=0.999) at 0.1x the small-LR ``adam_lr``.
    pko_q_split_rescale_v2_mode: str = "none"
    # V3 (SiLU-gated query): redefine q in PKO layers as
    #   q = (q_w @ x) * silu(g_w @ x)
    # where ``g_w`` has shape ``(hidden_dim, num_heads, 2)`` and the two
    # trailing scalars are independent multipliers for the rotating and
    # stationary halves of each q-head:
    #
    #   gate = silu(einsum("bsd,dn2->bsn2", x, g_w))    # (B, S, num_heads, 2)
    #   q_rot  *= gate[..., 0:1]
    #   q_stat *= gate[..., 1:2]
    #
    # Applied BEFORE rms_norm(q) and partial-RoPE, only on PKO layers.
    # ``g_w`` is allocated with ``initializer_std`` (matching other weight
    # matrices) so the gate has non-degenerate small random values at init.
    pko_q_silu_gate: bool = False
    # Sigmoid gate on the stationary half of ``q`` in PKO layers only.
    # ``w`` has shape ``(hidden_dim, num_heads)`` (zero-init); produces
    # per-token, per-q-head sigmoid in [0, 1]. Applied to the stationary
    # half BEFORE rms_norm(q) and partial-RoPE.
    #
    #   gate = sigmoid(x @ w)                # (B, S, num_heads), values in [0, 1]
    #   q_stat = q[..., half:] * gate
    #   (q_rot unchanged)
    #
    # At init, sigmoid(0) = 0.5 -> stationary half is halved at init.
    # Routed to the small-LR ``adam`` group.
    pko_q_stat_sigmoid_gate: bool = False

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
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
        if self.pko_q_split_rescale_mode not in ("none", "pre_norm", "post_norm"):
            raise ValueError(
                f"pko_q_split_rescale_mode must be 'none' / 'pre_norm' / 'post_norm'; "
                f"got {self.pko_q_split_rescale_mode!r}"
            )
        if self.pko_q_split_rescale_heads not in ("kv", "q"):
            raise ValueError(f"pko_q_split_rescale_heads must be 'kv' or 'q'; got {self.pko_q_split_rescale_heads!r}")
        if self.pko_q_split_rescale_v2_mode not in ("none", "pre_norm", "post_norm"):
            raise ValueError(
                f"pko_q_split_rescale_v2_mode must be 'none' / 'pre_norm' / 'post_norm'; "
                f"got {self.pko_q_split_rescale_v2_mode!r}"
            )

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Non-parametric RMS norm over the last dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "... N"] | None
    attn_gate_up: Float[Array, "R N"] | None
    # Per-token weight on the rotating vs stationary halves of q in PKO
    # layers; shape ``(hidden_dim, num_kv_heads)``. Only allocated when
    # ``cfg.pko_q_split_rescale_mode != "none"``. See GrugModelConfig docs.
    q_split_rescale_weight: Float[Array, "D M"] | None
    # V2: two independent (hidden_dim, num_heads) weights, each producing a
    # per-token, per-q-head sigmoid in [0, 1] for stationary and rotating.
    q_split_rescale_v2_stat_weight: Float[Array, "D N"] | None
    q_split_rescale_v2_rot_weight: Float[Array, "D N"] | None
    # V3 (SiLU-gated query): shape (hidden_dim, num_heads, 2). The trailing
    # 2 are (rotating-half, stationary-half) multipliers per q-head.
    q_silu_gate_weight: Float[Array, "D N two"] | None
    # V4 (stationary-only sigmoid gate): shape (hidden_dim, num_heads).
    # Gates only the stationary half of q in PKO layers.
    q_stat_sigmoid_gate_weight: Float[Array, "D N"] | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o, k_q_silu = random.split(key, 5)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim

        gate_mode = cfg.attn_gate_mode
        gate_frac = cfg.attn_gate_fraction
        attn_gate: jax.Array | None = None
        attn_gate_up: jax.Array | None = None

        if gate_mode == "full":
            attn_gate = reshard(jnp.zeros((d, n)), P(None, None))
        elif gate_mode == "truncated":
            gate_dim = max(1, int(d * gate_frac))
            attn_gate = reshard(jnp.zeros((gate_dim, n)), P(None, None))
        elif gate_mode == "lora":
            low_rank = max(1, int(d * gate_frac))
            attn_gate = reshard(jnp.zeros((d, low_rank)), P(None, None))
            attn_gate_up = reshard(jnp.zeros((low_rank, n)), P(None, None))
        elif gate_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown attn_gate_mode: {gate_mode!r}")

        q_split_rescale_weight: jax.Array | None = None
        if cfg.pko_q_split_rescale_mode != "none":
            heads_dim = m if cfg.pko_q_split_rescale_heads == "kv" else n
            q_split_rescale_weight = reshard(jnp.zeros((d, heads_dim)), P(None, None))

        q_split_rescale_v2_stat_weight: jax.Array | None = None
        q_split_rescale_v2_rot_weight: jax.Array | None = None
        if cfg.pko_q_split_rescale_v2_mode != "none":
            q_split_rescale_v2_stat_weight = reshard(jnp.zeros((d, n)), P(None, None))
            q_split_rescale_v2_rot_weight = reshard(jnp.zeros((d, n)), P(None, None))

        q_silu_gate_weight: jax.Array | None = None
        if cfg.pko_q_silu_gate:
            q_silu_gate_weight = reshard(_init_weight(k_q_silu, (d, n, 2), cfg.initializer_std), P(None, None, None))

        q_stat_sigmoid_gate_weight: jax.Array | None = None
        if cfg.pko_q_stat_sigmoid_gate:
            q_stat_sigmoid_gate_weight = reshard(jnp.zeros((d, n)), P(None, None))

        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            attn_gate=attn_gate,
            attn_gate_up=attn_gate_up,
            q_split_rescale_weight=q_split_rescale_weight,
            q_split_rescale_v2_stat_weight=q_split_rescale_v2_stat_weight,
            q_split_rescale_v2_rot_weight=q_split_rescale_v2_rot_weight,
            q_silu_gate_weight=q_silu_gate_weight,
            q_stat_sigmoid_gate_weight=q_stat_sigmoid_gate_weight,
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_partial_key_offset: bool = False,
        use_partial_rope: bool = False,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)

        # Compute q-half rescale weights once if enabled. Only used on PKO
        # layers. Shape ``(B, S, num_q_heads, 1)`` for broadcasting against
        # the (B, S, num_q_heads, half) q halves. ``num_q_heads = group_size
        # * num_kv_heads`` so we repeat each kv-head's scalar across the
        # group_size q-heads that share it.
        q_rot_w: jax.Array | None = None
        q_stat_w: jax.Array | None = None
        if use_partial_key_offset and self.q_split_rescale_weight is not None:
            # Logits shape: (B, S, heads) where heads is either num_kv_heads
            # (broadcast across GQA groups) or num_heads (per-q-head gate).
            weight_logits = jnp.einsum("bsd,dh->bsh", x, self.q_split_rescale_weight)
            stat_weight = 2 * jax.nn.sigmoid(weight_logits).astype(x.dtype)
            rot_weight = (2 - stat_weight).astype(x.dtype)
            if self.cfg.pko_q_split_rescale_heads == "kv":
                n_q, m_kv = self.cfg.num_heads, self.cfg.num_kv_heads
                group_size = n_q // m_kv
                # Broadcast each kv-head's scalar across the group_size q-heads
                # that share it.
                stat_weight = jnp.repeat(stat_weight, group_size, axis=-1)
                rot_weight = jnp.repeat(rot_weight, group_size, axis=-1)
            # Now (B, S, num_q_heads); expand for broadcasting against
            # (B, S, num_q_heads, half) q halves.
            q_stat_w = stat_weight[..., None]
            q_rot_w = rot_weight[..., None]
            if self.cfg.pko_q_split_rescale_mode == "pre_norm":
                half = head_dim // 2
                q = jnp.concatenate([q[..., :half] * q_rot_w, q[..., half:] * q_stat_w], axis=-1)
                # Consume now so we don't re-apply below.
                q_rot_w = None
                q_stat_w = None

        # V2: two independent sigmoid weights for stationary / rotating halves.
        q_v2_rot_w: jax.Array | None = None
        q_v2_stat_w: jax.Array | None = None
        if use_partial_key_offset and self.q_split_rescale_v2_stat_weight is not None:
            stat_logits = jnp.einsum("bsd,dn->bsn", x, self.q_split_rescale_v2_stat_weight)
            rot_logits = jnp.einsum("bsd,dn->bsn", x, self.q_split_rescale_v2_rot_weight)
            q_v2_stat_w = jax.nn.sigmoid(stat_logits).astype(x.dtype)[..., None]
            q_v2_rot_w = jax.nn.sigmoid(rot_logits).astype(x.dtype)[..., None]
            if self.cfg.pko_q_split_rescale_v2_mode == "pre_norm":
                half = head_dim // 2
                q = jnp.concatenate([q[..., :half] * q_v2_rot_w, q[..., half:] * q_v2_stat_w], axis=-1)
                q_v2_rot_w = None
                q_v2_stat_w = None

        # V3 (SiLU-gated query): q *= silu(g_w @ x), with two scalars per
        # q-head (one for rotating half, one for stationary). Applied BEFORE
        # rms_norm. Only on PKO layers.
        if use_partial_key_offset and self.q_silu_gate_weight is not None:
            gate_logits = jnp.einsum("bsd,dnk->bsnk", x, self.q_silu_gate_weight)
            gate = jax.nn.silu(gate_logits).astype(x.dtype)  # (B, S, num_heads, 2)
            half = head_dim // 2
            # gate[..., 0:1] is the rotating-half multiplier; gate[..., 1:2] is stationary.
            q = jnp.concatenate([q[..., :half] * gate[..., 0:1], q[..., half:] * gate[..., 1:2]], axis=-1)

        # V4 (stationary-only sigmoid gate): q_stat *= sigmoid(w @ x).
        # Applied BEFORE rms_norm and partial-RoPE. Only on PKO layers.
        # q_rot is unchanged.
        if use_partial_key_offset and self.q_stat_sigmoid_gate_weight is not None:
            gate_logits = jnp.einsum("bsd,dn->bsn", x, self.q_stat_sigmoid_gate_weight)
            gate = jax.nn.sigmoid(gate_logits).astype(x.dtype)[..., None]  # (B, S, num_heads, 1)
            half = head_dim // 2
            q = jnp.concatenate([q[..., :half], q[..., half:] * gate], axis=-1)

        q = rms_norm(q)
        k = rms_norm(k)

        if q_rot_w is not None:
            # post_norm: rescale after rms_norm and before partial-RoPE.
            half = head_dim // 2
            q = jnp.concatenate([q[..., :half] * q_rot_w, q[..., half:] * q_stat_w], axis=-1)

        if q_v2_rot_w is not None:
            # v2 post_norm: rescale after rms_norm and before partial-RoPE.
            half = head_dim // 2
            q = jnp.concatenate([q[..., :half] * q_v2_rot_w, q[..., half:] * q_v2_stat_w], axis=-1)

        if use_partial_key_offset:
            # Partial RoPE: only rotate the first half of head dims.
            # Concatenate rotated and stationary halves to avoid sharding issues
            # with .at[].set() on model-sharded arrays.
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                q[..., :half], k[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            q = jnp.concatenate([q_rot, q[..., half:]], axis=-1)
            # Shift stationary key dims forward by one position (enables 1-layer induction).
            k_stationary = k[..., half:]
            k_shifted = jnp.concatenate([k_stationary[:, :1, :, :], k_stationary[:, :-1, :, :]], axis=1)
            k = jnp.concatenate([k_rot, k_shifted], axis=-1)
        elif use_partial_rope:
            # Partial RoPE only (no key shift): rotate first half, leave rest stationary.
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                q[..., :half], k[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            q = jnp.concatenate([q_rot, q[..., half:]], axis=-1)
            k = jnp.concatenate([k_rot, k[..., half:]], axis=-1)
        else:
            q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask)
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = reshard(aligned_v, P(("data", "expert"), None, "model", None))
        # Exclusive Self Attention: subtract the component of yᵢ parallel to vᵢ.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: sigmoid produces one scalar per head.
        if self.attn_gate is not None:
            if self.attn_gate_up is not None:
                # LoRA: x @ W_down @ W_up -> [B, S, N]
                gate_logits = jnp.einsum("bsd,dr->bsr", x, self.attn_gate)
                gate_logits = jnp.einsum("bsr,rn->bsn", gate_logits, self.attn_gate_up)
            else:
                gate_dim = self.attn_gate.shape[0]
                if gate_dim < x.shape[-1]:
                    # Truncated: use first gate_dim elements of activation
                    gate_logits = jnp.einsum("bsg,gn->bsn", x[..., :gate_dim], self.attn_gate)
                else:
                    # Full: x @ attn_gate -> [B, S, N]
                    gate_logits = jnp.einsum("bsd,dn->bsn", x, self.attn_gate)
            gate = 2 * jax.nn.sigmoid(gate_logits)[..., None]
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
        return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


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


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | Histogram]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    out: dict[str, jax.Array | Histogram] = {
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


def _histogram_from_expert_counts(expert_counts: jax.Array) -> Histogram:
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
    return Histogram(
        min=min_value,
        max=max_value,
        num=num,
        sum=sum_values,
        sum_squares=sum_squares,
        bucket_limits=bucket_limits,
        bucket_counts=counts,
    )


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    w_gate_up: jax.Array
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_gate, k_up, k_down = random.split(key, 4)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim
        w_gate = _init_weight(k_gate, (e, d, i), cfg.initializer_std)
        w_up = _init_weight(k_up, (e, d, i), cfg.initializer_std)
        # TODO: Explore whether concatenating gate/up at init (instead of keeping separate params)
        # is (1) a meaningful MFU speedup and (2) a meaningful perf hit due to AdamH treating the
        # concatenated tensor as a single parameter for its scale-invariant norm computation.
        w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)

        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            w_gate_up=reshard(w_gate_up, P("expert", "data", "model")),
            w_down=reshard(_init_weight(k_down, (e, i, d), cfg.initializer_std), P("expert", "model", "data")),
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
        combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )
        # Sharded QB: compute beta locally per device, then average.
        s_minus_alpha = router_logits - qb_alpha
        mesh = get_abstract_mesh()
        batch_axes = ("data", "expert")
        num_devices = 1
        for a in batch_axes:
            if a in mesh.shape:
                num_devices *= mesh.shape[a]
        local_tokens = s_minus_alpha.shape[0] // num_devices
        qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

        def _local_qb_beta(s_ma):
            topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
            beta = topk_vals[:, -1]
            return jax.lax.pmean(beta, axis_name=batch_axes)

        router_stats["qb_beta"] = shard_map(
            _local_qb_beta,
            mesh=mesh,
            in_specs=(P(batch_axes, None),),
            out_specs=P(),
        )(s_minus_alpha)

        routed_flat = moe_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            self.w_gate_up,
            self.w_down,
            activation=ActivationFunctionEnum.silu,
            mesh=get_abstract_mesh(),
            capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
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
        use_partial_key_offset: bool = False,
        use_partial_rope: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(
            attn_in, mask, use_partial_key_offset=use_partial_key_offset, use_partial_rope=use_partial_rope
        )
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        pko_mode = cfg.partial_key_offset
        num_blocks = len(self.blocks)
        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            is_last = i == num_blocks - 1
            is_long = i % 4 == 3 or (cfg.last_layer_pko and is_last)
            layer_mask = long_mask if is_long else short_mask
            use_pko = (pko_mode == "every_layer") or (pko_mode == "every_4th" and is_long)
            partial_rope = cfg.use_partial_rope and not use_pko
            hidden, router_stats = eqx.filter_checkpoint(block)(hidden, layer_mask, use_pko, partial_rope)
            moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
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
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | Histogram]]:
        hidden, router_metrics = self(token_ids, mask=mask)
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
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
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
        axis_sizes=(data, expert, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("data", "expert"), None)


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
    "moe_mlp",
]
