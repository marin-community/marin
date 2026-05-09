# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nano transformer: JAX/Equinox port of the modded-nanogpt reference architecture.

Mirrors `experiments/grug/nanogpt_ref.py` exactly:
- Parametric RMSNorm with learnable gains, applied PreNorm in each block plus
  a norm right after the embed and a final norm before the output projection.
- Linear layers with bias on every projection.
- Half-truncated RoPE (base = 1/1024) applied to q,k after a non-parametric QK norm.
- Causal self-attention with a fixed scale of `attn_scale` (default 0.12)
  rather than the conventional 1/sqrt(head_dim).
- MLP with squared-ReLU activation and 4x intermediate dim.
- Zero-initialized "proj" weights (attn output, mlp output, lm head).
- Logit soft-cap: `cap * x * rsqrt(x^2 + cap^2)`, matching the ref exactly.
  (The loss is computed manually because the fused kernel does not support
  this form.) Note: this differs from the `cap * tanh(x / cap)` shape — the
  rsqrt form saturates *polynomially* (`1 - 0.5/(x/cap)^2` for large x),
  whereas tanh saturates *exponentially*. Gradient through the cap is
  `cap^3 / (x^2 + cap^2)^(3/2)`.

Keeps the file layout from `experiments/grug/base/model.py` (config dataclass,
eqx.Module submodules, `Transformer.next_token_loss`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask, align_kv_heads, attention
from levanter.grug.grug_moe import ActivationFunctionEnum, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, unshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map


@dataclass(frozen=True)
class NanoModelConfig:
    """Hyperparameters for the nano (modded-nanogpt) transformer."""

    vocab_size: int
    hidden_dim: int = 768
    intermediate_dim: int = 3072  # 4 * hidden_dim
    num_layers: int = 12
    num_heads: int = 6
    head_dim: int = 128
    max_seq_len: int = 1024
    layer_norm_eps: float = 1e-5
    attn_scale: float = 0.12
    logit_cap: float = 15.0
    rope_base: float = 1.0 / 1024.0
    initializer_std: float = 0.02
    zero_init_proj: bool = True
    # "default"    = nanogpt_ref Muon init: truncated normal, "proj" weights zeroed.
    # "adamh_ref"  = nanogpt_adamh_ref Kaiming-uniform init with per-module
    #                multipliers so AdamH has non-zero matrices to operate on from step 0.
    # "muon_tuned" = experiments/grug/muon_tuned.py init: embed ~ N(0, 1) (PyTorch
    #                Embedding default), non-proj weights ~ N(0, sqrt(0.33 / fan_in))
    #                (PyTorch nn.Linear default), proj weights / biases zeroed,
    #                RMSNorm gains = ones.
    init_scheme: str = "default"
    # Logit soft-cap shape:
    #   "rsqrt" : cap * x * rsqrt(x^2 + cap^2)   (modded-nanogpt ref formula)
    #   "tanh"  : cap * tanh(x / cap)            (alternative soft-cap, faster saturation)
    cap_form: str = "rsqrt"

    # ---- Optional moe-style features ----
    # `qk_mult`: multiply q by this scalar after QK norm + RoPE (before attention scale).
    # In moe convention this is set to 1.3.
    qk_mult: float = 1.0
    # `use_attn_gate`: per-head learned sigmoid gate on attention output (zero-init,
    # so the gate starts at 0.5 and the gating becomes identity at init since `2 * 0.5 = 1`).
    use_attn_gate: bool = False
    # `use_gated_norm`: insert a learnable rank-`gated_norm_rank` GatedNorm module after each
    # parametric RMSNorm (block attn-in / mlp-in / post-embed / pre-lm-head). Compensates for
    # AdamH's Frobenius-norm preservation by giving the model another scaling knob.
    use_gated_norm: bool = False
    gated_norm_rank: int = 128
    # `use_bias`: when False, every Linear in the model (q/k/v/proj/mlp.fc/mlp.proj/lm head)
    # drops its additive bias term. Matches the moe model layout.
    use_bias: bool = True
    # `sliding_window`: when set, 3-out-of-every-4 blocks use sliding-window causal
    # attention with window = `sliding_window // 2`; every 4th block (i % 4 == 3) uses
    # full causal attention with window = `sliding_window`. Matches the moe pattern in
    # `experiments/grug/moe/model.py:Transformer.__call__`. `None` disables the
    # pattern (all blocks use plain causal attention).
    sliding_window: int | None = None
    # `mlp_type`: "relu_squared" (default, modded-nanogpt) uses two matrices
    # (`fc` -> up, `proj` -> down) with `relu(x)**2` activation. "swiglu" uses
    # three matrices (`fc` = up, `gate`, `proj` = down) with
    # `silu(x @ gate) * (x @ fc)` then `out = ... @ proj`. Matches the moe DenseMLP.
    mlp_type: str = "relu_squared"
    # `num_kv_heads`: when None, defaults to `num_heads` (full multi-head).
    # When set to a divisor of `num_heads`, k and v project to `num_kv_heads * head_dim`
    # channels and are broadcast up to `num_heads` inside `levanter.grug.attention.attention`.
    # The moe heuristic uses `num_heads / 4` (then rounded down to the largest divisor).
    num_kv_heads: int | None = None
    # `use_xsa`: Exclusive Self-Attention. After attention, subtract the component
    # of each head's output parallel to its `aligned_v`:
    #   z = y - (y^T v / ||v||^2) * v   per head.
    # Matches `experiments/grug/moe/model.py:148-152`. Runs *before* the optional
    # `attn_gate` so the gate operates on the orthogonalized output.
    use_xsa: bool = False

    # ---- MoE (mixture-of-experts) ----
    # When `use_moe=True`, the per-block MLP is replaced by an `MoEMLP` that
    # routes each token to `num_experts_per_token` of `num_experts` experts via
    # a QB-style router (see `train.py:_apply_qb_betas`). Optionally a shared
    # expert (DenseMLP) is added in parallel. Routed expert MLPs use SwiGLU
    # (silu(gate) * up) at `expert_intermediate_dim`; the shared expert uses
    # the model's configured `mlp_type` at `shared_expert_intermediate_dim`.
    use_moe: bool = False
    num_experts: int = 64
    num_experts_per_token: int = 4
    # Intermediate dim for routed experts. None => use `intermediate_dim`.
    expert_intermediate_dim: int | None = None
    # >0 enables a shared dense expert at this intermediate dim, in parallel.
    shared_expert_intermediate_dim: int = 0
    # When True, store gate and up as separate (E, D, I) tensors and concat on
    # the forward pass before passing to `moe_mlp`. Matches the user's request.
    separate_gate_up: bool = True
    # Coefficient on the router-z-loss (logsumexp-squared on router logits).
    router_z_loss_coef: float = 0.001
    # When set, the shared dense expert overrides the global ``mlp_type`` with
    # this value (e.g. shared = ReLU² while routed experts stay SwiGLU). ``None``
    # means the shared expert inherits ``cfg.mlp_type``. Only meaningful when
    # ``use_moe=True`` and ``shared_expert_intermediate_dim > 0``.
    shared_expert_mlp_type: str | None = None

    # ---- Intra-document attention masking ----
    # Token id of the document-boundary marker in the cached token stream.
    # When set, `Transformer.__call__` derives `segment_ids` from
    # `cumsum(token_ids == intra_doc_bos_id, axis=-1)` and attaches them to
    # the attention mask. This makes every token in a document attend only to
    # earlier tokens in the *same* document (including the boundary marker
    # itself, which gets the same segment id as the doc it precedes).
    #
    # We do this in the model rather than relying on
    # `LmDataConfig.block_cross_document_attention=True` because:
    #
    #   1. Our `MarinTokenizer` for "gpt2" exposes neither `bos_token_id` nor
    #      `eos_token_id` (both are None). With `eos_id=None`, levanter's
    #      `GrugLmExample.causal` skips segment-id derivation entirely.
    #   2. Even if we passed an `eos_id`, the levanter logic uses
    #      `roll(tokens, 1) == eos_id` which places the marker in the
    #      *previous* segment — wrong for a cache that prepends the marker
    #      at the start of each doc (modded-nanogpt's fineweb10B-gpt2 layout:
    #      ``[<eot> doc1 <eot> doc2 …]``). We want the marker in the *same*
    #      segment as the doc it begins.
    #
    # Set to ``50256`` for the gpt2 cache. ``None`` disables segment masking.
    intra_doc_bos_id: int | None = None

    # ---- Fused cross-entropy ----
    # When True, route the lm-head + softmax + cross-entropy through
    # ``levanter.grug.loss.fused_linear_softmax_cross_entropy_loss`` (the same
    # kernel grug/moe uses). The fused kernel materializes per-shard CE
    # losses with `shard_map` + `pmean`, which both saves the (B, S, V)
    # logits HBM allocation *and* avoids the explicit-mesh-axes psum
    # mismatch that bit our manual CE path. When False, use the manual
    # path that's been driving the walks so far (materialized logits with
    # optional rsqrt/tanh soft-cap). The fused kernel does not support
    # the rsqrt soft-cap form; pass ``logit_cap=None`` (or use its tanh
    # form) when ``use_fused_ce=True``.
    use_fused_ce: bool = False

    # ---- PKO (Partial Key Offset) ----
    # When True, every 4th layer (the same long-window layers as the
    # sliding-window pattern: i % 4 == 3) shifts the second half of the key
    # tensor forward by one position after RoPE: ``k[:, t, :, head_dim//2:]
    # = k[:, t-1, :, head_dim//2:]`` for ``t >= 1`` (position 0 keeps its
    # own keys). The rotated first half is left untouched. This enables
    # 1-layer induction-head behaviour on those layers — a query at
    # position t can match against features that were *content-only* at
    # position t-1.
    #
    # Source semantics: ``experiments/grug/moe/model.py`` on the
    # `pko_every_4th` line of branches (commits 2b00190a9 + 15684cb39).
    # We don't switch nano's half-truncated RoPE to moe's full RoPE on
    # those layers — only the second-half key shift applies, matching the
    # user's spec ("1 position shift to half the key dim").
    use_pko_every_4th: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.head_dim % 4 != 0:
            # half-truncated RoPE splits head_dim into quarters.
            raise ValueError(f"head_dim must be divisible by 4 for half-truncated RoPE, got {self.head_dim}")
        if self.init_scheme not in ("default", "adamh_ref", "muon_tuned"):
            raise ValueError(f"init_scheme must be 'default', 'adamh_ref', or 'muon_tuned', got {self.init_scheme!r}")
        if self.init_scheme == "adamh_ref" and self.zero_init_proj:
            # AdamH preserves Frobenius norm: zero-init matrices stay zero forever.
            raise ValueError("init_scheme='adamh_ref' requires zero_init_proj=False")
        if self.cap_form not in ("rsqrt", "tanh"):
            raise ValueError(f"cap_form must be 'rsqrt' or 'tanh', got {self.cap_form!r}")
        if self.mlp_type not in ("relu_squared", "swiglu"):
            raise ValueError(f"mlp_type must be 'relu_squared' or 'swiglu', got {self.mlp_type!r}")
        if self.shared_expert_mlp_type is not None and self.shared_expert_mlp_type not in ("relu_squared", "swiglu"):
            raise ValueError(
                f"shared_expert_mlp_type must be None, 'relu_squared', or 'swiglu', got {self.shared_expert_mlp_type!r}"
            )
        if self.use_fused_ce and self.use_bias:
            # The fused kernel signature is ``(hidden, lm_head_weight, labels, ...)``;
            # there's no slot for an lm-head bias, so silently keeping ``use_bias=True``
            # would drop the bias contribution. Fail loudly instead.
            raise ValueError(
                "use_fused_ce=True requires use_bias=False (the fused kernel "
                "doesn't accept a bias term on the lm head)."
            )
        if self.num_kv_heads is not None:
            if self.num_kv_heads <= 0:
                raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

    @property
    def attn_hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def kv_hidden_dim(self) -> int:
        """Total k/v projection output channels (`num_kv_heads * head_dim`)."""
        return self.effective_num_kv_heads * self.head_dim


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:  # noqa: UP037
    # Quoted shape string is required by jaxtyping; ruff's UP037 would otherwise turn it into Ellipsis.
    return std * random.truncated_normal(key, -3, 3, shape)


def _apply_logit_cap(x: jax.Array, cap: float, form: str) -> jax.Array:
    """Apply a soft-cap to logits.

    `form="rsqrt"`: `cap * x * rsqrt(x^2 + cap^2)` — modded-nanogpt reference shape.
    `form="tanh"` : `cap * tanh(x / cap)`         — alternative soft-cap.
    Both saturate at ±cap and are linear near zero; tanh saturates faster.
    """
    if form == "tanh":
        return cap * jnp.tanh(x / cap)
    # default: rsqrt
    return cap * x * jax.lax.rsqrt(x**2 + cap**2)


# ---- Half-truncated RoPE ----
# First half of head_dim uses frequencies (rope_base)^linspace(0,1,head_dim/4),
# second half is zeros (no rotation), matching the modded-nanogpt reference.


def _build_rope_freqs(head_dim: int, base: float) -> jax.Array:
    quarter = head_dim // 4
    angular_freq = base ** jnp.linspace(0, 1, quarter)
    return jnp.concatenate([angular_freq, jnp.zeros(quarter)])


def _apply_half_truncated_rope(
    q: Float[Array, "B S N D"], k: Float[Array, "B S M D"], head_dim: int, base: float
) -> tuple[jax.Array, jax.Array]:
    seq_len = q.shape[1]
    freqs = _build_rope_freqs(head_dim, base)
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    theta = jnp.outer(pos, freqs)  # (S, head_dim/2)
    cos = jnp.cos(theta)[None, :, None, :]
    sin = jnp.sin(theta)[None, :, None, :]

    def rotate(x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x1, x2 = jnp.split(x, 2, axis=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return jnp.concatenate([y1, y2], axis=-1).astype(dtype)

    return rotate(q), rotate(k)


# ---- Norms ----


class RMSNorm(eqx.Module):
    """Parametric RMSNorm with learnable per-channel gains."""

    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> RMSNorm:
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


def _rms_norm(x: jax.Array) -> jax.Array:
    """Non-parametric RMS norm used for QK normalization inside attention."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)


# ---- Linear with bias ----


class LinearWithBias(eqx.Module):
    """Linear layer with optional bias; weights stored as (in, out) for fan-in/fan-out clarity.

    `bias` is `None` when the layer is initialized with `use_bias=False`. Forward-pass
    helpers must check for `None` before adding the bias term.
    """

    weight: jax.Array
    bias: jax.Array | None

    @staticmethod
    def init(
        in_dim: int,
        out_dim: int,
        std: float,
        *,
        key: PRNGKeyArray,
        weight_pspec: P = P(None, None),
        zero_init: bool = False,
        use_bias: bool = True,
    ) -> LinearWithBias:
        if zero_init:
            weight = jnp.zeros((in_dim, out_dim), dtype=jnp.float32)
        else:
            weight = _init_weight(key, (in_dim, out_dim), std)
        bias = jnp.zeros((out_dim,), dtype=jnp.float32) if use_bias else None
        return LinearWithBias(
            weight=reshard(weight, weight_pspec),
            bias=bias,
        )


def _add_bias(out: jax.Array, bias: jax.Array | None) -> jax.Array:
    """Add `bias` to `out` if `bias` is not None."""
    return out + bias if bias is not None else out


# ---- GatedNorm (moe convention) ----
#
# Learnable per-dimension gating with a low-rank MLP: silu(x @ W_down) @ W_up → sigmoid → mul x.
# See https://arxiv.org/abs/2601.22966v1; cargo-culted from `experiments/grug/moe/model.py`.


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating module."""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, rank: int, initializer_std: float, *, key: PRNGKeyArray) -> GatedNorm:
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, rank), initializer_std), P(None, None)),
            w_up=reshard(_init_weight(k_up, (rank, hidden_dim), initializer_std), P(None, None)),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


# ---- Attention ----


class CausalSelfAttention(eqx.Module):
    q: LinearWithBias
    k: LinearWithBias
    v: LinearWithBias
    proj: LinearWithBias
    # Optional per-head attention gate: 2 * sigmoid(x @ attn_gate)[..., None] * attn_out.
    # Zero-init so the gate starts at 0.5 (and `2 * 0.5 = 1`, identity at init).
    attn_gate: jax.Array | None
    cfg: NanoModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> CausalSelfAttention:
        k_q, k_k, k_v, k_proj = random.split(key, 4)
        hdim = cfg.attn_hidden_dim
        kv_hdim = cfg.kv_hidden_dim
        std = cfg.initializer_std
        attn_gate = (
            reshard(jnp.zeros((cfg.hidden_dim, cfg.num_heads), dtype=jnp.float32), P(None, None))
            if cfg.use_attn_gate
            else None
        )
        return CausalSelfAttention(
            q=LinearWithBias.init(
                cfg.hidden_dim, hdim, std, key=k_q, weight_pspec=P("data", "model"), use_bias=cfg.use_bias
            ),
            k=LinearWithBias.init(
                cfg.hidden_dim, kv_hdim, std, key=k_k, weight_pspec=P("data", "model"), use_bias=cfg.use_bias
            ),
            v=LinearWithBias.init(
                cfg.hidden_dim, kv_hdim, std, key=k_v, weight_pspec=P("data", "model"), use_bias=cfg.use_bias
            ),
            proj=LinearWithBias.init(
                hdim,
                cfg.hidden_dim,
                std,
                key=k_proj,
                weight_pspec=P("model", "data"),
                zero_init=cfg.zero_init_proj,
                use_bias=cfg.use_bias,
            ),
            attn_gate=attn_gate,
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        *,
        use_pko: bool = False,
    ) -> Float[Array, "B S D"]:
        cfg = self.cfg
        h = cfg.head_dim
        num_q_heads = cfg.num_heads
        num_kv_heads = cfg.effective_num_kv_heads

        def linear(mod: LinearWithBias, inp: jax.Array) -> jax.Array:
            return _add_bias(jnp.einsum("bsd,df->bsf", inp, mod.weight), mod.bias)

        # q has `num_q_heads` heads; k and v have `num_kv_heads` heads (GQA).
        # `levanter.grug.attention.attention` calls `align_kv_heads` internally.
        q = rearrange(linear(self.q, x), "b s (n d) -> b s n d", n=num_q_heads, d=h)
        k = rearrange(linear(self.k, x), "b s (m d) -> b s m d", m=num_kv_heads, d=h)
        v = rearrange(linear(self.v, x), "b s (m d) -> b s m d", m=num_kv_heads, d=h)
        # QK norm (non-parametric, matches the ref's `norm(x) = F.rms_norm(x, ...)`).
        q, k = _rms_norm(q), _rms_norm(k)
        q, k = _apply_half_truncated_rope(q, k, h, cfg.rope_base)
        # PKO: shift the second half of k forward by one position so token t's
        # query attends against token (t-1)'s second-half key features. Mirrors
        # `experiments/grug/moe/model.py` PKO commit (15684cb39) but skips
        # moe's rope swap — nano's half-truncated rope stays on PKO layers.
        if use_pko:
            half = h // 2
            k_stationary = k[..., half:]
            # Position 0 keeps its own k (no wrap-around); positions 1..S-1
            # see the previous token's stationary key dims.
            k_shifted = jnp.concatenate([k_stationary[:, :1, :, :], k_stationary[:, :-1, :, :]], axis=1)
            k = jnp.concatenate([k[..., :half], k_shifted], axis=-1)
        # Optional QK gain (moe convention; default 1.0 = no-op).
        if cfg.qk_mult != 1.0:
            q = q * cfg.qk_mult
        # Levanter's `attention` uses 1/sqrt(head_dim). Pre-scale q so the effective
        # softmax scale matches the ref's hard-coded `attn_scale`.
        levanter_scale = 1.0 / math.sqrt(h)
        q = q * (cfg.attn_scale / levanter_scale)
        attn_out = attention(q, k, v, mask)
        # Optional XSA: subtract the v-parallel component of attn_out per head.
        # `aligned_v` repeats kv heads up to num_q_heads via `align_kv_heads`,
        # then is resharded to match `attn_out`'s head-axis sharding so the
        # following pointwise mul/sum stay sharding-legal under explicit-mesh-axes.
        # Mirrors `experiments/grug/moe/model.py:148-152` (sans the "expert" axis,
        # which our mesh doesn't have).
        if cfg.use_xsa:
            aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
            aligned_v = reshard(aligned_v, P("data", None, "model", None))
            dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
            v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
            attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Optional per-head sigmoid gate (zero-init -> identity at step 0).
        if self.attn_gate is not None:
            gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
            attn_out = gate * attn_out
        attn_out = rearrange(attn_out, "b s n d -> b s (n d)")
        out = jnp.einsum("bsd,df->bsf", attn_out, self.proj.weight, out_sharding=Pbatch)
        return _add_bias(out, self.proj.bias)


# ---- MLP (squared ReLU, 4x intermediate) ----


class MLP(eqx.Module):
    # `fc` is the "up" projection. `gate` is None for relu_squared, present for swiglu.
    fc: LinearWithBias
    gate: LinearWithBias | None
    proj: LinearWithBias

    @staticmethod
    def init(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> MLP:
        if cfg.mlp_type == "swiglu":
            k_fc, k_gate, k_proj = random.split(key, 3)
            gate = LinearWithBias.init(
                cfg.hidden_dim,
                cfg.intermediate_dim,
                cfg.initializer_std,
                key=k_gate,
                weight_pspec=P("data", "model"),
                use_bias=cfg.use_bias,
            )
        else:
            k_fc, k_proj = random.split(key, 2)
            gate = None
        std = cfg.initializer_std
        return MLP(
            fc=LinearWithBias.init(
                cfg.hidden_dim,
                cfg.intermediate_dim,
                std,
                key=k_fc,
                weight_pspec=P("data", "model"),
                use_bias=cfg.use_bias,
            ),
            gate=gate,
            proj=LinearWithBias.init(
                cfg.intermediate_dim,
                cfg.hidden_dim,
                std,
                key=k_proj,
                weight_pspec=P("model", "data"),
                zero_init=cfg.zero_init_proj,
                use_bias=cfg.use_bias,
            ),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        h_up = _add_bias(jnp.einsum("bsd,df->bsf", x, self.fc.weight), self.fc.bias)
        if self.gate is not None:
            # SwiGLU: silu(x @ gate) * (x @ fc), then proj. Matches moe DenseMLP.
            h_gate = _add_bias(jnp.einsum("bsd,df->bsf", x, self.gate.weight), self.gate.bias)
            h = jax.nn.silu(h_gate) * h_up
        else:
            # ReLU² (modded-nanogpt default).
            h = jax.nn.relu(h_up) ** 2
        out = jnp.einsum("bsf,fd->bsd", h, self.proj.weight, out_sharding=Pbatch)
        return _add_bias(out, self.proj.bias)


# ---- MoE MLP (routed experts) ----
#
# Mirrors `experiments/grug/moe/model.py:MoEMLP` with the user's preferences:
# - `separate_gate_up=True`: gate and up live as separate (E, D, I) tensors and
#   are concatenated along the last axis on every forward pass before passing
#   into `levanter.grug.grug_moe.moe_mlp`. This lets the optimizer treat them
#   as independent params (matters under matrix-norm-preserving updates like AdamH
#   and Muon's per-matrix orthogonalization).
# - QB load balancing: the router computes a per-expert beta (the (K+1)-th
#   logit threshold across tokens) each step; the trainer stores it on the
#   train state and applies it as `router_bias = -beta` on the *next* step.
#   See `train.py:_apply_qb_betas`.
# - Sigmoid combine weights on unbiased router logits for the selected experts
#   (not softmax). Matches moe.


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
    p = jnp.mean(router_probs_f, axis=0)
    token_fraction = assignment_fraction * num_experts_per_token
    load_balancing_loss = num_experts * jnp.sum(token_fraction * p)
    z = jax.scipy.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.mean(z**2)
    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty or axis_name not in mesh.shape:
        raise ValueError(f"nano MoE requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


class MoEMLP(eqx.Module):
    """QB-routed MoE expert MLP.

    Returns ``(output, router_stats)`` from ``__call__`` so the surrounding
    Block / Transformer can collect per-layer stats for loss + logging.
    """

    router: jax.Array  # (D, E)
    router_bias: jax.Array  # (E,) — `stop_gradient` applied at use site
    w_gate: jax.Array  # (E, D, I)
    w_up: jax.Array  # (E, D, I)
    w_down: jax.Array  # (E, I, D)
    cfg: NanoModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> MoEMLP:
        k_router, k_gate, k_up, k_down = random.split(key, 4)
        mesh = jax.sharding.get_abstract_mesh()
        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")
        d = cfg.hidden_dim
        e = cfg.num_experts
        i = cfg.expert_intermediate_dim if cfg.expert_intermediate_dim is not None else cfg.intermediate_dim
        std = cfg.initializer_std
        # Expert weights are sharded along the leading "expert" axis (so each
        # expert lives on one device shard) plus the usual data/model split on
        # the matrix dims. Matches `experiments/grug/moe/model.py:MoEMLP.init`.
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), std), P(None, None)),
            router_bias=jnp.zeros((e,), dtype=jnp.float32),
            w_gate=reshard(_init_weight(k_gate, (e, d, i), std), P("expert", "data", "model")),
            w_up=reshard(_init_weight(k_up, (e, d, i), std), P("expert", "data", "model")),
            w_down=reshard(_init_weight(k_down, (e, i, d), std), P("expert", "model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")

        # Router runs in fp32 before top-k / softmax / QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)

        # Pick top-(K+1); the (K+1)-th biased logit is the QB threshold alpha.
        topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]

        # Sigmoid combine weights on *unbiased* logits for the selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)

        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )

        # QB beta: per-expert top-(local_tokens * K / E) of `router_logits - alpha`,
        # computed *locally* on each batch shard then `pmean`'d across batch axes.
        # Mirrors `experiments/grug/moe/model.py:MoEMLP.__call__`'s shard_map block.
        # Reused on the next step as `router_bias = -beta` (see `train.py:_apply_qb_betas`).
        s_minus_alpha = router_logits - qb_alpha
        mesh = jax.sharding.get_abstract_mesh()
        batch_axes = tuple(a for a in ("data", "expert") if a in mesh.shape)
        num_devices = 1
        for a in batch_axes:
            num_devices *= mesh.shape[a]
        # Reshard so `s_minus_alpha`'s leading axis spec exactly matches the
        # `in_specs` we hand to `shard_map`. Under explicit-mesh-axes, JAX
        # validates the input spec strictly. Resharding from `P("data", None)`
        # to `P(("data", "expert"), None)` is a no-op when expert size=1, but
        # it's required for the spec match.
        s_minus_alpha = reshard(s_minus_alpha, P(batch_axes, None))
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

        # Concat gate and up on the fly so the optimizer can keep them as
        # separate params; the kernel still sees a fused (E, D, 2I) tensor.
        w_gate_up = jnp.concatenate([self.w_gate, self.w_up], axis=-1)
        routed_flat = moe_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            w_gate_up,
            self.w_down,
            activation=ActivationFunctionEnum.silu,
            mesh=mesh,
        )
        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, Pbatch)
        return routed, router_stats


# ---- Block ----


class Block(eqx.Module):
    norm1: RMSNorm
    attn_gated_norm: GatedNorm | None
    attn: CausalSelfAttention
    norm2: RMSNorm
    mlp_gated_norm: GatedNorm | None
    # `mlp` is the dense feed-forward when `cfg.use_moe=False`, or the routed
    # `MoEMLP` when `cfg.use_moe=True`.
    mlp: MLP | MoEMLP
    # Optional dense "shared expert" that runs in parallel with the routed
    # experts (its output is added to the routed output before the residual).
    # Only present when `cfg.use_moe=True` and `shared_expert_intermediate_dim > 0`.
    shared: MLP | None

    @staticmethod
    def init(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> Block:
        keys = random.split(key, 5)
        attn_key, mlp_key, gn_attn_key, gn_mlp_key, shared_key = keys
        attn_gn = (
            GatedNorm.init(cfg.hidden_dim, cfg.gated_norm_rank, cfg.initializer_std, key=gn_attn_key)
            if cfg.use_gated_norm
            else None
        )
        mlp_gn = (
            GatedNorm.init(cfg.hidden_dim, cfg.gated_norm_rank, cfg.initializer_std, key=gn_mlp_key)
            if cfg.use_gated_norm
            else None
        )
        if cfg.use_moe:
            mlp: MLP | MoEMLP = MoEMLP.init(cfg, key=mlp_key)
            if cfg.shared_expert_intermediate_dim > 0:
                # Shared expert is a regular MLP at `shared_expert_intermediate_dim`.
                # Activation is ``cfg.shared_expert_mlp_type`` if set, else
                # ``cfg.mlp_type``. We override `intermediate_dim` and
                # `mlp_type` via `dataclasses.replace` so `MLP.init` reads the
                # shared-expert values without affecting the per-cfg routed
                # experts (which always use silu via `moe_mlp`).
                shared_mlp_type = cfg.shared_expert_mlp_type or cfg.mlp_type
                shared_cfg = dataclasses.replace(
                    cfg,
                    intermediate_dim=cfg.shared_expert_intermediate_dim,
                    mlp_type=shared_mlp_type,
                )
                shared: MLP | None = MLP.init(shared_cfg, key=shared_key)
            else:
                shared = None
        else:
            mlp = MLP.init(cfg, key=mlp_key)
            shared = None
        return Block(
            norm1=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=attn_gn,
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            norm2=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=mlp_gn,
            mlp=mlp,
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        *,
        use_pko: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array] | None]:
        attn_in = self.norm1(x)
        if self.attn_gated_norm is not None:
            attn_in = self.attn_gated_norm(attn_in)
        x = x + self.attn(attn_in, mask, use_pko=use_pko)
        mlp_in = self.norm2(x)
        if self.mlp_gated_norm is not None:
            mlp_in = self.mlp_gated_norm(mlp_in)
        if isinstance(self.mlp, MoEMLP):
            mlp_out, router_stats = self.mlp(mlp_in)
            if self.shared is not None:
                mlp_out = mlp_out + self.shared(mlp_in)
            return x + mlp_out, router_stats
        return x + self.mlp(mlp_in), None


# ---- Transformer ----


class Transformer(eqx.Module):
    embed: jax.Array
    norm1: RMSNorm  # post-embed norm
    embed_gated_norm: GatedNorm | None
    blocks: tuple[Block, ...]
    norm2: RMSNorm  # final norm
    final_gated_norm: GatedNorm | None
    proj: LinearWithBias  # lm head; "proj" matches the ref's GPT.proj name and gets zero-init
    config: NanoModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> Transformer:
        # Dispatch on init scheme. All schemes return a `Transformer` (same
        # eqx field layout) so train.py is agnostic to which one was used.
        if cfg.init_scheme == "adamh_ref":
            return _init_adamh_ref(cfg, key=key)
        if cfg.init_scheme == "muon_tuned":
            return _init_muon_tuned(cfg, key=key)
        return _init_default(cfg, key=key)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array] | None]:
        """Forward returns ``(hidden, router_metrics)``.

        ``router_metrics`` is ``None`` for dense models (use_moe=False) and a
        dict of per-layer router stats stacked along axis 0 for MoE models.
        Mirrors ``experiments/grug/moe/model.py:Transformer.__call__``.
        """
        if mask is None:
            mask = AttentionMask.causal()
        # Intra-doc attention masking: when `intra_doc_bos_id` is set, derive
        # segment ids from the boundary marker and attach to the mask BEFORE
        # any sliding-window override (`with_sliding_window` preserves
        # segment_ids cleanly). Each marker token shares its segment id with
        # the document it precedes — i.e., a doc's content tokens *can* attend
        # back to the marker at the start of their own document. See
        # `NanoModelConfig.intra_doc_bos_id` for why this lives here rather
        # than in `LmDataConfig.block_cross_document_attention`.
        bos_id = self.config.intra_doc_bos_id
        if bos_id is not None and isinstance(mask, AttentionMask) and mask.segment_ids is None:
            bos_marker = (token_ids == bos_id).astype(jnp.int32)
            seg_ids = jnp.cumsum(bos_marker, axis=-1)
            # Reshard the segment ids to match `Pbatch`. Inside attention the
            # mask broadcasts against scores sharded `("data", "model", q, k)`,
            # while `token_ids` arrives `(("data","expert"), None)` under the
            # MoE p13/p14/p15 launches. Dropping the expert axis (size 1 here)
            # makes the broadcast legal under explicit-mesh-axes.
            seg_ids = reshard(seg_ids, Pbatch)
            mask = mask.with_segment_ids(seg_ids)
        hidden = self.norm1(self.embed.at[token_ids].get(out_sharding=Pbatch))
        if self.embed_gated_norm is not None:
            hidden = self.embed_gated_norm(hidden)

        # Optional 3-short / 1-long sliding-window pattern (moe convention).
        # When `sliding_window` is set, blocks 0, 1, 2, 4, 5, 6, ... use a half-size
        # window; every 4th block (i % 4 == 3) uses full causal. Matches
        # `experiments/grug/moe/model.py:Transformer.__call__`.
        sw = self.config.sliding_window
        moe_router_stats: list[dict[str, jax.Array]] = []
        # PKO fires on the same long-window layers (i % 4 == 3) — matches the
        # `pko_mode == "every_4th"` rule in moe's Transformer.__call__.
        pko_on = self.config.use_pko_every_4th
        if sw is not None and isinstance(mask, AttentionMask):
            short_mask = mask.with_sliding_window(sw // 2)
            long_mask = mask.with_sliding_window(sw)
            for i, block in enumerate(self.blocks):
                is_long = i % 4 == 3
                layer_mask = long_mask if is_long else short_mask
                use_pko = pko_on and is_long
                hidden, router_stats = eqx.filter_checkpoint(block)(hidden, layer_mask, use_pko=use_pko)
                if router_stats is not None:
                    moe_router_stats.append(router_stats)
        else:
            for i, block in enumerate(self.blocks):
                use_pko = pko_on and (i % 4 == 3)
                hidden, router_stats = eqx.filter_checkpoint(block)(hidden, mask, use_pko=use_pko)
                if router_stats is not None:
                    moe_router_stats.append(router_stats)

        hidden = self.norm2(hidden)
        if self.final_gated_norm is not None:
            hidden = self.final_gated_norm(hidden)

        if not moe_router_stats:
            return hidden, None
        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
        }
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden, _ = self(token_ids, mask=mask)
        raw = _add_bias(jnp.einsum("bsd,dv->bsv", hidden, self.proj.weight), self.proj.bias)
        raw = raw.astype(jnp.float32)
        cap = self.config.logit_cap
        if cap is None or cap <= 0:
            return raw
        return _apply_logit_cap(raw, cap, self.config.cap_form)

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
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        """Manual cross-entropy with the logit soft-cap applied before softmax.

        Mirrors `nanogpt_ref.py` exactly: materialize logits, apply
        `cap * x * rsqrt(x^2 + cap^2)`, then `log_softmax` and gather the
        label's log-prob. The full (B, S, V) tensor lives in HBM, which is
        fine at the ref's vocab_size (50304) on a v5p-8.

        For MoE models (use_moe=True), adds ``router_z_loss_coef * mean(rzl_per_layer)``
        as an aux loss. When ``return_router_metrics=True`` (used by train.py to
        plumb QB betas), returns ``(loss, router_metrics)`` with the per-layer
        stats stacked along axis 0; the trainer reads ``qb_beta_per_layer`` from
        this dict to update router biases on the next step.
        """
        hidden, router_metrics = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], jnp.zeros_like(token_ids[:, :1])], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        if self.config.use_fused_ce:
            # Fused path: matches grug/moe exactly. The kernel reshards both
            # `hidden` and `labels` to a common `_batch_axis_spec(hidden)`
            # inside a `shard_map`, computes per-shard CE without ever
            # materializing the full `(B, S, V)` logits tensor, and `pmean`s
            # the loss correctly across the batch axes (no clip-by-norm
            # mismatch). Skips the manual rsqrt soft-cap — the fused kernel
            # only knows the tanh form via `logit_soft_cap`. Set
            # `logit_cap=None` when using this path.
            cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
                hidden,
                self.proj.weight,
                labels,
                weight=loss_weight,
                reduction=reduction,
                logsumexp_weight=logsumexp_weight,
                dtype=loss_dtype,
            )
        else:
            # Manual path: materialize logits, optionally soft-cap, compute CE.
            # Under MoE, the input batch arrives sharded as `P(("data", "expert"))`
            # (set by `train_batch_pspec`), but the model's hidden state is sharded
            # `Pbatch = P("data")`. The manual `take_along_axis` gather below would
            # then see operand `P("data", ...)` vs indices `P(("data","expert"), ...)`
            # and fail with `ShardingTypeError`. We explicitly drop the expert axis
            # from the label/weight specs to match `Pbatch`. With expert size=1
            # this is a no-op at runtime (just a sharding-spec change).
            labels = reshard(labels, Pbatch)
            loss_weight = reshard(loss_weight, Pbatch)

            raw = _add_bias(
                jnp.einsum("bsd,dv->bsv", hidden, self.proj.weight, out_sharding=Pbatch),
                self.proj.bias,
            )
            raw = raw.astype(loss_dtype)
            cap = self.config.logit_cap
            logits = _apply_logit_cap(raw, cap, self.config.cap_form) if cap is not None and cap > 0 else raw

            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_losses = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
            if logsumexp_weight is not None and logsumexp_weight > 0:
                lse = jax.scipy.special.logsumexp(logits, axis=-1)
                token_losses = token_losses + logsumexp_weight * lse**2
            weighted = token_losses * loss_weight
            if reduction == "none":
                cross_entropy_loss = weighted
            elif reduction == "sum":
                cross_entropy_loss = jnp.sum(weighted)
            else:
                denom = jnp.maximum(jnp.sum(loss_weight), 1.0)
                cross_entropy_loss = jnp.sum(weighted) / denom

        # MoE aux loss: router z-loss only (no load-balancing — QB handles balance).
        # Matches `experiments/grug/moe/model.py:next_token_loss`.
        if router_metrics is not None and reduction != "none":
            num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
            rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
            aux_loss = self.config.router_z_loss_coef * rzl
            loss = cross_entropy_loss + aux_loss
        else:
            loss = cross_entropy_loss

        if return_router_metrics:
            metrics: dict[str, jax.Array] = {"train/cross_entropy_loss": cross_entropy_loss}
            if router_metrics is not None:
                metrics.update(router_metrics)
            return loss, metrics
        return loss


# ---- AdamH-ref init variant ----
#
# Mirrors `experiments/grug/nanogpt_adamh_ref.py`'s init: PyTorch nn.Linear's
# default Kaiming-uniform (bound = 1/sqrt(fan_in)) on every linear, with
# per-module multipliers on the residual-side projections so AdamH has
# non-zero matrices to operate on from step 0.
#
#   - q, k, v.weight     : default Kaiming
#   - attn.proj.weight   : default Kaiming x 1.25
#   - mlp.fc.weight      : default Kaiming x 1.5
#   - mlp.proj.weight    : default Kaiming x 3.0
#   - attn.proj.bias, mlp.proj.bias, lm-head proj.bias : zeroed
#   - q/k/v/mlp.fc bias  : Kaiming-uniform with fan_in of the *weight* (matches
#                          PyTorch nn.Linear, where bias bound = 1/sqrt(fan_in_w))
#   - lm-head proj.weight : zeroed (lm head is on AdamW, not AdamH)
#   - embed              : N(0,1) (PyTorch nn.Embedding default)
#   - RMSNorm gains      : ones (default)


def _kaiming_uniform(key: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, multiplier: float = 1.0) -> jax.Array:
    bound = multiplier / math.sqrt(fan_in)
    return random.uniform(key, shape, minval=-bound, maxval=bound)


def _maybe_gated_norm(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> GatedNorm | None:
    """Build a `GatedNorm` if the config has `use_gated_norm=True`, else None."""
    if not cfg.use_gated_norm:
        return None
    return GatedNorm.init(cfg.hidden_dim, cfg.gated_norm_rank, cfg.initializer_std, key=key)


def _init_default(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> Transformer:
    """nanogpt_ref Muon-style init: truncated normal, optional zero-init on proj weights."""
    keys = random.split(key, cfg.num_layers + 4)
    embed_key, proj_key, embed_gn_key, final_gn_key = keys[:4]
    block_keys = keys[4:]
    embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab)
    proj = LinearWithBias.init(
        cfg.hidden_dim,
        cfg.vocab_size,
        cfg.initializer_std,
        key=proj_key,
        weight_pspec=Plm_head,
        zero_init=cfg.zero_init_proj,
        use_bias=cfg.use_bias,
    )
    blocks = tuple(Block.init(cfg, key=bk) for bk in block_keys)
    return Transformer(
        embed=embed,
        norm1=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
        embed_gated_norm=_maybe_gated_norm(cfg, key=embed_gn_key),
        blocks=blocks,
        norm2=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
        final_gated_norm=_maybe_gated_norm(cfg, key=final_gn_key),
        proj=proj,
        config=cfg,
    )


def _init_adamh_ref(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> Transformer:
    """nanogpt_adamh_ref Kaiming-uniform init with per-module multipliers.

    AdamH preserves Frobenius norm, so any matrix it touches must start non-zero;
    the multipliers below tune those starting norms to match the ref's recipe.
    LM head weight + every "proj" bias still get zeroed (lm head is on AdamW).
    """
    dim = cfg.hidden_dim
    ff = cfg.intermediate_dim
    hdim = cfg.attn_hidden_dim

    embed_key, *block_keys = random.split(key, cfg.num_layers + 1)

    # PyTorch nn.Embedding default: N(0, 1).
    embed = reshard(random.normal(embed_key, (cfg.vocab_size, dim)), Pembed_vocab)

    # LM head: zeroed weight, zeroed bias. Routed to AdamW (head group), not AdamH.
    proj = LinearWithBias(
        weight=reshard(jnp.zeros((dim, cfg.vocab_size), dtype=jnp.float32), Plm_head),
        bias=jnp.zeros((cfg.vocab_size,), dtype=jnp.float32),
    )

    blocks: list[Block] = []
    for bk in block_keys:
        keys = random.split(bk, 10)
        # qkv: default Kaiming, fan_in = dim. Bias uses the same fan_in (PyTorch convention).
        q = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[0], (dim, hdim), dim), P("data", "model")),
            bias=_kaiming_uniform(keys[1], (hdim,), dim),
        )
        k_layer = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[2], (dim, hdim), dim), P("data", "model")),
            bias=_kaiming_uniform(keys[3], (hdim,), dim),
        )
        v = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[4], (dim, hdim), dim), P("data", "model")),
            bias=_kaiming_uniform(keys[5], (hdim,), dim),
        )
        # attn.proj: Kaiming x 1.25, bias zeroed.
        attn_proj = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[6], (hdim, dim), hdim, 1.25), P("model", "data")),
            bias=jnp.zeros((dim,), dtype=jnp.float32),
        )
        attn = CausalSelfAttention(q=q, k=k_layer, v=v, proj=attn_proj, attn_gate=None, cfg=cfg)

        # mlp.fc: Kaiming x 1.5, fan_in = dim. Bias uses the same fan_in.
        fc = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[7], (dim, ff), dim, 1.5), P("data", "model")),
            bias=_kaiming_uniform(keys[8], (ff,), dim),
        )
        # mlp.proj: Kaiming x 3.0, fan_in = ff. Bias zeroed (matches "if 'proj' in name: zero_()").
        mlp_proj = LinearWithBias(
            weight=reshard(_kaiming_uniform(keys[9], (ff, dim), ff, 3.0), P("model", "data")),
            bias=jnp.zeros((dim,), dtype=jnp.float32),
        )
        mlp = MLP(fc=fc, gate=None, proj=mlp_proj)

        blocks.append(
            Block(
                norm1=RMSNorm.init(dim, cfg.layer_norm_eps),
                attn_gated_norm=None,
                attn=attn,
                norm2=RMSNorm.init(dim, cfg.layer_norm_eps),
                mlp_gated_norm=None,
                mlp=mlp,
                shared=None,
            )
        )

    return Transformer(
        embed=embed,
        norm1=RMSNorm.init(dim, cfg.layer_norm_eps),
        embed_gated_norm=None,
        blocks=tuple(blocks),
        norm2=RMSNorm.init(dim, cfg.layer_norm_eps),
        final_gated_norm=None,
        proj=proj,
        config=cfg,
    )


def _init_muon_tuned(cfg: NanoModelConfig, *, key: PRNGKeyArray) -> Transformer:
    """muon_tuned init from `experiments/grug/muon_tuned.py`.

    Per-parameter init rules (translated from the torch ref's name-based dispatch):

    - `embed`: ``N(0, 1)`` per element (PyTorch ``nn.Embedding`` default).
    - any ``proj.weight`` (lm head, attn.proj, mlp.proj): zeros.
    - any other ``.weight`` (q, k, v, mlp.fc): ``N(0, sqrt(0.33 / fan_in))``
      (PyTorch ``nn.Linear`` default).
    - any ``.bias``: zeros.
    - RMSNorm gains: ones.

    `fan_in` for our ``LinearWithBias(weight=(in_dim, out_dim))`` is the
    leading dimension `in_dim`, mirroring torch's ``w.size(-1)`` on a
    ``Linear`` weight stored as ``(out_dim, in_dim)``.
    """
    dim = cfg.hidden_dim
    ff = cfg.intermediate_dim
    hdim = cfg.attn_hidden_dim

    keys = random.split(key, cfg.num_layers + 3)
    embed_key, embed_gn_key, final_gn_key = keys[:3]
    block_keys = keys[3:]

    def _bias(shape: tuple[int, ...]) -> jax.Array | None:
        return jnp.zeros(shape, dtype=jnp.float32) if cfg.use_bias else None

    # PyTorch nn.Embedding default: N(0, 1) per element.
    embed = reshard(random.normal(embed_key, (cfg.vocab_size, dim)), Pembed_vocab)

    # LM head: zeros (matches the ref's "if 'proj' in name and endswith 'weight': zero_()").
    proj = LinearWithBias(
        weight=reshard(jnp.zeros((dim, cfg.vocab_size), dtype=jnp.float32), Plm_head),
        bias=_bias((cfg.vocab_size,)),
    )

    # PyTorch's nn.Linear default uses std = sqrt(0.33 / fan_in); for q/k/v/mlp.fc
    # in our model, fan_in = hidden_dim.
    std_dim = math.sqrt(0.33 / dim)
    kv_hdim = cfg.kv_hidden_dim

    blocks: list[Block] = []
    for bk in block_keys:
        # When MoE is enabled, the bespoke per-matrix init recipe below doesn't
        # apply to the routed expert tensors (those live in `MoEMLP` and use
        # truncated-normal init keyed off `cfg.initializer_std`). Fall through
        # to `Block.init`, which handles the MoE case end-to-end.
        if cfg.use_moe:
            blocks.append(Block.init(cfg, key=bk))
            continue

        # Per block: q, k, v, mlp.fc (always non-zero), attn.proj and mlp.proj
        # (zero per the "proj in name" rule), plus optionally attn_gated_norm /
        # mlp_gated_norm and an mlp.gate matrix when SwiGLU is enabled.
        keys = random.split(bk, 7)

        q = LinearWithBias(
            weight=reshard(std_dim * random.normal(keys[0], (dim, hdim)), P("data", "model")),
            bias=_bias((hdim,)),
        )
        k_layer = LinearWithBias(
            weight=reshard(std_dim * random.normal(keys[1], (dim, kv_hdim)), P("data", "model")),
            bias=_bias((kv_hdim,)),
        )
        v = LinearWithBias(
            weight=reshard(std_dim * random.normal(keys[2], (dim, kv_hdim)), P("data", "model")),
            bias=_bias((kv_hdim,)),
        )
        attn_proj = LinearWithBias(
            weight=reshard(jnp.zeros((hdim, dim), dtype=jnp.float32), P("model", "data")),
            bias=_bias((dim,)),
        )
        attn_gate = (
            reshard(jnp.zeros((dim, cfg.num_heads), dtype=jnp.float32), P(None, None)) if cfg.use_attn_gate else None
        )
        attn = CausalSelfAttention(q=q, k=k_layer, v=v, proj=attn_proj, attn_gate=attn_gate, cfg=cfg)

        fc = LinearWithBias(
            weight=reshard(std_dim * random.normal(keys[3], (dim, ff)), P("data", "model")),
            bias=_bias((ff,)),
        )
        # SwiGLU: gate matrix uses the same Kaiming-style init as fc; mlp.proj
        # stays zero per the "proj in name" rule.
        gate = (
            LinearWithBias(
                weight=reshard(std_dim * random.normal(keys[6], (dim, ff)), P("data", "model")),
                bias=_bias((ff,)),
            )
            if cfg.mlp_type == "swiglu"
            else None
        )
        mlp_proj = LinearWithBias(
            weight=reshard(jnp.zeros((ff, dim), dtype=jnp.float32), P("model", "data")),
            bias=_bias((dim,)),
        )
        mlp = MLP(fc=fc, gate=gate, proj=mlp_proj)

        blocks.append(
            Block(
                norm1=RMSNorm.init(dim, cfg.layer_norm_eps),
                attn_gated_norm=(
                    GatedNorm.init(dim, cfg.gated_norm_rank, cfg.initializer_std, key=keys[4])
                    if cfg.use_gated_norm
                    else None
                ),
                attn=attn,
                norm2=RMSNorm.init(dim, cfg.layer_norm_eps),
                mlp_gated_norm=(
                    GatedNorm.init(dim, cfg.gated_norm_rank, cfg.initializer_std, key=keys[5])
                    if cfg.use_gated_norm
                    else None
                ),
                mlp=mlp,
                shared=None,
            )
        )

    return Transformer(
        embed=embed,
        norm1=RMSNorm.init(dim, cfg.layer_norm_eps),
        embed_gated_norm=_maybe_gated_norm(cfg, key=embed_gn_key),
        blocks=tuple(blocks),
        norm2=RMSNorm.init(dim, cfg.layer_norm_eps),
        final_gated_norm=_maybe_gated_norm(cfg, key=final_gn_key),
        proj=proj,
        config=cfg,
    )


def debug_mesh_and_token_pspec(num_devices: int, model_axis_size: int = 1) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if model_axis_size <= 0:
        raise ValueError(f"model_axis_size must be positive, got {model_axis_size}")
    if num_devices % model_axis_size != 0:
        raise ValueError(f"num_devices ({num_devices}) must be divisible by model_axis_size ({model_axis_size})")
    data_axis_size = num_devices // model_axis_size
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(data_axis_size, model_axis_size),
        axis_names=("data", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("data",), None)


__all__ = [
    "MLP",
    "Block",
    "CausalSelfAttention",
    "LinearWithBias",
    "MoEMLP",
    "NanoModelConfig",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
