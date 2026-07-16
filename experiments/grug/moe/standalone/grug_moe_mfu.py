# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal standalone reproduction of issue #6979 row-13 (and Will's d5120/sonic_cute variant).
Self-contained MFU benchmark: imports only the levanter.grug kernels + levanter.optim + haliax/eqx/jax.
No real data -- deterministic synthetic tokens. Regenerate via rav/merge_minimal.py."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np

# Import TransformerEngine's JAX bindings before the Levanter/Haliax imports so TE's ep FFI
# handlers (e.g. te_ep_prepare_ffi) register with XLA before the JAX backend is initialized by
# Levanter's import side effects. Only the nccl_ep backend needs TE; the import is optional
# everywhere else (e.g. Blackwell sonic_cute runs without TE 2.17 installed).
try:
    import transformer_engine.jax
    import transformer_engine.jax.ep  # noqa: F401
except ImportError:
    pass

from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh


class _Stub:
    """Placeholder for names from dropped deps that appear only in unused HF/pipeline code
    (as base classes or module-level values). Never exercised by the benchmark path."""

    def __init__(self, *a, **k):
        pass

    def __class_getitem__(cls, item):
        return cls

    def __call__(self, *a, **k):
        return self


HfConfig = HFCheckpointConverter = _Stub


class Batch(eqx.Module):
    """Minimal replacement for levanter.data.text.GrugLmExample -- the 3 fields train_step reads."""

    tokens: jax.Array
    loss_weight: jax.Array
    attn_mask: object = eqx.field(static=False)


def synthetic_tokens(global_batch: int, seq_len: int, vocab_size: int, step: int) -> np.ndarray:
    """Deterministic synthetic token ids (no real data)."""
    stride = 9973
    base = np.arange(seq_len, dtype=np.int64)
    idx = step * global_batch + np.arange(global_batch, dtype=np.int64)
    return ((base[None, :] + idx[:, None] * stride) % vocab_size).astype(np.int32)


# ==================== inlined adamh.py ====================
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Local copy of AdamH for iteration without modifying Levanter.
# Adapted from levanter.optim.adamh.

from typing import Any, NamedTuple

import chex
import optax
from optax import tree_utils as otu


class ScaleByAdamHState(NamedTuple):
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


def scale_by_adamh(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        return ScaleByAdamHState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params):
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)

        adam_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)

        def _scale_invariant_2d(p, u):
            """Core update for a 2-D (matrix) parameter."""
            p_norm = jnp.linalg.norm(p)
            u_norm = jnp.linalg.norm(u)
            new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
            return new_p / jnp.linalg.norm(new_p) * p_norm - p

        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim <= 2:
                return _scale_invariant_2d(p, u)
            # For higher-rank tensors, vmap the 2-D logic over the leading axis.
            return jax.vmap(_scale_invariant_2d)(p, u)

        adamh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            adam_updates,
            is_leaf=lambda x: x is None,
        )

        return adamh_updates, ScaleByAdamHState(count=count_inc, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


__all__ = ["ScaleByAdamHState", "scale_by_adamh"]
# ==================== inlined model.py ====================
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
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call, tree_checkpoint_name
from haliax.nn import ArrayStacked
from haliax.nn.ragged_dot import ragged_dot

try:
    # QuACK SM100 cutlass grouped SwiGLU GEMM (custom_vjp; shard_map-safe) — much faster than the
    # Pallas Triton ragged_dot on Blackwell. Optional: needs quack-kernels + cutlass-dsl.
    from levanter.grug._moe.sonic_cute import _expert_mlp as _quack_expert_mlp
    from levanter.grug._moe.sonic_cute import _interleave_gate_up as _quack_interleave_gate_up
except ImportError:
    _quack_expert_mlp = None
    _quack_interleave_gate_up = None
from jax import random
from jax.sharding import get_abstract_mesh, reshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
)
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
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
_ROUTING_RENORM_SUM = 2.5

# NCCL_EP (TransformerEngine HybridEP) runtime state, populated by main() when
# --moe-implementation nccl_ep. The concrete (replica_dcn, data, expert, model)
# mesh and the per-rank recv capacity are read inside the jitted model by
# _moe_mlp_nccl_ep; ep_bootstrap must have run once under the same mesh.
_NCCL_EP_MESH: jax.sharding.Mesh | None = None
_NCCL_EP_RECV_CAP: int = 0
_NCCL_EP_RAGGED: bool = False
_NCCL_EP_QUACK: bool = False
GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY = "grugmoe_artifact_schema_version"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION = 1


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/moe requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


RematMode = Literal["recompute_all", "save_moe", "none"]


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window // 2), mask.with_sliding_window(sliding_window)


class GrugMoeHfConfig(HfConfig):
    model_type = GRUG_MOE_MODEL_TYPE


def _hf_config_attr(config: HfConfig, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer.

    Architecture choices (GatedNorm, XSA, QB routing) are hardcoded.
    Only shape/size knobs live here. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int | None = None
    max_seq_len: int = 8192
    sliding_window: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    router_z_loss_coef: float = 0.0
    disable_pko: bool = True
    """When True (default), the every-4th + last 'long' layers skip Partial
    Key Offset (no shift of the second half of K, no doc-start zeroing). Short
    layers never had PKO. Set to False to re-enable PKO on long layers."""
    disable_long_rope: bool = True
    """When True (default), the every-4th + last 'long' layers skip rotary
    embedding entirely (Q and K go into attention un-rotated). Short layers
    still apply half-RoPE. Set to False to keep RoPE on long layers."""
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    use_array_stacked_blocks: bool = False
    """Stack all transformer blocks into a single ``ArrayStacked[Block]`` and run them
    through one ``jax.lax.scan``. Collapses N per-layer subgraphs into one scan body so
    XLA only plans HBM for one iteration's intermediates -- needed at scale where the
    unrolled program OOMs. Requires ``disable_pko=True`` (PKO reads a per-layer flag at
    trace time, which the homogeneous scan body cannot express)."""
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

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
        resolve_moe_implementation(self.moe_implementation)

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def model_type(self) -> type[Transformer]:
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

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> Transformer:
        cfg = self if Vocab.size == self.vocab_size else dataclasses.replace(self, vocab_size=Vocab.size)
        return Transformer.init(cfg, key=key)

    def hf_checkpoint_converter(
        self,
        ref_checkpoint: str | None = None,
    ) -> HFCheckpointConverter[GrugModelConfig]:  # type: ignore[type-var]
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=ref_checkpoint,
            HfConfigClass=GrugMoeHfConfig,
            tokenizer=ref_checkpoint,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> GrugModelConfig:
        rope = RotaryConfig(theta=float(_hf_config_attr(hf_config, ("rope_theta",), 10000.0)))
        return cls(
            vocab_size=int(_hf_config_attr(hf_config, ("vocab_size",))),
            hidden_dim=int(_hf_config_attr(hf_config, ("hidden_dim", "hidden_size"), 2048)),
            intermediate_dim=int(
                _hf_config_attr(hf_config, ("intermediate_dim", "moe_intermediate_size", "intermediate_size"), 5632)
            ),
            shared_expert_intermediate_dim=int(
                _hf_config_attr(
                    hf_config,
                    ("shared_expert_intermediate_dim", "shared_expert_intermediate_size"),
                    5632,
                )
            ),
            num_experts=int(_hf_config_attr(hf_config, ("num_experts", "num_local_experts"), 8)),
            num_experts_per_token=int(_hf_config_attr(hf_config, ("num_experts_per_token", "num_experts_per_tok"), 2)),
            num_layers=int(_hf_config_attr(hf_config, ("num_layers", "num_hidden_layers"), 24)),
            num_heads=int(_hf_config_attr(hf_config, ("num_heads", "num_attention_heads"), 16)),
            num_kv_heads=int(_hf_config_attr(hf_config, ("num_kv_heads", "num_key_value_heads"), 16)),
            head_dim=_hf_config_attr(hf_config, ("head_dim", "attention_head_dim")),
            max_seq_len=int(_hf_config_attr(hf_config, ("max_seq_len", "max_position_embeddings"), 4096)),
            sliding_window=int(_hf_config_attr(hf_config, ("sliding_window",), 4096)),
            layer_norm_eps=float(_hf_config_attr(hf_config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)),
            initializer_std=float(_hf_config_attr(hf_config, ("initializer_std", "initializer_range"), 0.02)),
            qk_mult=float(_hf_config_attr(hf_config, ("qk_mult",), 1.0)),
            rope=rope,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict[str, Any] | None = None) -> GrugMoeHfConfig:
        config = {
            "architectures": [GRUG_MOE_ARCHITECTURE],
            "vocab_size": vocab_size,
            "hidden_dim": self.hidden_dim,
            "hidden_size": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "intermediate_size": self.intermediate_dim,
            "moe_intermediate_size": self.intermediate_dim,
            "shared_expert_intermediate_dim": self.shared_expert_intermediate_dim,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_dim,
            "num_experts": self.num_experts,
            "num_local_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "num_experts_per_tok": self.num_experts_per_token,
            "num_layers": self.num_layers,
            "num_hidden_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_attention_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.inferred_head_dim,
            "max_seq_len": self.max_seq_len,
            "max_position_embeddings": self.max_seq_len,
            "sliding_window": self.sliding_window,
            "layer_norm_eps": self.layer_norm_eps,
            "rms_norm_eps": self.layer_norm_eps,
            "initializer_std": self.initializer_std,
            "initializer_range": self.initializer_std,
            "qk_mult": self.qk_mult,
            "grugmoe_attention_mode": "production",
            GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY: GRUG_MOE_ARTIFACT_SCHEMA_VERSION,
            "rope_theta": self.rope.theta,
            "tie_word_embeddings": False,
        }
        if config_overrides is not None:
            config.update(config_overrides)
        return GrugMoeHfConfig(**config)


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
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> CausalSelfAttention:
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
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        disable_rope: bool = False,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)

        # Shift the second half of K's head_dim back by one position so the
        # query at position i sees K[i] on head_dim[:half] but K[i-1] on
        # head_dim[half:]. Zero the shifted half at document starts so the
        # cross-half look-back does not leak across docs. Runs before the
        # rms_norm on Q/K below.
        if use_pko:
            half = head_dim // 2
            k_stationary = k[..., half:]
            k_shifted = jnp.concatenate([k_stationary[:, :1, :, :], k_stationary[:, :-1, :, :]], axis=1)
            segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
            if segment_ids is None:
                # No segment info (raw-mask or unsegmented eval path): only position 0 is a doc start.
                is_doc_start_seq = jnp.zeros((seq_len,), dtype=bool).at[0].set(True)
                is_doc_start = jnp.broadcast_to(is_doc_start_seq, k_shifted.shape[:2])
            else:
                q_seg = segment_ids[0]
                if q_seg.ndim == 1:
                    is_doc_start_seq = jnp.concatenate([jnp.ones((1,), dtype=bool), q_seg[1:] != q_seg[:-1]])
                    is_doc_start = jnp.broadcast_to(is_doc_start_seq, k_shifted.shape[:2])
                else:
                    is_doc_start = jnp.concatenate(
                        [jnp.ones_like(q_seg[:, :1], dtype=bool), q_seg[:, 1:] != q_seg[:, :-1]],
                        axis=1,
                    )
            k_shifted = jnp.where(is_doc_start[..., None, None], jnp.zeros_like(k_shifted), k_shifted)
            k = jnp.concatenate([k[..., :half], k_shifted], axis=-1)

        q = rms_norm(q)
        k = rms_norm(k)
        # Half-RoPE: apply rotary embedding only to first half of Q/K head_dim
        # (second half is rope-free on every layer). ``disable_rope`` skips
        # the RoPE step entirely on this layer — used to opt long layers out
        # of rotary embedding when ``cfg.disable_long_rope`` is set.
        if not disable_rope:
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                q[..., :half], k[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            q = jnp.concatenate([q_rot, q[..., half:]], axis=-1)
            k = jnp.concatenate([k_rot, k[..., half:]], axis=-1)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # Half-RoPE's slice+concat on the head_dim axis can leave the explicit-mesh
        # propagator with ``model`` annotated on ``head_dim`` rather than
        # ``num_q_heads``; force the canonical TP layout so it matches ``aligned_v``.
        attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
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


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating. Compensates for AdamH's bounded activation norms.
    See https://arxiv.org/abs/2601.22966v1"""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> GatedNorm:
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
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> DenseMLP:
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
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    # Per-layer total assignments = sum of routing_counts over experts (= tokens * k).
    assignments_per_layer = jnp.sum(routing_counts.astype(jnp.float32), axis=-1)
    capacity_overflow_rate = capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "train/router/capacity_overflow_rate_mean": jnp.mean(capacity_overflow_rate),
        "qb_beta_per_layer": router_metrics.get("qb_beta_per_layer"),
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/layer_{i}/capacity_overflow_rate"] = capacity_overflow_rate[i]
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


def _warmup_nccl_ep(mesh, num_procs, t_local, hidden_dim, top_k, num_experts, recv_cap):
    """Force eager, synchronized creation of the EP NCCL communicator.

    Runs one tiny ep_dispatch immediately after ep_bootstrap so ncclCommInitRank
    happens while ranks are still barrier-aligned (see the call site for why the
    lazy trace-time init fails cross-node). Inputs are throwaway; only the side
    effect of building+caching the comm matters.
    """
    import transformer_engine.jax.ep as te_ep  # optional dep: nccl_ep backend only

    # Keep the warmup dispatch TINY so its jit compiles fast: the NCCL bootstrap-root
    # listener spawned by ep_bootstrap (ncclGetUniqueId) times out, so ncclCommInitRank
    # must fire soon after. A large warmup's slow compile lets the root go stale and the
    # cross-node connect is refused. The comm is sized by ep_bootstrap's max_tokens_per_rank
    # regardless of this TL, so 256 tokens still creates the full-size comm train_step reuses.
    tl = min(256, t_local)
    ep3 = P(_BATCH_AXES, None, None)
    r = jax.process_index()
    key = jax.random.PRNGKey(r)

    # recv_cap must equal the comm's bootstrap recv_capacity (a smaller one -> NCCL invalid argument).
    # But summing the full [num_procs, recv_cap, D] recv forces it to materialize replicated and OOMs
    # (~193 GiB at dp=2/capfac=3). Return the tiny token_counts instead: the dispatch still executes
    # (triggering ncclCommInitRank) without gathering recv.
    def _mk(local, shape):
        return jax.make_array_from_process_local_data(NamedSharding(mesh, ep3), np.asarray(local), shape)

    tokens = _mk(jax.random.normal(key, (1, tl, hidden_dim), dtype=jnp.bfloat16), (num_procs, tl, hidden_dim))
    idx = _mk(jax.random.randint(key, (1, tl, top_k), 0, num_experts, dtype=jnp.int32), (num_procs, tl, top_k))
    cw = _mk(jax.random.uniform(key, (1, tl, top_k), dtype=jnp.float32), (num_procs, tl, top_k))
    cfg = te_ep.EpLayerConfig(top_k=top_k, dispatch_output_per_expert_alignment=16)

    @jax.jit
    def _dispatch(idx, tokens, cw):
        _recv, _recv_w, _hm, tc = te_ep.ep_dispatch(cfg, idx, tokens, cw, recv_cap)
        return tc.astype(jnp.float32).sum()

    jax.block_until_ready(_dispatch(idx, tokens, cw))


def _ragged_expert_glu(recv, recv_w, wg, wu, wd, *, mesh, nle, slots, use_quack=False):
    """Ragged per-expert GLU over ACTUAL tokens, replacing the dense-over-capacity einsum.

    ``recv`` is ``[num_procs, recv_pr, D]`` (recv_pr = nle*slots), padded to ``slots`` per local
    expert with real tokens contiguous at the FRONT of each expert's slot block (verified against
    the TE dispatch layout). Per device: compact the padded tokens to a packed buffer, run two
    ``ragged_dot`` grouped matmuls sized by the actual per-expert counts, then scatter back to the
    padded layout ``ep_combine`` expects. Compute scales with real tokens, not padded capacity —
    unlike the dense einsum which pays for every capacity slot.
    """
    recv_pr = nle * slots
    ep3 = P(_BATCH_AXES, None, None)
    ep2 = P(_BATCH_AXES, None)
    kspec = P("expert", None, None, None)

    def _local(recv_l, recv_w_l, wg_l, wu_l, wd_l):
        recv_l = recv_l[0]  # [recv_pr, D]
        recv_w_l = recv_w_l[0]  # [recv_pr]
        wg_l, wu_l, wd_l = wg_l[0], wu_l[0], wd_l[0]  # [nle, D, I] / [nle, I, D]
        d = recv_l.shape[1]
        if use_quack:
            # QuACK SM100 cutlass grouped SwiGLU directly over the PADDED slot blocks — no
            # compaction. Each expert's group is its full `slots` block (cu = slot boundaries);
            # padding rows compute from padded recv and are masked out by recv_w before combine.
            # At the low capacity_factor dp=1/dp=2 use, paying ~capacity_factor x compute on a fast
            # cutlass GEMM is far cheaper than the gather/scatter the packed path needs (~279ms).
            moe_dim = wg_l.shape[-1]
            w13_il = _quack_interleave_gate_up(jnp.concatenate([wg_l, wu_l], axis=-1), moe_dim).astype(recv_l.dtype)
            cu = jnp.arange(0, recv_pr + 1, slots, dtype=jnp.int32)  # [nle+1] fixed slot-block bounds
            gs = jnp.full((nle,), slots, dtype=jnp.int32)  # every group is a full slot block
            out = _quack_expert_mlp(recv_l, w13_il, wd_l.astype(recv_l.dtype), gs, cu)  # [recv_pr, D]
            return out.reshape(1, recv_pr, d)
        # ragged (Triton) path: compact padded -> packed by real counts, then grouped GEMM.
        counts = (recv_w_l != 0).astype(jnp.int32).reshape(nle, slots).sum(axis=1)  # [nle] real tokens/expert
        cum = jnp.cumsum(counts)  # inclusive; cum[-1] = total real
        offsets = cum - counts  # exclusive cumsum: packed start of each expert
        p = jnp.arange(recv_pr)
        e_of_p = jnp.clip(jnp.searchsorted(cum, p, side="right"), 0, nle - 1)
        src = e_of_p * slots + (p - offsets[e_of_p])
        packed = jnp.where((p < cum[-1])[:, None], recv_l[src], 0).astype(recv_l.dtype)
        packed = tree_checkpoint_name(packed, _CHECKPOINT_DISPATCH_INPUT)
        rimpl = os.environ.get("SCALE_RAGGED_IMPL", "auto")  # auto|triton|xla|megablox
        gate = ragged_dot(packed, wg_l.astype(packed.dtype), counts, implementation=rimpl)
        up = ragged_dot(packed, wu_l.astype(packed.dtype), counts, implementation=rimpl)
        hidden = tree_checkpoint_name((jax.nn.silu(gate) * up).astype(packed.dtype), _CHECKPOINT_EXPERT_HIDDEN)
        out = ragged_dot(hidden, wd_l.astype(hidden.dtype), counts, implementation=rimpl)  # [recv_pr, D]
        q = jnp.arange(recv_pr)
        qe, qt = q // slots, q % slots
        padded = jnp.where((qt < counts[qe])[:, None], out[offsets[qe] + qt], 0).astype(recv_l.dtype)
        return padded.reshape(1, recv_pr, d)

    # Per-device compute is fully local (no cross-device dep), so disable the replication check.
    # The kwarg was renamed check_rep -> check_vma across jax versions; pass whichever exists.
    sm_kwargs = {"mesh": mesh, "in_specs": (ep3, ep2, kspec, kspec, kspec), "out_specs": ep3}
    if "check_vma" in inspect.signature(shard_map).parameters:
        sm_kwargs["check_vma"] = False
    elif "check_rep" in inspect.signature(shard_map).parameters:
        sm_kwargs["check_rep"] = False
    expert_out = shard_map(_local, **sm_kwargs)(recv, recv_w, wg, wu, wd)
    return tree_checkpoint_name(expert_out, _CHECKPOINT_DISPATCH_OUTPUT)


def _moe_mlp_nccl_ep(
    x_flat: Float[Array, "T D"],
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    num_experts: int,
) -> jax.Array:
    """Routed-expert MoE via TransformerEngine NCCL_EP (HybridEP) dispatch/combine.

    Multi-controller, one process per GPU. Maps grug's ``expert`` mesh axis to TE's
    EP axis and (``replica_dcn``, ``data``) to TE's DP axis. ``x_flat`` is ``[T, D]``
    batch-sharded over ``(replica_dcn, data, expert)``; it is reshaped to
    ``[num_procs, T_local, D]`` so dispatch axis 0 is the compound (dp, ep) rank.
    Experts use grug's separate-weight GLU (``w_gate``/``w_up`` ``[E, D, I]``,
    ``w_down`` ``[E, I, D]``). ``ep_bootstrap`` must have run once under the active
    mesh + ``MeshResource``. Returns routed output ``[T, D]``.
    """
    import transformer_engine.jax.ep as te_ep  # optional dep: imported only for the nccl_ep backend

    mesh = _NCCL_EP_MESH
    T, D = x_flat.shape
    K = selected_experts.shape[1]
    moe_dim = w_down.shape[1]
    ep_size = mesh.shape["expert"]
    dp_size = mesh.shape["replica_dcn"] * mesh.shape["data"]
    num_procs = dp_size * ep_size
    T_local = T // num_procs
    NLE = num_experts // ep_size
    recv_cap = _NCCL_EP_RECV_CAP

    ep3 = P(_BATCH_AXES, None, None)  # [num_procs, ., .] sharded over (replica_dcn, data, expert)
    ep2 = P(_BATCH_AXES, None)
    kernel_spec = P("expert", None, None, None)  # experts sharded on the ep-rank axis only

    cfg = te_ep.EpLayerConfig(top_k=K, dispatch_output_per_expert_alignment=16)

    tokens = reshard(x_flat.reshape(num_procs, T_local, D), ep3)
    idx = reshard(selected_experts.reshape(num_procs, T_local, K), ep3)
    cw = reshard(combine_weights.reshape(num_procs, T_local, K), ep3)

    recv, recv_w, handle_mem, transport = te_ep.ep_dispatch(cfg, idx, tokens, cw, recv_cap)
    # Tag *every* dispatch output, not just the tokens: under a save_only_these_names remat
    # policy this is what stops the backward from re-running ep_dispatch. handle_mem and
    # transport must be saved too — leaving them untagged makes the recompute re-issue the
    # collective just to rebuild them, and HybridEP's dispatch/combine is a paired protocol
    # over persistent shared buffers, so a second dispatch against the same handle corrupts it.
    recv, recv_w, handle_mem, transport = tree_checkpoint_name(
        (recv, recv_w, handle_mem, transport), _CHECKPOINT_DISPATCH_INPUT
    )
    recv = reshard(recv, ep3)  # [num_procs, recv_pr, D]
    recv_w = reshard(recv_w, ep2)

    recv_pr = recv.shape[1]
    slots = recv_pr // NLE
    wg = reshard(w_gate.reshape(ep_size, NLE, D, moe_dim), kernel_spec)
    wu = reshard(w_up.reshape(ep_size, NLE, D, moe_dim), kernel_spec)
    wd = reshard(w_down.reshape(ep_size, NLE, moe_dim, D), kernel_spec)

    if _NCCL_EP_RAGGED:
        # Ragged per-expert GLU over ACTUAL tokens (packed by count), not padded capacity.
        expert_out = _ragged_expert_glu(recv, recv_w, wg, wu, wd, mesh=mesh, nle=NLE, slots=slots, use_quack=_NCCL_EP_QUACK)
    else:
        # Dense per-expert GLU over padded capacity slots (recv is grouped by local expert).
        # Split the compound (dp,ep) rank axis: dp = (replica_dcn, data), ep = expert, so the
        # einsum batch axis (ep) shares the expert-only sharding of the ep-sharded weights.
        grouped_spec = P(("replica_dcn", "data"), "expert", None, None, None)
        # Splitting the sharded rank axis into (dp, ep) *and* the slot axis into (NLE, slots) at once
        # is ambiguous on an Explicit mesh; pin the output sharding so XLA keeps ep on the expert axis.
        grouped = jax.lax.reshape(recv, (dp_size, ep_size, NLE, slots, D), out_sharding=grouped_spec)
        gate = jnp.einsum("p e n s d, e n d i -> p e n s i", grouped, wg.astype(grouped.dtype))
        up = jnp.einsum("p e n s d, e n d i -> p e n s i", grouped, wu.astype(grouped.dtype))
        hidden = tree_checkpoint_name(reshard(jax.nn.silu(gate) * up, grouped_spec), _CHECKPOINT_EXPERT_HIDDEN)
        out = jnp.einsum("p e n s i, e n i d -> p e n s d", hidden, wd.astype(hidden.dtype))
        # Inverse merge (dp,ep)->rank and (NLE,slots)->recv_pr; pin the compound (dp,ep) sharding.
        expert_out = tree_checkpoint_name(
            jax.lax.reshape(out, (num_procs, recv_pr, D), out_sharding=ep3), _CHECKPOINT_DISPATCH_OUTPUT
        )

    # ep_combine is an unweighted scatter-sum: fold the per-slot combine weight in
    # here and zero the padded slots (recv_w == 0) before the combine.
    mask = (recv_w != 0).astype(jnp.float32)[..., None]
    weighted = (expert_out.astype(jnp.float32) * recv_w[..., None] * mask).astype(expert_out.dtype)
    weighted = reshard(weighted, ep3)

    combined = te_ep.ep_combine(
        cfg,
        handle_mem,
        transport,
        weighted,
        num_local_tokens=(num_procs, T_local),
        out_sharding=(_BATCH_AXES, None, None),
    )
    combined = tree_checkpoint_name(combined, _CHECKPOINT_MOE_OUTPUT)
    return combined.reshape(T, D)


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> MoEMLP:
        k_router, k_expert = random.split(key, 2)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e = cfg.hidden_dim, cfg.num_experts
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
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

        if self.cfg.moe_implementation == "nccl_ep":
            routed_flat = _moe_mlp_nccl_ep(
                x_flat,
                selected_experts.astype(jnp.int32),
                combine_weights,
                self.expert_mlp.w_gate,
                self.expert_mlp.w_up,
                self.expert_mlp.w_down,
                num_experts=self.cfg.num_experts,
            )
            router_stats["capacity_overflow"] = jnp.zeros((), dtype=jnp.float32)
        else:
            routed_flat, dropped_assignments = self.expert_mlp(
                x_flat,
                selected_experts.astype(jnp.int32),
                combine_weights,
                mesh=get_abstract_mesh(),
                report_capacity_overflow=True,
            )
            router_stats["capacity_overflow"] = dropped_assignments.astype(jnp.float32)

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


def _long_layer_schedule(num_layers: int) -> jax.Array:
    """Bool[num_layers] = True for every 4th layer and the last layer (the full-causal
    "long" layers); False elsewhere (sliding-window "short" layers)."""
    idx = jnp.arange(num_layers)
    return ((idx % 4) == 3) | (idx == num_layers - 1)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> Block:
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
        short_mask: AttentionMask | jax.Array,
        long_mask: AttentionMask | jax.Array,
        use_long_mask: Bool[Array, ""] | bool,
        use_pko: bool = False,
        disable_long_rope: bool = False,
        remat_attn: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        # lax.cond so the body has a uniform shape across scan iterations: long layers use
        # the full causal mask (and may PKO / drop RoPE); short layers use the sliding-window
        # mask, never PKO, and always RoPE.
        def _attn_residual(xx: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
            attn_in = self.attn_gated_norm(self.rms_attn(xx))
            attn_out = jax.lax.cond(
                jnp.asarray(use_long_mask, dtype=jnp.bool_),
                lambda _: self.attn(attn_in, long_mask, use_pko=use_pko, disable_rope=disable_long_rope),
                lambda _: self.attn(attn_in, short_mask, use_pko=False, disable_rope=False),
                operand=None,
            )
            return xx + attn_out

        # `remat_attn` checkpoints ONLY the attention residual and leaves the MoE outside any
        # checkpoint. The DeepEP dispatch/combine FFI is `has_side_effect=True`, which
        # jax.checkpoint's partial-eval rejects; keeping the MoE out of the remat'd region is
        # the way to run DeepEP under gradient checkpointing (at the cost of saving MoE acts).
        x = eqx.filter_checkpoint(_attn_residual)(x) if remat_attn else _attn_residual(x)
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
    # Exactly one is populated: ``blocks`` (unrolled per-layer) or ``stacked_blocks``
    # (homogeneous lax.scan), selected by ``cfg.use_array_stacked_blocks``.
    blocks: tuple[Block, ...] | None
    stacked_blocks: ArrayStacked[Block] | None
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(
        cfg_or_vocab: GrugModelConfig | Axis,
        config: GrugModelConfig | None = None,
        *,
        key: PRNGKeyArray,
    ) -> Transformer:
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

        if cfg.use_array_stacked_blocks and not cfg.disable_pko:
            raise ValueError(
                "use_array_stacked_blocks=True requires disable_pko=True because "
                "CausalSelfAttention reads use_pko at trace time (not scan-expressible)."
            )

        keys = random.split(key, cfg.num_layers + 4)
        embed_key, out_key, embed_gn_key, final_gn_key = keys[:4]
        block_keys = keys[4:]
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)

        blocks: tuple[Block, ...] | None
        stacked_blocks: ArrayStacked[Block] | None
        if cfg.use_array_stacked_blocks:
            blocks = None
            stacked_blocks = ArrayStacked.init(cfg.num_layers, Block)(cfg=cfg, key=block_keys)
        else:
            blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
            stacked_blocks = None

        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            stacked_blocks=stacked_blocks,
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

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        def _maybe_ckpt(fn):
            # "none" skips checkpointing entirely (keep all activations, no backward recompute);
            # "recompute_all"/"save_moe" checkpoint the block/layer with the chosen policy.
            return fn if cfg.remat_mode == "none" else eqx.filter_checkpoint(fn, policy=remat_policy)

        # DeepEP's dispatch/combine FFI is has_side_effect=True, which jax.checkpoint's partial-eval
        # rejects. For deepep we therefore checkpoint only the attention residual (inside Block via
        # remat_attn) and keep the MoE outside any checkpoint, rather than remat'ing the whole block.
        moe_outside_remat = cfg.moe_implementation == "deepep" and cfg.remat_mode != "none"

        if self.blocks is not None:
            num_blocks = len(self.blocks)
            moe_router_stats: list[dict[str, jax.Array]] = []
            for i, block in enumerate(self.blocks):
                is_last = i == num_blocks - 1
                is_long = i % 4 == 3 or is_last
                use_pko = is_long and not cfg.disable_pko
                if moe_outside_remat:
                    hidden, router_stats = block(
                        hidden, short_mask, long_mask, is_long, use_pko, cfg.disable_long_rope, remat_attn=True
                    )
                else:
                    hidden, router_stats = _maybe_ckpt(block)(
                        hidden, short_mask, long_mask, is_long, use_pko, cfg.disable_long_rope
                    )
                moe_router_stats.append(router_stats)
            router_metrics = {
                "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
                "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
                "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
                "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
                "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
                "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
            }
        else:
            assert self.stacked_blocks is not None
            # Homogeneous scan: one compiled Block body over the stacked layers; the per-layer
            # short/long mask choice rides in as a Bool[num_layers] scan input.
            mask_schedule = _long_layer_schedule(cfg.num_layers)

            def _scan_layers(
                carry_hidden: Float[Array, "B S D"],
                scan_inputs: tuple[Block, Bool[Array, ""]],
            ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
                layer, layer_use_long_mask = scan_inputs
                if moe_outside_remat:
                    return layer(
                        carry_hidden,
                        short_mask,
                        long_mask,
                        layer_use_long_mask,
                        False,
                        cfg.disable_long_rope,
                        remat_attn=True,
                    )
                return _maybe_ckpt(layer)(
                    carry_hidden, short_mask, long_mask, layer_use_long_mask, False, cfg.disable_long_rope
                )

            hidden, stacked_router_stats = jax.lax.scan(
                _scan_layers, hidden, xs=(self.stacked_blocks.stacked, mask_schedule)
            )
            router_metrics = {
                "routing_entropy_per_layer": stacked_router_stats["routing_entropy"],
                "routing_counts_per_layer": stacked_router_stats["routing_counts"],
                "load_balancing_loss_per_layer": stacked_router_stats["load_balancing_loss"],
                "router_z_loss_per_layer": stacked_router_stats["router_z_loss"],
                "qb_beta_per_layer": stacked_router_stats["qb_beta"],
                "capacity_overflow_per_layer": stacked_router_stats["capacity_overflow"],
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

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return grugmoe_inference_state_dict(self, prefix=prefix)

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


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, ...]:
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


def _with_state_dict_prefix(prefix: str | None, name: str) -> str:
    return name if prefix is None else f"{prefix}.{name}"


def _linear_inference_tensor(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, -1, -2)


def grugmoe_inference_state_dict(model: Transformer, prefix: str | None = None) -> dict[str, jax.Array]:
    tensors: dict[str, jax.Array] = {
        "model.embed_tokens.weight": model.token_embed,
        "model.embed_norm.weight": model.embed_norm.weight,
        "model.embed_gated_norm.down_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_down),
        "model.embed_gated_norm.up_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_up),
        "model.norm.weight": model.final_norm.weight,
        "model.final_gated_norm.down_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_down),
        "model.final_gated_norm.up_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_up),
        "lm_head.weight": _linear_inference_tensor(model.output_proj),
    }

    for layer_index, block in enumerate(model.blocks):
        layer_prefix = f"model.layers.{layer_index}"
        gate, up = jnp.split(block.mlp.expert_mlp.w_gate_up, [model.config.intermediate_dim], axis=-1)
        tensors.update(
            {
                f"{layer_prefix}.input_layernorm.weight": block.rms_attn.weight,
                f"{layer_prefix}.attn_gated_norm.down_proj.weight": _linear_inference_tensor(
                    block.attn_gated_norm.w_down
                ),
                f"{layer_prefix}.attn_gated_norm.up_proj.weight": _linear_inference_tensor(block.attn_gated_norm.w_up),
                f"{layer_prefix}.self_attn.q_proj.weight": _linear_inference_tensor(block.attn.w_q),
                f"{layer_prefix}.self_attn.k_proj.weight": _linear_inference_tensor(block.attn.w_k),
                f"{layer_prefix}.self_attn.v_proj.weight": _linear_inference_tensor(block.attn.w_v),
                f"{layer_prefix}.self_attn.o_proj.weight": _linear_inference_tensor(block.attn.w_o),
                f"{layer_prefix}.self_attn.attn_gate.weight": _linear_inference_tensor(block.attn.attn_gate),
                f"{layer_prefix}.post_attention_layernorm.weight": block.rms_mlp.weight,
                f"{layer_prefix}.mlp_gated_norm.down_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_down),
                f"{layer_prefix}.mlp_gated_norm.up_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_up),
                f"{layer_prefix}.mlp.router.weight": _linear_inference_tensor(block.mlp.router),
                f"{layer_prefix}.mlp.router.bias": block.mlp.router_bias,
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(up),
                f"{layer_prefix}.mlp.experts.down_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_down),
            }
        )
        if block.shared is not None:
            tensors.update(
                {
                    f"{layer_prefix}.shared_expert.gate_proj.weight": _linear_inference_tensor(block.shared.w_gate),
                    f"{layer_prefix}.shared_expert.up_proj.weight": _linear_inference_tensor(block.shared.w_up),
                    f"{layer_prefix}.shared_expert.down_proj.weight": _linear_inference_tensor(block.shared.w_down),
                }
            )

    return {_with_state_dict_prefix(prefix, name): value for name, value in tensors.items()}


__all__ = [
    "GRUG_MOE_ARCHITECTURE",
    "GRUG_MOE_ARTIFACT_SCHEMA_VERSION",
    "GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY",
    "GRUG_MOE_MODEL_TYPE",
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "GrugMoeHfConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "grugmoe_inference_state_dict",
]
# ==================== inlined optimizer.py ====================
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax.numpy as jnp
from levanter.optim.config import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(
        scale_invariant_update,
        params,
        direction_updates,
        is_leaf=lambda x: x is None,
    )


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        muon_updates, next_state = muon_transform.update(updates, state, params)
        muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate)
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate,
      mlp.expert_mlp.w_up, mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            # GatedNorm projections are matrices -> adamh; checked before the rms/norm test
            # (model.py names them *_gated_norm).
            if "gated" in path_lower:
                return "adamh"
            # RMSNorm/LayerNorm gains -> plain Adam (model.py rms_*, model_mid norm_*).
            if "rms" in path_lower or "norm" in path_lower:
                return "adam"
            if "expert_mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer: 3 LR groups (muonh / adamh / adam).

    Three LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all GatedNorms.
      Newton-Schulz orthogonalisation + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate``
      / 1-D norm weights.

    ``max_grad_norm`` defaults to ``None`` here (no clipping) for the 1pct-noclip
    schedule used by the May Recipe baseline.
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "router_bias" in path_lower
                or path_lower.endswith(".attn_gate")
                or ".router" in path_lower
            ):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorm low-rank projections are real matrices -> MuonH. Checked before the
            # norm test since model.py names them *_gated_norm (contains "norm"); model_mid
            # names them gated_*.
            if "gated" in path_lower:
                return "muonh"
            # RMSNorm/LayerNorm gains are per-dimension scales (stack to 2D under scan), not
            # orthogonalizable matrices -> plain Adam. Handles model.py's rms_attn/rms_mlp
            # and model_mid's norm_* naming.
            if "rms" in path_lower or "norm" in path_lower:
                return "adam"
            # Matrices -> MuonH: attention, MoE experts (4D when stacked under scan), shared.
            if hasattr(param, "ndim") and param.ndim in (2, 3, 4):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "scale_with_grug_muonh",
]
# ==================== inlined train.py (run_grug pipeline stripped) ====================
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import levanter.callbacks as callbacks
import levanter.tracker
from jax.sharding import Mesh
from jax.tree_util import register_dataclass
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator

# This file intentionally mirrors `experiments/grug/base/train.py` with
# variant-specific model/loss/FLOP wiring, per the grug copy-first workflow in
# `.agents/skills/change-grug/`.

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrugTrainerConfig:
    """Runtime knobs for grug training."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(use_explicit_mesh_axes=True))
    data_seed: int | None = None
    log_every: int = 1
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    z_loss_weight: float = 1e-4  # Weight on final-logit logsumexp z-loss stabilization term.

    # Grug builds its own compact (replica_dcn, data, expert, model) mesh instead of using
    # the Trainer's logical axis mapping; `data` absorbs whatever these two leave free.
    # Defaults reproduce the historical layout: no expert parallelism and full replication
    # across slices (replica_axis_size=None -> jax.process_count()), i.e. parameters
    # replicated per slice and sharded only over the intra-slice `data` axis. For a model
    # too large to replicate within one slice, set replica_axis_size=1 (FSDP across every
    # slice) and expert_axis_size>1 (expert parallelism over the intra-slice devices).
    expert_axis_size: int = 1
    replica_axis_size: int | None = None
    sharding_dump_path: str | None = None


@dataclass(frozen=True)
class GrugEvalConfig:
    """Perplexity eval settings for grug training."""

    eval_batch_size: int = 512
    steps_per_eval: int | None = 1000
    max_eval_batches: int | None = None
    prefix: str = "eval"
    eval_current: bool = True
    eval_ema: bool = True
    compute_bpb: bool = True


@dataclass(frozen=True)
class GrugRunConfig:
    """Top-level config for grug training."""

    model: GrugModelConfig
    data: LmDataConfig
    resources: ResourceConfig
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # GPU processes per task: > 1 runs one JAX process per GPU (multi-controller)
    # via the iris.runtime.multigpu supervisor instead of one process per node.
    processes_per_task: int = 1


def build_train_dataset(
    data_config: LmDataConfig,
    *,
    max_seq_len: int,
    batch_schedule: BatchSchedule,
    key: PRNGKeyArray,
) -> MixtureDataset[GrugLmExample]:
    pos = Axis("position", max_seq_len)
    mix_key, shuffle_key = jax.random.split(key)
    weights = data_config.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

    initial_batch_size = batch_schedule.batch_size_at_step(0)
    datasets = data_config.train_sets(pos, key=shuffle_key, initial_batch_size=initial_batch_size)
    return MixtureDataset(
        datasets=datasets,
        weights=weights,
        stop_strategy=data_config.stop_strategy,
        key=mix_key,
        block_size=data_config.mixture_block_size,
    )


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def build_train_loader(
    dataset: AsyncDataset[GrugLmExample],
    *,
    batch_schedule: BatchSchedule,
    mesh: Mesh,
) -> DataLoader[GrugLmExample]:
    # DataLoader uses this batch axis mapping to shard batches across the distributed mesh.
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    return DataLoader(
        dataset,
        batch_schedule.schedule,
        mesh=mesh,
        axis_resources={"__BATCH__": _BATCH_AXES},
        batch_axis_name="__BATCH__",
        allow_nondivisible_batch_size=False,
    )


def build_tagged_evaluator(
    *,
    data_config: LmDataConfig,
    max_seq_len: int,
    mesh: Mesh,
    eval_cfg: GrugEvalConfig,
) -> TaggedEvaluator[LmExample | GrugLmExample, Transformer] | None:
    pos = Axis("position", max_seq_len)
    tagged_eval_sets = data_config.tagged_eval_sets(pos)
    if len(tagged_eval_sets) == 0:
        logger.warning("No evaluation datasets provided.")
        return None

    max_examples_per_dataset = None
    if eval_cfg.max_eval_batches is not None:
        max_examples_per_dataset = eval_cfg.max_eval_batches * eval_cfg.eval_batch_size

    tokenizer = data_config.the_tokenizer if eval_cfg.compute_bpb else None
    # `compact_grug_mesh` always carries (replica_dcn, data, expert, model); length-1 axes
    # are kept so we can name "expert" unconditionally.
    eval_axis_mapping = {"batch": _BATCH_AXES}
    eval_batch = Axis("batch", eval_cfg.eval_batch_size)
    eval_array_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))

    def eval_loss_fn(model: Transformer, batch: LmExample | GrugLmExample) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(batch, LmExample):
            batch = grug_lm_example_from_named(batch)
        per_pos_loss = model.next_token_loss(
            batch.tokens,
            batch.loss_weight,
            mask=batch.attn_mask,
            reduction="none",
            logsumexp_weight=None,
        )
        per_pos_loss = jax.sharding.reshard(per_pos_loss, eval_array_sharding)
        per_pos_weight = jax.sharding.reshard(batch.loss_weight, eval_array_sharding)
        per_pos_token_id = jnp.roll(batch.tokens, -1, axis=-1)
        return per_pos_loss, per_pos_weight, per_pos_token_id

    return TaggedEvaluator(
        EvalBatch=eval_batch,
        tagged_eval_sets=tagged_eval_sets,
        loss_fn=eval_loss_fn,
        tokenizer=tokenizer,
        device_mesh=mesh,
        axis_mapping=eval_axis_mapping,
        max_examples_per_dataset=max_examples_per_dataset,
    )


def _compute_flops(
    *,
    model_config: GrugModelConfig,
) -> tuple[float, dict[str, float]]:
    flops_per_token = lm_flops_per_token(
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        shared_intermediate_dim=model_config.shared_expert_intermediate_dim,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        num_heads=model_config.num_heads,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        glu=True,
        num_experts=model_config.num_experts,
        num_shared_experts=1 if model_config.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model_config.num_experts_per_token,
    )
    flops_per_example = 3 * flops_per_token * model_config.max_seq_len

    flops_summary: dict[str, float] = {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }

    return flops_per_example, flops_summary


def _make_mixture_stage_callback(train_dataset: MixtureDataset, batch_schedule: BatchSchedule):
    last_mixture_stage = -1

    def log_mixture_stage(step_info):
        nonlocal last_mixture_stage
        seq_index = batch_schedule.global_data_offset_by_step(step_info.step)
        block_id = seq_index // train_dataset.block_size
        stage = train_dataset._get_stage_for_block(block_id)
        if stage == last_mixture_stage:
            return

        weights = train_dataset.weight_stages[stage][1]
        mixture_log = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
        mixture_log["mixture/stage"] = stage
        levanter.tracker.log(mixture_log, step=step_info.step)
        last_mixture_stage = stage

    return log_mixture_stage


@register_dataclass
@dataclass(frozen=True)
class GrugTrainState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    ema_params: Transformer | None
    pending_qb_betas: jax.Array


def _apply_qb_betas(model: Transformer, qb_betas: jax.Array) -> Transformer:
    """Set router biases from QB betas (computed on previous step)."""
    new_biases = -qb_betas
    new_biases = new_biases - jnp.mean(new_biases, axis=-1, keepdims=True)
    if model.stacked_blocks is not None:
        # Stacked path: router_bias is a single [num_layers, num_experts] leaf.
        return eqx.tree_at(lambda t: t.stacked_blocks.stacked.mlp.router_bias, model, new_biases)
    assert model.blocks is not None
    new_blocks = list(model.blocks)
    for i, block in enumerate(model.blocks):
        if block.mlp is None:
            continue
        new_mlp = eqx.tree_at(lambda m: m.router_bias, block.mlp, new_biases[i])
        new_blocks[i] = eqx.tree_at(lambda b: b.mlp, block, new_mlp)
    return eqx.tree_at(lambda t: t.blocks, model, tuple(new_blocks))


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
) -> GrugTrainState:
    params = mp.cast_to_param(Transformer.init(model_config, key=key))
    if params.blocks is not None:
        num_moe_layers = sum(1 for b in params.blocks if b.mlp is not None)
    else:
        num_moe_layers = model_config.num_layers
    return GrugTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(params),
        ema_params=params if ema_beta is not None else None,
        pending_qb_betas=jnp.zeros((num_moe_layers, model_config.num_experts)),
    )


def _make_train_step(
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    *,
    z_loss_weight: float,
    ema_beta: float | None,
    watch_config: WatchConfig | None = None,
):
    one = jnp.array(1, dtype=jnp.int32)
    z_loss = z_loss_weight if z_loss_weight > 0 else None
    if watch_config is not None:
        if isinstance(watch_config.watch_targets, str):
            watch_targets = tuple(t.strip() for t in watch_config.watch_targets.split(","))
        else:
            watch_targets = tuple(watch_config.watch_targets)
    else:
        watch_targets = ()

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("compute_watch",))
    def train_step(state: GrugTrainState, batch, *, compute_watch: bool = False):
        # Apply pending QB betas to router biases inside JIT (avoids eager
        # host-side TPU kernel launches that can cause SPMD sync issues).
        qb_params = _apply_qb_betas(state.params, state.pending_qb_betas)
        if ema_beta is not None:
            qb_ema_params = _apply_qb_betas(state.ema_params, state.pending_qb_betas)
        else:
            qb_ema_params = None

        def loss_fn(params):
            compute_params = mp.cast_to_compute(params)
            return compute_params.next_token_loss(
                batch.tokens,
                batch.loss_weight,
                mask=batch.attn_mask,
                reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )

        (loss, summarized_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qb_params)
        metrics = {"train/loss": loss, **summarized_metrics}
        updates, opt_state = optimizer.update(grads, state.opt_state, qb_params)
        params = optax.apply_updates(qb_params, updates)

        if ema_beta is None:
            ema_params = None
        else:
            if qb_ema_params is None:
                raise ValueError("ema_params must be initialized when ema_beta is set.")
            ema_params = jax.tree_util.tree_map(
                lambda old, new: ema_beta * old + (1.0 - ema_beta) * new,
                qb_ema_params,
                params,
            )

        watch_stats = None
        if watch_config is not None and compute_watch:
            watch_stats = compute_watch_stats(
                watch_targets=watch_targets,
                include_norms=watch_config.include_norms,
                include_per_parameter_norms=watch_config.include_per_parameter_norms,
                include_histogram=watch_config.include_histograms,
                split_scan_layers=watch_config.split_scan_layers,
                params=qb_params,
                grads=grads,
                updates=updates,
                opt_state=state.opt_state,
                model_tree_type=type(state.params),
            )

        next_state = dataclasses.replace(
            state,
            step=state.step + one,
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=metrics["qb_beta_per_layer"],
        )

        return next_state, metrics, watch_stats

    return train_step


def _run_grug_local(config: GrugRunConfig) -> None:
    """Entry point for the grug template training loop."""
    trainer = config.trainer.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)

    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    optimizer = config.optimizer.build(trainer.num_train_steps)
    watch_config = trainer.watch
    train_step = _make_train_step(
        optimizer,
        trainer.mp,
        z_loss_weight=config.trainer.z_loss_weight,
        ema_beta=config.trainer.ema_beta,
        watch_config=watch_config if watch_config.is_enabled else None,
    )

    data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
    if config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(config.trainer.data_seed)

    # Grug uses raw PartitionSpecs rather than Trainer's logical axis mapping.
    # Keep the mesh compact so the batch pspec derived by `_batch_spec(mesh)` spans slices directly.
    # replica_axis_size=None lets compact_grug_mesh default to jax.process_count() (full
    # cross-slice replication); set it to 1 on GrugTrainerConfig for cross-slice FSDP.
    mesh = compact_grug_mesh(
        expert_axis_size=config.trainer.expert_axis_size,
        replica_axis_size=config.trainer.replica_axis_size,
    )
    with set_mesh(mesh):
        batch_schedule = trainer.batch_schedule

        train_dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=batch_schedule,
            key=data_key,
        )
        train_loader = build_train_loader(
            train_dataset,
            batch_schedule=batch_schedule,
            mesh=mesh,
        )

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=trainer.mp,
                key=model_rng,
                ema_beta=config.trainer.ema_beta,
            )

        state = _init_state(model_key)

        checkpointer = trainer.checkpointer.create(run_id)
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        dump_grug_state_sharding_run_artifact(
            state,
            log_dir=trainer.log_dir,
            run_id=run_id,
            path_override=config.trainer.sharding_dump_path,
        )

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.params)})

        flops_per_example, flops_summary = _compute_flops(model_config=config.model)
        levanter.tracker.log_summary(flops_summary)

        eval_cfg = config.eval
        evaluator = None
        if eval_cfg is not None:
            evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.model.max_seq_len,
                mesh=mesh,
                eval_cfg=eval_cfg,
            )

        profiler_cfg = trainer.profiler
        profiler_num_steps = profiler_cfg.resolve_num_profile_steps(num_train_steps=trainer.num_train_steps)
        profiler_enabled = profiler_cfg.is_enabled and profiler_num_steps > 0

        log_every = max(1, config.trainer.log_every)
        iterator = LoadingTimeTrackerIterator(train_loader.iter_from_step(int(state.step)))

        state_callbacks = StateCallbackRunner[GrugTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.ema_params if s.ema_params is not None else s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        state_callbacks.add_hook(
            callbacks.log_performance_stats(config.model.max_seq_len, batch_schedule, flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=trainer.num_train_steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(trainer.num_train_steps), every=log_every)
        if profiler_enabled:
            state_callbacks.add_hook(
                callbacks.profile(
                    str(trainer.log_dir / run_id / "profiler"),
                    profiler_cfg.start_step,
                    profiler_num_steps,
                    profiler_cfg.perfetto_link,
                ),
                every=1,
            )
        state_callbacks.add_hook(_make_mixture_stage_callback(train_dataset, batch_schedule), every=1)
        if evaluator is not None and eval_cfg is not None:
            interval = eval_cfg.steps_per_eval
            eval_ema = eval_cfg.eval_ema and config.trainer.ema_beta is not None
            if interval is not None and interval > 0 and (eval_cfg.eval_current or eval_ema):
                state_callbacks.add_hook(
                    cb_tagged_evaluate(
                        evaluator,
                        prefix=eval_cfg.prefix,
                        eval_current=eval_cfg.eval_current,
                        eval_ema=eval_ema,
                    ),
                    every=interval,
                )

        last_loss: float | jax.Array = 0.0
        last_step_duration = 0.0

        # Main optimization loop.
        try:
            while int(state.step) < trainer.num_train_steps:
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch = next(iterator)
                step_start = time.perf_counter()
                current_step = int(state.step)
                # grad_watch runs only on its configured interval.
                compute_watch = (
                    watch_config.is_enabled and watch_config.interval > 0 and current_step % watch_config.interval == 0
                )
                state, metrics, watch_stats = train_step(state, batch, compute_watch=compute_watch)
                step = int(state.step) - 1

                jax.block_until_ready(metrics["train/loss"])

                if jnp.isnan(metrics["train/loss"]):
                    logger.error(f"NaN loss at step {int(state.step)}. Stopping training.")
                    break
                duration = time.perf_counter() - step_start
                hook_start = time.perf_counter()
                with jax.profiler.TraceAnnotation("callbacks"):
                    state_callbacks.run(state, loss=metrics["train/loss"], step_duration=duration)
                    last_loss = metrics["train/loss"]
                    last_step_duration = duration
                    levanter.tracker.log({"throughput/hook_time": time.perf_counter() - hook_start}, step=step)
                    levanter.tracker.log({"throughput/loading_time": iterator.this_load_time}, step=step)
                    router_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if (key.startswith("train/router/") or key.startswith("moe_bias/"))
                        and key not in ("train/router/routing_counts_per_layer", "qb_beta_per_layer")
                    }
                    if router_metrics:
                        levanter.tracker.log(router_metrics, step=step)
                    if "train/cross_entropy_loss" in metrics:
                        levanter.tracker.log(
                            {"train/cross_entropy_loss": metrics["train/cross_entropy_loss"]},
                            step=step,
                        )

                    if watch_stats is not None:
                        levanter.tracker.log(watch_stats, step=step)

                if checkpointer is not None:
                    checkpointer.on_step(tree=state, step=int(state.step))
        except BaseException:
            logger.exception(
                "Fatal error in grug training loop; skipping final callbacks/checkpoint to preserve root cause"
            )
            raise
        else:
            # Mirror classic trainer behavior: force callbacks on the last completed step.
            state_callbacks.run(state, loss=last_loss, step_duration=last_step_duration, force=True)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=int(state.step), force=True)
                checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_grug(config: GrugRunConfig) -> None:
    """Dispatch grug training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
    )


__all__ = [
    "GrugEvalConfig",
    "GrugRunConfig",
    "GrugTrainState",
    "GrugTrainerConfig",
    "initial_state",
    "run_grug",
]


# ==================== minimal synthetic-data MFU harness ====================
_B200_BF16_PEAK_FLOPS = 2.25e15
_H100_BF16_PEAK_FLOPS = 9.89e14
_BATCH_AXES = ("replica_dcn", "data", "expert")
_PROFILE_TRACE_STEPS = 1  # steady steps to trace; 1 keeps trace.json.gz untruncated (~1 step = all kernels once)


def _cuda_profiler():
    """Return a cudaProfilerStart/Stop pair, or no-ops when cuda-python is unavailable.

    Pairs with ``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop``
    so the report holds only the steady steps rather than setup and JIT compilation.
    """
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart  # older cuda-python layout
        except ImportError:
            return (lambda: None), (lambda: None)
    return cudart.cudaProfilerStart, cudart.cudaProfilerStop


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=2560)
    p.add_argument("--num-layers", type=int, default=26)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--num-experts-per-token", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--num-kv-heads", type=int, default=0, help="GQA KV-head count; 0 = full MHA (num_kv_heads = num_heads)"
    )
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--moe-implementation", default="sonic")
    p.add_argument("--attention-implementation", default="gpu_fa4_cute")
    p.add_argument("--expert-axis-size", type=int, default=1, help="EP degree; 1 = EP1 (pure FSDP)")
    p.add_argument(
        "--replica-axis-size",
        type=int,
        default=1,
        help="Replica/DDP degree (cross-rack data parallel). data axis (FSDP) = world/(replica*expert). "
        "Set to #racks so one model copy shards within a rack (NVLink FSDP) and gradients all-reduce across racks (IB).",
    )
    p.add_argument(
        "--capacity-factor",
        type=float,
        default=1.25,
        help="NCCL_EP per-expert recv capacity as a multiple of expected balanced load.",
    )
    p.add_argument("--max-num-sms", type=int, default=0, help="NCCL_EP kernel SM budget (0=auto)")
    p.add_argument(
        "--expert-glu",
        choices=["dense", "ragged", "quack"],
        default="dense",
        help="NCCL_EP expert GLU: 'dense' (einsum over padded capacity), 'ragged' (Triton grouped GEMM), or 'quack' (SM100 cutlass grouped GEMM).",
    )
    p.add_argument(
        "--blocks",
        choices=["auto", "stacked", "unrolled"],
        default="auto",
        help="Layer layout: 'stacked' (one lax.scan body, fast compile), 'unrolled' (per-layer "
        "subgraphs), or 'auto' (stacked, except unrolled for nccl_ep's per-layer EpLayerConfig).",
    )
    p.add_argument(
        "--remat-mode",
        default="recompute_all",
        choices=["recompute_all", "save_moe", "none"],
        help="gradient checkpointing: recompute_all (baseline), save_moe, or none (no recompute)",
    )
    p.add_argument("--profile", action="store_true", help="capture an xprof trace over a few steady steps")
    p.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="cudaProfilerStart/Stop around the steady steps, to pair with "
        "`nsys profile --capture-range=cudaProfilerApi` (keeps clone/pip/compile out of the report).",
    )
    p.add_argument("--muon-ns-steps", type=int, default=5, help="MuonH Newton-Schulz iterations (baseline 5)")
    # Manual multi-controller (1 process per GPU) — required for NCCL_EP. Each rank runs this
    # script; jax.distributed coordinates and local_device_ids=[process_id] pins one GPU per rank.
    p.add_argument(
        "--coordinator-address", default=None, help="jax.distributed coordinator host:port (1-proc/GPU multi-controller)"
    )
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--process-id", type=int, default=0)
    return p.parse_args()


def _make_batch(bs, seq_len, vocab, step, mesh):
    tokens = jnp.asarray(synthetic_tokens(bs, seq_len, vocab, step))
    loss_weight = jnp.ones((bs, seq_len), dtype=jnp.float32)
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    segment_ids = jax.device_put(jnp.zeros((bs, seq_len), dtype=jnp.int32), sharding)
    return Batch(
        tokens=jax.device_put(tokens, sharding),
        loss_weight=jax.device_put(loss_weight, sharding),
        attn_mask=AttentionMask.causal(sliding_window=seq_len).with_segment_ids(segment_ids),
    )


def main():
    a = _parse()
    # Manual 1-proc/GPU multi-controller (for NCCL_EP): each rank runs this script with an explicit
    # jax.distributed coordinator; local_device_ids=[process_id] pins exactly one GPU to this rank.
    if a.coordinator_address:
        jax.distributed.initialize(
            coordinator_address=a.coordinator_address,
            num_processes=a.num_processes,
            process_id=a.process_id,
            local_device_ids=[a.process_id],
        )
        print(
            f"MC process_count={jax.process_count()} process_index={jax.process_index()} "
            f"device_count={jax.device_count()} local={jax.local_device_count()}",
            flush=True,
        )
    # Multi-controller JAX under `iris job run --replicas N` (one process per node, 8 local GPUs):
    # task 0 registers the coordinator in the iris endpoint registry, tasks 1..N-1 poll. Must run
    # before any jax backend use. No-op / skipped for a single-task job (and outside iris entirely).
    elif int(os.environ.get("IRIS_NUM_TASKS", "1")) > 1:
        from iris.runtime.jax_init import initialize_jax

        initialize_jax()
        print(
            f"MN process_count={jax.process_count()} process_index={jax.process_index()} "
            f"device_count={jax.device_count()} local={jax.local_device_count()}",
            flush=True,
        )
    jax.config.update("jax_threefry_partitionable", True)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nh = a.hidden_dim // a.head_dim
    n_kv = a.num_kv_heads or nh
    if nh % n_kv != 0:
        raise ValueError(f"num_heads={nh} must be divisible by num_kv_heads={n_kv} for GQA")

    mesh = compact_grug_mesh(expert_axis_size=a.expert_axis_size, replica_axis_size=a.replica_axis_size)

    # NCCL_EP: bootstrap the EP communicator IMMEDIATELY after JAX init, BEFORE building the model.
    # ncclCommInitRank fires at compile time of the first ep_dispatch; run here (right after the
    # init barrier) all ranks reach it tightly synced. Deferred until after model construction,
    # accumulated per-rank skew makes the compile-time init miss NCCL's hardcoded ~63s bootstrap
    # connect window -> cross-node "connection refused". (Bisected 2026-07-16: every component works
    # when the bootstrap is early; only a late bootstrap fails.)
    nccl_ep = a.moe_implementation == "nccl_ep"
    if nccl_ep:
        import transformer_engine.jax.ep as te_ep
        from transformer_engine.jax.ep import ep_bootstrap
        from transformer_engine.jax.sharding import MeshResource, global_shard_guard

        # TE's ep dispatch/combine call with_sharding_constraint to pin output shardings, which
        # asserts (rather than reshards) on grug's Explicit-axis mesh. Route TE's wsc through
        # jax.sharding.reshard so it establishes the sharding on the Explicit mesh instead.
        _te_orig_wsc = te_ep.with_sharding_constraint

        def _wsc_or_reshard(arr, sharding):
            try:
                return _te_orig_wsc(arr, sharding)
            except AssertionError as exc:
                if "Explicit" not in str(exc):
                    raise
                return reshard(arr, sharding)

        te_ep.with_sharding_constraint = _wsc_or_reshard

        global _NCCL_EP_MESH, _NCCL_EP_RECV_CAP, _NCCL_EP_RAGGED, _NCCL_EP_QUACK
        _NCCL_EP_RAGGED = a.expert_glu in ("ragged", "quack")
        _NCCL_EP_QUACK = a.expert_glu == "quack"
        ep_size = mesh.shape["expert"]
        dp_size = mesh.shape["replica_dcn"] * mesh.shape["data"]
        num_procs = dp_size * ep_size
        t_local = (a.batch_size * a.seq_len) // num_procs
        nle = a.num_experts // ep_size
        # Per-expert capacity = capacity_factor * expected tokens/expert, aligned to 16.
        # Expected load is per EP GROUP: each dp replica is an independent ep_size-rank group
        # with its own expert copies, so a given expert only receives from its group's ep_size
        # ranks (ep_size * t_local tokens), NOT all num_procs. Using num_procs oversizes recv_cap
        # by dp_size at multi-rack (dp>1) — the replicated-recv OOM driver at two racks.
        expected_per_expert = ep_size * t_local * a.num_experts_per_token / a.num_experts
        slots_per_expert = max(16, math.ceil(a.capacity_factor * expected_per_expert / 16) * 16)
        _NCCL_EP_MESH = mesh
        _NCCL_EP_RECV_CAP = nle * slots_per_expert
        # Store the MeshResource, not a guard instance: a global_shard_guard is single-use
        # (its generator can be entered once), and it is needed for two separate `with` blocks
        # (bootstrap here + the training loop below). Build a fresh guard at each entry.
        ep_mesh_resource = MeshResource(dp_resource="data", ep_resource="expert")
        print(
            f"NCCL_EP dp={dp_size} ep={ep_size} num_procs={num_procs} T_local={t_local} "
            f"NLE={nle} slots/expert={slots_per_expert} recv_cap={_NCCL_EP_RECV_CAP}",
            flush=True,
        )
        # The EP-group rank must be this process's GLOBAL rank. Under multi-node iris the launcher
        # supplies it via IRIS_MULTIGPU_PROCESS_INDEX (consumed by initialize_jax), not --process-id,
        # so read jax.process_index() rather than a.process_id — otherwise every rank bootstraps as
        # rank 0 and ncclCommInitRank cannot form the bootstrap ring (cross-node connect refused).
        ep_rank = jax.process_index()
        with set_mesh(mesh), global_shard_guard(ep_mesh_resource):
            ep_bootstrap(
                world_size=num_procs,
                rank=ep_rank,
                num_experts=a.num_experts,
                max_tokens_per_rank=t_local,
                recv_capacity_per_rank=_NCCL_EP_RECV_CAP,
                hidden_dim=a.hidden_dim,
                max_num_sms=a.max_num_sms,
            )
            _warmup_nccl_ep(
                mesh, num_procs, t_local, a.hidden_dim, a.num_experts_per_token, a.num_experts, _NCCL_EP_RECV_CAP
            )
            print(f"NCCL_EP warmup dispatch OK rank={ep_rank}", flush=True)

    inter = a.hidden_dim // 2
    model = GrugModelConfig(
        vocab_size=128256,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
        num_heads=nh,
        num_kv_heads=n_kv,
        head_dim=a.head_dim,
        intermediate_dim=inter,
        shared_expert_intermediate_dim=inter,
        num_experts=a.num_experts,
        num_experts_per_token=a.num_experts_per_token,
        max_seq_len=a.seq_len,
        sliding_window=a.seq_len,
        initializer_std=0.5 / (a.hidden_dim**0.5),
        qk_mult=1.3,
        attention_implementation=a.attention_implementation,
        moe_implementation=a.moe_implementation,
        # NCCL_EP defaults to unrolled layers (each gets its own EpLayerConfig); 'stacked' folds
        # them into one lax.scan body (much faster compile) — valid because the ep handle_mem is a
        # per-iteration value, not per-layer static state. Override with --blocks.
        use_array_stacked_blocks=(a.blocks == "stacked" if a.blocks != "auto" else a.moe_implementation != "nccl_ep"),
        disable_pko=True,
        remat_mode=a.remat_mode,
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1, backend_steps=a.muon_ns_steps
    )
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    opt = optimizer.build(a.steps)
    train_step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    peak = a.num_gpus * _B200_BF16_PEAK_FLOPS
    metrics = []
    tps = a.batch_size * a.seq_len

    train_guard = global_shard_guard(ep_mesh_resource) if nccl_ep else contextlib.nullcontext()
    with set_mesh(mesh), train_guard:

        @jax.jit
        def init(rng):
            return initial_state(model, optimizer=opt, mp=mp, key=rng, ema_beta=None)

        state = init(jax.random.PRNGKey(0))
        trace_stop = min(a.warmup_steps + _PROFILE_TRACE_STEPS, a.steps) - 1
        profiling = False
        prof_start, prof_stop = _cuda_profiler()
        cuda_profiling = False
        for step in range(a.steps):
            batch = _make_batch(a.batch_size, a.seq_len, model.vocab_size, step, mesh)
            if a.cuda_profiler_range and step == a.warmup_steps:
                prof_start()  # nsys --capture-range=cudaProfilerApi collects only from here
                cuda_profiling = True
            if a.profile and step == a.warmup_steps:
                jax.profiler.start_trace(str(out / "xprof"))
                profiling = True
            t0 = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                state, sm, _w = train_step(state, batch, compute_watch=False)
                loss = sm["train/loss"]
                jax.block_until_ready(loss)
            dur = time.perf_counter() - t0
            if profiling and step == trace_stop:
                jax.profiler.stop_trace()
                profiling = False
            if cuda_profiling and step == trace_stop:
                prof_stop()
                cuda_profiling = False
            s = int(state.step) - 1
            eps = a.batch_size / dur
            achieved = flops_per_example * eps
            m = {
                "step": s,
                "duration": dur,
                "tokens_per_second": tps / dur,
                "achieved_flops_per_second": achieved,
                "mfu_b200": achieved / peak,
                "mfu_h100_equiv": achieved / (a.num_gpus * _H100_BF16_PEAK_FLOPS),
                "loss": float(loss),
            }
            metrics.append(m)
            print(json.dumps(m, sort_keys=True), flush=True)
    steady = [m for m in metrics if m["step"] >= a.warmup_steps]

    def med(xs):
        return None if not xs else sorted(xs)[len(xs) // 2]

    summary = {
        "args": vars(a),
        "config": {
            **flops_summary,
            "hidden_dim": model.hidden_dim,
            "moe_implementation": model.moe_implementation,
            "num_gpus": a.num_gpus,
        },
        "steady_median_mfu_b200": med([m["mfu_b200"] for m in steady]),
        "steady_median_mfu_h100_equiv": med([m["mfu_h100_equiv"] for m in steady]),
        "steady_median_achieved_tflops": (
            None if not steady else med([m["achieved_flops_per_second"] for m in steady]) / 1e12
        ),
        "steady_median_tokens_per_second": med([m["tokens_per_second"] for m in steady]),
        "steady_median_duration": med([m["duration"] for m in steady]),
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
