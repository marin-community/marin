# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball: a first-class Levanter snapshot of the production 67/2 GrugMoE architecture.

Snowball is a *pinned* architectural snapshot of the June 67B A2B GrugMoE recipe, exposed as a
Levanter ``LmConfig`` / ``LmHeadModel`` so the existing HF BF16 export can be discovered, loaded,
and scored (and eventually served) through the standard Levanter stack.

"Grug" is a family, not a single model: the dense base template and the MoE variant differ, and
within MoE there are genuinely different architectures (attention mode, PKO on/off, long-layer
RoPE on/off, expert count, shared-expert on/off, sliding-window schedule). Snowball pins one
validated recipe -- the June production choices -- and reuses the shared array-first primitives in
``levanter.grug`` verbatim so its forward is numerically identical to the training path. Off-recipe
checkpoints are rejected rather than silently reinterpreted with June semantics.

The forward math here is a faithful copy of ``experiments/grug/moe/model.py`` (which cannot be
imported from ``levanter`` due to the dependency direction). A parity harness on the marin side
guards against drift.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import core, random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
)
from levanter.grug.grug_moe import MoeImplementation, MoEExpertMlp
from levanter.grug.sharding import (
    Pembed_vocab,
    Plm_head,
    _current_mesh,
    _drop_absent_mesh_axes,
    _mesh_axis_size,
    _reshard_for_init,
    unshard,
)
from levanter.layers.attention import AttentionMask as LmHeadAttentionMask
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import AutoConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


# --- Pinned recipe constants (June 67B A2B production choices) ---------------------------------

GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_ATTENTION_MODE = "production"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY = "grugmoe_artifact_schema_version"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION = 1

_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5
_EP_CAPACITY_FACTOR = 1.0
_QK_RMS_NORM_EPS = 1e-6  # q/k rms_norm uses the function default 1e-6, NOT layer_norm_eps
# June 67B qk_mult (YaRN mscale 1.3*(0.1*ln(65536/8192)+1)); used as the field default and the
# from_hf_config fallback so the two never drift. Real grug_moe exports always carry qk_mult.
_DEFAULT_QK_MULT = 1.5703274004183786
_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _bspec(*tail: Any) -> P:
    """Tolerant batch partition spec: grug batch axes + trailing axes, absent axes dropped."""
    mesh = _current_mesh()
    spec = P(_BATCH_AXES, *tail)
    if mesh is None or getattr(mesh, "empty", True):
        return spec
    return _drop_absent_mesh_axes(mesh, spec)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _bspec())


def _hf_attr(config: HfConfig, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


class GrugMoeHfConfig(HfConfig):
    """HF config class for the ``grug_moe`` model type (registered with ``AutoConfig``)."""

    model_type = GRUG_MOE_MODEL_TYPE


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def rms_norm(x: jax.Array, eps: float = _QK_RMS_NORM_EPS) -> jax.Array:
    """Non-parametric RMS norm over the last dimension (used on Q/K, eps=1e-6)."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


# --- Config ------------------------------------------------------------------------------------


@LmConfig.register_subclass("snowball")
@dataclass(frozen=True)
class SnowballConfig(HFCompatConfig):
    """Pinned snapshot of the June 67B A2B GrugMoE architecture.

    Only shape/size knobs are configurable; the architectural switches (QB-routed MoE, GatedNorm,
    XSA, half-RoPE on short layers, NoPE + full-causal on long layers, PKO disabled) are pinned to
    the June recipe. All fields are defaulted so the config is no-arg constructible, which the HF
    discovery path (``HFCheckpointConverter.from_hf``) requires.
    """

    vocab_size: int = 128256
    hidden_dim: int = 2560
    intermediate_dim: int = 1280
    shared_expert_intermediate_dim: int = 2560
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 26
    num_heads: int = 20
    num_kv_heads: int = 5
    head_dim: Optional[int] = 128
    max_seq_len: int = 65536
    sliding_window: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.0098821
    qk_mult: float = _DEFAULT_QK_MULT
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    attention_implementation: Optional[GrugAttentionImplementation] = None
    # Runtime knob, not an architectural switch: selects the MoE dispatch backend (None -> "ring").
    # The June H100 golden was produced with "sonic"; match it for exact-tolerance parity there.
    moe_implementation: Optional[MoeImplementation] = None

    reference_checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim <= 0:
            raise ValueError("snowball requires an always-on shared expert (shared_expert_intermediate_dim > 0)")
        _ = self.inferred_head_dim

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    @property
    def model_type(self) -> Type["SnowballLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return SnowballLMHeadModel

    @property
    def requires_explicit_mesh_axes(self) -> bool:
        # Snowball reshards every leaf and activation with jax.sharding.reshard(out_sharding=...)
        # over raw Grug PartitionSpecs, which only lower under an AxisType.Explicit mesh.
        return True

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "SnowballConfig":
        _assert_snowball_recipe(hf_config)
        rope = RotaryConfig(theta=float(_hf_attr(hf_config, ("rope_theta",), 10000.0)))
        return cls(
            vocab_size=int(_hf_attr(hf_config, ("vocab_size",), 128256)),
            hidden_dim=int(_hf_attr(hf_config, ("hidden_dim", "hidden_size"), 2560)),
            intermediate_dim=int(
                _hf_attr(hf_config, ("intermediate_dim", "moe_intermediate_size", "intermediate_size"), 1280)
            ),
            shared_expert_intermediate_dim=int(
                _hf_attr(hf_config, ("shared_expert_intermediate_dim", "shared_expert_intermediate_size"), 2560)
            ),
            num_experts=int(_hf_attr(hf_config, ("num_experts", "num_local_experts"), 256)),
            num_experts_per_token=int(_hf_attr(hf_config, ("num_experts_per_token", "num_experts_per_tok"), 4)),
            num_layers=int(_hf_attr(hf_config, ("num_layers", "num_hidden_layers"), 26)),
            num_heads=int(_hf_attr(hf_config, ("num_heads", "num_attention_heads"), 20)),
            num_kv_heads=int(_hf_attr(hf_config, ("num_kv_heads", "num_key_value_heads"), 5)),
            head_dim=_hf_attr(hf_config, ("head_dim", "attention_head_dim"), 128),
            max_seq_len=int(_hf_attr(hf_config, ("max_seq_len", "max_position_embeddings"), 65536)),
            sliding_window=int(_hf_attr(hf_config, ("sliding_window",), 2048)),
            layer_norm_eps=float(_hf_attr(hf_config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)),
            initializer_std=float(_hf_attr(hf_config, ("initializer_std", "initializer_range"), 0.0098821)),
            qk_mult=float(_hf_attr(hf_config, ("qk_mult",), _DEFAULT_QK_MULT)),
            rope=rope,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> GrugMoeHfConfig:
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
            "grugmoe_attention_mode": GRUG_MOE_ATTENTION_MODE,
            GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY: GRUG_MOE_ARTIFACT_SCHEMA_VERSION,
            "rope_theta": self.rope.theta,
            "tie_word_embeddings": False,
        }
        if config_overrides is not None:
            config.update(config_overrides)
        return GrugMoeHfConfig(**config)

    def hf_checkpoint_converter(  # pyrefly: ignore[bad-override]
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["SnowballConfig"]:  # type: ignore[type-var]
        ref = self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=ref,
            HfConfigClass=GrugMoeHfConfig,
            tokenizer=self.tokenizer if self.tokenizer is not None else ref,
        )


def _assert_snowball_recipe(hf_config: HfConfig) -> None:
    """Reject checkpoints whose HF-visible architecture is not the pinned June recipe.

    ``disable_pko`` / ``disable_long_rope`` are NOT written to config.json, so absence implies June
    semantics; only explicit contradicting values are rejected. Environment/runtime facts (expert
    mesh axis > 1, MoE impl) are validated at model/mesh build, not here.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type != GRUG_MOE_MODEL_TYPE:
        raise ValueError(f"snowball only loads model_type={GRUG_MOE_MODEL_TYPE!r}, got {model_type!r}")
    mode = getattr(hf_config, "grugmoe_attention_mode", GRUG_MOE_ATTENTION_MODE)
    if mode != GRUG_MOE_ATTENTION_MODE:
        raise ValueError(
            f"snowball pins grugmoe_attention_mode={GRUG_MOE_ATTENTION_MODE!r}, got {mode!r}; "
            "this checkpoint uses a different Grug attention recipe and needs its own named snapshot."
        )
    schema = getattr(hf_config, GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY, GRUG_MOE_ARTIFACT_SCHEMA_VERSION)
    if int(schema) != GRUG_MOE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"snowball supports artifact schema {GRUG_MOE_ARTIFACT_SCHEMA_VERSION}, got {schema}.")
    for flag in ("disable_pko", "disable_long_rope"):
        val = getattr(hf_config, flag, True)
        if val is not True:
            raise ValueError(
                f"snowball pins {flag}=True (June recipe); checkpoint has {flag}={val!r}. "
                "Off-recipe Grug variants need their own named snapshot."
            )


# --- Layers (faithful array-first snapshot of experiments/grug/moe/model.py) --------------------


class RMSNorm(eqx.Module):
    """Parametric RMS norm (eps = layer_norm_eps = 1e-5)."""

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
    """Learnable per-dimension gating (rank-128 low-rank gate on the RMS-normed input)."""

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


class SnowballAttention(eqx.Module):
    """GQA + half-RoPE + XSA + per-head sigmoid gate (pinned production attention).

    ``is_long`` layers (every 4th + last) run full-causal with RoPE disabled (NoPE); short layers
    run sliding-window with half-RoPE. PKO is disabled in the June recipe, so it is not implemented.
    """

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    cfg: SnowballConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return SnowballAttention(
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
        mask: AttentionMask,
        disable_rope: bool = False,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)

        q = rms_norm(q)
        k = rms_norm(k)
        # Half-RoPE: rotary on the first half of head_dim only (second half is rope-free).
        if not disable_rope:
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                q[..., :half], k[..., :half], seq_len=seq_len, head_dim=half, rope=self.cfg.rope
            )
            q = jnp.concatenate([q_rot, q[..., half:]], axis=-1)
            k = jnp.concatenate([k_rot, k[..., half:]], axis=-1)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = _partition_match(aligned_v, attn_out)
        # Exclusive Self-Attention: subtract the component of y parallel to v, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: 2 * sigmoid(x @ attn_gate), one scalar per head.
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:-2], attn_out.shape[-2] * attn_out.shape[-1]),
            out_sharding=_bspec(None, "model"),
        )
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_bspec())


def _partition_match(aligned_v: jax.Array, attn_out: jax.Array) -> jax.Array:
    """Match aligned_v's sharding to attn_out (backend attention can pick its own head sharding)."""
    sharding = jax.typeof(attn_out).sharding if isinstance(attn_out, core.Tracer) else attn_out.sharding
    if isinstance(sharding, NamedSharding):
        return reshard(aligned_v, sharding.spec)
    return reshard(aligned_v, _bspec(None, "model"))


class DenseMLP(eqx.Module):
    """Always-on shared expert (SwiGLU)."""

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
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        out_flat = jnp.einsum("tm,md->td", jax.nn.silu(gate) * up, self.w_down, out_sharding=_bspec())
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


class SnowballMoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights (inference forward only).

    Drops the training-only QB-beta statistics, router metrics, and capacity-overflow reporting;
    only the routed output is needed for scoring/serving. The loaded ``router_bias`` already has the
    QB betas baked in (from the export), so we add it directly and never re-apply the update.
    """

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: SnowballConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballMoEMLP":
        k_router, k_expert = random.split(key, 2)
        expert_axis_size = _mesh_axis_size(_current_mesh(), "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")
        d, e = cfg.hidden_dim, cfg.num_experts
        return SnowballMoEMLP(
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
                capacity_factor=_EP_CAPACITY_FACTOR,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        # router_bias is [E]; replicate it (like the norm weights) so the add keeps the expert axis
        # unsharded. A safetensors load auto-shards [E] over `data` when E % data == 0, which would
        # otherwise make router_logits + router_bias illegally sharded on multi-device meshes.
        biased_logits = router_logits + unshard(self.router_bias)
        # Select top-(K+1) on biased logits; the (K+1)-th is only the QB threshold (unused at inference).
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on UNbiased logits for the selected experts, renormed to sum to 2.5.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = jax.nn.sigmoid(unbiased_topk)
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_f.astype(x.dtype)

        routed_flat = self.expert_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=False,
        )
        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        return _batch_reshard(routed)


class SnowballBlock(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: SnowballAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: SnowballMoEMLP
    shared: DenseMLP

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballBlock":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        return SnowballBlock(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=SnowballAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=SnowballMoEMLP.init(cfg, key=mlp_key),
            shared=DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            ),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        short_mask: AttentionMask,
        long_mask: AttentionMask,
        use_long: Bool[Array, ""],
    ) -> Float[Array, "B S D"]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        # ``lax.cond`` keeps a uniform per-layer body so the transformer can scan the layers:
        # long layers use the full causal mask and disable RoPE (NoPE); short layers use the
        # sliding-window mask and keep RoPE. Both branches are the same shape.
        attn_out = jax.lax.cond(
            jnp.asarray(use_long, dtype=jnp.bool_),
            lambda _: self.attn(attn_in, long_mask, disable_rope=True),
            lambda _: self.attn(attn_in, short_mask, disable_rope=False),
            operand=None,
        )
        x = x + attn_out
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out = self.mlp(mlp_in) + self.shared(mlp_in)
        return x + mlp_out


class SnowballTransformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[SnowballBlock, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: SnowballConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballTransformer":
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = _reshard_for_init(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = _reshard_for_init(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head
        )
        blocks = tuple(SnowballBlock.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return SnowballTransformer(
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
    def __call__(self, token_ids: Int[Array, "B S"], mask: Optional[AttentionMask] = None) -> Float[Array, "B S D"]:
        cfg = self.config
        if mask is None:
            mask = AttentionMask.causal()
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        hidden = self.token_embed.at[token_ids].get(out_sharding=_bspec())
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Scan the layers instead of a Python loop so XLA plans HBM for ONE layer's MoE expert
        # buffers at a time. Unrolling keeps every layer's buffers live simultaneously (temp scales
        # linearly with depth), which OOMs the 67B on 8xH100. The stacked blocks + per-layer long
        # schedule feed a single uniform scan body (June recipe: long layers = every 4th + the last).
        num_blocks = len(self.blocks)
        stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *self.blocks)
        idx = jnp.arange(num_blocks)
        long_schedule = ((idx % 4) == 3) | (idx == num_blocks - 1)

        def _scan_layer(carry: Float[Array, "B S D"], layer_and_flag) -> tuple[Float[Array, "B S D"], None]:
            layer, use_long = layer_and_flag
            return layer(carry, short_mask, long_mask, use_long), None

        hidden, _ = jax.lax.scan(_scan_layer, hidden, (stacked, long_schedule))
        return self.final_gated_norm(self.final_norm(hidden))


# --- LmHeadModel adapter -----------------------------------------------------------------------


class SnowballLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[SnowballConfig]):
    """Levanter ``LmHeadModel`` boundary over the array-first Snowball transformer.

    Normalizes named-axis inputs to raw ``[B, S]`` arrays for the grug forward and wraps the raw
    hidden state back into a ``NamedArray`` so the standard LM head / loss / scoring path works.
    """

    transformer: SnowballTransformer
    _config: SnowballConfig = eqx.field(static=True)

    @property
    def config(self) -> SnowballConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self._config.vocab_size)

    @classmethod
    def init(cls, Vocab: Axis, config: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballLMHeadModel":
        cfg = config if Vocab.size == config.vocab_size else dataclasses.replace(config, vocab_size=Vocab.size)
        return SnowballLMHeadModel(SnowballTransformer.init(cfg, key=key), cfg)

    def activations(  # pyrefly: ignore[bad-override]  # narrows the MoE-optional aux return to a plain NamedArray
        self,
        input_ids: NamedArray,
        attn_mask: Optional[LmHeadAttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        # attn_mask is ignored: the pinned recipe builds its own per-layer short/long causal masks
        # inside the transformer core. Segmented/packed inputs are a follow-up (see plan).
        Pos = input_ids.resolve_axis(self.Pos.name)
        raw = input_ids.array
        lead = raw.shape[:-1]
        s = raw.shape[-1]
        b = int(np.prod(lead)) if lead else 1
        tokens = raw.reshape(b, s)
        hidden = self.transformer(tokens)  # [B, S, D]
        hidden = hidden.reshape(*lead, s, self.Embed.size) if lead else hidden.reshape(s, self.Embed.size)
        out_axes = (*input_ids.axes, self.Embed) if lead else (Pos, self.Embed)
        return hax.named(hidden, out_axes)

    def get_lm_head(self) -> NamedArray:
        return hax.named(self.transformer.output_proj, (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "SnowballLMHeadModel":
        old = self._config.vocab_size
        if new_size == old:
            return self
        te = _resize_axis(self.transformer.token_embed, 0, new_size, self._config.initializer_std, key)
        op = _resize_axis(self.transformer.output_proj, 1, new_size, self._config.initializer_std, key)
        new_cfg = dataclasses.replace(self._config, vocab_size=new_size)
        new_tf = eqx.tree_at(lambda t: (t.token_embed, t.output_proj, t.config), self.transformer, (te, op, new_cfg))
        return SnowballLMHeadModel(new_tf, new_cfg)

    # --- state dict (bidirectional HF serialization) ---
    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        return snowball_to_state_dict(self.transformer, prefix=prefix)

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "SnowballLMHeadModel":
        new_tf = snowball_from_state_dict(self.transformer, state_dict, prefix=prefix)
        return SnowballLMHeadModel(new_tf, self._config)


def _resize_axis(arr: jax.Array, axis: int, new_size: int, std: float, key) -> jax.Array:
    old = arr.shape[axis]
    if new_size == old:
        return arr
    if new_size < old:
        return jax.lax.slice_in_dim(arr, 0, new_size, axis=axis)
    pad_shape = list(arr.shape)
    pad_shape[axis] = new_size - old
    if key is None:
        pad = jnp.zeros(pad_shape, dtype=arr.dtype)
    else:
        pad = (std * random.truncated_normal(key, -3, 3, tuple(pad_shape))).astype(arr.dtype)
    return jnp.concatenate([arr, pad], axis=axis)


# --- State-dict mapping (canonical HF keys; inverse of grugmoe_inference_state_dict) ------------


def _with_prefix(prefix: Optional[str], name: str) -> str:
    return name if prefix is None else f"{prefix}.{name}"


def _T(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, -1, -2)


def snowball_to_state_dict(model: SnowballTransformer, prefix: Optional[str] = None) -> StateDict:
    tensors: dict[str, jax.Array] = {
        "model.embed_tokens.weight": model.token_embed,
        "model.embed_norm.weight": model.embed_norm.weight,
        "model.embed_gated_norm.down_proj.weight": _T(model.embed_gated_norm.w_down),
        "model.embed_gated_norm.up_proj.weight": _T(model.embed_gated_norm.w_up),
        "model.norm.weight": model.final_norm.weight,
        "model.final_gated_norm.down_proj.weight": _T(model.final_gated_norm.w_down),
        "model.final_gated_norm.up_proj.weight": _T(model.final_gated_norm.w_up),
        "lm_head.weight": _T(model.output_proj),
    }
    for i, block in enumerate(model.blocks):
        p = f"model.layers.{i}"
        tensors.update(
            {
                f"{p}.input_layernorm.weight": block.rms_attn.weight,
                f"{p}.attn_gated_norm.down_proj.weight": _T(block.attn_gated_norm.w_down),
                f"{p}.attn_gated_norm.up_proj.weight": _T(block.attn_gated_norm.w_up),
                f"{p}.self_attn.q_proj.weight": _T(block.attn.w_q),
                f"{p}.self_attn.k_proj.weight": _T(block.attn.w_k),
                f"{p}.self_attn.v_proj.weight": _T(block.attn.w_v),
                f"{p}.self_attn.o_proj.weight": _T(block.attn.w_o),
                f"{p}.self_attn.attn_gate.weight": _T(block.attn.attn_gate),
                f"{p}.post_attention_layernorm.weight": block.rms_mlp.weight,
                f"{p}.mlp_gated_norm.down_proj.weight": _T(block.mlp_gated_norm.w_down),
                f"{p}.mlp_gated_norm.up_proj.weight": _T(block.mlp_gated_norm.w_up),
                f"{p}.mlp.router.weight": _T(block.mlp.router),
                f"{p}.mlp.router.bias": block.mlp.router_bias,
                f"{p}.mlp.experts.gate_proj.weight": _T(block.mlp.expert_mlp.w_gate),
                f"{p}.mlp.experts.up_proj.weight": _T(block.mlp.expert_mlp.w_up),
                f"{p}.mlp.experts.down_proj.weight": _T(block.mlp.expert_mlp.w_down),
                f"{p}.shared_expert.gate_proj.weight": _T(block.shared.w_gate),
                f"{p}.shared_expert.up_proj.weight": _T(block.shared.w_up),
                f"{p}.shared_expert.down_proj.weight": _T(block.shared.w_down),
            }
        )
    return {_with_prefix(prefix, name): value for name, value in tensors.items()}


def _get(state_dict: StateDict, prefix: Optional[str], name: str) -> jax.Array:
    return jnp.asarray(state_dict[_with_prefix(prefix, name)])


def snowball_from_state_dict(
    template: SnowballTransformer, state_dict: StateDict, prefix: Optional[str] = None
) -> SnowballTransformer:
    """Populate the template's leaves from canonical HF keys, resharding each to its Grug spec.

    Each leaf must be resharded to its Grug partition spec on load: the generic loader only shards
    NamedArray-keyed axes, so Snowball's raw-array leaves would otherwise load replicated (and the
    67B would not fit).
    """
    g = lambda name: _get(state_dict, prefix, name)  # noqa: E731

    m = template
    m = eqx.tree_at(lambda t: t.token_embed, m, _reshard_for_init(g("model.embed_tokens.weight"), Pembed_vocab))
    m = eqx.tree_at(lambda t: t.embed_norm.weight, m, g("model.embed_norm.weight"))
    m = eqx.tree_at(
        lambda t: t.embed_gated_norm.w_down, m, _reshard_replicated(_T(g("model.embed_gated_norm.down_proj.weight")))
    )
    m = eqx.tree_at(
        lambda t: t.embed_gated_norm.w_up, m, _reshard_replicated(_T(g("model.embed_gated_norm.up_proj.weight")))
    )
    m = eqx.tree_at(lambda t: t.final_norm.weight, m, g("model.norm.weight"))
    m = eqx.tree_at(
        lambda t: t.final_gated_norm.w_down, m, _reshard_replicated(_T(g("model.final_gated_norm.down_proj.weight")))
    )
    m = eqx.tree_at(
        lambda t: t.final_gated_norm.w_up, m, _reshard_replicated(_T(g("model.final_gated_norm.up_proj.weight")))
    )
    m = eqx.tree_at(lambda t: t.output_proj, m, _reshard_for_init(_T(g("lm_head.weight")), Plm_head))

    for i in range(len(m.blocks)):
        p = f"model.layers.{i}"
        m = eqx.tree_at(lambda t, i=i: t.blocks[i].rms_attn.weight, m, g(f"{p}.input_layernorm.weight"))
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn_gated_norm.w_down,
            m,
            _reshard_replicated(_T(g(f"{p}.attn_gated_norm.down_proj.weight"))),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn_gated_norm.w_up,
            m,
            _reshard_replicated(_T(g(f"{p}.attn_gated_norm.up_proj.weight"))),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn.w_q, m, _reshard(_T(g(f"{p}.self_attn.q_proj.weight")), P("data", "model"))
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn.w_k, m, _reshard(_T(g(f"{p}.self_attn.k_proj.weight")), P("data", "model"))
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn.w_v, m, _reshard(_T(g(f"{p}.self_attn.v_proj.weight")), P("data", "model"))
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn.w_o, m, _reshard(_T(g(f"{p}.self_attn.o_proj.weight")), P("model", "data"))
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].attn.attn_gate, m, _reshard_replicated(_T(g(f"{p}.self_attn.attn_gate.weight")))
        )
        m = eqx.tree_at(lambda t, i=i: t.blocks[i].rms_mlp.weight, m, g(f"{p}.post_attention_layernorm.weight"))
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].mlp_gated_norm.w_down,
            m,
            _reshard_replicated(_T(g(f"{p}.mlp_gated_norm.down_proj.weight"))),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].mlp_gated_norm.w_up,
            m,
            _reshard_replicated(_T(g(f"{p}.mlp_gated_norm.up_proj.weight"))),
        )
        m = eqx.tree_at(lambda t, i=i: t.blocks[i].mlp.router, m, _reshard_replicated(_T(g(f"{p}.mlp.router.weight"))))
        m = eqx.tree_at(lambda t, i=i: t.blocks[i].mlp.router_bias, m, g(f"{p}.mlp.router.bias"))
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].mlp.expert_mlp.w_gate,
            m,
            _reshard(_T(g(f"{p}.mlp.experts.gate_proj.weight")), _EXPERT_GATE_UP_SPEC),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].mlp.expert_mlp.w_up,
            m,
            _reshard(_T(g(f"{p}.mlp.experts.up_proj.weight")), _EXPERT_GATE_UP_SPEC),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].mlp.expert_mlp.w_down,
            m,
            _reshard(_T(g(f"{p}.mlp.experts.down_proj.weight")), _EXPERT_DOWN_SPEC),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].shared.w_gate,
            m,
            _reshard(_T(g(f"{p}.shared_expert.gate_proj.weight")), P("data", "model")),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].shared.w_up,
            m,
            _reshard(_T(g(f"{p}.shared_expert.up_proj.weight")), P("data", "model")),
        )
        m = eqx.tree_at(
            lambda t, i=i: t.blocks[i].shared.w_down,
            m,
            _reshard(_T(g(f"{p}.shared_expert.down_proj.weight")), P("model", "data")),
        )
    return m


_EXPERT_GATE_UP_SPEC = P("expert", "data", "model")
_EXPERT_DOWN_SPEC = P("expert", "model", "data")


def _reshard_replicated(arr: jax.Array) -> jax.Array:
    return _reshard_for_init(arr, P(None, None))


def _reshard(arr: jax.Array, spec: P) -> jax.Array:
    return _reshard_for_init(arr, spec)


# Register the HF config so ``AutoConfig.from_pretrained`` resolves ``grug_moe`` without remote code.
AutoConfig.register(GRUG_MOE_MODEL_TYPE, GrugMoeHfConfig, exist_ok=True)


__all__ = [
    "GRUG_MOE_ARCHITECTURE",
    "GRUG_MOE_MODEL_TYPE",
    "GrugMoeHfConfig",
    "SnowballConfig",
    "SnowballLMHeadModel",
    "snowball_from_state_dict",
    "snowball_to_state_dict",
]
