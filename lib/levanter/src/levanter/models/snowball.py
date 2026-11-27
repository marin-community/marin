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
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import math
from typing import Any, Literal, NamedTuple, Optional, Type, cast

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
from haliax import Axis, NamedArray, NamedOrNumeric
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.grug._moe.common import MoeDispatchCounts, _TRANSPORT_CAPACITY_MOE_IMPLEMENTATIONS, _init_weight
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
    ragged_paged_attention,
)
from levanter.grug.attention._paged import (
    FUSED_KV_APPEND_MAX_TOKENS,
    FUSED_KV_PREFILL_MAX_TOKENS,
    RPA_V3_ALIGNMENT,
    PagedAttentionImplementation,
    ragged_paged_attention_append,
)
from levanter.grug.attention.fixed_schedule import (
    FixedScheduleSpec,
    fixed_attention_row_axis,
    fixed_schedule_attention,
    materialize_contiguous_kv,
)
from levanter.grug.grug_moe import MoEExpertMlpPspecs, MoeImplementation, MoEExpertMlp, validate_captured_experts
from levanter.grug.sharding import (
    Pembed_vocab,
    Plm_head,
    _current_mesh,
    _current_sharding,
    _mesh_axis_size,
    unshard,
)
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.kernels.logsumexp import batch_invariant_logsumexp, batch_invariant_selected_logprobs
from levanter.kernels.pallas.fixed_schedule_linear import fixed_schedule_linear

from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.kernels.pallas.rms_norm import Implementation as RmsNormImplementation
from levanter.kernels.pallas.rms_norm import batch_invariant_rms_norm
from levanter.kernels.pallas.rms_norm.reference import MAX_FLATTENED_ROWS
from levanter.kernels.pallas.xsa import xsa_forward
from levanter.layers.attention import AttentionMask as LmHeadAttentionMask
from levanter.layers.kv_cache import KvPageCache, ListCache, StackedKvPageCache
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, split_activations
from levanter.models.loss import next_token_ids, next_token_loss_weight
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

# A published config.json carries exactly one spelling per field. ``from_hf_config`` deliberately
# accepts more spellings than the exporter may write, so the two lists are not interchangeable:
# reads are tolerant (old artifacts keep loading), writes are strict. A config carrying both
# spellings of a field can load as two different models, since backends disagree on which wins.
#
# (published name, config attribute) for every field that has more than one accepted spelling.
GRUG_MOE_CANONICAL_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("hidden_size", "hidden_dim"),
    ("num_hidden_layers", "num_layers"),
    ("num_attention_heads", "num_heads"),
    ("num_key_value_heads", "num_kv_heads"),
    ("max_position_embeddings", "max_seq_len"),
    ("rms_norm_eps", "layer_norm_eps"),
    ("initializer_range", "initializer_std"),
    ("num_experts", "num_experts"),
    ("num_experts_per_tok", "num_experts_per_token"),
    ("moe_intermediate_size", "intermediate_dim"),
    ("shared_expert_intermediate_size", "shared_expert_intermediate_dim"),
)

# Spellings the exporter must never emit, though from_hf_config still reads them.
GRUG_MOE_BANNED_CONFIG_ALIASES = frozenset(
    {
        "attention_head_dim",
        "hidden_dim",
        "intermediate_dim",
        "intermediate_size",
        "shared_expert_intermediate_dim",
        "num_local_experts",
        "num_experts_per_token",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "max_seq_len",
        "layer_norm_eps",
        "initializer_std",
    }
)


def validate_single_name_config(serialized: dict, config: Any) -> None:
    """Check a serialized config.json against the single-name contract.

    Each canonical name must be present with the value held by the corresponding attribute of
    ``config``, and no banned alias may appear. ``config`` may be any object exposing the grug
    attribute names (``SnowballConfig`` and the experiment ``GrugModelConfig`` both do).
    """
    for published, attribute in GRUG_MOE_CANONICAL_CONFIG_FIELDS:
        expected = getattr(config, attribute)
        actual = serialized.get(published)
        if actual != expected:
            raise ValueError(f"config.json {published}={actual!r}, expected {expected!r}")
    leaked = GRUG_MOE_BANNED_CONFIG_ALIASES & serialized.keys()
    if leaked:
        raise ValueError(f"banned config aliases in config.json: {sorted(leaked)}")


_GATED_NORM_RANK = 128
# June recipe: every _LONG_ATTENTION_INTERVAL-th layer and the last layer run full-causal attention.
_CACHE_LAYER_AXIS = "layer"
_LONG_ATTENTION_INTERVAL = 4
_ROUTING_RENORM_SUM = 2.5
_QK_RMS_NORM_EPS = 1e-6  # q/k rms_norm uses the function default 1e-6, NOT layer_norm_eps
# June 67B qk_mult (YaRN mscale 1.3*(0.1*ln(65536/8192)+1)); used as the field default and the
# from_hf_config fallback so the two never drift. Real grug_moe exports always carry qk_mult.
_DEFAULT_QK_MULT = 1.5703274004183786
_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")
_FSDP_AXES: tuple[str, ...] = ("data", "context")


def _context_axis() -> str | None:
    mesh = _current_mesh()
    return "context" if mesh is not None and not mesh.empty and _mesh_axis_size(mesh, "context") > 1 else None


def _activation_spec(*tail: Any) -> P:
    return P(_BATCH_AXES, _context_axis(), *tail)


def _token_spec() -> P:
    context = _context_axis()
    return P((*_BATCH_AXES, context) if context is not None else _BATCH_AXES)


def _reshard_sequence(x: jax.Array, axis: str | None) -> jax.Array:
    sharding = jax.typeof(x).sharding if isinstance(x, core.Tracer) else x.sharding
    if not isinstance(sharding, NamedSharding):
        return x
    spec = sharding.spec
    return reshard(x, P(spec[0], axis, *spec[2:]))


def _hf_attr(config: HfConfig, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


class GrugMoeHfConfig(HfConfig):
    """HF config class for the ``grug_moe`` model type (registered with ``AutoConfig``)."""

    model_type = GRUG_MOE_MODEL_TYPE


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
    projection_implementation: Literal["native", "fixed_schedule"] = dataclasses.field(
        default="native", metadata={"fingerprint_omit_default": True}
    )
    """Use fixed per-row dense projections in both learner and serving paths on TPUv4."""
    replay_precision: Literal["standard", "zero_kl"] = dataclasses.field(
        default="standard", metadata={"fingerprint_omit_default": True}
    )
    shared_attention_implementation: Literal["native", "fixed_schedule"] = dataclasses.field(
        default="native", metadata={"fingerprint_omit_default": True}
    )
    """Use the bounded shared TPUv4 schedule for both paged attention and learner forward."""
    # Runtime knob, not an architectural switch: selects the MoE dispatch backend (None -> "ring").
    # The June H100 golden was produced with "sonic"; match it for exact-tolerance parity there.
    moe_implementation: Optional[MoeImplementation] = None
    router_dot_fp32: bool = False
    """Upcast compute-view router operands before the dot; does not change storage or HF metadata."""
    moe_capacity_factor: float = 1.0
    """Expert-parallel capacity: each expert shard keeps at most ``factor x assignments / shards`` routed
    assignments and silently drops the rest. At the expert-axis size it covers every assignment (dropless). A
    runtime knob like ``router_dot_fp32``: not exported in HF metadata."""
    fp32_activations: bool = False
    """Compute the SiLU and sigmoid activations in FP32 and cast to the input dtype once: GatedNorm, the attention
    head gate, the shared MLP and the routed experts. Off, they run in the compute dtype. A runtime knob like
    ``router_dot_fp32``: not exported in HF metadata."""
    router_combine_fp32: bool = False
    """Keep the routed combine in FP32: FP32 combine weights, weighted expert outputs, per-token sum and
    expert-axis reduction, cast to the compute dtype once at the end. Off, the weights are cast to the compute
    dtype first. A runtime knob like ``router_dot_fp32``: not exported in HF metadata."""

    reference_checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    def __post_init__(self) -> None:
        if self.replay_precision == "zero_kl":
            _validate_fixed_shared_attention_config(self)
            if (
                self.projection_implementation != "fixed_schedule"
                or self.shared_attention_implementation != "fixed_schedule"
                or self.moe_implementation != "ring"
            ):
                raise ValueError("Use SnowballConfig.with_zero_kl_replay() to select the complete replay profile")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if not self.moe_capacity_factor > 0:
            raise ValueError(f"moe_capacity_factor must be positive, got {self.moe_capacity_factor}")
        if self.shared_expert_intermediate_dim <= 0:
            raise ValueError("snowball requires an always-on shared expert (shared_expert_intermediate_dim > 0)")
        if self.fp32_activations and self.moe_implementation == "sonic_cute":
            raise ValueError("fp32_activations cannot use moe_implementation='sonic_cute': its kernel fuses BF16 SiLU")
        if self.router_combine_fp32 and self.moe_implementation in ("sonic", "sonic_cute"):
            raise ValueError(f"router_combine_fp32 cannot use moe_implementation={self.moe_implementation!r}")
        if self.moe_implementation in _TRANSPORT_CAPACITY_MOE_IMPLEMENTATIONS:
            raise ValueError(
                f"snowball cannot use moe_implementation={self.moe_implementation!r}: it requires "
                "pooled_transport_capacity_factor, which SnowballConfig does not set"
            )
        _ = self.inferred_head_dim

    def with_zero_kl_replay(self) -> "SnowballConfig":
        """Select shared TPUv4 arithmetic for the bounded four-slot replay fixture.

        Use FP32 parameters and activations, BF16 serving KV, and highest native
        matmul precision on both sides. Serving must return raw policy scores.
        """
        _validate_fixed_shared_attention_config(self)
        return dataclasses.replace(
            self,
            replay_precision="zero_kl",
            shared_attention_implementation="fixed_schedule",
            projection_implementation="fixed_schedule",
            moe_implementation="ring",
        )

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
        # One name per field: core fields take the universal transformers spelling, MoE fields the
        # most common public spelling, and grug-specific extras keep their bare names. Emitting a
        # second spelling for any of these is what GRUG_MOE_BANNED_CONFIG_ALIASES forbids.
        config = {
            "architectures": [GRUG_MOE_ARCHITECTURE],
            "vocab_size": vocab_size,
            # core — universal transformers names
            "hidden_size": self.hidden_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.inferred_head_dim,
            "max_position_embeddings": self.max_seq_len,
            "sliding_window": self.sliding_window,
            "rms_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_std,
            "rope_theta": self.rope.theta,
            "tie_word_embeddings": False,
            # MoE — most common public spelling per field
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_token,
            "moe_intermediate_size": self.intermediate_dim,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_dim,
            # grug-specific (no public equivalent)
            "qk_mult": self.qk_mult,
            "grugmoe_attention_mode": GRUG_MOE_ATTENTION_MODE,
            GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY: GRUG_MOE_ARTIFACT_SCHEMA_VERSION,
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


def silu_fp32(x: jax.Array) -> jax.Array:
    """Apply SiLU in FP32 and cast once to the input dtype."""
    return jax.nn.silu(x.astype(jnp.float32)).astype(x.dtype)


def sigmoid_fp32(x: jax.Array) -> jax.Array:
    """Sigmoid computed in FP32 and cast to the input dtype once."""
    return jax.nn.sigmoid(x.astype(jnp.float32)).astype(x.dtype)


def _replay_norm(cfg: SnowballConfig) -> RmsNormImplementation | None:
    return "pallas_tpu" if cfg.replay_precision == "zero_kl" else None


def _bounded_rms_norm(x: jax.Array, weight: jax.Array, eps: float, implementation: RmsNormImplementation) -> jax.Array:
    rows = x.size // x.shape[-1]
    if rows <= MAX_FLATTENED_ROWS:
        return batch_invariant_rms_norm(x, weight, eps, implementation)
    flat = x.reshape(rows, x.shape[-1])
    sharding = named_sharding_of(flat)
    chunks = []
    for start in range(0, rows, MAX_FLATTENED_ROWS):
        stop = start + MAX_FLATTENED_ROWS
        chunk = flat.at[start:stop].get(out_sharding=sharding.spec) if sharding is not None else flat[start:stop]
        chunks.append(batch_invariant_rms_norm(chunk, weight, eps, implementation))
    return jnp.concatenate(chunks, axis=0).reshape(x.shape)


def _normalize_qk(x: jax.Array, cfg: SnowballConfig) -> jax.Array:
    implementation = _replay_norm(cfg)
    if implementation is None:
        return rms_norm(x)
    original_sharding = named_sharding_of(x)
    if original_sharding is not None and not original_sharding.mesh.empty:
        row_axes = next(iter(original_sharding.spec), None)
        x = reshard(x, P(row_axes, None, None, None))
    weight = jnp.ones((x.shape[-1],), dtype=jnp.float32)
    output = _bounded_rms_norm(x, weight, _QK_RMS_NORM_EPS, implementation)
    if original_sharding is not None and not original_sharding.mesh.empty:
        return reshard(output, original_sharding)
    return output


class RMSNorm(eqx.Module):
    """Parametric RMS norm (eps = layer_norm_eps = 1e-5)."""

    weight: jax.Array
    eps: float = eqx.field(static=True)
    implementation: RmsNormImplementation | None = eqx.field(static=True, default=None)

    @staticmethod
    def init(dim: int, eps: float, implementation: RmsNormImplementation | None = None) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps, implementation=implementation)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        if self.implementation is not None:
            if self.implementation == "pallas_tpu" and _context_axis() is not None:
                raise ValueError("Batch-invariant TPU RMSNorm requires an unsharded context axis")
            output = _bounded_rms_norm(x, weight, self.eps, self.implementation)
            sharding = named_sharding_of(x)
            return reshard(output, sharding) if sharding is not None and not sharding.mesh.empty else output
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


def _fixed_projection(x: jax.Array, weight: jax.Array) -> jax.Array:
    contraction, output = weight.shape
    flat = x.reshape(-1, contraction)
    sharding = jax.typeof(flat).sharding if isinstance(flat, core.Tracer) else flat.sharding
    if isinstance(sharding, NamedSharding):
        flat = reshard(flat, P(next(iter(sharding.spec), None), None))
        weight = reshard(weight, P())
    projected = fixed_schedule_linear(flat, weight)
    return projected.reshape((*x.shape[:-1], output))


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating (rank-128 low-rank gate on the RMS-normed input)."""

    w_down: jax.Array
    w_up: jax.Array
    fp32_activations: bool = eqx.field(static=True, default=False)
    projection_implementation: Literal["native", "fixed_schedule"] = eqx.field(static=True, default="native")

    @staticmethod
    def init(
        hidden_dim: int,
        initializer_std: float,
        *,
        key: PRNGKeyArray,
        fp32_activations: bool = False,
        projection_implementation: Literal["native", "fixed_schedule"] = "native",
    ) -> "GatedNorm":
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=_init_weight(k_down, (hidden_dim, _GATED_NORM_RANK), initializer_std, P(None, None)),
            w_up=_init_weight(k_up, (_GATED_NORM_RANK, hidden_dim), initializer_std, P(None, None)),
            fp32_activations=fp32_activations,
            projection_implementation=projection_implementation,
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        silu, sigmoid = (silu_fp32, sigmoid_fp32) if self.fp32_activations else (jax.nn.silu, jax.nn.sigmoid)
        if self.projection_implementation == "fixed_schedule":
            gate = sigmoid(_fixed_projection(silu(_fixed_projection(x, self.w_down)), self.w_up))
            return x * _partition_match(gate.astype(x.dtype), x)
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        gate_hidden = silu(gate_hidden)
        gate = sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


def _inference_kv_heads(config: SnowballConfig, kv_axis: str) -> int:
    if kv_axis == "model":
        return config.num_kv_heads
    mesh = _current_mesh()
    if _mesh_axis_size(mesh, "model") != 1:
        raise ValueError("Expert-sharded KV requires model_axis_size=1")
    expert_size = _mesh_axis_size(mesh, "expert")
    return (config.num_kv_heads + expert_size - 1) // expert_size * expert_size


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
            w_q=_init_weight(k_q, (d, n * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_k=_init_weight(k_k, (d, m * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_v=_init_weight(k_v, (d, m * h), cfg.initializer_std, P(_FSDP_AXES, "model")),
            w_o=_init_weight(k_o, (n * h, d), cfg.initializer_std, P("model", _FSDP_AXES)),
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
        q, k, v = self._project_qkv(x, disable_rope=disable_rope)
        if self.cfg.shared_attention_implementation == "fixed_schedule":
            attn_out = _fixed_shared_attention(q, k, v, mask, cfg=self.cfg)
            return self._project_output(x, attn_out, v)
        context = _context_axis()
        local_v = v
        if context is not None:
            q = _reshard_sequence(q, context)
            k = _reshard_sequence(k, None)
            v = _reshard_sequence(v, None)
        implementation = self.cfg.attention_implementation
        if implementation is None and jax.default_backend() == "gpu":
            implementation = "gpu_fa4_cute"
        attn_out = attention(q, k, v, mask, implementation=implementation)
        if context is not None:
            attn_out = _reshard_sequence(attn_out, context)
        return self._project_output(x, attn_out, local_v)

    def _project_qkv(
        self, x: jax.Array, *, disable_rope: bool, position_ids: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        head_dim = self.cfg.inferred_head_dim

        def project(weight):
            if self.cfg.projection_implementation == "fixed_schedule":
                return _fixed_projection(x, weight)
            return jnp.einsum("bsh,hd->bsd", x, weight)

        q = rearrange(project(self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(project(self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(project(self.w_v), "... (m d) -> ... m d", d=head_dim)

        q = _normalize_qk(q, self.cfg)
        k = _normalize_qk(k, self.cfg)
        # Half-RoPE: rotary on the first half of head_dim only (second half is rope-free).
        if not disable_rope:
            half = head_dim // 2
            q_rot, k_rot = apply_rotary_embedding(
                q[..., :half],
                k[..., :half],
                seq_len=x.shape[1],
                head_dim=half,
                rope=self.cfg.rope,
                position_ids=position_ids,
            )
            q_tail, k_tail = q[..., half:], k[..., half:]
            q = jnp.concatenate([_partition_match(q_rot, q_tail), q_tail], axis=-1)
            k = jnp.concatenate([_partition_match(k_rot, k_tail), k_tail], axis=-1)
        return q * self.cfg.qk_mult, k, v

    def _project_output(self, x: jax.Array, attn_out: jax.Array, v: jax.Array) -> jax.Array:
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = _partition_match(aligned_v, attn_out)
        # Exclusive Self-Attention: subtract the component of y parallel to v, per head.
        if self.cfg.replay_precision == "zero_kl":
            attn_out = xsa_forward(attn_out, aligned_v, "pallas_tpu")
        else:
            dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
            v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
            attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: 2 * sigmoid(x @ attn_gate), one scalar per head.
        sigmoid = sigmoid_fp32 if self.cfg.fp32_activations else jax.nn.sigmoid
        if self.cfg.projection_implementation == "fixed_schedule":
            gate_logits = _fixed_projection(x, self.attn_gate)
            gate = _partition_match((2 * sigmoid(gate_logits))[..., None], attn_out)
        else:
            gate = 2 * sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        attn_out = jnp.reshape(
            attn_out,
            (*attn_out.shape[:-2], attn_out.shape[-2] * attn_out.shape[-1]),
            out_sharding=_activation_spec("model"),
        )
        if self.cfg.projection_implementation == "fixed_schedule":
            return _partition_match(_fixed_projection(attn_out, self.w_o), x)
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_activation_spec())

    @named_call
    def decode(
        self,
        x: jax.Array,
        cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: jax.Array,
        *,
        mode: Literal["short", "long"],
        implementation: PagedAttentionImplementation | None = None,
        kv_axis: Literal["model", "expert"] = "model",
        fused_kv_tokens: int = FUSED_KV_APPEND_MAX_TOKENS,
    ) -> tuple[jax.Array, KvPageCache]:
        """Write the new keys and values, then attend over each request's cached prefix."""
        cfg = self.cfg
        head_dim = cfg.inferred_head_dim
        q, k, v = self._project_qkv(x, disable_rope=mode == "long", position_ids=pos_ids[:, None])
        token_axis = batch_info.new_token_dests.axes[0]
        if cfg.shared_attention_implementation == "fixed_schedule":
            kv_axes = (token_axis, Axis("kv_head", cfg.num_kv_heads), Axis("head_size", head_dim))
            cache_sharding = P() if cfg.replay_precision == "zero_kl" else P(None, kv_axis, None)
            cache_k = reshard(k[:, 0], cache_sharding)
            cache_v = reshard(v[:, 0], cache_sharding)
            cache = cache.update(batch_info, hax.named(cache_k, kv_axes), hax.named(cache_v, kv_axes))
            out = _fixed_shared_attention_decode(q, cache, batch_info, pos_ids, use_long=mode == "long", cfg=cfg)
            return self._project_output(x, out, v), cache
        kv_heads = _inference_kv_heads(cfg, kv_axis)
        extra_heads = kv_heads - cfg.num_kv_heads
        kv_axes = (token_axis, Axis("kv_head", kv_heads), Axis("head_size", head_dim))
        cache_k, cache_v = k[:, 0], v[:, 0]
        q = q.reshape(x.shape[0], cfg.num_kv_heads, cfg.num_heads // cfg.num_kv_heads, head_dim)
        if extra_heads:
            cache_k = jnp.pad(cache_k, ((0, 0), (0, extra_heads), (0, 0)))
            cache_v = jnp.pad(cache_v, ((0, 0), (0, extra_heads), (0, 0)))
            q = jnp.pad(q, ((0, 0), (0, extra_heads), (0, 0), (0, 0)))
        cache_k = reshard(cache_k, P(None, kv_axis, None))
        cache_v = reshard(cache_v, P(None, kv_axis, None))
        q = reshard(q, P(None, kv_axis, None, None))
        metadata = (
            batch_info.seq_lens.array,
            batch_info.page_indices.array,
            batch_info.cu_q_lens.array,
            batch_info.num_seqs,
        )
        sliding_window = cfg.sliding_window if mode == "short" else None
        scale = head_dim**-0.5
        if (
            implementation == "tpu_v3"
            and q.shape[0] <= fused_kv_tokens
            and q.dtype == cache.kv_pages.dtype == jnp.bfloat16
            and head_dim == batch_info.page_size == RPA_V3_ALIGNMENT
        ):
            out, pages = ragged_paged_attention_append(
                q,
                cache_k.astype(q.dtype),
                cache_v.astype(q.dtype),
                cache.kv_pages.array,
                *metadata,
                sm_scale=scale,
                sliding_window=sliding_window,
            )
            cache = KvPageCache(hax.named(pages, cache.kv_pages.axes))
        else:
            cache = cache.update(batch_info, hax.named(cache_k, kv_axes), hax.named(cache_v, kv_axes))
            out = ragged_paged_attention(
                q,
                cache.kv_pages.array,
                *metadata,
                sm_scale=scale,
                sliding_window=sliding_window,
                implementation=implementation,
            )
        if kv_axis == "expert":
            out = reshard(out, P(_BATCH_AXES, None, None, None))[:, : cfg.num_kv_heads]
        out = out.reshape(x.shape[0], 1, cfg.num_heads, head_dim)
        return self._project_output(x, out, v), cache


def _fixed_shared_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: AttentionMask, *, cfg: SnowballConfig
) -> jax.Array:
    """Run the bounded four-slot fixed forward with a reference reverse mode."""
    if not mask.is_causal or mask.segment_ids is not None:
        raise ValueError("Snowball fixed shared attention requires an unsegmented causal mask")
    return _fixed_shared_attention_with_vjp(q, k, v, mask, cfg)


def _fixed_shared_attention_forward(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: AttentionMask, *, cfg: SnowballConfig
) -> jax.Array:
    """Run the fixed forward on one complete sequence per row shard."""
    slots, length, _, _ = q.shape
    if length != cfg.max_seq_len:
        raise ValueError("Fixed attention requires learner rows padded to the configured context limit")
    expected_kv_shape = (slots, length, cfg.num_kv_heads, cfg.inferred_head_dim)
    if k.shape != expected_kv_shape or v.shape != expected_kv_shape:
        raise ValueError("K/V leading dimensions must match the fixed attention query geometry")
    pages_per_seq = length // 128
    spec = FixedScheduleSpec(
        max_num_seqs=slots,
        max_kv_len=length,
        num_physical_pages=slots * pages_per_seq,
        pages_per_seq=pages_per_seq,
        num_q_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.inferred_head_dim,
        sliding_window=None if mask.sliding_window is None else min(mask.sliding_window, length),
    )
    page_indices = jnp.arange(spec.num_physical_pages, dtype=jnp.int32).reshape(spec.max_num_seqs, spec.pages_per_seq)
    pages = materialize_contiguous_kv(k, v, page_indices, spec=spec)
    positions = jnp.broadcast_to(jnp.arange(spec.max_kv_len, dtype=jnp.int32), (spec.max_num_seqs, spec.max_kv_len))
    valid = jnp.ones((spec.max_num_seqs, spec.max_kv_len), dtype=jnp.bool_)
    return fixed_schedule_attention(
        q.astype(jnp.float32),
        pages,
        jnp.full((spec.max_num_seqs,), spec.max_kv_len, dtype=jnp.int32),
        page_indices,
        jnp.full((spec.max_num_seqs,), spec.max_kv_len, dtype=jnp.int32),
        positions,
        valid,
        spec=spec,
    )


def _fixed_shared_attention_reference(
    q: jax.Array, k: jax.Array, v: jax.Array, sliding_window: int | None
) -> jax.Array:
    """Evaluate the cast-aware contiguous grouped-query attention used by the VJP."""
    if not get_abstract_mesh().empty:
        # Training can annotate the singleton expert axis on sequence. Q and K
        # form independent sequence dimensions, so contract with batch-only sharding.
        q, k, v = (reshard(value, P(fixed_attention_row_axis(q.shape[0]), None, None, None)) for value in (q, k, v))
    scale = jnp.asarray(1.0 / math.sqrt(q.shape[-1]), dtype=jnp.float32)
    q_compute = q.astype(jnp.float32)
    k_compute = k.astype(jnp.bfloat16)
    v_compute = v.astype(jnp.bfloat16)
    # Keep the BF16 head-reduction adjoint while making dot operands explicitly FP32.
    k_aligned = align_kv_heads(k_compute, num_q_heads=q.shape[2]).astype(jnp.float32)
    v_aligned = align_kv_heads(v_compute, num_q_heads=q.shape[2]).astype(jnp.float32)
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q_compute * scale,
        k_aligned,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    positions = jnp.arange(q.shape[1])
    allowed = positions[None, None, None, :] <= positions[None, None, :, None]
    if sliding_window is not None:
        allowed &= positions[None, None, None, :] > positions[None, None, :, None] - sliding_window
    scores = jnp.where(allowed, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum(
        "bhqk,bkhd->bqhd",
        weights,
        v_aligned,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _fixed_shared_attention_with_vjp(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: AttentionMask, cfg: SnowballConfig
) -> jax.Array:
    return _fixed_shared_attention_forward(q, k, v, mask, cfg=cfg)


def _fixed_shared_attention_with_vjp_fwd(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: AttentionMask, cfg: SnowballConfig
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    return _fixed_shared_attention_forward(q, k, v, mask, cfg=cfg), (q, k, v)


def _fixed_shared_attention_with_vjp_bwd(
    mask: AttentionMask,
    cfg: SnowballConfig,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del cfg
    q, k, v = residuals
    _, pullback = jax.vjp(
        lambda q_, k_, v_: _fixed_shared_attention_reference(q_, k_, v_, mask.sliding_window), q, k, v
    )
    cotangent = cotangent.astype(jnp.float32)
    if not get_abstract_mesh().empty:
        cotangent = reshard(cotangent, P(fixed_attention_row_axis(q.shape[0]), None, None, None))
    dq, dk, dv = pullback(cotangent)
    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


_fixed_shared_attention_with_vjp.defvjp(_fixed_shared_attention_with_vjp_fwd, _fixed_shared_attention_with_vjp_bwd)


def _fixed_shared_attention_decode(
    q: jax.Array,
    cache: KvPageCache,
    batch_info: PageBatchInfo,
    pos_ids: jax.Array,
    *,
    use_long: bool,
    cfg: SnowballConfig,
) -> jax.Array:
    """Run fixed shared attention over cached sequence slots."""
    slots, pages_per_seq = batch_info.page_indices.array.shape
    length = cfg.max_seq_len
    spec = FixedScheduleSpec(
        max_num_seqs=slots,
        max_kv_len=length,
        num_physical_pages=cache.kv_pages.array.shape[0],
        pages_per_seq=pages_per_seq,
        page_size=batch_info.page_size,
        num_q_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.inferred_head_dim,
        sliding_window=None if use_long else min(cfg.sliding_window, length),
    )
    if batch_info.slot_ids.array.shape != (slots,):
        raise ValueError("Fixed attention requires one physical slot ID per sequence")
    metadata = (batch_info.page_indices, batch_info.slot_ids, batch_info.seq_lens, batch_info.cu_q_lens)
    if any(value.array.dtype != jnp.int32 for value in metadata):
        raise ValueError("Snowball fixed shared decode requires int32 page and sequence metadata")
    if batch_info.seq_lens.array.shape != (slots,) or batch_info.cu_q_lens.array.shape != (slots + 1,):
        raise ValueError("Snowball fixed shared decode requires one set of lengths per sequence slot")
    if pos_ids.shape != (q.shape[0],) or pos_ids.dtype != jnp.int32:
        raise ValueError("Snowball fixed shared decode positions must be int32")
    token_rows = jnp.arange(q.shape[0], dtype=jnp.int32)
    starts = batch_info.cu_q_lens.array[:-1]
    used = jnp.arange(spec.max_num_seqs, dtype=jnp.int32) < batch_info.num_seqs
    sequences = jnp.clip(jnp.sum((token_rows[:, None] >= starts[None, :]) & used[None, :], axis=1) - 1, 0, slots - 1)
    safe_positions = jnp.clip(pos_ids, 0, spec.max_kv_len - 1)
    position_valid = (pos_ids >= 0) & (pos_ids < spec.max_kv_len)
    active = token_rows < batch_info.num_new_tokens
    grid = jnp.zeros((slots, length, cfg.num_heads, cfg.inferred_head_dim), dtype=jnp.float32)
    valid = jnp.zeros((slots, length), dtype=jnp.bool_)
    q_update = q[:, 0].astype(jnp.float32)
    q_sharding = jax.typeof(q_update).sharding if isinstance(q_update, core.Tracer) else q_update.sharding
    if not isinstance(q_sharding, NamedSharding):
        raise ValueError("Snowball fixed shared decode requires an Explicit query sharding")
    q_spec = tuple(q_sharding.spec)
    grid_sharding = NamedSharding(q_sharding.mesh, P(q_spec[0], None, q_spec[1], q_spec[2]))
    grid = jax.sharding.reshard(grid, grid_sharding)
    grid = jax.lax.with_sharding_constraint(grid, grid_sharding)
    # The engine pads packed batches with INVALID rows. Drop their writes;
    # clipping them into the grid could overwrite a real query in that slot.
    write_positions = jnp.where(active & position_valid, pos_ids, spec.max_kv_len)
    grid = grid.at[sequences, write_positions].set(q_update, mode="drop", out_sharding=grid_sharding)
    valid = valid.at[sequences, write_positions].set(True, mode="drop")
    counts = jnp.zeros((slots, length), dtype=jnp.int32).at[sequences, write_positions].add(1, mode="drop")
    slot_ids = batch_info.slot_ids.array
    pages = batch_info.page_indices.array
    needed = jnp.arange(spec.pages_per_seq)[None, :] * spec.page_size < batch_info.seq_lens.array[:, None]
    visible_pages_valid = jnp.all(~(used[:, None] & needed) | ((pages >= 0) & (pages < spec.num_physical_pages)))
    # Future pages may remain unallocated until a generated input reaches them.
    # The fixed kernel still prefetches them, so provide an in-range address;
    # its unchanged causal mask excludes these future keys from every query.
    safe_pages = jnp.clip(pages, 0, spec.num_physical_pages - 1)
    metadata_valid = (
        (batch_info.num_seqs >= 0)
        & (batch_info.num_seqs <= spec.max_num_seqs)
        & (batch_info.cu_q_lens.array[0] == 0)
        & (batch_info.num_new_tokens >= 0)
        & (batch_info.num_new_tokens <= q.shape[0])
        & jnp.all(
            ~used
            | (
                (batch_info.cu_q_lens.array[1:] >= batch_info.cu_q_lens.array[:-1])
                & (batch_info.cu_q_lens.array[1:] <= q.shape[0])
            )
        )
        & jnp.all(~used | ((batch_info.seq_lens.array >= 0) & (batch_info.seq_lens.array <= length)))
        & jnp.all(~used | ((slot_ids >= 0) & (slot_ids < slots)))
        & jnp.all(
            (slot_ids[:, None] != slot_ids[None, :])
            | ~used[:, None]
            | ~used[None, :]
            | jnp.eye(slots, dtype=jnp.bool_)
        )
        & visible_pages_valid
        & jnp.all(~active | position_valid)
        & jnp.all(~active | (pos_ids < batch_info.seq_lens.array[jnp.clip(sequences, 0, slots - 1)]))
        & jnp.all(counts <= 1)
    )
    positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (slots, length))
    out = fixed_schedule_attention(
        grid,
        cache.kv_pages.array,
        jnp.full((slots,), length, dtype=jnp.int32),
        safe_pages,
        jnp.full((slots,), length, dtype=jnp.int32),
        positions,
        valid,
        spec=spec,
    )
    out_sharding = NamedSharding(q_sharding.mesh, P(q_spec[0], q_spec[1], q_spec[2]))
    out = out.at[sequences, safe_positions].get(out_sharding=out_sharding)[:, None, ...]
    return jnp.where(metadata_valid, jnp.where(active[:, None, None, None], out, 0), jnp.nan)


def _validate_fixed_shared_attention_config(cfg: SnowballConfig) -> None:
    """Validate the static head and page geometry used by fixed attention."""
    if cfg.inferred_head_dim != 128 or cfg.max_seq_len < 128 or cfg.max_seq_len % 128:
        raise ValueError("Snowball fixed attention requires width 128 and complete 128-token pages")


def _validate_fixed_shared_attention_public_mask(mask: AttentionMask) -> None:
    if not mask.is_causal or mask.sliding_window is not None or mask.segment_ids is not None:
        raise ValueError("Snowball fixed shared attention requires an unsegmented causal mask")


def _partition_match(aligned_v: jax.Array, attn_out: jax.Array) -> jax.Array:
    """Match aligned_v's sharding to attn_out (backend attention can pick its own head sharding)."""
    sharding = jax.typeof(attn_out).sharding if isinstance(attn_out, core.Tracer) else attn_out.sharding
    if isinstance(sharding, NamedSharding):
        return reshard(aligned_v, sharding.spec)
    return reshard(aligned_v, _activation_spec(None, "model"))


class DenseMLP(eqx.Module):
    """Always-on shared expert (SwiGLU)."""

    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    fp32_activations: bool = eqx.field(static=True, default=False)
    projection_implementation: Literal["native", "fixed_schedule"] = eqx.field(static=True, default="native")

    @staticmethod
    def init(
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
        *,
        key: PRNGKeyArray,
        fp32_activations: bool = False,
        projection_implementation: Literal["native", "fixed_schedule"] = "native",
    ) -> "DenseMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        return DenseMLP(
            w_gate=_init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std, P(_FSDP_AXES, "model")),
            w_up=_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std, P(_FSDP_AXES, "model")),
            w_down=_init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std, P("model", _FSDP_AXES)),
            fp32_activations=fp32_activations,
            projection_implementation=projection_implementation,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        silu = silu_fp32 if self.fp32_activations else jax.nn.silu
        if self.projection_implementation == "fixed_schedule":
            gate = _fixed_projection(x, self.w_gate)
            up = _fixed_projection(x, self.w_up)
            return _partition_match(_fixed_projection(silu(gate) * up, self.w_down), x)
        b, s, _ = x.shape
        x_flat = jnp.reshape(x, (b * s, x.shape[-1]), out_sharding=_token_spec())
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        out_flat = jnp.einsum("tm,md->td", silu(gate) * up, self.w_down, out_sharding=_token_spec())
        return jax.lax.reshape(out_flat, (b, s, out_flat.shape[-1]), out_sharding=_activation_spec())


class MoeForwardStats(eqx.Module):
    """Selected routes and globally counted valid dispatch losses from one forward.

    Backends mask invalid assignments before counting drops. Receiver losses are
    counted after sender selection, so their sum counts each lost assignment once.
    Padding skips are not capacity drops, and padding does not consume capacity in
    this forward. The unmasked training forward (``activations``,
    ``compute_next_token_loss``) lets padding consume capacity, so these counts
    describe it only at dropless capacity or on a batch with no padding.
    """

    routes: Array
    sender_dropped: Array
    receiver_dropped: Array

    @property
    def dropped(self) -> Array:
        return self.sender_dropped + self.receiver_dropped


class SnowballDecodeRoutes(eqx.Module):
    """Expert IDs across the padded query rows and their validity mask."""

    expert_ids: Array  # [layer, position, top_k]
    token_valid: Array  # [position]


class SnowballPagedResult(NamedTuple):
    logits: NamedArray
    cache: StackedKvPageCache | ListCache[KvPageCache]
    routes: SnowballDecodeRoutes | None


class SnowballRouteReplay(eqx.Module):
    """Layer-ordered expert IDs and token validity from a packed rollout."""

    expert_ids: Array  # [layers, *token_axes, top_k]
    token_valid: NamedArray


class MoeForwardOutput(NamedTuple):
    """One MoE-bearing forward's output with its optional observers.

    ``routes`` is set when the caller asked for the selected expert ids. ``counts``
    is set when the caller passed ``token_valid`` or enabled capacity reporting.
    """

    output: Array
    routes: Array | None
    counts: MoeDispatchCounts | None


class SnowballMoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights and frozen routing biases.

    The loaded ``router_bias`` already includes the exported QB updates. This forward
    never updates it or computes QB-beta statistics. The checked dispatch path also
    returns capacity losses for valid assignments.
    """

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: SnowballConfig = eqx.field(static=True)
    selected_decode: bool = eqx.field(static=True, default=False)

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballMoEMLP":
        k_router, k_expert = random.split(key, 2)
        expert_axis_size = _mesh_axis_size(_current_mesh(), "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")
        d, e = cfg.hidden_dim, cfg.num_experts
        return SnowballMoEMLP(
            router=_init_weight(k_router, (d, e), cfg.initializer_std, P(None, None)),
            router_bias=jnp.zeros((e,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=cfg.moe_implementation,
                activation=silu_fp32 if cfg.fp32_activations else ActivationFunctionEnum.silu,
                capacity_factor=cfg.moe_capacity_factor,
                combine_in_fp32=cfg.router_combine_fp32,
                pspecs=MoEExpertMlpPspecs(hidden=_FSDP_AXES),
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        return self._forward(x, return_routes=False).output

    def with_routes(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], Int[Array, "B S K"]]:
        """The routed output and the selected expert ids [B, S, K], as used (an observer)."""
        result = self._forward(x, return_routes=True)
        assert result.routes is not None
        return result.output, result.routes

    def _forward(
        self,
        x: Float[Array, "B S D"],
        return_routes: bool,
        token_valid: Array | None = None,
        *,
        collect_counts: bool = False,
        replay_experts: Array | None = None,
    ) -> MoeForwardOutput:
        b, s, _ = x.shape
        x_flat = jnp.reshape(x, (b * s, x.shape[-1]), out_sharding=_token_spec())
        router_input = x_flat
        router_weight = reshard(self.router, P(None, None))
        if self.cfg.router_dot_fp32:
            router_input = router_input.astype(jnp.float32)
            router_weight = router_weight.astype(jnp.float32)
        router_logits = jnp.einsum("td,de->te", router_input, router_weight).astype(jnp.float32)
        # router_bias is [E]; replicate it (like the norm weights) so the add keeps the expert axis
        # unsharded. A safetensors load auto-shards [E] over `data` when E % data == 0, which would
        # otherwise make router_logits + router_bias illegally sharded on multi-device meshes.
        if replay_experts is None:
            biased_logits = router_logits + unshard(self.router_bias)
            _, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
            selected_experts = selected_experts[:, :-1]
        else:
            selected_experts = reshard(replay_experts, P(_token_spec()[0], None))
        # Sigmoid combine weights on UNbiased logits for the selected experts, renormed to sum to 2.5.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = jax.nn.sigmoid(unbiased_topk)
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_f if self.cfg.router_combine_fp32 else combine_weights_f.astype(x.dtype)

        valid_flat = None if token_valid is None else reshard(token_valid.reshape(b * s), P(_token_spec()[0]))
        counts = None
        if self.selected_decode:
            assert valid_flat is not None
            routed_flat = self.expert_mlp.decode_small_ep(
                x_flat,
                selected_experts.astype(jnp.int32),
                combine_weights,
                token_valid=valid_flat,
                mesh=get_abstract_mesh(),
            )
        else:
            report_capacity_overflow = token_valid is not None or collect_counts
            result = self.expert_mlp(
                x_flat,
                selected_experts.astype(jnp.int32),
                combine_weights,
                mesh=get_abstract_mesh(),
                token_valid=valid_flat,
                report_capacity_overflow=report_capacity_overflow,
            )
            if report_capacity_overflow:
                routed_flat, counts = cast(tuple[Array, MoeDispatchCounts], result)
            else:
                routed_flat = cast(Array, result)
        out = jax.lax.reshape(routed_flat, (b, s, routed_flat.shape[-1]), out_sharding=_activation_spec())
        if not return_routes:
            return MoeForwardOutput(out, None, counts)
        # Replicated: a small int array, so the observer adds no sharding constraint to the forward.
        ids = reshard(selected_experts.astype(jnp.int32), P(None, None))
        return MoeForwardOutput(out, jnp.reshape(ids, (b, s, ids.shape[-1])), counts)


class SnowballBlock(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: SnowballAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: SnowballMoEMLP
    shared: DenseMLP

    def with_packed_experts(self) -> "SnowballBlock":
        """Pack the two routed expert input projections for inference."""
        return eqx.tree_at(lambda block: block.mlp.expert_mlp, self, self.mlp.expert_mlp.with_packed_gate_up())

    @staticmethod
    def init(cfg: SnowballConfig, *, key: PRNGKeyArray) -> "SnowballBlock":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        return SnowballBlock(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps, _replay_norm(cfg)),
            attn_gated_norm=GatedNorm.init(
                cfg.hidden_dim,
                cfg.initializer_std,
                key=gn_attn_key,
                fp32_activations=cfg.fp32_activations,
                projection_implementation=cfg.projection_implementation,
            ),
            attn=SnowballAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps, _replay_norm(cfg)),
            mlp_gated_norm=GatedNorm.init(
                cfg.hidden_dim,
                cfg.initializer_std,
                key=gn_mlp_key,
                fp32_activations=cfg.fp32_activations,
                projection_implementation=cfg.projection_implementation,
            ),
            mlp=SnowballMoEMLP.init(cfg, key=mlp_key),
            shared=DenseMLP.init(
                cfg.hidden_dim,
                cfg.shared_expert_intermediate_dim,
                cfg.initializer_std,
                key=shared_key,
                fp32_activations=cfg.fp32_activations,
                projection_implementation=cfg.projection_implementation,
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
        return self.forward(x, short_mask, long_mask, use_long, return_routes=False)[0]

    def forward(
        self,
        x: Float[Array, "B S D"],
        short_mask: AttentionMask,
        long_mask: AttentionMask,
        use_long: Bool[Array, ""],
        return_routes: bool,
    ) -> tuple[Float[Array, "B S D"], Array | None]:
        """The block output and, with ``return_routes``, the MoE's selected expert ids [B, S, K] (an observer)."""
        result = self._forward(x, short_mask, long_mask, use_long, return_routes)
        return result.output, result.routes

    def _forward(
        self,
        x: Float[Array, "B S D"],
        short_mask: AttentionMask,
        long_mask: AttentionMask,
        use_long: Bool[Array, ""],
        return_routes: bool,
        token_valid: Array | None = None,
        *,
        collect_counts: bool = False,
        replay_experts: Array | None = None,
    ) -> MoeForwardOutput:
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
        return self._after_attention(
            x, attn_out, return_routes, token_valid, collect_counts=collect_counts, replay_experts=replay_experts
        )

    def _after_attention(
        self,
        x: Array,
        attn_out: Array,
        return_routes: bool,
        token_valid: Array | None = None,
        *,
        collect_counts: bool = False,
        replay_experts: Array | None = None,
    ) -> MoeForwardOutput:
        x = x + attn_out
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        routed = self.mlp._forward(
            mlp_in, return_routes, token_valid, collect_counts=collect_counts, replay_experts=replay_experts
        )
        return MoeForwardOutput(x + (routed.output + self.shared(mlp_in)), routed.routes, routed.counts)

    def decode(
        self,
        x: Array,
        cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: Array,
        use_long: bool | Bool[Array, ""],
        token_valid: Array,
        *,
        return_routes: bool = False,
        implementation: PagedAttentionImplementation | None = None,
        kv_axis: Literal["model", "expert"] = "model",
        fused_kv_tokens: int = FUSED_KV_APPEND_MAX_TOKENS,
    ) -> tuple[MoeForwardOutput, KvPageCache]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))

        def attend(long_context: bool):
            return self.attn.decode(
                attn_in,
                cache,
                batch_info,
                pos_ids,
                mode="long" if long_context else "short",
                implementation=implementation,
                kv_axis=kv_axis,
                fused_kv_tokens=fused_kv_tokens,
            )

        if isinstance(use_long, bool):
            attn_out, cache = attend(use_long)
        else:
            attn_out, cache = jax.lax.cond(use_long, lambda: attend(True), lambda: attend(False))
        return self._after_attention(x, attn_out, return_routes, token_valid), cache


def _is_long_attention_layer(index, num_layers: int):
    return ((index + 1) % _LONG_ATTENTION_INTERVAL == 0) | (index == num_layers - 1)


def long_attention_layer_mask(num_layers: int) -> jnp.ndarray:
    """Bool[num_layers], True where a layer runs full-causal attention instead of the sliding window."""
    return _is_long_attention_layer(jnp.arange(num_layers), num_layers)


def num_long_attention_layers(num_layers: int) -> int:
    """Number of layers that run full-causal attention, counted without JAX for FLOP estimates."""
    return sum(_is_long_attention_layer(i, num_layers) for i in range(num_layers))


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
        token_embed = _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std, Pembed_vocab)
        output_proj = _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std, Plm_head)
        blocks = tuple(SnowballBlock.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return SnowballTransformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps, _replay_norm(cfg)),
            embed_gated_norm=GatedNorm.init(
                cfg.hidden_dim,
                cfg.initializer_std,
                key=embed_gn_key,
                fp32_activations=cfg.fp32_activations,
                projection_implementation=cfg.projection_implementation,
            ),
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps, _replay_norm(cfg)),
            final_gated_norm=GatedNorm.init(
                cfg.hidden_dim,
                cfg.initializer_std,
                key=final_gn_key,
                fp32_activations=cfg.fp32_activations,
                projection_implementation=cfg.projection_implementation,
            ),
            config=cfg,
        )

    def __call__(self, token_ids: Int[Array, "B S"], mask: Optional[AttentionMask] = None) -> Float[Array, "B S D"]:
        return self._forward(token_ids, mask, return_routes=False).output

    def with_stats(
        self, token_ids: Array, token_valid: Array, mask: AttentionMask | None = None
    ) -> tuple[Array, MoeForwardStats]:
        """Run causal attention with document segments, dispatch validity and counts."""
        result = self._forward(token_ids, mask, True, token_valid)
        assert result.routes is not None and result.counts is not None
        stats = MoeForwardStats(result.routes, result.counts.sender_dropped, result.counts.receiver_dropped)
        return result.output, stats

    @named_call(name="SnowballTransformer")
    def _forward(
        self,
        token_ids: Int[Array, "B S"],
        mask: Optional[AttentionMask],
        return_routes: bool,
        token_valid: Array | None = None,
        *,
        collect_counts: bool = False,
        replay_experts: Array | None = None,
    ) -> MoeForwardOutput:
        cfg = self.config
        if mask is None:
            mask = AttentionMask.causal()
        if cfg.shared_attention_implementation == "fixed_schedule":
            _validate_fixed_shared_attention_public_mask(mask)
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        token_ids = reshard(token_ids, P(_BATCH_AXES, _context_axis()))
        if token_valid is not None:
            token_valid = reshard(token_valid, P(_BATCH_AXES, _context_axis()))
        hidden = self.token_embed.at[token_ids].get(out_sharding=_activation_spec())
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Scan the layers instead of a Python loop so XLA plans HBM for ONE layer's MoE expert
        # buffers at a time. Unrolling keeps every layer's buffers live simultaneously (temp scales
        # linearly with depth), which OOMs the 67B on 8xH100. The stacked blocks + per-layer long
        # schedule feed a single uniform scan body (June recipe: long layers = every 4th + the last).
        num_blocks = len(self.blocks)
        stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *self.blocks)
        long_schedule = long_attention_layer_mask(num_blocks)

        def _scan_layer(carry: Float[Array, "B S D"], layer_data):
            layer, use_long, layer_replay = layer_data
            if token_valid is not None or collect_counts or layer_replay is not None:
                result = layer._forward(
                    carry,
                    short_mask,
                    long_mask,
                    use_long,
                    return_routes,
                    token_valid,
                    collect_counts=collect_counts,
                    replay_experts=layer_replay,
                )
                return result.output, (result.routes, result.counts)
            if return_routes:
                hidden, routes = layer.forward(carry, short_mask, long_mask, use_long, return_routes=True)
                return hidden, (routes, None)
            return layer(carry, short_mask, long_mask, use_long), (None, None)

        # Bound routed-expert activation memory during the backward pass as well as forward.
        hidden, (routes, counts) = jax.lax.scan(
            jax.checkpoint(_scan_layer), hidden, (stacked, long_schedule, replay_experts)
        )
        return MoeForwardOutput(self.final_gated_norm(self.final_norm(hidden)), routes, counts)


# --- LmHeadModel adapter -----------------------------------------------------------------------


class SnowballLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[SnowballConfig]):
    """Levanter ``LmHeadModel`` boundary over the array-first Snowball transformer.

    Normalizes named-axis inputs to raw ``[B, S]`` arrays for the grug forward and wraps the raw
    hidden state back into a ``NamedArray`` so the standard LM head / loss / scoring path works.
    """

    transformer: SnowballTransformer
    _config: SnowballConfig = eqx.field(static=True)
    _unroll_decode: bool = eqx.field(static=True, default=False)
    _split_decode_cache: bool = eqx.field(static=True, default=False)
    _paged_attention: PagedAttentionImplementation | None = eqx.field(static=True, default=None)
    _kv_axis: Literal["model", "expert"] = eqx.field(static=True, default="model")
    _selected_ep_decode: bool = eqx.field(static=True, default=False)

    @property
    def has_expert_sharded_kv(self) -> bool:
        return self._kv_axis == "expert"

    def with_expert_sharded_kv(self) -> "SnowballLMHeadModel":
        """Shard paged KV heads across the expert axis during generation."""
        return dataclasses.replace(self, _kv_axis="expert")

    def with_paged_attention(self, implementation: PagedAttentionImplementation) -> "SnowballLMHeadModel":
        """Select the attention backend for cache-based generation."""
        return dataclasses.replace(self, _paged_attention=implementation)

    def with_unrolled_decode(self) -> "SnowballLMHeadModel":
        """Return a model that unrolls its layer loop during token decode."""
        return dataclasses.replace(self, _unroll_decode=True)

    def with_split_decode_cache(self) -> "SnowballLMHeadModel":
        """Keep one KV buffer per layer for unrolled token decode."""
        return dataclasses.replace(self, _unroll_decode=True, _split_decode_cache=True)

    def with_packed_experts(self) -> "SnowballLMHeadModel":
        """Keep routed gate and up weights in the grouped inference layout."""
        blocks = tuple(block.with_packed_experts() for block in self.transformer.blocks)
        return eqx.tree_at(lambda model: model.transformer.blocks, self, blocks)

    def with_selected_ep_decode(self) -> "SnowballLMHeadModel":
        """Read selected matrix slices for eligible small expert-parallel decode batches."""
        return dataclasses.replace(self, _selected_ep_decode=True)

    def with_batch_invariant_rmsnorm(self, implementation: RmsNormImplementation) -> "SnowballLMHeadModel":
        """Select the same weighted normalization for learner and paged decode."""
        transformer = jax.tree.map(
            lambda layer: (
                dataclasses.replace(layer, implementation=implementation) if isinstance(layer, RMSNorm) else layer
            ),
            self.transformer,
            is_leaf=lambda layer: isinstance(layer, RMSNorm),
        )
        return dataclasses.replace(self, transformer=transformer)

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
        return self._activations(input_ids, return_routes=False, attn_mask=attn_mask)[0]

    def activations_with_replay(
        self, input_ids: NamedArray, attn_mask: LmHeadAttentionMask, route_replay: SnowballRouteReplay
    ) -> NamedArray:
        """Compute hidden states with prepared expert IDs and live combine weights."""
        return self._activations(input_ids, False, attn_mask, route_replay=route_replay)[0]

    def activations_with_routes(
        self, input_ids: NamedArray, attn_mask: LmHeadAttentionMask | NamedArray | None = None
    ) -> tuple[NamedArray, Array]:
        """The activations and every layer's selected expert ids ``[L, B, S, K]`` from this forward (an observer)."""
        hidden, routes, _ = self._activations(input_ids, return_routes=True, attn_mask=attn_mask)
        assert routes is not None
        return hidden, routes

    def activations_with_stats(
        self,
        input_ids: NamedArray,
        token_valid: NamedArray,
        *,
        attn_mask: LmHeadAttentionMask | NamedArray | None = None,
    ) -> tuple[NamedArray, MoeForwardStats]:
        """Return hidden states, routes and raw per-call valid-assignment counts.

        Callers must sum the count arrays across microbatches for update totals.
        ``token_valid`` excludes padding from capacity here; the training forward
        passes no mask, so zero drops here prove it dropped nothing only at dropless
        capacity or with no padding (see ``MoeForwardStats``).
        """
        if token_valid.axes != input_ids.axes or token_valid.dtype != jnp.bool_:
            raise ValueError("token_valid must be boolean with the token axes")
        sequence = input_ids.array.shape[-1]
        hidden, stats = self.transformer.with_stats(
            input_ids.array.reshape(-1, sequence),
            token_valid.array.reshape(-1, sequence),
            mask=self._attention_mask(input_ids, attn_mask),
        )
        hidden = hidden.reshape(*input_ids.array.shape, self.Embed.size)
        return hax.named(hidden, (*input_ids.axes, self.Embed)), stats

    def _attention_mask(
        self, input_ids: NamedArray, attn_mask: LmHeadAttentionMask | NamedArray | None
    ) -> AttentionMask | None:
        """Preserve document IDs while the model retains its short/long causal windows."""
        if attn_mask is None:
            return None
        if not isinstance(attn_mask, LmHeadAttentionMask):
            raise ValueError("Snowball accepts structured causal document masks, not dense masks")
        if not attn_mask.is_causal or attn_mask.explicit_mask is not None or attn_mask.causal_offset is not None:
            raise ValueError("Snowball document masks cannot override the model's causal/window policy")
        if attn_mask.segment_ids is None:
            return AttentionMask.causal()
        sequence = input_ids.array.shape[-1]
        query, keys = attn_mask.segment_ids
        # The native Grug -> named reader deliberately gives KV its own axis.
        # Normalize that alias before broadcasting/flattening, preserving values.
        if "key_position" in keys.axis_names:
            key_position = keys.resolve_axis("key_position")
            position = input_ids.axes[-1]
            if key_position.size != position.size or position.name in keys.axis_names:
                raise ValueError("Snowball key_position must match the input position axis exactly")
            keys = keys.rename({"key_position": position})
        query = query.broadcast_to(input_ids.axes).array.reshape(-1, sequence)
        keys = keys.broadcast_to(input_ids.axes).array.reshape(-1, sequence)
        query = reshard(query, P(_BATCH_AXES, _context_axis()))
        keys = reshard(keys, P(_BATCH_AXES, None))
        return AttentionMask.causal().with_segment_ids(query, keys)

    def prepare_route_replay(self, expert_ids: Array, token_valid: NamedArray) -> SnowballRouteReplay:
        """Validate a packed rollout's route IDs on the host before a learner JIT."""
        expected = (self.config.num_layers, *token_valid.array.shape, self.config.num_experts_per_token)
        ids = np.asarray(expert_ids)
        if ids.shape != expected or ids.dtype.kind not in "iu" or token_valid.dtype != jnp.bool_:
            raise ValueError(f"Expected integer expert IDs of shape {expected} and a boolean token mask")
        active = np.broadcast_to(np.asarray(token_valid.array), ids.shape[:-1])
        validate_captured_experts(ids[active], self.config.num_experts)
        # Padding is excluded from dispatch, but gather still needs valid IDs.
        ids = np.where(active[..., None], ids, np.arange(expected[-1], dtype=np.int32))
        return SnowballRouteReplay(jnp.asarray(ids, dtype=jnp.int32), token_valid)

    def _policy_matmul_precision(self):
        if self.config.replay_precision == "zero_kl":
            return jax.default_matmul_precision("highest")
        return nullcontext()

    def _activations(
        self,
        input_ids: NamedArray,
        return_routes: bool,
        attn_mask: LmHeadAttentionMask | NamedArray | None,
        *,
        collect_counts: bool = False,
        route_replay: SnowballRouteReplay | None = None,
    ) -> tuple[NamedArray, Array | None, MoeDispatchCounts | None]:
        Pos = input_ids.resolve_axis(self.Pos.name)
        raw = input_ids.array
        lead = raw.shape[:-1]
        s = raw.shape[-1]
        b = int(np.prod(lead)) if lead else 1
        tokens = reshard(raw.reshape(b, s), P(_BATCH_AXES, _context_axis()))
        # [B, S, D], and [L, B, S, K] with return_routes
        mask = self._attention_mask(input_ids, attn_mask)
        token_valid = replay_experts = None
        if route_replay is not None:
            if route_replay.token_valid.axes != input_ids.axes:
                raise ValueError("Route replay must use the learner's token axes")
            token_valid = route_replay.token_valid.array.reshape(b, s)
            replay_experts = route_replay.expert_ids.reshape(
                self.config.num_layers, b * s, self.config.num_experts_per_token
            )
        with self._policy_matmul_precision():
            result = self.transformer._forward(
                tokens,
                mask,
                return_routes,
                token_valid,
                collect_counts=collect_counts,
                replay_experts=replay_experts,
            )
        hidden = (
            result.output.reshape(*lead, s, self.Embed.size) if lead else result.output.reshape(s, self.Embed.size)
        )
        out_axes = (*input_ids.axes, self.Embed) if lead else (Pos, self.Embed)
        return hax.named(hidden, out_axes), result.routes, result.counts

    def next_token_loss_and_metrics(
        self, example: LmExample, *, key=None, route_replay: SnowballRouteReplay | None = None, **kwargs
    ):
        """Return loss and the fraction of routed assignments dropped. Replay counts valid positions."""
        activations, _, counts = self._activations(
            example.tokens, False, example.attn_mask, collect_counts=True, route_replay=route_replay
        )
        assert counts is not None
        loss = self._loss_from_activations(example, activations, **kwargs)
        num_tokens = (
            float(example.tokens.array.size)
            if route_replay is None
            else jnp.sum(route_replay.token_valid.array, dtype=jnp.float32)
        )
        assignments = num_tokens * self.config.num_experts_per_token * self.config.num_layers
        return loss, {
            "moe/dropped_assignment_fraction": Metric.from_value(
                counts.dropped.sum(dtype=jnp.float32) / jnp.maximum(assignments, 1.0), ReductionType.MEAN
            )
        }

    def project_logits(self, activations: NamedArray) -> NamedArray:
        if self.config.projection_implementation == "native":
            return super().project_logits(activations)
        logits = _fixed_projection(activations.array, self.transformer.output_proj)
        return hax.named(logits, (*activations.axes[:-1], self.Vocab))

    def fixed_policy_logprobs(self, activations: NamedArray, targets: NamedArray) -> NamedArray:
        """Score target IDs with the fixed projection and policy normalization."""
        logits = self.project_logits(activations).astype(jnp.float32)
        return self._fixed_target_logprobs(logits, targets)

    def _fixed_target_logprobs(self, logits: NamedArray, targets: NamedArray) -> NamedArray:
        row_axes = logits.axes[:-1]
        selected = batch_invariant_selected_logprobs(logits.array, targets.rearrange(row_axes).array)
        return hax.named(selected, row_axes)

    def _loss_from_activations(
        self,
        example: LmExample,
        activations: NamedArray | tuple[NamedArray, NamedOrNumeric],
        *,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ) -> jnp.ndarray | NamedArray:
        if self.config.projection_implementation == "native":
            return super()._loss_from_activations(
                example,
                activations,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )
        hidden, aux_loss = split_activations(activations)
        loss_type = example.loss_weight.dtype if example.loss_weight is not None else loss_dtype
        logits = self.project_logits(hidden)
        if loss_type is not None:
            logits = logits.astype(loss_type)
        if logit_soft_cap is not None:
            logits = hax.tanh(logits / logit_soft_cap) * logit_soft_cap
        targets = next_token_ids(self.Pos.name, example.tokens)
        loss = -self._fixed_target_logprobs(logits, targets)
        if logsumexp_weight:
            normalizers = hax.named(batch_invariant_logsumexp(logits.array), logits.axes[:-1])
            loss = loss + logsumexp_weight * hax.square(normalizers)
        weights = next_token_loss_weight(logits.resolve_axis(self.Pos.name), example.loss_weight)
        return hax.nn.loss.reduce_loss(loss, reduction, reduction_axis, weight=weights) + aux_loss

    def get_lm_head(self) -> NamedArray:
        return hax.named(reshard(self.transformer.output_proj, P(None, "model")), (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "SnowballLMHeadModel":
        old = self._config.vocab_size
        if new_size == old:
            return self
        te = _resize_axis(self.transformer.token_embed, 0, new_size, self._config.initializer_std, key)
        op = _resize_axis(self.transformer.output_proj, 1, new_size, self._config.initializer_std, key)
        new_cfg = dataclasses.replace(self._config, vocab_size=new_size)
        new_tf = eqx.tree_at(lambda t: (t.token_embed, t.output_proj, t.config), self.transformer, (te, op, new_cfg))
        return dataclasses.replace(self, transformer=new_tf, _config=new_cfg)

    def initial_layer_cache(self, spec: PageTableSpec, *, dtype: jnp.dtype) -> KvPageCache:
        """Allocate interleaved KV pages directly on the model's head sharding."""
        if _context_axis() is not None:
            raise ValueError("Snowball paged inference requires context_axis_size=1")
        axes = (
            Axis("page", spec.num_pages),
            Axis("slot", spec.page_size),
            Axis("kv_head", 2 * _inference_kv_heads(self.config, self._kv_axis)),
            Axis("head_size", self.config.inferred_head_dim),
        )
        kv_axis = None if self.config.replay_precision == "zero_kl" else self._kv_axis
        sharding = _current_sharding(P(None, None, kv_axis, None))
        pages = jnp.zeros(tuple(axis.size for axis in axes), dtype, out_sharding=sharding)
        return KvPageCache(hax.named(pages, axes))

    def layer_cache_specs(self, spec: PageTableSpec, window_cache_pages: int) -> tuple[PageTableSpec, ...]:
        """Select full-context or sliding-window capacity for each split layer."""
        layers = len(self.transformer.blocks)
        if not self._split_decode_cache or num_long_attention_layers(layers) == layers:
            raise ValueError("A compact window cache requires split decode and sliding-window layers")
        window_spec = dataclasses.replace(spec, num_pages=window_cache_pages)
        return tuple(spec if _is_long_attention_layer(i, layers) else window_spec for i in range(layers))

    def initial_cache(
        self, spec: PageTableSpec, *, dtype: jnp.dtype, window_cache_pages: int | None = None
    ) -> StackedKvPageCache | ListCache[KvPageCache]:
        """Allocate paged KV state on a serving mesh with context parallelism disabled."""
        if self.config.replay_precision == "zero_kl":
            if dtype != jnp.bfloat16 or self.transformer.token_embed.dtype != jnp.float32:
                raise ValueError("The zero-KL replay profile requires FP32 compute and BF16 serving KV")
            if window_cache_pages is not None:
                raise ValueError("The zero-KL replay profile uses the complete cache for each layer")
            return ListCache(tuple(self.initial_layer_cache(spec, dtype=dtype) for _ in self.transformer.blocks))
        if window_cache_pages is not None:
            specs = self.layer_cache_specs(spec, window_cache_pages)
            return ListCache(tuple(self.initial_layer_cache(layer, dtype=dtype) for layer in specs))
        layer = Axis(_CACHE_LAYER_AXIS, len(self.transformer.blocks))
        pages = self.initial_layer_cache(spec, dtype=dtype).kv_pages
        stacked = pages.broadcast_axis(layer).rearrange((layer, ...))
        stacked = hax.named(reshard(stacked.array, P(None, None, None, self._kv_axis, None)), stacked.axes)
        cache = StackedKvPageCache(stacked)
        return _split_layer_cache(cache, self._kv_axis) if self._split_decode_cache else cache

    @named_call
    def decode(
        self,
        input_ids: NamedArray,
        kv_cache: StackedKvPageCache | ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        window_batch: PageBatchInfo | None = None,
    ) -> tuple[NamedArray, StackedKvPageCache | ListCache[KvPageCache]]:
        """Prefill or decode packed sequences using absolute positions and paged KV state.

        Tokens occupy a flat position axis, with a valid prefix described by batch_info.
        Expert assignments are independent of other requests and padding.
        """
        result = self._decode_paged(input_ids, kv_cache, batch_info, pos_ids, window_batch=window_batch)
        return result.logits, result.cache

    @named_call
    def decode_for_sampling(
        self,
        input_ids: NamedArray,
        kv_cache: StackedKvPageCache | ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        sample_indices: NamedArray,
        *,
        window_batch: PageBatchInfo | None = None,
    ) -> tuple[NamedArray, StackedKvPageCache | ListCache[KvPageCache]]:
        """Return only sampled logits, preserving all valid rows in the updated KV cache."""
        result = self._decode_paged(
            input_ids,
            kv_cache,
            batch_info,
            pos_ids,
            window_batch=window_batch,
            sample_indices=sample_indices,
            prefill=True,
        )
        return result.logits, result.cache

    def decode_with_routes(
        self,
        input_ids: NamedArray,
        kv_cache: StackedKvPageCache | ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        sample_indices: NamedArray | None = None,
        window_batch: PageBatchInfo | None = None,
        prefill: bool = False,
    ) -> SnowballPagedResult:
        """Return query-row expert IDs independently of the vocabulary projection rows."""
        return self._decode_paged(
            input_ids,
            kv_cache,
            batch_info,
            pos_ids,
            sample_indices=sample_indices,
            window_batch=window_batch,
            prefill=prefill,
            capture_routes=True,
        )

    def _decode_paged(
        self,
        input_ids: NamedArray,
        kv_cache: StackedKvPageCache | ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        sample_indices: NamedArray | None = None,
        window_batch: PageBatchInfo | None = None,
        prefill: bool = False,
        capture_routes: bool = False,
    ) -> SnowballPagedResult:
        unroll = window_batch is not None or (self._unroll_decode and not prefill)
        with self._policy_matmul_precision():
            hidden, caches, routes = self._decode_hidden(
                input_ids,
                kv_cache,
                batch_info,
                pos_ids,
                unroll=unroll,
                window_batch=window_batch,
                capture_routes=capture_routes,
                fused_kv_tokens=(FUSED_KV_PREFILL_MAX_TOKENS if prefill else FUSED_KV_APPEND_MAX_TOKENS),
            )
        if self.config.projection_implementation == "fixed_schedule":
            rows = hax.named(hidden[:, 0], (*input_ids.axes, self.Embed))
            if sample_indices is not None:
                sampled = hidden[:, 0].at[sample_indices.array].get(out_sharding=P())
                rows = hax.named(sampled, (*sample_indices.axes, self.Embed))
            output = self.project_logits(rows)
        elif sample_indices is None:
            logits = jnp.einsum(
                "bsd,dv->bsv", hidden, self.transformer.output_proj, out_sharding=_activation_spec("model")
            )
            output = hax.named(logits[:, 0], (*input_ids.axes, self.Vocab))
        else:
            sampled = hidden[:, 0].at[sample_indices.array].get(out_sharding=P())
            logits = jnp.einsum("pd,dv->pv", sampled, self.transformer.output_proj, out_sharding=P())
            output = hax.named(logits, (*sample_indices.axes, self.Vocab))
        return SnowballPagedResult(output, caches, routes)

    def _decode_hidden(
        self,
        input_ids: NamedArray,
        kv_cache: StackedKvPageCache | ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        unroll: bool,
        window_batch: PageBatchInfo | None,
        capture_routes: bool,
        fused_kv_tokens: int,
    ) -> tuple[Array, StackedKvPageCache | ListCache[KvPageCache], SnowballDecodeRoutes | None]:
        if input_ids.ndim != 1 or pos_ids.axes != input_ids.axes:
            raise ValueError("Snowball decode requires matching flat token and position arrays")
        num_layers = len(self.transformer.blocks)
        cache_layers = (
            len(kv_cache) if isinstance(kv_cache, ListCache) else kv_cache.kv_pages.axis_size(_CACHE_LAYER_AXIS)
        )
        if cache_layers != num_layers:
            raise ValueError("Snowball decode requires one KV cache per transformer layer")
        tokens = reshard(input_ids.array[:, None], P(_BATCH_AXES, None))
        hidden = self.transformer.token_embed.at[tokens].get(out_sharding=_activation_spec())
        hidden = self.transformer.embed_gated_norm(self.transformer.embed_norm(hidden))
        token_valid = jnp.arange(tokens.shape[0]) < batch_info.num_new_tokens
        if isinstance(kv_cache, ListCache) and unroll:
            layers = []
            routes_by_layer = []
            for layer_index, (block, cache) in enumerate(zip(self.transformer.blocks, kv_cache, strict=True)):
                use_long = _is_long_attention_layer(layer_index, num_layers)
                layer_batch = window_batch if window_batch is not None and not use_long else batch_info
                result, cache = self._inference_block(block, tokens.shape[0]).decode(
                    hidden,
                    cache,
                    layer_batch,
                    pos_ids.array,
                    use_long,
                    token_valid,
                    return_routes=capture_routes,
                    implementation=self._paged_attention,
                    kv_axis=self._kv_axis,
                    fused_kv_tokens=fused_kv_tokens,
                )
                hidden = result.output
                layers.append(cache)
                if result.routes is not None:
                    routes_by_layer.append(result.routes[:, 0])
            hidden = self.transformer.final_gated_norm(self.transformer.final_norm(hidden))
            routes = SnowballDecodeRoutes(jnp.stack(routes_by_layer), token_valid) if capture_routes else None
            return hidden, ListCache(tuple(layers)), routes

        split_cache = isinstance(kv_cache, ListCache)
        kv_axis = None if self.config.replay_precision == "zero_kl" else self._kv_axis
        stacked_cache = _stack_layer_cache(kv_cache, kv_axis) if split_cache else kv_cache
        cache_axes = stacked_cache.kv_pages.axes[1:]

        def layer_step(carry, layer_data):
            x, pages = carry
            layer, block, use_long = layer_data
            layer_pages = jax.lax.dynamic_index_in_dim(pages, layer, keepdims=False)
            cache = KvPageCache(hax.named(layer_pages, cache_axes))
            result, cache = block.decode(
                x,
                cache,
                batch_info,
                pos_ids.array,
                use_long,
                token_valid,
                return_routes=capture_routes,
                implementation=self._paged_attention,
                kv_axis=self._kv_axis,
                fused_kv_tokens=fused_kv_tokens,
            )
            pages = jax.lax.dynamic_update_index_in_dim(pages, cache.kv_pages.array, layer, axis=0)
            routes = None if result.routes is None else result.routes[:, 0]
            return (result.output, pages), routes

        initial = (hidden, stacked_cache.kv_pages.array)
        if unroll:
            state = initial
            routes_by_layer = []
            for layer_index, block in enumerate(self.transformer.blocks):
                use_long = _is_long_attention_layer(layer_index, num_layers)
                state, routes = layer_step(
                    state, (layer_index, self._inference_block(block, tokens.shape[0]), use_long)
                )
                if routes is not None:
                    routes_by_layer.append(routes)
            hidden, cache_pages = state
            expert_ids = jnp.stack(routes_by_layer) if capture_routes else None
        else:
            stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *self.transformer.blocks)
            layer_data = (
                jnp.arange(num_layers),
                self._inference_block(stacked, tokens.shape[0]),
                long_attention_layer_mask(num_layers),
            )
            (hidden, cache_pages), expert_ids = jax.lax.scan(layer_step, initial, layer_data)
        hidden = self.transformer.final_gated_norm(self.transformer.final_norm(hidden))
        caches = StackedKvPageCache(hax.named(cache_pages, stacked_cache.kv_pages.axes))
        routes = SnowballDecodeRoutes(expert_ids, token_valid) if expert_ids is not None else None
        return hidden, _split_layer_cache(caches, kv_axis) if split_cache else caches, routes

    def _inference_block(self, block: SnowballBlock, token_rows: int) -> SnowballBlock:
        expert_size = _mesh_axis_size(_current_mesh(), "expert")
        if expert_size <= 1:
            return block
        # A receiver can get every sender's assignments. Reserve that worst-case
        # capacity so expert dispatch cannot drop tokens at serving time.
        experts = dataclasses.replace(block.mlp.expert_mlp, implementation="ring", capacity_factor=float(expert_size))
        selected = (
            self._selected_ep_decode
            and token_rows == expert_size
            and not self.config.router_combine_fp32
            and _mesh_axis_size(_current_mesh(), "data") == 1
            and _mesh_axis_size(_current_mesh(), "replica_dcn") == 1
            and _mesh_axis_size(_current_mesh(), "replica") == 1
            and _mesh_axis_size(_current_mesh(), "model") == 1
        )
        mlp = dataclasses.replace(block.mlp, expert_mlp=experts, selected_decode=selected)
        return eqx.tree_at(lambda candidate: candidate.mlp, block, mlp)

    # --- state dict (bidirectional HF serialization) ---
    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        return snowball_to_state_dict(self.transformer, prefix=prefix)

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "SnowballLMHeadModel":
        new_tf = snowball_from_state_dict(self.transformer, state_dict, prefix=prefix)
        return dataclasses.replace(self, transformer=new_tf)


def _split_layer_cache(cache: StackedKvPageCache, kv_axis: str | None) -> ListCache[KvPageCache]:
    pages = cache.kv_pages
    layer_sharding = P(None, None, kv_axis, None)
    return ListCache(
        tuple(
            KvPageCache(hax.named(reshard(layer, layer_sharding), pages.axes[1:]))
            for layer in jnp.unstack(pages.array, axis=0)
        )
    )


def _stack_layer_cache(caches: ListCache[KvPageCache], kv_axis: str | None) -> StackedKvPageCache:
    layer = Axis(_CACHE_LAYER_AXIS, len(caches))
    pages = hax.stack(layer, [cache.kv_pages for cache in caches])
    return StackedKvPageCache(hax.named(reshard(pages.array, P(None, None, None, kv_axis, None)), pages.axes))


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


def snowball_embeddings_to_state_dict(model: SnowballTransformer, prefix: Optional[str] = None) -> StateDict:
    """Export the embeddings boundary using canonical HF keys."""
    tensors = {
        "model.embed_tokens.weight": model.token_embed,
        "model.embed_norm.weight": model.embed_norm.weight,
        "model.embed_gated_norm.down_proj.weight": _T(model.embed_gated_norm.w_down),
        "model.embed_gated_norm.up_proj.weight": _T(model.embed_gated_norm.w_up),
    }
    return {_with_prefix(prefix, name): value for name, value in tensors.items()}


def snowball_final_to_state_dict(model: SnowballTransformer, prefix: Optional[str] = None) -> StateDict:
    """Export the final boundary using canonical HF keys."""
    tensors = {
        "model.norm.weight": model.final_norm.weight,
        "model.final_gated_norm.down_proj.weight": _T(model.final_gated_norm.w_down),
        "model.final_gated_norm.up_proj.weight": _T(model.final_gated_norm.w_up),
        "lm_head.weight": _T(model.output_proj),
    }
    return {_with_prefix(prefix, name): value for name, value in tensors.items()}


def snowball_block_to_state_dict(block: SnowballBlock, layer_index: int, prefix: Optional[str] = None) -> StateDict:
    """Export one layer under its global HF layer index."""
    gate, up = block.mlp.expert_mlp.split_gate_up()
    p = f"model.layers.{layer_index}"
    tensors = {
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
        f"{p}.mlp.experts.gate_proj.weight": _T(gate),
        f"{p}.mlp.experts.up_proj.weight": _T(up),
        f"{p}.mlp.experts.down_proj.weight": _T(block.mlp.expert_mlp.w_down),
        f"{p}.shared_expert.gate_proj.weight": _T(block.shared.w_gate),
        f"{p}.shared_expert.up_proj.weight": _T(block.shared.w_up),
        f"{p}.shared_expert.down_proj.weight": _T(block.shared.w_down),
    }
    return {_with_prefix(prefix, name): value for name, value in tensors.items()}


def _get(state_dict: StateDict, prefix: Optional[str], name: str) -> jax.Array:
    return jnp.asarray(state_dict[_with_prefix(prefix, name)])


def snowball_embeddings_from_state_dict(
    template: SnowballTransformer, state_dict: StateDict, prefix: Optional[str] = None
) -> SnowballTransformer:
    """Load canonical HF embeddings weights into the current mesh."""
    g = lambda name: _get(state_dict, prefix, name)  # noqa: E731
    m = template
    m = eqx.tree_at(lambda t: t.token_embed, m, _reshard(g("model.embed_tokens.weight"), Pembed_vocab))
    m = eqx.tree_at(lambda t: t.embed_norm.weight, m, g("model.embed_norm.weight"))
    m = eqx.tree_at(
        lambda t: t.embed_gated_norm.w_down, m, _reshard_replicated(_T(g("model.embed_gated_norm.down_proj.weight")))
    )
    m = eqx.tree_at(
        lambda t: t.embed_gated_norm.w_up, m, _reshard_replicated(_T(g("model.embed_gated_norm.up_proj.weight")))
    )
    return m


def snowball_final_from_state_dict(
    template: SnowballTransformer, state_dict: StateDict, prefix: Optional[str] = None
) -> SnowballTransformer:
    """Load canonical HF final weights into the current mesh."""
    g = lambda name: _get(state_dict, prefix, name)  # noqa: E731
    m = template
    m = eqx.tree_at(lambda t: t.final_norm.weight, m, g("model.norm.weight"))
    m = eqx.tree_at(
        lambda t: t.final_gated_norm.w_down, m, _reshard_replicated(_T(g("model.final_gated_norm.down_proj.weight")))
    )
    m = eqx.tree_at(
        lambda t: t.final_gated_norm.w_up, m, _reshard_replicated(_T(g("model.final_gated_norm.up_proj.weight")))
    )
    m = eqx.tree_at(lambda t: t.output_proj, m, _reshard(_T(g("lm_head.weight")), Plm_head))

    return m


def snowball_block_from_state_dict(
    template: SnowballBlock, state_dict: StateDict, layer_index: int, prefix: Optional[str] = None
) -> SnowballBlock:
    """Load canonical HF block weights into the current mesh."""
    g = lambda name: _get(state_dict, prefix, name)  # noqa: E731
    m = template
    p = f"model.layers.{layer_index}"
    m = eqx.tree_at(lambda t: t.rms_attn.weight, m, g(f"{p}.input_layernorm.weight"))
    m = eqx.tree_at(
        lambda t: t.attn_gated_norm.w_down,
        m,
        _reshard_replicated(_T(g(f"{p}.attn_gated_norm.down_proj.weight"))),
    )
    m = eqx.tree_at(
        lambda t: t.attn_gated_norm.w_up,
        m,
        _reshard_replicated(_T(g(f"{p}.attn_gated_norm.up_proj.weight"))),
    )
    m = eqx.tree_at(lambda t: t.attn.w_q, m, _reshard(_T(g(f"{p}.self_attn.q_proj.weight")), P(_FSDP_AXES, "model")))
    m = eqx.tree_at(lambda t: t.attn.w_k, m, _reshard(_T(g(f"{p}.self_attn.k_proj.weight")), P(_FSDP_AXES, "model")))
    m = eqx.tree_at(lambda t: t.attn.w_v, m, _reshard(_T(g(f"{p}.self_attn.v_proj.weight")), P(_FSDP_AXES, "model")))
    m = eqx.tree_at(lambda t: t.attn.w_o, m, _reshard(_T(g(f"{p}.self_attn.o_proj.weight")), P("model", _FSDP_AXES)))
    m = eqx.tree_at(lambda t: t.attn.attn_gate, m, _reshard_replicated(_T(g(f"{p}.self_attn.attn_gate.weight"))))
    m = eqx.tree_at(lambda t: t.rms_mlp.weight, m, g(f"{p}.post_attention_layernorm.weight"))
    m = eqx.tree_at(
        lambda t: t.mlp_gated_norm.w_down,
        m,
        _reshard_replicated(_T(g(f"{p}.mlp_gated_norm.down_proj.weight"))),
    )
    m = eqx.tree_at(
        lambda t: t.mlp_gated_norm.w_up,
        m,
        _reshard_replicated(_T(g(f"{p}.mlp_gated_norm.up_proj.weight"))),
    )
    m = eqx.tree_at(lambda t: t.mlp.router, m, _reshard_replicated(_T(g(f"{p}.mlp.router.weight"))))
    m = eqx.tree_at(lambda t: t.mlp.router_bias, m, g(f"{p}.mlp.router.bias"))
    experts = m.mlp.expert_mlp
    loaded = dataclasses.replace(
        experts,
        w_gate=_reshard(_T(g(f"{p}.mlp.experts.gate_proj.weight")), _EXPERT_GATE_UP_SPEC),
        w_up=_reshard(_T(g(f"{p}.mlp.experts.up_proj.weight")), _EXPERT_GATE_UP_SPEC),
        w_down=_reshard(_T(g(f"{p}.mlp.experts.down_proj.weight")), _EXPERT_DOWN_SPEC),
    )
    if experts.w_up is None:
        loaded = loaded.with_packed_gate_up()
    m = eqx.tree_at(lambda t: t.mlp.expert_mlp, m, loaded)
    m = eqx.tree_at(
        lambda t: t.shared.w_gate,
        m,
        _reshard(_T(g(f"{p}.shared_expert.gate_proj.weight")), P(_FSDP_AXES, "model")),
    )
    m = eqx.tree_at(
        lambda t: t.shared.w_up,
        m,
        _reshard(_T(g(f"{p}.shared_expert.up_proj.weight")), P(_FSDP_AXES, "model")),
    )
    m = eqx.tree_at(
        lambda t: t.shared.w_down,
        m,
        _reshard(_T(g(f"{p}.shared_expert.down_proj.weight")), P("model", _FSDP_AXES)),
    )
    return m


def snowball_to_state_dict(model: SnowballTransformer, prefix: Optional[str] = None) -> StateDict:
    tensors = snowball_embeddings_to_state_dict(model, prefix)
    tensors.update(snowball_final_to_state_dict(model, prefix))
    for index, block in enumerate(model.blocks):
        tensors.update(snowball_block_to_state_dict(block, index, prefix))
    return tensors


def snowball_from_state_dict(
    template: SnowballTransformer, state_dict: StateDict, prefix: Optional[str] = None
) -> SnowballTransformer:
    """Load HF weights with their Grug partition specs rather than replicating raw leaves."""
    model = snowball_embeddings_from_state_dict(template, state_dict, prefix)
    model = snowball_final_from_state_dict(model, state_dict, prefix)
    blocks = tuple(
        snowball_block_from_state_dict(block, state_dict, index, prefix) for index, block in enumerate(model.blocks)
    )
    return eqx.tree_at(lambda model: model.blocks, model, blocks)


_EXPERT_GATE_UP_SPEC = P("expert", _FSDP_AXES, "model")
_EXPERT_DOWN_SPEC = P("expert", "model", _FSDP_AXES)


def _reshard_replicated(arr: jax.Array) -> jax.Array:
    return _reshard(arr, P(None, None))


def _reshard(arr: jax.Array, spec: P) -> jax.Array:
    sharding = _current_sharding(spec)
    return arr if sharding is None else reshard(arr, sharding)


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
