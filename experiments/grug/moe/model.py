# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: MoE with GatedNorm, XSA, sigmoid combine weights, optional QB
router-bias balancing, and optional eligibility-conditioned load balancing.
All layers are MoE (no dense layers).
"""

import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call
from jax import core, random
from jax.sharding import NamedSharding, get_abstract_mesh, reshard
from jax.sharding import PartitionSpec as P

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
    fa4_cute_segment_bounds,
)
from levanter.grug.grug_moe import (
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import unshard
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum
from transformers import PretrainedConfig as HfConfig

_INELIGIBLE_ROUTER_LOGIT = -1e9
_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5
_TOKEN_EMBED_SHARDING = P(None, None)
_LM_HEAD_SHARDING = P("data", "model")
GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY = "grugmoe_artifact_schema_version"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION = 1


class NestedSubsetSchedule(StrEnum):
    """How multi-size nested expert subsets are selected during training."""

    FIXED = "fixed"
    PREFIX = "prefix"
    ROTATING = "rotating"

    @property
    def is_stable(self) -> bool:
        """Whether each nested size reuses one extractable expert subset."""
        return self is not NestedSubsetSchedule.ROTATING


class RouterBalanceMode(StrEnum):
    """How router assignment balance is controlled."""

    ELIGIBILITY_AUX = "eligibility_aux"
    ELIGIBILITY_QB = "eligibility_qb"
    NONE = "none"
    QB = "qb"


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


def _embedding_gather(token_embed: jax.Array, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
    """Look up embeddings locally from a replicated table."""

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


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window), mask.with_sliding_window(None)


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

    GatedNorm and XSA are fixed architecture choices. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 256
    num_experts_per_token: int = 4
    capacity_factor: float = 1.0
    """Expert-parallel dispatch capacity relative to the average assignment load."""
    nested_expert_count: int | None = None
    """Fixed extractable expert subset size.

    Experts are interleaved through the full bank. Rank balance additionally
    requires the expert-parallel axis to divide the smallest fixed subset.
    """
    nested_expert_counts: tuple[int, ...] = ()
    """Expert-subset sizes used by multi-level nesting experiments."""
    nested_subset_schedule: NestedSubsetSchedule = NestedSubsetSchedule.ROTATING
    """Whether each nested size uses one fixed subset or rotates across cosets."""
    nested_batch_fraction: float = 0.0
    """Fraction of training rows restricted to a configured nested subset."""
    router_balance_mode: RouterBalanceMode = RouterBalanceMode.QB
    """Router assignment balancing applied between optimizer updates."""
    router_load_balancing_loss_coef: float = 0.0
    """Weight for eligibility-conditioned router load balancing."""
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
        if self.capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive")
        if not 0.0 <= self.nested_batch_fraction <= 1.0:
            raise ValueError("nested_batch_fraction must be between 0 and 1")
        if self.router_load_balancing_loss_coef < 0.0:
            raise ValueError("router_load_balancing_loss_coef must be non-negative")
        if self.router_balance_mode is RouterBalanceMode.ELIGIBILITY_AUX:
            if self.router_load_balancing_loss_coef == 0.0:
                raise ValueError("eligibility_aux router balancing requires a positive loss coefficient")
            if self.nested_expert_counts and not self.nested_subset_schedule.is_stable:
                raise ValueError("eligibility_aux router balancing requires fixed multi-level expert subsets")
        elif self.router_balance_mode is RouterBalanceMode.ELIGIBILITY_QB:
            if self.router_load_balancing_loss_coef != 0.0:
                raise ValueError("router_load_balancing_loss_coef requires eligibility_aux router balancing")
            if self.nested_expert_counts and not self.nested_subset_schedule.is_stable:
                raise ValueError("eligibility_qb router balancing requires fixed multi-level expert subsets")
        elif self.router_load_balancing_loss_coef != 0.0:
            raise ValueError("router_load_balancing_loss_coef requires eligibility_aux router balancing")
        if self.nested_expert_count is not None and self.nested_expert_counts:
            raise ValueError("configure either nested_expert_count or nested_expert_counts, not both")
        if self.nested_expert_count is None and not self.nested_expert_counts:
            if self.nested_batch_fraction != 0.0:
                raise ValueError("nested_batch_fraction requires a nested expert configuration")
        else:
            nested_counts = self.nested_expert_sizes
            if len(set(nested_counts)) != len(nested_counts):
                raise ValueError("nested expert counts must be unique")
            if tuple(sorted(nested_counts, reverse=True)) != nested_counts:
                raise ValueError("nested expert counts must be in descending order")
            for nested_count in nested_counts:
                if nested_count <= 0 or nested_count >= self.num_experts:
                    raise ValueError("nested expert counts must be positive and smaller than num_experts")
                if self.num_experts % nested_count != 0:
                    raise ValueError("num_experts must be divisible by every nested expert count")
            if self.nested_expert_count is not None and self.nested_expert_count < self.num_experts_per_token + 1:
                raise ValueError("nested_expert_count must exceed num_experts_per_token for QB top-k")
            if self.nested_batch_fraction not in (0.0, 1.0):
                period = round(1.0 / self.nested_batch_fraction)
                if not math.isclose(self.nested_batch_fraction, 1.0 / period):
                    raise ValueError("nested_batch_fraction must be zero, one, or the reciprocal of an integer")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        resolve_moe_implementation(self.moe_implementation)

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
    def router_balance_group_counts(self) -> tuple[int, ...]:
        """Expert counts whose routing biases are balanced independently."""
        if self.router_balance_mode is not RouterBalanceMode.ELIGIBILITY_QB:
            return (self.num_experts,)
        return (self.num_experts, *self.nested_expert_sizes)

    @property
    def nested_expert_sizes(self) -> tuple[int, ...]:
        """Configured nested expert counts in descending order."""
        if self.nested_expert_count is not None:
            return (self.nested_expert_count,)
        return self.nested_expert_counts

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "Transformer":
        cfg = self if Vocab.size == self.vocab_size else dataclasses.replace(self, vocab_size=Vocab.size)
        return Transformer.init(cfg, key=key)

    def hf_checkpoint_converter(
        self,
        ref_checkpoint: str | None = None,
    ) -> HFCheckpointConverter["GrugModelConfig"]:  # type: ignore[type-var]
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=ref_checkpoint,
            HfConfigClass=GrugMoeHfConfig,
            tokenizer=ref_checkpoint,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "GrugModelConfig":
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
            capacity_factor=float(_hf_config_attr(hf_config, ("capacity_factor",), 1.0)),
            nested_expert_count=_hf_config_attr(hf_config, ("nested_expert_count",)),
            nested_expert_counts=tuple(_hf_config_attr(hf_config, ("nested_expert_counts",), ())),
            nested_subset_schedule=NestedSubsetSchedule(
                _hf_config_attr(hf_config, ("nested_subset_schedule",), NestedSubsetSchedule.ROTATING)
            ),
            nested_batch_fraction=float(_hf_config_attr(hf_config, ("nested_batch_fraction",), 0.0)),
            router_balance_mode=RouterBalanceMode(
                _hf_config_attr(hf_config, ("router_balance_mode",), RouterBalanceMode.QB)
            ),
            router_load_balancing_loss_coef=float(_hf_config_attr(hf_config, ("router_load_balancing_loss_coef",), 0.0)),
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
        # One name per field: core fields take the universal transformers spelling, MoE fields the
        # most common public spelling, and grug-specific extras keep their bare names. from_hf_config
        # stays tolerant of the older spellings so existing artifacts keep loading.
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
            "capacity_factor": self.capacity_factor,
            "nested_expert_count": self.nested_expert_count,
            "nested_expert_counts": self.nested_expert_counts,
            "nested_subset_schedule": self.nested_subset_schedule.value,
            "nested_batch_fraction": self.nested_batch_fraction,
            "router_balance_mode": self.router_balance_mode.value,
            "router_load_balancing_loss_coef": self.router_load_balancing_loss_coef,
            "moe_intermediate_size": self.intermediate_dim,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_dim,
            # grug-specific (no public equivalent)
            "qk_mult": self.qk_mult,
            "grugmoe_attention_mode": "production",
            GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY: GRUG_MOE_ARTIFACT_SCHEMA_VERSION,
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
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        # GPU XSA with GQA can give attn_out a backend-specific head sharding;
        # match v to that dynamic sharding before the per-head projection math.
        aligned_v = reshard(aligned_v, _partition_spec_of(attn_out) or P(_BATCH_AXES, None, None, "model"))
        # Exclusive Self Attention: subtract the component of yᵢ parallel to vᵢ.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ, per head.
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
    active_assignments: jax.Array | None = None,
    expert_eligibility: jax.Array | None = None,
    eligibility_group_counts: tuple[int, ...] = (),
) -> dict[str, jax.Array]:
    router_probs_f = router_probs.astype(jnp.float32)
    router_logits_f = router_logits.astype(jnp.float32)
    assignment_weights = (
        jnp.ones_like(selected_experts, dtype=jnp.float32)
        if active_assignments is None
        else active_assignments.astype(jnp.float32)
    )
    expert_counts = jnp.sum(
        jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32) * assignment_weights[..., None],
        axis=(0, 1),
    )
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))
    if expert_eligibility is None or not eligibility_group_counts:
        token_fraction = assignment_fraction * num_experts_per_token
        p = jnp.mean(router_probs_f, axis=0)
        load_balancing_loss = num_experts * jnp.sum(token_fraction * p)
    else:
        token_eligibility = expert_eligibility.astype(jnp.bool_)
        eligible_counts = jnp.sum(token_eligibility, axis=-1)
        assignment_one_hot = (
            jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32) * assignment_weights[..., None]
        )
        weighted_loss = jnp.asarray(0.0, dtype=jnp.float32)
        total_tokens = jnp.asarray(0.0, dtype=jnp.float32)
        for eligible_count in eligibility_group_counts:
            group_mask = eligible_counts == eligible_count
            group_tokens = jnp.sum(group_mask.astype(jnp.float32))
            denominator = jnp.maximum(group_tokens, 1.0)
            group_counts = jnp.sum(
                assignment_one_hot * group_mask[:, None, None],
                axis=(0, 1),
            )
            token_fraction = group_counts / denominator
            p = jnp.sum(router_probs_f * group_mask[:, None], axis=0) / denominator
            group_loss = eligible_count * jnp.sum(token_fraction * p)
            weighted_loss += group_tokens * group_loss
            total_tokens += group_tokens
        load_balancing_loss = weighted_loss / jnp.maximum(total_tokens, 1.0)
    z = jsp.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.mean(z**2)

    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def nested_expert_eligibility(
    num_experts: int,
    nested_expert_count: int,
    subset_schedule: NestedSubsetSchedule = NestedSubsetSchedule.FIXED,
) -> jax.Array:
    """Return the offset-zero eligibility mask for one nested expert subset."""
    if nested_expert_count <= 0:
        raise ValueError("nested_expert_count must be positive")
    if num_experts % nested_expert_count != 0:
        raise ValueError("num_experts must be divisible by nested_expert_count")
    if subset_schedule is NestedSubsetSchedule.PREFIX:
        return jnp.arange(num_experts) < nested_expert_count
    stride = num_experts // nested_expert_count
    return jnp.arange(num_experts) % stride == 0


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
        "qb_beta_per_layer": router_metrics["qb_beta_per_layer"],
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


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert = random.split(key, 2)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e = cfg.hidden_dim, cfg.num_experts
        router_bias_shape = (
            (len(cfg.router_balance_group_counts), e)
            if cfg.router_balance_mode is RouterBalanceMode.ELIGIBILITY_QB
            else (e,)
        )
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros(router_bias_shape),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        expert_eligibility: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        if expert_eligibility is None:
            token_eligibility = jnp.ones_like(router_logits, dtype=jnp.bool_)
        else:
            if expert_eligibility.shape != (b, self.cfg.num_experts):
                raise ValueError(
                    f"expert_eligibility must have shape {(b, self.cfg.num_experts)}, " f"got {expert_eligibility.shape}"
                )
            token_eligibility = jnp.repeat(expert_eligibility.astype(jnp.bool_), s, axis=0)
        eligible_router_logits = jnp.where(token_eligibility, router_logits, _INELIGIBLE_ROUTER_LOGIT)
        eligible_counts = jnp.sum(token_eligibility, axis=-1)
        if self.router_bias.ndim == 1:
            token_router_bias = self.router_bias
        else:
            group_ids = jnp.zeros_like(eligible_counts)
            for group_id, group_count in enumerate(self.cfg.router_balance_group_counts[1:], start=1):
                group_ids = jnp.where(eligible_counts == group_count, group_id, group_ids)
            token_router_bias = self.router_bias.at[group_ids].get(out_sharding=P(_BATCH_AXES, None))
        biased_logits = eligible_router_logits + jax.lax.stop_gradient(token_router_bias)
        router_probs = jax.nn.softmax(eligible_router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        active_assignments = jnp.arange(self.cfg.num_experts_per_token)[None, :] < jnp.minimum(
            eligible_counts[:, None],
            self.cfg.num_experts_per_token,
        )
        token_ids = jnp.arange(selected_experts.shape[0], dtype=selected_experts.dtype)[:, None]
        dispatch_slots = jnp.arange(self.cfg.num_experts_per_token, dtype=selected_experts.dtype)[None, :]
        balanced_padding_experts = (token_ids * self.cfg.num_experts_per_token + dispatch_slots) % self.cfg.num_experts
        selected_experts = reshard(selected_experts, P(None, None))
        active_assignments = reshard(active_assignments, P(None, None))
        balanced_padding_experts = reshard(balanced_padding_experts, P(None, None))
        selected_experts = jnp.where(active_assignments, selected_experts, balanced_padding_experts)
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights_f = jnp.where(active_assignments, jax.nn.sigmoid(unbiased_topk), 0.0)
        # Renormalize K combine weights to sum to ``_ROUTING_RENORM_SUM`` (baked in).
        denom = jnp.sum(combine_weights_f, axis=-1, keepdims=True)
        combine_weights_f = combine_weights_f * (_ROUTING_RENORM_SUM / (denom + 1e-9))
        combine_weights = combine_weights_f.astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            eligible_router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
            active_assignments=active_assignments,
            expert_eligibility=(
                token_eligibility
                if self.cfg.router_balance_mode in (RouterBalanceMode.ELIGIBILITY_AUX, RouterBalanceMode.ELIGIBILITY_QB)
                else None
            ),
            eligibility_group_counts=(self.cfg.num_experts, *self.cfg.nested_expert_sizes),
        )
        if self.cfg.router_balance_mode is RouterBalanceMode.QB:
            # Sharded QB: compute beta locally per device, then average.
            mesh = get_abstract_mesh()
            qb_observation = token_eligibility & (eligible_counts > self.cfg.num_experts_per_token)[:, None]
            s_minus_alpha = reshard(
                jnp.where(qb_observation, eligible_router_logits - qb_alpha, _INELIGIBLE_ROUTER_LOGIT),
                P(_BATCH_AXES, None),
            )
            num_devices = 1
            for a in _BATCH_AXES:
                num_devices *= mesh.shape[a]
            local_tokens = s_minus_alpha.shape[0] // num_devices
            qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

            def _local_qb_beta(s_ma):
                topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
                beta = topk_vals[:, -1]
                beta = jnp.where(jnp.isfinite(beta), beta, 0.0)
                return jax.lax.pmean(beta, axis_name=_BATCH_AXES)

            router_stats["qb_beta"] = shard_map(
                _local_qb_beta,
                mesh=mesh,
                in_specs=(P(_BATCH_AXES, None),),
                out_specs=P(),
            )(s_minus_alpha)
        elif self.cfg.router_balance_mode is RouterBalanceMode.ELIGIBILITY_QB:
            mesh = get_abstract_mesh()
            num_devices = math.prod(mesh.shape[axis_name] for axis_name in _BATCH_AXES)
            local_tokens = eligible_router_logits.shape[0] // num_devices
            group_betas = []
            for group_count in self.cfg.router_balance_group_counts:
                group_experts = nested_expert_eligibility(
                    self.cfg.num_experts,
                    group_count,
                    self.cfg.nested_subset_schedule,
                )
                group_indices = jnp.nonzero(group_experts, size=group_count)[0]
                group_observation = eligible_counts == group_count
                group_scores = reshard(
                    jnp.where(
                        group_observation[:, None],
                        eligible_router_logits[:, group_indices] - qb_alpha,
                        _INELIGIBLE_ROUTER_LOGIT,
                    ),
                    P(_BATCH_AXES, None),
                )
                max_qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // group_count)

                def _local_group_qb_beta(
                    scores,
                    *,
                    group_count=group_count,
                    max_qb_count=max_qb_count,
                ):
                    local_group_tokens = jnp.sum(jnp.any(scores > _INELIGIBLE_ROUTER_LOGIT / 2, axis=-1))
                    qb_count = jnp.clip(
                        local_group_tokens * self.cfg.num_experts_per_token // group_count,
                        1,
                        max_qb_count,
                    )
                    topk_vals, _ = jax.lax.top_k(scores.T, max_qb_count)
                    beta = jax.lax.dynamic_index_in_dim(topk_vals, qb_count - 1, axis=1, keepdims=False)
                    present = local_group_tokens > 0
                    beta = jnp.where(present, beta, 0.0)
                    mean_beta = jax.lax.pmean(beta, axis_name=_BATCH_AXES)
                    mean_presence = jax.lax.pmean(present.astype(jnp.float32), axis_name=_BATCH_AXES)
                    return mean_beta / jnp.maximum(mean_presence, 1.0 / num_devices)

                compact_beta = shard_map(
                    _local_group_qb_beta,
                    mesh=mesh,
                    in_specs=(P(_BATCH_AXES, None),),
                    out_specs=P(),
                )(group_scores)
                group_betas.append(
                    jnp.zeros((self.cfg.num_experts,), dtype=router_logits.dtype).at[group_indices].set(compact_beta)
                )
            router_stats["qb_beta"] = jnp.stack(group_betas)
        else:
            router_stats["qb_beta"] = jnp.zeros_like(self.router_bias)

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
        use_pko: bool = False,
        disable_rope: bool = False,
        expert_eligibility: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(attn_in, mask, use_pko=use_pko, disable_rope=disable_rope)
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in, expert_eligibility=expert_eligibility)
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
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), _TOKEN_EMBED_SHARDING
        )
        output_proj = reshard(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), _LM_HEAD_SHARDING
        )
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

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        expert_eligibility: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        hidden = _embedding_gather(self.token_embed, token_ids)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        if not isinstance(mask, AttentionMask):
            raise NotImplementedError("Grug MoE requires a structured attention mask.")
        short_mask, long_mask = _layer_attention_masks(mask, sliding_window=cfg.sliding_window)
        if cfg.attention_implementation == "gpu_fa4_cute":
            batch_size, seq_len = hidden.shape[:2]
            long_lower_bounds, valid = fa4_cute_segment_bounds(
                long_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                sliding_window=None,
            )
            short_lower_bounds, _ = fa4_cute_segment_bounds(
                short_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                sliding_window=cfg.sliding_window,
            )
            valid = _batch_reshard(valid)
            long_mask = long_mask.with_fa4_bounds(_batch_reshard(long_lower_bounds), valid)
            short_mask = short_mask.with_fa4_bounds(_batch_reshard(short_lower_bounds), valid)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        num_blocks = len(self.blocks)
        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            is_last = i == num_blocks - 1
            is_long = i % 4 == 3 or is_last
            layer_mask = long_mask if is_long else short_mask
            use_pko = is_long and not cfg.disable_pko
            disable_rope = is_long and cfg.disable_long_rope
            hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(
                hidden, layer_mask, use_pko, disable_rope, expert_eligibility
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
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        expert_eligibility: jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask, expert_eligibility=expert_eligibility)
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
        expert_eligibility: jax.Array | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(token_ids, mask=mask, expert_eligibility=expert_eligibility)
        labels = jnp.zeros_like(token_ids).at[:, :-1].set(token_ids[:, 1:]).astype(jnp.int32)
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
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        load_balancing_loss = jnp.sum(router_metrics["load_balancing_loss_per_layer"]) / num_moe_layers
        aux_loss = (
            self.config.router_z_loss_coef * rzl + self.config.router_load_balancing_loss_coef * load_balancing_loss
        )
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            if self.config.nested_expert_count is not None:
                nested_experts = nested_expert_eligibility(
                    self.config.num_experts,
                    self.config.nested_expert_count,
                    self.config.nested_subset_schedule,
                )
                routing_counts = router_metrics["routing_counts_per_layer"].astype(jnp.float32)
                core_counts = jnp.sum(routing_counts * nested_experts[None, :], axis=-1)
                outer_counts = jnp.sum(routing_counts * ~nested_experts[None, :], axis=-1)
                core_mean = core_counts / self.config.nested_expert_count
                outer_mean = outer_counts / (self.config.num_experts - self.config.nested_expert_count)
                nested_counts = jnp.where(nested_experts[None, :], routing_counts, jnp.nan)
                outer_expert_counts = jnp.where(~nested_experts[None, :], routing_counts, jnp.nan)
                summarized_metrics["train/router/nested_assignment_fraction"] = jnp.mean(
                    core_counts / jnp.maximum(core_counts + outer_counts, 1.0)
                )
                summarized_metrics["train/router/outer_to_nested_assignment_ratio"] = jnp.mean(
                    outer_mean / jnp.maximum(core_mean, 1.0)
                )
                summarized_metrics["train/router/nested_assignment_cv"] = jnp.nanmean(
                    jnp.nanstd(nested_counts, axis=-1) / jnp.maximum(core_mean, 1.0)
                )
                summarized_metrics["train/router/outer_assignment_cv"] = jnp.nanmean(
                    jnp.nanstd(outer_expert_counts, axis=-1) / jnp.maximum(outer_mean, 1.0)
                )
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            return loss, summarized_metrics
        return loss


def extract_nested_expert_model(model: Transformer, nested_expert_count: int | None = None) -> Transformer:
    """Compact one configured nested expert subset into a standalone model.

    Args:
        model: Source model containing the full expert bank.
        nested_expert_count: Nested size to extract. This is required for a
            multi-level chain and inferred for a single configured subset.
    """
    configured_counts = model.config.nested_expert_sizes
    if nested_expert_count is None and len(configured_counts) == 1:
        nested_expert_count = configured_counts[0]
    if nested_expert_count is None:
        raise ValueError("nested_expert_count is required for a multi-level nested model")
    if nested_expert_count not in configured_counts:
        raise ValueError("nested_expert_count must be one of the model's configured nested sizes")

    nested_experts = nested_expert_eligibility(
        model.config.num_experts,
        nested_expert_count,
        model.config.nested_subset_schedule,
    )
    nested_indices = jnp.nonzero(nested_experts, size=nested_expert_count)[0]
    extracted_config = dataclasses.replace(
        model.config,
        num_experts=nested_expert_count,
        nested_expert_count=None,
        nested_expert_counts=(),
        nested_batch_fraction=0.0,
        router_balance_mode=(
            RouterBalanceMode.QB
            if model.config.router_balance_mode is RouterBalanceMode.ELIGIBILITY_QB
            else model.config.router_balance_mode
        ),
    )

    extracted_blocks: list[Block] = []
    for block in model.blocks:
        expert_mlp = MoEExpertMlp(
            w_gate=block.mlp.expert_mlp.w_gate.at[nested_indices].get(out_sharding=P("expert", "data", "model")),
            w_up=block.mlp.expert_mlp.w_up.at[nested_indices].get(out_sharding=P("expert", "data", "model")),
            w_down=block.mlp.expert_mlp.w_down.at[nested_indices].get(out_sharding=P("expert", "model", "data")),
            implementation=block.mlp.expert_mlp.implementation,
            activation=block.mlp.expert_mlp.activation,
            capacity_factor=block.mlp.expert_mlp.capacity_factor,
        )
        source_router_bias = block.mlp.router_bias
        if source_router_bias.ndim == 2:
            group_index = configured_counts.index(nested_expert_count) + 1
            source_router_bias = source_router_bias[group_index]
        mlp = MoEMLP(
            router=block.mlp.router.at[:, nested_indices].get(out_sharding=P(None, None)),
            router_bias=source_router_bias.at[nested_indices].get(out_sharding=P(None)),
            expert_mlp=expert_mlp,
            cfg=extracted_config,
        )
        extracted_blocks.append(
            dataclasses.replace(
                block,
                attn=dataclasses.replace(block.attn, cfg=extracted_config),
                mlp=mlp,
            )
        )

    return dataclasses.replace(
        model,
        blocks=tuple(extracted_blocks),
        config=extracted_config,
    )


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
                f"{layer_prefix}.mlp.router.bias": (
                    block.mlp.router_bias[0] if block.mlp.router_bias.ndim == 2 else block.mlp.router_bias
                ),
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_up),
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
    "extract_nested_expert_model",
    "grugmoe_inference_state_dict",
    "nested_expert_eligibility",
]
