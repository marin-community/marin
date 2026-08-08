# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

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
from transformers import PretrainedConfig as HfConfig

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5
_SELECTED_EXPERTS_KEY = "selected_experts"
GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY = "grugmoe_artifact_schema_version"
GRUG_MOE_ARTIFACT_SCHEMA_VERSION = 1


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


@dataclass(frozen=True)
class _BlockInitKeys:
    attention: PRNGKeyArray
    router: PRNGKeyArray
    expert_bank: PRNGKeyArray
    shared: PRNGKeyArray
    attention_gated_norm: PRNGKeyArray
    mlp_gated_norm: PRNGKeyArray
    expert_adapter: PRNGKeyArray


def _block_init_keys(key: PRNGKeyArray) -> _BlockInitKeys:
    attention, mlp, shared, attention_gated_norm, mlp_gated_norm = random.split(key, 5)
    router, expert_bank = random.split(mlp, 2)
    return _BlockInitKeys(
        attention=attention,
        router=router,
        expert_bank=expert_bank,
        shared=shared,
        attention_gated_norm=attention_gated_norm,
        mlp_gated_norm=mlp_gated_norm,
        # Keep the existing split unchanged so enabling adapters cannot perturb
        # any pre-existing parameter initialization.
        expert_adapter=random.fold_in(key, 0xADAF7),
    )


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


def _partition_spec_of(x: jax.Array) -> P | None:
    sharding = jax.typeof(x).sharding if isinstance(x, core.Tracer) else x.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.spec
    return None


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

    GatedNorm, XSA, and QB routing are fixed. Shape, attention, and routed-expert
    sharing topology are configurable. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 6
    expert_bank_for_layer: tuple[int, ...] | None = None
    """Routed-expert bank ID used by each layer. ``None`` gives one bank per layer."""
    expert_adapter_rank_for_layer: tuple[int, ...] | None = None
    """Rank of the routed-expert input/output adapter at each layer. ``None`` disables adapters."""
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
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
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
        bank_for_layer = self.resolved_expert_bank_for_layer
        bank_ids = set(bank_for_layer)
        expected_bank_ids = set(range(len(bank_ids)))
        if bank_ids != expected_bank_ids:
            raise ValueError(
                "expert_bank_for_layer must use contiguous bank IDs starting at zero; " f"got {bank_for_layer}"
            )
        _ = self.resolved_expert_adapter_rank_for_layer
        resolve_moe_implementation(self.moe_implementation)

    @property
    def resolved_expert_bank_for_layer(self) -> tuple[int, ...]:
        if self.expert_bank_for_layer is None:
            return tuple(range(self.num_layers))
        if len(self.expert_bank_for_layer) != self.num_layers:
            raise ValueError(
                "expert_bank_for_layer must contain one bank ID per layer; "
                f"got {len(self.expert_bank_for_layer)} entries for {self.num_layers} layers"
            )
        if any(bank_id < 0 for bank_id in self.expert_bank_for_layer):
            raise ValueError("expert_bank_for_layer bank IDs must be non-negative")
        return self.expert_bank_for_layer

    @property
    def expert_bank_group_sizes(self) -> tuple[int, ...]:
        bank_for_layer = self.resolved_expert_bank_for_layer
        return tuple(bank_for_layer.count(bank_id) for bank_id in range(max(bank_for_layer) + 1))

    @property
    def resolved_expert_adapter_rank_for_layer(self) -> tuple[int, ...]:
        if self.expert_adapter_rank_for_layer is None:
            return (0,) * self.num_layers
        if len(self.expert_adapter_rank_for_layer) != self.num_layers:
            raise ValueError(
                "expert_adapter_rank_for_layer must contain one rank per layer; "
                f"got {len(self.expert_adapter_rank_for_layer)} entries for {self.num_layers} layers"
            )
        if any(rank < 0 for rank in self.expert_adapter_rank_for_layer):
            raise ValueError("expert_adapter_rank_for_layer ranks must be non-negative")
        return self.expert_adapter_rank_for_layer

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
        expert_bank_for_layer = _hf_config_attr(hf_config, ("expert_bank_for_layer",))
        expert_adapter_rank_for_layer = _hf_config_attr(hf_config, ("expert_adapter_rank_for_layer",))
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
            expert_bank_for_layer=(
                tuple(int(bank_id) for bank_id in expert_bank_for_layer) if expert_bank_for_layer is not None else None
            ),
            expert_adapter_rank_for_layer=(
                tuple(int(rank) for rank in expert_adapter_rank_for_layer)
                if expert_adapter_rank_for_layer is not None
                else None
            ),
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
            "moe_intermediate_size": self.intermediate_dim,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_dim,
            "expert_bank_for_layer": list(self.resolved_expert_bank_for_layer),
            "expert_adapter_rank_for_layer": list(self.resolved_expert_adapter_rank_for_layer),
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
    activation_norm = router_metrics["activation_norm_per_layer"]
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
        "tying/top1_cross_loop_agreement": router_metrics["top1_cross_loop_agreement"],
        "tying/topk_set_overlap": router_metrics["topk_set_overlap"],
        "qb_beta_per_layer": router_metrics["qb_beta_per_layer"],
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/layer_{i}/capacity_overflow_rate"] = capacity_overflow_rate[i]
        out[f"tying/activation_norm_by_layer/layer_{i}"] = activation_norm[i]
    return out


class CrossLoopAgreement(NamedTuple):
    top1: jax.Array
    topk_set_overlap: jax.Array


def _cross_loop_agreement(
    router_stats: list[dict[str, jax.Array]],
    bank_for_layer: tuple[int, ...],
) -> CrossLoopAgreement:
    """Return top-1 agreement and normalized top-k overlap across tied layers.

    Both values are NaN when no bank is reused because cross-layer agreement is
    undefined for an untied topology.
    """
    top1_agreements: list[jax.Array] = []
    topk_overlaps: list[jax.Array] = []
    for bank_id in range(max(bank_for_layer) + 1):
        layer_indices = [layer_index for layer_index, layer_bank in enumerate(bank_for_layer) if layer_bank == bank_id]
        for first_offset, first_layer in enumerate(layer_indices):
            first_selected = router_stats[first_layer][_SELECTED_EXPERTS_KEY]
            for second_layer in layer_indices[first_offset + 1 :]:
                second_selected = router_stats[second_layer][_SELECTED_EXPERTS_KEY]
                top1_agreements.append(jnp.mean(first_selected[:, 0] == second_selected[:, 0]))
                intersection_size = jnp.sum(
                    jnp.any(first_selected[:, :, None] == second_selected[:, None, :], axis=-1),
                    axis=-1,
                )
                topk_overlaps.append(jnp.mean(intersection_size / first_selected.shape[-1]))

    if not top1_agreements:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return CrossLoopAgreement(top1=nan, topk_set_overlap=nan)
    return CrossLoopAgreement(
        top1=jnp.mean(jnp.stack(top1_agreements)),
        topk_set_overlap=jnp.mean(jnp.stack(topk_overlaps)),
    )


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
    """Per-layer QB router and dispatch logic for an explicit expert bank."""

    router: jax.Array
    router_bias: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e = cfg.hidden_dim, cfg.num_experts
        return MoEMLP(
            router=reshard(_init_weight(key, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            cfg=cfg,
        )

    @named_call
    def route(self, x: Float[Array, "B S D"]) -> "MoERouting":
        """Compute routed expert IDs, combine weights, and QB statistics."""
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
        router_stats[_SELECTED_EXPERTS_KEY] = selected_experts
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

        return MoERouting(
            x_flat=x_flat,
            selected_experts=selected_experts,
            combine_weights=combine_weights,
            router_stats=router_stats,
        )

    @named_call
    def forward_with_trace(
        self,
        x: Float[Array, "B S D"],
        expert_bank: MoEExpertMlp,
        expert_adapter: "RoutedExpertAdapter | None" = None,
    ) -> "MoEForwardTrace":
        b, s, _ = x.shape
        routing = self.route(x)
        expert_input = routing.x_flat
        if expert_adapter is not None:
            expert_input = expert_adapter.adapt_input(expert_input)

        routed_flat, dropped_assignments = expert_bank(
            expert_input,
            routing.selected_experts.astype(jnp.int32),
            routing.combine_weights,
            mesh=get_abstract_mesh(),
            report_capacity_overflow=True,
        )
        if expert_adapter is not None:
            routed_flat = expert_adapter.adapt_output(routed_flat)
        router_stats = routing.router_stats
        router_stats["capacity_overflow"] = dropped_assignments.astype(jnp.float32)

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return MoEForwardTrace(routed_output=routed, routing=routing, router_stats=router_stats)

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        expert_bank: MoEExpertMlp,
        expert_adapter: "RoutedExpertAdapter | None" = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        trace = self.forward_with_trace(x, expert_bank, expert_adapter)
        return trace.routed_output, trace.router_stats


class MoERouting(NamedTuple):
    x_flat: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array
    router_stats: dict[str, jax.Array]


class MoEForwardTrace(NamedTuple):
    routed_output: jax.Array
    routing: MoERouting
    router_stats: dict[str, jax.Array]


def _init_expert_bank(cfg: GrugModelConfig, *, block_key: PRNGKeyArray) -> MoEExpertMlp:
    return MoEExpertMlp.init(
        num_experts=cfg.num_experts,
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        initializer_std=cfg.initializer_std,
        key=_block_init_keys(block_key).expert_bank,
        implementation=cfg.moe_implementation,
        activation=ActivationFunctionEnum.silu,
        capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
    )


class RoutedExpertAdapter(eqx.Module):
    """Layer-specific low-rank residual maps around a routed shared expert bank."""

    input_a: jax.Array
    input_b: jax.Array
    output_a: jax.Array
    output_b: jax.Array

    @staticmethod
    def init(hidden_dim: int, rank: int, *, key: PRNGKeyArray) -> "RoutedExpertAdapter":
        if rank <= 0:
            raise ValueError(f"adapter rank must be positive, got {rank}")
        input_key, output_key = random.split(key)
        a_std = hidden_dim**-0.5
        return RoutedExpertAdapter(
            input_a=reshard(_init_weight(input_key, (hidden_dim, rank), a_std), P(None, None)),
            input_b=reshard(jnp.zeros((rank, hidden_dim)), P(None, None)),
            output_a=reshard(_init_weight(output_key, (hidden_dim, rank), a_std), P(None, None)),
            output_b=reshard(jnp.zeros((rank, hidden_dim)), P(None, None)),
        )

    def adapt_input(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        hidden = jnp.einsum("...d,dr->...r", x, self.input_a)
        correction = jnp.einsum("...r,rd->...d", hidden, self.input_b)
        return x + correction.astype(x.dtype)

    def adapt_output(self, y: Float[Array, "... D"]) -> Float[Array, "... D"]:
        hidden = jnp.einsum("...d,dr->...r", y, self.output_a)
        correction = jnp.einsum("...r,rd->...d", hidden, self.output_b)
        return y + correction.astype(y.dtype)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None
    routed_expert_adapter: RoutedExpertAdapter | None
    expert_bank_index: int = eqx.field(static=True)

    @staticmethod
    def init(
        cfg: GrugModelConfig,
        *,
        expert_bank_index: int,
        expert_adapter_rank: int,
        key: PRNGKeyArray,
    ) -> "Block":
        keys = _block_init_keys(key)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=keys.shared
            )
        routed_expert_adapter = (
            RoutedExpertAdapter.init(cfg.hidden_dim, expert_adapter_rank, key=keys.expert_adapter)
            if expert_adapter_rank > 0
            else None
        )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=keys.attention_gated_norm),
            attn=CausalSelfAttention.init(cfg, key=keys.attention),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=keys.mlp_gated_norm),
            mlp=MoEMLP.init(cfg, key=keys.router),
            shared=shared,
            routed_expert_adapter=routed_expert_adapter,
            expert_bank_index=expert_bank_index,
        )

    @named_call
    def forward_with_moe_trace(
        self,
        x: Float[Array, "B S D"],
        expert_bank: MoEExpertMlp,
        options: "BlockCallOptions",
    ) -> "MoeBlockTrace":
        """Run a block and expose the exact routed-MoE calibration boundary."""
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(
            attn_in,
            options.mask,
            use_pko=options.use_pko,
            disable_rope=options.disable_rope,
        )
        mlp_input = self.mlp_gated_norm(self.rms_mlp(x))
        moe_trace = self.mlp.forward_with_trace(mlp_input, expert_bank, self.routed_expert_adapter)
        mlp_output = moe_trace.routed_output
        if self.shared is not None:
            mlp_output = mlp_output + self.shared(mlp_input, activation=ActivationFunctionEnum.silu)
        return MoeBlockTrace(
            hidden=x + mlp_output,
            mlp_input=mlp_input,
            selected_experts=moe_trace.routing.selected_experts,
            combine_weights=moe_trace.routing.combine_weights,
            routed_output=moe_trace.routed_output,
            router_stats=moe_trace.router_stats,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        expert_bank: MoEExpertMlp,
        options: "BlockCallOptions",
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        trace = self.forward_with_moe_trace(x, expert_bank, options)
        return trace.hidden, trace.router_stats


class MoeBlockTrace(NamedTuple):
    hidden: jax.Array
    mlp_input: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array
    routed_output: jax.Array
    router_stats: dict[str, jax.Array]


class BlockCallOptions(NamedTuple):
    mask: AttentionMask
    use_pko: bool
    disable_rope: bool


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    expert_banks: tuple[MoEExpertMlp, ...]
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
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        bank_for_layer = cfg.resolved_expert_bank_for_layer
        adapter_rank_for_layer = cfg.resolved_expert_adapter_rank_for_layer
        blocks = tuple(
            Block.init(
                cfg,
                expert_bank_index=bank_for_layer[i],
                expert_adapter_rank=adapter_rank_for_layer[i],
                key=block_keys[i],
            )
            for i in range(cfg.num_layers)
        )
        first_layer_for_bank = tuple(bank_for_layer.index(bank_id) for bank_id in range(max(bank_for_layer) + 1))
        expert_banks = tuple(
            _init_expert_bank(cfg, block_key=block_keys[layer_index]) for layer_index in first_layer_for_bank
        )
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            expert_banks=expert_banks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    def embed_inputs(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        hidden = self.token_embed.at[token_ids].get(out_sharding=_batch_spec())
        return self.embed_gated_norm(self.embed_norm(hidden))

    def block_call_options(
        self,
        mask: AttentionMask | jax.Array,
        layer_index: int,
    ) -> BlockCallOptions:
        """Resolve the attention mask and long-layer switches for one block."""
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        is_last = layer_index == len(self.blocks) - 1
        is_long = layer_index % 4 == 3 or is_last
        layer_mask = AttentionMask(
            is_causal=True,
            sliding_window=None if is_long else self.config.sliding_window,
            segment_ids=segment_ids,
        )
        return BlockCallOptions(
            mask=layer_mask,
            use_pko=is_long and not self.config.disable_pko,
            disable_rope=is_long and self.config.disable_long_rope,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        hidden = self.embed_inputs(token_ids)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        moe_router_stats: list[dict[str, jax.Array]] = []
        activation_norms: list[jax.Array] = []
        for i, block in enumerate(self.blocks):
            options = self.block_call_options(mask, i)
            expert_bank = self.expert_banks[block.expert_bank_index]
            hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(
                hidden,
                expert_bank,
                options,
            )
            moe_router_stats.append(router_stats)
            activation_norms.append(jnp.sqrt(jnp.mean(jnp.square(hidden.astype(jnp.float32)))))

        cross_loop_agreement = _cross_loop_agreement(
            moe_router_stats,
            cfg.resolved_expert_bank_for_layer,
        )

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
            "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
            "activation_norm_per_layer": jnp.stack(activation_norms, axis=0),
            "top1_cross_loop_agreement": cross_loop_agreement.top1,
            "topk_set_overlap": cross_loop_agreement.topk_set_overlap,
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
        final_label = jnp.zeros_like(token_ids[:, :1])
        if not get_abstract_mesh().empty:
            final_label = jax.sharding.reshard(final_label, _batch_spec())
        labels = jnp.concatenate([token_ids[:, 1:], final_label], axis=1).astype(jnp.int32)
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
        expert_bank = model.expert_banks[block.expert_bank_index]
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
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(expert_bank.w_gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(expert_bank.w_up),
                f"{layer_prefix}.mlp.experts.down_proj.weight": _linear_inference_tensor(expert_bank.w_down),
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
        if block.routed_expert_adapter is not None:
            adapter = block.routed_expert_adapter
            tensors.update(
                {
                    f"{layer_prefix}.mlp.expert_adapter.input_a.weight": _linear_inference_tensor(adapter.input_a),
                    f"{layer_prefix}.mlp.expert_adapter.input_b.weight": _linear_inference_tensor(adapter.input_b),
                    f"{layer_prefix}.mlp.expert_adapter.output_a.weight": _linear_inference_tensor(adapter.output_a),
                    f"{layer_prefix}.mlp.expert_adapter.output_b.weight": _linear_inference_tensor(adapter.output_b),
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
    "RoutedExpertAdapter",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "grugmoe_inference_state_dict",
]
