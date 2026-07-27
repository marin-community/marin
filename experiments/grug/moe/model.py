# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call, tree_checkpoint_name
from haliax.quantization import Fp8RaggedDotOp
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
    CHECKPOINT_ROUTER_COMBINE_WEIGHTS,
    CHECKPOINT_ROUTER_SELECTED_EXPERTS,
    CHECKPOINT_ROUTER_SIGMOID_WEIGHTS,
    CHECKPOINT_ROUTER_UNBIASED_TOPK,
    CHECKPOINT_ROUTER_WEIGHT_DENOMINATOR,
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    MoeRaggedDotOps,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum
from transformers import PretrainedConfig as HfConfig

try:
    import jaxpp.api as jaxpp
except ModuleNotFoundError:
    jaxpp = None

GRUG_MOE_EP_CAPACITY_FACTOR = 1.0
GRUG_MOE_NCCL_EP_DROP_CAPACITY_FACTOR = 1.25
_GATED_NORM_RANK = 128
_ROUTING_RENORM_SUM = 2.5
_FP8_EXPERT_GEMM_ALIGNMENT = 128
_EFFECTFUL_MOE_IMPLEMENTATIONS = ("deepep", "ubx")
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


RematMode = Literal["recompute_all", "save_moe"]
Fp8ExpertGemmRevDtype = Literal["e4m3", "e5m2"]


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


def _pipeline_stage_end_layers(
    num_blocks: int,
    pipeline_stages: int | None,
    stage_layer_counts: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if pipeline_stages is None:
        if stage_layer_counts is not None:
            raise ValueError("stage_layer_counts requires pipeline_stages")
        return ()
    if pipeline_stages <= 0:
        raise ValueError(f"pipeline_stages must be positive, got {pipeline_stages}")
    if pipeline_stages > num_blocks:
        raise ValueError(f"pipeline_stages ({pipeline_stages}) must be <= num_layers ({num_blocks})")
    if stage_layer_counts is None:
        base_layers_per_stage, extra_layers = divmod(num_blocks, pipeline_stages)
        stage_layer_counts = tuple(
            base_layers_per_stage + (1 if stage_index < extra_layers else 0) for stage_index in range(pipeline_stages)
        )
    else:
        if len(stage_layer_counts) != pipeline_stages:
            raise ValueError(
                f"stage_layer_counts must have one entry per pipeline stage; "
                f"got {len(stage_layer_counts)} counts for {pipeline_stages} stages"
            )
        if any(layer_count <= 0 for layer_count in stage_layer_counts):
            raise ValueError(f"stage_layer_counts must be positive, got {stage_layer_counts}")
        if sum(stage_layer_counts) != num_blocks:
            raise ValueError(
                f"stage_layer_counts must sum to num_layers={num_blocks}, "
                f"got {stage_layer_counts} (sum={sum(stage_layer_counts)})"
            )
    stage_end_layers = []
    layer_count = 0
    for stage_size in stage_layer_counts:
        layer_count += stage_size
        stage_end_layers.append(layer_count - 1)
    return tuple(stage_end_layers)


def _pipeline_stage_bounds(
    num_blocks: int,
    pipeline_stages: int,
    stage_layer_counts: tuple[int, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    stage_end_layers = _pipeline_stage_end_layers(num_blocks, pipeline_stages, stage_layer_counts)
    start = 0
    bounds = []
    for end_layer in stage_end_layers:
        end = end_layer + 1
        bounds.append((start, end))
        start = end
    return tuple(bounds)


def _mark_pipeline_stage_end(hidden: jax.Array, *, layer_index: int) -> jax.Array:
    if jaxpp is None:
        raise ModuleNotFoundError(
            "jaxpp is required when pipeline_stages is set. Install NVIDIA/jaxpp in the training environment."
        )
    return jaxpp.mark_stage_end(hidden, name=f"grug_moe_layer_{layer_index}")


class GrugMoeHfConfig(HfConfig):
    model_type = GRUG_MOE_MODEL_TYPE


def _hf_config_attr(config: HfConfig, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


@dataclass(frozen=True)
class ResearchFp8ExpertGemmConfig:
    """Research-only delayed-scaling FP8 for the two routed expert GEMMs.

    This path uses Hopper-only Mosaic GPU kernels and is not a portable model
    default. The E4M3 reverse dtype matches the benchmarked JAX 0.10.x setup.
    """

    amax_history_length: int = 1024
    rev_dtype: Fp8ExpertGemmRevDtype = "e4m3"

    def __post_init__(self) -> None:
        if self.amax_history_length <= 0:
            raise ValueError("FP8 expert GEMM amax_history_length must be positive")
        if self.rev_dtype not in ("e4m3", "e5m2"):
            raise ValueError(f"unknown FP8 expert GEMM reverse dtype: {self.rev_dtype}")


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
    research_fp8_expert_gemm: ResearchFp8ExpertGemmConfig | None = None
    """Opt-in research FP8 expert GEMMs; None preserves the BF16 ring path."""
    loss_implementation: str | tuple[str, ...] | None = None
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
        if self.research_fp8_expert_gemm is not None:
            if resolve_moe_implementation(self.moe_implementation) != "ring":
                raise ValueError("research FP8 expert GEMMs require moe_implementation='ring'")
            if (
                self.hidden_dim % _FP8_EXPERT_GEMM_ALIGNMENT != 0
                or self.intermediate_dim % _FP8_EXPERT_GEMM_ALIGNMENT != 0
            ):
                raise ValueError(
                    "research FP8 expert GEMMs require hidden_dim and intermediate_dim divisible by "
                    f"{_FP8_EXPERT_GEMM_ALIGNMENT}"
                )

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
        # A selectively preserved FP32 master gate keeps its cancellation-heavy
        # gradient out of BF16 while leaving the activation wire dtype unchanged.
        if self.attn_gate.dtype == jnp.float32 and x.dtype != jnp.float32:
            gate_logits = jnp.einsum("bsd,dn->bsn", x.astype(jnp.float32), self.attn_gate)
            gate = (2 * jax.nn.sigmoid(gate_logits)).astype(attn_out.dtype)[..., None]
        else:
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


def _stack_router_metrics(moe_router_stats: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    if not moe_router_stats:
        raise ValueError("at least one block is required to produce router metrics")
    return {
        "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
        "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
        "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
        "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
        "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
        "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
    }


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
        moe_implementation = resolve_moe_implementation(cfg.moe_implementation)

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        ragged_dot_ops = None
        if cfg.research_fp8_expert_gemm is not None:
            reverse_dtype = {
                "e4m3": jnp.float8_e4m3fn,
                "e5m2": jnp.float8_e5m2,
            }[cfg.research_fp8_expert_gemm.rev_dtype]
            ragged_dot_ops = MoeRaggedDotOps(
                w13=Fp8RaggedDotOp.init(
                    amax_history_length=cfg.research_fp8_expert_gemm.amax_history_length,
                    rev_dtype=reverse_dtype,
                ),
                w2=Fp8RaggedDotOp.init(
                    amax_history_length=cfg.research_fp8_expert_gemm.amax_history_length,
                    rev_dtype=reverse_dtype,
                ),
            )

        d, e = cfg.hidden_dim, cfg.num_experts
        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=reshard(jnp.zeros((e,)), P(None)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=(
                    GRUG_MOE_NCCL_EP_DROP_CAPACITY_FACTOR
                    if moe_implementation == "nccl_ep_drop"
                    else GRUG_MOE_EP_CAPACITY_FACTOR
                ),
                ragged_dot_ops=ragged_dot_ops,
            ),
            cfg=cfg,
        )

    @named_call
    def route(
        self,
        x: Float[Array, "B S D"],
    ) -> "MoERoutingState":
        x_flat = rearrange(x, "b s d -> (b s) d")
        # Keep every expert score visible on each token shard. In particular,
        # top-k indices must remain global expert IDs when the token axis is
        # sharded over the expert mesh axis.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        router_logits = reshard(router_logits, P(_BATCH_AXES, None))
        biased_logits = router_logits + jax.lax.stop_gradient(reshard(self.router_bias, P(None)))
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K+1) on biased logits; the (K+1)-th is the QB threshold alpha.
        _topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = _topk_logits[:, -1:]
        selected_experts = tree_checkpoint_name(
            reshard(selected_experts[:, :-1], P(_BATCH_AXES, None)),
            CHECKPOINT_ROUTER_SELECTED_EXPERTS,
        )
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = tree_checkpoint_name(
            jnp.take_along_axis(router_logits, selected_experts, axis=-1),
            CHECKPOINT_ROUTER_UNBIASED_TOPK,
        )
        sigmoid_weights = tree_checkpoint_name(
            jax.nn.sigmoid(unbiased_topk),
            CHECKPOINT_ROUTER_SIGMOID_WEIGHTS,
        )
        # Renormalize K combine weights to sum to ``_ROUTING_RENORM_SUM`` (baked in).
        denom = tree_checkpoint_name(
            jnp.sum(sigmoid_weights, axis=-1, keepdims=True),
            CHECKPOINT_ROUTER_WEIGHT_DENOMINATOR,
        )
        combine_weights_f = tree_checkpoint_name(
            sigmoid_weights * (_ROUTING_RENORM_SUM / (denom + 1e-9)),
            CHECKPOINT_ROUTER_COMBINE_WEIGHTS,
        )
        combine_dtype = jnp.float32 if self.expert_mlp.implementation == "ubx" else x.dtype
        combine_weights = reshard(combine_weights_f.astype(combine_dtype), P(_BATCH_AXES, None))
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
        return MoERoutingState(
            router_logits=router_logits,
            selected_experts=selected_experts.astype(jnp.int32),
            combine_weights=combine_weights,
            boundary_margin=_topk_logits[:, self.cfg.num_experts_per_token - 1] - _topk_logits[:, -1],
            router_stats=router_stats,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        return apply_moe_routing(self.expert_mlp, x, self.route(x))

    @named_call
    def accumulating_weight_gradient(
        self,
        x: Float[Array, "B S D"],
        w13_accumulator: Float[Array, "E D I2"],
        w2_accumulator: Float[Array, "E I D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
        """Run routed experts with data-local FP32 expert-gradient accumulation."""
        return apply_moe_routing_accumulating_weight_gradient(
            self.expert_mlp,
            x,
            self.route(x),
            w13_accumulator,
            w2_accumulator,
        )


class MoERoutingState(eqx.Module):
    """Transient router output consumed by an explicit expert task."""

    router_logits: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array
    boundary_margin: jax.Array
    router_stats: dict[str, jax.Array]


def apply_moe_routing(
    expert_mlp: MoEExpertMlp,
    x: Float[Array, "B S D"],
    routing: MoERoutingState,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    """Run experts from explicit router state without recomputing routing."""
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    routed_flat, dropped_assignments = expert_mlp(
        x_flat,
        routing.selected_experts,
        routing.combine_weights,
        mesh=get_abstract_mesh(),
        report_capacity_overflow=True,
    )
    router_stats = dict(routing.router_stats)
    router_stats["capacity_overflow"] = dropped_assignments.astype(jnp.float32)
    routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
    return reshard(routed, _batch_spec()), router_stats


def apply_moe_routing_accumulating_weight_gradient(
    expert_mlp: MoEExpertMlp,
    x: Float[Array, "B S D"],
    routing: MoERoutingState,
    w13_accumulator: Float[Array, "E D I2"],
    w2_accumulator: Float[Array, "E I D"],
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
    """Run explicit routing while accumulating data-local FP32 expert gradients.

    The returned token must be added to the normalized loss with coefficient
    exactly one.
    """
    b, s, _ = x.shape
    x_flat = rearrange(x, "b s d -> (b s) d")
    routed_flat, dropped_assignments, accumulation_token = expert_mlp.accumulating_weight_gradient(
        x_flat,
        routing.selected_experts,
        routing.combine_weights,
        w13_accumulator,
        w2_accumulator,
        mesh=get_abstract_mesh(),
    )
    router_stats = dict(routing.router_stats)
    router_stats["capacity_overflow"] = dropped_assignments.astype(jnp.float32)
    routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
    return reshard(routed, _batch_spec()), router_stats, accumulation_token


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
    def attention_residual(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        disable_rope: bool = False,
    ) -> Float[Array, "B S D"]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        return x + self.attn(attn_in, mask, use_pko=use_pko, disable_rope=disable_rope)

    @named_call
    def moe_residual(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats

    @named_call
    def moe_residual_accumulating_weight_gradient(
        self,
        x: Float[Array, "B S D"],
        w13_accumulator: Float[Array, "E D I2"],
        w2_accumulator: Float[Array, "E I D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats, accumulation_token = self.mlp.accumulating_weight_gradient(
            mlp_in,
            w13_accumulator,
            w2_accumulator,
        )
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        return x + mlp_out, router_stats, accumulation_token

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        disable_rope: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        x = self.attention_residual(x, mask, use_pko=use_pko, disable_rope=disable_rope)
        return self.moe_residual(x)

    @named_call
    def accumulating_weight_gradient(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        w13_accumulator: Float[Array, "E D I2"],
        w2_accumulator: Float[Array, "E I D"],
        use_pko: bool = False,
        disable_rope: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
        x = self.attention_residual(x, mask, use_pko=use_pko, disable_rope=disable_rope)
        return self.moe_residual_accumulating_weight_gradient(x, w13_accumulator, w2_accumulator)

    @named_call
    def grouped_call(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        use_pko: bool = False,
        disable_rope: bool = False,
    ) -> tuple[Float[Array, "B S D"], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
        x = self.attention_residual(x, mask, use_pko=use_pko, disable_rope=disable_rope)
        if self.mlp.expert_mlp.implementation != "ring":
            raise ValueError("grouped block requires the exact bulk-ring implementation")
        if x.shape[0] % 2:
            raise ValueError(f"grouped block requires an even batch dimension, got {x.shape[0]}")

        microbatch_size = x.shape[0] // 2
        grouped_hidden = jnp.reshape(x, (microbatch_size, 2, *x.shape[1:]))
        microbatch_shape = (microbatch_size, *x.shape[1:])
        first_hidden = jnp.reshape(jax.lax.slice_in_dim(grouped_hidden, 0, 1, axis=1), microbatch_shape)
        second_hidden = jnp.reshape(jax.lax.slice_in_dim(grouped_hidden, 1, 2, axis=1), microbatch_shape)
        first_output, first_stats = self.moe_residual(first_hidden)
        second_output, second_stats = self.moe_residual(second_hidden)
        output = jnp.reshape(jnp.stack((first_output, second_output), axis=1), x.shape)
        return output, (first_stats, second_stats)


class _BlockAttentionView(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention

    @staticmethod
    def from_block(block: Block) -> "_BlockAttentionView":
        return _BlockAttentionView(
            rms_attn=block.rms_attn,
            attn_gated_norm=block.attn_gated_norm,
            attn=block.attn,
        )

    @named_call
    def __call__(
        self,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        *,
        use_pko: bool,
        disable_rope: bool,
    ) -> Float[Array, "B S D"]:
        attn_input = self.attn_gated_norm(self.rms_attn(hidden))
        return hidden + self.attn(attn_input, mask, use_pko=use_pko, disable_rope=disable_rope)


class _BlockMoeDenseView(eqx.Module):
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    shared: DenseMLP | None

    @staticmethod
    def from_block(block: Block) -> "_BlockMoeDenseView":
        return _BlockMoeDenseView(
            rms_mlp=block.rms_mlp,
            mlp_gated_norm=block.mlp_gated_norm,
            shared=block.shared,
        )

    @named_call
    def __call__(
        self,
        hidden: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], Float[Array, "B S D"]]:
        mlp_input = self.mlp_gated_norm(self.rms_mlp(hidden))
        if self.shared is None:
            return mlp_input, jnp.zeros_like(mlp_input)
        return mlp_input, self.shared(mlp_input, activation=ActivationFunctionEnum.silu)


def _require_component_pair(values: tuple[Any, ...], *, name: str) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two microbatches, got {len(values)}")


@named_call
def _paired_moe_calls(
    mlp: MoEMLP,
    mlp_inputs: tuple[jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    first_output, first_stats = mlp(mlp_inputs[0])
    second_output, second_stats = mlp(mlp_inputs[1])
    return (first_output, second_output), (first_stats, second_stats)


def paired_moe_component_forward(
    mlp: MoEMLP,
    mlp_inputs: tuple[jax.Array, jax.Array],
    *,
    remat_mode: RematMode,
) -> tuple[tuple[jax.Array, jax.Array], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    """Run exactly two learned-router MoE calls under one rematerialization boundary."""
    _require_component_pair(mlp_inputs, name="MoE inputs")
    if mlp.expert_mlp.implementation != "ring":
        raise ValueError("paired MoE components require the exact bulk-ring implementation")

    remat_policy = None
    if remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    return eqx.filter_checkpoint(_paired_moe_calls, policy=remat_policy)(mlp, mlp_inputs)


@named_call
def _paired_explicit_moe_calls(
    expert_mlp: MoEExpertMlp,
    mlp_inputs: tuple[jax.Array, jax.Array],
    routing_states: tuple[MoERoutingState, MoERoutingState],
) -> tuple[tuple[jax.Array, jax.Array], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    first_output, first_stats = apply_moe_routing(expert_mlp, mlp_inputs[0], routing_states[0])
    second_output, second_stats = apply_moe_routing(expert_mlp, mlp_inputs[1], routing_states[1])
    return (first_output, second_output), (first_stats, second_stats)


def paired_explicit_moe_component_forward(
    expert_mlp: MoEExpertMlp,
    mlp_inputs: tuple[jax.Array, jax.Array],
    routing_states: tuple[MoERoutingState, MoERoutingState],
    *,
    remat_mode: RematMode,
) -> tuple[tuple[jax.Array, jax.Array], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    """Run two expert calls from explicit per-microbatch routing state."""
    _require_component_pair(mlp_inputs, name="MoE inputs")
    _require_component_pair(routing_states, name="MoE routing states")
    if expert_mlp.implementation != "ring":
        raise ValueError("paired explicit MoE components require the exact bulk-ring implementation")

    remat_policy = None
    if remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    return eqx.filter_checkpoint(_paired_explicit_moe_calls, policy=remat_policy)(
        expert_mlp,
        mlp_inputs,
        routing_states,
    )


def _component_projection(output: jax.Array, cotangent: jax.Array) -> jax.Array:
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


def paired_moe_component_value_and_grads(
    mlp: MoEMLP,
    mlp_inputs: tuple[jax.Array, jax.Array],
    output_cotangents: tuple[jax.Array, jax.Array],
    *,
    remat_mode: RematMode,
    router_z_loss_scale: float,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[dict[str, jax.Array], dict[str, jax.Array]],
    MoEMLP,
    tuple[jax.Array, jax.Array],
]:
    """Differentiate two MoE calls jointly while retaining per-microbatch values and metrics."""
    _require_component_pair(mlp_inputs, name="MoE inputs")
    _require_component_pair(output_cotangents, name="MoE output cotangents")

    def projected_pair(current_mlp: MoEMLP, current_inputs: tuple[jax.Array, jax.Array]):
        outputs, router_stats = paired_moe_component_forward(
            current_mlp,
            current_inputs,
            remat_mode=remat_mode,
        )
        losses = tuple(
            _component_projection(output, cotangent) + router_z_loss_scale * stats["router_z_loss"]
            for output, cotangent, stats in zip(outputs, output_cotangents, router_stats, strict=True)
        )
        return losses[0] + losses[1], (losses, outputs, router_stats)

    (_, auxiliary), (mlp_gradient, input_gradients) = jax.value_and_grad(
        projected_pair,
        argnums=(0, 1),
        has_aux=True,
    )(mlp, mlp_inputs)
    losses, outputs, router_stats = auxiliary
    return losses, outputs, router_stats, mlp_gradient, input_gradients


def paired_block_forward(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    masks: tuple[AttentionMask | jax.Array, AttentionMask | jax.Array],
    *,
    use_pko: bool,
    disable_rope: bool,
    remat_mode: RematMode,
) -> tuple[tuple[jax.Array, jax.Array], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    """Run a block pair while joining only the two learned-router MoE calls."""
    _require_component_pair(hiddens, name="block inputs")
    _require_component_pair(masks, name="attention masks")
    attention = _BlockAttentionView.from_block(block)
    dense_moe_side = _BlockMoeDenseView.from_block(block)

    post_attention = tuple(
        attention(hidden, mask, use_pko=use_pko, disable_rope=disable_rope)
        for hidden, mask in zip(hiddens, masks, strict=True)
    )
    dense_outputs = tuple(dense_moe_side(hidden) for hidden in post_attention)
    mlp_inputs = tuple(output[0] for output in dense_outputs)
    shared_outputs = tuple(output[1] for output in dense_outputs)
    routed_outputs, router_stats = paired_moe_component_forward(
        block.mlp,
        mlp_inputs,
        remat_mode=remat_mode,
    )

    if block.shared is None:
        updates = routed_outputs
    else:
        updates = tuple(routed + shared for routed, shared in zip(routed_outputs, shared_outputs, strict=True))
    outputs = tuple(hidden + update for hidden, update in zip(post_attention, updates, strict=True))
    return outputs, router_stats


def paired_block_value_and_grads(
    block: Block,
    hiddens: tuple[jax.Array, jax.Array],
    masks: tuple[AttentionMask | jax.Array, AttentionMask | jax.Array],
    output_cotangents: tuple[jax.Array, jax.Array],
    *,
    use_pko: bool,
    disable_rope: bool,
    remat_mode: RematMode,
    router_z_loss_scale: float,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[dict[str, jax.Array], dict[str, jax.Array]],
    Block,
    tuple[jax.Array, jax.Array],
]:
    """Compose separate dense VJPs around one joined exact-ring MoE VJP."""
    _require_component_pair(hiddens, name="block inputs")
    _require_component_pair(masks, name="attention masks")
    _require_component_pair(output_cotangents, name="block output cotangents")
    if block.mlp.expert_mlp.implementation != "ring":
        raise ValueError("paired block components require the exact bulk-ring implementation")

    attention = _BlockAttentionView.from_block(block)
    dense_moe_side = _BlockMoeDenseView.from_block(block)
    post_attention = []
    attention_pullbacks = []
    for hidden, mask in zip(hiddens, masks, strict=True):
        output, pullback = jax.vjp(
            lambda current_attention, current_hidden, mask=mask: current_attention(
                current_hidden,
                mask,
                use_pko=use_pko,
                disable_rope=disable_rope,
            ),
            attention,
            hidden,
        )
        post_attention.append(output)
        attention_pullbacks.append(pullback)

    dense_outputs = []
    dense_pullbacks = []
    for hidden in post_attention:
        output, pullback = jax.vjp(_BlockMoeDenseView.__call__, dense_moe_side, hidden)
        dense_outputs.append(output)
        dense_pullbacks.append(pullback)

    mlp_inputs = tuple(output[0] for output in dense_outputs)
    shared_outputs = tuple(output[1] for output in dense_outputs)
    _, routed_outputs, router_stats, mlp_gradient, mlp_input_gradients = paired_moe_component_value_and_grads(
        block.mlp,
        mlp_inputs,
        output_cotangents,
        remat_mode=remat_mode,
        router_z_loss_scale=router_z_loss_scale,
    )

    dense_gradients = []
    post_attention_gradients = []
    for dense_pullback, mlp_input_gradient, output_cotangent in zip(
        dense_pullbacks,
        mlp_input_gradients,
        output_cotangents,
        strict=True,
    ):
        shared_cotangent = output_cotangent if block.shared is not None else jnp.zeros_like(output_cotangent)
        dense_gradient, dense_hidden_gradient = dense_pullback((mlp_input_gradient, shared_cotangent))
        dense_gradients.append(dense_gradient)
        post_attention_gradients.append(output_cotangent + dense_hidden_gradient)

    attention_gradients = []
    input_gradients = []
    for attention_pullback, post_attention_gradient in zip(
        attention_pullbacks,
        post_attention_gradients,
        strict=True,
    ):
        attention_gradient, input_gradient = attention_pullback(post_attention_gradient)
        attention_gradients.append(attention_gradient)
        input_gradients.append(input_gradient)

    attention_gradient = jax.tree.map(lambda first, second: first + second, *attention_gradients)
    dense_gradient = jax.tree.map(lambda first, second: first + second, *dense_gradients)
    block_gradient = Block(
        rms_attn=attention_gradient.rms_attn,
        attn_gated_norm=attention_gradient.attn_gated_norm,
        attn=attention_gradient.attn,
        rms_mlp=dense_gradient.rms_mlp,
        mlp_gated_norm=dense_gradient.mlp_gated_norm,
        mlp=mlp_gradient,
        shared=dense_gradient.shared,
    )

    if block.shared is None:
        updates = routed_outputs
    else:
        updates = tuple(routed + shared for routed, shared in zip(routed_outputs, shared_outputs, strict=True))
    outputs = tuple(hidden + update for hidden, update in zip(post_attention, updates, strict=True))
    losses = tuple(
        _component_projection(output, cotangent) + router_z_loss_scale * stats["router_z_loss"]
        for output, cotangent, stats in zip(outputs, output_cotangents, router_stats, strict=True)
    )
    return losses, outputs, router_stats, block_gradient, tuple(input_gradients)


def _run_block_with_remat(
    block: Block,
    hidden: Float[Array, "B S D"],
    mask: AttentionMask | jax.Array,
    *,
    use_pko: bool,
    disable_rope: bool,
    remat_mode: RematMode,
    effectful_moe: bool,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
    if effectful_moe:
        hidden = eqx.filter_checkpoint(Block.attention_residual)(
            block,
            hidden,
            mask,
            use_pko,
            disable_rope,
        )
        return block.moe_residual(hidden)

    remat_policy = None
    if remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    return eqx.filter_checkpoint(block, policy=remat_policy)(hidden, mask, use_pko, disable_rope)


def _run_block_accumulating_weight_gradient_with_remat(
    block: Block,
    hidden: Float[Array, "B S D"],
    mask: AttentionMask | jax.Array,
    w13_accumulator: Float[Array, "E D I2"],
    w2_accumulator: Float[Array, "E I D"],
    *,
    use_pko: bool,
    disable_rope: bool,
    remat_mode: RematMode,
) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
    remat_policy = None
    if remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    return eqx.filter_checkpoint(Block.accumulating_weight_gradient, policy=remat_policy)(
        block,
        hidden,
        mask,
        w13_accumulator,
        w2_accumulator,
        use_pko,
        disable_rope,
    )


def _run_grouped_block_with_remat(
    block: Block,
    hidden: Float[Array, "B S D"],
    mask: AttentionMask | jax.Array,
    *,
    use_pko: bool,
    disable_rope: bool,
    remat_mode: RematMode,
) -> tuple[Float[Array, "B S D"], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
    remat_policy = None
    if remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    return eqx.filter_checkpoint(Block.grouped_call, policy=remat_policy)(
        block,
        hidden,
        mask,
        use_pko,
        disable_rope,
    )


class TransformerPipelineStage(eqx.Module):
    """Stage-local subset of a Transformer for explicit MPMD pipeline training."""

    token_embed: jax.Array | None
    embed_norm: RMSNorm | None
    embed_gated_norm: GatedNorm | None
    output_proj: jax.Array | None
    blocks: tuple[Block, ...]
    final_norm: RMSNorm | None
    final_gated_norm: GatedNorm | None
    config: GrugModelConfig = eqx.field(static=True)
    stage_index: int = eqx.field(static=True)
    start_layer: int = eqx.field(static=True)
    end_layer: int = eqx.field(static=True)
    pipeline_stages: int = eqx.field(static=True)

    @property
    def is_first_stage(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage_index == self.pipeline_stages - 1

    @named_call
    def embed_tokens(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        if self.token_embed is None or self.embed_norm is None or self.embed_gated_norm is None:
            raise ValueError("only the first pipeline stage owns token embedding parameters")
        batch_spec = _batch_spec()
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        return self.embed_gated_norm(self.embed_norm(hidden))

    @named_call
    def run_block(
        self,
        local_index: int,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """Run one stage-local block with the same policy as :meth:`block_range`."""
        if mask is None:
            mask = AttentionMask.causal()
        if not 0 <= local_index < len(self.blocks):
            raise ValueError(f"local block index must be in [0, {len(self.blocks)}), got {local_index}")

        layer_index = self.start_layer + local_index
        is_last = layer_index == self.config.num_layers - 1
        is_long = layer_index % 4 == 3 or is_last
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        layer_mask = AttentionMask(
            is_causal=True,
            sliding_window=None if is_long else self.config.sliding_window,
            segment_ids=segment_ids,
        )
        return _run_block_with_remat(
            self.blocks[local_index],
            hidden,
            layer_mask,
            use_pko=is_long and not self.config.disable_pko,
            disable_rope=is_long and self.config.disable_long_rope,
            remat_mode=self.config.remat_mode,
            effectful_moe=self.config.moe_implementation in _EFFECTFUL_MOE_IMPLEMENTATIONS,
        )

    @named_call
    def run_block_accumulating_weight_gradient(
        self,
        local_index: int,
        hidden: Float[Array, "B S D"],
        w13_accumulator: Float[Array, "E D I2"],
        w2_accumulator: Float[Array, "E I D"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
        """Run one Ring block with a data-local FP32 accumulator pair."""
        if mask is None:
            mask = AttentionMask.causal()
        if not 0 <= local_index < len(self.blocks):
            raise ValueError(f"local block index must be in [0, {len(self.blocks)}), got {local_index}")
        if self.blocks[local_index].mlp.expert_mlp.implementation != "ring":
            raise ValueError("FP32 expert-gradient accumulation requires the exact Ring EP implementation")

        layer_index = self.start_layer + local_index
        is_last = layer_index == self.config.num_layers - 1
        is_long = layer_index % 4 == 3 or is_last
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        layer_mask = AttentionMask(
            is_causal=True,
            sliding_window=None if is_long else self.config.sliding_window,
            segment_ids=segment_ids,
            thd_segment_metadata=mask.thd_segment_metadata if isinstance(mask, AttentionMask) else None,
        )
        return _run_block_accumulating_weight_gradient_with_remat(
            self.blocks[local_index],
            hidden,
            layer_mask,
            w13_accumulator,
            w2_accumulator,
            use_pko=is_long and not self.config.disable_pko,
            disable_rope=is_long and self.config.disable_long_rope,
            remat_mode=self.config.remat_mode,
        )

    @named_call
    def run_grouped_block(
        self,
        local_index: int,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
        """Run attention once and exact ring MoE separately for two microbatches."""
        if mask is None:
            mask = AttentionMask.causal()
        if not 0 <= local_index < len(self.blocks):
            raise ValueError(f"local block index must be in [0, {len(self.blocks)}), got {local_index}")

        layer_index = self.start_layer + local_index
        is_last = layer_index == self.config.num_layers - 1
        is_long = layer_index % 4 == 3 or is_last
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        layer_mask = AttentionMask(
            is_causal=True,
            sliding_window=None if is_long else self.config.sliding_window,
            segment_ids=segment_ids,
            thd_segment_metadata=mask.thd_segment_metadata if isinstance(mask, AttentionMask) else None,
        )
        return _run_grouped_block_with_remat(
            self.blocks[local_index],
            hidden,
            layer_mask,
            use_pko=is_long and not self.config.disable_pko,
            disable_rope=is_long and self.config.disable_long_rope,
            remat_mode=self.config.remat_mode,
        )

    @named_call
    def block_range(
        self,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array | None = None,
        *,
        mark_stage_end: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()
        if not self.blocks:
            raise ValueError("pipeline stages must own at least one transformer block")

        moe_router_stats: list[dict[str, jax.Array]] = []
        for local_index in range(len(self.blocks)):
            hidden, router_stats = self.run_block(local_index, hidden, mask)
            moe_router_stats.append(router_stats)

        if mark_stage_end:
            hidden = _mark_pipeline_stage_end(hidden, layer_index=self.end_layer - 1)
        return hidden, _stack_router_metrics(moe_router_stats)

    @named_call
    def block_range_accumulating_weight_gradient(
        self,
        hidden: Float[Array, "B S D"],
        w13_accumulators: tuple[jax.Array, ...],
        w2_accumulators: tuple[jax.Array, ...],
        mask: AttentionMask | jax.Array | None = None,
        *,
        mark_stage_end: bool = False,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], Float[Array, ""]]:
        """Run all stage blocks with one data-local accumulator pair per block.

        Accumulator values use ordinary expert-weight shapes and may be
        physically divergent across the data axis. They must stay in the
        controlled ``check_vma=False`` task chain until explicit synchronization.
        The returned token must be added to the normalized loss with coefficient
        exactly one.
        """
        if mask is None:
            mask = AttentionMask.causal()
        if not self.blocks:
            raise ValueError("pipeline stages must own at least one transformer block")
        if len(w13_accumulators) != len(self.blocks) or len(w2_accumulators) != len(self.blocks):
            raise ValueError(
                "expert-gradient accumulator counts must match the stage block count; "
                f"got {len(w13_accumulators)} W13, {len(w2_accumulators)} W2, and {len(self.blocks)} blocks"
            )

        moe_router_stats: list[dict[str, jax.Array]] = []
        accumulation_token = jnp.zeros((), dtype=jnp.float32)
        for local_index, (w13_accumulator, w2_accumulator) in enumerate(
            zip(w13_accumulators, w2_accumulators, strict=True)
        ):
            hidden, router_stats, block_token = self.run_block_accumulating_weight_gradient(
                local_index,
                hidden,
                w13_accumulator,
                w2_accumulator,
                mask,
            )
            moe_router_stats.append(router_stats)
            accumulation_token = accumulation_token + block_token

        if mark_stage_end:
            hidden = _mark_pipeline_stage_end(hidden, layer_index=self.end_layer - 1)
        return hidden, _stack_router_metrics(moe_router_stats), accumulation_token

    @named_call
    def finalize_hidden(self, hidden: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        if self.final_norm is None or self.final_gated_norm is None:
            raise ValueError("only the final pipeline stage owns final norm parameters")
        return self.final_gated_norm(self.final_norm(hidden))

    @named_call
    def hidden_next_token_loss(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        router_metrics: dict[str, jax.Array],
        *,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        if self.output_proj is None:
            raise ValueError("only the final pipeline stage owns output projection parameters")

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
            implementation=self.config.loss_implementation,
        )
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

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    def split_for_pipeline(
        self,
        pipeline_stages: int,
        stage_layer_counts: tuple[int, ...] | None = None,
    ) -> tuple[TransformerPipelineStage, ...]:
        """Partition Transformer weights into contiguous pipeline-stage pytrees."""
        bounds = _pipeline_stage_bounds(len(self.blocks), pipeline_stages, stage_layer_counts)
        stages = []
        for stage_index, (start_layer, end_layer) in enumerate(bounds):
            is_first = stage_index == 0
            is_last = stage_index == pipeline_stages - 1
            stages.append(
                TransformerPipelineStage(
                    token_embed=self.token_embed if is_first else None,
                    embed_norm=self.embed_norm if is_first else None,
                    embed_gated_norm=self.embed_gated_norm if is_first else None,
                    output_proj=self.output_proj if is_last else None,
                    blocks=self.blocks[start_layer:end_layer],
                    final_norm=self.final_norm if is_last else None,
                    final_gated_norm=self.final_gated_norm if is_last else None,
                    config=self.config,
                    stage_index=stage_index,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    pipeline_stages=pipeline_stages,
                )
            )
        return tuple(stages)

    @staticmethod
    def merge_pipeline_stages(stages: tuple[TransformerPipelineStage, ...]) -> "Transformer":
        """Reconstruct a full Transformer from stage-local weights."""
        if not stages:
            raise ValueError("at least one pipeline stage is required")
        for expected_index, stage in enumerate(stages):
            if stage.stage_index != expected_index:
                raise ValueError(
                    f"pipeline stages must be provided in order; expected stage {expected_index}, "
                    f"got stage {stage.stage_index}"
                )

        first = stages[0]
        last = stages[-1]
        if first.token_embed is None or first.embed_norm is None or first.embed_gated_norm is None:
            raise ValueError("first pipeline stage is missing embedding parameters")
        if last.output_proj is None or last.final_norm is None or last.final_gated_norm is None:
            raise ValueError("last pipeline stage is missing output head parameters")

        blocks = tuple(block for stage in stages for block in stage.blocks)
        return Transformer(
            token_embed=first.token_embed,
            embed_norm=first.embed_norm,
            embed_gated_norm=first.embed_gated_norm,
            output_proj=last.output_proj,
            blocks=blocks,
            final_norm=last.final_norm,
            final_gated_norm=last.final_gated_norm,
            config=first.config,
        )

    @named_call
    def embed_tokens(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        batch_spec = _batch_spec()
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        return self.embed_gated_norm(self.embed_norm(hidden))

    @named_call
    def block_range(
        self,
        hidden: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array | None = None,
        *,
        start_layer: int,
        end_layer: int,
        pipeline_stage_end_layers: tuple[int, ...] = (),
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        cfg = self.config
        num_blocks = len(self.blocks)
        if not 0 <= start_layer < end_layer <= num_blocks:
            raise ValueError(
                f"block range must satisfy 0 <= start_layer < end_layer <= {num_blocks}, "
                f"got start_layer={start_layer}, end_layer={end_layer}"
            )

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks[start_layer:end_layer], start=start_layer):
            is_last = i == num_blocks - 1
            is_long = i % 4 == 3 or is_last
            layer_mask = long_mask if is_long else short_mask
            use_pko = is_long and not cfg.disable_pko
            disable_rope = is_long and cfg.disable_long_rope
            hidden, router_stats = _run_block_with_remat(
                block,
                hidden,
                layer_mask,
                use_pko=use_pko,
                disable_rope=disable_rope,
                remat_mode=cfg.remat_mode,
                effectful_moe=cfg.moe_implementation in _EFFECTFUL_MOE_IMPLEMENTATIONS,
            )
            if i in pipeline_stage_end_layers:
                hidden = _mark_pipeline_stage_end(hidden, layer_index=i)
            moe_router_stats.append(router_stats)

        return hidden, _stack_router_metrics(moe_router_stats)

    @named_call
    def finalize_hidden(self, hidden: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        return self.final_gated_norm(self.final_norm(hidden))

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        *,
        pipeline_stages: int | None = None,
        pipeline_stage_layer_counts: tuple[int, ...] | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        num_blocks = len(self.blocks)
        hidden = self.embed_tokens(token_ids)
        hidden, router_metrics = self.block_range(
            hidden,
            mask=mask,
            start_layer=0,
            end_layer=num_blocks,
            pipeline_stage_end_layers=_pipeline_stage_end_layers(
                num_blocks,
                pipeline_stages,
                pipeline_stage_layer_counts,
            ),
        )
        hidden = self.finalize_hidden(hidden)
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        *,
        pipeline_stages: int | None = None,
        pipeline_stage_layer_counts: tuple[int, ...] | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(
            token_ids,
            mask=mask,
            pipeline_stages=pipeline_stages,
            pipeline_stage_layer_counts=pipeline_stage_layer_counts,
        )
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
        pipeline_stages: int | None = None,
        pipeline_stage_layer_counts: tuple[int, ...] | None = None,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(
            token_ids,
            mask=mask,
            pipeline_stages=pipeline_stages,
            pipeline_stage_layer_counts=pipeline_stage_layer_counts,
        )
        return self.hidden_next_token_loss(
            hidden,
            token_ids,
            loss_weight,
            router_metrics,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            loss_dtype=loss_dtype,
            return_router_metrics=return_router_metrics,
        )

    @named_call
    def hidden_next_token_loss(
        self,
        hidden: Float[Array, "B S D"],
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        router_metrics: dict[str, jax.Array],
        *,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
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
            implementation=self.config.loss_implementation,
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
    "MoERoutingState",
    "MoeActivation",
    "RMSNorm",
    "ResearchFp8ExpertGemmConfig",
    "Transformer",
    "TransformerPipelineStage",
    "apply_moe_routing",
    "debug_mesh_and_token_pspec",
    "grugmoe_inference_state_dict",
    "paired_block_forward",
    "paired_block_value_and_grads",
    "paired_explicit_moe_component_forward",
    "paired_moe_component_forward",
    "paired_moe_component_value_and_grads",
]
