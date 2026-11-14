# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Normal Levanter LM boundaries around the array-only Llama implementation."""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.nn.scan import ScanCheckpointPolicy as NamedScanCheckpointPolicy
from haliax.partitioning import ResourceMapping
from haliax.state_dict import ModuleWithStateDictSerialization
from jax.sharding import AxisType, Mesh, PartitionSpec as P, auto_axes, get_abstract_mesh, reshard
from jaxtyping import PRNGKeyArray
from transformers import PretrainedConfig as HfConfig
from transformers import LlamaConfig as HfLlamaConfig

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.compat.hf_config import hf_rope_config
from levanter.data.text.examples import grug_attention_mask_from_named
from levanter.grug.attention import AttentionMask as RawAttentionMask
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.layers.attention import AttentionBackend, AttentionMask, ragged_paged_attention
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.layers.scan_checkpoint import ScanCheckpointPolicy
from levanter.layers.rotary import (
    DefaultRotaryEmbeddingsConfig,
    Llama3RotaryEmbeddingsConfig,
    RotaryEmbeddingsConfig,
    YarnRotaryEmbeddingsConfig,
)
from levanter.layers.rotary_native import (
    Llama3RotaryConfig,
    NativeRotaryConfig,
    RotaryConfig,
    YarnRotaryConfig,
    yarn_attention_factor,
)
from levanter.models.llama import LlamaConfig
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.tokenizers import MarinTokenizer
from levanter.utils.activation import ActivationFunctionEnum


_LEVANTER_MODEL_METADATA_KEY = "levanter_model"
_NATIVE_CHECKPOINT_VERSION = 1
_NATIVE_MODEL_TYPE = "llama_native"


def _paged_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    model_axis: str | None,
    sm_scale: float,
) -> tuple[jax.Array, KvPageCache]:
    # Named cache/reference math uses Auto axes; return to the native explicit
    # layout at this shared infrastructure boundary, including donated pages.
    def attend(query, keys, values, pages, info):
        query = hax.named(query, ("position", "kv_head", "q_heads_per_group", "head_size"))
        cache = KvPageCache(hax.named(pages, ("page", "slot", "kv_head", "head_size")))
        with hax.axis_mapping({} if model_axis is None else {"kv_head": model_axis}, merge=False):
            cache = cache.update(
                info,
                hax.named(keys, ("position", "kv_head", "head_size")),
                hax.named(values, ("position", "kv_head", "head_size")),
            )
            output = ragged_paged_attention(
                query,
                cache.kv_pages,
                info.seq_lens,
                info.page_indices,
                info.cu_q_lens,
                info.num_seqs,
                sm_scale=sm_scale,
            )
        return output.array, cache.kv_pages.array

    # pyrefly: ignore[bad-assignment]  # auto_axes decorator overload erases the callable signature.
    wrapped: Callable[..., tuple[jax.Array, jax.Array]] = auto_axes(
        attend, out_sharding=(named_sharding_of(q), named_sharding_of(cache.kv_pages.array))
    )
    output, pages = wrapped(q, k, v, cache.kv_pages.array, batch_info)
    return output, dataclasses.replace(cache, kv_pages=hax.named(pages, cache.kv_pages.axes))


def native_llama_sharding(
    parameter_axis_mapping: ResourceMapping,
    compute_axis_mapping: ResourceMapping,
    batch_axis_name: str = "batch",
) -> NativeLlamaSharding:
    """Resolve native storage and computation roles from Levanter's named mappings."""
    batch = compute_axis_mapping.get(batch_axis_name, compute_axis_mapping.get("batch"))
    batch_axes = () if batch is None else (batch,) if isinstance(batch, str) else tuple(batch)
    fsdp_axis = parameter_axis_mapping.get("embed")
    model_axis = compute_axis_mapping.get("mlp")
    if fsdp_axis is not None and not isinstance(fsdp_axis, str):
        raise ValueError("llama_native requires one FSDP axis in mesh.param_mapping.embed")
    if model_axis is not None and not isinstance(model_axis, str):
        raise ValueError("llama_native requires one tensor-parallel axis in mesh.compute_mapping.mlp")
    if model_axis is not None and (model_axis in batch_axes or model_axis == fsdp_axis):
        raise ValueError("llama_native tensor-parallel placement must be distinct from batch/FSDP axes")
    if compute_axis_mapping.get("position") is not None:
        raise ValueError("llama_native does not support sequence parallelism")
    if compute_axis_mapping.get("embed") is not None:
        raise ValueError("llama_native gathers the FSDP embed dimension for computation")
    for mapping in (compute_axis_mapping, parameter_axis_mapping):
        for dimension in ("mlp", "heads", "kv_head", "vocab"):
            if dimension in mapping and mapping[dimension] != model_axis:
                raise ValueError(f"llama_native requires {dimension} to use the tensor-parallel axis {model_axis}")
    return NativeLlamaSharding(batch_axes=batch_axes, fsdp_axis=fsdp_axis, model_axis=model_axis)


def _checkpoint_architecture(config: NativeLlamaConfig) -> dict[str, Any]:
    architecture: dict[str, Any] = {
        "hidden_dim": config.hidden_dim,
        "intermediate_dim": config.intermediate_dim,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.actual_head_dim,
        "layer_norm_epsilon": float(config.layer_norm_epsilon),
        "tie_word_embeddings": config.tie_word_embeddings,
        "use_bias": config.use_bias,
        "use_layer_norm_weight": config.use_layer_norm_weight,
        "input_embedding_norm": config.input_embedding_norm,
        "hybrid_norm": config.hybrid_norm,
        "use_qk_norm": config.use_qk_norm,
        "upcast_attn": config.upcast_attn,
        "rope_theta": float(config.rope.theta),
        "rope_factor": float(config.rope.factor),
    }
    # SiLU/default-linear metadata is already a checkpoint wire contract.
    # Other activations and scaled RoPE add their semantic fields without
    # changing that existing shape.
    if config.activation_function is not ActivationFunctionEnum.silu:
        architecture["activation_function"] = config.activation_function.value
    if not isinstance(config.rope, RotaryConfig):
        architecture["rope_type"] = "llama3" if isinstance(config.rope, Llama3RotaryConfig) else "yarn"
        architecture.update(
            {name: value for name, value in dataclasses.asdict(config.rope).items() if name not in {"theta", "factor"}}
        )
    return architecture


def _native_rotary_config(config: RotaryEmbeddingsConfig) -> NativeRotaryConfig:
    if isinstance(config, DefaultRotaryEmbeddingsConfig):
        return RotaryConfig(theta=config.theta, factor=config.factor)
    if isinstance(config, Llama3RotaryEmbeddingsConfig):
        return Llama3RotaryConfig(**dataclasses.asdict(config))
    if isinstance(config, YarnRotaryEmbeddingsConfig):
        return YarnRotaryConfig(**dataclasses.asdict(config))
    raise ValueError(f"Unsupported native Llama RoPE configuration: {type(config).__name__}")


@LmConfig.register_subclass(_NATIVE_MODEL_TYPE)
@dataclass(frozen=True)
class NativeLlamaLmConfig(LlamaConfig):
    """Llama training configuration with raw-array parameters and explicit placement.

    The default attention backend is the portable reference implementation.
    Select ``splash`` explicitly to use the native TPU Splash implementation.
    """

    def __post_init__(self):
        super().__post_init__()
        if not self.scan_layers:
            raise ValueError("llama_native requires scan_layers: true")
        if self.attn_backend not in (
            None,
            AttentionBackend.DEFAULT,
            AttentionBackend.VANILLA,
            AttentionBackend.SPLASH,
        ):
            raise ValueError("llama_native supports vanilla/reference or splash attention")
        if self.flash_attention_block_size is not None:
            raise ValueError("llama_native does not support flash_attention_block_size")
        self.native_config(1)

    @property
    def requires_explicit_mesh_axes(self) -> bool:
        return True

    @property
    def model_type(self) -> type["NativeLlamaLmHeadModel"]:  # pyrefly: ignore[bad-override]
        return NativeLlamaLmHeadModel

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "NativeLlamaLmConfig":
        named_config = LlamaConfig.from_hf_config(hf_config)
        fields = {field.name: getattr(named_config, field.name) for field in dataclasses.fields(named_config)}
        fields["head_dim"] = hf_config.head_dim
        config = cls(**fields)
        config.validate_hf_config(hf_config)
        return config

    def validate_hf_export(self) -> None:
        """Require the standard HF Llama normalization and bias conventions."""
        if (
            self.hybrid_norm
            or self.input_embedding_norm
            or self.use_qk_norm
            or not self.use_layer_norm_weight
            or self.use_bias
        ):
            raise ValueError("llama_native HF export requires standard weighted RMSNorm and bias-free projections")

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfLlamaConfig:
        self.validate_hf_export()
        # Reuse the architecture translation without its untyped legacy linear
        # scaling dictionary, then emit matching Transformers 4/5 RoPE metadata.
        unscaled = dataclasses.replace(self, rope=DefaultRotaryEmbeddingsConfig(theta=self.rope.theta))
        fields = LlamaConfig.to_hf_config(unscaled, vocab_size).to_dict()
        _, scaling = self.rope.to_hf_config()
        if isinstance(self.rope, DefaultRotaryEmbeddingsConfig):
            scaling = None if self.rope.factor == 1 else {"rope_type": "linear", "factor": self.rope.factor}
        elif isinstance(self.rope, YarnRotaryEmbeddingsConfig):
            assert scaling is not None
            scaling["attention_factor"] = yarn_attention_factor(self.rope.factor, self.rope.mscale)
        fields.update(
            architectures=["LlamaForCausalLM"],
            head_dim=self.actual_head_size,
            rope_theta=self.rope.theta,
            rope_scaling=scaling,
            rope_parameters={"rope_theta": self.rope.theta, **(scaling or {"rope_type": "default"})},
        )
        fields.update(config_overrides or {})
        return HfLlamaConfig.from_dict(fields)

    def hf_checkpoint_converter(
        self, ref_checkpoint: str | None = None, *, tokenizer: MarinTokenizer | None = None
    ) -> HFCheckpointConverter:
        return HFCheckpointConverter(
            type(self),
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            HfConfigClass=HfLlamaConfig,
            tokenizer=tokenizer if tokenizer is not None else self.tokenizer,
        )

    def validate_hf_config(self, hf_config: HfConfig) -> None:
        """Reject HF architectures whose weights would change meaning under this configuration."""
        if hf_config.model_type != "llama":
            raise ValueError("llama_native HF initialization requires a Llama checkpoint")
        if (
            self.hybrid_norm
            or self.input_embedding_norm
            or self.use_qk_norm
            or not self.use_layer_norm_weight
            or any(getattr(hf_config, name, False) for name in ("hybrid_norm", "input_embedding_norm", "use_qk_norm"))
            or not getattr(hf_config, "use_layer_norm_weight", True)
        ):
            raise ValueError("llama_native HF initialization requires standard weighted RMSNorm")
        # The native use_bias also enables norm biases, which standard HF Llama
        # does not serialize even when attention_bias or mlp_bias is enabled.
        if self.use_bias or hf_config.attention_bias or hf_config.mlp_bias:
            raise ValueError("llama_native HF initialization supports only bias-free Llama")
        if hf_config.attention_dropout != 0:
            raise ValueError("llama_native HF initialization does not support attention dropout")
        expected = {
            "hidden_size": self.hidden_dim,
            "intermediate_size": self.intermediate_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.actual_head_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_act": self.activation_function.value,
            "rms_norm_eps": self.layer_norm_epsilon,
        }
        for name, value in expected.items():
            actual = getattr(hf_config, name)
            if actual != value:
                raise ValueError(f"llama_native HF {name} mismatch: checkpoint={actual}, configured={value}")
        theta, scaling = hf_rope_config(hf_config)
        if getattr(hf_config, "partial_rotary_factor", 1.0) != 1.0 or (
            scaling is not None and scaling.get("partial_rotary_factor", 1.0) != 1.0
        ):
            raise ValueError("llama_native HF initialization requires full-head RoPE")
        rope_type = (scaling.get("rope_type") or scaling.get("type") or "default") if scaling else "default"
        if rope_type == "default" and scaling is not None and scaling.get("factor", 1.0) != 1.0:
            raise ValueError("llama_native HF default RoPE ignores factor; scaled RoPE must declare linear")
        rope = RotaryEmbeddingsConfig.from_hf_config(theta, scaling)
        if rope != self.rope:
            raise ValueError("llama_native HF RoPE parameters must match the configured RoPE")
        if isinstance(rope, YarnRotaryEmbeddingsConfig) and scaling is not None:
            if not scaling.get("truncate", True):
                raise ValueError("llama_native HF YaRN requires the integer correction ramp")
            expected_factor = yarn_attention_factor(rope.factor, rope.mscale)
            attention_factor = scaling.get("attention_factor")
            if attention_factor is None:
                mscale, mscale_all_dim = scaling.get("mscale"), scaling.get("mscale_all_dim")
                attention_factor = (
                    yarn_attention_factor(rope.factor, mscale) / yarn_attention_factor(rope.factor, mscale_all_dim)
                    if mscale and mscale_all_dim
                    else yarn_attention_factor(rope.factor, 1.0)
                )
            if attention_factor != expected_factor:
                raise ValueError("llama_native HF YaRN attention factor does not match the configured RoPE")

    def validate_sharding(self, sharding: NativeLlamaSharding, mesh: Mesh, vocab_size: int | None = None) -> None:
        """Validate a native model's dimensions against its concrete explicit mesh."""
        if mesh.empty or any(axis_type != AxisType.Explicit for axis_type in mesh.axis_types):
            raise ValueError("llama_native requires an active mesh with explicit axis types")
        for axis in (*sharding.batch_axes, sharding.fsdp_axis, sharding.model_axis):
            if axis is not None and axis not in mesh.shape:
                raise ValueError(f"llama_native placement axis {axis} is missing from the mesh")
        if sharding.fsdp_axis is not None and self.hidden_dim % mesh.shape[sharding.fsdp_axis]:
            raise ValueError("llama_native hidden_dim must be divisible by the FSDP mesh size")
        if sharding.model_axis is not None:
            model_size = mesh.shape[sharding.model_axis]
            if self.num_kv_heads % model_size:
                raise ValueError("llama_native num_kv_heads must be divisible by the tensor-parallel mesh size")
            if self.intermediate_dim % model_size:
                raise ValueError("llama_native intermediate_dim must be divisible by the tensor-parallel mesh size")
            if vocab_size is not None and vocab_size % model_size:
                raise ValueError("llama_native source vocabulary must be divisible by the tensor-parallel mesh size")

    def checkpoint_vocab_size(self, metadata: Mapping[str, Any]) -> int | None:
        """Validate native checkpoint semantics and recover its complete model vocabulary.

        Checkpoints without model metadata retain the explicit-config initialization path.
        """
        if _LEVANTER_MODEL_METADATA_KEY not in metadata:
            return None
        identity = metadata[_LEVANTER_MODEL_METADATA_KEY]
        if not isinstance(identity, dict) or set(identity) != {"version", "type", "vocab_size", "architecture"}:
            raise ValueError("Malformed native checkpoint model metadata")
        if (
            type(identity["version"]) is not int
            or identity["version"] != _NATIVE_CHECKPOINT_VERSION
            or identity["type"] != _NATIVE_MODEL_TYPE
        ):
            raise ValueError("Unsupported checkpoint model metadata version or implementation")
        vocab_size = identity["vocab_size"]
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Malformed native checkpoint vocabulary")
        expected = _checkpoint_architecture(self.native_config(vocab_size))
        actual = identity["architecture"]
        if (
            not isinstance(actual, dict)
            or set(actual) != set(expected)
            or any(
                type(value) in (int, bool) and type(actual[name]) is not type(value)
                for name, value in expected.items()
            )
            or any(
                type(actual[name]) not in (int, float)
                or (type(actual[name]) is float and not math.isfinite(actual[name]))
                for name, value in expected.items()
                if type(value) is float
            )
            or actual != expected
        ):
            raise ValueError("Native checkpoint architecture does not match the configured model semantics")
        return vocab_size

    def native_config(self, vocab_size: int) -> NativeLlamaConfig:
        """Resolve tokenizer vocabulary and the supported architecture fields."""
        fields = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(NativeLlamaConfig)
            if field.name not in {"vocab_size", "rope", "attention_implementation"}
        }
        if isinstance(self.gradient_checkpointing, NamedScanCheckpointPolicy):
            fields["gradient_checkpointing"] = ScanCheckpointPolicy(**dataclasses.asdict(self.gradient_checkpointing))
        return NativeLlamaConfig(
            vocab_size=vocab_size,
            rope=_native_rotary_config(self.rope),
            attention_implementation="tpu_splash" if self.attn_backend == AttentionBackend.SPLASH else "reference",
            **fields,
        )

    def build(  # pyrefly: ignore[bad-override]  # This config selects a different LmHeadModel implementation.
        self,
        Vocab: Axis,
        *,
        key: PRNGKeyArray,
        sharding: NativeLlamaSharding = NativeLlamaSharding(),
    ) -> "NativeLlamaLmHeadModel":
        return NativeLlamaLmHeadModel.init(Vocab, self, key=key, sharding=sharding)


class NativeLlamaLmHeadModel(ModuleWithStateDictSerialization, LmHeadModel[NativeLlamaLmConfig]):
    """Named input/loss/evaluation boundary; every model parameter is a raw array."""

    native: NativeLlamaModel
    _config: NativeLlamaLmConfig = eqx.field(static=True)

    @property
    def config(self) -> NativeLlamaLmConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.native.config.vocab_size)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        """Describe the actual stored model without source paths or runtime placement policy."""
        return {
            _LEVANTER_MODEL_METADATA_KEY: {
                "version": _NATIVE_CHECKPOINT_VERSION,
                "type": _NATIVE_MODEL_TYPE,
                "vocab_size": self.Vocab.size,
                "architecture": _checkpoint_architecture(self.native.config),
            }
        }

    @property
    def _batch_partition(self) -> tuple[str, ...] | None:
        mesh = get_abstract_mesh()
        return tuple(axis for axis in self.native.sharding.batch_axes if mesh.shape[axis] > 1) or None

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: NativeLlamaLmConfig,
        *,
        key: PRNGKeyArray,
        sharding: NativeLlamaSharding = NativeLlamaSharding(),
    ) -> "NativeLlamaLmHeadModel":
        return cls(config.native_config(Vocab.size).build(key=key, sharding=sharding), config)

    def _attention_mask(
        self, mask: AttentionMask | NamedArray | None, token_axes: tuple[Axis, ...]
    ) -> RawAttentionMask | jax.Array:
        if mask is None:
            # The normal LM API treats None as unmasked. The raw model's default
            # is causal, so represent the unmasked case explicitly.
            return RawAttentionMask()
        Pos = token_axes[-1]
        KeyPos = Pos.alias(self.KeyPos.name)
        batch_axes = token_axes[:-1]
        dense = mask if isinstance(mask, NamedArray) else None
        if isinstance(mask, AttentionMask):
            if (
                mask.causal_offset is not None
                or mask.explicit_mask is not None
                or mask.bidirectional_window is not None
            ):
                dense = mask.materialize(Pos, KeyPos)
            else:
                if mask.segment_ids is not None:
                    query, key = mask.segment_ids
                    query = query.broadcast_to((*batch_axes, Pos))
                    if key.has_axis(Pos.name):
                        key = key.rename({Pos.name: KeyPos.name})
                    key = key.broadcast_to((*batch_axes, KeyPos))
                    mask = dataclasses.replace(mask, segment_ids=(query, key))
                return grug_attention_mask_from_named(mask)
        if self.native.config.attention_implementation != "reference":
            raise ValueError(
                "Dense, offset and bidirectional attention masks require llama_native reference attention"
            )
        assert dense is not None
        # Named masks use truth values; raw attention also accepts additive biases.
        return dense.astype(jnp.bool_).broadcast_to((*batch_axes, Pos, KeyPos)).array

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        Pos = input_ids.resolve_axis(self.Pos.name)
        ordered = input_ids.rearrange((..., Pos))
        if ordered.ndim not in (1, 2):
            raise ValueError("llama_native expects tokens with position and at most one batch axis")
        tokens = ordered.array
        unbatched = tokens.ndim == 1
        if unbatched:
            # Keep the unbatched LM API usable on a data-sharded model. Identical
            # examples occupy each batch partition; only one output is returned.
            batch_size = math.prod(get_abstract_mesh().shape[axis] for axis in self.native.sharding.batch_axes)
            tokens = jnp.broadcast_to(tokens, (batch_size, tokens.shape[0]))
        positions = None
        if pos_ids is not None:
            positions = pos_ids.broadcast_to(ordered.axes).array
            if unbatched:
                positions = jnp.broadcast_to(positions, tokens.shape)
        mask = self._attention_mask(attn_mask, ordered.axes)
        hidden = self.native.activations(
            tokens,
            mask,
            pos_ids=positions,
            key=key,
        )
        if unbatched:
            # Gather before removing the data-sharded dimension so its transpose
            # restores the original cotangent layout, including unit mesh axes.
            hidden = reshard(hidden, P(None, None, None))[0]
        else:
            hidden = reshard(hidden, P(self._batch_partition, None, None))
        return hax.named(hidden, (*ordered.axes, self.Embed)).rearrange((*input_ids.axes, self.Embed))

    def get_lm_head(self) -> NamedArray:
        # The shared CE kernel partitions examples and requires a complete head
        # on each participant. Storage remains FSDP/tensor-parallel.
        return hax.named(reshard(self.native.get_lm_head(), P(None, None)), (self.Vocab, self.Embed))

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        """Allocate per-layer paged KV storage, replicated except for TP heads."""
        model_axis = self.native.sharding.model_axis
        model_size = 1 if model_axis is None else get_abstract_mesh().shape[model_axis]
        if self.native.config.num_kv_heads % model_size:
            raise ValueError("Paged native Llama requires KV heads divisible by tensor parallel size")
        caches = []
        with hax.axis_mapping({} if model_axis is None else {"kv_head": model_axis}, merge=False):
            for _ in range(self.native.config.num_layers):
                cache = KvPageCache.init(
                    spec,
                    Axis("kv_head", self.native.config.num_kv_heads),
                    Axis("head_size", self.native.config.actual_head_dim),
                    dtype=dtype,
                )
                pages = reshard(cache.kv_pages.array, P(None, None, model_axis, None))
                caches.append(dataclasses.replace(cache, kv_pages=hax.named(pages, cache.kv_pages.axes)))
        return ListCache(caches)

    def decode(
        self,
        input_ids: NamedArray,
        kv_cache: ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        """Run packed causal prefill or decode using the existing paged engine.

        The packed stream and page table are replicated over data axes; weights
        retain FSDP storage and attention/cache heads retain tensor parallelism.
        Positions are absolute within each sequence. Only the valid prefix in
        ``batch_info`` updates cache; trailing padded logits are not sampled.
        """
        # Paged decode shares the public key callback, but this route has no stochastic operation.
        del key
        if input_ids.ndim != 1 or input_ids.axes[0].name != self.Pos.name or pos_ids.axes != input_ids.axes:
            raise ValueError("Paged native Llama expects matching flat position token/position arrays")
        if len(kv_cache) != self.native.config.num_layers:
            raise ValueError("Paged native Llama requires one cache per decoder layer")
        layout = dataclasses.replace(self.native.sharding, batch_axes=())
        tokens = reshard(input_ids.array[None], layout.tokens)
        positions = reshard(pos_ids.array[None], layout.tokens)
        hidden = self.native.embed_tokens(tokens, sharding=layout)
        config = self.native.config
        count = input_ids.axes[0].size
        caches = []
        for index in range(config.num_layers):
            layer = self.native.layers.get_layer(index)
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(hidden), positions, sharding=layout)
            q = q.reshape(count, config.num_kv_heads, config.num_heads // config.num_kv_heads, config.actual_head_dim)
            k = k.reshape(count, config.num_kv_heads, config.actual_head_dim)
            v = v.reshape(count, config.num_kv_heads, config.actual_head_dim)
            output, cache = _paged_attention(
                q,
                k,
                v,
                kv_cache[index],
                batch_info,
                model_axis=layout.model_axis,
                sm_scale=1 / math.sqrt(config.actual_head_dim),
            )
            output = output.reshape(1, count, config.num_heads, config.actual_head_dim)
            attn = layer.self_attn.project_output(output, dtype=hidden.dtype, sharding=layout)
            hidden = layer.finish_attention(hidden, attn, sharding=layout)
            caches.append(cache)
        logits = self.native.unembed(self.native.norm(hidden), sharding=layout)
        logits = logits.reshape(count, self.Vocab.size)
        return hax.named(logits, (input_ids.axes[0], self.Vocab)), ListCache(caches)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(hax.ReductionFunction, hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ):
        mapping = {}
        tokens = example.tokens.rearrange((..., self.Pos.name))
        weights = example.loss_weight.broadcast_to(tokens.axes)
        batch_axes = tuple(axis for axis in tokens.axes if axis.name != self.Pos.name)
        batch_partition = self._batch_partition
        if batch_axes and batch_partition is not None:
            mapping[batch_axes[0].name] = batch_partition
        token_spec = P(batch_partition, None) if batch_axes else P(None)
        # Unit-size mesh axes are physically replicated. Canonicalize them before
        # jnp.roll: its empty slice can otherwise lose those axes and make the
        # explicit-sharding concatenate reject a single-example microbatch.
        example = dataclasses.replace(
            example,
            tokens=hax.named(reshard(tokens.array, token_spec), tokens.axes).rearrange(example.tokens.axes),
            loss_weight=hax.named(reshard(weights.array, token_spec), weights.axes),
        )
        # The loss's shard_map must agree with explicit array types, independently
        # of a caller's named-axis compute mapping.
        with hax.axis_mapping(mapping):
            return super().compute_next_token_loss(
                example,
                key=key,
                reduction=reduction,
                reduction_axis=reduction_axis,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype,
                logit_soft_cap=logit_soft_cap,
            )

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "NativeLlamaLmHeadModel":
        if new_size == self.Vocab.size:
            return self
        embed_key, head_key = maybe_rng_split(key, 2)
        # The existing resize helper concatenates old rows with padding. Gather
        # its resize axis so both pieces share a layout, then restore storage.
        embeddings = hax.named(
            reshard(self.native.token_embeddings, P(None, self.native.sharding.fsdp_axis)), (self.Vocab, self.Embed)
        )
        embeddings = hax.tree_util.resize_axis(embeddings, self.Vocab, new_size, key=embed_key)
        head = self.native.lm_head
        if head is not None:
            weight = hax.tree_util.resize_axis(self.get_lm_head(), self.Vocab, new_size, key=head_key)
            head = dataclasses.replace(head, weight=reshard(weight.array, self.native.sharding.column_storage))
        native = dataclasses.replace(
            self.native,
            token_embeddings=reshard(embeddings.array, self.native.sharding.column_storage),
            lm_head=head,
            config=dataclasses.replace(self.native.config, vocab_size=new_size),
        )
        return dataclasses.replace(self, native=native)

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return self.native.to_state_dict(prefix)

    def from_state_dict(self, state_dict, prefix: str | None = None) -> "NativeLlamaLmHeadModel":
        return dataclasses.replace(self, native=self.native.from_state_dict(state_dict, prefix))
