"""Versioned precision and architecture policy for inference exports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

POLICY_VERSION = 1


class PrecisionMode(StrEnum):
    BF16 = "bf16"
    FP32 = "fp32"
    PRESERVE = "preserve"


_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclass(frozen=True)
class TensorSpec:
    """Expected HF tensor shape and minimum source precision."""

    shape: tuple[int, ...]
    source_dtype: str
    minimum_dtype: str | None = None


@dataclass(frozen=True)
class ReleasePolicy:
    """Immutable precision decision shared by all inference exporters."""

    mode: PrecisionMode = PrecisionMode.PRESERVE
    version: int = POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != POLICY_VERSION:
            raise ValueError(f"unsupported HF release policy version {self.version}")
        if not isinstance(self.mode, PrecisionMode):
            raise ValueError(f"unsupported HF release precision {self.mode!r}")

    def target_dtype(self, name: str, spec: TensorSpec) -> str:
        """Resolve a tensor's release dtype, rejecting lost required precision."""
        source = spec.source_dtype
        minimum = spec.minimum_dtype
        if source not in _DTYPE_BYTES:
            raise ValueError(f"unsupported source dtype for {name}: {source}")
        if minimum == "F32" and source not in ("F32", "F64"):
            raise ValueError(f"{name} requires source dtype {minimum}, got {source}")
        if minimum is not None and minimum != "F32" and source != minimum:
            raise ValueError(f"{name} requires source dtype {minimum}, got {source}")
        if source not in _FLOAT_DTYPES:
            return source
        if self.mode is PrecisionMode.PRESERVE:
            return source
        if minimum == "F32":
            return "F32"
        return "BF16" if self.mode is PrecisionMode.BF16 else "F32"

    def config_dtype(self, specs: Mapping[str, TensorSpec]) -> str | None:
        """Return scalar HF dtype only when regular floating weights agree."""
        dtypes = {
            self.target_dtype(name, spec)
            for name, spec in specs.items()
            if spec.source_dtype in _FLOAT_DTYPES and spec.minimum_dtype != "F32"
        }
        if len(dtypes) != 1:
            return None
        return {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}[dtypes.pop()]


def dtype_size(dtype: str) -> int:
    return _DTYPE_BYTES[dtype]


_GRUG_ALIASES = {
    "hidden_size": ("hidden_dim",),
    "num_hidden_layers": ("num_layers",),
    "num_attention_heads": ("num_heads",),
    "num_key_value_heads": ("num_kv_heads",),
    "head_dim": ("attention_head_dim",),
    "max_position_embeddings": ("max_seq_len",),
    "rms_norm_eps": ("layer_norm_eps",),
    "initializer_range": ("initializer_std",),
    "num_experts": ("num_local_experts",),
    "num_experts_per_tok": ("num_experts_per_token",),
    "moe_intermediate_size": ("intermediate_size", "intermediate_dim"),
    "shared_expert_intermediate_size": ("shared_expert_intermediate_dim",),
}
_GRUG_REQUIRED = frozenset(
    {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "sliding_window",
        "grugmoe_attention_mode",
        "grugmoe_artifact_schema_version",
        "qk_mult",
    }
)


def canonical_grug_config(config: Mapping[str, object]) -> dict[str, object]:
    """Write one spelling per Grug field and retain its architecture variant."""
    if config.get("model_type") != "grug_moe":
        raise ValueError(f"unsupported Grug conversion from {config.get('model_type')!r}")
    result = dict(config)
    for canonical, aliases in _GRUG_ALIASES.items():
        values = {key: result[key] for key in (canonical, *aliases) if key in result}
        if values and any(value != next(iter(values.values())) for value in values.values()):
            raise ValueError(f"conflicting Grug aliases for {canonical}: {values}")
        if values:
            result[canonical] = next(iter(values.values()))
        for alias in aliases:
            result.pop(alias, None)
    missing = _GRUG_REQUIRED - result.keys()
    if missing:
        raise ValueError(f"Grug config is missing required fields: {sorted(missing)}")
    version = result["grugmoe_artifact_schema_version"]
    if version not in (1, 2):
        raise ValueError(f"unsupported Grug artifact schema version {version!r}")
    rope = result.get("rope_parameters")
    if isinstance(rope, dict) and rope.get("rope_theta", result["rope_theta"]) != result["rope_theta"]:
        raise ValueError("conflicting rope_theta and rope_parameters.rope_theta")
    return result


def grug_tensor_specs(config: Mapping[str, object], source: Mapping[str, TensorSpec]) -> dict[str, TensorSpec]:
    """Bind a backend's full expected layout to Grug's FP32 router state."""
    canonical = canonical_grug_config(config)
    layers = int(canonical["num_hidden_layers"])
    experts = int(canonical["num_experts"])
    required = {"model.embed_tokens.weight", "lm_head.weight"}
    required.update(f"model.layers.{layer}.mlp.router.bias" for layer in range(layers))
    missing = required - source.keys()
    if missing:
        raise ValueError(f"Grug export is missing required tensors: {sorted(missing)[:5]}")
    result = dict(source)
    for layer in range(layers):
        name = f"model.layers.{layer}.mlp.router.bias"
        spec = source[name]
        if spec.shape != (experts,):
            raise ValueError(f"{name} has shape {spec.shape}, expected {(experts,)}")
        result[name] = TensorSpec(spec.shape, spec.source_dtype, minimum_dtype="F32")
    return result
