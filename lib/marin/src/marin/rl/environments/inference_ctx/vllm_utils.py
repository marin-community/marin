# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

logger = logging.getLogger(__name__)


def levanter_llama_to_vllm_mapping():
    return {
        "lm_head": ("model.lm_head", (None, "model")),
        "model.embed_tokens": (
            "model.embed.embedding",
            ("model", None),
        ),
        "model.layers.*.input_layernorm": (
            "model.layers.*.input_layernorm.scale",
            (None,),
        ),
        "model.layers.*.mlp.down_proj": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", None),
        ),
        "model.layers.*.mlp.gate_proj": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.mlp.up_proj": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.post_attention_layernorm": (
            "model.layers.*.post_attention_layernorm.scale",
            (None,),
        ),
        "model.layers.*.self_attn.k_proj": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.o_proj": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", None, None),
        ),
        "model.layers.*.self_attn.q_proj": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.v_proj": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "model", None),
        ),
        "model.norm": ("model.norm.scale", (None,)),
    }


def levanter_qwen_to_vllm_mapping():
    mapping = levanter_llama_to_vllm_mapping()
    mapping.update(
        {
            "model.layers.*.self_attn.q_norm": ("model.layers.*.self_attn.q_norm.scale", (None,)),
            "model.layers.*.self_attn.k_norm": ("model.layers.*.self_attn.k_norm.scale", (None,)),
            "model.layers.*.self_attn.q_proj_bias": (
                "model.layers.*.self_attn.q_proj.bias",
                ("model", None),
            ),
            "model.layers.*.self_attn.k_proj_bias": (
                "model.layers.*.self_attn.k_proj.bias",
                ("model", None),
            ),
            "model.layers.*.self_attn.v_proj_bias": (
                "model.layers.*.self_attn.v_proj.bias",
                ("model", None),
            ),
        }
    )
    return mapping


llama_transpose_keys = {
    "lm_head": (1, 0),
    "gate_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "q_proj": (2, 0, 1),
    "k_proj": (2, 0, 1),
    "v_proj": (2, 0, 1),
    "o_proj": (1, 2, 0),
    "q_proj_bias": (0, 1),
    "k_proj_bias": (0, 1),
    "v_proj_bias": (0, 1),
}

_MODEL_MAPPINGS = {
    "meta-llama/Llama-3.2-1B-Instruct": levanter_llama_to_vllm_mapping(),
    "meta-llama/Llama-3.2-3B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-0.6B": levanter_qwen_to_vllm_mapping(),
    "Qwen/Qwen3-1.7B": levanter_qwen_to_vllm_mapping(),
    "meta-llama/Llama-3.1-8B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-8B": levanter_qwen_to_vllm_mapping(),
    "marin-community/marin-8b-instruct": levanter_llama_to_vllm_mapping(),
}

_MODEL_TRANSPOSE_KEYS = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_transpose_keys,
    "meta-llama/Llama-3.2-3B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-0.6B": llama_transpose_keys,
    "Qwen/Qwen3-1.7B": llama_transpose_keys,
    "meta-llama/Llama-3.1-8B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-8B": llama_transpose_keys,
    "marin-community/marin-8b-instruct": llama_transpose_keys,
}


def _infer_mapping(model_name: str) -> dict:
    """Infer the vLLM mapping for a model name, falling back to substring matching."""
    if model_name in _MODEL_MAPPINGS:
        return _MODEL_MAPPINGS[model_name]
    if "Qwen2.5" in model_name:
        return levanter_qwen_to_vllm_mapping()
    raise KeyError(f"No MODEL_MAPPING registered for model: {model_name}")


def _infer_transpose_keys(model_name: str) -> dict:
    """Infer the transpose keys for a model name, falling back to substring matching."""
    if model_name in _MODEL_TRANSPOSE_KEYS:
        return _MODEL_TRANSPOSE_KEYS[model_name]
    if "Qwen2.5" in model_name:
        return llama_transpose_keys
    raise KeyError(f"No MODEL_TRANSPOSE_KEYS registered for model: {model_name}")


class _FallbackDict:
    """Dict-like object that supports fallback lookup by substring matching."""

    def __init__(self, fallback):
        self._fallback = fallback

    def __getitem__(self, key):
        return self._fallback(key)

    def __contains__(self, key):
        try:
            self._fallback(key)
            return True
        except KeyError:
            return False


MODEL_MAPPINGS = _FallbackDict(_infer_mapping)
MODEL_TRANSPOSE_KEYS = _FallbackDict(_infer_transpose_keys)


def _state_path_to_str(path: Any) -> str:
    if isinstance(path, str):
        return path
    if hasattr(path, "to_str"):
        return path.to_str()
    if isinstance(path, tuple | list):
        return ".".join(str(part) for part in path)
    return str(path)


def _flat_state_items(state: Any):
    if not hasattr(state, "flat_state"):
        return []

    flat_state = state.flat_state()
    return flat_state.items() if hasattr(flat_state, "items") else flat_state


def _flat_state_keys(state: Any) -> set[str]:
    return {_state_path_to_str(path) for path, _leaf in _flat_state_items(state)}


def _native_nnx_target_candidates(source_key: str) -> tuple[str, ...]:
    candidates = [source_key]
    if source_key.endswith("_bias"):
        candidates.append(f"{source_key.removesuffix('_bias')}.bias")
    else:
        candidates.append(f"{source_key}.weight")
    return tuple(candidates)


def _native_nnx_mapping(source_state: Any, target_state: Any) -> dict[str, tuple[str, None]] | None:
    source_keys = _flat_state_keys(source_state)
    target_keys = _flat_state_keys(target_state)
    if not source_keys or not target_keys:
        return None

    mapping = {}
    missing = []
    for source_key in source_keys:
        for target_key in _native_nnx_target_candidates(source_key):
            if target_key in target_keys:
                mapping[source_key] = (target_key, None)
                break
        else:
            missing.append(source_key)

    if missing:
        logger.info(
            "reload_model: live target state looked native NNX, but %d/%d source keys had no native target match",
            len(missing),
            len(source_keys),
        )
        return None

    logger.info("reload_model: using native NNX weight mapping for %d params", len(mapping))
    return mapping


def _target_state_from_worker(worker: Any) -> Any | None:
    candidates = [
        worker,
        getattr(worker, "model_runner", None),
        getattr(worker, "worker", None),
    ]
    nested_worker = getattr(worker, "worker", None)
    if nested_worker is not None:
        candidates.append(getattr(nested_worker, "model_runner", None))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "state"):
            return candidate.state
    return None


def sync_weight_mapping_kwargs(model_name: str, source_state: Any, worker: Any) -> dict[str, Any]:
    target_state = _target_state_from_worker(worker)
    native_mapping = _native_nnx_mapping(source_state, target_state)
    if native_mapping is not None:
        return {
            "mappings": native_mapping,
            "transpose_keys": {},
        }

    return {
        "mappings": MODEL_MAPPINGS[model_name],
        "transpose_keys": MODEL_TRANSPOSE_KEYS[model_name],
    }
