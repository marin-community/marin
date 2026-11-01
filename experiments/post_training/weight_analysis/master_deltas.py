# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug FP32 optimizer masters as HF-named delta blocks for seed alignment.

The caller supplies the existing Megatron master reader. Each stream holds one
native record at a time; three Snowball streams can hold about 522 GB for the
largest fused expert record. Run this CPU analysis with sufficient host memory.
"""

from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache

import numpy as np

from experiments.post_training.weight_analysis.checkpoint_io import SafetensorsCheckpoint
from experiments.post_training.weight_analysis.components import Block, Component, component_of, layer_of
from experiments.post_training.weight_analysis.delta_stats import _is_bf16_exact

_ROOT_NAMES = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "output_layer.weight": "lm_head.weight",
    "embed_norm.norm.weight": "model.embed_norm.weight",
    "embed_norm.down_proj.weight": "model.embed_gated_norm.down_proj.weight",
    "embed_norm.up_proj.weight": "model.embed_gated_norm.up_proj.weight",
    "decoder.final_layernorm.norm.weight": "model.norm.weight",
    "decoder.final_layernorm.down_proj.weight": "model.final_gated_norm.down_proj.weight",
    "decoder.final_layernorm.up_proj.weight": "model.final_gated_norm.up_proj.weight",
}
_LAYER_NAMES = {
    "input_layernorm.norm.weight": "input_layernorm.weight",
    "input_layernorm.down_proj.weight": "attn_gated_norm.down_proj.weight",
    "input_layernorm.up_proj.weight": "attn_gated_norm.up_proj.weight",
    "pre_mlp_layernorm.norm.weight": "post_attention_layernorm.weight",
    "pre_mlp_layernorm.down_proj.weight": "mlp_gated_norm.down_proj.weight",
    "pre_mlp_layernorm.up_proj.weight": "mlp_gated_norm.up_proj.weight",
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "self_attention.attn_gate.weight": "self_attn.attn_gate.weight",
    "mlp.router.weight": "mlp.router.weight",
    "mlp.shared_experts.linear_fc2.weight": "shared_expert.down_proj.weight",
}
_LAYER_PREFIX = "decoder.layers."
_QKV = "self_attention.linear_qkv.weight"
_ROUTED_GATE_UP = "mlp.experts.experts.linear_fc1.weight"
_ROUTED_DOWN = "mlp.experts.experts.linear_fc2.weight"
_SHARED_GATE_UP = "mlp.shared_experts.linear_fc1.weight"


@lru_cache(maxsize=2)
def _base_array(base: SafetensorsCheckpoint, name: str, expert: int | None) -> np.ndarray:
    value = base.read(name) if expert is None else base.read_expert(name, expert)
    value = value.astype(np.float32)
    if component_of(name) is not Component.SELECTION_BIAS and not _is_bf16_exact(value):
        raise ValueError(f"B has values outside its BF16 view: {name}")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite base weights: {name}")
    return value


def _gate_up_views(
    value: np.ndarray, prefix: str, base: SafetensorsCheckpoint, expert: int | None
) -> Iterator[tuple[str, int | None, np.ndarray]]:
    gate = prefix + ".gate_proj.weight"
    up = prefix + ".up_proj.weight"
    shape = base.spec(gate).shape
    gate_rows = shape[-2]
    if value.shape[0] != gate_rows + base.spec(up).shape[-2]:
        raise ValueError(f"Fused gate/up shape differs from HF layout: {prefix}")
    yield gate, expert, value[:gate_rows]
    yield up, expert, value[gate_rows:]


def _qkv_views(
    value: np.ndarray, prefix: str, base: SafetensorsCheckpoint
) -> Iterator[tuple[str, int | None, np.ndarray]]:
    names = [prefix + f".{letter}_proj.weight" for letter in ("q", "k", "v")]
    q_shape, k_shape, v_shape = [base.spec(name).shape for name in names]
    heads = base.spec(prefix + ".attn_gate.weight").shape[0]
    head_size = q_shape[0] // heads
    groups = k_shape[0] // head_size
    if q_shape[0] % heads or k_shape != v_shape or k_shape[0] % head_size or heads % groups:
        raise ValueError(f"Unsupported Grug grouped-query attention geometry: {prefix}")
    per_group = heads // groups
    packed = value.reshape(groups, per_group + 2, head_size, q_shape[1])
    views = [
        packed[:, :per_group].reshape(q_shape),
        packed[:, per_group].reshape(k_shape),
        packed[:, per_group + 1].reshape(v_shape),
    ]
    for name, array in zip(names, views, strict=True):
        yield name, None, array


def hf_master_views(
    name: str, value: np.ndarray, base: SafetensorsCheckpoint
) -> Iterator[tuple[str, int | None, np.ndarray]]:
    """Yield HF tensor names, expert indices, and views of one fused native record.

    Layout follows Grug's Megatron Bridge: gate/up concatenation and grouped
    Q/K/V packing. Stacked layer and expert axes retain their recorded order.
    """
    if name in _ROOT_NAMES:
        yield _ROOT_NAMES[name], None, value
        return
    if not name.startswith(_LAYER_PREFIX):
        raise ValueError(f"Unknown Grug master record: {name}")
    suffix = name.removeprefix(_LAYER_PREFIX)
    for layer, layer_value in enumerate(value):
        prefix = f"model.layers.{layer}"
        if suffix in _LAYER_NAMES:
            yield prefix + "." + _LAYER_NAMES[suffix], None, layer_value
        elif suffix == _QKV:
            yield from _qkv_views(layer_value, prefix + ".self_attn", base)
        elif suffix == _SHARED_GATE_UP:
            yield from _gate_up_views(layer_value, prefix + ".shared_expert", base, None)
        elif suffix == _ROUTED_GATE_UP:
            for expert, expert_value in enumerate(layer_value):
                yield from _gate_up_views(expert_value, prefix + ".mlp.experts", base, expert)
        elif suffix == _ROUTED_DOWN:
            for expert, expert_value in enumerate(layer_value):
                yield prefix + ".mlp.experts.down_proj.weight", expert, expert_value
        else:
            raise ValueError(f"Unknown Grug layer master record: {name}")


def fp32_master_delta_blocks(
    master_names: Iterable[str],
    read_master: Callable[[str], np.ndarray],
    base: SafetensorsCheckpoint,
) -> Iterator[Block]:
    """Read FP32 masters as ``master - B`` blocks for the existing C5 analysis.

    All trainable HF tensors must be covered. Frozen selection biases are emitted
    as zero deltas under the experiment's frozen-bias contract. A stream whose
    values are entirely BF16-representable cannot pass as unrounded masters.
    """
    seen = set()
    unrounded = False
    for native_name in sorted(master_names):
        value = read_master(native_name)
        if value.dtype != np.float32:
            raise ValueError(f"Master {native_name} must store FP32 values")
        for name, expert, view in hf_master_views(native_name, value, base):
            expected = base.spec(name).shape
            if expert is not None:
                if expert >= expected[0]:
                    raise ValueError(f"Expert index outside HF tensor: {name}/{expert}")
                expected = expected[1:]
            if view.shape != expected or not np.isfinite(view).all():
                raise ValueError(f"Invalid master block for {name}/{expert}: {view.shape}, expected {expected}")
            identity = (name, expert)
            if identity in seen:
                raise ValueError(f"Duplicate master block: {identity}")
            seen.add(identity)
            if not unrounded:
                unrounded = not _is_bf16_exact(view)
            component = component_of(name)
            key = (layer_of(name), expert) if expert is not None else None
            yield Block(name, component, key, view - _base_array(base, name, expert))
        del value, view
    expected = set()
    for name in base.names():
        component = component_of(name)
        if component is Component.SELECTION_BIAS:
            yield Block(name, component, None, np.zeros(base.spec(name).shape, dtype=np.float32))
        elif component is Component.ROUTED_EXPERTS:
            expected.update((name, expert) for expert in range(base.spec(name).shape[0]))
        else:
            expected.add((name, None))
    if seen != expected:
        raise ValueError(
            f"Master inventory differs from trainable HF weights: "
            f"missing={sorted(expected-seen)[:5]}, extra={sorted(seen-expected)[:5]}"
        )
    if not unrounded:
        raise ValueError("Masters contain only BF16-representable values; unrounded FP32 updates were not established")
