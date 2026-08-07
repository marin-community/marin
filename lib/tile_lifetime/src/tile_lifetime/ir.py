# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic tensor graph independent of GPU backend types."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from math import prod


class DType(StrEnum):
    """Dtypes whose precision affects rewrite legality."""

    BOOL = "bool"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"
    INT32 = "int32"


@dataclass(frozen=True)
class TensorValue:
    """A typed logical tensor value."""

    id: int
    name: str
    shape: tuple[int, ...]
    dtype: DType


@dataclass(frozen=True)
class LinearOp:
    """Right-multiplication by a rank-two weight matrix."""

    id: int
    input: TensorValue
    weight: TensorValue
    output: TensorValue
    accumulation_dtype: DType
    source_location: str | None


@dataclass(frozen=True)
class ResidualAddOp:
    """Elementwise residual addition."""

    id: int
    left: TensorValue
    right: TensorValue
    output: TensorValue
    source_location: str | None


@dataclass(frozen=True)
class RMSNormOp:
    """RMS normalization over one tensor dimension."""

    id: int
    input: TensorValue
    gamma: TensorValue
    output: TensorValue
    axis: int
    epsilon: float
    reduction_dtype: DType
    source_location: str | None


@dataclass(frozen=True)
class ScaledDotProductAttentionOp:
    """Exact scaled dot-product attention over rank-four Q, K, and V tensors."""

    id: int
    query: TensorValue
    key: TensorValue
    value: TensorValue
    output: TensorValue
    scale: float
    causal: bool
    accumulation_dtype: DType
    source_location: str | None

    @property
    def query_heads(self) -> int:
        return self.query.shape[2]

    @property
    def key_value_heads(self) -> int:
        return self.key.shape[2]

    @property
    def head_dimension(self) -> int:
        return self.query.shape[3]


@dataclass(frozen=True)
class SwiGLUOp:
    """SwiGLU over separately projected gate and up activations."""

    id: int
    gate: TensorValue
    up: TensorValue
    output: TensorValue
    source_location: str | None


@dataclass(frozen=True)
class PairwiseSwiGLUOp:
    """SwiGLU over adjacent ``(gate, up)`` pairs in one projection."""

    id: int
    input: TensorValue
    output: TensorValue
    source_location: str | None


@dataclass(frozen=True)
class QKVProjectionOp:
    """One projection whose output is partitioned into separate Q, K, and V views."""

    id: int
    input: TensorValue
    weight: TensorValue
    output: TensorValue
    query: TensorValue
    key: TensorValue
    value: TensorValue
    accumulation_dtype: DType
    source_location: str | None


@dataclass(frozen=True)
class RoPEOp:
    """Adjacent-pair rotary transform over Q and K head dimensions."""

    id: int
    query: TensorValue
    key: TensorValue
    sine: TensorValue
    cosine: TensorValue
    output: TensorValue
    key_output: TensorValue
    rotary_dimension: int
    source_location: str | None


@dataclass(frozen=True)
class ViewOp:
    """Zero-cost reshape or flattening view with unchanged element order."""

    id: int
    input: TensorValue
    output: TensorValue
    source_location: str | None


@dataclass(frozen=True)
class TopKRouterOp:
    """Select expert indices and weights from per-token router logits."""

    id: int
    logits: TensorValue
    expert_indices: TensorValue
    output: TensorValue
    top_k: int
    normalize_weights: bool
    source_location: str | None


@dataclass(frozen=True)
class SharedExpertMLPOp:
    """One shared gated MLP evaluated for every token."""

    id: int
    input: TensorValue
    gate_weight: TensorValue
    up_weight: TensorValue
    down_weight: TensorValue
    gate: TensorValue
    up: TensorValue
    hidden: TensorValue
    output: TensorValue
    accumulation_dtype: DType
    source_location: str | None


@dataclass(frozen=True)
class RoutedExpertMLPOp:
    """Top-k gated MLP evaluation using one selected expert per route."""

    id: int
    input: TensorValue
    expert_indices: TensorValue
    gate_weight: TensorValue
    up_weight: TensorValue
    down_weight: TensorValue
    gate: TensorValue
    up: TensorValue
    hidden: TensorValue
    output: TensorValue
    accumulation_dtype: DType
    source_location: str | None


@dataclass(frozen=True)
class WeightedExpertCombineOp:
    """Add the shared expert result to the weighted routed-expert sum."""

    id: int
    shared: TensorValue
    routed: TensorValue
    route_weights: TensorValue
    output: TensorValue
    source_location: str | None


SemanticOp = (
    LinearOp
    | ResidualAddOp
    | RMSNormOp
    | ScaledDotProductAttentionOp
    | SwiGLUOp
    | PairwiseSwiGLUOp
    | QKVProjectionOp
    | RoPEOp
    | ViewOp
    | TopKRouterOp
    | SharedExpertMLPOp
    | RoutedExpertMLPOp
    | WeightedExpertCombineOp
)


@dataclass
class TensorGraph:
    """Builder and query interface for a normalized semantic tensor graph."""

    _values: list[TensorValue] = field(default_factory=list)
    _operations: list[SemanticOp] = field(default_factory=list)

    @property
    def values(self) -> tuple[TensorValue, ...]:
        return tuple(self._values)

    @property
    def operations(self) -> tuple[SemanticOp, ...]:
        return tuple(self._operations)

    def input(self, name: str, *, shape: tuple[int, ...], dtype: DType) -> TensorValue:
        """Add a runtime input value."""
        return self._new_value(name, shape=shape, dtype=dtype)

    def parameter(self, name: str, *, shape: tuple[int, ...], dtype: DType) -> TensorValue:
        """Add a parameter value."""
        return self._new_value(name, shape=shape, dtype=dtype)

    def linear(
        self,
        value: TensorValue,
        weight: TensorValue,
        *,
        name: str,
        accumulation_dtype: DType,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add a right-multiplication with explicit accumulation precision."""
        if len(value.shape) != 2 or len(weight.shape) != 2:
            raise ValueError("the prototype linear operation requires rank-two inputs")
        if value.shape[1] != weight.shape[0]:
            raise ValueError(f"linear reduction dimensions differ: {value.shape[1]} != {weight.shape[0]}")

        output = self._new_value(name, shape=(value.shape[0], weight.shape[1]), dtype=value.dtype)
        self._operations.append(
            LinearOp(
                id=len(self._operations),
                input=value,
                weight=weight,
                output=output,
                accumulation_dtype=accumulation_dtype,
                source_location=source_location,
            )
        )
        return output

    def residual_add(
        self, left: TensorValue, right: TensorValue, *, name: str, source_location: str | None = None
    ) -> TensorValue:
        """Add two identically shaped tensors."""
        if left.shape != right.shape:
            raise ValueError(f"residual shapes differ: {left.shape} != {right.shape}")
        if left.dtype != right.dtype:
            raise ValueError(f"residual dtypes differ: {left.dtype} != {right.dtype}")

        output = self._new_value(name, shape=left.shape, dtype=left.dtype)
        self._operations.append(
            ResidualAddOp(
                id=len(self._operations),
                left=left,
                right=right,
                output=output,
                source_location=source_location,
            )
        )
        return output

    def view(
        self,
        value: TensorValue,
        *,
        shape: tuple[int, ...],
        name: str,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add a zero-copy reshape preserving contiguous logical element order."""
        if prod(value.shape) != prod(shape):
            raise ValueError(f"view element counts differ: {value.shape} -> {shape}")
        output = self._new_value(name, shape=shape, dtype=value.dtype)
        self._operations.append(
            ViewOp(
                id=len(self._operations),
                input=value,
                output=output,
                source_location=source_location,
            )
        )
        return output

    def rms_norm(
        self,
        value: TensorValue,
        gamma: TensorValue,
        *,
        name: str,
        axis: int,
        epsilon: float,
        reduction_dtype: DType,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add RMS normalization with an explicit reduction axis and dtype."""
        normalized_axis = axis % len(value.shape)
        if gamma.shape != (value.shape[normalized_axis],):
            raise ValueError(
                f"gamma shape {gamma.shape} does not match normalization dimension {value.shape[normalized_axis]}"
            )
        if epsilon < 0:
            raise ValueError("RMSNorm epsilon must be non-negative")

        output = self._new_value(name, shape=value.shape, dtype=value.dtype)
        self._operations.append(
            RMSNormOp(
                id=len(self._operations),
                input=value,
                gamma=gamma,
                output=output,
                axis=normalized_axis,
                epsilon=epsilon,
                reduction_dtype=reduction_dtype,
                source_location=source_location,
            )
        )
        return output

    def scaled_dot_product_attention(
        self,
        query: TensorValue,
        key: TensorValue,
        value: TensorValue,
        *,
        name: str,
        scale: float,
        causal: bool,
        accumulation_dtype: DType,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add exact MHA or grouped-query attention with explicit semantics."""
        if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
            raise ValueError("attention requires Q, K, and V in [batch, sequence, heads, dimension] layout")

        query_batch, query_length, query_heads, query_dimension = query.shape
        key_batch, key_length, key_value_heads, key_dimension = key.shape
        value_batch, value_length, value_heads, value_dimension = value.shape
        if query_batch != key_batch or query_batch != value_batch:
            raise ValueError("attention batch dimensions differ")
        if key_length != value_length:
            raise ValueError("attention K and V sequence dimensions differ")
        if key_value_heads != value_heads:
            raise ValueError("attention K and V head counts differ")
        if query_dimension != key_dimension:
            raise ValueError("attention Q and K head dimensions differ")
        if query_heads % key_value_heads != 0:
            raise ValueError("query head count must be divisible by the KV head count")
        if causal and query_length != key_length:
            raise ValueError("the initial causal-attention prototype requires equal Q and KV sequence lengths")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("attention Q, K, and V dtypes differ")
        if scale <= 0:
            raise ValueError("attention scale must be positive")

        output = self._new_value(
            name,
            shape=(query_batch, query_length, query_heads, value_dimension),
            dtype=query.dtype,
        )
        self._operations.append(
            ScaledDotProductAttentionOp(
                id=len(self._operations),
                query=query,
                key=key,
                value=value,
                output=output,
                scale=scale,
                causal=causal,
                accumulation_dtype=accumulation_dtype,
                source_location=source_location,
            )
        )
        return output

    def swiglu(
        self,
        gate: TensorValue,
        up: TensorValue,
        *,
        name: str,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add SwiGLU over separate, identically shaped gate and up values."""
        if gate.shape != up.shape:
            raise ValueError(f"SwiGLU gate/up shapes differ: {gate.shape} != {up.shape}")
        if gate.dtype != up.dtype:
            raise ValueError(f"SwiGLU gate/up dtypes differ: {gate.dtype} != {up.dtype}")

        output = self._new_value(name, shape=gate.shape, dtype=gate.dtype)
        self._operations.append(
            SwiGLUOp(
                id=len(self._operations),
                gate=gate,
                up=up,
                output=output,
                source_location=source_location,
            )
        )
        return output

    def pairwise_swiglu(
        self,
        value: TensorValue,
        *,
        name: str,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add SwiGLU over adjacent ``(gate, up)`` pairs in the last dimension."""
        if value.shape[-1] % 2 != 0:
            raise ValueError(f"pairwise SwiGLU input width must be even, got {value.shape[-1]}")

        output = self._new_value(name, shape=(*value.shape[:-1], value.shape[-1] // 2), dtype=value.dtype)
        self._operations.append(
            PairwiseSwiGLUOp(
                id=len(self._operations),
                input=value,
                output=output,
                source_location=source_location,
            )
        )
        return output

    def qkv_projection(
        self,
        value: TensorValue,
        weight: TensorValue,
        *,
        name: str,
        query_heads: int,
        key_value_heads: int,
        head_dimension: int,
        accumulation_dtype: DType,
        source_location: str | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Add a combined QKV projection producing logical BSHD segment views."""
        if len(value.shape) != 3 or len(weight.shape) != 2:
            raise ValueError("QKV projection requires a BSH input and a rank-two weight")
        if value.shape[-1] != weight.shape[0]:
            raise ValueError(f"QKV reduction dimensions differ: {value.shape[-1]} != {weight.shape[0]}")
        if query_heads <= 0 or key_value_heads <= 0 or head_dimension <= 0:
            raise ValueError("QKV head counts and head dimension must be positive")
        if query_heads % key_value_heads != 0:
            raise ValueError("QKV query head count must be divisible by the KV head count")

        expected_width = (query_heads + 2 * key_value_heads) * head_dimension
        if weight.shape[1] != expected_width:
            raise ValueError(f"QKV projection width {weight.shape[1]} does not match expected width {expected_width}")

        batch, sequence, _ = value.shape
        packed = self._new_value(name, shape=(batch, sequence, expected_width), dtype=value.dtype)
        query = self._new_value(
            f"{name}.query",
            shape=(batch, sequence, query_heads, head_dimension),
            dtype=value.dtype,
        )
        key = self._new_value(
            f"{name}.key",
            shape=(batch, sequence, key_value_heads, head_dimension),
            dtype=value.dtype,
        )
        projected_value = self._new_value(
            f"{name}.value",
            shape=(batch, sequence, key_value_heads, head_dimension),
            dtype=value.dtype,
        )
        self._operations.append(
            QKVProjectionOp(
                id=len(self._operations),
                input=value,
                weight=weight,
                output=packed,
                query=query,
                key=key,
                value=projected_value,
                accumulation_dtype=accumulation_dtype,
                source_location=source_location,
            )
        )
        return query, key, projected_value

    def rope(
        self,
        query: TensorValue,
        key: TensorValue,
        sine: TensorValue,
        cosine: TensorValue,
        *,
        name: str,
        rotary_dimension: int,
        source_location: str | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Add an adjacent-pair RoPE transform to Q and K."""
        if len(query.shape) != 4 or len(key.shape) != 4:
            raise ValueError("RoPE requires Q and K in BSHD layout")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError("RoPE Q and K batch, sequence, or head dimensions differ")
        if query.dtype != key.dtype:
            raise ValueError("RoPE Q and K dtypes differ")
        if rotary_dimension <= 0 or rotary_dimension % 2 != 0 or rotary_dimension > query.shape[-1]:
            raise ValueError("RoPE rotary dimension must be positive, even, and no larger than the head dimension")
        expected_table_shape = (query.shape[1], rotary_dimension // 2)
        if sine.shape != expected_table_shape or cosine.shape != expected_table_shape:
            raise ValueError(
                f"RoPE sine/cosine tables must have shape {expected_table_shape}, got {sine.shape} and {cosine.shape}"
            )
        if sine.dtype != query.dtype or cosine.dtype != query.dtype:
            raise ValueError("RoPE sine/cosine dtype must match Q and K")

        query_output = self._new_value(f"{name}.query", shape=query.shape, dtype=query.dtype)
        key_output = self._new_value(f"{name}.key", shape=key.shape, dtype=key.dtype)
        self._operations.append(
            RoPEOp(
                id=len(self._operations),
                query=query,
                key=key,
                sine=sine,
                cosine=cosine,
                output=query_output,
                key_output=key_output,
                rotary_dimension=rotary_dimension,
                source_location=source_location,
            )
        )
        return query_output, key_output

    def top_k_router(
        self,
        logits: TensorValue,
        *,
        name: str,
        top_k: int,
        normalize_weights: bool = True,
        source_location: str | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Add deterministic top-k selection over the expert dimension."""
        if len(logits.shape) != 2:
            raise ValueError("top-k routing requires [tokens, experts] logits")
        if not 0 < top_k <= logits.shape[1]:
            raise ValueError(f"top-k must be in [1, {logits.shape[1]}], got {top_k}")

        route_shape = (logits.shape[0], top_k)
        expert_indices = self._new_value(f"{name}.expert_indices", shape=route_shape, dtype=DType.INT32)
        route_weights = self._new_value(f"{name}.weights", shape=route_shape, dtype=DType.FP32)
        self._operations.append(
            TopKRouterOp(
                id=len(self._operations),
                logits=logits,
                expert_indices=expert_indices,
                output=route_weights,
                top_k=top_k,
                normalize_weights=normalize_weights,
                source_location=source_location,
            )
        )
        return expert_indices, route_weights

    def shared_expert_mlp(
        self,
        value: TensorValue,
        gate_weight: TensorValue,
        up_weight: TensorValue,
        down_weight: TensorValue,
        *,
        name: str,
        accumulation_dtype: DType,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add an ordinary shared gate/up, SwiGLU, and down projection."""
        if len(value.shape) != 2:
            raise ValueError("shared expert input must have shape [tokens, hidden]")
        hidden = value.shape[1]
        if len(gate_weight.shape) != 2 or gate_weight.shape[1] != hidden:
            raise ValueError("shared gate weight must have shape [intermediate, hidden]")
        intermediate = gate_weight.shape[0]
        if up_weight.shape != (intermediate, hidden):
            raise ValueError("shared up weight must match the shared gate weight")
        if down_weight.shape != (hidden, intermediate):
            raise ValueError("shared down weight must have shape [hidden, intermediate]")

        gate = self._new_value(f"{name}.gate", shape=(value.shape[0], intermediate), dtype=value.dtype)
        up = self._new_value(f"{name}.up", shape=gate.shape, dtype=value.dtype)
        activated = self._new_value(f"{name}.hidden", shape=gate.shape, dtype=value.dtype)
        output = self._new_value(name, shape=value.shape, dtype=value.dtype)
        self._operations.append(
            SharedExpertMLPOp(
                id=len(self._operations),
                input=value,
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=down_weight,
                gate=gate,
                up=up,
                hidden=activated,
                output=output,
                accumulation_dtype=accumulation_dtype,
                source_location=source_location,
            )
        )
        return output

    def routed_expert_mlp(
        self,
        value: TensorValue,
        expert_indices: TensorValue,
        gate_weight: TensorValue,
        up_weight: TensorValue,
        down_weight: TensorValue,
        *,
        name: str,
        accumulation_dtype: DType,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add top-k routed gate/up, SwiGLU, and down projections."""
        if len(value.shape) != 2:
            raise ValueError("routed expert input must have shape [tokens, hidden]")
        if len(expert_indices.shape) != 2 or expert_indices.shape[0] != value.shape[0]:
            raise ValueError("expert indices must have shape [tokens, top_k]")
        if expert_indices.dtype is not DType.INT32:
            raise ValueError("expert indices must have INT32 dtype")
        if len(gate_weight.shape) != 3 or gate_weight.shape[2] != value.shape[1]:
            raise ValueError("routed gate weight must have shape [local_experts, intermediate, hidden]")
        local_experts, intermediate, hidden = gate_weight.shape
        if up_weight.shape != (local_experts, intermediate, hidden):
            raise ValueError("routed up weight must match the routed gate weight")
        if down_weight.shape != (local_experts, hidden, intermediate):
            raise ValueError("routed down weight must have shape [local_experts, hidden, intermediate]")

        route_shape = (value.shape[0], expert_indices.shape[1])
        intermediate_shape = (*route_shape, intermediate)
        gate = self._new_value(f"{name}.gate", shape=intermediate_shape, dtype=value.dtype)
        up = self._new_value(f"{name}.up", shape=intermediate_shape, dtype=value.dtype)
        activated = self._new_value(f"{name}.hidden", shape=intermediate_shape, dtype=value.dtype)
        output = self._new_value(name, shape=(*route_shape, hidden), dtype=value.dtype)
        self._operations.append(
            RoutedExpertMLPOp(
                id=len(self._operations),
                input=value,
                expert_indices=expert_indices,
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=down_weight,
                gate=gate,
                up=up,
                hidden=activated,
                output=output,
                accumulation_dtype=accumulation_dtype,
                source_location=source_location,
            )
        )
        return output

    def weighted_expert_combine(
        self,
        shared: TensorValue,
        routed: TensorValue,
        route_weights: TensorValue,
        *,
        name: str,
        source_location: str | None = None,
    ) -> TensorValue:
        """Add shared output and the route-weighted top-k expert outputs."""
        if len(shared.shape) != 2 or len(routed.shape) != 3:
            raise ValueError("expert combine expects shared [tokens, hidden] and routed [tokens, top_k, hidden]")
        if routed.shape[0] != shared.shape[0] or routed.shape[2] != shared.shape[1]:
            raise ValueError("shared and routed expert output shapes are incompatible")
        if route_weights.shape != routed.shape[:2] or route_weights.dtype is not DType.FP32:
            raise ValueError("route weights must be FP32 with shape [tokens, top_k]")
        if shared.dtype != routed.dtype:
            raise ValueError("shared and routed expert output dtypes differ")

        output = self._new_value(name, shape=shared.shape, dtype=shared.dtype)
        self._operations.append(
            WeightedExpertCombineOp(
                id=len(self._operations),
                shared=shared,
                routed=routed,
                route_weights=route_weights,
                output=output,
                source_location=source_location,
            )
        )
        return output

    def producer(self, value: TensorValue) -> SemanticOp | None:
        """Return the operation producing a value, if any."""
        for operation in self._operations:
            if value in _operation_outputs(operation):
                return operation
        return None

    def consumers(self, value: TensorValue) -> tuple[SemanticOp, ...]:
        """Return operations that consume a value."""
        return tuple(operation for operation in self._operations if value in _operation_inputs(operation))

    def _new_value(self, name: str, *, shape: tuple[int, ...], dtype: DType) -> TensorValue:
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"tensor {name!r} has invalid shape {shape}")
        if any(value.name == name for value in self._values):
            raise ValueError(f"tensor name {name!r} is already in use")

        value = TensorValue(id=len(self._values), name=name, shape=shape, dtype=dtype)
        self._values.append(value)
        return value


def operation_inputs(operation: SemanticOp) -> tuple[TensorValue, ...]:
    """Return the logical inputs of a semantic operation."""
    return _operation_inputs(operation)


def _operation_inputs(operation: SemanticOp) -> tuple[TensorValue, ...]:
    if isinstance(operation, LinearOp):
        return operation.input, operation.weight
    if isinstance(operation, ResidualAddOp):
        return operation.left, operation.right
    if isinstance(operation, RMSNormOp):
        return operation.input, operation.gamma
    if isinstance(operation, ScaledDotProductAttentionOp):
        return operation.query, operation.key, operation.value
    if isinstance(operation, SwiGLUOp):
        return operation.gate, operation.up
    if isinstance(operation, PairwiseSwiGLUOp):
        return (operation.input,)
    if isinstance(operation, QKVProjectionOp):
        return operation.input, operation.weight
    if isinstance(operation, RoPEOp):
        return operation.query, operation.key, operation.sine, operation.cosine
    if isinstance(operation, TopKRouterOp):
        return (operation.logits,)
    if isinstance(operation, SharedExpertMLPOp):
        return operation.input, operation.gate_weight, operation.up_weight, operation.down_weight
    if isinstance(operation, RoutedExpertMLPOp):
        return (
            operation.input,
            operation.expert_indices,
            operation.gate_weight,
            operation.up_weight,
            operation.down_weight,
        )
    if isinstance(operation, WeightedExpertCombineOp):
        return operation.shared, operation.routed, operation.route_weights
    return (operation.input,)


def _operation_outputs(operation: SemanticOp) -> tuple[TensorValue, ...]:
    if isinstance(operation, QKVProjectionOp):
        return operation.output, operation.query, operation.key, operation.value
    if isinstance(operation, RoPEOp):
        return operation.output, operation.key_output
    if isinstance(operation, TopKRouterOp):
        return operation.expert_indices, operation.output
    if isinstance(operation, SharedExpertMLPOp | RoutedExpertMLPOp):
        return operation.gate, operation.up, operation.hidden, operation.output
    return (operation.output,)


def operation_outputs(operations: Iterable[SemanticOp]) -> tuple[TensorValue, ...]:
    """Return operation outputs in graph order."""
    return tuple(output for operation in operations for output in _operation_outputs(operation))
