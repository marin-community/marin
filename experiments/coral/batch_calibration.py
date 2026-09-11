# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate dense-transformer HBM and select a TPU batch configuration."""

import math
from dataclasses import dataclass

from fray.types import get_tpu_topology, tpu_hbm_capacity_bytes

BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class TpuBatchConfig:
    """Parallelism settings selected for a TPU training batch."""

    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def adam_optimizer_bytes(
    parameter_count: int,
    *,
    first_moment_dtype_bytes: int = 4,
    second_moment_dtype_bytes: int = 4,
) -> int:
    """Adam-family optimizer-state bytes: the first- and second-moment buffers."""
    first_moment_bytes = parameter_count * first_moment_dtype_bytes
    second_moment_bytes = parameter_count * second_moment_dtype_bytes
    return first_moment_bytes + second_moment_bytes


def dense_transformer_bytes(
    *,
    parameter_count: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    parameter_dtype_bytes: int = 4,
    activation_dtype_bytes: int = 2,
    activation_layer_fraction: float = 0.75,
    activation_layer_floor: int = 4,
) -> tuple[int, int]:
    """Estimate parameter and activation bytes for a dense transformer.

    Args:
        parameter_count: Number of trainable model parameters.
        batch_size: Global training batch size.
        seq_len: Tokens per training example.
        hidden_dim: Transformer hidden dimension.
        intermediate_dim: MLP intermediate dimension.
        num_layers: Number of transformer layers.
        parameter_dtype_bytes: Bytes used by each model parameter.
        activation_dtype_bytes: Bytes used by each activation element.
        activation_layer_fraction: Fraction of layers whose activations are resident simultaneously.
        activation_layer_floor: Minimum number of layers whose activations are resident simultaneously.

    Returns:
        A ``(parameter_bytes, activation_bytes)`` tuple.

    Raises:
        ValueError: If ``activation_layer_fraction`` is outside ``[0, 1]``.
    """
    _validate_activation_layer_fraction(activation_layer_fraction)

    hidden_activation_bytes = batch_size * seq_len * hidden_dim * activation_dtype_bytes
    attention_activation_bytes = batch_size * seq_len * hidden_dim * 4 * activation_dtype_bytes
    mlp_activation_bytes = batch_size * seq_len * intermediate_dim * activation_dtype_bytes
    per_layer_activation_bytes = hidden_activation_bytes + attention_activation_bytes + mlp_activation_bytes
    # With gradient checkpointing, only a fraction of layer activations are resident at once;
    # the explicit floor keeps the estimate useful for shallow models.
    saved_activation_layers = max(
        math.floor(num_layers * activation_layer_fraction),
        activation_layer_floor,
    )
    parameter_bytes = parameter_count * parameter_dtype_bytes
    activation_bytes = per_layer_activation_bytes * saved_activation_layers
    return parameter_bytes, activation_bytes


def batch_memory_bytes(
    *,
    parameter_bytes: int,
    optimizer_bytes: int,
    activation_bytes: int,
    correction_factor: float = 1.0,
) -> int:
    """Return total HBM bytes for a global batch.

    Args:
        parameter_bytes: Model parameter memory.
        optimizer_bytes: Optimizer-state memory.
        activation_bytes: Activation memory for the global batch.
        correction_factor: Empirical scale factor for the raw bucket sum. The default ``1.0`` means no
            ad hoc correction is applied.

    Returns:
        The corrected global-batch memory estimate in bytes.

    Raises:
        ValueError: If a byte bucket is negative or ``correction_factor`` is not positive.
    """
    if min(parameter_bytes, optimizer_bytes, activation_bytes) < 0:
        raise ValueError("Memory byte counts must be non-negative.")
    if correction_factor <= 0:
        raise ValueError(f"correction_factor must be positive, got {correction_factor}")
    return math.ceil((parameter_bytes + optimizer_bytes + activation_bytes) * correction_factor)


def tpu_batch_config(
    tpu: str,
    batch_size: int,
    batch_bytes: int,
    *,
    context_parallelism: int = 1,
    slice_count: int = 1,
) -> TpuBatchConfig:
    """Select Levanter parallelism settings that fit a global batch on TPUs.

    Args:
        tpu: TPU topology name accepted by Fray, such as ``"v5litepod-8"``.
        batch_size: Global training batch size.
        batch_bytes: HBM needed for the full global batch. When it does not fit, memory is assumed to
            scale linearly with microbatch size.
        context_parallelism: Number of chips that split each example's sequence positions within a slice.
        slice_count: Number of identical TPU slices used by the training job.

    Returns:
        The effective data, tensor, per-device, and gradient-accumulation parallelism. Tensor parallelism
        uses the chips not assigned to data or context parallelism. All returned values are positive and
        explicit, including when the full batch fits in one microstep.

    Raises:
        ValueError: If the inputs are invalid or no microbatch fits the requested slices.
    """
    topology = get_tpu_topology(tpu)
    _validate_parallelism_inputs(
        batch_size=batch_size,
        chips_per_slice=topology.chip_count,
        context_parallelism=context_parallelism,
        slice_count=slice_count,
    )
    if batch_bytes <= 0:
        raise ValueError(f"batch_bytes must be positive, got {batch_bytes}")

    examples_per_slice = batch_size // slice_count
    parallel_capacity_per_slice = topology.chip_count // context_parallelism
    data_parallelism_per_slice = math.gcd(examples_per_slice, parallel_capacity_per_slice)
    data_parallelism = slice_count * data_parallelism_per_slice
    tensor_parallelism = parallel_capacity_per_slice // data_parallelism_per_slice

    capacity_bytes = tpu_hbm_capacity_bytes(tpu) * slice_count
    if batch_bytes <= capacity_bytes:
        return TpuBatchConfig(
            data_parallelism=data_parallelism,
            tensor_parallelism=tensor_parallelism,
            per_device_parallelism=batch_size // data_parallelism,
            gradient_accumulation=1,
        )

    full_per_device_batch = batch_size // data_parallelism
    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism != 0:
            continue
        microbatch_size = per_device_parallelism * data_parallelism
        if _batch_bytes_for_microbatch(batch_bytes, full_batch_size=batch_size, microbatch_size=microbatch_size) <= (
            capacity_bytes
        ):
            return TpuBatchConfig(
                data_parallelism=data_parallelism,
                tensor_parallelism=tensor_parallelism,
                per_device_parallelism=per_device_parallelism,
                gradient_accumulation=batch_size // microbatch_size,
            )

    minimum_microbatch_bytes = _batch_bytes_for_microbatch(
        batch_bytes,
        full_batch_size=batch_size,
        microbatch_size=data_parallelism,
    )
    raise ValueError(
        f"Batch does not fit on {tpu}: even per_device_parallelism=1 "
        f"(microbatch_size={data_parallelism}) needs {_format_gib(minimum_microbatch_bytes)}, "
        f"but target HBM capacity is {_format_gib(capacity_bytes)}. Use a larger TPU slice, "
        "more slices, a larger batch size, or more model/context parallelism."
    )


def _batch_bytes_for_microbatch(batch_bytes: int, *, full_batch_size: int, microbatch_size: int) -> int:
    return math.ceil(batch_bytes * microbatch_size / full_batch_size)


def _validate_parallelism_inputs(
    *,
    batch_size: int,
    chips_per_slice: int,
    context_parallelism: int,
    slice_count: int,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_parallelism <= 0:
        raise ValueError(f"context_parallelism must be positive, got {context_parallelism}")
    if slice_count <= 0:
        raise ValueError(f"slice_count must be positive, got {slice_count}")
    if chips_per_slice % context_parallelism != 0:
        raise ValueError(f"context_parallelism ({context_parallelism}) must divide chips per slice ({chips_per_slice})")
    if batch_size % slice_count != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by slice_count ({slice_count})")


def _validate_activation_layer_fraction(activation_layer_fraction: float) -> None:
    if not (0.0 <= activation_layer_fraction <= 1.0):
        raise ValueError(
            f"activation_layer_fraction must be in the interval [0.0, 1.0], got {activation_layer_fraction}"
        )


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GIB:.2f} GiB"
