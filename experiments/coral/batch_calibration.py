# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate dense-transformer HBM and select a TPU batch configuration."""

import math

from fray.types import get_tpu_topology, tpu_hbm_capacity_bytes

BYTES_PER_GIB = 1024**3


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
    data_axis_size: int | None = None,
) -> tuple[int, int]:
    """Select Levanter batch settings that fit a global batch on a TPU slice.

    Args:
        tpu: TPU topology name accepted by Fray, such as ``"v5litepod-8"``.
        batch_size: Global training batch size.
        batch_bytes: HBM needed for the full global batch. When it does not fit, memory is assumed to
            scale linearly with microbatch size.
        data_axis_size: Chips assigned to Levanter's batch axis. Defaults to the slice chip count. For
            example, ``"v5p-32"`` has 16 chips, so tensor parallelism of two uses ``data_axis_size=8``.

    Returns:
        A ``(per_device_parallelism, gradient_accumulation)`` tuple. ``per_device_parallelism`` is ``-1``
        when the full batch fits without microbatching.

    Raises:
        ValueError: If the inputs are invalid or no microbatch fits the slice.
    """
    topology = get_tpu_topology(tpu)
    data_axis_size = _resolve_data_axis_size(topology.chip_count, data_axis_size)
    _validate_batch_divisible_by_data_axis(batch_size, data_axis_size)
    if batch_bytes <= 0:
        raise ValueError(f"batch_bytes must be positive, got {batch_bytes}")

    capacity_bytes = tpu_hbm_capacity_bytes(tpu)
    if batch_bytes <= capacity_bytes:
        return -1, 1

    full_per_device_batch = batch_size // data_axis_size
    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism != 0:
            continue
        microbatch_size = per_device_parallelism * data_axis_size
        if _batch_bytes_for_microbatch(batch_bytes, full_batch_size=batch_size, microbatch_size=microbatch_size) <= (
            capacity_bytes
        ):
            return per_device_parallelism, batch_size // microbatch_size

    minimum_microbatch_bytes = _batch_bytes_for_microbatch(
        batch_bytes,
        full_batch_size=batch_size,
        microbatch_size=data_axis_size,
    )
    raise ValueError(
        f"Batch does not fit on {tpu}: even per_device_parallelism=1 "
        f"(microbatch_size={data_axis_size}) needs {_format_gib(minimum_microbatch_bytes)}, "
        f"but target HBM capacity is {_format_gib(capacity_bytes)}. Use a larger TPU slice, "
        "reduce model/sequence size, or use more model/context parallelism and pass the resulting data_axis_size."
    )


def _batch_bytes_for_microbatch(batch_bytes: int, *, full_batch_size: int, microbatch_size: int) -> int:
    return math.ceil(batch_bytes * microbatch_size / full_batch_size)


def _resolve_data_axis_size(chip_count: int, data_axis_size: int | None) -> int:
    if data_axis_size is None:
        return chip_count
    if data_axis_size <= 0:
        raise ValueError(f"data_axis_size must be positive, got {data_axis_size}")
    if chip_count % data_axis_size != 0:
        raise ValueError(
            f"data_axis_size ({data_axis_size}) must divide TPU chip count ({chip_count}). "
            "For tensor/model/context parallelism, pass the number of chips left on Levanter's batch axis."
        )
    return data_axis_size


def _validate_batch_divisible_by_data_axis(batch_size: int, data_axis_size: int) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size % data_axis_size == 0:
        return
    next_valid = math.ceil(batch_size / data_axis_size) * data_axis_size
    raise ValueError(
        f"batch_size ({batch_size}) must be divisible by data_axis_size ({data_axis_size}) for Levanter "
        f"microbatching. Use batch_size={next_valid}, choose another batch size, or pass the correct data_axis_size."
    )


def _validate_activation_layer_fraction(activation_layer_fraction: float) -> None:
    if not (0.0 <= activation_layer_fraction <= 1.0):
        raise ValueError(
            f"activation_layer_fraction must be in the interval [0.0, 1.0], got {activation_layer_fraction}"
        )


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GIB:.2f} GiB"
