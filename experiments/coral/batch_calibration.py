# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

from fray.types import get_tpu_topology, tpu_family

BYTES_PER_GIB = 1024**3


TPU_GENERATION_MEMORY: dict[str, int] = {
    "v4": 32 * BYTES_PER_GIB,
    "v5e": 16 * BYTES_PER_GIB,
    "v5p": 95 * BYTES_PER_GIB,
    "v6e": 32 * BYTES_PER_GIB,
}


def adam_optimizer_bytes(
    parameter_count: int,
    *,
    first_moment_dtype_bytes: int = 4,
    second_moment_dtype_bytes: int = 4,
) -> int:
    """Adam-family optimizer-state bytes: the first- and second-moment buffers, excluding gradients."""
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
    activation_layer_fraction: float = 0.75,
) -> tuple[int, int]:
    """Estimate ``(param_bytes, activation_bytes)`` for a dense transformer."""
    _validate_activation_layer_fraction(activation_layer_fraction)

    hidden_activation_bytes = batch_size * seq_len * hidden_dim * 2
    attention_activation_bytes = batch_size * seq_len * hidden_dim * 4 * 2
    mlp_activation_bytes = batch_size * seq_len * intermediate_dim * 2
    per_layer_activation_bytes = hidden_activation_bytes + attention_activation_bytes + mlp_activation_bytes
    # Layers whose activations are resident at once: with gradient checkpointing only a fraction are,
    # floored at 4 for shallow models.
    saved_activation_layers = max(math.floor(num_layers * activation_layer_fraction), 4)
    param_bytes = parameter_count * parameter_dtype_bytes
    activation_bytes = per_layer_activation_bytes * saved_activation_layers
    return param_bytes, activation_bytes


def batch_memory_bytes(
    *,
    param_bytes: int,
    optimizer_bytes: int,
    activation_bytes: int,
    correction_factor: float = 1.0,
) -> int:
    """Total HBM for one global batch: the byte buckets scaled by ``correction_factor``.

    ``correction_factor`` scales the (rough) bucket sum to absorb HBM the buckets omit — framework
    runtime, allocator fragmentation, and estimate error; calibrate it empirically. ``1.0`` is the raw sum.
    """
    return math.ceil((param_bytes + optimizer_bytes + activation_bytes) * correction_factor)


def tpu_batch_config(
    tpu: str,
    batch_size: int,
    batch_bytes: int,
    *,
    data_axis_size: int | None = None,
) -> tuple[int, int]:
    """Return ``(per_device_parallelism, gradient_accumulation)`` to fit ``batch_size`` on ``tpu``.

    ``batch_bytes`` is the HBM to place the full global batch (from ``batch_memory_bytes`` or a measured
    run); when it does not fit, memory is assumed to scale linearly with microbatch size.
    ``data_axis_size`` defaults to the slice chip count; pass a smaller value when some chips are off the
    batch axis — e.g. ``"v5p-32"`` (16 chips) with tensor-parallel size 2 uses ``data_axis_size=8``.
    ``per_device_parallelism`` is ``-1`` when the full batch fits without microbatching.
    """
    topology = get_tpu_topology(tpu)
    data_axis_size = _resolve_data_axis_size(topology.chip_count, data_axis_size)
    _validate_batch_divisible_by_data_axis(batch_size, data_axis_size)

    capacity_bytes = _slice_hbm_capacity_bytes(tpu)
    if batch_bytes <= capacity_bytes:
        return -1, 1

    full_per_device_batch = batch_size // data_axis_size
    for pdp in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % pdp != 0:
            continue
        microbatch_size = pdp * data_axis_size
        if _batch_bytes_for_microbatch(batch_bytes, full_batch_size=batch_size, microbatch_size=microbatch_size) <= (
            capacity_bytes
        ):
            return pdp, batch_size // microbatch_size

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


def tpu_hbm_capacity_bytes(tpu: str) -> int:
    return _slice_hbm_capacity_bytes(tpu)


def _batch_bytes_for_microbatch(batch_bytes: int, *, full_batch_size: int, microbatch_size: int) -> int:
    return math.ceil(batch_bytes * microbatch_size / full_batch_size)


def _slice_hbm_capacity_bytes(tpu: str) -> int:
    topology = get_tpu_topology(tpu)
    family = tpu_family(tpu)
    generation_memory = TPU_GENERATION_MEMORY.get(family)
    if generation_memory is None:
        raise ValueError(
            f"No HBM memory spec for TPU family {family!r}. Known families: {sorted(TPU_GENERATION_MEMORY)}"
        )
    return topology.chip_count * generation_memory


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
