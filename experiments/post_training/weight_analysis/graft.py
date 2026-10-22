# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grafting builder: ``B + Δ·1[S]`` checkpoints over Grug component groups.

With ``Δ = θ - B``, ``B + Δ·1[S]`` is θ's tensor on every component in ``S``
and B's elsewhere, so the builder copies tensors instead of adding deltas. In
fp32, ``B + (θ - B)`` is not θ for weights near zero; copying is exact, and
grafting every group reproduces θ bit for bit.

The hybrid is written in fp32 (B upcast where it is kept) as a sharded
safetensors checkpoint with its index plus B's config, generation config and
tokenizer, so vLLM can serve it, casting to its serving dtype at load. Both
inputs pass the C8 checks of
:func:`~experiments.post_training.weight_analysis.delta_stats.paired_blocks`
on the tensors read: θ in fp32, B bf16 or its exact fp32 upcast.
"""

from collections.abc import Iterator

import ml_dtypes
import numpy as np

from experiments.post_training.weight_analysis.checkpoint_io import (
    SafetensorsCheckpoint,
    TensorSpec,
    copy_serving_files,
    write_sharded_checkpoint,
)
from experiments.post_training.weight_analysis.components import Component, component_of

_FP32 = np.dtype(np.float32)
_BF16 = np.dtype(ml_dtypes.bfloat16)


def _validated_plan(base: SafetensorsCheckpoint, checkpoint: SafetensorsCheckpoint) -> dict[str, TensorSpec]:
    names = base.names()
    if names != checkpoint.names():
        raise ValueError(
            f"{base.path} and {checkpoint.path} cover different tensors; "
            f"only in base: {sorted(set(names) - set(checkpoint.names()))[:5]}, "
            f"only in checkpoint: {sorted(set(checkpoint.names()) - set(names))[:5]}"
        )
    plan: dict[str, TensorSpec] = {}
    for name in names:
        component_of(name)
        base_spec, checkpoint_spec = base.spec(name), checkpoint.spec(name)
        if base_spec.shape != checkpoint_spec.shape:
            raise ValueError(f"{name!r}: base shape {base_spec.shape} != checkpoint shape {checkpoint_spec.shape}")
        if checkpoint_spec.dtype != _FP32:
            raise ValueError(f"{name!r} is {checkpoint_spec.dtype} in {checkpoint.path}; C8 needs fp32 masters")
        if base_spec.dtype not in (_FP32, _BF16):
            raise ValueError(f"{name!r} is {base_spec.dtype} in {base.path}; B must be bf16 or its fp32 upcast")
        plan[name] = TensorSpec(base_spec.shape, _FP32)
    return plan


def _base_tensor(base: SafetensorsCheckpoint, name: str) -> np.ndarray:
    stored = base.read(name)
    upcast = stored.astype(np.float32)
    if stored.dtype == _FP32 and not np.array_equal(upcast, upcast.astype(_BF16).astype(np.float32)):
        raise ValueError(f"{name!r} in {base.path} is not the fp32 upcast of a bf16 tensor (C8)")
    return upcast


def graft_checkpoint(
    base_path: str,
    checkpoint_path: str,
    components: frozenset[Component],
    output_dir: str,
    *,
    serving_files_dir: str,
    shard_size_bytes: int,
) -> dict:
    """Write ``B + Δ·1[components]`` to ``output_dir``; returns the index json.

    ``serving_files_dir`` is B's HF release directory, the source of
    :data:`~checkpoint_io.REQUIRED_SERVING_FILES`. Every tensor name must belong
    to a component group and both checkpoints must hold the same names and
    shapes; anything else raises before a byte is written.
    """
    if not components:
        raise ValueError("an empty component set grafts nothing; the hybrid would be B")
    base = SafetensorsCheckpoint(base_path)
    checkpoint = SafetensorsCheckpoint(checkpoint_path)
    plan = _validated_plan(base, checkpoint)

    def tensors() -> Iterator[tuple[str, np.ndarray]]:
        for name in plan:
            if component_of(name) in components:
                yield name, checkpoint.read(name)
            else:
                yield name, _base_tensor(base, name)

    index = write_sharded_checkpoint(tensors(), output_dir, plan=plan, shard_size_bytes=shard_size_bytes)
    copy_serving_files(serving_files_dir, output_dir)
    return index
