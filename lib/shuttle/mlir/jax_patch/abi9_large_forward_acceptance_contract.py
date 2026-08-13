# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed ordinary-JAX contract for the unbuilt ABI 9 Host slice."""

import hashlib
import json
from collections.abc import Mapping

import numpy as np
from shuttle_jaxlib_target1_acceptance import Shape, acceptance_tuning, fixed_inputs, output_digest

from shuttle import ExecutionMode, Numerics, compiler_options

PIPELINE_ABI_VERSION = 9
SHAPE = Shape(2048, 4096, "44d152ecc3e9ff18")
BOUNDARY = "forward"
POLICY = Numerics.SOURCE_ORDERED
WORKERS = ("baseline", "populate", "reuse")
EXPECTED_INPUT_DIGESTS = (
    "144528f4214fdd038bd51c5bcb6c653fcef1ab067cad0c6ae45924ff2949772a",
    "e90137a7f830b75f00ee6dae7c89bbd7fba30dfb31ff7951fec129441c170821",
)
EXPECTED_JAX_OUTPUT_DIGEST = "b9665bb7204c79fbedaa49e07062ed6a65436face5961f198d6223ba51828f6f"
EXPECTED_SERIAL_MISMATCHES = 9884
EXPECTED_TEMPORARY_BYTES = 201_416_716


def subject_options() -> dict[str, object]:
    """Return the one canonical ABI 9 subject option payload."""
    options = compiler_options(
        execution_mode=ExecutionMode.CPU_EXECUTABLE_BUNDLE,
        numerics=POLICY,
        tuning=acceptance_tuning(),
    )
    encoded = options["xla_shuttle_options"]
    if not isinstance(encoded, str):
        raise TypeError("xla_shuttle_options must be canonical JSON text")
    payload = json.loads(encoded)
    if payload.get("pipeline_abi_version") != PIPELINE_ABI_VERSION:
        raise AssertionError("representative-shape Host proof requires pipeline ABI 9")
    if payload.get("execution_mode") != "cpu_executable_bundle":
        raise AssertionError("representative-shape Host proof lost its execution mode")
    if payload.get("numerics") != POLICY.value:
        raise AssertionError("representative-shape Host proof requires source_ordered")
    return options


def balanced_adjacent_forward() -> np.ndarray:
    """Evaluate the exact cpu_bytecode_v2 Fold realization in binary32."""
    x, gamma = fixed_inputs(SHAPE, BOUNDARY)
    leaves = np.square(x.astype(np.float32), dtype=np.float32)
    while leaves.shape[1] > 1:
        if leaves.shape[1] % 2:
            paired = np.add(leaves[:, 0:-1:2], leaves[:, 1::2], dtype=np.float32)
            leaves = np.concatenate((paired, leaves[:, -1:]), axis=1)
        else:
            leaves = np.add(leaves[:, 0::2], leaves[:, 1::2], dtype=np.float32)
    # This is a chosen order_free realization. It is not a blanket claim that
    # initializer placement is fixed by StableHLO source semantics.
    total = np.add(leaves[:, 0], np.float32(0.0), dtype=np.float32)
    mean = np.divide(total, np.float32(SHAPE.features), dtype=np.float32)
    inverse = np.divide(
        np.float32(1.0),
        np.sqrt(np.add(mean, np.float32(9.99999974e-6), dtype=np.float32), dtype=np.float32),
        dtype=np.float32,
    )
    return np.multiply(
        np.multiply(x.astype(np.float32), inverse[:, None], dtype=np.float32),
        gamma.astype(np.float32),
        dtype=np.float32,
    ).astype("bfloat16")


def validate_cache_transition(
    worker: str,
    before: Mapping[str, tuple[int, str]],
    after: Mapping[str, tuple[int, str]],
    hits_before: int,
    hits_after: int,
    expected_key: str | None,
) -> str:
    """Require one ABI 9 miss followed by an immutable cross-process hit."""
    if worker == "populate":
        added = set(after) - set(before)
        if before or len(added) != 1 or hits_after != hits_before:
            raise AssertionError("ABI 9 populate must create exactly one cache entry on a miss")
        return added.pop()
    if worker == "reuse":
        if expected_key is None or set(after) != {expected_key} or before != after:
            raise AssertionError("ABI 9 reuse must preserve the exact populated cache entry")
        if hits_after != hits_before + 1:
            raise AssertionError("ABI 9 reuse must report exactly one public cache hit")
        return expected_key
    raise ValueError(f"unsupported subject worker: {worker!r}")


def output_identity(value: np.ndarray) -> tuple[str, str]:
    """Return the raw and contract digests of the one BF16 result."""
    return hashlib.sha256(value.tobytes(order="C")).hexdigest(), output_digest((value,))
