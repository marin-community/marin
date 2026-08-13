# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Red behavior contracts for the unbuilt ABI 9 representative Host slice."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import shuttle_jaxlib_target1_acceptance
from abi9_large_forward_acceptance_contract import (
    BOUNDARY,
    EXPECTED_INPUT_DIGESTS,
    EXPECTED_JAX_OUTPUT_DIGEST,
    EXPECTED_SERIAL_MISMATCHES,
    EXPECTED_TEMPORARY_BYTES,
    PIPELINE_ABI_VERSION,
    POLICY,
    SHAPE,
    WORKERS,
    balanced_adjacent_forward,
    output_identity,
    subject_options,
    validate_cache_transition,
)
from shuttle_jaxlib_target1_acceptance import boundary_function, fixed_inputs


def test_large_forward_subject_is_one_exact_abi9_cell() -> None:
    assert PIPELINE_ABI_VERSION == 9
    assert (SHAPE.rows, SHAPE.features, BOUNDARY, POLICY.value) == (2048, 4096, "forward", "source_ordered")
    assert WORKERS == ("baseline", "populate", "reuse")
    payload = json.loads(subject_options()["xla_shuttle_options"])
    assert payload["pipeline_abi_version"] == 9
    assert payload["execution_mode"] == "cpu_executable_bundle"
    assert payload["numerics"] == "source_ordered"


def test_balanced_adjacent_fold_preserves_frozen_ordinary_jax_bits() -> None:
    arguments = fixed_inputs(SHAPE, BOUNDARY)
    assert tuple(hashlib.sha256(value.tobytes(order="C")).hexdigest() for value in arguments) == EXPECTED_INPUT_DIGESTS
    assert Path(shuttle_jaxlib_target1_acceptance.__file__).resolve() == (
        Path(__file__).resolve().parent / "shuttle_jaxlib_target1_acceptance.py"
    )
    ordinary_jax = np.asarray(jax.jit(boundary_function(BOUNDARY))(*arguments))
    balanced = balanced_adjacent_forward()
    assert np.array_equal(balanced.view(np.uint16), ordinary_jax.view(np.uint16))
    assert output_identity(balanced) == (
        "b3cbbf50c3b6f6025dbaf3840c0c1a606b8e61e492d139b6c80217c1faae4226",
        EXPECTED_JAX_OUTPUT_DIGEST,
    )


def test_serial_left_fold_is_not_the_abi9_numerical_contract() -> None:
    x, gamma = fixed_inputs(SHAPE, BOUNDARY)
    total = np.zeros(SHAPE.rows, dtype=np.float32)
    for feature in range(SHAPE.features):
        leaf = np.square(x[:, feature].astype(np.float32), dtype=np.float32)
        total = np.add(total, leaf, dtype=np.float32)
    inverse = np.divide(
        np.float32(1.0),
        np.sqrt(total / np.float32(SHAPE.features) + np.float32(9.99999974e-6), dtype=np.float32),
        dtype=np.float32,
    )
    serial = (x.astype(np.float32) * inverse[:, None] * gamma.astype(np.float32)).astype("bfloat16")
    balanced = balanced_adjacent_forward()
    assert np.count_nonzero(serial.view(np.uint16) != balanced.view(np.uint16)) == EXPECTED_SERIAL_MISMATCHES


def test_no_reuse_census_is_exactly_192_086_mib() -> None:
    assert EXPECTED_TEMPORARY_BYTES == 201_416_716
    assert EXPECTED_TEMPORARY_BYTES / 1024**2 == pytest.approx(192.0859489440918)


def test_cache_contract_is_one_miss_then_one_immutable_hit() -> None:
    populated = {"jit_forward-" + "a" * 64 + "-cache": (4096, "b" * 64)}
    key = validate_cache_transition("populate", {}, populated, 0, 0, None)
    assert validate_cache_transition("reuse", populated, populated, 0, 1, key) == key
    with pytest.raises(AssertionError, match="exact populated cache entry"):
        validate_cache_transition("reuse", populated, {key: (4096, "c" * 64)}, 0, 1, key)
