# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import jax.numpy as jnp
import pytest

from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.jax_normalized_exp_contract_reverse_ffi import call_cuda_normalized_exp_contract_reverse_ffi
from tile_lifetime.xla_normalized_exp_contract_reverse import plan_normalized_exp_contract_reverse_hlo_replacement

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)


def _generated():
    hlo = gzip.decompress(_HLO.read_bytes()).decode()
    plan = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    return generate_cuda_normalized_exp_contract_reverse_ffi(
        plan,
        target="shuttle.generic.normalized_exp_contract_reverse.validation",
    )


def _inputs():
    return {
        "lhs": jnp.zeros((8, 32), dtype=jnp.bfloat16),
        "rhs": jnp.zeros((32, 128), dtype=jnp.bfloat16),
        "saved_state": jnp.zeros((8,), dtype=jnp.float32),
        "fold_validity": jnp.ones((128,), dtype=jnp.bool_),
        "row_cotangent": jnp.ones((8,), dtype=jnp.float32),
        "selected_indices": jnp.zeros((8,), dtype=jnp.int32),
        "row_validity": jnp.ones((8,), dtype=jnp.bool_),
    }


def test_jax_normalized_exp_reverse_call_rejects_wrong_shape_before_dispatch() -> None:
    inputs = _inputs()
    inputs["lhs"] = jnp.zeros((7, 32), dtype=jnp.bfloat16)

    with pytest.raises(ValueError, match=r"lhs.*shape"):
        call_cuda_normalized_exp_contract_reverse_ffi(_generated(), inputs)


def test_jax_normalized_exp_reverse_call_rejects_wrong_dtype_before_dispatch() -> None:
    inputs = _inputs()
    inputs["selected_indices"] = jnp.zeros((8,), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"selected_indices.*dtype"):
        call_cuda_normalized_exp_contract_reverse_ffi(_generated(), inputs)
