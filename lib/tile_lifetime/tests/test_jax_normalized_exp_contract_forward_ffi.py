# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import jax.numpy as jnp
import pytest

from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.jax_normalized_exp_contract_forward_ffi import call_cuda_normalized_exp_contract_forward_ffi
from tile_lifetime.xla_normalized_exp_contract_forward import plan_normalized_exp_contract_forward_hlo_replacement

_HLO = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "original-gpu-pre-scheduler-hlo.txt.gz"
)


def _generated():
    plan = plan_normalized_exp_contract_forward_hlo_replacement(gzip.decompress(_HLO.read_bytes()).decode())
    return generate_cuda_normalized_exp_contract_forward_ffi(
        plan,
        target="shuttle.generic.normalized_exp_contract_forward.validation",
    )


def _inputs():
    return {
        "lhs": jnp.zeros((8, 32), dtype=jnp.bfloat16),
        "rhs": jnp.zeros((32, 128), dtype=jnp.bfloat16),
        "fold_validity": jnp.ones((128,), dtype=jnp.bool_),
        "selected_indices": jnp.zeros((8,), dtype=jnp.int32),
    }


def test_jax_normalized_exp_forward_call_rejects_wrong_shape_before_dispatch() -> None:
    inputs = _inputs()
    inputs["fold_validity"] = jnp.ones((127,), dtype=jnp.bool_)

    with pytest.raises(ValueError, match=r"fold_validity.*shape"):
        call_cuda_normalized_exp_contract_forward_ffi(_generated(), inputs)


def test_jax_normalized_exp_forward_call_rejects_wrong_dtype_before_dispatch() -> None:
    inputs = _inputs()
    inputs["selected_indices"] = jnp.zeros((8,), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"selected_indices.*dtype"):
        call_cuda_normalized_exp_contract_forward_ffi(_generated(), inputs)
