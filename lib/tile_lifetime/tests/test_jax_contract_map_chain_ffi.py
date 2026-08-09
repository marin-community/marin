# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
from pathlib import Path

import jax.numpy as jnp
import pytest

from tile_lifetime.contract_map_chain import form_two_contract_map_training_program
from tile_lifetime.cuda_contract_map_chain_codegen import generate_cuda_contract_map_chain_ffi
from tile_lifetime.jax_contract_map_chain_ffi import (
    call_cuda_contract_map_chain_forward_ffi,
    call_cuda_contract_map_chain_reverse_ffi,
)
from tile_lifetime.xla_low_rank_gated_product import recover_low_rank_gated_product_training
from tile_lifetime.xla_normalized_exp_contract_forward import (
    plan_normalized_exp_contract_forward_hlo_replacement,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)

_ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_narrowed_unaccepted_da49b94c_v0/"
    "transformed-gpu-pre-scheduler-hlo.txt.gz"
)


def _generated():
    hlo = gzip.decompress(_ARTIFACT.read_bytes()).decode()
    forward_replacement = plan_normalized_exp_contract_forward_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo,
        forward_replacement,
        target="shuttle.test.normalized_exp.forward",
    )
    reverse_replacement = plan_normalized_exp_contract_reverse_hlo_replacement(hlo)
    hlo = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
        hlo,
        reverse_replacement,
        target="shuttle.test.normalized_exp.reverse",
    )
    report = recover_low_rank_gated_product_training(hlo)
    reverse = report.reverse_families[0]
    program = form_two_contract_map_training_program(reverse.primal, reverse)
    return generate_cuda_contract_map_chain_ffi(
        program,
        forward_target="shuttle.generic.contract_map_chain.forward.validation",
        reverse_target="shuttle.generic.contract_map_chain.reverse.validation",
    )


def _inputs():
    return (
        jnp.zeros((8, 32), dtype=jnp.bfloat16),
        jnp.zeros((32, 128), dtype=jnp.bfloat16),
        jnp.zeros((128, 32), dtype=jnp.bfloat16),
    )


def test_jax_contract_map_chain_forward_rejects_wrong_shape_before_dispatch() -> None:
    _, first_weight, second_weight = _inputs()

    with pytest.raises(ValueError, match=r"activation.*shape"):
        call_cuda_contract_map_chain_forward_ffi(
            _generated(),
            jnp.zeros((7, 32), dtype=jnp.bfloat16),
            first_weight,
            second_weight,
        )


def test_jax_contract_map_chain_reverse_rejects_non_bf16_save_before_dispatch() -> None:
    activation, first_weight, second_weight = _inputs()

    with pytest.raises(ValueError, match=r"hidden.*dtype"):
        call_cuda_contract_map_chain_reverse_ffi(
            _generated(),
            activation,
            first_weight,
            second_weight,
            jnp.zeros((8, 128), dtype=jnp.bfloat16),
            jnp.zeros((8, 128), dtype=jnp.float32),
            jnp.zeros((8, 32), dtype=jnp.bfloat16),
            jnp.zeros((8, 32), dtype=jnp.bfloat16),
        )
