# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.grug.moe.check_jaxpp_group2_moe_boundary_parity import (
    EXPERT_AXIS_SIZE,
    GLOBAL_MICROBATCH_SIZE,
    LOCAL_MICROBATCH_SIZE,
    host_microbatches,
    target_model_config,
    validate_kernel_environment,
)


def test_group2_gpu_gate_uses_target_model_and_microbatch() -> None:
    config = target_model_config()

    assert config.hidden_dim == 2560
    assert config.intermediate_dim == 1280
    assert config.shared_expert_intermediate_dim == 2560
    assert config.num_layers == 2
    assert config.num_experts == 64
    assert config.num_experts_per_token == 4
    assert config.num_heads == 20
    assert config.num_kv_heads == 5
    assert config.max_seq_len == 4096
    assert config.vocab_size == 8192
    assert config.attention_implementation == "gpu_fa4_cute"
    assert config.moe_implementation == "ring"
    assert config.loss_implementation == "xla"
    assert config.remat_mode == "save_moe"
    assert GLOBAL_MICROBATCH_SIZE == EXPERT_AXIS_SIZE * LOCAL_MICROBATCH_SIZE


def test_group2_gpu_gate_uses_unequal_nonzero_loss_denominators() -> None:
    microbatches = host_microbatches()
    denominators = [float(loss_weight.sum()) for _, loss_weight in microbatches]

    assert denominators == [131040.0, 87360.0]
    assert denominators[0] != denominators[1]
    for tokens, loss_weight in microbatches:
        assert tokens.shape == (32, 4096)
        assert loss_weight.shape == tokens.shape
        assert np.all(loss_weight[:, -1] == 0)


def test_group2_gpu_gate_requires_matched_pallas_triton_geometry() -> None:
    environment = {
        "RAGGED_DOT_IMPL": "triton",
        "HALIAX_RAGGED_DOT_TRITON_BLOCK_K": "32",
        "HALIAX_RAGGED_DOT_TRITON_NUM_WARPS": "8",
    }

    validate_kernel_environment(environment)
    with pytest.raises(ValueError, match="target kernel environment"):
        validate_kernel_environment({**environment, "HALIAX_RAGGED_DOT_TRITON_BLOCK_K": "64"})
