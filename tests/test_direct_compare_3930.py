# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the direct comparison configs from issue #3930."""

import pytest

from experiments.grug.moe.direct_compare_3930 import (
    ES3R2_MODEL,
    Q30A3B_PROXY_MODEL,
    Q35A3B_FA_MODEL,
)


@pytest.mark.parametrize(
    "config,expected_hidden,expected_layers,expected_experts",
    [
        (ES3R2_MODEL, 4096, 27, 64),
        (Q30A3B_PROXY_MODEL, 2048, 48, 128),
        (Q35A3B_FA_MODEL, 2048, 40, 256),
    ],
    ids=["es3r2", "q30a3b-proxy", "q35a3b-fa"],
)
def test_config_dimensions(config, expected_hidden, expected_layers, expected_experts):
    assert config.hidden_dim == expected_hidden
    assert config.num_layers == expected_layers
    assert config.num_experts == expected_experts
    assert config.vocab_size == 128_256
    assert config.max_seq_len == 4096
    # Verify head_dim is consistent with num_heads.
    assert config.inferred_head_dim == config.head_dim


@pytest.mark.parametrize(
    "config",
    [ES3R2_MODEL, Q30A3B_PROXY_MODEL, Q35A3B_FA_MODEL],
    ids=["es3r2", "q30a3b-proxy", "q35a3b-fa"],
)
def test_gqa_ratio_is_valid(config):
    assert config.num_heads % config.num_kv_heads == 0
    assert config.num_experts_per_token <= config.num_experts
