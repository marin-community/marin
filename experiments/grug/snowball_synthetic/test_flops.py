# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Snowball synthetic harness and the June Snowball trainer must charge the same FLOPs per example, attention
included."""

import dataclasses

import pytest
from levanter.models.snowball import SnowballConfig, num_long_attention_layers

from experiments.grug.snowball_synthetic.train import TINY, snowball_attention_flops_per_token, snowball_flops_per_token
from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

SHAPE_FIELDS = (
    "vocab_size",
    "hidden_dim",
    "intermediate_dim",
    "shared_expert_intermediate_dim",
    "num_experts",
    "num_experts_per_token",
    "num_layers",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "sliding_window",
)


@pytest.mark.parametrize(
    ("snowball", "seq_len", "long_layers"), [(SnowballConfig(), 4096, 7), (TINY, 32, 2)], ids=["full", "tiny"]
)
def test_harness_flops_match_the_june_trainer(snowball, seq_len, long_layers):
    june = GrugModelConfig(**{name: getattr(snowball, name) for name in SHAPE_FIELDS}, max_seq_len=seq_len)
    june_flops_per_example, summary = june_train._compute_flops(model_config=june)
    harness_flops_per_example = (
        3 * snowball_flops_per_token(snowball, snowball.vocab_size, june.max_seq_len) * june.max_seq_len
    )
    assert harness_flops_per_example == june_flops_per_example
    assert num_long_attention_layers(snowball.num_layers) == long_layers
    assert summary["throughput/num_full_attention_layers"] == long_layers
    attention = snowball_attention_flops_per_token(snowball, june.max_seq_len)
    assert 3 * attention.noncausal * june.max_seq_len == summary[june_train.ATTENTION_FLOPS_PER_EXAMPLE_KEY]
    assert 3 * attention.causal * june.max_seq_len == summary[june_train.CAUSAL_ATTENTION_FLOPS_PER_EXAMPLE_KEY]


def test_attention_flops_use_the_configured_head_width():
    """The tiny preset has 8 heads of 16 at hidden_dim 64: its attention maps are as wide as at hidden_dim 128."""
    assert TINY.hidden_dim // TINY.num_heads != TINY.head_dim
    same_heads = dataclasses.replace(TINY, hidden_dim=TINY.num_heads * TINY.head_dim, head_dim=None)
    assert snowball_attention_flops_per_token(TINY, 32) == snowball_attention_flops_per_token(same_heads, 32)
