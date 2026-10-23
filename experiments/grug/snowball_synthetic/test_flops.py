# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Snowball synthetic harness and the June Snowball trainer must charge the same FLOPs per example."""

from levanter.models.snowball import SnowballConfig, num_long_attention_layers

from experiments.grug.snowball_synthetic.train import snowball_flops_per_token
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
    "sliding_window",
)


def test_harness_flops_match_the_june_trainer():
    snowball = SnowballConfig()
    june = GrugModelConfig(**{name: getattr(snowball, name) for name in SHAPE_FIELDS}, max_seq_len=4096)
    june_flops_per_example, summary = june_train._compute_flops(model_config=june)
    harness_flops_per_example = (
        3 * snowball_flops_per_token(snowball, snowball.vocab_size, june.max_seq_len) * june.max_seq_len
    )
    assert harness_flops_per_example == june_flops_per_example
    assert num_long_attention_layers(snowball.num_layers) == summary["throughput/num_full_attention_layers"] == 7
