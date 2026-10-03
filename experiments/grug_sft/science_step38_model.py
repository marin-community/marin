# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Step38 architecture shared by the science SFT trainer and exporter."""

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

MODEL_PATH = "s3://marin-us-east-02a/models/open-athena--Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38"
MODEL_REVISION = "cfc1d845dae89b067cdc7250d0164abefa5a69cf"
CONTEXT = 32_768


def science_model_config(*, trainable_router_bias: bool = False) -> GrugModelConfig:
    """Return the pinned Step38 architecture for training and export."""
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=2_560,
        intermediate_dim=1_280,
        shared_expert_intermediate_dim=2_560,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=26,
        num_heads=20,
        num_kv_heads=5,
        head_dim=None,
        max_seq_len=CONTEXT,
        sliding_window=2_048,
        layer_norm_eps=1e-5,
        initializer_std=0.009882117688026186,
        qk_mult=1.5703274004183787,
        qk_mult_long_scale=1.0,
        router_z_loss_coef=0.0,
        trainable_router_bias=trainable_router_bias,
        disable_pko=True,
        disable_long_rope=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation=None,
        capacity_factor=1.0,
        ce_implementation=None,
        remat_mode="recompute_all",
        replicate_attn_weights=False,
        split_w_gate_up=True,
        use_array_stacked_blocks=True,
    )
