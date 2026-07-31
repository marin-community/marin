# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed model and optimizer recipes for the coupon-clipping pyramid arms."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from levanter.grug.attention import GrugAttentionImplementation
from levanter.grug.grug_moe import resolve_moe_implementation
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.coupon_clipping.model import GrugModelConfig
from experiments.grug.depth_growth import DepthGrowthConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.llama import llama3_tokenizer_vocab_size

HIDDEN_DIM = 3072
NUM_LAYERS = 48
NUM_EXPERTS = 64
NUM_EXPERTS_PER_TOKEN = 4
ROUTED_INTERMEDIATE_DIM = 1536
AVERAGE_SHARED_INTERMEDIATE_DIM = 1280
NUM_HEADS = 24
NUM_KV_HEADS = 6
HEAD_DIM = 128
SEQUENCE_LENGTH = 4096
SLIDING_WINDOW = 2048

SEGMENT_LENGTHS = (4, 18, 4, 22)
UNIFORM_SHARED_WIDTHS = (1280, 1280, 1280, 1280)
FAT_FIRST_SHARED_WIDTHS = (4096, 1024, 1024, 1024)
FAT_MIDDLE_SHARED_WIDTHS = (1024, 1024, 4096, 1024)

TRAIN_BATCH_SIZE = 256
TRAIN_STEPS = 6400
TRAIN_TOKENS = TRAIN_BATCH_SIZE * SEQUENCE_LENGTH * TRAIN_STEPS
EXPECTED_TRAIN_TOKENS = 6_710_886_400
DEPTH_SOURCE_LAYERS = 1
DEPTH_TRANSITION_STEP = 4480
DEPTH_TRANSITION_DATA_OFFSET = DEPTH_TRANSITION_STEP * TRAIN_BATCH_SIZE

MUONH_LEARNING_RATE = 0.006423539
ADAM_LEARNING_RATE = 0.001482355
LOW_MUONH_LEARNING_RATE = 0.005768679
LOW_ADAM_LEARNING_RATE = 0.001331234
HIGH_MUONH_LEARNING_RATE = 0.007210848
HIGH_ADAM_LEARNING_RATE = 0.001664041
WARMUP_STEPS = 64
DECAY_STEPS = 640

_GATED_NORM_RANK = 128


class CouponClippingArm(StrEnum):
    C0_P0 = "cc16-c0-p0"
    P1 = "cc16-p1-fat-first"
    P2 = "cc16-p2-fat-middle"


class CouponClippingLearningRate(StrEnum):
    LOW = "lr0500"
    CENTER = "lr0557"
    HIGH = "lr0625"


_SHARED_WIDTHS_BY_ARM = {
    CouponClippingArm.C0_P0: UNIFORM_SHARED_WIDTHS,
    CouponClippingArm.P1: FAT_FIRST_SHARED_WIDTHS,
    CouponClippingArm.P2: FAT_MIDDLE_SHARED_WIDTHS,
}

_LEARNING_RATES = {
    CouponClippingLearningRate.LOW: (LOW_MUONH_LEARNING_RATE, LOW_ADAM_LEARNING_RATE),
    CouponClippingLearningRate.CENTER: (MUONH_LEARNING_RATE, ADAM_LEARNING_RATE),
    CouponClippingLearningRate.HIGH: (HIGH_MUONH_LEARNING_RATE, HIGH_ADAM_LEARNING_RATE),
}


@dataclass(frozen=True)
class ModelAccounting:
    stored_parameters: int
    active_parameters: int
    forward_flops_per_token: float


def build_model_config(arm: CouponClippingArm) -> GrugModelConfig:
    """Build an arm with fixed four-segment scan topology and arm-specific shared widths."""
    return GrugModelConfig(
        vocab_size=llama3_tokenizer_vocab_size,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=ROUTED_INTERMEDIATE_DIM,
        shared_expert_intermediate_dim=AVERAGE_SHARED_INTERMEDIATE_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=SEQUENCE_LENGTH,
        sliding_window=SLIDING_WINDOW,
        initializer_std=0.5 / math.sqrt(HIDDEN_DIM),
        qk_mult=1.3,
        attention_implementation=cast(GrugAttentionImplementation, "gpu_fa4_cute"),
        moe_implementation=resolve_moe_implementation("sonic_cute"),
        remat_mode="recompute_all",
        use_array_stacked_blocks=True,
        block_segment_lengths=SEGMENT_LENGTHS,
        block_segment_shared_expert_intermediate_dims=_SHARED_WIDTHS_BY_ARM[arm],
    )


def build_optimizer_config(
    learning_rate: CouponClippingLearningRate = CouponClippingLearningRate.CENTER,
) -> GrugMoeMuonHConfig:
    """Return the pre-registered 6.71B-token MuonH/Adam schedule."""
    muonh_learning_rate, adam_learning_rate = _LEARNING_RATES[learning_rate]
    return GrugMoeMuonHConfig(
        learning_rate=muonh_learning_rate,
        adam_lr=adam_learning_rate,
        beta1=0.9062,
        beta2=0.992027944,
        epsilon=7.7408e-16,
        lr_schedule="linear",
        warmup=WARMUP_STEPS,
        decay=DECAY_STEPS,
        min_lr_ratio=0.05,
        max_grad_norm=None,
    )


def model_accounting(config: GrugModelConfig) -> ModelAccounting:
    """Compute exact parameter counts and the standard analytic forward FLOP estimate."""
    hidden_dim = config.hidden_dim
    head_dim = config.inferred_head_dim

    embedding_parameters = 2 * config.vocab_size * hidden_dim
    endpoint_norm_parameters = 2 * hidden_dim + 4 * hidden_dim * _GATED_NORM_RANK

    attention_parameters_per_layer = (
        2 * hidden_dim * config.num_heads * head_dim
        + 2 * hidden_dim * config.num_kv_heads * head_dim
        + hidden_dim * config.num_heads
    )
    norm_parameters_per_layer = 2 * hidden_dim + 4 * hidden_dim * _GATED_NORM_RANK
    router_parameters_per_layer = hidden_dim * config.num_experts + config.num_experts
    routed_parameters_per_layer = 3 * config.num_experts * hidden_dim * config.intermediate_dim
    active_routed_parameters_per_layer = 3 * config.num_experts_per_token * hidden_dim * config.intermediate_dim
    shared_parameters = 3 * hidden_dim * sum(config.shared_expert_intermediate_dims_by_layer)

    common_layer_parameters = attention_parameters_per_layer + norm_parameters_per_layer + router_parameters_per_layer
    stored_parameters = (
        embedding_parameters
        + endpoint_norm_parameters
        + config.num_layers * (common_layer_parameters + routed_parameters_per_layer)
        + shared_parameters
    )
    active_parameters = (
        embedding_parameters
        + endpoint_norm_parameters
        + config.num_layers * (common_layer_parameters + active_routed_parameters_per_layer)
        + shared_parameters
    )
    forward_flops_per_token = lm_flops_per_token(
        hidden_dim=hidden_dim,
        intermediate_dim=config.intermediate_dim,
        shared_intermediate_dim=config.shared_expert_intermediate_dim,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        num_heads=config.num_heads,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        glu=True,
        num_experts=config.num_experts,
        num_shared_experts=1,
        num_experts_per_tok=config.num_experts_per_token,
    )
    return ModelAccounting(stored_parameters, active_parameters, forward_flops_per_token)


def assert_matched_pyramid_accounting() -> ModelAccounting:
    """Fail at config import if an arm no longer matches the control's size or FLOPs."""
    models = [build_model_config(arm) for arm in CouponClippingArm]
    if any(model.resolved_block_segment_lengths != SEGMENT_LENGTHS for model in models):
        raise AssertionError("all pyramid arms must use identical scan segment boundaries")
    accounting = [model_accounting(model) for model in models]
    if any(candidate != accounting[0] for candidate in accounting[1:]):
        raise AssertionError(f"pyramid parameter/FLOP mismatch: {accounting}")
    return accounting[0]


if TRAIN_TOKENS != EXPECTED_TRAIN_TOKENS:
    raise AssertionError(f"token horizon changed: got {TRAIN_TOKENS}, expected {EXPECTED_TRAIN_TOKENS}")

MATCHED_MODEL_ACCOUNTING = assert_matched_pyramid_accounting()
DEPTH_GROWTH_CONFIG = DepthGrowthConfig(
    source_layers=DEPTH_SOURCE_LAYERS,
    target_layers=NUM_LAYERS,
    expected_step=DEPTH_TRANSITION_STEP,
    expected_data_offset=DEPTH_TRANSITION_DATA_OFFSET,
)
