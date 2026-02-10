# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto-tuned block sizes for Fused MoE kernel."""

import logging

logger = logging.getLogger(__name__)


# TODO(jevinjiang): create sharable util
def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


# Key:
#   - hidden_size: int, padded hidden dimension size
#   - intermediate_size: int, padded intermediate dimension size
#   - num_experts: int, total number of experts
#   - top_k: int, selected number of top experts
#   - t_packing: int, token packing
#   - w_packing: int, weight packing
#   - num_tokens: int, total number of tokens
#   - ep_size: int, expert parallelism size
# Value:
#   - bt: int, block size of local_num_tokens
#   - bf: int, block size of intermediate_size
#   - bd1: int, block size of hidden_size in w1
#   - bd2: int, block size of hidden_size in w2
#   - btc: int, compute size of block tokens for active expert
#   - bfc: int, compute size of block intermediate_size
#   - bd1c: int, compute size of block hidden_size in w1
#   - bd2c: int, compute size of block hidden_size in w2
TUNED_BLOCK_SIZES = {
    (3072, 3072, 128, 4, 2, 2, 128, 4): (
        32,
        256 * 6,
        256 * 12,
        256 * 12,
        16,
        256 * 6,
        256 * 12,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 4): (
        64,
        256 * 6,
        256 * 6,
        256 * 12,
        16,
        256 * 6,
        256 * 6,
        256 * 12,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 4): (
        128,
        256 * 6,
        256 * 6,
        256 * 6,
        64,
        256 * 6,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 4): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 4): (
        128,
        256 * 3,
        256 * 6,
        256 * 6,
        64,
        256 * 3,
        256 * 6,
        256 * 6,
    ),
    (3072, 3072, 128, 4, 2, 2, 128, 8): (
        16,
        1536,
        1536,
        1536 * 2,
        16,
        1536,
        1536,
        1536,
    ),
    (3072, 3072, 128, 4, 2, 2, 256, 8): (
        32,
        1536,
        1536,
        1536 * 2,
        32,
        1536,
        1536,
        1536,
    ),
    (3072, 3072, 128, 4, 2, 2, 512, 8): (
        64,
        1536,
        1536,
        1536 * 2,
        64,
        1536,
        1536,
        1536,
    ),
    (3072, 3072, 128, 4, 2, 2, 1024, 8): (
        64,
        1536,
        1536,
        1536 * 2,
        64,
        1536,
        1536,
        1536,
    ),
    (3072, 3072, 128, 4, 2, 2, 2048, 8): (
        64,
        1536,
        1536,
        1536 * 2,
        64,
        1536,
        1536,
        1536,
    ),
    # for qwen3-coder-480b, we use a temporary huristic block size here. Need to tune for better performance.
    (6144, 2560, 160, 8, 2, 4, 16, 4): (
        256 * 1,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 16, 8): (
        32,
        256 * 10,
        256 * 6,
        256 * 12,
        16,
        256 * 10,
        256 * 6,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 4): (
        128,
        256 * 5,
        256 * 12,
        256 * 24,
        32,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 32, 8): (
        32,
        256 * 10,
        256 * 4,
        256 * 8,
        32,
        256 * 10,
        256 * 4,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 4): (
        64,
        256 * 10,
        256 * 12,
        256 * 12,
        32,
        256 * 10,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 64, 8): (
        256 * 1,
        256 * 5,
        256 * 24,
        256 * 24,
        16,
        256 * 5,
        256 * 24,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 4): (
        64,
        256 * 5,
        256 * 24,
        256 * 24,
        16,
        256 * 5,
        256 * 24,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 128, 8): (
        128,
        256 * 5,
        256 * 12,
        256 * 24,
        16,
        256 * 5,
        256 * 12,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 4): (
        256 * 1,
        256 * 10,
        256 * 8,
        256 * 8,
        32,
        256 * 10,
        256 * 8,
        256 * 8,
    ),
    (6144, 2560, 160, 8, 2, 4, 256, 8): (
        128,
        256 * 10,
        256 * 8,
        256 * 12,
        64,
        256 * 10,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 4): (
        256 * 1,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 512, 8): (
        256 * 1,
        256 * 2,
        256 * 24,
        256 * 24,
        64,
        256 * 2,
        256 * 24,
        256 * 24,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 1024, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 2048, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        64,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        64,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 4096, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 4): (
        128,
        256 * 2,
        256 * 12,
        256 * 12,
        32,
        256 * 2,
        256 * 12,
        256 * 12,
    ),
    (6144, 2560, 160, 8, 2, 4, 8192, 8): (
        64,
        256 * 5,
        256 * 8,
        256 * 12,
        32,
        256 * 5,
        256 * 8,
        256 * 12,
    ),
}


def get_default_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
):
    assert num_experts % ep_size == 0
    assert num_tokens % ep_size == 0
    assert 0 < top_k <= num_experts
    assert 0 < t_packing <= 8
    assert 0 < w_packing <= 8
    if hidden_size % 256:
        hidden_size = align_to(hidden_size, 256)
    if intermediate_size % 256:
        intermediate_size = align_to(intermediate_size, 256)
    # TODO(jevinjiang): the formula is only applied to tpu-v7 and we need to add
    # other formulas for other generations.
    local_num_tokens = num_tokens // ep_size

    d = hidden_size // 256
    f = intermediate_size // 256

    return (
        min(local_num_tokens, 128),
        min(256 * f // 2, 1024),
        min(256 * d // 2, 1024),
        min(256 * d // 2, 2048),
        min(local_num_tokens // 2, 64),
        min(256 * f // 2, 1024),
        min(256 * d // 2, 1024),
        min(256 * d // 2, 2048),
    )


def get_tuned_block_sizes(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    t_packing: int,
    w_packing: int,
    num_tokens: int,
    ep_size: int,
):
    assert num_experts % ep_size == 0
    assert num_tokens % ep_size == 0
    assert 0 < top_k <= num_experts
    assert 0 < t_packing <= 8
    assert 0 < w_packing <= 8
    # TODO(b/467431118): currently we just use manual tuned block sizes for some
    # specific models.
    hidden_size = align_to(hidden_size, 256)
    intermediate_size = align_to(intermediate_size, 256)

    key = (
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        t_packing,
        w_packing,
        num_tokens,
        ep_size,
    )

    if key not in TUNED_BLOCK_SIZES:
        logger.warning_once(
            f"[Fused MOE kernel] using default block sizes for key: {key}: {get_default_block_sizes(*key)}"
        )
    return TUNED_BLOCK_SIZES.get(key, get_default_block_sizes(*key))
