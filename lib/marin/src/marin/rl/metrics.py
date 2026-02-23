# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL evaluation metrics shared across environments."""

import math


def pass_at_k_estimator(correct_list: list[bool], k: int) -> float:
    """Compute the standard combinatorial pass@k estimator (DeepMath-style)."""
    assert k > 0, "k must be greater than 0"
    assert k <= len(correct_list), "k must be less than or equal to the length of correct_list"

    num_samples = len(correct_list)
    num_correct = sum(correct_list)
    if num_correct == 0:
        return 0.0
    if (num_samples - num_correct) < k:
        return 1.0

    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(num_samples - num_correct - i) - math.log(num_samples - i)
    return 1.0 - math.exp(log_ratio)
