# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
