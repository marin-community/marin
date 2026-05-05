# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deep networks d768 on v5p-8.

GitHub issue: https://github.com/marin-community/marin/issues/5423
"""

from marin.execution.executor import executor_main

from experiments.grug.moe.deep_network_sweep import _make_steps

DEEP_CONFIGS_D768 = [
    (768, 12, 3.63e18, 128, 7357),
    (768, 17, 6.96e18, 128, 9958),
]

d768_steps = _make_steps(DEEP_CONFIGS_D768, tpu_type="v5p-8")

if __name__ == "__main__":
    executor_main(steps=d768_steps, description="Deep networks d768 (v5p-8).")
