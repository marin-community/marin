# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deep networks d768+d1024 on v5p-32.

GitHub issue: https://github.com/marin-community/marin/issues/5423
"""

from marin.execution.executor import executor_main

from experiments.grug.moe.deep_network_sweep import d768_d1024_steps

if __name__ == "__main__":
    executor_main(steps=d768_d1024_steps, description="Deep networks d768+d1024 (v5p-32).")
