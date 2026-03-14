# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
https://github.com/marin-community/marin/issues/977

Codename: sensible-starling

This experiment is a cooldown run for the tootsie-8b model starting from adept-phoenix. It is trained on the
same mix as exp934_hq_vs_pt's best mix's full mix

We also add z-loss, since in spoonbill we found that to be very helpful
"""

from experiments.tootsie.exp600_tootsie import tootsie_8b_sensible_starling
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        [tootsie_8b_sensible_starling],
        description="Train Tootsie 8b with cooldown from 1.7e-4 to 1.7e-5 over 125B tokens",
    )
