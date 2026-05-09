# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p14 for both walks.

p14 = p13 (full MoE) trained out to the moe d768 compute-optimal step count
(10343 steps, 2.711B tokens). Submits both p14 ExecutorSteps under a single
iris job (muon p14, then adamh p14).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p14 import nano_adamh_heuristic_p14_trial
from experiments.grug.nano.launch_muon_tuned_walk_p14 import nano_muon_tuned_p14_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p14_trial, nano_adamh_heuristic_p14_trial],
        description="Walk p14 (full MoE at moe d768 compute-optimal: b=64, steps=10343, tokens=2.71B).",
    )
