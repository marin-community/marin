# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p11 (num_layers 12 -> 8) for both walks.

Submits both p11 ExecutorSteps under a single iris job (muon p11, then adamh
p11).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p11 import nano_adamh_heuristic_p11_trial
from experiments.grug.nano.launch_muon_tuned_walk_p11 import nano_muon_tuned_p11_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p11_trial, nano_adamh_heuristic_p11_trial],
        description="Walk p11 (num_layers=8) for muon and adamh.",
    )
