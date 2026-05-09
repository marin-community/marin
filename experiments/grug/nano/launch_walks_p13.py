# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p13 (full MoE) for both walks.

Submits both p13 ExecutorSteps under a single iris job (muon p13, then adamh
p13).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p13 import nano_adamh_heuristic_p13_trial
from experiments.grug.nano.launch_muon_tuned_walk_p13 import nano_muon_tuned_p13_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p13_trial, nano_adamh_heuristic_p13_trial],
        description="Walk p13 (full MoE on top of p12's halved-batch / doubled-steps schedule).",
    )
