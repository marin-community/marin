# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p15 for both walks.

p15 = p14 (full MoE at moe d768 compute-optimal) + intra-doc attention
masking. Submits both p15 ExecutorSteps under a single iris job (muon p15,
then adamh p15).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p15 import nano_adamh_heuristic_p15_trial
from experiments.grug.nano.launch_muon_tuned_walk_p15 import nano_muon_tuned_p15_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p15_trial, nano_adamh_heuristic_p15_trial],
        description="Walk p15 (intra-doc attention masking on p14's compute-optimal MoE) for muon and adamh.",
    )
