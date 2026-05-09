# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p12 (halved batch / doubled steps, no MoE) for both walks.

Submits both p12 ExecutorSteps under a single iris job (muon p12, then adamh
p12). The schedule change (b=64, s=6700 keeping total_tokens fixed) is the
isolated ablation here; p13 layers MoE on top of this same schedule.
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p12 import nano_adamh_heuristic_p12_trial
from experiments.grug.nano.launch_muon_tuned_walk_p12 import nano_muon_tuned_p12_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p12_trial, nano_adamh_heuristic_p12_trial],
        description="Walk p12 (batch halved, steps doubled, no MoE) for muon and adamh.",
    )
