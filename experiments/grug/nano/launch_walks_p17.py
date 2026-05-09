# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p17 for both walks.

p17 = p16 (nemotron + llama3 + fused CE + intra-doc mask + full MoE) minus
the two AdamH-compensator features (attn_gate, gated_norm). Submits both
p17 ExecutorSteps under a single iris job (muon p17, then adamh p17).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p17 import nano_adamh_heuristic_p17_trial
from experiments.grug.nano.launch_muon_tuned_walk_p17 import nano_muon_tuned_p17_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p17_trial, nano_adamh_heuristic_p17_trial],
        description="Walk p17 (p16 minus attn_gate and gated_norm).",
    )
