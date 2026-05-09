# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p8 / p9 / p10 for both muon-tuned and adamh-heuristic.

Submits all 6 ExecutorSteps under a single iris job. They run sequentially
in the order listed (muon p8 -> muon p9 -> muon p10 -> adamh p8 -> adamh p9 ->
adamh p10). Total wall-clock ~5h on one v5p-8 (~50min/run).

Each step's run_id, ExecutorStep config, and optimizer is unchanged from its
individual launch file -- this file just batches them onto one reservation.
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p8 import nano_adamh_heuristic_p8_trial
from experiments.grug.nano.launch_adamh_heuristic_walk_p9 import nano_adamh_heuristic_p9_trial
from experiments.grug.nano.launch_adamh_heuristic_walk_p10 import nano_adamh_heuristic_p10_trial
from experiments.grug.nano.launch_muon_tuned_walk_p8 import nano_muon_tuned_p8_trial
from experiments.grug.nano.launch_muon_tuned_walk_p9 import nano_muon_tuned_p9_trial
from experiments.grug.nano.launch_muon_tuned_walk_p10 import nano_muon_tuned_p10_trial

if __name__ == "__main__":
    executor_main(
        steps=[
            nano_muon_tuned_p8_trial,
            nano_muon_tuned_p9_trial,
            nano_muon_tuned_p10_trial,
            nano_adamh_heuristic_p8_trial,
            nano_adamh_heuristic_p9_trial,
            nano_adamh_heuristic_p10_trial,
        ],
        description="Walks p8 (XSA), p9 (GQA), p10 (attn_scale=1/sqrt(h) + qk_mult=1.3) for muon and adamh.",
    )
