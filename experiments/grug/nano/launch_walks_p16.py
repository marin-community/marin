# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p16 for both walks.

p16 = p14's compute-optimal MoE + grug/moe data path: nemotron_mix tokenized
with llama3 (vocab 128_256), fused softmax+CE kernel, and levanter's native
intra-doc segment masking. Submits both p16 ExecutorSteps under a single
iris job (muon p16, then adamh p16).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p16 import nano_adamh_heuristic_p16_trial
from experiments.grug.nano.launch_muon_tuned_walk_p16 import nano_muon_tuned_p16_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p16_trial, nano_adamh_heuristic_p16_trial],
        description="Walk p16 (nemotron_mix + llama3 vocab + fused CE + levanter intra-doc mask).",
    )
