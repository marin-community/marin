# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: walk p18 for both walks.

p18 = p17 (no attn_gate / gated_norm) with the shared expert reshaped to
ReLU² @ 1.5·D (instead of SwiGLU @ 1·D). Isoflop, isoparam swap of the
shared dense expert. Routed experts continue to use SwiGLU @ D/2.
Submits both p18 ExecutorSteps under a single iris job (muon p18, then
adamh p18).
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_adamh_heuristic_walk_p18 import nano_adamh_heuristic_p18_trial
from experiments.grug.nano.launch_muon_tuned_walk_p18 import nano_muon_tuned_p18_trial

if __name__ == "__main__":
    executor_main(
        steps=[nano_muon_tuned_p18_trial, nano_adamh_heuristic_p18_trial],
        description="Walk p18 (shared expert -> ReLU² @ 1.5·D).",
    )
