# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined launch: muon walks p16/p17/p18 with final-logit z_loss enabled.

Three runs (p16-muonz, p17-muonz, p18-muonz) under one iris job. Each is
identical to its base muon walk except ``z_loss_weight=1e-4``, which makes
the trainer pass ``logsumexp_weight=1e-4`` into ``next_token_loss``. The
fused kernel then adds ``1e-4 * logsumexp(logits)**2`` per position to the
cross-entropy.
"""

from marin.execution.executor import executor_main

from experiments.grug.nano.launch_muon_tuned_walk_p16_muonz import nano_muon_tuned_p16_muonz_trial
from experiments.grug.nano.launch_muon_tuned_walk_p17_muonz import nano_muon_tuned_p17_muonz_trial
from experiments.grug.nano.launch_muon_tuned_walk_p18_muonz import nano_muon_tuned_p18_muonz_trial

if __name__ == "__main__":
    executor_main(
        steps=[
            nano_muon_tuned_p16_muonz_trial,
            nano_muon_tuned_p17_muonz_trial,
            nano_muon_tuned_p18_muonz_trial,
        ],
        description="Muon p16/p17/p18 with final-logit z_loss=1e-4 (matches adamh recipe).",
    )
