# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the approved loop-guard stages against their regional task selections."""

import os

from experiments.post_training import rl, rl_compose
from experiments.post_training.mochi_loop_guards.presets import (
    S0,
    S0_TB2_RECOVERY,
    S1_RP,
    S1_STOP,
    S2_CONTROL,
    S2_TREATMENT,
)
from experiments.post_training.rl_presets import ModelPresets

REGIONAL_ROOT = "s3://marin-us-east-08a/marin"
GUARDS = ModelPresets(
    model="mochi",
    presets=(S0, S0_TB2_RECOVERY, S1_STOP, S1_RP, S2_CONTROL, S2_TREATMENT),
    namespace="mochi-loop-guards",
    control_pins={},
    needs_completed_default_export=False,
)


if __name__ == "__main__":
    # These experiment selections are published in the execution region, rather than
    # task curation's canonical 02a root. Every stage uses the same regional binding.
    rl_compose.SELECTION_ROOT = f"{REGIONAL_ROOT}/rl-data/task-selection"
    os.environ.setdefault("MARIN_PREFIX", REGIONAL_ROOT)
    os.environ.setdefault("MARIN_TEMP_PREFIX", "s3://marin-us-east-08a/tmp/ttl=3d")
    raise SystemExit(rl.main(GUARDS, forward_env=("MARIN_PREFIX", "MARIN_TEMP_PREFIX")))
