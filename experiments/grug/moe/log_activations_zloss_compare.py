# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log activations for baseline vs no-router-zloss d512 across all checkpoints."""


from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

from experiments.grug.moe.log_activations import (
    LogActivationsSweepConfig,
    run_sweep,
)

_BL_BASE = "gs://marin-us-east5/grug/baseline-d512-ckpt-e5d2da/checkpoints"
_NZ_BASE = "gs://marin-us-east5/grug/no-router-zloss-d512-ckpt-0e67a0/checkpoints"

_STEPS = [
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    500,
    750,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    6387,
]

_TEXT = "The quick brown fox jumps over the lazy dog."
_RESOURCES = ResourceConfig.with_tpu("v5p-8")

bl_sweep = ExecutorStep(
    name="grug/activations-baseline-d512-zloss-compare",
    fn=run_sweep,
    config=LogActivationsSweepConfig(
        checkpoint_base=versioned(_BL_BASE),
        output_base="grug/activations-zloss-compare/baseline-d512",
        hidden_dim=versioned(512),
        budget=versioned(2.19e17),
        steps=versioned(_STEPS),
        text=versioned(_TEXT),
        resources=versioned(_RESOURCES),
    ),
)

nz_sweep = ExecutorStep(
    name="grug/activations-no-router-zloss-d512-zloss-compare",
    fn=run_sweep,
    config=LogActivationsSweepConfig(
        checkpoint_base=versioned(_NZ_BASE),
        output_base="grug/activations-zloss-compare/no-router-zloss-d512",
        hidden_dim=versioned(512),
        budget=versioned(2.19e17),
        steps=versioned(_STEPS),
        text=versioned(_TEXT),
        resources=versioned(_RESOURCES),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[nz_sweep],
        description="Activation logging: no-router-zloss d512 across all checkpoints.",
    )
