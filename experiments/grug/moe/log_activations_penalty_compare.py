# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log activations for router penalty variants across all checkpoints."""

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

from experiments.grug.moe.log_activations import (
    LogActivationsSweepConfig,
    run_sweep,
)

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

_VARIANTS = [
    ("zloss-warmdown", "gs://marin-us-east5/grug/zloss-warmdown-d512-ckpt-5b2e69/checkpoints"),
    ("router-l2-v1", "gs://marin-us-east5/grug/router-l2-d512-ckpt-24d527/checkpoints"),
    ("router-l2-v2", "gs://marin-us-east5/grug/router-l2-v2-d512-ckpt-3305e4/checkpoints"),
]

all_steps = []
for label, ckpt_base in _VARIANTS:
    all_steps.append(
        ExecutorStep(
            name=f"grug/activations-{label}-d512",
            fn=run_sweep,
            config=LogActivationsSweepConfig(
                checkpoint_base=versioned(ckpt_base),
                output_base=f"grug/activations-penalty-compare/{label}",
                hidden_dim=versioned(512),
                budget=versioned(2.19e17),
                steps=versioned(_STEPS),
                text=versioned(_TEXT),
                resources=versioned(_RESOURCES),
            ),
        )
    )

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Activation logging: router penalty variants across all checkpoints.",
    )
