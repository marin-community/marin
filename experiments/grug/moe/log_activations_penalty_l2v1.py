# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Activation logging for router-l2-v1 variant."""

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

from experiments.grug.moe.log_activations import LogActivationsSweepConfig, run_sweep
from experiments.grug.moe.log_activations_penalty_compare import _STEPS, _TEXT

step = ExecutorStep(
    name="grug/activations-router-l2-v1-d512-v2",
    fn=run_sweep,
    config=LogActivationsSweepConfig(
        checkpoint_base=versioned("gs://marin-us-east5/grug/router-l2-d512-ckpt-24d527/checkpoints"),
        output_base="grug/activations-penalty-compare-v2/router-l2-v1",
        hidden_dim=versioned(512),
        budget=versioned(2.19e17),
        steps=versioned(_STEPS),
        text=versioned(_TEXT),
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[step], description="Activation logging: router-l2-v1.")
