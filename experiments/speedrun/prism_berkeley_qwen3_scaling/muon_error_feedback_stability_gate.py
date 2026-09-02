# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one 100-step 300M Hesscorr cell through the first corrected update."""

from __future__ import annotations

import dataclasses

from fray.cluster import ResourceConfig
from marin.execution.step_runner import StepRunner

from experiments.speedrun.prism_berkeley_qwen3_scaling.muon_error_feedback_sweep import (
    FeedbackVariant,
    build_sweep_configs,
)
from experiments.speedrun.prism_berkeley_qwen3_scaling.submission_support import SpeedrunConfig, default_speedrun

GATE_LEARNING_RATE = 0.004
GATE_STEPS = 100
GATE_TPU_VARIANTS = ("v4-8", "v5p-8")
GATE_VERSION = "2026.08.29.1"


def build_gate_config() -> tuple[str, SpeedrunConfig]:
    """Build a 100-step gate that can use either supported four-chip TPU slice."""
    selected = build_sweep_configs(
        size="300m",
        learning_rates=(GATE_LEARNING_RATE,),
        variants=(FeedbackVariant("hesscorr", 0.1),),
    )
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one stability-gate cell, got {len(selected)}.")

    name, config = selected[0]
    archived_resources = config.train_config.resources
    resources = ResourceConfig.with_tpu(
        GATE_TPU_VARIANTS,
        cpu=archived_resources.cpu,
        ram=archived_resources.ram,
        disk=archived_resources.disk,
        preemptible=archived_resources.preemptible,
    )
    train_config = dataclasses.replace(
        config.train_config,
        num_train_steps=GATE_STEPS,
        resources=resources,
    )
    config = dataclasses.replace(config, train_config=train_config)
    return name, config


def main() -> None:
    name, config = build_gate_config()
    config.print_run_info()
    train_step, _ = default_speedrun(
        f"{name}_stability_gate",
        config,
        tags=["error-aware-muon", "300m-stability-gate"],
        version=GATE_VERSION,
    )
    StepRunner().run([train_step.lower()], max_concurrent=1)


if __name__ == "__main__":
    main()
