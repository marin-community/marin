# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""June TPU 67B cooldown using the length-partitioned Datakit mixture."""

import dataclasses

from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import _phase_weights
from experiments.june_tpu_67b_a2b.moe.long_context_datakit_moe_mix import (
    LONG_CONTEXT_SKEW,
    long_context_datakit_components,
    long_context_phase_weights,
)
from experiments.june_tpu_67b_a2b.moe.moe_67b_a2b_d2560_cooldown_step39k_seq64k_bs1024_rep8_muon_10T import (
    build as base_build,
)

_RUN_SUFFIX = "long_context"


def build(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    name = f"grug/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k_{_RUN_SUFFIX}"
    version = resolve_version(name, version)
    base = base_build(version=version)

    def build_config(ctx: StepContext):
        config = base.build_config(ctx)
        original_train_components = _phase_weights(1)
        validation_components = {
            name: component for name, component in config.data.components.items() if name not in original_train_components
        }
        validation_weights = {name: 0.0 for name in validation_components}
        data = dataclasses.replace(
            config.data,
            components={**long_context_datakit_components(), **validation_components},
            train_weights=[(0, {**long_context_phase_weights(), **validation_weights})],
        )
        return dataclasses.replace(
            config,
            data=data,
            run_id=f"{config.run_id}_{_RUN_SUFFIX}_skew{LONG_CONTEXT_SKEW:g}",
        )

    return dataclasses.replace(
        base,
        name=user_namespaced_name(name, version),
        build_config=build_config,
    )


if __name__ == "__main__":
    experiment_main(build)()
