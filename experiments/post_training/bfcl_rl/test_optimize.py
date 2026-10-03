# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest

from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.post_training.bfcl_rl.optimize import RecoveryOptimization, recovery_optimizer_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache


def test_recovery_mesh_fits_eight_gpu_nodes_and_preserves_batch_parallelism():
    # The live 64-GPU recovery run failed because preflight assumed one slice.
    cache = ArtifactStep.adopt("preferences", "2026.10.03.16", "preferences", kind=RecoveryPreferenceCache)
    with build_context(BuildContext(VersionCodex("2026.10.03.18"))):
        step = recovery_optimizer_step(
            cache, selection_name="full", optimization=RecoveryOptimization(1, 16, 0.1, 8, 8, 4)
        )
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    mesh = config.train_config.trainer.mesh
    ici, dcn = mesh.axis_shapes(config.resources.chip_count(), config.resources.replicas)
    assert prod(ici.values()) == 8
    assert prod(dcn.values()) == 8
    assert ici["expert"] == 8
    assert dcn["context"] == 4
    batch_axes = mesh.resolved_compute_mapping["batch"]
    width = prod(ici.get(axis, 1) * dcn.get(axis, 1) for axis in batch_axes)
    assert width == config.train_config.trainer.train_batch_size == 16

    with build_context(BuildContext(VersionCodex("2026.10.03.18"))):
        with pytest.raises(ValueError, match="ICI product"):
            recovery_optimizer_step(
                cache, selection_name="full", optimization=RecoveryOptimization(1, 16, 0.1, 8, 16, 4)
            )
