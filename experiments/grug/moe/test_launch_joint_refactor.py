# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import materialized_config
from rigging.filesystem import prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.launch_joint_refactor import build_joint_refactor_pipeline
from experiments.grug.moe.merge_recovery import (
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)

_PREFIX = "gs://marin-us-central1/test"


def test_joint_refactor_graph_has_no_matching_dependency_and_stays_in_central1(monkeypatch) -> None:
    monkeypatch.setattr("experiments.grug.moe.launch_joint_refactor.mixture", lambda *_args, **_kwargs: None)
    pipeline = build_joint_refactor_pipeline(version="dev")
    offline = materialized_config(pipeline.offline, _PREFIX)
    screen = materialized_config(pipeline.screen, _PREFIX)

    assert offline.calibration_path.startswith("gs://marin-us-central1/")
    assert offline.resources.regions == ["us-central1"]
    assert offline.representative_layer == 2
    assert offline.source_layer == 3
    assert offline.train_examples_per_layer == 256
    assert offline.heldout_examples_per_layer == 512
    assert offline.steps == 2_000
    assert offline.heldout_loss_gate == 0.349847

    assert screen.matching_path is None
    assert screen.init_checkpoint_dir == prefix_join(pipeline.offline.path(_PREFIX), "checkpoints")
    assert screen.resources.regions == ["us-central1"]
    assert screen.initialization is RecoveryInitialization.JOINT_REFACTORIZATION
    assert screen.stage is RecoveryStage.PRESERVATION
    assert screen.trainable_scope is RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS
    assert screen.assignment_mode is AssignmentMode.IDENTITY
    assert screen.training_tokens == 25_034_752
    assert screen.checkpoint_token_milestones == (25_034_752,)
    assert pipeline.offline in pipeline.screen.deps
    assert all("matching" not in dependency.name for dependency in pipeline.screen.deps)
