# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import ArtifactStep, materialized_config
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.expert_prefit import PrefitObjective
from experiments.grug.moe.launch_merge_recovery import MergeBranchName, build_merge_recovery_pipeline
from experiments.grug.moe.launch_tied_experts import TiedExpertPhase, tied_expert_runs
from experiments.grug.moe.merge_recovery import RecoveryStage

_PREFIX = "gs://marin-us-central1/test"


def _pipeline(*, teacher_commit: str | None = None):
    teacher = ArtifactStep.adopt(
        "test/grug-merge-teacher",
        "dev",
        "gs://marin-us-central1/test/teacher",
        kind=LevanterCheckpoint,
    )
    return build_merge_recovery_pipeline(teacher, version="dev", teacher_commit=teacher_commit)


def test_merge_pipeline_reuses_calibration_and_matching_across_all_ablation_branches() -> None:
    pipeline = _pipeline(teacher_commit="teacher-sha")

    assert {branch.name for branch in pipeline.branches} == set(MergeBranchName)
    assert len({branch.converted.fingerprint() for branch in pipeline.branches}) == 5
    for branch in pipeline.branches:
        assert pipeline.calibration in branch.converted.deps
        assert pipeline.matching in branch.converted.deps
        config = materialized_config(branch.converted, _PREFIX)
        assert config.calibration_path == pipeline.calibration.path(_PREFIX)
        assert config.matching_path == pipeline.matching.path(_PREFIX)
        assert config.assignment_mode is branch.assignment_mode
        assert config.resources.regions == ["us-central1"]
        assert config.source.source_commit == "teacher-sha"
        assert (config.prefit_path is not None) is branch.prefit_applied
        expected_prefit = {
            MergeBranchName.SPECTRAL_PREFIT: pipeline.prefit,
            MergeBranchName.NATIVE_AGGREGATE_PREFIT: pipeline.native_aggregate_prefit,
        }.get(branch.name)
        if expected_prefit is None:
            assert not branch.prefit_applied
        else:
            assert expected_prefit in branch.converted.deps


def test_stage_b_initializes_from_stage_a_permanent_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr("experiments.grug.moe.launch_merge_recovery.mixture", lambda *_args, **_kwargs: None)
    branch = next(branch for branch in _pipeline().branches if branch.name is MergeBranchName.SPECTRAL_PREFIT)

    stage_a = materialized_config(branch.stage_a, _PREFIX)
    stage_b = materialized_config(branch.stage_b, _PREFIX)

    assert stage_a.stage is RecoveryStage.LOCAL
    assert stage_a.init_checkpoint_dir == prefix_join(branch.converted.path(_PREFIX), "checkpoints")
    assert stage_b.stage is RecoveryStage.PRESERVATION
    assert stage_b.init_checkpoint_dir == prefix_join(branch.stage_a.path(_PREFIX), "checkpoints")
    assert stage_b.init_checkpoint_dir != stage_a.init_checkpoint_dir


def test_pipeline_modes_preserve_the_required_identity_native_spectral_comparisons() -> None:
    branches = {branch.name: branch for branch in _pipeline().branches}

    assert branches[MergeBranchName.IDENTITY].assignment_mode is AssignmentMode.IDENTITY
    assert branches[MergeBranchName.NATIVE].assignment_mode is AssignmentMode.NATIVE
    assert branches[MergeBranchName.SPECTRAL].assignment_mode is AssignmentMode.SPECTRAL
    assert not branches[MergeBranchName.SPECTRAL].prefit_applied
    assert branches[MergeBranchName.SPECTRAL_PREFIT].assignment_mode is AssignmentMode.SPECTRAL
    assert branches[MergeBranchName.SPECTRAL_PREFIT].prefit_applied
    aggregate = branches[MergeBranchName.NATIVE_AGGREGATE_PREFIT]
    assert aggregate.assignment_mode is AssignmentMode.NATIVE
    assert aggregate.prefit_applied
    aggregate_prefit_config = materialized_config(_pipeline().native_aggregate_prefit, _PREFIX)
    assert aggregate_prefit_config.assignment_mode is AssignmentMode.NATIVE
    assert aggregate_prefit_config.objective is PrefitObjective.AGGREGATE_ROUTED
    assert aggregate_prefit_config.run_id == "grug-xem-native-aggregate-prefit-d512-l2-l3"
    assert aggregate_prefit_config.resources.regions == ["us-central1"]


def test_no_launch_graph_construction_does_not_resolve_remote_checkpoint(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("checkpoint storage was accessed during graph construction")

    monkeypatch.setattr("levanter.checkpoint.latest_checkpoint_path", fail_if_called)
    pipeline = _pipeline()
    for branch in pipeline.branches:
        branch.stage_b.fingerprint()
        branch.stage_b.lower()

    monkeypatch.setattr("experiments.grug.moe.launch_merge_recovery.mixture", lambda *_args, **_kwargs: None)
    for branch in pipeline.branches:
        materialized_config(branch.converted, _PREFIX)
        materialized_config(branch.stage_a, _PREFIX)
        materialized_config(branch.stage_b, _PREFIX)


def test_merge_source_matches_full_untied_teacher_config(monkeypatch) -> None:
    monkeypatch.setattr("experiments.grug.moe.launch_merge_recovery.mixture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("experiments.grug.moe.launch_tied_experts.mixture", lambda *_args, **_kwargs: None)
    pipeline = _pipeline()
    baseline = tied_expert_runs(
        version="dev",
        phase=TiedExpertPhase.FULL,
        variant_names=("baseline",),
    )[0]

    merge_source = materialized_config(pipeline.calibration, _PREFIX).source
    baseline_config = materialized_config(baseline, _PREFIX)

    assert merge_source.model == baseline_config.model
    assert merge_source.optimizer == baseline_config.optimizer
    assert merge_source.training_steps == baseline_config.steps
