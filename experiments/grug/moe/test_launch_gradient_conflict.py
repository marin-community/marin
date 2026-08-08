# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import ArtifactStep, materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch_gradient_conflict import build_gradient_conflict_diagnostic

_PREFIX = "gs://marin-us-central1/test"
_VERSION = "2026.08.06"
_TEACHER_NAME = "grug/tied_experts/d512/full/baseline"
_STAGE_A_NAME = "grug/expert_merge/d512/native_local_ce_kl/stage-a"
_CONTROL_NAME = "grug/expert_merge/d512/native_local_ce_kl_adapter_control/stage-b"


def _adopt(name: str, path: str) -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep.adopt(name, _VERSION, path, kind=LevanterCheckpoint)


def test_gradient_conflict_graph_pins_checkpoints_batches_statistics_and_region(monkeypatch) -> None:
    monkeypatch.setattr("experiments.grug.moe.launch_gradient_conflict.mixture", lambda *_args, **_kwargs: None)
    teacher = _adopt(_TEACHER_NAME, "gs://marin-us-central1/test/teacher")
    stage_a = _adopt(_STAGE_A_NAME, "gs://marin-us-central1/test/stage-a")
    control = _adopt(_CONTROL_NAME, "gs://marin-us-central1/test/control")
    diagnostic = build_gradient_conflict_diagnostic(
        teacher,
        stage_a,
        control,
        version="dev",
        teacher_commit="884b213ff4",
    )
    config = materialized_config(diagnostic, _PREFIX)

    assert config.source.source_commit == "884b213ff4"
    assert config.teacher_artifact.fingerprint == "38d1fe9b"
    assert config.affected_layers == (2, 3)
    assert config.batch_size == 32
    assert config.num_batches == 16
    assert config.loader_start_step == 382
    assert config.bootstrap_samples == 10_000
    assert config.bootstrap_seed == 8_032
    assert config.seed == 0
    assert config.resources.regions == ["us-central1"]
    assert [
        (checkpoint.label, checkpoint.expected_step, checkpoint.continuation_tokens) for checkpoint in config.checkpoints
    ] == [
        ("selected_stage_a", 382, 0),
        ("shared_control_midpoint", 191, 25_034_752),
        ("shared_control_endpoint", 382, 50_069_504),
    ]
    assert config.checkpoints[0].checkpoint_path == "gs://marin-us-central1/test/stage-a/checkpoints/step-382"
    assert config.checkpoints[0].artifact.fingerprint == "62564720"
    assert config.checkpoints[1].checkpoint_path == "gs://marin-us-central1/test/control/checkpoints/step-191"
    assert config.checkpoints[2].checkpoint_path == "gs://marin-us-central1/test/control/checkpoints/step-382"
    assert config.checkpoints[1].artifact.fingerprint == "59969d51"
    assert teacher in diagnostic.deps
    assert stage_a in diagnostic.deps
    assert control in diagnostic.deps


def test_gradient_conflict_graph_lowering_does_not_read_checkpoint_storage(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("checkpoint storage was accessed during graph construction")

    monkeypatch.setattr("levanter.checkpoint.latest_checkpoint_path", fail_if_called)
    diagnostic = build_gradient_conflict_diagnostic(
        _adopt(_TEACHER_NAME, "gs://marin-us-central1/test/teacher"),
        _adopt(_STAGE_A_NAME, "gs://marin-us-central1/test/stage-a"),
        _adopt(_CONTROL_NAME, "gs://marin-us-central1/test/control"),
        version="dev",
        teacher_commit="884b213ff4",
    )
    diagnostic.fingerprint()
    diagnostic.lower()
