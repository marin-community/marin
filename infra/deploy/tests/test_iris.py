# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for checkpoint-paired Iris controller deployment."""

import subprocess
from pathlib import Path

import pytest
from iris.cluster.controller.rollout import RolloutPhase, RolloutRecord
from marin_deploy.iris import (
    IrisActivationError,
    IrisActivationSpec,
    IrisDeployError,
    activation_marker_path,
    apply_pulumi_activation,
    run_forward_activation,
    run_rollback_activation,
)


def _activation(image: str = "controller:new") -> IrisActivationSpec:
    return IrisActivationSpec(
        cluster="cluster",
        controller_image=image,
        worker_image="worker:new",
        task_image="task:new",
        activation_id="activation",
    )


def _prior_record() -> RolloutRecord:
    return RolloutRecord(
        phase=RolloutPhase.COMMITTED,
        image="controller:old",
        previous_image="controller:older",
        rollback_checkpoint="gs://state/older-checkpoint",
    )


def test_forward_activation_commits_image_and_paired_checkpoint() -> None:
    records: list[RolloutRecord] = []
    applied: list[IrisActivationSpec] = []

    run_forward_activation(
        remote_state_dir="gs://state",
        activation=_activation(),
        prior_record=_prior_record(),
        checkpoint="gs://state/pre-deploy",
        apply_candidate=applied.append,
        apply_rollback=applied.append,
        writer=lambda _remote, record: records.append(record),
    )

    assert [record.phase for record in records] == [RolloutPhase.PENDING, RolloutPhase.COMMITTED]
    assert [(record.image, record.previous_image, record.rollback_checkpoint) for record in records] == [
        ("controller:new", "controller:old", "gs://state/pre-deploy"),
        ("controller:new", "controller:old", "gs://state/pre-deploy"),
    ]
    assert applied == [_activation()]


def test_forward_activation_failure_restores_previous_image_with_new_checkpoint() -> None:
    records: list[RolloutRecord] = []
    applied: list[IrisActivationSpec] = []

    def apply(spec: IrisActivationSpec) -> None:
        applied.append(spec)
        if spec.controller_image == "controller:new":
            raise RuntimeError("candidate unhealthy")

    with pytest.raises(IrisDeployError):
        run_forward_activation(
            remote_state_dir="gs://state",
            activation=_activation(),
            prior_record=_prior_record(),
            checkpoint="gs://state/pre-deploy",
            apply_candidate=apply,
            apply_rollback=apply,
            writer=lambda _remote, record: records.append(record),
        )

    assert [spec.controller_image for spec in applied] == ["controller:new", "controller:old"]
    assert [record.phase for record in records] == [RolloutPhase.PENDING, RolloutPhase.ROLLBACK_REQUESTED]
    assert records[-1].image == "controller:old"
    assert records[-1].rollback_checkpoint == "gs://state/pre-deploy"


def test_forward_activation_failure_before_mutation_restores_prior_record_without_rollback() -> None:
    prior = _prior_record()
    records: list[RolloutRecord] = []
    rollback_attempts: list[IrisActivationSpec] = []

    def fail_before_mutation(_spec: IrisActivationSpec) -> None:
        raise IrisActivationError("stack is not initialized", started=False)

    with pytest.raises(IrisDeployError):
        run_forward_activation(
            remote_state_dir="gs://state",
            activation=_activation(),
            prior_record=prior,
            checkpoint="gs://state/pre-deploy",
            apply_candidate=fail_before_mutation,
            apply_rollback=rollback_attempts.append,
            writer=lambda _remote, record: records.append(record),
        )

    assert [record.phase for record in records] == [RolloutPhase.PENDING, RolloutPhase.COMMITTED]
    assert records[-1] == prior
    assert rollback_attempts == []


def test_first_activation_failure_before_mutation_does_not_create_false_rollout_record() -> None:
    records: list[RolloutRecord] = []

    def fail_before_mutation(_spec: IrisActivationSpec) -> None:
        raise IrisActivationError("stack is not initialized", started=False)

    with pytest.raises(IrisDeployError):
        run_forward_activation(
            remote_state_dir="gs://state",
            activation=_activation(),
            prior_record=None,
            checkpoint="gs://state/pre-deploy",
            apply_candidate=fail_before_mutation,
            apply_rollback=lambda _spec: None,
            writer=lambda _remote, record: records.append(record),
        )

    assert records == []


def test_pending_record_write_failure_prevents_controller_mutation() -> None:
    applied: list[IrisActivationSpec] = []

    def fail_write(_remote: str, _record: RolloutRecord) -> None:
        raise OSError("GCS unavailable")

    with pytest.raises(IrisDeployError):
        run_forward_activation(
            remote_state_dir="gs://state",
            activation=_activation(),
            prior_record=_prior_record(),
            checkpoint="gs://state/pre-deploy",
            apply_candidate=applied.append,
            apply_rollback=applied.append,
            writer=fail_write,
        )

    assert applied == []


def test_explicit_rollback_uses_recorded_image_and_checkpoint() -> None:
    records: list[RolloutRecord] = []
    applied: list[IrisActivationSpec] = []

    run_rollback_activation(
        remote_state_dir="gs://state",
        activation=_activation(),
        prior_record=_prior_record(),
        apply=applied.append,
        writer=lambda _remote, record: records.append(record),
    )

    assert [spec.controller_image for spec in applied] == ["controller:older"]
    assert [(record.phase, record.image, record.rollback_checkpoint) for record in records] == [
        (RolloutPhase.ROLLBACK_REQUESTED, "controller:older", "gs://state/older-checkpoint")
    ]


@pytest.mark.parametrize("started", [False, True])
def test_pulumi_failure_reports_whether_controller_mutation_started(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    started: bool,
) -> None:
    activation = _activation()
    (tmp_path / "Pulumi.cluster.yaml").write_text("config:\n  iris:cluster: cluster\n")

    def fail(_arguments, **_kwargs) -> None:
        if started:
            activation_marker_path(activation).write_text("started\n")
        raise subprocess.CalledProcessError(1, ["pulumi", "up"])

    monkeypatch.setattr("marin_deploy.iris.IRIS_PULUMI_DIRECTORY", tmp_path)
    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(IrisActivationError) as exc_info:
        apply_pulumi_activation(activation, yes=True)

    assert exc_info.value.started is started
    assert not activation_marker_path(activation).exists()
