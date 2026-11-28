# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller restart failure behavior at the CLI boundary."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.cli import cluster as cluster_cli
from iris.cluster.config import ControllerVmConfig, IrisClusterConfig, ManualControllerConfig, StorageConfig
from iris.cluster.controller.rollout import RolloutPhase, RolloutRecord, read_rollout_record, write_rollout_record
from iris.cluster.platforms.protocols import ControllerRestartNotStarted


@pytest.mark.parametrize(
    ("reached_vm", "prior_phase", "rollback_probe_fails"),
    [
        (False, RolloutPhase.COMMITTED, False),
        (False, RolloutPhase.PENDING, False),
        (False, RolloutPhase.ROLLBACK_REQUESTED, False),
        (True, RolloutPhase.COMMITTED, False),
        (True, RolloutPhase.COMMITTED, True),
    ],
)
def test_controller_restart_failure_preserves_or_requests_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reached_vm: bool,
    prior_phase: RolloutPhase,
    rollback_probe_fails: bool,
) -> None:
    remote_state_dir = f"file://{tmp_path}"
    checkpoint = tmp_path / "pre-deploy-checkpoint"
    checkpoint.mkdir()
    checkpoint_url = f"file://{checkpoint}"
    prior_record = RolloutRecord(
        phase=prior_phase,
        image="controller:old",
        previous_image="controller:older",
        rollback_checkpoint="file://previous-checkpoint",
        updated_at_ms=123,
    )
    write_rollout_record(remote_state_dir, prior_record)
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="controller:old", manual=ManualControllerConfig(host="controller")),
        storage=StorageConfig(remote_state_dir=remote_state_dir),
    )

    class ControllerProvider:
        def preflight_controller(self, _config: IrisClusterConfig) -> None:
            pass

        def restart_controller(self, restarted_config: IrisClusterConfig) -> str:
            if restarted_config.controller.image == "controller:new":
                if reached_vm:
                    raise RuntimeError("health check failed")
                raise ControllerRestartNotStarted("SSH authorization failed")
            if rollback_probe_fails:
                raise ControllerRestartNotStarted("SSH authorization failed")
            return "http://controller:10000"

    def build_image(_ctx, built_config: IrisClusterConfig, **_kwargs) -> str:
        built_config.controller.image = "controller:new"
        return "controller:new"

    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))
    monkeypatch.setattr(cluster_cli, "_build_forward_image", build_image)
    monkeypatch.setattr(
        cluster_cli,
        "rpc_client_for_ctx",
        lambda _ctx, **_kwargs: nullcontext(
            SimpleNamespace(
                begin_checkpoint=lambda *_args, **_kwargs: SimpleNamespace(
                    checkpoint_path=checkpoint_url, job_count=1, worker_count=1
                )
            )
        ),
    )

    result = CliRunner().invoke(
        cluster_cli.controller_restart,
        obj={"config": config, "controller_url": "http://controller:10000"},
    )

    assert result.exit_code == 1
    if rollback_probe_fails:
        assert isinstance(result.exception, SystemExit)
    record = read_rollout_record(remote_state_dir)
    if rollback_probe_fails:
        assert record is not None
        assert record.phase is RolloutPhase.PENDING
        assert record.image == "controller:new"
        assert record.previous_image == "controller:old"
        assert record.rollback_checkpoint == checkpoint_url
    elif reached_vm:
        assert record is not None
        assert record.phase is RolloutPhase.ROLLBACK_REQUESTED
        assert record.image == "controller:old"
        assert record.rollback_checkpoint == checkpoint_url
    else:
        assert record == prior_record


def test_controller_restart_before_vm_with_no_prior_record_keeps_original_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_state_dir = f"file://{tmp_path}"
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="controller:old", manual=ManualControllerConfig(host="controller")),
        storage=StorageConfig(remote_state_dir=remote_state_dir),
    )

    class ControllerProvider:
        def preflight_controller(self, _config: IrisClusterConfig) -> None:
            pass

        def restart_controller(self, _config: IrisClusterConfig) -> str:
            raise ControllerRestartNotStarted("SSH authorization failed")

    def build_image(_ctx, built_config: IrisClusterConfig, **_kwargs) -> str:
        built_config.controller.image = "controller:new"
        return "controller:new"

    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))
    monkeypatch.setattr(cluster_cli, "_build_forward_image", build_image)

    result = CliRunner().invoke(
        cluster_cli.controller_restart,
        ["--skip-checkpoint"],
        obj={"config": config, "controller_url": "http://controller:10000"},
    )

    assert result.exit_code == 1
    record = read_rollout_record(remote_state_dir)
    assert record is not None
    assert record.phase is RolloutPhase.COMMITTED
    assert record.image == "controller:old"
    assert record.rollback_checkpoint is None


def test_explicit_rollback_prebootstrap_failure_restores_prior_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_state_dir = f"file://{tmp_path}"
    prior_record = RolloutRecord(
        phase=RolloutPhase.COMMITTED,
        image="controller:new",
        previous_image="controller:old",
        rollback_checkpoint="file://pre-deploy-checkpoint",
    )
    write_rollout_record(remote_state_dir, prior_record)
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="controller:new", manual=ManualControllerConfig(host="controller")),
        storage=StorageConfig(remote_state_dir=remote_state_dir),
    )

    class ControllerProvider:
        def preflight_controller(self, _config: IrisClusterConfig) -> None:
            pass

        def restart_controller(self, _config: IrisClusterConfig) -> str:
            raise ControllerRestartNotStarted("SSH authorization failed")

    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))

    result = CliRunner().invoke(
        cluster_cli.controller_restart,
        ["--rollback"],
        obj={"config": config, "controller_url": "http://controller:10000"},
    )

    assert result.exit_code == 1
    assert read_rollout_record(remote_state_dir) == prior_record
