# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for snapshot CLI commands and periodic checkpointing."""


from click.testing import CliRunner

from iris.cli.main import iris
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.snapshot import read_latest_snapshot, read_snapshot_from_path
from iris.cluster.controller.state import ControllerState
from iris.time_utils import Duration, Timestamp


class _FakeStubFactory:
    """Minimal stub factory satisfying the WorkerStubFactory protocol."""

    def get_stub(self, address: str):
        raise NotImplementedError("stub not needed in unit tests")

    def evict(self, address: str) -> None:
        pass


def _make_controller(tmp_path: str, checkpoint_interval: Duration | None = None) -> Controller:
    config = ControllerConfig(
        bundle_prefix=f"file://{tmp_path}/bundles",
        checkpoint_interval=checkpoint_interval,
    )
    return Controller(config=config, worker_stub_factory=_FakeStubFactory())


# =============================================================================
# CLI checkpoint command tests
# =============================================================================


def test_checkpoint_cli_invokes_rpc(tmp_path):
    """The CLI checkpoint command calls BeginCheckpoint and prints the path."""
    controller = _make_controller(str(tmp_path))
    controller.start()
    try:
        port = controller._server.servers[0].sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        runner = CliRunner()
        result = runner.invoke(
            iris,
            ["--controller-url", url, "cluster", "controller", "checkpoint"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "Snapshot written:" in result.output
        assert "Jobs:" in result.output
        assert "Tasks:" in result.output
        assert "Workers:" in result.output
    finally:
        controller.stop()


# =============================================================================
# Periodic checkpointing tests
# =============================================================================


def test_periodic_checkpoint_writes_snapshot(tmp_path):
    """Periodic checkpointing writes a snapshot when the interval elapses."""
    # Use a very short interval so the test completes quickly.
    checkpoint_interval = Duration.from_ms(50)
    controller = _make_controller(str(tmp_path), checkpoint_interval=checkpoint_interval)
    controller.start()
    try:
        # Drive periodic checkpoint directly via the private method instead of
        # waiting on the autoscaler loop (which requires a configured autoscaler).
        # Force the last checkpoint timestamp to zero so the interval check triggers.
        controller._last_periodic_checkpoint = Timestamp.from_ms(0)
        controller._maybe_periodic_checkpoint()

        storage_prefix = f"file://{tmp_path}/bundles"
        snapshot = read_latest_snapshot(storage_prefix)
        assert snapshot is not None
        assert snapshot.schema_version == 1
    finally:
        controller.stop()


def test_periodic_checkpoint_respects_interval(tmp_path):
    """Periodic checkpoint is skipped when the interval has not elapsed."""
    checkpoint_interval = Duration.from_seconds(9999)
    controller = _make_controller(str(tmp_path), checkpoint_interval=checkpoint_interval)
    controller.start()
    try:
        # Set last checkpoint to now so the interval hasn't elapsed yet.
        controller._last_periodic_checkpoint = Timestamp.now()
        controller._maybe_periodic_checkpoint()

        storage_prefix = f"file://{tmp_path}/bundles"
        snapshot = read_latest_snapshot(storage_prefix)
        assert snapshot is None, "Should not have written a snapshot when interval not elapsed"
    finally:
        controller.stop()


def test_periodic_checkpoint_disabled_when_no_interval(tmp_path):
    """No snapshot is written when checkpoint_interval is None."""
    controller = _make_controller(str(tmp_path), checkpoint_interval=None)
    controller.start()
    try:
        controller._last_periodic_checkpoint = Timestamp.from_ms(0)
        controller._maybe_periodic_checkpoint()

        storage_prefix = f"file://{tmp_path}/bundles"
        snapshot = read_latest_snapshot(storage_prefix)
        assert snapshot is None
    finally:
        controller.stop()


# =============================================================================
# read_snapshot_from_path tests
# =============================================================================


def test_read_snapshot_from_path_returns_none_for_missing(tmp_path):
    """read_snapshot_from_path returns None for a path that does not exist."""
    from iris.cluster.controller.snapshot import read_snapshot_from_path

    result = read_snapshot_from_path(f"file://{tmp_path}/nonexistent.json")
    assert result is None


def test_read_snapshot_from_path_roundtrip(tmp_path):
    """write_snapshot + read_snapshot_from_path roundtrip returns the same data."""
    from iris.cluster.controller.snapshot import create_snapshot, write_snapshot

    state = ControllerState()
    result = create_snapshot(state)
    storage_prefix = f"file://{tmp_path}/bundles"
    path = write_snapshot(result.proto, storage_prefix)

    # Normalize file:// path for read_snapshot_from_path
    read_proto = read_snapshot_from_path(path)
    assert read_proto is not None
    assert read_proto.schema_version == result.proto.schema_version
    assert read_proto.created_at.epoch_ms == result.proto.created_at.epoch_ms
