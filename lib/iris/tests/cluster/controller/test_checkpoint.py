# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller checkpoint upload to remote storage."""

from pathlib import Path
from unittest.mock import patch

from iris.cluster.controller.checkpoint import (
    maybe_periodic_checkpoint,
)
from iris.cluster.controller.controller import (
    Controller,
    ControllerConfig,
)
from iris.time_utils import Duration, RateLimiter


class FakeStubFactory:
    def create(self, worker_id, address):
        pass

    def close(self):
        pass


def _make_controller(bundle_prefix: str = "file:///tmp/iris-test", **kwargs) -> Controller:
    config = ControllerConfig(bundle_prefix=bundle_prefix, **kwargs)
    return Controller(config=config, worker_stub_factory=FakeStubFactory())


def test_periodic_checkpoint_writes_local_and_uploads(tmp_path):
    """Periodic checkpoint writes a local copy and uploads to remote."""
    controller = _make_controller(
        bundle_prefix="gs://test-bucket/bundles",
        checkpoint_interval=Duration.from_seconds(0),  # always eligible
        log_dir=tmp_path,
    )
    limiter = RateLimiter(interval_seconds=0)
    limiter._last_run = 0

    uploaded = []

    def track_upload(local_path, created_at, bundle_prefix):
        uploaded.append((str(local_path), created_at))

    with patch("iris.cluster.controller.checkpoint.upload_checkpoint_to_remote", side_effect=track_upload):
        maybe_periodic_checkpoint(
            controller._db,
            "gs://test-bucket/bundles",
            limiter,
            checkpoint_in_progress=False,
        )

    # Local checkpoint should exist
    checkpoint_dir = tmp_path / "controller-checkpoints"
    assert checkpoint_dir.exists()
    checkpoints = list(checkpoint_dir.glob("checkpoint-*.sqlite3"))
    assert len(checkpoints) == 1

    # Remote upload should have been called
    assert len(uploaded) == 1
    assert uploaded[0][0] == str(checkpoints[0])

    controller._transitions.close()


def test_begin_checkpoint_uploads_to_remote(tmp_path):
    """begin_checkpoint writes a local copy and uploads to remote."""
    controller = _make_controller(
        bundle_prefix="gs://test-bucket/bundles",
        log_dir=tmp_path,
    )

    uploaded = []

    def track_upload(local_path, created_at, bundle_prefix):
        uploaded.append((str(local_path), created_at))

    with patch("iris.cluster.controller.checkpoint.upload_checkpoint_to_remote", side_effect=track_upload):
        path, _result = controller.begin_checkpoint()

    # Local checkpoint should exist
    assert Path(path).exists()

    # Remote upload should have been called
    assert len(uploaded) == 1
    assert uploaded[0][0] == path

    controller._transitions.close()


def test_atexit_checkpoint_writes_and_uploads(tmp_path):
    """_atexit_checkpoint writes a local copy and uploads to remote."""
    controller = _make_controller(
        bundle_prefix="gs://test-bucket/bundles",
        log_dir=tmp_path,
    )

    uploaded = []

    def track_upload(local_path, created_at, bundle_prefix):
        uploaded.append((str(local_path), created_at))

    with patch("iris.cluster.controller.checkpoint.upload_checkpoint_to_remote", side_effect=track_upload):
        controller._atexit_checkpoint()

    checkpoint_dir = tmp_path / "controller-checkpoints"
    checkpoints = list(checkpoint_dir.glob("checkpoint-*.sqlite3"))
    assert len(checkpoints) == 1
    assert len(uploaded) == 1

    controller._transitions.close()


def test_no_upload_for_local_bundle_prefix(tmp_path):
    """No remote upload when bundle_prefix is local."""
    controller = _make_controller(
        bundle_prefix="file:///tmp/local-bundles",
        checkpoint_interval=Duration.from_seconds(0),
        log_dir=tmp_path,
    )
    limiter = RateLimiter(interval_seconds=0)
    limiter._last_run = 0

    maybe_periodic_checkpoint(
        controller._db,
        "file:///tmp/local-bundles",
        limiter,
        checkpoint_in_progress=False,
    )

    # Local checkpoint should exist
    checkpoint_dir = tmp_path / "controller-checkpoints"
    assert checkpoint_dir.exists()
    checkpoints = list(checkpoint_dir.glob("checkpoint-*.sqlite3"))
    assert len(checkpoints) == 1

    controller._transitions.close()
