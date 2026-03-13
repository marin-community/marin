# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the periodic profiling loop."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB, recent_profiles
from iris.cluster.types import WorkerId
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp


def _worker_metadata() -> cluster_pb2.WorkerMetadata:
    return cluster_pb2.WorkerMetadata(
        hostname="test-host",
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )


def _register_worker(
    controller: Controller,
    worker_id: str,
    address: str = "host:8080",
) -> None:
    controller._transitions.register_or_refresh_worker(
        worker_id=WorkerId(worker_id),
        address=address,
        metadata=_worker_metadata(),
        ts=Timestamp.now(),
    )


def _make_stub_factory(profile_data: bytes = b"thread dump", error: str = "") -> MagicMock:
    """Build a mock stub factory whose stubs return the given profile response."""
    stub = MagicMock()
    stub.profile_task.return_value = cluster_pb2.ProfileTaskResponse(
        profile_data=profile_data,
        error=error,
    )
    factory = MagicMock()
    factory.get_stub.return_value = stub
    return factory


@pytest.fixture
def controller(tmp_path: Path) -> Controller:
    """Controller with a mock stub factory that returns successful profiles."""
    stub_factory = _make_stub_factory(profile_data=b"stack trace here")
    threads = ThreadContainer()
    config = ControllerConfig(
        bundle_prefix="gs://test-bucket/bundles",
        profiling_enabled=True,
        profiling_interval=Duration.from_seconds(600.0),
    )
    db = ControllerDB(tmp_path / "controller.sqlite3")
    ctrl = Controller(
        config=config,
        worker_stub_factory=stub_factory,
        threads=threads,
        db=db,
    )
    yield ctrl
    threads.stop()


def test_profile_all_workers_stores_profiles(controller: Controller) -> None:
    """Register workers, run one profiling round, verify profiles in DB."""
    _register_worker(controller, "w1", "host1:8080")
    _register_worker(controller, "w2", "host2:8080")

    controller._profile_all_workers()

    w1_profiles = recent_profiles(controller._db, "w1", "threads")
    w2_profiles = recent_profiles(controller._db, "w2", "threads")
    assert len(w1_profiles) == 1
    assert len(w2_profiles) == 1
    assert w1_profiles[0].data == b"stack trace here"
    assert w2_profiles[0].data == b"stack trace here"

    assert controller.stub_factory.get_stub.call_count == 2


def test_profile_all_workers_no_workers(controller: Controller) -> None:
    """Profiling with no workers should be a no-op."""
    controller._profile_all_workers()
    assert controller.stub_factory.get_stub.call_count == 0


def test_profile_failure_does_not_store(tmp_path: Path) -> None:
    """When profiling returns an error, no profile is stored."""
    stub_factory = _make_stub_factory(error="py-spy not found")
    threads = ThreadContainer()
    config = ControllerConfig(
        bundle_prefix="gs://test-bucket/bundles",
        profiling_enabled=True,
    )
    db = ControllerDB(tmp_path / "controller.sqlite3")
    ctrl = Controller(config=config, worker_stub_factory=stub_factory, threads=threads, db=db)

    _register_worker(ctrl, "w1", "host1:8080")
    ctrl._profile_all_workers()

    assert len(recent_profiles(ctrl._db, "w1", "threads")) == 0
    threads.stop()


def test_profile_rpc_exception_does_not_store(tmp_path: Path) -> None:
    """RPC exception during profiling should not store a profile or raise."""
    stub = MagicMock()
    stub.profile_task.side_effect = ConnectionError("worker unreachable")
    factory = MagicMock()
    factory.get_stub.return_value = stub

    threads = ThreadContainer()
    config = ControllerConfig(bundle_prefix="gs://test-bucket/bundles", profiling_enabled=True)
    db = ControllerDB(tmp_path / "controller.sqlite3")
    ctrl = Controller(config=config, worker_stub_factory=factory, threads=threads, db=db)

    _register_worker(ctrl, "w1", "host1:8080")
    ctrl._profile_all_workers()

    assert len(recent_profiles(ctrl._db, "w1", "threads")) == 0
    threads.stop()


def test_partial_failure_stores_successful_profiles(tmp_path: Path) -> None:
    """When some workers fail, successful ones still get profiles stored."""
    good_stub = MagicMock()
    good_stub.profile_task.return_value = cluster_pb2.ProfileTaskResponse(profile_data=b"good data")
    bad_stub = MagicMock()
    bad_stub.profile_task.side_effect = ConnectionError("unreachable")

    factory = MagicMock()
    factory.get_stub.side_effect = lambda addr: good_stub if "host1" in addr else bad_stub

    threads = ThreadContainer()
    config = ControllerConfig(bundle_prefix="gs://test-bucket/bundles", profiling_enabled=True)
    db = ControllerDB(tmp_path / "controller.sqlite3")
    ctrl = Controller(config=config, worker_stub_factory=factory, threads=threads, db=db)

    _register_worker(ctrl, "w1", "host1:8080")
    _register_worker(ctrl, "w2", "host2:8080")

    ctrl._profile_all_workers()

    assert len(recent_profiles(ctrl._db, "w1", "threads")) == 1
    assert len(recent_profiles(ctrl._db, "w2", "threads")) == 0
    threads.stop()


def test_profiling_loop_skips_during_checkpoint(tmp_path: Path) -> None:
    """The profiling loop should skip when checkpoint_in_progress is set."""
    stub_factory = _make_stub_factory(profile_data=b"data")
    threads = ThreadContainer()
    config = ControllerConfig(
        bundle_prefix="gs://test-bucket/bundles",
        profiling_enabled=True,
        profiling_interval=Duration.from_seconds(0.01),
    )
    db = ControllerDB(tmp_path / "controller.sqlite3")
    ctrl = Controller(config=config, worker_stub_factory=stub_factory, threads=threads, db=db)

    _register_worker(ctrl, "w1", "host1:8080")

    # Set checkpoint flag so the loop skips
    ctrl._checkpoint_in_progress = True

    stop_event = threading.Event()
    ran = threading.Event()

    original = ctrl._profile_all_workers

    def patched_profile_all() -> None:
        ran.set()
        original()

    ctrl._profile_all_workers = patched_profile_all

    t = threading.Thread(target=ctrl._run_profiling_loop, args=(stop_event,))
    t.start()

    # Give the loop time to run at least one iteration (interval is 0.01s)
    time.sleep(0.1)
    stop_event.set()
    t.join(timeout=2.0)

    # _profile_all_workers should NOT have been called because checkpoint was in progress
    assert not ran.is_set()
    assert len(recent_profiles(ctrl._db, "w1", "threads")) == 0
    threads.stop()
