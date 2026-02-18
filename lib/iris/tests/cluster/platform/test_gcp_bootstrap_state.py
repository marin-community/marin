# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from iris.cluster.platform.base import CloudSliceState, SliceStatus
from iris.cluster.platform.gcp import GcpSliceHandle
from iris.rpc import config_pb2
from iris.time_utils import Timestamp


def _make_handle(*, bootstrapping: bool) -> GcpSliceHandle:
    return GcpSliceHandle(
        _slice_id="slice-1",
        _zone="us-central1-a",
        _project_id="proj",
        _labels={"iris-scale-group": "g"},
        _created_at=Timestamp.now(),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-8",
        _ssh_config=config_pb2.SshConfig(),
        _state="READY",
        _bootstrapping=bootstrapping,
    )


def test_describe_ready_for_discovered_slice_without_bootstrap(monkeypatch) -> None:
    handle = _make_handle(bootstrapping=False)
    monkeypatch.setattr(
        handle,
        "_describe_cloud",
        lambda: SliceStatus(state=CloudSliceState.READY, worker_count=1, workers=[]),
    )

    status = handle.describe()
    assert status.state == CloudSliceState.READY


def test_describe_bootstrapping_while_bootstrap_pending(monkeypatch) -> None:
    handle = _make_handle(bootstrapping=True)
    monkeypatch.setattr(
        handle,
        "_describe_cloud",
        lambda: SliceStatus(state=CloudSliceState.READY, worker_count=1, workers=[]),
    )

    status = handle.describe()
    assert status.state == CloudSliceState.BOOTSTRAPPING
