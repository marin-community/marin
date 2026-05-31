# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from iris.dev_tpu import DevTpuState, DevTpuWorker, GcpNodeRef, parse_worker_host
from iris.rpc import job_pb2, query_pb2


def _install_marin_gcp_stub_when_unavailable() -> None:
    try:
        marin_gcp_spec = importlib.util.find_spec("marin.cluster.gcp")
    except ModuleNotFoundError:
        marin_gcp_spec = None
    if marin_gcp_spec is not None:
        return

    marin_module = types.ModuleType("marin")
    marin_cluster_module = types.ModuleType("marin.cluster")
    marin_gcp_module = types.ModuleType("marin.cluster.gcp")
    marin_gcp_module.get_project_id = lambda: None
    marin_gcp_module.find_tpu_by_name = lambda _name, _project, *, zone="-": None
    marin_gcp_module.find_tpu_by_ip = lambda _ip, _project, *, zone="-": None
    marin_cluster_module.gcp = marin_gcp_module
    marin_module.cluster = marin_cluster_module
    sys.modules.setdefault("marin", marin_module)
    sys.modules.setdefault("marin.cluster", marin_cluster_module)
    sys.modules.setdefault("marin.cluster.gcp", marin_gcp_module)


def _install_watchdog_stub_when_unavailable() -> None:
    try:
        watchdog_spec = importlib.util.find_spec("watchdog")
    except ModuleNotFoundError:
        watchdog_spec = None
    if watchdog_spec is not None:
        return

    watchdog_module = types.ModuleType("watchdog")
    watchdog_events_module = types.ModuleType("watchdog.events")
    watchdog_observers_module = types.ModuleType("watchdog.observers")
    watchdog_events_module.FileSystemEventHandler = type("FileSystemEventHandler", (), {})
    watchdog_observers_module.Observer = type("Observer", (), {})
    watchdog_module.events = watchdog_events_module
    watchdog_module.observers = watchdog_observers_module
    sys.modules.setdefault("watchdog", watchdog_module)
    sys.modules.setdefault("watchdog.events", watchdog_events_module)
    sys.modules.setdefault("watchdog.observers", watchdog_observers_module)


_DEV_TPU_SCRIPT = Path(__file__).parents[3] / "scripts" / "iris" / "dev_tpu.py"
_SPEC = importlib.util.spec_from_file_location("dev_tpu_script", _DEV_TPU_SCRIPT)
assert _SPEC is not None
dev_tpu_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dev_tpu_script
assert _SPEC.loader is not None
_install_marin_gcp_stub_when_unavailable()
_install_watchdog_stub_when_unavailable()
_SPEC.loader.exec_module(dev_tpu_script)


def _worker_query_response(row: list[str]) -> query_pb2.RawQueryResponse:
    return query_pb2.RawQueryResponse(
        columns=[
            query_pb2.ColumnMeta(name="address"),
            query_pb2.ColumnMeta(name="md_ip_address"),
            query_pb2.ColumnMeta(name="md_tpu_name"),
            query_pb2.ColumnMeta(name="md_tpu_worker_id"),
            query_pb2.ColumnMeta(name="md_gce_instance_name"),
            query_pb2.ColumnMeta(name="md_gce_zone"),
        ],
        rows=[json.dumps(row)],
    )


def test_parse_worker_host_accepts_http_address():
    assert parse_worker_host("http://10.0.0.12:10001") == "10.0.0.12"


def test_parse_worker_host_accepts_host_port_without_scheme():
    assert parse_worker_host("10.0.0.13:10001") == "10.0.0.13"


def test_parse_worker_host_rejects_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_worker_host("http://:10001")


def test_dev_tpu_state_json_roundtrip():
    state = DevTpuState(
        session_name="dlwh-branch-123456",
        config_file="/tmp/iris.yaml",
        job_id="/dlwh/dev-tpu-dlwh-branch-123456",
        tpu_type="v5p-8",
        workers=[
            DevTpuWorker(
                task_id="/dlwh/dev-tpu-dlwh-branch-123456/0",
                worker_id="10.0.0.12",
                worker_address="http://10.0.0.12:10001",
                host="10.0.0.12",
                node=GcpNodeRef(
                    kind="tpu",
                    name="iris-dev-tpu",
                    zone="us-east5-a",
                    project="hai-gcp-models",
                    tpu_worker_id=0,
                ),
            )
        ],
    )

    restored = DevTpuState.from_json(state.to_json())

    assert restored == state


def test_resolve_node_ref_from_tpu_metadata_with_zone():
    metadata = job_pb2.WorkerMetadata(
        tpu_name="iris-v5p-slice",
        tpu_worker_id="3",
        gce_zone="us-east5-a",
    )

    assert dev_tpu_script.resolve_node_ref_from_worker_metadata(metadata, "hai-gcp-models") == GcpNodeRef(
        kind="tpu",
        name="iris-v5p-slice",
        zone="us-east5-a",
        project="hai-gcp-models",
        tpu_worker_id=3,
    )


def test_resolve_node_ref_from_tpu_metadata_falls_back_to_gcloud_name_lookup(monkeypatch):
    metadata = job_pb2.WorkerMetadata(
        tpu_name="iris-v5p-slice",
        tpu_worker_id="1",
    )

    def find_tpu_by_name(name, project, *, zone="-"):
        assert project == "hai-gcp-models"
        assert zone == "-"
        return name, "us-central1-a"

    monkeypatch.setattr(
        dev_tpu_script.gcp,
        "find_tpu_by_name",
        find_tpu_by_name,
    )

    assert dev_tpu_script.resolve_node_ref_from_worker_metadata(metadata, "hai-gcp-models") == GcpNodeRef(
        kind="tpu",
        name="iris-v5p-slice",
        zone="us-central1-a",
        project="hai-gcp-models",
        tpu_worker_id=1,
    )


def test_resolve_node_ref_from_vm_metadata():
    metadata = job_pb2.WorkerMetadata(
        gce_instance_name="iris-worker-vm",
        gce_zone="us-central1-a",
    )

    assert dev_tpu_script.resolve_node_ref_from_worker_metadata(metadata, "hai-gcp-models") == GcpNodeRef(
        kind="vm",
        name="iris-worker-vm",
        zone="us-central1-a",
        project="hai-gcp-models",
    )


def test_worker_address_lookup_values_include_rpc_address_forms():
    assert dev_tpu_script._worker_address_lookup_values("http://10.202.0.55:10001") == [
        "http://10.202.0.55:10001",
        "10.202.0.55:10001",
    ]
    assert dev_tpu_script._worker_address_lookup_values("10.202.0.55:10001") == ["10.202.0.55:10001"]


def test_worker_resolution_metadata_decodes_live_tpu_worker_row_shape():
    response = _worker_query_response(
        ["10.202.0.225:10001", "10.202.0.225", "t1v-n-6d49d11b", "0", "t1v-n-6d49d11b-w-0", ""]
    )

    result = dev_tpu_script.worker_resolution_metadata_from_response(response)

    assert result is not None
    assert result.address == "10.202.0.225:10001"
    assert result.metadata.ip_address == "10.202.0.225"
    assert result.metadata.tpu_name == "t1v-n-6d49d11b"
    assert result.metadata.tpu_worker_id == "0"
    assert result.metadata.gce_instance_name == "t1v-n-6d49d11b-w-0"
    assert result.metadata.gce_zone == ""


def test_live_tpu_worker_row_shape_resolves_to_tpu_node(monkeypatch):
    response = _worker_query_response(
        ["10.202.0.225:10001", "10.202.0.225", "t1v-n-6d49d11b", "0", "t1v-n-6d49d11b-w-0", ""]
    )

    def find_tpu_by_name(name, project, *, zone="-"):
        assert project == "hai-gcp-models"
        assert zone == "-"
        return name, "us-east5-a"

    monkeypatch.setattr(dev_tpu_script.gcp, "find_tpu_by_name", find_tpu_by_name)

    result = dev_tpu_script.worker_resolution_metadata_from_response(response)
    assert result is not None
    node = dev_tpu_script.resolve_node_ref_from_worker_metadata(result.metadata, "hai-gcp-models")

    assert node == GcpNodeRef(
        kind="tpu",
        name="t1v-n-6d49d11b",
        zone="us-east5-a",
        project="hai-gcp-models",
        tpu_worker_id=0,
    )
