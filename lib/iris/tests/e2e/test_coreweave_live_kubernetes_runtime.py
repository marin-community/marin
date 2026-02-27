# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live CoreWeave validation using KubernetesRuntime directly."""

from __future__ import annotations

import io
import os
import posixpath
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
import zipfile

import pytest
from iris.cluster.config import load_config
import fsspec.core
from iris.cluster.runtime.kubernetes import KubernetesRuntime
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def _entrypoint(argv: list[str]) -> cluster_pb2.RuntimeEntrypoint:
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv))
    return ep


def _wait_finished(handle, timeout_seconds: float) -> cluster_pb2.TaskState:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = handle.status()
        if not status.running:
            return cluster_pb2.TASK_STATE_SUCCEEDED if status.exit_code == 0 else cluster_pb2.TASK_STATE_FAILED
        time.sleep(2.0)
    raise TimeoutError(f"pod {handle.container_id} did not finish in {timeout_seconds}s")


@pytest.fixture
def coreweave_runtime() -> KubernetesRuntime:
    """Create a runtime configured from examples/coreweave.yaml.

    Simulates the env vars that CoreweavePlatform._s3_env_vars() sets on
    worker pods so KubernetesRuntime forwards them to task pods.
    """
    import json
    from iris.cluster.platform.coreweave import _needs_virtual_host_addressing

    iris_root = Path(__file__).resolve().parents[2]
    config = load_config(iris_root / "examples" / "coreweave.yaml")
    namespace = config.platform.coreweave.namespace or "iris"

    endpoint = config.platform.coreweave.object_storage_endpoint
    saved: dict[str, str | None] = {}
    if endpoint:
        saved["AWS_ENDPOINT_URL"] = os.environ.get("AWS_ENDPOINT_URL")
        os.environ["AWS_ENDPOINT_URL"] = endpoint
        fsspec_conf: dict = {"endpoint_url": endpoint}
        if _needs_virtual_host_addressing(endpoint):
            fsspec_conf["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
        saved["FSSPEC_S3"] = os.environ.get("FSSPEC_S3")
        os.environ["FSSPEC_S3"] = json.dumps(fsspec_conf)

    runtime = KubernetesRuntime(namespace=namespace)
    try:
        yield runtime
    finally:
        runtime.cleanup()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cache_mounts() -> list[tuple[str, str, str]]:
    return [
        ("/var/lib/iris/cache/uv", "/uv/cache", "rw"),
        ("/var/lib/iris/cache/cargo", "/root/.cargo/registry", "rw"),
        ("/var/lib/iris/cache/cargo-target", "/root/.cargo/target", "rw"),
    ]


def _bundle_access_env(config) -> dict[str, str]:
    """Return env vars task pods need to fetch bundles from object storage."""
    env: dict[str, str] = {}
    endpoint = config.platform.coreweave.object_storage_endpoint
    if endpoint:
        env["AWS_ENDPOINT_URL"] = endpoint

    key_id = os.environ.get("R2_ACCESS_KEY_ID", "")
    key_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if key_id and key_secret:
        env["AWS_ACCESS_KEY_ID"] = key_id
        env["AWS_SECRET_ACCESS_KEY"] = key_secret
    return env


def _upload_test_bundle(bundle_prefix: str, run_id: str) -> tuple[str, object, str]:
    """Upload a tiny zip bundle for live runtime extraction checks."""
    if not bundle_prefix:
        raise ValueError("storage.bundle_prefix is required")

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hello.txt", "hello-from-bundle\n")
        zf.writestr("sub/nested.txt", "nested-from-bundle\n")
    bundle_url = f"{bundle_prefix.rstrip('/')}/live-runtime/{run_id}/bundle.zip"
    fs, path = fsspec.core.url_to_fs(bundle_url)
    parent = posixpath.dirname(path)
    if parent:
        fs.mkdirs(parent, exist_ok=True)
    with fs.open(path, "wb") as f:
        f.write(zip_bytes.getvalue())
    return bundle_url, fs, path


@contextmanager
def _coreweave_upload_env(config) -> object:
    """Temporarily configure env vars and fsspec config so S3 uploads use R2 credentials."""
    import json
    import fsspec.config
    from iris.cluster.platform.coreweave import _needs_virtual_host_addressing

    previous = {
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "FSSPEC_S3": os.environ.get("FSSPEC_S3"),
    }
    previous_conf = dict(fsspec.config.conf.get("s3", {}))
    try:
        endpoint = config.platform.coreweave.object_storage_endpoint
        if endpoint:
            os.environ["AWS_ENDPOINT_URL"] = endpoint
            fsspec_s3: dict = {"endpoint_url": endpoint}
            if _needs_virtual_host_addressing(endpoint):
                fsspec_s3["config_kwargs"] = {"s3": {"addressing_style": "virtual"}}
            os.environ["FSSPEC_S3"] = json.dumps(fsspec_s3)
            fsspec.config.conf.setdefault("s3", {}).update(fsspec_s3)

        r2_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        if r2_key and r2_secret:
            os.environ["AWS_ACCESS_KEY_ID"] = r2_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = r2_secret
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_conf:
            fsspec.config.conf["s3"] = previous_conf
        else:
            fsspec.config.conf.pop("s3", None)


@pytest.mark.timeout(120)
def test_kubernetes_runtime_lifecycle(k8s_runtime: KubernetesRuntime):
    """Full KubernetesRuntime lifecycle: create pod, run, succeed, read logs."""
    run_id = uuid.uuid4().hex[:8]
    config = ContainerConfig(
        image="python:3.11-slim",
        entrypoint=_entrypoint(["bash", "-c", "echo lifecycle-test-ok && sleep 2"]),
        env={},
        workdir="/app",
        task_id=f"lifecycle-{run_id}",
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=100, memory_bytes=64 * 1024**2),
    )

    handle = k8s_runtime.create_container(config)
    handle.run()

    state = _wait_finished(handle, timeout_seconds=60)
    assert state == cluster_pb2.TASK_STATE_SUCCEEDED

    logs = handle.log_reader().read_all()
    assert any("lifecycle-test-ok" in line.data for line in logs)


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not available")
@pytest.mark.timeout(1800)
def test_coreweave_kubernetes_runtime_cpu_job_live(coreweave_runtime: KubernetesRuntime):
    """CPU pod should extract bundle and complete successfully via KubernetesRuntime."""
    config = load_config(Path(__file__).resolve().parents[2] / "examples" / "coreweave.yaml")
    image = config.defaults.worker.default_task_image
    run_id = uuid.uuid4().hex[:8]
    bundle_url = ""
    bundle_fs = None
    bundle_path = ""
    with _coreweave_upload_env(config):
        bundle_url, bundle_fs, bundle_path = _upload_test_bundle(config.storage.bundle_prefix, run_id)

    try:
        cpu_config = ContainerConfig(
            image=image,
            entrypoint=_entrypoint(
                [
                    "bash",
                    "-lc",
                    (
                        "set -euo pipefail; "
                        "echo BUNDLE_FILES_BEGIN; "
                        "find /app -type f | sort; "
                        "echo BUNDLE_HELLO; "
                        "cat /app/hello.txt; "
                        "echo BUNDLE_NESTED; "
                        "cat /app/sub/nested.txt; "
                        "echo runtime-cpu-ok"
                    ),
                ]
            ),
            env={
                "IRIS_BUNDLE_GCS_PATH": bundle_url,
                **_bundle_access_env(config),
            },
            workdir="/app",
            mounts=_cache_mounts(),
            network_mode="bridge",
            task_id=f"runtime-live-cpu-{run_id}",
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1 * 1024**3),
        )
        cpu_handle = coreweave_runtime.create_container(cpu_config)
        cpu_handle.run()
        cpu_state = _wait_finished(cpu_handle, timeout_seconds=900)
        reader = cpu_handle.log_reader()
        all_logs = reader.read_all()
        assert cpu_state == cluster_pb2.TASK_STATE_SUCCEEDED, f"cpu pod failed logs={all_logs}"

        logs = "\n".join(line.data for line in all_logs)
        assert "hello.txt" in logs
        assert "sub/nested.txt" in logs
        assert "hello-from-bundle" in logs
        assert "nested-from-bundle" in logs
        assert "runtime-cpu-ok" in logs
    finally:
        if bundle_fs is not None and bundle_path:
            try:
                bundle_fs.rm(bundle_path)
            except Exception:
                pass


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not available")
@pytest.mark.timeout(1800)
def test_incremental_log_reader_no_duplicates(coreweave_runtime: KubernetesRuntime):
    """Incremental log reads via byte-offset cursor must not produce duplicate lines.

    Runs a pod that emits numbered lines with sub-second timing, then polls
    with the log reader multiple times. The union of all incremental reads
    must equal the full log with zero duplicates.
    """
    config = load_config(Path(__file__).resolve().parents[2] / "examples" / "coreweave.yaml")
    image = config.defaults.worker.default_task_image
    run_id = uuid.uuid4().hex[:8]

    # Emit 20 numbered lines with 0.1s spacing so multiple land in the same second
    line_count = 20
    script = "; ".join(f"echo LINE_{i:04d}" for i in range(line_count)) + "; sleep 1"

    handle = coreweave_runtime.create_container(
        ContainerConfig(
            image=image,
            entrypoint=_entrypoint(["bash", "-lc", script]),
            env={},
            workdir="/app",
            mounts=_cache_mounts(),
            network_mode="bridge",
            task_id=f"log-dedup-{run_id}",
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=512 * 1024**2),
        )
    )
    handle.run()

    reader = handle.log_reader()
    collected: list[str] = []

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        new_lines = reader.read()
        for line in new_lines:
            collected.append(line.data)
        status = handle.status()
        if not status.running:
            # One final read to drain remaining
            for line in reader.read():
                collected.append(line.data)
            break
        time.sleep(0.5)

    # Extract numbered lines (ignore non-matching like timestamps or empty)
    numbered = [line for line in collected if line.startswith("LINE_")]

    assert len(numbered) == line_count, (
        f"Expected {line_count} lines, got {len(numbered)}. "
        f"Duplicates: {[line for line in numbered if numbered.count(line) > 1]}"
    )
    assert len(set(numbered)) == line_count, f"Duplicate lines found: {numbered}"

    # Verify ordering preserved
    expected = [f"LINE_{i:04d}" for i in range(line_count)]
    assert numbered == expected


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not available")
@pytest.mark.timeout(3600)
def test_coreweave_kubernetes_runtime_gpu_job_live(coreweave_runtime: KubernetesRuntime):
    """GPU pod should request GPU and prove device access via nvidia-smi."""
    config = load_config(Path(__file__).resolve().parents[2] / "examples" / "coreweave.yaml")
    image = config.defaults.worker.default_task_image
    run_id = uuid.uuid4().hex[:8]

    gpu_resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)
    gpu_resources.device.gpu.CopyFrom(cluster_pb2.GpuDevice(variant="H100", count=1))
    gpu_config = ContainerConfig(
        image=image,
        entrypoint=_entrypoint(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-gpu=name --format=csv,noheader && echo runtime-gpu-ok",
            ]
        ),
        env={},
        workdir="/app",
        mounts=_cache_mounts(),
        network_mode="bridge",
        task_id=f"runtime-live-gpu-{run_id}",
        resources=gpu_resources,
    )
    gpu_handle = coreweave_runtime.create_container(gpu_config)
    gpu_handle.run()

    pod_name = gpu_handle.container_id
    assert pod_name is not None
    pod = coreweave_runtime._kubectl.get_json("pod", pod_name)
    assert pod is not None
    limits = pod["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "1"

    gpu_state = _wait_finished(gpu_handle, timeout_seconds=2400)
    gpu_logs = gpu_handle.log_reader().read_all()
    assert gpu_state == cluster_pb2.TASK_STATE_SUCCEEDED, f"gpu pod failed logs={gpu_logs}"


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not available")
@pytest.mark.timeout(600)
def test_tensorstore_s3_roundtrip():
    """Verify tensorstore can write and read zarr3 data via S3-compatible storage.

    This exercises the build_kvstore_spec code path that maps s3:// URIs to
    tensorstore specs with the correct endpoint/region, catching regressions
    like the JAX regex bug (s3 contains a digit) or missing endpoint forwarding.
    """
    import numpy as np
    import tensorstore as ts
    from levanter.tensorstore_serialization import build_kvstore_spec

    config = load_config(Path(__file__).resolve().parents[2] / "examples" / "coreweave.yaml")

    endpoint = config.platform.coreweave.object_storage_endpoint
    if not endpoint:
        pytest.skip("No object_storage_endpoint configured")

    r2_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    if not r2_key or not r2_secret:
        pytest.skip("R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY not set")

    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", r2_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", r2_secret)

    # Derive bucket from config
    bucket = config.storage.bundle_prefix.removeprefix("s3://").split("/")[0]
    run_id = uuid.uuid4().hex[:8]
    test_path = f"s3://{bucket}/iris/test-tensorstore/{run_id}/data"

    kvstore = build_kvstore_spec(test_path)
    spec = {
        "driver": "zarr3",
        "kvstore": kvstore,
        "metadata": {
            "shape": [100],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100]}},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "data_type": "int32",
        },
    }

    # Write
    store = ts.open(spec, create=True, delete_existing=True).result()
    data = np.arange(100, dtype=np.int32)
    store[:].write(data).result()

    # Read back
    store2 = ts.open(spec, open=True).result()
    result = store2[:].read().result()
    np.testing.assert_array_equal(result, data)

    # Cleanup
    try:
        fs, path = fsspec.core.url_to_fs(test_path)
        fs.rm(path, recursive=True)
    except Exception:
        pass
