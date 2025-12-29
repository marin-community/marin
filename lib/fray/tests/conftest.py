# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest fixtures for fray tests."""


import logging
import os
import secrets
import shutil
import tempfile
from pathlib import Path

import pytest
from fray.cluster.local_cluster import LocalCluster, LocalClusterConfig

# Ensure Ray subprocesses can pick up our test-only `sitecustomize.py`.
tests_dir = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{tests_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)

# Ray token auth is now assumed to be enabled in Marin/Fray.
# For tests, ensure there's a stable token available *before* importing Ray.
os.environ.setdefault("RAY_AUTH_MODE", "token")
if "RAY_AUTH_TOKEN" not in os.environ and "RAY_AUTH_TOKEN_PATH" not in os.environ:
    token_dir = Path(tempfile.mkdtemp(prefix="fray-ray-auth-"))
    token_path = token_dir / "auth_token"
    token_path.write_text(secrets.token_hex(32))
    token_path.chmod(0o600)
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(token_path)

    import atexit

    atexit.register(lambda: shutil.rmtree(token_dir, ignore_errors=True))

# In some environments Ray's job supervisor uses process-scanning to discover a
# local cluster address ("auto"), which can be blocked by sandboxing. Provide a
# small side-channel file that our test-only `sitecustomize.py` can use instead.
if "FRAY_RAY_BOOTSTRAP_ADDRESS_PATH" not in os.environ:
    bootstrap_dir = Path(tempfile.mkdtemp(prefix="fray-ray-bootstrap-"))
    bootstrap_path = bootstrap_dir / "ray_address"
    bootstrap_path.write_text("")
    bootstrap_path.chmod(0o600)
    os.environ["FRAY_RAY_BOOTSTRAP_ADDRESS_PATH"] = str(bootstrap_path)

    import atexit

    atexit.register(lambda: shutil.rmtree(bootstrap_dir, ignore_errors=True))


@pytest.fixture(scope="module")
def ray_cluster():
    from fray.cluster.ray import RayCluster
    import ray

    patched_ray_node = False
    orig_get_system_pids = None
    if not ray.is_initialized():
        # Ray 2.53+ calls psutil to enumerate dashboard child processes during
        # startup, which can fail in sandboxed macOS environments (PermissionError
        # from sysctl). This process listing is only used for Linux cgroup-based
        # resource isolation, so it's safe to best-effort it for tests.
        from ray._private import node as ray_node

        orig_get_system_pids = ray_node.Node._get_system_processes_for_resource_isolation

        def _safe_get_system_pids(self: ray_node.Node) -> str:
            try:
                return orig_get_system_pids(self)
            except PermissionError:
                system_process_pids = [str(p[0].process.pid) for p in self.all_processes.values()]
                return ",".join(system_process_pids)

        ray_node.Node._get_system_processes_for_resource_isolation = _safe_get_system_pids
        patched_ray_node = True

        logging.info("Initializing Ray cluster")
        ray.init(
            address="local",
            num_cpus=8,
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            resources={"head_node": 1},
        )
        bootstrap_path = os.environ.get("FRAY_RAY_BOOTSTRAP_ADDRESS_PATH")
        if bootstrap_path:
            try:
                Path(bootstrap_path).write_text(ray._private.worker.global_worker.node.address_info["gcs_address"])
            except Exception:
                pass
    try:
        yield RayCluster()
    finally:
        ray.shutdown()
        if patched_ray_node and orig_get_system_pids is not None:
            from ray._private import node as ray_node

            ray_node.Node._get_system_processes_for_resource_isolation = orig_get_system_pids


@pytest.fixture(scope="module")
def local_cluster():
    yield LocalCluster(LocalClusterConfig(use_isolated_env=True))


@pytest.fixture(scope="module", params=["local", "ray"])
def cluster(request, local_cluster, ray_cluster):
    if request.param == "local":
        return local_cluster
    elif request.param == "ray":
        return ray_cluster
