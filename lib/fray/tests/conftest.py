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

# Note: For best performance and to avoid pulling in heavyweight workspace deps,
# run Fray tests from `lib/fray/` (e.g. `cd lib/fray && uv run pytest ...`).
# Ensure Ray subprocesses can pick up our test-only `sitecustomize.py`.
tests_dir = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = f"{tests_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)


def _configure_ray_auth_for_tests() -> None:
    """Configure Ray auth environment for tests.

    By default, run tests against a token-authenticated local Ray cluster. This
    matches our production assumptions and avoids relying on whatever token may
    exist in `~/.ray/auth_token`. If `~/.ray/auth_token` exists, we default to it
    (so developers can use a stable token across local runs).

    To force token auth on for tests, set `FRAY_TEST_RAY_AUTH_MODE=token`.
    To force auth off for tests, set `FRAY_TEST_RAY_AUTH_MODE=disabled`.
    `FRAY_TEST_RAY_AUTH_MODE=auto` attempts to mirror the parent environment.
    """
    auth_mode = os.environ.get("FRAY_TEST_RAY_AUTH_MODE", "token").strip().lower()
    if auth_mode not in {"auto", "token", "disabled"}:
        raise ValueError(f"Invalid FRAY_TEST_RAY_AUTH_MODE={auth_mode!r}; expected auto|token|disabled")

    if auth_mode == "auto":
        ray_auth_mode = os.environ.get("RAY_AUTH_MODE")
        if ray_auth_mode in {"token", "disabled"}:
            wants_token = ray_auth_mode == "token"
        else:
            wants_token = (
                "RAY_AUTH_TOKEN" in os.environ
                or "RAY_AUTH_TOKEN_PATH" in os.environ
                or (Path.home() / ".ray" / "auth_token").exists()
            )
    else:
        wants_token = auth_mode == "token"

    if wants_token:
        # Use a test-only token rather than relying on whatever happens to be in
        # ~/.ray/auth_token, to avoid flakiness when developers are connected to
        # other clusters.
        os.environ["RAY_AUTH_MODE"] = "token"
        if "RAY_AUTH_TOKEN" in os.environ and "RAY_AUTH_TOKEN_PATH" in os.environ:
            return

        default_token_path = Path.home() / ".ray" / "auth_token"
        if default_token_path.exists():
            os.environ.setdefault("RAY_AUTH_TOKEN_PATH", str(default_token_path))
            return

        token_dir = Path(tempfile.mkdtemp(prefix="fray-ray-auth-"))
        token_path = token_dir / "auth_token"
        token = secrets.token_hex(32)
        token_path.write_text(token)
        token_path.chmod(0o600)
        os.environ["RAY_AUTH_TOKEN"] = token
        os.environ["RAY_AUTH_TOKEN_PATH"] = str(token_path)

        import atexit

        atexit.register(lambda: shutil.rmtree(token_dir, ignore_errors=True))
        return

    # Unauthenticated cluster. Clear any token state inherited from the parent
    # environment before importing Ray.
    os.environ["RAY_AUTH_MODE"] = "disabled"
    os.environ.pop("RAY_AUTH_TOKEN", None)
    os.environ.pop("RAY_AUTH_TOKEN_PATH", None)


_configure_ray_auth_for_tests()

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
    patched_ray_uv_hook = False
    orig_get_system_pids = None
    orig_uv_pids = None
    if not ray.is_initialized():
        # Ray 2.53+ calls psutil to enumerate dashboard child processes during
        # startup, which can fail in sandboxed macOS environments (PermissionError
        # from sysctl). This process listing is only used for Linux cgroup-based
        # resource isolation, so it's safe to best-effort it for tests.
        from ray._private import node as ray_node

        orig_get_system_pids = getattr(ray_node.Node, "_get_system_processes_for_resource_isolation", None)
        if orig_get_system_pids is not None:
            # Ray 2.53+ uses psutil in its uv runtime-env hook to walk parent
            # processes. In sandboxed macOS environments, psutil's PID enumeration
            # can fail with PermissionError from sysctl and prevent ray.init from
            # starting.
            from ray._private.runtime_env import uv_runtime_env_hook as uv_hook

            orig_uv_pids = uv_hook.psutil.pids

            def _safe_uv_pids() -> list[int]:
                try:
                    return orig_uv_pids()
                except PermissionError:
                    return [1]

            uv_hook.psutil.pids = _safe_uv_pids  # type: ignore[assignment]
            patched_ray_uv_hook = True

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
            # ray declares this as int, but its default value is a string!
            logging_level="info",  # type: ignore
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
        if patched_ray_uv_hook and orig_uv_pids is not None:
            from ray._private.runtime_env import uv_runtime_env_hook as uv_hook

            uv_hook.psutil.pids = orig_uv_pids  # type: ignore[assignment]


@pytest.fixture(scope="module")
def local_cluster():
    from fray.cluster.local_cluster import LocalCluster, LocalClusterConfig

    yield LocalCluster(
        LocalClusterConfig(
            isolated_env_mode="shared",
        )
    )


@pytest.fixture(scope="module", params=["local", "ray"])
def cluster(request):
    if request.param == "local":
        return request.getfixturevalue("local_cluster")
    elif request.param == "ray":
        return request.getfixturevalue("ray_cluster")
    raise RuntimeError(f"Unknown cluster param: {request.param!r}")
