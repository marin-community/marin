import re
import sys
from pathlib import Path

import pytest
from ray._private.parameter import RayParams

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import task_runtime as runtime  # noqa: E402
from cloud.iris.ray_ports import RayPort  # noqa: E402
from cloud.iris.ray_storage import (  # noqa: E402
    DEFAULT_RAY_SPILL_DIR,
    LocalRaySpillTarget,
    R2RaySpillTarget,
    RaySpillBackend,
    resolve_ray_spill_target,
)


pytestmark = pytest.mark.usefixtures("iris_ray_ports")


def test_concurrent_ray_heads_use_disjoint_scheduler_ports(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **_kwargs: commands.append(command))
    endpoints = []
    for first_port in (12000, 12100):
        allocated = {port: first_port + index for index, port in enumerate(RayPort)}
        for port, value in allocated.items():
            monkeypatch.setenv(f"IRIS_PORT_{port.value.upper()}", str(value))
        runtime.ray_start_head("127.0.0.1", LocalRaySpillTarget(str(tmp_path / str(first_port))))
        flags = dict(argument.split("=", 1) for argument in commands[-1] if "=" in argument)
        ports = {
            "gcs_server_port": int(flags["--port"]),
            "dashboard_port": int(flags["--dashboard-port"]),
            "ray_client_server_port": int(flags["--ray-client-server-port"]),
            "metrics_export_port": int(flags["--metrics-export-port"]),
            "runtime_env_agent_port": int(flags["--runtime-env-agent-port"]),
            "metrics_agent_port": int(flags["--dashboard-agent-grpc-port"]),
            "dashboard_agent_listen_port": int(flags["--dashboard-agent-listen-port"]),
            "node_manager_port": int(flags["--node-manager-port"]),
            "object_manager_port": int(flags["--object-manager-port"]),
        }
        # Iris allocates in Ray's old default worker range. Validate the actual
        # pinned Ray parameters so agents cannot conflict with worker ports.
        parameters = RayParams(
            **ports,
            min_worker_port=int(flags["--min-worker-port"]),
            max_worker_port=int(flags["--max-worker-port"]),
        )
        parameters.update_pre_selected_port()
        assert ports["gcs_server_port"] == allocated[RayPort.HEAD]
        assert set(ports.values()) == set(allocated.values())
        endpoints.append(set(ports.values()))
    assert endpoints[0].isdisjoint(endpoints[1])


def test_remote_spilling_requires_explicit_opt_in():
    local = resolve_ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.LOCAL,
        DEFAULT_RAY_SPILL_DIR,
    )

    remote = resolve_ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.R2,
        DEFAULT_RAY_SPILL_DIR,
    )

    assert local == LocalRaySpillTarget(DEFAULT_RAY_SPILL_DIR)
    assert remote == R2RaySpillTarget("s3://marin-us-east-02a/iris/rl-rdv/job/ray_spill")


def test_remote_spilling_fails_when_backend_dependency_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)

    target = resolve_ray_spill_target(
        "s3://marin-us-east-02a/iris/rl-rdv/job",
        RaySpillBackend.R2,
        DEFAULT_RAY_SPILL_DIR,
    )

    with pytest.raises(RuntimeError, match="requires boto3"):
        target.prepare_node()


@pytest.mark.parametrize("role", ["head", "worker"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_local_spill_directory_exists_before_ray_start(tmp_path, monkeypatch, role, preexisting):
    spill_dir = tmp_path / "node-scratch" / "ray-spill"
    if preexisting:
        spill_dir.mkdir(parents=True)
    directory_state_at_start = []

    def observe_ray_start(_command, **_kwargs):
        directory_state_at_start.append(spill_dir.is_dir())

    monkeypatch.setattr(runtime.subprocess, "run", observe_ray_start)
    target = LocalRaySpillTarget(str(spill_dir))

    if role == "head":
        runtime.ray_start_head("10.0.0.1", target)
    else:
        runtime.ray_start_worker("10.0.0.1", 6379, "10.0.0.2", target)

    assert directory_state_at_start == [True]


def test_local_spill_directory_creation_failure_names_configured_path(tmp_path, monkeypatch):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file blocks directory creation")
    spill_dir = blocked_parent / "ray-spill"

    def reject_ray_start(_command, **_kwargs):
        pytest.fail("ray start was invoked before its spill directory was prepared")

    monkeypatch.setattr(runtime.subprocess, "run", reject_ray_start)

    with pytest.raises(RuntimeError, match=re.escape(str(spill_dir))):
        runtime.ray_start_worker(
            "10.0.0.1",
            6379,
            "10.0.0.2",
            LocalRaySpillTarget(str(spill_dir)),
        )
