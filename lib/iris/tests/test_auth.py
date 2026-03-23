# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Auth tests for Iris controller with static token authentication."""

from pathlib import Path

import pytest
from iris.cluster.config import load_config, make_local_config
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

IRIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"

_AUTH_TOKEN = "e2e-test-token"
_AUTH_USER = "test-user"


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["placeholder"]
    sg.name = "placeholder"
    sg.num_vms = 1
    sg.min_slices = 0
    sg.max_slices = 0
    sg.resources.cpu_millicores = 1000
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


def _login_for_jwt(url: str, identity_token: str) -> str:
    """Exchange a raw identity token for a JWT via the Login RPC."""
    client = ControllerServiceClientSync(address=url, timeout_ms=10000)
    try:
        resp = client.login(cluster_pb2.LoginRequest(identity_token=identity_token))
        return resp.token
    finally:
        client.close()


def _quick():
    return 1


def test_static_auth_rpc_access():
    """Static auth rejects unauthenticated and wrong-token RPCs, accepts valid JWT."""
    from connectrpc.errors import ConnectError
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        list_req = cluster_pb2.Controller.ListWorkersRequest()

        unauth_client = ControllerServiceClientSync(address=url, timeout_ms=5000)
        with pytest.raises(ConnectError, match=r"(?i)authorization"):
            unauth_client.list_workers(list_req)
        unauth_client.close()

        wrong_injector = AuthTokenInjector(StaticTokenProvider("wrong-token"))
        wrong_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[wrong_injector])
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            wrong_client.list_workers(list_req)
        wrong_client.close()

        jwt_token = _login_for_jwt(url, _AUTH_TOKEN)
        valid_injector = AuthTokenInjector(StaticTokenProvider(jwt_token))
        valid_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[valid_injector])
        response = valid_client.list_workers(list_req)
        assert response is not None
        valid_client.close()
    finally:
        controller.close()


def test_static_auth_job_ownership():
    """Job ownership: user A cannot terminate user B's job."""
    from connectrpc.errors import ConnectError
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    _TOKEN_A = "token-user-a"
    _TOKEN_B = "token-user-b"

    config = _make_controller_only_config()
    config.auth.static.tokens[_TOKEN_A] = "user-a"
    config.auth.static.tokens[_TOKEN_B] = "user-b"
    controller = LocalCluster(config)
    url = controller.start()

    try:
        jwt_a = _login_for_jwt(url, _TOKEN_A)
        jwt_b = _login_for_jwt(url, _TOKEN_B)

        injector_a = AuthTokenInjector(StaticTokenProvider(jwt_a))
        client_a = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_a])

        entrypoint = Entrypoint.from_callable(_quick)
        launch_req = cluster_pb2.Controller.LaunchJobRequest(
            name="/user-a/auth-owned-job",
            entrypoint=entrypoint.to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
        )
        resp = client_a.launch_job(launch_req)
        job_id = resp.job_id

        injector_b = AuthTokenInjector(StaticTokenProvider(jwt_b))
        client_b = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_b])
        with pytest.raises(ConnectError, match="cannot access resources owned by"):
            client_b.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.terminate_job(cluster_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.close()
        client_b.close()
    finally:
        controller.close()
