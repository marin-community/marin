# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real adapter against an external Kubernetes API boundary."""

import asyncio
import socket
import threading
from contextlib import asynccontextmanager

import pytest
import uvicorn
import yaml
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.exec_gateway.auth import ExecPodTarget
from iris.cluster.exec_gateway.kubernetes import KubernetesExecConnector
from iris.cluster.platforms.k8s.service import CloudK8sService
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute


@asynccontextmanager
async def kubernetes_api(tmp_path, *, actual_uid="pod-uid", replace_on_connect=False):
    observations = []
    uid = actual_uid

    async def read_pod(request):
        observations.append(("read", request.headers.get("authorization")))
        return JSONResponse(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": "pod", "uid": uid, "labels": {"iris.attempt_uid": "attempt"}},
                "spec": {"containers": [{"name": "task", "image": "python"}]},
                "status": {
                    "containerStatuses": [
                        {
                            "name": "task",
                            "image": "python",
                            "imageID": "python",
                            "ready": True,
                            "restartCount": 0,
                            "state": {"running": {}},
                        }
                    ]
                },
            }
        )

    async def execute(websocket):
        nonlocal uid
        observations.append(("exec", list(websocket.query_params.getlist("command"))))
        await websocket.accept(subprotocol="v5.channel.k8s.io")
        if replace_on_connect:
            uid = "replacement-pod"
        while True:
            event = await websocket.receive()
            if event["type"] == "websocket.disconnect":
                return
            frame = event["bytes"]
            if frame == b"\xff\x00":
                await websocket.send_bytes(b'\x03{"status":"Success"}')
                await websocket.close()
                return
            await websocket.send_bytes(b"\x01" + frame[1:])
            await websocket.send_bytes(b"\x02err")

    app = Starlette(
        routes=[
            Route("/api/v1/namespaces/{namespace}/pods/{pod}", read_pod),
            WebSocketRoute("/api/v1/namespaces/{namespace}/pods/{pod}/exec", execute),
        ]
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        address = f"http://127.0.0.1:{listener.getsockname()[1]}"
        config_path = tmp_path / "kubeconfig.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "apiVersion": "v1",
                    "kind": "Config",
                    "current-context": "test",
                    "clusters": [{"name": "test", "cluster": {"server": address}}],
                    "contexts": [{"name": "test", "context": {"cluster": "test", "user": "test"}}],
                    "users": [{"name": "test", "user": {"token": "gateway-service-account"}}],
                }
            )
        )
        server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="off"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        connector = KubernetesExecConnector(CloudK8sService("iris", kubeconfig_path=str(config_path)))
        try:
            yield connector, observations
        finally:
            await connector.close()
            server.should_exit = True
            await serving


@pytest.mark.asyncio
async def test_kubernetes_exec_uses_incremental_binary_v5_channels_and_pinned_pod(tmp_path):
    async with kubernetes_api(tmp_path) as (connector, observations):
        transport = await connector.open(
            ExecPodTarget("iris", "pod", "pod-uid", "task", "attempt"), ("python", "-u", "two words")
        )
        try:
            await transport.send(b"\x00\xff", eof=False)
            assert (await transport.receive()).stdout == b"\x00\xff"
            assert (await transport.receive()).stderr == b"err"
            await transport.send(b"", eof=True)
            assert (await transport.receive()).exit_code == 0
        finally:
            await transport.close()
    assert observations == [
        ("read", "Bearer gateway-service-account"),
        ("exec", ["python", "-u", "two words"]),
        ("read", "Bearer gateway-service-account"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_on_connect", [False, True])
async def test_kubernetes_exec_rejects_replaced_pod_before_or_during_connection(tmp_path, replace_on_connect):
    async with kubernetes_api(
        tmp_path, actual_uid="pod-uid" if replace_on_connect else "replacement", replace_on_connect=replace_on_connect
    ) as (connector, observations):
        with pytest.raises(ConnectError) as rejected:
            await connector.open(ExecPodTarget("iris", "pod", "pod-uid", "task", "attempt"), ("python",))
        assert rejected.value.code == Code.FAILED_PRECONDITION
        assert any(observation[0] == "exec" for observation in observations) == replace_on_connect


@pytest.mark.asyncio
async def test_kubernetes_exec_multiplexes_32_idle_streams_with_bounded_setup_threads(tmp_path):
    baseline_threads = threading.active_count()
    async with kubernetes_api(tmp_path) as (connector, _):
        target = ExecPodTarget("iris", "pod", "pod-uid", "task", "attempt")
        transports = await asyncio.gather(*(connector.open(target, ("python",)) for _ in range(32)))
        try:
            assert threading.active_count() - baseline_threads <= 8
            # All streams are idle until the last receives data. Its output must
            # progress without waiting for any of the other 31 streams.
            async with asyncio.timeout(2):
                await transports[-1].send(b"ready", eof=False)
                assert (await transports[-1].receive()).stdout == b"ready"
        finally:
            await asyncio.gather(*(transport.close() for transport in transports))
