# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone execution gateway and its operator manifests."""

import asyncio
from pathlib import Path

import click
import uvicorn
import yaml
from rigging.token_authority import JwksVerifier

from iris.cli.connect import rpc_client_for_ctx
from iris.cluster.exec_gateway.auth import exec_audience
from iris.cluster.exec_gateway.main import gateway_application
from iris.cluster.exec_gateway.manifests import GATEWAY_PORT, gateway_manifests
from iris.cluster.exec_gateway.service import MAX_CHUNK_BYTES, ExecGatewayService
from iris.rpc import controller_pb2, exec_gateway_pb2
from iris.rpc.exec_gateway_connect import ExecGatewayServiceClient


@click.group("exec-gateway")
def exec_gateway():
    """Serve persistent execution separately from the controller."""


@exec_gateway.command("serve")
@click.option("--namespace", required=True)
@click.option("--gateway-id", required=True)
@click.option("--issuer", required=True)
@click.option("--public-key", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--kubeconfig", type=click.Path(exists=True))
@click.option("--context")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=GATEWAY_PORT, show_default=True)
def serve(
    namespace: str,
    gateway_id: str,
    issuer: str,
    public_key: Path,
    kubeconfig: str | None,
    context: str | None,
    host: str,
    port: int,
):
    """Serve Connect RPC with an operator-pinned public verification key."""
    # Keep optional Kubernetes dependencies off the general CLI import path.
    from iris.cluster.exec_gateway.kubernetes import KubernetesExecConnector  # noqa: PLC0415
    from iris.cluster.platforms.k8s.service import CloudK8sService  # noqa: PLC0415

    verifier = JwksVerifier(
        issuers={issuer: [public_key.read_text()]},
        expected_audiences=frozenset({exec_audience(gateway_id)}),
        leeway_seconds=0,
    )
    connector = KubernetesExecConnector(CloudK8sService(namespace, kubeconfig_path=kubeconfig, context=context))
    uvicorn.run(
        gateway_application(ExecGatewayService(verifier, connector)),
        host=host,
        port=port,
        limit_concurrency=1024,
        timeout_keep_alive=30,
    )


@exec_gateway.command("manifests")
@click.option("--namespace", required=True)
@click.option("--image", required=True)
@click.option("--gateway-id", required=True)
@click.option("--issuer", required=True)
@click.option("--public-key", type=click.Path(exists=True, path_type=Path), required=True)
def manifests(namespace: str, image: str, gateway_id: str, issuer: str, public_key: Path):
    """Print Kubernetes manifests for operator review and deployment."""
    documents = gateway_manifests(
        namespace=namespace, image=image, gateway_id=gateway_id, issuer=issuer, public_key=public_key.read_text()
    )
    click.echo(yaml.safe_dump_all(documents, sort_keys=False))


async def _probe_session(address: str, grant: str) -> None:
    payload = bytes(range(256)) * 1024
    async with ExecGatewayServiceClient(address=address) as client:
        opened = await client.open(exec_gateway_pb2.OpenRequest(grant=grant))
        try:
            for offset in range(0, len(payload), MAX_CHUNK_BYTES):
                await client.send(
                    exec_gateway_pb2.SendRequest(
                        grant=grant,
                        session_id=opened.session_id,
                        stdin=payload[offset : offset + MAX_CHUNK_BYTES],
                    )
                )
            await client.send(exec_gateway_pb2.SendRequest(grant=grant, session_id=opened.session_id, eof=True))
            stdout, stderr = bytearray(), bytearray()
            async with asyncio.timeout(60):
                while True:
                    result = await client.receive(
                        exec_gateway_pb2.ReceiveRequest(
                            grant=grant,
                            session_id=opened.session_id,
                            wait_ms=20_000,
                        )
                    )
                    stdout.extend(result.stdout)
                    stderr.extend(result.stderr)
                    if result.exited:
                        break
            if bytes(stdout) != payload or bytes(stderr) != b"gateway-probe" or result.exit_code != 23 or result.error:
                raise click.ClickException(f"Execution probe failed: exit={result.exit_code}, error={result.error!r}")
        finally:
            await client.close_session(exec_gateway_pb2.CloseRequest(grant=grant, session_id=opened.session_id))
    click.echo("Execution gateway: 262144 binary bytes round-tripped; stderr and exit status verified")


@exec_gateway.command("probe")
@click.option("--task-id", required=True)
@click.option("--container-name", default="task", show_default=True)
@click.option("--python", "python_command", default="python3", show_default=True)
@click.pass_context
def probe(ctx: click.Context, task_id: str, container_name: str, python_command: str):
    """Prove a deployed gateway against an existing running Python container."""
    command = [
        python_command,
        "-u",
        "-c",
        "import sys; data=sys.stdin.buffer.read(); "
        "sys.stdout.buffer.write(data); sys.stdout.buffer.flush(); "
        "sys.stderr.buffer.write(b'gateway-probe'); sys.exit(23)",
    ]
    with rpc_client_for_ctx(ctx) as controller:
        response = controller.open_exec_session(
            controller_pb2.Controller.OpenExecSessionRequest(
                task_id=task_id,
                container_name=container_name,
                command=command,
            )
        )
    asyncio.run(_probe_session(response.gateway_address, response.grant))
