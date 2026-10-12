# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from iris.cli import cluster as cluster_cli
from iris.cluster.config import (
    AuthConfig,
    ControllerVmConfig,
    CoreweaveControllerConfig,
    IapAuthConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    UserBudgetTier,
)
from iris.rpc import job_pb2
from rigging.auth import StaticTokenProvider
from rigging.credentials import ClientCredentials
from rigging.timing import Duration, ExponentialBackoff


@pytest.mark.parametrize(
    ("command", "directory", "signing_key", "auth", "budgets"),
    [
        (command, directory, signing_key, AuthConfig(user_roles={"principal:human-999": "user"}), [])
        for command in (["start"], ["controller", "restart"], ["restart"])
        for directory, signing_key in (
            (None, ""),
            ("not-json", ""),
            ('{"human-998":"private@example.com"}', ""),
            ("missing-pod-projection", ""),
            ('{"human-999":"private@example.com"}', "unused-inline-key"),
        )
    ]
    + [
        (
            ["start"],
            '{"human-998":"private@example.com"}',
            "",
            AuthConfig(allowed_submitters=["principal:human-999"]),
            [],
        ),
        (
            ["start"],
            '{"human-998":"private@example.com"}',
            "",
            AuthConfig(iap=IapAuthConfig(signed_header_audience="/projects/1/global/backendServices/2")),
            [UserBudgetTier(user_ids=["principal:human-999"], max_band=job_pb2.PRIORITY_BAND_BATCH)],
        ),
    ],
)
def test_controller_deploy_with_invalid_auth_leaves_infrastructure_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    directory: str | None,
    signing_key: str,
    auth: AuthConfig,
    budgets: list[UserBudgetTier],
) -> None:
    activity = tmp_path / "infrastructure-requests"
    auth = auth.model_copy(update={"principal_directory": "env:TEST_PRINCIPAL_DIRECTORY", "signing_key": signing_key})
    if directory == "missing-pod-projection":
        directory = '{"human-999":"private@example.com"}'
        controller = ControllerVmConfig(coreweave=CoreweaveControllerConfig())
    else:
        controller = ControllerVmConfig()
    if directory is None:
        monkeypatch.delenv("TEST_PRINCIPAL_DIRECTORY", raising=False)
    else:
        monkeypatch.setenv("TEST_PRINCIPAL_DIRECTORY", directory)
    config = IrisClusterConfig(
        auth=auth, user_budgets=budgets, controller=controller, kubernetes_provider=KubernetesProviderConfig()
    )
    real_run = cluster_cli.subprocess.run

    class ControllerProvider:
        def preflight_controller(self, _config):
            activity.write_text("cloud preflight")

        def start_controller(self, _config, **_kwargs):
            activity.write_text("cloud deployment")
            return "https://controller.example"

        restart_controller = start_controller

        def stop_all(self, _config, **_kwargs):
            activity.write_text("cluster deletion")
            return []

        def shutdown(self):
            pass

    def run(command, **_kwargs):
        if command[0] == "git":
            return real_run(command, **_kwargs)
        activity.write_text(str(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))
    monkeypatch.setattr(cluster_cli.subprocess, "run", run)
    result = CliRunner().invoke(cluster_cli.cluster, command, obj={"config": config})

    assert result.exit_code == 1
    assert not activity.exists()
    assert "private@example.com" not in result.output
    assert "unused-inline-key" not in result.output


@pytest.mark.requires_cluster
@pytest.mark.parametrize("local_flag", [True, False], ids=["converted", "already-local"])
@pytest.mark.parametrize(
    "auth_yaml",
    [
        "  signing_key: unused-inline-key\n",
        "  principal_directory: env:TEST_PRINCIPAL_DIRECTORY\n"
        "  signing_key: env:TEST_UNUSED_SIGNING_KEY\n"
        "  user_roles: {principal:human-999: user}\n",
    ],
    ids=["inline-key-without-principals", "principal-directory"],
)
def test_local_start_uses_ephemeral_key_and_resolves_personal_policy(
    tmp_path: Path, local_flag: bool, auth_yaml: str
) -> None:
    config = tmp_path / "cluster.yaml"
    config.write_text(
        "name: personal-local\nauth:\n" + auth_yaml + ("controller: {local: {}}\n" if not local_flag else "")
    )
    credentials = tmp_path / "credentials"
    # Isolate the credential-store filesystem boundary while running the real CLI.
    entrypoint = (
        "from pathlib import Path; import sys; from rigging import credential_store; "
        "credential_store._CREDENTIALS_DIR = Path(sys.argv.pop(1)); "
        "from iris.cli.main import main; main()"
    )
    command = [sys.executable, "-c", entrypoint, str(credentials), "--config", str(config), "cluster", "start"]
    if local_flag:
        command.append("--local")
    env = dict(os.environ, TEST_PRINCIPAL_DIRECTORY='{"human-999":"private@example.com"}')
    env.pop("TEST_UNUSED_SIGNING_KEY", None)
    log = tmp_path / "controller.log"
    with log.open("w") as output, subprocess.Popen(command, stdout=output, stderr=output, env=env) as process:

        def started() -> bool:
            if process.poll() is not None:
                return True
            record = credentials / "personal-local.json"
            if not record.exists():
                return False
            endpoint = json.loads(record.read_text())["endpoint"]
            return httpx.get(f"{endpoint}/health", timeout=1).status_code == 200

        try:
            assert ExponentialBackoff(initial=0.02, maximum=0.2).wait_until(
                started, timeout=Duration.from_seconds(15)
            ), log.read_text()
            assert process.poll() is None, log.read_text()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        assert process.returncode == 0, log.read_text()


@pytest.fixture
def authenticated_upstream() -> Iterator[tuple[str, list[str | None]]]:
    proxy_authorizations: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            proxy_authorizations.append(self.headers.get("Proxy-Authorization"))
            body = json.dumps({"enabled": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", proxy_authorizations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_proxy_sends_context_credentials_to_upstream(
    monkeypatch: pytest.MonkeyPatch,
    authenticated_upstream: tuple[str, list[str | None]],
) -> None:
    upstream_url, proxy_authorizations = authenticated_upstream
    credentials = ClientCredentials(iap_provider=StaticTokenProvider("iap-token"))

    monkeypatch.setattr(cluster_cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cluster_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(terminate=lambda: None, wait=lambda: None),
    )

    async def request_through_proxy(app) -> int:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
                return (await client.get("/auth/config")).status_code

    def serve_proxy(app, **_kwargs) -> None:
        assert asyncio.run(request_through_proxy(app)) == 200

    monkeypatch.setattr(cluster_cli.uvicorn, "run", serve_proxy)

    result = CliRunner().invoke(
        cluster_cli.cluster_dashboard_proxy,
        obj={"controller_url": upstream_url, "credentials": credentials},
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert proxy_authorizations == ["Bearer iap-token"]
