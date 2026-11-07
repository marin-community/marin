# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated receiving-service proof; synthetic data and no external messaging.

Iris runs its real listener, controller, and database with an execution backend
that has no workers. This proves job admission and cancellation, not Kubernetes
task execution. Finelog query authority covers the entire disposable instance.
"""

import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pyarrow.ipc as paipc
import requests
from config import LoomAlertConfig, SharedLoomAuth, SlackAlertConfig
from iris.cluster.config import AuthConfig
from iris.cluster.controller.auth import create_controller_auth
from iris.managed_thread import thread_container_scope
from iris.testing.controller import controller_factory
from iris.version import client_revision_date
from loom_alerts import LoomAlertClient, OperatorBehavior
from marina.applets import AppletStore, package_applet, read_applet_package
from marina.database_setup import ensure_applet_provisioning
from marina.db import UrlDatabase
from marina.server import MCP_PATH, MarinaConfig, create_app
from rigging.grant_auth import JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.shared_grants import SharedGrantConfig

from infra.auth.proof_fga import free_port
from infra.auth.proof_http import serve

APP_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
OTHER_APP_ID = uuid.UUID("00000000-0000-4000-8000-000000000002")
MEMBERSHIP = {"user": "group:marin-team#member", "relation": "member", "object": "group:marin-research"}


def rpc(url: str, method: str, payload: dict, token: str | None = None) -> requests.Response:
    return requests.post(
        url + method,
        json=payload,
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=15,
        allow_redirects=False,
    )


@contextmanager
def postgres(binary_dir: Path, directory: Path):
    directory.mkdir()
    data = directory / "data"
    port = free_port()
    subprocess.run(
        [str(binary_dir / "initdb"), "-D", str(data), "-U", "proof", "--auth=trust", "--no-locale"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(binary_dir / "pg_ctl"),
            "-D",
            str(data),
            "-l",
            str(directory / "postgres.log"),
            "-o",
            f"-p {port} -h 127.0.0.1 -k {directory}",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
    )
    try:
        yield UrlDatabase(f"postgresql+pg8000://proof@127.0.0.1:{port}/postgres")
    finally:
        subprocess.run(
            [str(binary_dir / "pg_ctl"), "-D", str(data), "-m", "immediate", "-w", "stop"],
            check=True,
            capture_output=True,
        )


@contextmanager
def iris(directory: Path, trust: SharedGrantConfig | None):
    directory.mkdir()
    with thread_container_scope():
        factory = controller_factory(directory)
        make = next(factory)
        try:
            auth = create_controller_auth(
                AuthConfig(shared_grants=trust) if trust else None, cluster_name="isolated-proof"
            )
            controller = make(auth=auth, host="127.0.0.1", port=0)
            controller.start()
            yield controller.url
        finally:
            next(factory, None)


@contextmanager
def finelog(executable: Path, directory: Path, trust: SharedGrantConfig | None):
    directory.mkdir(exist_ok=True)
    port = free_port()
    env = {name: os.environ[name] for name in ("PATH", "HOME", "LANG") if name in os.environ}
    command = [str(executable), "--port", str(port), "--log-dir", str(directory / "data")]
    if trust is not None:
        command += [
            "--auth-policy",
            json.dumps([{"type": "shared_grants", "trust": trust.model_dump(), "instance": "proof"}]),
        ]
    with (directory / "server.log").open("a") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Finelog exited: {(directory / 'server.log').read_text()}")
                try:
                    if requests.get(url + "/health", timeout=1).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Private Finelog did not start")
                time.sleep(0.05)
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)


class AlertTransport(httpx.AsyncBaseTransport):
    """Fake only Slack; all grant and Loom traffic must use real loopback APIs."""

    def __init__(self):
        self.local = httpx.AsyncHTTPTransport()
        self.messages = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "slack.com":
            self.messages += 1
            return httpx.Response(200, json={"ok": True, "ts": f"1700000000.{self.messages:06d}"})
        if request.url.host != "127.0.0.1":
            raise RuntimeError("The alert proof attempted external traffic")
        return await self.local.handle_async_request(request)

    async def aclose(self):
        await self.local.aclose()


def grant_token(client: httpx.Client, caller: str, permissions: list[dict]) -> str:
    grant = client.post(
        "/v1/grants",
        headers={"Authorization": f"Bearer {caller}"},
        json={"actor": "launcher", "permissions": permissions, "expires_at": time.time() + 300},
    )
    grant.raise_for_status()
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": TOKEN_EXCHANGE_GRANT,
            "subject_token_type": JWT_TOKEN_TYPE,
            "subject_token": caller,
            "audience": permissions[0]["service"],
            "grant_id": grant.json()["id"],
        },
    )
    token.raise_for_status()
    return token.json()["access_token"]


def permission(service: str, actions: list[str], resource: str) -> dict:
    return {"service": service, "actions": actions, "resource": resource}


def run_service_proof(
    directory: Path,
    *,
    trust_issuer: str,
    public_keys: tuple[str, ...],
    client: httpx.Client,
    caller: str,
    fga,
    fga_config: dict,
    bao_address: str,
    bao_token_file: Path,
    bao_ca: Path,
    postgres_bin: Path,
    finelog_executable: Path,
    loom,
) -> dict:
    directory.mkdir()
    trust = SharedGrantConfig(issuer=trust_issuer, public_keys=public_keys, authority_url=trust_issuer)
    with ExitStack() as stack:
        iris_url = stack.enter_context(iris(directory / "iris", trust))
        local_iris = stack.enter_context(iris(directory / "local-iris", None))
        database = stack.enter_context(postgres(postgres_bin, directory / "postgres"))
        store = AppletStore(database)
        ensure_applet_provisioning(store.engine)
        store.migrate()
        for applet_id, name in [(APP_ID, "auth-demo"), (OTHER_APP_ID, "other-demo")]:
            source = directory / name
            (source / "dist").mkdir(parents=True)
            (source / "applet.toml").write_text(
                f'name = "{name}"\ntitle = "Synthetic auth app"\ndescription = "Disposable authorization proof"\n'
            )
            (source / "dist/index.html").write_text(f"<p>{name}</p>")
            store.publish(read_applet_package(package_applet(source)), "seed", applet_id, 0, frozenset({"seed"}))
        store.close()
        marina_config = MarinaConfig(data_root=str(directory / "marina-data"), iap_audience=None, database=database)
        marina_url = stack.enter_context(serve(create_app(replace(marina_config, shared_grants=trust)), free_port()))
        local_marina = stack.enter_context(serve(create_app(marina_config), free_port()))
        log_dir = directory / "finelog"
        with finelog(finelog_executable, log_dir, None) as seed_url:
            for cluster in ["cluster-a", "cluster-b"]:
                response = rpc(
                    seed_url,
                    "/finelog.logging.LogService/PushLogs",
                    {
                        "key": "auth-proof",
                        "cluster": cluster,
                        "entries": [
                            {"timestamp": {"epochMs": str(int(time.time() * 1000))}, "source": "stdout", "data": cluster}
                        ],
                    },
                )
                response.raise_for_status()
        finelog_url = stack.enter_context(finelog(finelog_executable, log_dir, trust))
        local_finelog = stack.enter_context(finelog(finelog_executable, directory / "local-finelog", None))
        iris_token = grant_token(
            client,
            caller,
            [permission("iris", ["job.submit", "job.read", "job.cancel"], "/clusters/isolated-proof/jobs/proof/")],
        )
        marina_token = grant_token(
            client, caller, [permission("marina", ["app.read", "app.release"], f"/apps/{APP_ID}/")]
        )
        finelog_token = grant_token(client, caller, [permission("finelog", ["logs.query"], "/instances/proof/")])
        job = {
            "name": "/proof/auth-demo",
            "clientRevisionDate": client_revision_date(),
            "containerProfile": "CONTAINER_PROFILE_SANDBOX",
            "taskImage": "registry.invalid/auth-proof:synthetic",
            "resources": {"cpuMillicores": 1000, "memoryBytes": "134217728"},
            "entrypoint": {"runCommand": {"argv": ["true"]}},
            "replicas": 1,
        }
        launch = rpc(iris_url, "/iris.cluster.ControllerService/LaunchJob", job, iris_token)
        assert launch.status_code == 200, launch.text
        assert launch.json()["jobId"] == job["name"]
        assert (
            rpc(
                iris_url, "/iris.cluster.ControllerService/LaunchJob", {**job, "name": "/other/forbidden"}, iris_token
            ).status_code
            == 403
        )
        assert (
            rpc(
                iris_url,
                "/iris.cluster.ControllerService/LaunchJob",
                {**job, "name": "/proof/unsafe", "containerProfile": "CONTAINER_PROFILE_DEFAULT"},
                iris_token,
            ).status_code
            == 403
        )
        assert rpc(iris_url, "/iris.cluster.ControllerService/ListWorkers", {}, iris_token).status_code == 403
        assert (
            rpc(
                iris_url, "/iris.cluster.ControllerService/LaunchJob", {**job, "name": "/proof/private/job"}, iris_token
            ).status_code
            == 403
        )
        assert rpc(iris_url, "/iris.cluster.ControllerService/GetJobStatus", {"jobId": job["name"]}).status_code == 401
        marina_headers = {"Authorization": f"Bearer {marina_token}"}
        assert requests.get(marina_url + "/auth-demo/", headers=marina_headers, timeout=10).status_code == 200
        assert requests.get(marina_url + "/other-demo/", headers=marina_headers, timeout=10).status_code == 403
        apps = requests.get(marina_url + "/api/marina/apps", headers=marina_headers, timeout=10).json()["apps"]
        assert [app["name"] for app in apps] == ["auth-demo"]
        assert requests.get(marina_url + "/auth-demo/", timeout=10).status_code == 401
        assert requests.post(marina_url + MCP_PATH + "/", headers=marina_headers, timeout=10).status_code == 403
        updated = requests.post(
            marina_url + f"/api/marina/applets/{APP_ID}",
            params={"base_version": 1},
            headers=marina_headers,
            data=package_applet(directory / "auth-demo"),
            timeout=10,
        )
        assert updated.status_code == 201, updated.text
        query = {"sql": "SELECT DISTINCT cluster FROM log ORDER BY cluster"}
        queried = rpc(finelog_url, "/finelog.stats.StatsService/Query", query, finelog_token)
        assert queried.status_code == 200, queried.text
        rows = paipc.open_stream(base64.b64decode(queried.json()["arrowIpc"])).read_all().to_pylist()
        assert [row["cluster"] for row in rows] == ["cluster-a", "cluster-b"]
        assert rpc(finelog_url, "/finelog.stats.StatsService/Query", query).status_code == 401
        cluster_token = grant_token(
            client, caller, [permission("finelog", ["logs.query"], "/instances/proof/clusters/cluster-a/")]
        )
        assert rpc(finelog_url, "/finelog.stats.StatsService/Query", query, cluster_token).status_code == 403
        assert (
            rpc(finelog_url, "/finelog.logging.LogService/PushLogs", {"key": "forbidden"}, finelog_token).status_code
            == 403
        )

        checks = [
            lambda: rpc(iris_url, "/iris.cluster.ControllerService/GetJobStatus", {"jobId": job["name"]}, iris_token),
            lambda: requests.get(marina_url + "/auth-demo/", headers=marina_headers, timeout=10),
            lambda: rpc(finelog_url, "/finelog.stats.StatsService/Query", query, finelog_token),
        ]
        assert [check().status_code for check in checks] == [200, 200, 200]
        fga.write(fga_config["store_id"], fga_config["model_id"], writes=[], deletes=[MEMBERSHIP])
        assert [check().status_code for check in checks] == [403, 403, 403]
        fga.write(fga_config["store_id"], fga_config["model_id"], writes=[MEMBERSHIP], deletes=[])
        assert [check().status_code for check in checks] == [200, 200, 200]
        credential_path = Path(fga_config["token_file"])
        credential = credential_path.read_text()
        try:
            credential_path.write_text("synthetic-invalid-check-credential")
            assert [check().status_code for check in checks] == [503, 503, 503]
            assert rpc(local_iris, "/iris.cluster.ControllerService/ListJobs", {}).status_code == 200
            assert requests.get(local_marina + "/auth-demo/", timeout=10).status_code == 200
            assert (
                rpc(local_finelog, "/finelog.stats.StatsService/Query", {"sql": "SELECT 1 AS value"}).status_code == 200
            )
        finally:
            credential_path.write_text(credential)
        assert [check().status_code for check in checks] == [200, 200, 200]
        assert (
            rpc(iris_url, "/iris.cluster.ControllerService/TerminateJob", {"jobId": job["name"]}, iris_token).status_code
            == 200
        )
        alert_config = LoomAlertConfig(
            url=loom.base_url,
            profile="auth-proof",
            repository="synthetic",
            http_timeout=20,
            slack=SlackAlertConfig(bot_token="synthetic-unused-slack", channel="synthetic"),
            shared_auth=SharedLoomAuth(
                authority_url=trust_issuer,
                bao_address=bao_address,
                bao_token_file=bao_token_file,
                ca_bundle=str(bao_ca),
                actor="launcher",
            ),
        )
        alerts = LoomAlertClient(
            alert_config,
            behaviors=[
                OperatorBehavior(
                    name="default",
                    channel="operator",
                    session_title="Synthetic alert",
                    operator_name="Synthetic operator",
                )
            ],
            transport=AlertTransport(),
        )
        payload = {
            "receiver": "auth-proof",
            "status": "firing",
            "groupKey": "auth-proof",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {"alertname": "SyntheticAuthProof"},
                    "annotations": {},
                    "startsAt": "2026-11-06T00:00:00Z",
                    "fingerprint": "synthetic",
                }
            ],
        }
        run = asyncio.run(alerts.submit(payload))
        retry_alerts = LoomAlertClient(
            alert_config,
            behaviors=[
                OperatorBehavior(
                    name="default",
                    channel="operator",
                    session_title="Synthetic alert",
                    operator_name="Synthetic operator",
                )
            ],
            transport=AlertTransport(),
        )
        retry = asyncio.run(retry_alerts.submit(payload))
        assert run is not None and retry == run
        assert loom.post("sessions/get", {"session": run["session_id"]})["profile"] == "auth-proof"
        loom.post("sessions/archive", {"session": run["session_id"]})
        return {
            "iris": {
                "job_id": job["name"],
                "result": "native listener -> scoped sandbox admission -> persisted status -> cancellation",
                "execution": "not scheduled; isolated backend has no workers",
            },
            "marina": {
                "app_id": str(APP_ID),
                "result": (
                    "UUID-scoped read and release, named route enforcement, filtered directory, other app and MCP denied"
                ),
            },
            "finelog": {
                "resource": "/instances/proof/",
                "result": "real SQL returns both seeded clusters; unauthenticated reads and grant-based writes denied",
                "scope": "whole instance, no per-cluster SQL filtering",
            },
            "grafana": {
                "run_id": run["id"],
                "session_id": run["session_id"],
                "result": "real OpenBao -> idempotent grant -> native Loom; retry returns same run",
                "slack": "fake HTTP endpoint; no external messages",
            },
            "revocation": (
                "removing nested group edge denies existing Iris/Marina/Finelog JWTs; restoring edge restores access"
            ),
            "availability": (
                "failed OpenFGA credential produces 503; recovery preserves grants; local backends remain usable"
            ),
        }
