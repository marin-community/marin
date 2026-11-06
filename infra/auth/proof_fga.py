# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a real OpenFGA engine for the isolated grant proof."""

import json
import os
import secrets
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from infra.auth.fga_operator import OpenFGAOperator
from infra.auth.openfga import scope_object


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def isolated_openfga(executable: Path, directory: Path):
    api_url = f"http://127.0.0.1:{free_port()}"
    token_file = directory / "openfga.token"
    token_file.touch(mode=0o600)
    token_file.write_text(secrets.token_urlsafe(32))
    env = {**os.environ, "OPENFGA_AUTHN_PRESHARED_KEYS": token_file.read_text()}
    command = [
        str(executable),
        "run",
        "--datastore-engine",
        "memory",
        "--http-addr",
        api_url.removeprefix("http://"),
        "--grpc-addr",
        f"127.0.0.1:{free_port()}",
        "--metrics-enabled=false",
        "--playground-enabled=false",
        "--authn-method=preshared",
    ]
    with (directory / "openfga.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"OpenFGA exited: {(directory / 'openfga.log').read_text()}")
                try:
                    if requests.get(api_url + "/healthz", timeout=1).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("OpenFGA did not become ready")
                time.sleep(0.1)
            yield OpenFGAOperator(api_url, token_file)
        finally:
            process.terminate()
            process.wait(timeout=15)


def configure_proof(operator: OpenFGAOperator) -> dict:
    directory = Path(__file__).parent / "policy"
    store_id, model_id = operator.publish("marin-auth-proof", json.loads((directory / "model.json").read_text()))
    relationships = [
        {"user": "principal:launcher", "relation": "member", "object": "group:marin-team"},
        {"user": "group:marin-team#member", "relation": "member", "object": "group:marin-research"},
        *(
            {
                "user": "group:marin-research#member",
                "relation": "accessor",
                "object": scope_object(service, action, resource),
            }
            for service, action, resource in [
                ("loom", "session.create", "/auth-proof/"),
                ("iris", "job.submit", "/proof/"),
                ("iris", "job.read", "/proof/"),
                ("iris", "job.cancel", "/proof/"),
                ("iris", "job.submit", "/clusters/isolated-proof/jobs/proof/"),
                ("iris", "job.read", "/clusters/isolated-proof/jobs/proof/"),
                ("iris", "job.cancel", "/clusters/isolated-proof/jobs/proof/"),
                ("marina", "app.read", "/apps/"),
                ("marina", "app.release", "/apps/"),
                ("finelog", "logs.query", "/instances/proof/"),
                ("cw_s3", "credentials.issue", "diagnostics"),
            ]
        ),
        {
            "user": "group:marin-research#member",
            "relation": "blocked",
            "object": scope_object("iris", "job.submit", "/proof/private/"),
        },
        {
            "user": "group:marin-research#member",
            "relation": "blocked",
            "object": scope_object("iris", "job.submit", "/clusters/isolated-proof/jobs/proof/private/"),
        },
    ]
    operator.reconcile(store_id, model_id, relationships)
    return {
        "api_url": operator.api_url,
        "store_id": store_id,
        "model_id": model_id,
        "token_file": operator.token_file,
        "principals": [
            {"principal": "launcher", "owner": "proof-owner", "max_ttl": 3600, "delegation_depth": 2},
            {"principal": "other-session", "owner": "other-owner", "max_ttl": 3600, "delegation_depth": 0},
        ],
    }
