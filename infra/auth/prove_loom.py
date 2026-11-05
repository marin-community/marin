# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repeatable Loom launch proof using a private server and scripted ACP agent."""

import json
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import quote

import httpx
import requests
from rigging.grants import Grant

from infra.auth.loom import LoomGrantService


@dataclass(frozen=True)
class IsolatedLoom:
    base_url: str
    observer_token_file: Path
    repo: Path
    storage_marker: Path | None
    session: requests.Session = field(repr=False)

    def post(self, operation: str, payload: dict):
        response = self.session.post(
            f"{self.base_url}/api/{operation}",
            json=payload,
            headers={"Authorization": f"Bearer {self.observer_token_file.read_text().strip()}"},
            timeout=30,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.HTTPError("Isolated Loom redirected a credential-bearing request", response=response)
        response.raise_for_status()
        return response.json()


@contextmanager
def isolated_loom(
    directory: Path,
    *,
    executable: Path,
    source: Path,
    verify_storage_binding: str | None = None,
):
    """Start the real Loom backend before constructing its lifecycle observer."""
    directory.mkdir(parents=True, exist_ok=False)
    home = directory / "home"
    repo = directory / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Auth proof",
            "-c",
            "user.email=auth@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "Initialize isolated auth proof",
        ],
        check=True,
    )
    fixture = source / "crates/loom/tests/fixtures/fake-acp-agent.mjs"
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    marker = directory / "storage-proof.json"
    command = f"node {shlex.quote(str(fixture))}"
    if verify_storage_binding is not None:
        wrapper = directory / "credential-agent.mjs"
        wrapper.write_text(
            "import {spawn} from 'node:child_process';\n"
            "import {createInterface} from 'node:readline';\n"
            "import {writeFileSync} from 'node:fs';\n"
            f"const child = spawn('node', [{json.dumps(str(fixture))}], {{stdio:['pipe','inherit','inherit']}});\n"
            "child.on('exit', code => process.exit(code ?? 1));\n"
            "for await (const line of createInterface({input:process.stdin})) {\n"
            "  const message = JSON.parse(line);\n"
            "  if (message.method === 'session/prompt' && process.env.LOOM_SESSION_ID) {\n"
            "    const response = await fetch(process.env.MARIN_AUTH_URL + "
            f"{json.dumps('/v1/storage/' + quote(verify_storage_binding, safe='') + '/token')}, {{\n"
            "      method:'POST', redirect:'error',\n"
            "      headers:{Authorization:'Bearer ' + process.env.MARIN_GRANT_TOKEN}\n"
            "    });\n"
            "    if (!response.ok) throw new Error('Storage authority request failed: ' + response.status);\n"
            "    const identity = await response.json();\n"
            "    if (!identity.token || !identity.expires_at) throw new Error('Missing native identity');\n"
            f"    writeFileSync({json.dumps(str(marker))}, JSON.stringify({{\n"
            "      status:response.status, expires_at:identity.expires_at, session_id:process.env.LOOM_SESSION_ID\n"
            "    }), {mode:0o600});\n"
            "  }\n"
            "  child.stdin.write(line + '\\n');\n"
            "}\n"
            "child.stdin.end();\n"
        )
        command = f"node {shlex.quote(str(wrapper))}"
    env = {name: os.environ[name] for name in ("PATH", "HOME", "LANG") if name in os.environ}
    env.update(
        WEAVER_HOME=str(home),
        LOOM_OWNER_GITHUB="isolated-auth-proof",
        WEAVER_CLAUDE_ACP_CMD=command,
    )
    log_path = directory / "loom.log"
    with log_path.open("w") as log, requests.Session() as session:
        process = subprocess.Popen(
            [str(executable), "server", "run", "--addr", "127.0.0.1:0"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base_url = ""
        fixture = None
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"isolated Loom exited; see {log_path}")
                listening = [line for line in log_path.read_text().splitlines() if line.startswith("loom listening on ")]
                if listening:
                    base_url = listening[-1].removeprefix("loom listening on ")
                    break
                time.sleep(0.05)
            if not base_url:
                raise TimeoutError(f"isolated Loom did not start; see {log_path}")
            fixture = IsolatedLoom(
                base_url, home / "loom-token", repo, marker if verify_storage_binding is not None else None, session
            )
            observer_token = fixture.post(
                "auth/tokens/create", {"name": "isolated lifecycle observer", "expires_in_days": 1}
            )
            observer_token_file = directory / "observer.token"
            observer_token_file.touch(mode=0o600, exist_ok=False)
            observer_token_file.write_text(observer_token["token"])
            fixture = replace(fixture, observer_token_file=observer_token_file)
            yield fixture
        finally:
            try:
                if fixture is not None:
                    for run in fixture.post("runs/list", {}):
                        if run["session_id"]:
                            try:
                                fixture.post("sessions/archive", {"session": run["session_id"]})
                            except requests.HTTPError as error:
                                if error.response.status_code != 404:
                                    raise
            finally:
                process.terminate()
                process.wait(timeout=10)


def run_loom_proof(
    loom: IsolatedLoom,
    *,
    client: httpx.Client,
    grants: LoomGrantService,
    loom_token: str,
    grant: Grant,
    auth_url: str,
    public_keys: tuple[str, ...],
    template_name: str,
) -> dict[str, str | bool]:
    """Exercise the real broker launch, native callback, and background cleanup."""
    stock = loom.post("profiles/get", {"name": "github_comment"})
    excluded = {"env", "revision", "lifetime", "created_at", "updated_at"}
    profile = {key: value for key, value in stock.items() if key not in excluded}
    profile.update(name=template_name, model="fake-fast", mcp_access={"mode": "none", "groups": []})
    loom.post("profiles/create", profile)
    preview = loom.post("sessions/launches/resolve", {"selection": {"profile": template_name, "overrides": {}}})
    loom.post(
        "settings/patch",
        {
            "changes": {
                "auth.trust_loopback": False,
                "auth.shared_grants": json.dumps(
                    {
                        "issuer": auth_url,
                        "public_keys": list(public_keys),
                        "authorize_url": auth_url + "/v1/authorize",
                        "credentials_url": auth_url + "/v1/loom/credentials",
                        "profiles": {
                            template_name: {
                                "cwd": str(loom.repo),
                                "revision": preview["profile_revision"],
                                "resolver_revision": preview["resolver_revision"],
                            }
                        },
                    }
                ),
            }
        },
    )
    response = client.post(
        "/v1/loom/launch",
        headers={"Authorization": f"Bearer {loom_token}"},
        json={"template": template_name, "title": "Isolated grant proof", "goal": "say:grant launch worked"},
    )
    response.raise_for_status()
    run = response.json()
    if run["status"] != "running" or not run["run_id"] or not run["session_id"] or run["grant_id"] != grant.id:
        raise RuntimeError("broker did not return the actual grant-bound Loom run")
    observed = loom.post("sessions/get", {"session": run["session_id"]})
    if observed["profile"] != template_name:
        raise RuntimeError("Loom session did not retain its approved profile")
    if loom.storage_marker is not None:
        deadline = time.monotonic() + 30
        while not loom.storage_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not loom.storage_marker.exists():
            raise RuntimeError("Loom session did not obtain a native storage identity")
        consumed = json.loads(loom.storage_marker.read_text())
        if consumed["status"] != 200 or consumed["session_id"] != run["session_id"]:
            raise RuntimeError("native identity was not obtained by the actual Loom session")
    loom.post("sessions/archive", {"session": run["session_id"]})
    deadline = time.monotonic() + 10
    while grants.active(grant.id, grant.actor) and time.monotonic() < deadline:
        time.sleep(0.05)
    if grants.active(grant.id, grant.actor):
        raise RuntimeError("the broker lifecycle observer did not revoke the completed grant")
    if loom.post("profiles/get", {"name": template_name})["env"]:
        raise RuntimeError("Loom persisted a shared credential in its profile")
    return {
        "run_id": run["run_id"],
        "session_id": run["session_id"],
        "profile": template_name,
        "result": "broker HTTP launch; native shared grant accepted; background observer revoked completed grant",
        "limit": "scripted ACP agent; restricted Loom has no CoreWeave storage tool",
        "session_obtained_native_identity": loom.storage_marker is not None,
    }
