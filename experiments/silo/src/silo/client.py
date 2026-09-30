# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The worker-side client: a work-alike of the Daytona SDK surface the pipeline calls.

Scope is exactly brief section 2 -- the calls ``capability_pipeline/daytona_*.py``
and ``dt.py`` make -- and nothing else. It accepts the pipeline's existing
parameter objects unchanged (``CreateSnapshotParams``, ``CreateSandboxFromSnapshotParams``,
``SessionExecuteRequest`` from the ``daytona`` package, or plain mappings), reading
them by attribute, so swapping the provider does not touch a single call site.

stdlib only, because ``dt.py`` is loaded with ``exec()`` from one file.

Control-plane calls go to the broker. Data-plane calls (exec, sessions, files) go
straight to the host that owns the sandbox, authenticated with the per-sandbox
capability the broker returned, so the broker never carries a build's output.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from silo.auth import HEADER_API_TOKEN, HEADER_SANDBOX_TOKEN, bearer
from silo.errors import NETWORK_REFUSAL, SiloError
from silo.http import HttpClient, TransportError

logger = logging.getLogger(__name__)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dockerfile_text(image: Any) -> str:
    """Pull the recipe text out of whatever the caller passed as ``image``.

    The pipeline passes ``daytona.Image.from_dockerfile(path)``, which reads the
    file eagerly (the SDK stores it on ``_dockerfile``, with a ``dockerfile()``
    accessor) and the pipeline deletes the temp file right after. A plain string
    is accepted too.
    """
    if isinstance(image, str):
        return image
    accessor = getattr(image, "dockerfile", None)
    text = accessor() if callable(accessor) else getattr(image, "_dockerfile", None)
    if isinstance(text, str):
        return text
    raise SiloError(f"cannot read a Dockerfile from image argument of type {type(image).__name__}", status_code=400)


# --------------------------------------------------------------------------- #
# Records returned to callers. Attribute access, like the SDK's models.
# --------------------------------------------------------------------------- #


class _Record(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _snapshot(body: Mapping[str, Any]) -> _Record:
    data = dict(body)
    data["build_info"] = _Record(**(body.get("build_info") or {}))
    return _Record(**data)


class ProcessApi:
    def __init__(self, http: HttpClient, sandbox_id: str) -> None:
        self._http = http
        self._base = f"/sandboxes/{sandbox_id}"

    def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Record:
        body = self._http.post(
            f"{self._base}/exec",
            {"command": command, "cwd": cwd, "env": dict(env) if env else None, "timeout": timeout},
            timeout=(timeout or 600) + 90,
        )
        if body.get("timed_out"):
            # Daytona raises on an exec deadline and the pipeline classifies it by
            # the word "timeout" in the message.
            raise SiloError(f"sandbox exec timeout after {timeout}s", status_code=408)
        return _Record(exit_code=body["exit_code"], result=body["result"])

    def create_session(self, session_id: str) -> None:
        self._http.post(f"{self._base}/sessions", {"session_id": session_id})

    def execute_session_command(self, session_id: str, req: Any, timeout: float | None = None) -> _Record:
        command = _field(req, "command")
        run_async = _field(req, "run_async")
        if run_async is None:
            run_async = _field(req, "var_async", True)
        body = self._http.post(
            f"{self._base}/sessions/{session_id}/exec",
            {"command": command, "run_async": bool(run_async)},
            timeout=timeout or 120,
        )
        return _Record(cmd_id=body["cmd_id"], exit_code=body.get("exit_code"), command=body.get("command"))

    def get_session_command(self, session_id: str, cmd_id: str) -> _Record:
        body = self._http.get(f"{self._base}/sessions/{session_id}/commands/{cmd_id}")
        return _Record(id=body["cmd_id"], cmd_id=body["cmd_id"], command=body["command"], exit_code=body["exit_code"])

    def get_session_command_logs(self, session_id: str, cmd_id: str) -> _Record:
        body = self._http.get(f"{self._base}/sessions/{session_id}/commands/{cmd_id}/logs", timeout=300)
        return _Record(stdout=body["stdout"], stderr=body["stderr"], output=body["stdout"] + body["stderr"])

    def delete_session(self, session_id: str) -> None:
        self._http.delete(f"{self._base}/sessions/{session_id}")


class FsApi:
    def __init__(self, http: HttpClient, sandbox_id: str) -> None:
        self._http = http
        self._base = f"/sandboxes/{sandbox_id}/files"

    def upload_file(self, data: bytes, target_path: str) -> None:
        if isinstance(data, str):
            data = data.encode()
        self._http.put(self._base, data, query={"path": target_path}, timeout=900)

    def download_file(self, source_path: str) -> bytes:
        return self._http.get(self._base, query={"path": source_path}, raw=True, timeout=900)


class Sandbox:
    """A sandbox handle. Attribute names match what the pipeline reads."""

    def __init__(self, client: SiloClient, body: Mapping[str, Any]) -> None:
        self._client = client
        self._apply(body)
        host = HttpClient(body["host_url"], headers={HEADER_SANDBOX_TOKEN: body["token"]}, timeout=120)
        self.process = ProcessApi(host, self.id)
        self.fs = FsApi(host, self.id)

    def _apply(self, body: Mapping[str, Any]) -> None:
        self.id: str = body["id"]
        self.state: str = body["state"]
        self.snapshot: str = body["snapshot"]
        self.cpu: int = body["cpu"]
        self.memory: int = body["memory"]
        self.disk: int = body["disk"]
        # Read back by several gates as the isolation attestation. Always True:
        # there is no code path that starts a networked sandbox.
        self.network_block_all: bool = bool(body["network_block_all"])
        self.labels: dict[str, str] = dict(body.get("labels") or {})
        self.created_at: str = body.get("created_at", "")
        self.host_id: str = body.get("host_id", "")

    def refresh_data(self) -> None:
        self._apply(self._client._broker.get(f"/sandboxes/{self.id}"))

    def delete(self, timeout: float | None = None) -> None:
        self._client._broker.delete(f"/sandboxes/{self.id}", timeout=timeout or 120)


class SnapshotApi:
    def __init__(self, client: SiloClient) -> None:
        self._client = client

    def create(self, params: Any, *, timeout: float = 3600, on_logs: Callable[[str], None] | None = None) -> _Record:
        """Register a snapshot and wait until it is usable, as the SDK does.

        For a FROM-only recipe this is a pre-pull of a pinned digest and returns
        in seconds. For a recipe with RUN steps it waits for the build.
        """
        name = _field(params, "name")
        resources = _field(params, "resources")
        if resources is None:
            raise SiloError("snapshot create requires resources (cpu, memory, disk)", status_code=400)
        body = {
            "name": name,
            "dockerfile_content": _dockerfile_text(_field(params, "image")),
            "cpu": int(_field(resources, "cpu")),
            "memory_gb": int(_field(resources, "memory")),
            "disk_gb": int(_field(resources, "disk")),
        }
        self._client._broker.post("/snapshots", body)
        deadline = time.monotonic() + timeout
        seen = 0
        # Short first waits: a FROM-only snapshot is active in about a second.
        delays = iter((0.1, 0.25, 0.5, 1.0))
        while True:
            record = self.get(name)
            if on_logs is not None:
                lines = self._client._broker.get(f"/snapshots/{name}/build-logs", raw=True).decode().splitlines()
                for line in lines[seen:]:
                    on_logs(line)
                seen = len(lines)
            if record.state == "active":
                return record
            if record.state not in ("pending", "building", "creating", "queued"):
                raise SiloError(f"snapshot {name!r} failed: {getattr(record, 'error_message', '')}", status_code=500)
            if time.monotonic() >= deadline:
                raise SiloError(f"snapshot {name!r} did not become active within {timeout}s timeout", status_code=408)
            time.sleep(next(delays, 2.0))

    def get(self, name: str) -> _Record:
        return _snapshot(self._client._broker.get(f"/snapshots/{name}"))

    def list(self, page: int = 1, limit: int = 100) -> _Record:
        body = self._client._broker.get("/snapshots", query={"page": page, "limit": limit})
        return _Record(
            items=[_snapshot(item) for item in body["items"]],
            total=body["total"],
            page=body["page"],
            total_pages=body["total_pages"],
        )

    def delete(self, snapshot: Any) -> None:
        name = snapshot if isinstance(snapshot, str) else _field(snapshot, "name")
        self._client._broker.delete(f"/snapshots/{name}")

    def build_logs(self, name: str) -> str:
        """Build logs, returned directly.

        The Daytona path reaches them through a private SDK attribute and a
        hardcoded proxy hostname (``daytona_verifier.py:89-105``); this replaces
        both with one call.
        """
        return self._client._broker.get(f"/snapshots/{name}/build-logs", raw=True).decode()


class SiloClient:
    """The ``Daytona`` object of the old SDK, pointed at a silo broker.

    ``broker_url`` is the broker's direct address. ``resolve_url``, if given, is
    the broker reached by NAME through the Iris controller's proxy; on a transport
    failure the client asks it for the broker's current address and retries once.
    That is what keeps a worker from being stranded on a dead address if the
    broker restarts -- the failure mode that cost 49 jobs per relay cut.
    """

    def __init__(self, broker_url: str | None, api_token: str, *, resolve_url: str | None = None) -> None:
        self._api_token = api_token
        self._resolve_url = resolve_url
        if not broker_url:
            # Start from the name alone: ask the proxy where the broker is now.
            broker_url = self._reresolve()
            if not broker_url:
                raise SiloError(f"cannot resolve the silo broker via {resolve_url!r}", status_code=503)
        self._broker = _ReresolvingClient(self, broker_url)
        self.snapshot = SnapshotApi(self)

    def _new_http(self, url: str) -> HttpClient:
        return HttpClient(url, headers={HEADER_API_TOKEN: bearer(self._api_token)}, timeout=120)

    def _reresolve(self) -> str | None:
        if not self._resolve_url:
            return None
        try:
            body = HttpClient(self._resolve_url, timeout=30).get("/whoami")
        except SiloError:
            return None
        return body.get("url")

    def create(self, params: Any, timeout: float = 600, *, runtime: str | None = None) -> Sandbox:
        """Start a sandbox from a snapshot.

        ``runtime`` ("runsc" or "runc") is not a Daytona field; the pipeline never
        passes it and gets the host's default. It exists so acceptance runs can
        exercise both inner runtimes explicitly.
        """
        if _field(params, "network_block_all") is not True:
            # The provider has exactly one network posture. Saying so is better
            # than handing back a sandbox whose isolation differs from the request.
            raise SiloError(NETWORK_REFUSAL, status_code=400)
        for name in ("network_allow_list", "domain_allow_list"):
            if _field(params, name):
                raise SiloError(
                    f"{name} is not supported: sandboxes have no network interface, so an "
                    "allow-list cannot be honoured, and approximating it as block-all would "
                    "misstate the isolation the caller asked for",
                    status_code=400,
                )
        if _field(params, "env_vars"):
            raise SiloError("sandbox env_vars are not supported; pass env per exec", status_code=400)
        body = {
            "snapshot": _field(params, "snapshot"),
            "labels": dict(_field(params, "labels") or {}),
            "ttl_minutes": _field(params, "ttl_minutes") or 180,
            "network_block_all": True,
            "timeout": timeout,
            "runtime": runtime,
        }
        created = self._broker.post("/sandboxes", body, timeout=timeout + 60)
        return Sandbox(self, created)

    def get(self, sandbox_id: str) -> Sandbox:
        return Sandbox(self, self._broker.get(f"/sandboxes/{sandbox_id}"))

    def list(self) -> list[Sandbox]:
        return [Sandbox(self, body) for body in self._broker.get("/sandboxes")["items"]]

    def delete(self, sandbox: Sandbox | str) -> None:
        sandbox_id = sandbox if isinstance(sandbox, str) else sandbox.id
        self._broker.delete(f"/sandboxes/{sandbox_id}")

    def capacity(self) -> dict[str, Any]:
        """The named ceiling. There was no equivalent on Daytona."""
        return self._broker.get("/capacity")


class _ReresolvingClient:
    """Broker HTTP client that re-resolves the broker once on a transport error."""

    def __init__(self, owner: SiloClient, url: str) -> None:
        self._owner = owner
        self._http = owner._new_http(url)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._http, method)(*args, **kwargs)
        except TransportError as error:
            # GET and DELETE are idempotent. A POST is retried only if it was
            # refused before reaching a server; one that timed out may already
            # have created a sandbox, and a blind retry would create a second.
            if method == "post" and not error.refused:
                raise
            fresh = self._owner._reresolve()
            if not fresh or fresh.rstrip("/") == self._http.base_url:
                raise
            logger.warning("broker unreachable; re-resolved to %s", fresh)
            self._http = self._owner._new_http(fresh)
            return getattr(self._http, method)(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("get", *args, **kwargs)

    def post(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("post", *args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("delete", *args, **kwargs)
