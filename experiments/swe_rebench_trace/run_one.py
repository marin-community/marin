# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-row map function for SWE-rebench tracing.

For one SWE-rebench-V2 row this:

1. Pulls the row's docker image into a local OCI layout via ``skopeo``.
2. Unpacks it to a runtime rootfs via ``umoci``.
3. Injects the marin tracer (`tracer.py`) and an entrypoint script that
   exports ``MARIN_TRACE_FD`` and runs the row's ``test_cmd``.
4. Generates an OCI ``config.json`` that mounts the tracer, sets
   ``PYTHONSTARTUP``/``HTTPS_PROXY`` env vars, and runs the entrypoint.
5. Invokes ``runsc run`` (rootless, no network) and captures stdout, stderr,
   and the framed trace stream over a unix pipe.
6. Returns a single dict — suitable for direct emission via Zephyr's
   ``.write_jsonl()`` / ``.write_parquet()``.

The function is intentionally a single linear top-level function: this is
a one-shot data-generation script, not a library.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

# Cap captured stdout/stderr per run. The trace stream is bounded separately
# by tracer.py via MARIN_TRACE_MAX_EVENTS.
DEFAULT_STDOUT_CAP_BYTES = 8 * 1024 * 1024  # 8 MiB

# Hard wall-clock timeout for one container.
DEFAULT_TIMEOUT_S = 1800

# Default trace event cap (≈ 200 MB JSON-framed at ~40 bytes / event).
DEFAULT_MAX_TRACE_EVENTS = 5_000_000

# Path inside the sandbox where the tracer is bind-mounted. Kept underneath
# /_marin/ so it can't collide with anything in real OSS projects.
SANDBOX_TRACER_DIR = "/_marin"
SANDBOX_TRACER_PATH = f"{SANDBOX_TRACER_DIR}/tracer.py"
SANDBOX_ENTRYPOINT_PATH = f"{SANDBOX_TRACER_DIR}/entrypoint.sh"
SANDBOX_TRACE_FIFO = f"{SANDBOX_TRACER_DIR}/trace.fifo"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TraceResult:
    """Output of one trace_swe_row call. Plain dict-friendly for serialization."""

    instance_id: str
    image_name: str
    test_cmd: str
    runtime: str  # "runsc"
    tracer: str  # "sys.monitoring" | "sys.settrace" | "unknown"
    sandbox_python: str  # e.g. "3.11.8"; empty if no metadata record was emitted
    returncode: int
    duration_s: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    trace_events: list[dict]
    trace_event_count: int
    trace_truncated: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: float, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run that always captures output."""
    logger.debug("exec: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _skopeo_copy(image: str, oci_dir: Path, *, timeout: float = 600.0) -> None:
    """Pull an image from a docker registry into a local OCI layout.

    The destination uses the OCI image layout format that umoci understands.
    """
    oci_dir.mkdir(parents=True, exist_ok=True)
    dest = f"oci:{oci_dir}:latest"
    proc = _run(
        [
            "skopeo",
            "copy",
            "--retry-times",
            "3",
            f"docker://{image}",
            dest,
        ],
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"skopeo copy failed for {image}: {proc.stderr.strip()}")


def _umoci_unpack(oci_dir: Path, bundle_dir: Path, *, timeout: float = 600.0) -> None:
    """Unpack an OCI image to a runtime bundle (rootfs/ + config.json)."""
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            "umoci",
            "unpack",
            "--rootless",
            "--image",
            f"{oci_dir}:latest",
            str(bundle_dir),
        ],
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"umoci unpack failed: {proc.stderr.strip()}")


# ---------------------------------------------------------------------------
# OCI spec generation
# ---------------------------------------------------------------------------


def _load_image_config(oci_dir: Path) -> dict:
    """Read the image manifest's config blob to get default WORKDIR/USER."""
    index = json.loads((oci_dir / "index.json").read_text())
    manifest_digest = index["manifests"][0]["digest"]
    _, sha = manifest_digest.split(":", 1)
    manifest = json.loads((oci_dir / "blobs" / "sha256" / sha).read_text())
    config_digest = manifest["config"]["digest"]
    _, sha = config_digest.split(":", 1)
    return json.loads((oci_dir / "blobs" / "sha256" / sha).read_text())


def _build_oci_config(
    *,
    bundle_dir: Path,
    test_cmd: str,
    image_config: dict,
    extra_env: dict[str, str],
    rootfs_subdir: str = "rootfs",
) -> dict:
    """Build a minimal OCI runtime spec for runsc.

    The rootfs is mounted **read-only**. Anything the test command needs to
    write to (the project source under /testbed, the trace stream at
    /tmp/marin-trace.bin, pip's site-packages cache, etc.) lives on a
    tmpfs overlay so a misbehaving test can't fill the worker's disk via
    arbitrary writes to the rootfs. The tmpfs caps are deliberately small
    so OOM hits before disk pressure does.
    """
    cfg = image_config.get("config", {}) or {}
    image_env = list(cfg.get("Env") or [])
    image_workdir = cfg.get("WorkingDir") or "/"

    # Convert image env to a dict so user-provided values can override.
    env_pairs: dict[str, str] = {}
    for entry in image_env:
        if "=" in entry:
            key, value = entry.split("=", 1)
            env_pairs[key] = value
    env_pairs.update(extra_env)

    return {
        "ociVersion": "1.0.2",
        "process": {
            "terminal": False,
            "user": {"uid": 0, "gid": 0},
            "args": ["/bin/sh", SANDBOX_ENTRYPOINT_PATH],
            "env": [f"{k}={v}" for k, v in env_pairs.items()],
            "cwd": image_workdir,
            "capabilities": {
                "bounding": ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_SETUID", "CAP_SETGID"],
                "effective": ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_SETUID", "CAP_SETGID"],
                "permitted": ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_SETUID", "CAP_SETGID"],
            },
            "rlimits": [{"type": "RLIMIT_NOFILE", "hard": 65536, "soft": 65536}],
            "noNewPrivileges": True,
        },
        "root": {"path": rootfs_subdir, "readonly": True},
        "hostname": "swe-trace",
        "mounts": [
            {"destination": "/proc", "type": "proc", "source": "proc"},
            {
                "destination": "/dev",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "strictatime", "mode=755", "size=65536k"],
            },
            {
                "destination": "/tmp",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "size=2g"],
            },
            # Writable overlay where SWE-rebench projects live. tmpfs-backed
            # so the rootfs stays read-only and the worker disk is protected.
            {
                "destination": "/testbed",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "size=8g"],
            },
            # Writable overlays for caches the test command may try to populate
            # (pip / cargo / npm under HOME, /var/tmp build dirs, etc.).
            {
                "destination": "/root",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "size=4g"],
            },
            {
                "destination": "/var/tmp",
                "type": "tmpfs",
                "source": "tmpfs",
                "options": ["nosuid", "nodev", "size=2g"],
            },
        ],
        "linux": {
            "namespaces": [
                {"type": "pid"},
                {"type": "ipc"},
                {"type": "uts"},
                {"type": "mount"},
                {"type": "network"},  # private netns; no host network
            ],
        },
        "annotations": {"marin.test_cmd": test_cmd},
    }


# ---------------------------------------------------------------------------
# Tracer / entrypoint injection
# ---------------------------------------------------------------------------


def _inject_tracer_and_entrypoint(
    *,
    rootfs: Path,
    test_cmd: str,
) -> None:
    """Copy tracer.py and write the entrypoint shell script into the rootfs."""
    tracer_dir = rootfs / SANDBOX_TRACER_DIR.lstrip("/")
    tracer_dir.mkdir(parents=True, exist_ok=True)

    src = Path(__file__).parent / "tracer.py"
    shutil.copy(src, tracer_dir / "tracer.py")

    # The entrypoint redirects fd 9 to a file inside the sandbox so the tracer
    # can write framed trace records via ``MARIN_TRACE_FD=9`` without
    # colliding with stdout. The host reads the file out of the rootfs after
    # the container exits.
    #
    # We deliberately do NOT enable ``set -e``: we want the test command's
    # actual exit code, not a hard-stop on the first non-zero return. The
    # ``; exit_code=$?`` form is part of the same statement so the exit-code
    # capture and the trace-fd close always run, regardless of test outcome.
    entrypoint_script = f"""#!/bin/sh
export MARIN_TRACE_FD=9
exec 9>>/tmp/marin-trace.bin
{test_cmd}; exit_code=$?
exec 9>&- 2>/dev/null
exit $exit_code
"""
    entrypoint_path = tracer_dir / "entrypoint.sh"
    entrypoint_path.write_text(entrypoint_script)
    entrypoint_path.chmod(0o755)


# ---------------------------------------------------------------------------
# Trace stream reader
# ---------------------------------------------------------------------------


def _iter_trace_records(stream: Iterable[bytes]) -> Iterator[dict]:
    """Decode the length-prefixed JSON record stream produced by tracer.py.

    ``stream`` is an iterable of byte chunks. Yields one decoded dict per
    record. Stops on truncation or partial record at EOF.
    """
    buf = bytearray()
    for chunk in stream:
        buf.extend(chunk)
        while True:
            if len(buf) < 4:
                break
            (payload_len,) = struct.unpack(">I", bytes(buf[:4]))
            if len(buf) < 4 + payload_len:
                break
            payload = bytes(buf[4 : 4 + payload_len])
            del buf[: 4 + payload_len]
            try:
                yield json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue


def _read_trace_file(path: Path, *, max_events: int) -> tuple[list[dict], int, bool, dict]:
    """Read a captured trace stream from disk.

    Returns (events, total_count, truncated, meta). ``events`` is capped at
    ``max_events`` to keep the per-row dict bounded; ``total_count`` is the
    full count read off disk before capping. ``meta`` is the first ``e=meta``
    record emitted by tracer.py at install time, or an empty dict if no
    metadata record was seen (e.g. the test command never ran a Python
    interpreter or the tracer failed to install).

    The meta record is consumed from the stream — it does NOT count toward
    ``total`` or appear in ``events``.
    """
    if not path.exists():
        return [], 0, False, {}
    events: list[dict] = []
    total = 0
    truncated = False
    meta: dict = {}
    with path.open("rb") as f:
        for record in _iter_trace_records(iter(lambda: f.read(65536), b"")):
            if not meta and record.get("e") == "meta":
                meta = record
                continue
            total += 1
            if len(events) < max_events:
                events.append(record)
            else:
                truncated = True
    return events, total, truncated, meta


# ---------------------------------------------------------------------------
# stdout/stderr capping
# ---------------------------------------------------------------------------


def _cap_text(blob: bytes, cap: int) -> tuple[str, bool]:
    """Truncate to ``cap`` bytes and return (text, truncated)."""
    if len(blob) <= cap:
        return blob.decode("utf-8", errors="replace"), False
    head = blob[: cap // 2]
    tail = blob[-cap // 2 :]
    sep = b"\n[... truncated ...]\n"
    return (head + sep + tail).decode("utf-8", errors="replace"), True


# ---------------------------------------------------------------------------
# Container ID sanitization
# ---------------------------------------------------------------------------

# OCI container IDs need to be filesystem-safe single path components: runsc
# uses them for state-dir names. SWE-rebench instance_ids contain `/`, `.`,
# and other characters that don't survive that path mapping, so we replace
# anything outside the safe alphabet with `-` and append a UUID4 to guarantee
# uniqueness across retries / parallel workers.
_CONTAINER_ID_SAFE_CHARS = re.compile(r"[^a-z0-9-]")
_CONTAINER_ID_MULTIPLE_DASHES = re.compile(r"-+")
_CONTAINER_ID_MAX_INSTANCE_LEN = 48


def _sanitize_container_id(instance_id: str) -> str:
    """Build a runsc-safe container ID from a SWE-rebench instance_id.

    Replaces any character outside ``[a-z0-9-]`` with ``-``, collapses runs
    of dashes, trims length to ``_CONTAINER_ID_MAX_INSTANCE_LEN``, and
    appends a uuid4 suffix to guarantee uniqueness across retries and
    parallel workers.
    """
    base = _CONTAINER_ID_SAFE_CHARS.sub("-", instance_id.lower())
    base = _CONTAINER_ID_MULTIPLE_DASHES.sub("-", base).strip("-")
    if not base:
        base = "row"
    return f"swe-{base[:_CONTAINER_ID_MAX_INSTANCE_LEN]}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# runsc invocation
# ---------------------------------------------------------------------------


def _runsc_run(
    *,
    bundle_dir: Path,
    container_id: str,
    timeout_s: float,
) -> tuple[int, bytes, bytes]:
    """Invoke runsc rootless and capture stdout, stderr, and the exit code."""
    # We use --network=none on the runsc command line; the OCI spec also
    # requests its own network namespace. The host can still reach the proxy
    # if the caller adds a host-network mount, but for the prototype we
    # assume the proxy is unreachable from the sandbox (we test selective
    # network with --network=host once we wire up host loopback bridging).
    cmd = [
        "runsc",
        "--rootless=true",
        "--network=none",
        "--ignore-cgroups=true",
        "run",
        "--bundle",
        str(bundle_dir),
        container_id,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -9, stdout, stderr + b"\n[marin] killed: timeout exceeded\n"
    return proc.returncode, stdout, stderr


# ---------------------------------------------------------------------------
# Per-row map function
# ---------------------------------------------------------------------------


def trace_swe_row(
    row: dict,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    stdout_cap_bytes: int = DEFAULT_STDOUT_CAP_BYTES,
    max_trace_events: int = DEFAULT_MAX_TRACE_EVENTS,
) -> dict:
    """Pull, sandbox, and trace one SWE-rebench-V2 row.

    Returns a dict that can be written directly via Zephyr's map-only
    streaming output (JSONL or Parquet).
    """
    instance_id = str(row.get("instance_id", "unknown"))
    image_name = row["image_name"]
    install_config = row.get("install_config") or {}
    test_cmd = install_config.get("test_cmd") or row.get("test_cmd") or ""
    if not test_cmd:
        return _make_error_result(
            instance_id=instance_id,
            image_name=image_name,
            test_cmd="",
            error="row has no install_config.test_cmd",
        ).to_dict()

    started_at = time.monotonic()

    with tempfile.TemporaryDirectory(prefix=f"swe-{instance_id}-") as tmp:
        tmp_path = Path(tmp)
        oci_dir = tmp_path / "oci"
        bundle_dir = tmp_path / "bundle"
        rootfs_dir = bundle_dir / "rootfs"

        try:
            _skopeo_copy(image_name, oci_dir)
            _umoci_unpack(oci_dir, bundle_dir)
            _inject_tracer_and_entrypoint(rootfs=rootfs_dir, test_cmd=test_cmd)

            image_config = _load_image_config(oci_dir)
            spec = _build_oci_config(
                bundle_dir=bundle_dir,
                test_cmd=test_cmd,
                image_config=image_config,
                extra_env={
                    "PYTHONSTARTUP": SANDBOX_TRACER_PATH,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONFAULTHANDLER": "1",
                    "MARIN_TRACE_ROOTS": "/testbed",
                    "MARIN_TRACE_MAX_EVENTS": str(max_trace_events),
                },
            )
            (bundle_dir / "config.json").write_text(json.dumps(spec))

            container_id = _sanitize_container_id(instance_id)
            returncode, stdout_bytes, stderr_bytes = _runsc_run(
                bundle_dir=bundle_dir,
                container_id=container_id,
                timeout_s=timeout_s,
            )

            stdout, stdout_truncated = _cap_text(stdout_bytes, stdout_cap_bytes)
            stderr, stderr_truncated = _cap_text(stderr_bytes, stdout_cap_bytes)

            trace_path = rootfs_dir / "tmp" / "marin-trace.bin"
            trace_events, trace_total, trace_truncated, trace_meta = _read_trace_file(
                trace_path,
                max_events=max_trace_events,
            )

            return TraceResult(
                instance_id=instance_id,
                image_name=image_name,
                test_cmd=test_cmd,
                runtime="runsc",
                tracer=str(trace_meta.get("tracer") or "unknown"),
                sandbox_python=str(trace_meta.get("py") or ""),
                returncode=returncode,
                duration_s=time.monotonic() - started_at,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                trace_events=trace_events,
                trace_event_count=trace_total,
                trace_truncated=trace_truncated,
                error=None,
            ).to_dict()
        except Exception as exc:
            logger.exception("trace_swe_row failed for %s", instance_id)
            return _make_error_result(
                instance_id=instance_id,
                image_name=image_name,
                test_cmd=test_cmd,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - started_at,
            ).to_dict()


def _make_error_result(
    *,
    instance_id: str,
    image_name: str,
    test_cmd: str,
    error: str,
    duration_s: float = 0.0,
) -> TraceResult:
    return TraceResult(
        instance_id=instance_id,
        image_name=image_name,
        test_cmd=test_cmd,
        runtime="runsc",
        tracer="unknown",
        sandbox_python="",
        returncode=-1,
        duration_s=duration_s,
        stdout="",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        trace_events=[],
        trace_event_count=0,
        trace_truncated=False,
        error=error,
    )
