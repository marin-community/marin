# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, parent-owned persistence of one retired shutdown observer."""

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import os
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROUTE_NAME = ".native-observer-route.json"
MAX_RECEIPT_BYTES = 65536
MAX_TOTAL_BYTES = 2 * 1024 * 1024
UPLOAD_TIMEOUT_SECONDS = 5.0
REAP_TIMEOUT_SECONDS = 1.0
FILE_LIMITS = {
    "observer-config.json": MAX_RECEIPT_BYTES,
    "tracer-ready.json": MAX_RECEIPT_BYTES,
    "ack.json": MAX_RECEIPT_BYTES,
    "lldb-receipt.json": MAX_RECEIPT_BYTES,
    "guardian-receipt.json": MAX_RECEIPT_BYTES,
    "stack.txt": 1024 * 1024,
    "debugger.log": 32768,
    "guardian.stderr": MAX_RECEIPT_BYTES,
}


@contextlib.contextmanager
def _directory(path: Path, *, private: bool = False):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("Observer artifact directory must be canonical")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        row = os.fstat(descriptor)
        if row.st_uid != os.getuid() or (private and stat.S_IMODE(row.st_mode) != 0o700):
            raise ValueError("Observer artifact directory ownership differs")
        yield descriptor
    finally:
        os.close(descriptor)


def _file_version(row: os.stat_result) -> tuple[int, int, int, int, int]:
    return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns


def _read_file(directory: int, name: str, limit: int) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1:
            raise ValueError("Observer artifact must be an owned regular file")
        if before.st_size > limit:
            raise ValueError("Observer artifact exceeds its byte limit")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if _file_version(before) != _file_version(after) or len(data) != before.st_size:
            raise ValueError("Observer artifact changed during snapshot")
        return data


def _write_json(directory: Path, name: str, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    if len(data) > MAX_RECEIPT_BYTES:
        raise ValueError("Observer handoff receipt exceeds its byte limit")
    with _directory(directory) as descriptor:
        pending = name + ".pending"
        output = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
        try:
            with os.fdopen(output, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, name, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
            os.fsync(descriptor)
        finally:
            os.unlink(pending, dir_fd=descriptor)


def write_route(root: Path, rendezvous: str, node_id: str, task_id: str, attempt_uid: str) -> None:
    """Bind a local debug root to the task runtime's exact remote destination."""
    rank = int(task_id.rsplit("/", 1)[-1].split(":", 1)[0])
    if (
        not rendezvous
        or not task_id.startswith("/")
        or not attempt_uid
        or node_id != f"rank{rank}-{socket.gethostname()}"
    ):
        raise ValueError("Observer route requires the current physical task identity")
    row = {
        "rendezvous": rendezvous,
        "node_id": node_id,
        "task_id": task_id,
        "attempt_uid": attempt_uid,
        "hostname": socket.gethostname(),
    }
    with _directory(root) as descriptor:
        try:
            existing = json.loads(_read_file(descriptor, ROUTE_NAME, MAX_RECEIPT_BYTES))
        except FileNotFoundError:
            existing = None
    if existing is not None:
        if existing != row:
            raise ValueError("Observer route already belongs to another task")
        return
    _write_json(root, ROUTE_NAME, row)


def _bound_route(request: dict) -> tuple[Path, dict, str]:
    config = request["config"]
    directory = Path(request["directory"])
    expected = f"rank{request['process_id']}-pid{config['pid']}-{config['nonce']}"
    if not re.fullmatch(r"[0-9a-f]{32}", config["nonce"]) or directory.name != expected:
        raise ValueError("Observer directory identity differs")
    root = directory.parent.parent
    if directory.parent.name != "native-shutdown-observer":
        raise ValueError("Observer directory lies outside its managed root")
    for key, name in (
        ("ready_path", "tracer-ready.json"),
        ("ack_path", "ack.json"),
        ("trace_path", "stack.txt"),
        ("lldb_receipt_path", "lldb-receipt.json"),
        ("guardian_receipt_path", "guardian-receipt.json"),
        ("debugger_log_path", "debugger.log"),
        ("cancel_path", "cancel"),
    ):
        if config[key] != str(directory / name):
            raise ValueError("Observer configured path differs")
    with _directory(directory, private=True), _directory(root) as descriptor:
        route = json.loads(_read_file(descriptor, ROUTE_NAME, MAX_RECEIPT_BYTES))
    if any(route.get(key) != request[key] for key in ("task_id", "attempt_uid", "hostname")):
        raise ValueError("Observer route task identity differs")
    rank = int(route["task_id"].rsplit("/", 1)[-1].split(":", 1)[0])
    if route["node_id"] != f"rank{rank}-{request['hostname']}" or not route["rendezvous"]:
        raise ValueError("Observer route physical node differs")
    destination = (
        route["rendezvous"].rstrip("/")
        + "/debug_artifacts/"
        + route["node_id"]
        + "/native-shutdown-observer/"
        + directory.name
    )
    return directory, route, destination


def upload_selected(request: dict, filesystem_factory=None) -> dict:
    """Upload only stable, identity-validated observer files; commit the index last."""
    directory, route, destination = _bound_route(request)
    if filesystem_factory is None:
        # Keep the isolated entry point stdlib-only until its route is validated.
        from rigging.filesystem.factory import url_to_fs  # noqa: PLC0415 - optional storage dependency

        filesystem_factory = url_to_fs
    filesystem, destination_path = filesystem_factory(destination)
    config = request["config"]
    receipt = {
        "schema_version": 1,
        "pid": config["pid"],
        "starttime": config["starttime"],
        "nonce": config["nonce"],
        "owner_pid": config["owner_pid"],
        "owner_starttime": config["owner_starttime"],
        "process_id": request["process_id"],
        "task_id": route["task_id"],
        "attempt_uid": route["attempt_uid"],
        "destination": destination,
        "cancel": request["cancel"],
        "retired_safe": request["retired_safe"],
        "primary_error_type": request.get("primary_error_type"),
        "observation_completed": request["observation_completed"],
        "files": [],
        "missing": [],
        "errors": [],
    }
    if not request["retired_safe"]:
        receipt["errors"].append({"stage": "retirement", "type": request["retirement_error"] or "UnsafeRetirement"})
    else:
        required = {"observer-config.json", "guardian-receipt.json"}
        guardian = request["guardian_receipt"]
        if guardian.get("safe_cleanup") is not True:
            raise ValueError("Observer retirement safety differs")
        if guardian.get("lldb"):
            required.add("lldb-receipt.json")
        if guardian.get("lldb", {}).get("status") == "collected":
            required.update(("stack.txt", "ack.json"))
        total = 0
        with (
            _directory(directory, private=True) as descriptor,
            tempfile.TemporaryDirectory(prefix="observer-handoff-") as stage,
        ):
            for name, limit in FILE_LIMITS.items():
                try:
                    data = _read_file(descriptor, name, limit)
                    if name.endswith(".json"):
                        row = json.loads(data)
                        # LLDB records bound_pid, not the guardian's nonce tuple.
                        # Bind its exact bytes to the validated guardian copy.
                        if name != "lldb-receipt.json" and any(
                            row.get(key) != config[key] for key in ("pid", "starttime", "nonce")
                        ):
                            raise ValueError("Observer artifact identity differs")
                        if name == "observer-config.json" and row != config:
                            raise ValueError("Observer configuration differs")
                        if name == "guardian-receipt.json" and row != guardian:
                            raise ValueError("Observer retirement receipt differs")
                        if name == "lldb-receipt.json" and row != guardian.get("lldb"):
                            raise ValueError("Observer debugger receipt differs")
                        if name == "lldb-receipt.json" and row.get("bound_pid", config["pid"]) != config["pid"]:
                            raise ValueError("Observer debugger target differs")
                        if name == "ack.json" and (
                            row.get("ack_path") != config["ack_path"]
                            or row.get("event") != "shutdown_ack_written"
                            or row.get("process_id") != request["process_id"]
                        ):
                            raise ValueError("Observer acknowledgement differs")
                    if total + len(data) > MAX_TOTAL_BYTES - MAX_RECEIPT_BYTES:
                        raise ValueError("Observer handoff exceeds its total byte limit")
                    snapshot = Path(stage) / name
                    snapshot.write_bytes(data)
                    filesystem.put(str(snapshot), destination_path + "/" + name)
                    receipt["files"].append(
                        {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    )
                    total += len(data)
                except FileNotFoundError:
                    receipt["missing"].append(name)
                    if name in required:
                        receipt["errors"].append({"file": name, "type": "RequiredFileMissing"})
                except Exception as error:
                    receipt["errors"].append({"file": name, "type": type(error).__name__})
    receipt["status"] = "uploaded" if not receipt["errors"] else "partial"
    receipt["bytes"] = sum(row["bytes"] for row in receipt["files"])
    data = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    if len(data) > MAX_RECEIPT_BYTES or receipt["bytes"] + len(data) > MAX_TOTAL_BYTES:
        raise ValueError("Observer handoff commit exceeds its byte limit")
    with filesystem.open(destination_path + "/handoff-commit.json", "wb") as stream:
        stream.write(data)
    return receipt


def handoff_artifacts(
    directory: Path,
    config: dict,
    process_id: int,
    *,
    retired_safe: bool,
    guardian_receipt: dict | None,
    observation_completed: bool,
    cancel: bool,
    retirement_error: str | None,
    primary_error_type: str | None = None,
) -> dict:
    """Wait for one separately owned uploader without changing a shutdown outcome."""
    request = {
        "directory": str(directory),
        "config": config,
        "process_id": process_id,
        "retired_safe": retired_safe,
        "guardian_receipt": guardian_receipt,
        "observation_completed": observation_completed,
        "cancel": cancel,
        "retirement_error": retirement_error,
        "task_id": os.environ.get("IRIS_TASK_ID"),
        "primary_error_type": primary_error_type,
        "attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
        "hostname": socket.gethostname(),
    }
    result = {"status": "failed", "pid": config["pid"], "starttime": config["starttime"], "nonce": config["nonce"]}
    started = time.monotonic()
    process = None
    binding = None
    exited = False
    try:
        if sys.platform != "linux" or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise RuntimeError("Observer uploader requires Linux and protocol-owned child reaping")
        # Marin is supplied through PYTHONPATH only in the parent process.
        from marin.rl.native_shutdown.stack_safety import LinuxBinding, identity  # noqa: PLC0415

        payload = json.dumps(request).encode()
        if len(payload) > 3 * MAX_RECEIPT_BYTES:
            raise ValueError("Observer upload request exceeds its byte limit")
        with tempfile.TemporaryFile() as source:
            source.write(payload)
            source.seek(0)
            process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--parent-pid", str(os.getpid())],
                stdin=source,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            expected = identity(process.pid)
            if expected.pgrp != process.pid:
                raise RuntimeError("Observer uploader is not its owned process group leader")
            binding = LinuxBinding(expected)
            # pidfd readiness observes exit without reaping the leader. Its PID
            # remains reserved even if a credential helper outlives that leader.
            remaining = max(0, UPLOAD_TIMEOUT_SECONDS - (time.monotonic() - started))
            exited = bool(select.select([binding.fd], [], [], remaining)[0])
        if not exited:
            result.update(status="timed_out", remote_commit="unknown")
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        if process is not None:
            # Never poll/communicate/wait before this group signal: reaping the
            # leader could release its numeric PGID while descendants still run.
            try:
                if binding is not None:
                    binding.verify()
                os.killpg(process.pid, signal.SIGKILL)
                result["uploader_group_cleanup"] = "signalled"
            except ProcessLookupError:
                result["uploader_group_cleanup"] = "already_absent"
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, remote_commit="unknown")
            try:
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                result.update(status="failed", error_type="UploaderReapTimeout", remote_commit="unknown")
            result["uploader_exit_code"] = process.returncode
        if binding is not None:
            binding.close()
        result["uploader_reaped"] = process is None or process.returncode is not None
    if exited and "error_type" not in result:
        try:
            with _directory(directory, private=True) as descriptor:
                uploaded = json.loads(_read_file(descriptor, "handoff-upload.json", MAX_RECEIPT_BYTES))
            if any(uploaded.get(key) != config[key] for key in ("pid", "starttime", "nonce")):
                raise ValueError("Observer uploader result identity differs")
            result.update(status=uploaded["status"] if process.returncode == 0 else "failed", uploader=uploaded)
        except Exception as error:
            result["error_type"] = type(error).__name__
    result["seconds"] = time.monotonic() - started
    _write_json(directory, "handoff-parent.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    args = parser.parse_args()
    if sys.platform == "linux" and ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
        raise OSError("Could not bind observer uploader lifetime")
    if os.getppid() != args.parent_pid:
        raise RuntimeError("Observer uploader parent changed")
    data = sys.stdin.buffer.read(3 * MAX_RECEIPT_BYTES + 1)
    if len(data) > 3 * MAX_RECEIPT_BYTES:
        raise ValueError("Observer upload request exceeds its byte limit")
    request = json.loads(data)
    if request["config"]["owner_pid"] != args.parent_pid:
        raise ValueError("Observer uploader owner differs")
    if sys.platform == "linux":
        # The parent bootstrap supplies Marin through PYTHONPATH, which -I
        # deliberately ignores. Use the same owned sibling as the guardian.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        safety = importlib.import_module("stack_safety")

        if safety.identity(args.parent_pid).starttime != request["config"]["owner_starttime"]:
            raise ValueError("Observer uploader owner starttime differs")
    directory = Path(request["directory"])
    receipt = {"status": "failed", **{key: request["config"][key] for key in ("pid", "starttime", "nonce")}}
    try:
        receipt = upload_selected(request)
    except Exception as error:
        receipt["error_type"] = type(error).__name__
    _write_json(directory, "handoff-upload.json", receipt)
    if receipt["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
