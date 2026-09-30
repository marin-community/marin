"""Trusted-side rootfs capture through silo's own data plane.

The Daytona capture (``capture-tools/capture_rootfs.py``) creates the source
sandbox with egress allow-listed to the object store and has the sandbox PUT
``tar | gzip`` to presigned multipart URLs.  A silo sandbox has no network
interface at all (there is no allow-list to ask for), so that design cannot run
there.  This module keeps the artefact identical -- a content-addressed
``s3://<bucket>/<prefix>/<sha256>.tar.gz`` object plus a raw receipt with the
same fields ``capture_rootfs.py`` writes -- but moves every network step to the
trusted harness:

* inside the sandbox, a stream helper cuts the archive into parts in /tmp
  (excluded from the archive) and waits while a few are outstanding;
* the harness downloads each complete part through silo's file API, uploads it
  as one multipart part with its own object-store credentials, and deletes it;
* the harness hashes the bytes it actually received and requires that digest
  and length to equal the sandbox's own count before completing the upload.

Two helpers implement the same protocol (``part-NNNNN`` renamed into place when
complete, a result marker written last): ``silo_rootfs_stream.py`` for guests
with python3, and the POSIX-sh ``SHELL_HELPER`` below (GNU coreutils: ``dd
iflag=fullblock``, ``stat``, ``sha256sum``, ``mkfifo``, ``tee``) for guests
without it -- R, node and plain Debian/Ubuntu images, where the python helper
exited 127 and every capture failed with "stream ended without a result".

Evidence behind the assumptions (construct-003, 2026-09-29): silo bind-mounts
its busybox at ``/.silo`` on a distinct device in every sandbox (host
``runtime.py`` ``TOOLS_MOUNT``; the ``stat -c %d`` check passed in every
receipt); ``tar --one-file-system`` under gVisor archived complete rootfs trees
(a 68 MB capture listed 7,972 entries including /usr/lib/python3*, with an empty
tar stderr in all 78 successful captures); the stream ran at 7-9 MB/s of
compressed output, gzip-bound.  ``du -xs /`` under gVisor under-reports (hundreds
of KB for a >100 MB rootfs), so ``unpacked_kb`` is informational only.

No credential, presigned URL or registry secret enters the sandbox, and nothing
here touches a registry: publication stays with the isolated publisher.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BUCKET = "marin-us-east-02a"  # the same object tree capture_rootfs.py writes
PREFIX = "users/muchanem/envrootfs"
TRANSPORT = "silo-exec-download-v1"
HELPER = Path(__file__).with_name("silo_rootfs_stream.py")
WORK_DIR = "/tmp/capability-rootfs-capture"
PART_BYTES = 64 * 1024 * 1024
MAX_OUTSTANDING = 3
# silo bind-mounts its static busybox here in every sandbox; it is provider
# plumbing, never image content.  --one-file-system drops it (distinct mount),
# and it is excluded by name as well so the archive cannot depend on that.
PROVIDER_EXCLUDES = ("./.silo", "./.silo/*")
PROVIDER_BOUND_PATHS = ("/.silo",)
HELPER_PYTHON = "python3"
HELPER_SHELL = "posix-sh"
_SHELL_BLOCK = 65536

SHELL_HELPER = r"""# capability rootfs stream: POSIX sh + GNU coreutils, for guests without python3.
# usage: sh silo_rootfs_stream.sh ROOT OUT_DIR PART_BYTES MAX_OUTSTANDING STALL_SECONDS GZIP_LEVEL [--exclude=P ...]
root=$1; out=$2; part=$3; maxo=$4; stall=$5; level=$6
shift 6
mkdir -p "$out" || exit 1
started=$(date +%s)
fifo="$out/.digest.fifo"
rm -f "$fifo" "$out/.digest" "$out/.tar_exit" "$out/.gzip_exit" "$out/.cut"
mkfifo "$fifo" || exit 1
sha256sum < "$fifo" > "$out/.digest" &
digest_pid=$!
block=65536
blocks=$((part / block))
cut_parts() {
  n=0; total=0; err=
  while :; do
    waited=0
    while [ "$(ls "$out" | grep -c '^part-[0-9]*$')" -ge "$maxo" ]; do
      if [ "$waited" -ge "$stall" ]; then err=consumer_stalled; break; fi
      sleep 1
      waited=$((waited + 1))
    done
    [ -n "$err" ] && break
    n=$((n + 1))
    name=$(printf 'part-%05d' "$n")
    if ! dd bs="$block" count="$blocks" iflag=fullblock of="$out/$name.tmp" 2>/dev/null; then
      err=cut_failed; break
    fi
    size=$(stat -c %s "$out/$name.tmp")
    if [ "$size" -eq 0 ] && [ "$n" -gt 1 ]; then
      rm -f "$out/$name.tmp"; n=$((n - 1)); break
    fi
    total=$((total + size))
    mv "$out/$name.tmp" "$out/$name"
    [ "$size" -lt "$part" ] && break
  done
  printf 'parts=%s\ncompressed_bytes=%s\nerror=%s\n' "$n" "$total" "$err" > "$out/.cut"
}
{ tar --one-file-system --numeric-owner -cf - -C "$root" "$@" . 2> "$out/tar.stderr"; echo $? > "$out/.tar_exit"; } \
  | { gzip "-$level" -c; echo $? > "$out/.gzip_exit"; } \
  | tee "$fifo" | cut_parts
wait "$digest_pid"
parts=0; compressed_bytes=0; error=cut_missing
[ -f "$out/.cut" ] && . "$out/.cut"
tar_exit=$(cat "$out/.tar_exit" 2>/dev/null || echo -1)
gzip_exit=$(cat "$out/.gzip_exit" 2>/dev/null || echo -1)
digest=$(cut -d' ' -f1 < "$out/.digest")
ok=false
if [ -z "$error" ] && [ "$gzip_exit" = 0 ]; then ok=true; fi
seconds=$(( $(date +%s) - started ))
printf 'ok=%s\nerror=%s\ntar_exit=%s\ngzip_exit=%s\nparts=%s\ncompressed_bytes=%s\nsha256=%s\nseconds=%s\n' \
  "$ok" "$error" "$tar_exit" "$gzip_exit" "$parts" "$compressed_bytes" "$digest" "$seconds" > "$out/result.txt.tmp"
mv "$out/result.txt.tmp" "$out/result.txt"
"""

# One probe for every tool either helper needs; `gnu-dd` is `dd iflag=fullblock`.
_TOOL_PROBE = (
    "for t in python3 tar gzip sha256sum stat mkfifo tee; do "
    'command -v "$t" >/dev/null 2>&1 && echo "have:$t"; done; '
    "dd if=/dev/null of=/dev/null iflag=fullblock 2>/dev/null && echo have:gnu-dd; true"
)
_NEEDS = {
    HELPER_PYTHON: {"python3", "tar", "gzip"},
    HELPER_SHELL: {"tar", "gzip", "gnu-dd", "stat", "sha256sum", "mkfifo", "tee"},
}


class SiloCaptureError(RuntimeError):
    """Sanitized failure; never carries provider or object-store text.

    ``kind`` names the step, ``failure_class`` is transient/content/harness,
    ``result`` carries the stream helper's own report when it produced one.
    """

    def __init__(self, message: str, *, kind: str = "transport", failure_class: str = "transient",
                 detail: dict | None = None, result: dict | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.failure_class = failure_class
        self.detail = dict(detail or {})
        self.result = result


def helper_sha256() -> str:
    return hashlib.sha256(HELPER.read_bytes()).hexdigest()


def transport_identity() -> dict[str, str]:
    return {
        "transport": TRANSPORT,
        "helper_sha256": helper_sha256(),
        "shell_helper_sha256": hashlib.sha256(SHELL_HELPER.encode()).hexdigest(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _guest_tools(sh: Callable[..., dict], sandbox: Any) -> set[str]:
    try:
        probe = sh(sandbox, _TOOL_PROBE, timeout=60)
    except Exception as error:  # noqa: BLE001 - recorded by type only
        raise SiloCaptureError(f"guest tool probe failed ({type(error).__name__})", kind="tool_probe") from None
    if probe.get("exit") != 0:
        raise SiloCaptureError("guest tool probe did not complete", kind="tool_probe",
                               detail={"exit_code": probe.get("exit")})
    return {line[5:].strip() for line in (probe.get("stdout") or "").splitlines() if line.startswith("have:")}


def _shell_result(text: str) -> dict:
    fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)

    def integer(name: str) -> int | None:
        try:
            return int(fields.get(name, ""))
        except ValueError:
            return None

    seconds = integer("seconds")
    compressed = integer("compressed_bytes")
    return {
        "ok": fields.get("ok") == "true",
        "error": fields.get("error") or None,
        "tar_exit": integer("tar_exit"),
        "gzip_exit": integer("gzip_exit"),
        "parts": integer("parts"),
        "compressed_bytes": compressed,
        "sha256": fields.get("sha256", ""),
        "seconds": seconds,
        "mb_per_s": round((compressed or 0) / 1e6 / max(seconds or 0, 1), 1),
        "bad_parts": {},
    }


def capture_rootfs(
    *, sandbox: Any, sh: Callable[..., dict], snapshot_name: str, tag: str,
    excludes: list[str], s3: Any, bucket: str = BUCKET, prefix: str = PREFIX,
    gzip_level: int = 6, part_bytes: int = PART_BYTES,
    max_outstanding: int = MAX_OUTSTANDING, poll_seconds: float = 2.0,
    stall_seconds: float = 1800.0, timeout_seconds: float = 4200.0,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    helper: str | None = None, root: str = "/",
) -> dict:
    """Capture ``sandbox``'s rootfs; return a ``capture_rootfs.py``-shaped receipt.

    ``s3`` is a botocore S3 client built from the harness's own credentials.
    ``helper`` forces the python3 or posix-sh stream helper; by default the
    guest's tools decide.  Raises ``SiloCaptureError`` on any failure after
    aborting the upload.
    """
    if part_bytes % _SHELL_BLOCK:
        raise ValueError("part size must be a multiple of 64 KiB")
    rec: dict[str, Any] = {
        "snapshot": snapshot_name, "tag": tag, "gzip_level": gzip_level,
        "excludes": list(excludes), "started_utc": _utc(), "sandbox": sandbox.id,
        "sandbox_create_s": None,
    }
    try:
        probe = sh(sandbox, "env | sort; echo '--- DU ---'; du -xs / 2>/dev/null | cut -f1", timeout=300)
    except Exception as error:  # noqa: BLE001 - recorded by type only
        raise SiloCaptureError(f"source environment probe failed ({type(error).__name__})", kind="env_probe") from None
    if probe.get("exit") is None:
        raise SiloCaptureError("source environment probe did not complete", kind="env_probe")
    env_text, _, du = (probe.get("stdout") or "").partition("--- DU ---")
    rec["source_env"] = dict(line.split("=", 1) for line in env_text.strip().splitlines() if "=" in line)
    try:
        rec["unpacked_kb"] = int(du.strip().splitlines()[-1])
    except (ValueError, IndexError):
        rec["unpacked_kb"] = None

    tools = _guest_tools(sh, sandbox)
    helper = helper or (HELPER_PYTHON if "python3" in tools else HELPER_SHELL)
    missing = sorted(_NEEDS[helper] - tools)
    if missing:
        # Deterministic for this image, and a limit of our transport: loud, not retried.
        raise SiloCaptureError("the source image lacks tools the rootfs stream needs",
                               kind="guest_tools_missing", failure_class="harness",
                               detail={"helper": helper, "missing": missing, "present": sorted(tools)})
    rec["stream_helper"] = helper

    out_dir = f"{WORK_DIR}/{tag}"
    marker = "result.json" if helper == HELPER_PYTHON else "result.txt"
    try:
        if helper == HELPER_PYTHON:
            job_path = f"{WORK_DIR}/{tag}.job.json"
            helper_path = f"{WORK_DIR}/silo_rootfs_stream.py"
            job = {"excludes": list(excludes), "gzip_level": gzip_level, "out_dir": out_dir,
                   "part_bytes": part_bytes, "max_outstanding": max_outstanding,
                   "stall_seconds": stall_seconds, "root": root}
            sandbox.fs.upload_file(HELPER.read_bytes(), helper_path)
            sandbox.fs.upload_file(json.dumps(job).encode(), job_path)
            command = f"python3 {shlex.quote(helper_path)} {shlex.quote(job_path)}"
        else:
            helper_path = f"{WORK_DIR}/silo_rootfs_stream.sh"
            sandbox.fs.upload_file(SHELL_HELPER.encode(), helper_path)
            arguments = [helper_path, root, out_dir, str(part_bytes), str(max_outstanding),
                         str(int(stall_seconds)), str(int(gzip_level)), *[f"--exclude={p}" for p in excludes]]
            command = "sh " + " ".join(shlex.quote(argument) for argument in arguments)
    except Exception as error:  # noqa: BLE001
        raise SiloCaptureError(f"rootfs stream helper upload failed ({type(error).__name__})", kind="upload") from None

    run: dict[str, Any] = {}

    def producer() -> None:
        try:
            run["result"] = sh(sandbox, command, timeout=int(timeout_seconds))
        except Exception as error:  # noqa: BLE001 - recorded by type only.
            run["error_type"] = type(error).__name__

    staging = f"{prefix}/_staging-{tag}-{int(time.time())}.tar.gz"
    try:
        upload_id = s3.create_multipart_upload(Bucket=bucket, Key=staging)["UploadId"]
    except Exception as error:  # noqa: BLE001
        raise SiloCaptureError(f"object store multipart start failed ({type(error).__name__})", kind="object_store") from None
    thread = threading.Thread(target=producer, name=f"silo-capture-{tag}", daemon=True)
    started = clock()
    thread.start()
    digest = hashlib.sha256()
    received = 0
    etags: list[dict[str, Any]] = []
    try:
        next_part = 1
        last_progress = clock()
        while True:
            now = clock()
            if now - started > timeout_seconds:
                raise SiloCaptureError("rootfs capture exceeded its deadline", kind="deadline")
            if now - last_progress > stall_seconds:
                raise SiloCaptureError("rootfs capture produced no part within the stall window", kind="stall")
            listing = sh(sandbox, f"ls -1 {shlex.quote(out_dir)} 2>/dev/null || true", timeout=60)
            if listing.get("exit") != 0:
                raise SiloCaptureError("rootfs part listing failed", kind="listing")
            names = set((listing.get("stdout") or "").split())
            part_name = f"part-{next_part:05d}"
            if part_name in names:
                part_path = f"{out_dir}/{part_name}"
                data = sandbox.fs.download_file(part_path)
                if not isinstance(data, (bytes, bytearray)):
                    raise SiloCaptureError("rootfs part download returned no bytes", kind="download")
                response = s3.upload_part(Bucket=bucket, Key=staging, UploadId=upload_id,
                                          PartNumber=next_part, Body=bytes(data))
                etags.append({"PartNumber": next_part, "ETag": response["ETag"]})
                digest.update(data)
                received += len(data)
                if sh(sandbox, f"rm -f {shlex.quote(part_path)}", timeout=60).get("exit") != 0:
                    raise SiloCaptureError("rootfs part cleanup failed", kind="part_cleanup")
                next_part += 1
                last_progress = clock()
                continue
            if marker in names:
                # Parts are renamed into place before the marker is, so a
                # listing that shows the marker and not the next part is final.
                break
            if not thread.is_alive():
                detail = {"helper": helper}
                exit_code = (run.get("result") or {}).get("exit")
                if isinstance(exit_code, int):
                    detail["helper_exit"] = exit_code
                if run.get("error_type"):
                    detail["helper_error_type"] = run["error_type"]
                raise SiloCaptureError("rootfs stream ended without a result", kind="no_result", detail=detail)
            sleeper(poll_seconds)
        payload = sandbox.fs.download_file(f"{out_dir}/{marker}")
        if helper == HELPER_PYTHON:
            result = json.loads(payload)
        else:
            result = _shell_result(bytes(payload).decode("utf-8", "replace"))
            with contextlib.suppress(Exception):
                result["tar_stderr_tail"] = bytes(sandbox.fs.download_file(f"{out_dir}/tar.stderr")).decode("utf-8", "replace")[-4000:]
            result.setdefault("tar_stderr_tail", "")
        thread.join(timeout=300)
        rec["push_wall_s"] = round(clock() - started, 1)
        if not isinstance(result, dict):
            raise SiloCaptureError("rootfs stream result is malformed", kind="result")
        rec["capture"] = {key: value for key, value in result.items() if key != "error"}
        if result.get("error") is not None or result.get("ok") is not True:
            raise SiloCaptureError("rootfs stream reported failure", kind="stream_failed",
                                   result={**rec["capture"], "error": result.get("error")})
        if (result.get("parts") != next_part - 1 or result.get("compressed_bytes") != received
                or result.get("sha256") != digest.hexdigest()):
            raise SiloCaptureError("received rootfs bytes differ from the sandbox's own count", kind="byte_mismatch")
        s3.complete_multipart_upload(Bucket=bucket, Key=staging, UploadId=upload_id,
                                     MultipartUpload={"Parts": etags})
        upload_id = None
        final = f"{prefix}/{result['sha256']}.tar.gz"
        s3.copy_object(Bucket=bucket, Key=final, CopySource={"Bucket": bucket, "Key": staging})
        s3.delete_object(Bucket=bucket, Key=staging)
        head = s3.head_object(Bucket=bucket, Key=final)
        rec.update(object=f"s3://{bucket}/{final}", object_key=final,
                   object_bytes=head["ContentLength"], sha256=result["sha256"], ok=True)
    except Exception as error:
        if upload_id is not None:
            # Best effort; staging keys are unique per capture tag.
            with contextlib.suppress(Exception):
                s3.abort_multipart_upload(Bucket=bucket, Key=staging, UploadId=upload_id)
        if isinstance(error, SiloCaptureError):
            raise
        raise SiloCaptureError(f"rootfs capture transport failed ({type(error).__name__})",
                               kind="transport", detail={"error_type": type(error).__name__}) from None
    finally:
        rec["finished_utc"] = _utc()
    return rec
