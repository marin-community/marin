#!/usr/bin/env python3
"""Deployment kit for the capability image publisher service (envreg, cw-us-east-02a).

Subcommands (run under the Marin project for the S3 ones, exactly like other
object-store scripts, with CW_KEY_ID/CW_KEY_SECRET exported):

  build-source            deterministic, sha-pinned publisher source archive (prints the sha)
  upload-source           content-addressed upload + readback verification
  render-registry-secret  stdin credential JSON -> Secret manifest on a pipe (never a TTY)
  render-s3-secret        CW_KEY_ID/CW_KEY_SECRET -> Secret manifest on a pipe (never a TTY)
  render-deployment       Deployment manifest from deployment.template.yaml
  health                  heartbeat age + queue depth + returns summary; alarm exit code
  requeue                 operator override: request one more attempt after exhaustion

No subcommand prints a credential; the secret renderers refuse to write to a terminal.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability_pipeline import publication_exchange as exchange
from capability_pipeline.image_publication_handoff import (
    deterministic_targz,
)

NAMESPACE = "envreg"
REGISTRY_SECRET = "capability-registry-publisher"
DEFAULT_REGISTRY = "envreg.208261-marin-gpu.coreweave.app"
DEFAULT_CAPTURE_PREFIX = "s3://marin-us-east-02a/users/muchanem/envrootfs"
# Digests resolved from Docker Hub 2026-09-29 (multi-arch indexes of these tags).
PYTHON_IMAGE = "python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
AWS_CLI_IMAGE = "amazon/aws-cli:2.31.26@sha256:cf1851fa3162c35009b2dc6d2df2797e5b0e9723fe546f545c9fa34a3dc03477"
SOURCE_SCRIPTS = (
    "scripts/__init__.py",
    "scripts/capture_task_images.py",
    "scripts/image_publication_handoff.py",
    "scripts/publish_generic_task_image.py",
    "scripts/run_oneoff_image_publisher.py",
    "scripts/run_image_publisher_service.py",
    "ops/publisher/requirements.txt",
)
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_HOST = re.compile(r"[a-z0-9.-]+(?::[0-9]+)?\Z")


def source_files(root: Path) -> dict[str, bytes]:
    """Pinned publisher source: every controller module plus the publisher scripts."""
    paths = sorted((root / "capability_pipeline").glob("*.py")) + [root / name for name in SOURCE_SCRIPTS]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("trusted publisher source is incomplete or linked")
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}


def build_source(root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("publisher source archive already exists")
    files = source_files(root)
    data = deterministic_targz(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return {"sha256": exchange.sha256(data), "bytes": len(data), "files": sorted(files)}


def upload_source(archive: Path, uri: str, *, fs_factory=None) -> dict:
    data = archive.read_bytes()
    digest = exchange.sha256(data)
    if f"/{digest}/" not in uri:
        raise ValueError("source URI must be content-addressed by the archive sha256")
    bucket, key = exchange.split_s3_uri(uri)
    if fs_factory is None:
        from rigging.filesystem.buckets import filesystem_for

        fs, path = filesystem_for(uri)
    else:
        fs, path = fs_factory(uri)
    try:
        existing = fs.cat_file(path)
    except FileNotFoundError:
        existing = None
    if existing is None or exchange.sha256(existing) != digest:
        fs.pipe_file(path, data)
        if exchange.sha256(fs.cat_file(path)) != digest:
            raise OSError("publisher source readback differs")
    return {"state": "uploaded_verified", "uri": f"s3://{bucket}/{key}", "sha256": digest}


def _refuse_tty() -> None:
    if sys.stdout.isatty():
        raise SystemExit("refusing to write a Secret manifest to a terminal; pipe it to kubectl apply -f -")


def _secret_manifest(name: str, data: dict[str, bytes]) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError("Secret name is invalid")
    return json.dumps({
        "apiVersion": "v1", "kind": "Secret", "type": "Opaque",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": {
            "app.kubernetes.io/name": "capability-image-publisher",
            "app.kubernetes.io/part-of": "capability-env-gen"}},
        "data": {key: base64.b64encode(value).decode() for key, value in data.items()},
    })


def render_registry_secret(raw: bytes, expected_registry: str) -> str:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("registry credential is not JSON") from None
    if (not isinstance(value, dict) or value.get("registry") != expected_registry
            or not isinstance(value.get("user"), str) or not value["user"] or ":" in value["user"]
            or not isinstance(value.get("password"), str) or not value["password"]):
        raise ValueError("registry credential lacks registry/user/password for the expected registry")
    body = json.dumps({"registry": value["registry"], "user": value["user"],
                       "password": value["password"]}).encode()
    return _secret_manifest(REGISTRY_SECRET, {"credentials.json": body})


def render_s3_secret(name: str, environ=os.environ) -> str:
    key, secret = environ.get("CW_KEY_ID"), environ.get("CW_KEY_SECRET")
    if not key or not secret:
        raise ValueError("CW_KEY_ID/CW_KEY_SECRET must be exported")
    return _secret_manifest(name, {"accesskey": key.encode(), "secretkey": secret.encode()})


def render_deployment(*, source_sha256: str, source_uri: str, queue_uri: str, s3_secret: str,
                      registry_host: str = DEFAULT_REGISTRY, capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
                      max_rootfs_bytes: int = 20 << 30, concurrency: int = 2, poll_seconds: int = 15,
                      python_image: str = PYTHON_IMAGE, aws_cli_image: str = AWS_CLI_IMAGE) -> str:
    exchange._checked_sha(source_sha256)
    exchange.split_s3_uri(source_uri)
    exchange.split_s3_uri(queue_uri)
    exchange.split_s3_uri(capture_prefix)
    if f"/{source_sha256}/" not in source_uri:
        raise ValueError("source URI must carry the pinned sha256")
    if not _NAME.fullmatch(s3_secret) or not _HOST.fullmatch(registry_host):
        raise ValueError("Secret name or registry host is invalid")
    for number in (max_rootfs_bytes, concurrency, poll_seconds):
        if type(number) is not int or number <= 0:
            raise ValueError("numeric settings must be positive integers")
    if not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be between 1 and 16")
    for image in (python_image, aws_cli_image):
        if not re.fullmatch(r"[a-z0-9./:_-]+(@sha256:[0-9a-f]{64})?", image):
            raise ValueError("container image reference is invalid")
    template = string.Template((HERE / "deployment.template.yaml").read_text())
    return template.substitute(
        SOURCE_SHA256=source_sha256, SOURCE_URI=source_uri, QUEUE_URI=queue_uri.rstrip("/"),
        S3_SECRET=s3_secret, REGISTRY_HOST=registry_host, CAPTURE_PREFIX=capture_prefix.rstrip("/"),
        MAX_ROOTFS_BYTES=str(max_rootfs_bytes), CONCURRENCY=str(concurrency),
        POLL_SECONDS=str(poll_seconds), PYTHON_IMAGE=python_image, AWS_CLI_IMAGE=aws_cli_image,
    )


def queue_summary(queue, *, clock=time.time, workers: int = 16) -> dict:
    """Heartbeat plus every packet's disposition (requests joined with returns)."""
    raw = queue.get(exchange.HEARTBEAT_KEY)
    heartbeat = None
    if raw is not None:
        try:
            heartbeat = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            heartbeat = {"unparseable": True}
    age = exchange.heartbeat_age(queue, clock)
    request_keys = queue.keys("requests")
    by_sha: dict[str, list[str]] = {}
    for key in request_keys:
        parts = key.split("/")
        if len(parts) == 3 and re.fullmatch(r"[0-9a-f]{64}", parts[1]):
            by_sha.setdefault(parts[1], []).append(key)

    def one(sha: str) -> dict:
        keys = by_sha[sha]
        row = {"packet_sha": sha, "requested_attempt": exchange.requested_attempt(keys, sha),
               "has_request": exchange.request_key(sha) in keys}
        try:
            result = exchange.parse_result(queue.get(exchange.result_key(sha)), sha)
        except ValueError:
            return {**row, "disposition": "result_invalid"}
        if result is None:
            return {**row, "disposition": "pending"}
        row.update(attempt=result["attempt"], reason=result["reason"], completed_utc=result["completed_utc"])
        if result["state"] == "published":
            return {**row, "disposition": "published", "images": result.get("images", {})}
        if result["state"] == "rejected":
            return {**row, "disposition": "rejected", "rejection_class": result["rejection_class"]}
        if result["attempt"] < row["requested_attempt"]:
            return {**row, "disposition": "pending"}
        if result["attempt"] >= exchange.MAX_PUBLICATION_ATTEMPTS:
            return {**row, "disposition": "transient_exhausted"}
        return {**row, "disposition": "transient_awaiting_resubmission"}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, sorted(by_sha)))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return {"heartbeat": heartbeat, "heartbeat_age_seconds": None if age is None else round(age, 1),
            "counts": counts, "packets": rows}


def health(queue, *, max_age: float, expect_source: str | None = None, clock=time.time) -> tuple[int, dict]:
    """Exit 0 healthy; 1 heartbeat stale/absent/degraded or wrong source."""
    summary = queue_summary(queue, clock=clock)
    heartbeat, age = summary["heartbeat"] or {}, summary["heartbeat_age_seconds"]
    problems = []
    if age is None:
        problems.append("heartbeat_absent")
    elif age > max_age:
        problems.append(f"heartbeat_stale_{int(age)}s")
    if heartbeat.get("state") not in (None, "running"):
        problems.append(f"service_{heartbeat.get('state')}: {heartbeat.get('reason')}")
    if expect_source and heartbeat.get("source_sha256") != expect_source:
        problems.append("heartbeat_from_other_source")
    summary["problems"] = problems
    return (1 if problems else 0), summary


def _print_health(code: int, summary: dict) -> None:
    heartbeat = summary["heartbeat"] or {}
    age = summary["heartbeat_age_seconds"]
    print(f"{'OK' if code == 0 else 'ALARM'} heartbeat age={'absent' if age is None else f'{age}s'} "
          f"state={heartbeat.get('state')} reason={heartbeat.get('reason')} host={heartbeat.get('host')} "
          f"source={str(heartbeat.get('source_sha256'))[:12]} loop={heartbeat.get('loop')} "
          f"in_flight={len(heartbeat.get('in_flight') or [])} processed={heartbeat.get('processed')}")
    counts = summary["counts"]
    depth = counts.get("pending", 0)
    print(f"queue depth={depth} " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for row in summary["packets"]:
        if row["disposition"] == "published":
            continue
        detail = row.get("rejection_class") or row.get("reason") or ""
        print(f"  {row['disposition'].upper()} {row['packet_sha'][:16]} attempt={row.get('attempt', '-')}"
              f"/{row['requested_attempt']} {detail[:160]}")
    for problem in summary.get("problems", []):
        print(f"  PROBLEM {problem}")


def requeue(queue, packet_sha: str) -> dict:
    result = exchange.parse_result(queue.get(exchange.result_key(packet_sha)), packet_sha)
    if result is None or result["state"] != "transient_failure":
        raise ValueError("only a packet whose last result is a transient failure can be requeued")
    keys = queue.keys(f"requests/{packet_sha}")
    if exchange.request_key(packet_sha) not in keys:
        raise ValueError("packet request is absent from the queue")
    attempt = max(result["attempt"], exchange.requested_attempt(keys, packet_sha)) + 1
    marker = {"schema_version": exchange.RESUBMIT_SCHEMA, "packet_sha256": packet_sha,
              "attempt": attempt, "requested_utc": exchange.utc_now(), "operator_requeue": True}
    queue.put(exchange.resubmit_key(packet_sha, attempt), json.dumps(marker, sort_keys=True).encode() + b"\n")
    return {"state": "requeued", "packet_sha": packet_sha, "attempt": attempt}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-source")
    build.add_argument("--root", type=Path, default=ROOT)
    build.add_argument("--output", type=Path, required=True)
    upload = commands.add_parser("upload-source")
    upload.add_argument("--archive", type=Path, required=True)
    upload.add_argument("--uri", required=True)
    registry = commands.add_parser("render-registry-secret")
    registry.add_argument("--expected-registry", default=DEFAULT_REGISTRY)
    s3 = commands.add_parser("render-s3-secret")
    s3.add_argument("--name", default="capability-publisher-s3")
    deployment = commands.add_parser("render-deployment")
    deployment.add_argument("--source-sha256", required=True)
    deployment.add_argument("--source-uri", required=True)
    deployment.add_argument("--queue-uri", default=exchange.DEFAULT_QUEUE_URI)
    deployment.add_argument("--s3-secret", default="capability-publisher-s3")
    deployment.add_argument("--registry-host", default=DEFAULT_REGISTRY)
    deployment.add_argument("--capture-prefix", default=DEFAULT_CAPTURE_PREFIX)
    deployment.add_argument("--max-rootfs-bytes", type=int, default=20 << 30)
    deployment.add_argument("--concurrency", type=int, default=2)
    deployment.add_argument("--poll-seconds", type=int, default=15)
    deployment.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("health")
    check.add_argument("--queue", default=exchange.DEFAULT_QUEUE_URI)
    check.add_argument("--max-age", type=float, default=120.0)
    check.add_argument("--expect-source")
    check.add_argument("--wait", type=float, default=0.0, help="poll up to this many seconds for health")
    check.add_argument("--json", action="store_true")
    again = commands.add_parser("requeue")
    again.add_argument("--queue", default=exchange.DEFAULT_QUEUE_URI)
    again.add_argument("--packet", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-source":
        result = build_source(args.root, args.output)
        print(json.dumps({key: value for key, value in result.items() if key != "files"}), file=sys.stderr)
        print(result["sha256"])
    elif args.command == "upload-source":
        print(json.dumps(upload_source(args.archive, args.uri)))
    elif args.command == "render-registry-secret":
        _refuse_tty()
        sys.stdout.write(render_registry_secret(sys.stdin.buffer.read(), args.expected_registry))
    elif args.command == "render-s3-secret":
        _refuse_tty()
        sys.stdout.write(render_s3_secret(args.name))
    elif args.command == "render-deployment":
        if args.output.exists():
            raise SystemExit("deployment output already exists")
        args.output.write_text(render_deployment(
            source_sha256=args.source_sha256, source_uri=args.source_uri, queue_uri=args.queue_uri,
            s3_secret=args.s3_secret, registry_host=args.registry_host, capture_prefix=args.capture_prefix,
            max_rootfs_bytes=args.max_rootfs_bytes, concurrency=args.concurrency,
            poll_seconds=args.poll_seconds))
        print(str(args.output))
    elif args.command == "health":
        queue = exchange.open_queue(args.queue)
        deadline = time.time() + args.wait
        while True:
            code, summary = health(queue, max_age=args.max_age, expect_source=args.expect_source)
            if code == 0 or time.time() >= deadline:
                break
            time.sleep(min(15.0, max(1.0, deadline - time.time())))
        if args.json:
            print(json.dumps(summary, sort_keys=True, indent=2))
        else:
            _print_health(code, summary)
        return code
    else:
        print(json.dumps(requeue(exchange.open_queue(args.queue), args.packet)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - never echo provider or credential text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "error": str(error)[:300] if isinstance(error, ValueError) else None}),
              file=sys.stderr)
        raise SystemExit(2) from None
