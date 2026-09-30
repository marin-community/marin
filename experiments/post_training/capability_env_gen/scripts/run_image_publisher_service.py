#!/usr/bin/env python3
"""Long-running trusted task-image publisher: S3 request queue -> registry -> returns.

Runs as the single-replica Deployment in namespace ``envreg`` (see
``ops/publisher/``). It is the only holder of the registry credential. Every
packet is untrusted data: it is validated by the same ``open_packet`` /
``publish_packet_role`` path as the one-off publisher, layer bytes are read
by pinned code confined to the reviewed capture-object prefixes, and nothing
from a packet is imported or executed.

Idempotence: packets are content-addressed; ``returns/<sha>/result.json`` is
written last and a ``published``/``rejected`` result is final. Publication is
by digest (re-pushing identical bytes is harmless), so a crash mid-publish is
redone from scratch; ``started-<attempt>.json`` bounds crash loops.

``--queue DIR --dry-run-object-root DIR --dry-run-registry-root DIR`` runs the
same loop against a local directory queue, local capture objects and a local
directory registry (tests and rehearsal); ``--drain`` exits when idle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import signal
import tarfile
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from capability_pipeline.oci_artifact import ImageArtifactError
from capability_pipeline.publication_exchange import (
    FINAL_STATES,
    HEARTBEAT_KEY,
    HEARTBEAT_SCHEMA,
    STARTED_SCHEMA,
    LocalObjectClient,
    LocalQueue,
    ObjectStoreReader,
    open_queue,
    parse_result,
    receipt_key,
    request_key,
    requested_attempt,
    result_document,
    result_key,
    s3_client,
    sha256,
    started_key,
    utc_now,
)
from capability_pipeline.rootfs_review import RootfsReviewError
from scripts.run_oneoff_image_publisher import open_packet, publish_packet_role

DEFAULT_REGISTRY = "envreg.208261-marin-gpu.coreweave.app"
DEFAULT_CAPTURE_PREFIX = "s3://marin-us-east-02a/users/muchanem/envrootfs"
DEFAULT_MAX_ROOTFS_BYTES = 20 << 30  # the reviewed c32 publication bound
# Capacity guards, not verdicts: a packet carries the retained GLM review input
# (a workspace snapshot), and real construct-003 packets reach 600+ MiB. Going
# over a bound is transient (bounded retries, then an operator hold) because
# the fix is raising the bound, not failing the item. unpack_handoff holds the
# expanded packet in memory: keep the expansion bound x concurrency under the
# pod's memory limit.
MAX_PACKET_BYTES = 8 << 30
MAX_PACKET_MEMBERS = 200_000
MAX_PACKET_EXPANDED_BYTES = 6 << 30
MAX_STARTS_PER_ATTEMPT = 3

_PACKET_ERRORS = (ValueError, TypeError, KeyError, tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile)
# ValueErrors that describe this publisher's own environment, never the packet.
_TRANSIENT_MESSAGES = frozenset({
    "isolated trusted publisher credentials are absent",
    "publisher credential mount is unavailable",
    "publisher rootfs limit is invalid",
    "layer download directory is linked",
    "layer download directory is unavailable",
    "layer download target already exists or is linked",
    "capture object body is unavailable",
    # A short or corrupted stream is indistinguishable from a network fault;
    # a real mismatch repeats and exhausts the bounded attempts instead.
    "capture object byte count or SHA-256 differs",
    "capture object exceeds receipt byte count",
    # Publisher policy, not the packet: widen --capture-object-prefix and requeue.
    "capture object is outside the publisher's allowed prefixes",
})
_REJECTION_MESSAGES = {
    "publisher credential host differs from reviewed plan": "registry_host_mismatch",
    "capture object size differs before download": "capture_object_mismatch",
    "capture receipt lacks a complete sanitized object identity": "capture_receipt_invalid",
    "capture receipt object identity is unsafe": "capture_receipt_invalid",
}


class Rejected(Exception):
    def __init__(self, rejection_class: str, reason: str):
        super().__init__(reason)
        self.rejection_class, self.reason = rejection_class, reason


class Transient(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def safe_reason(error: BaseException, secrets: tuple[str, ...] = ()) -> str:
    """Our own validation messages are safe; provider errors keep only type/code."""
    if isinstance(error, (ValueError, TypeError, KeyError)) or type(error).__name__ == "RegistryError":
        text = f"{type(error).__name__}: {error}"
    else:
        code = (getattr(error, "response", None) or {}).get("Error", {}).get("Code")
        text = type(error).__name__ + (f" ({code})" if isinstance(code, str) else "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text[:500]


def _classify(error: Exception, secrets: tuple[str, ...]) -> Exception:
    """Pre-registry failure -> Rejected (packet's fault) or Transient (ours/network)."""
    if isinstance(error, (Rejected, Transient)):
        return error
    reason = safe_reason(error, secrets)
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return Rejected("capture_object_missing", reason)
        return Transient("object_store_error: " + reason)
    if isinstance(error, ImageArtifactError):
        return Rejected("layer_invalid", reason)
    if isinstance(error, RootfsReviewError):
        return Rejected("rootfs_review_failed", reason)
    if isinstance(error, ValueError):
        message = str(error)
        if message in _TRANSIENT_MESSAGES:
            return Transient(reason)
        if message in _REJECTION_MESSAGES:
            return Rejected(_REJECTION_MESSAGES[message], reason)
        return Rejected("publication_validation_failed", reason)
    if isinstance(error, (TypeError, KeyError)):
        return Rejected("publication_validation_failed", reason)
    return Transient(reason)


def precheck_archive(path: Path) -> None:
    """Bound a packet's expanded size before ``unpack_handoff`` reads it into memory."""
    total = members = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            members += 1
            total += max(0, member.size)
            if members > MAX_PACKET_MEMBERS or total > MAX_PACKET_EXPANDED_BYTES:
                raise Transient(f"publisher_capacity_exceeded: handoff expands past "
                                f"{MAX_PACKET_EXPANDED_BYTES} bytes or {MAX_PACKET_MEMBERS} members")


class _RegistryPhase:
    """Registry-client factory wrapper: failures after ``publish_layer`` starts
    are transport (transient), never a verdict on the packet."""

    def __init__(self, factory):
        self.factory, self.entered = factory, False

    def __call__(self, host, repository, user, password):
        client = self.factory(host, repository, user, password)
        phase = self

        class _Tracked:
            def __init__(self):
                self.host, self.repository = client.host, client.repository

            def publish_layer(self, *args, **kwargs):
                phase.entered = True
                return client.publish_layer(*args, **kwargs)

        return _Tracked()


class LocalDirectoryRegistry:
    """Dry-run registry: validates and stores exactly what ``RegistryClient`` pushes."""

    def __init__(self, root: Path, host: str, repository: str):
        self.root, self.host, self.repository = Path(root), host, repository

    @classmethod
    def factory(cls, root: Path):
        return lambda host, repository, user, password: cls(root, host, repository)

    def publish_layer(self, path: Path, *, expected_digest: str, expected_bytes: int,
                      max_uncompressed_bytes: int, image_config, architecture: str,
                      operating_system: str) -> dict:
        from capability_pipeline.oci_artifact import (
            build_oci_metadata,
            digest,
            validate_layer,
        )

        with path.open("rb") as stream:
            layer = validate_layer(stream, expected_digest=expected_digest, expected_bytes=expected_bytes,
                                   max_uncompressed_bytes=max_uncompressed_bytes)
        config, manifest = build_oci_metadata(layer, image_config=image_config,
                                              architecture=architecture, operating_system=operating_system)
        repository = self.root / self.repository
        (repository / "blobs").mkdir(parents=True, exist_ok=True)
        (repository / "manifests").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, repository / "blobs" / layer.compressed_digest.replace(":", "-"))
        (repository / "blobs" / digest(config).replace(":", "-")).write_bytes(config)
        manifest_digest = digest(manifest)
        (repository / "manifests" / manifest_digest.replace(":", "-")).write_bytes(manifest)
        return {"schema_version": "capability-oci-transport-v1", "state": "integrity_verified",
                "image": self.host + "/" + self.repository + "@" + manifest_digest,
                "manifest_digest": manifest_digest, "manifest_bytes": len(manifest),
                "config_digest": digest(config), "config_bytes": len(config),
                "layer_digest": layer.compressed_digest, "layer_bytes": layer.compressed_bytes,
                "diff_id": layer.diff_id, "uncompressed_bytes": layer.uncompressed_bytes,
                "task_acceptance": "not_evaluated", "privacy_review": "caller_required"}


class PublisherService:
    def __init__(self, *, queue, work_root: Path, credentials_file: Path, private_credentials: Path,
                 expected_registry: str, reader, registry_client_factory=None,
                 max_rootfs_bytes: int = DEFAULT_MAX_ROOTFS_BYTES, concurrency: int = 2,
                 source_sha256: str | None = None, host: str | None = None, clock=time.time,
                 log=None):
        if type(concurrency) is not int or not 1 <= concurrency <= 16:
            raise ValueError("publisher concurrency must be between 1 and 16")
        if type(max_rootfs_bytes) is not int or max_rootfs_bytes <= 0:
            raise ValueError("publisher rootfs limit must be positive")
        self.queue, self.work_root = queue, Path(work_root)
        self.credentials_file, self.private_credentials = Path(credentials_file), Path(private_credentials)
        self.expected_registry, self.reader = expected_registry, reader
        self.registry_client_factory = registry_client_factory
        self.max_rootfs_bytes, self.concurrency = max_rootfs_bytes, concurrency
        self.source_sha256, self.host = source_sha256, host or os.environ.get("HOSTNAME", "local")
        self.clock, self._log = clock, log or self._print
        self.final: set[str] = set()
        self.in_flight: dict[str, Future] = {}
        self.counts: Counter[str] = Counter()
        self.queue_stats: dict[str, int] = {}
        self._secrets: tuple[str, ...] = ()

    # ------------------------------------------------------------------ logs

    @staticmethod
    def _print(record: dict) -> None:
        print(json.dumps(record, sort_keys=True), flush=True)

    def log(self, event: str, **fields) -> None:
        record = {"utc": utc_now(self.clock), "event": event, **fields}
        text = json.dumps(record, sort_keys=True)
        if any(secret and secret in text for secret in self._secrets):
            record = {"utc": record["utc"], "event": event, "redacted": True}
        self._log(record)

    # ----------------------------------------------------------- credential

    def refresh_credentials(self) -> str | None:
        """Re-read the mounted Secret (rotation-safe) into a private regular file.

        Kubernetes Secret volume entries are symlinks, which the shared publish
        path rejects, so a 0400 copy lives on a memory-backed volume.
        """
        try:
            value = json.loads(self.credentials_file.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return "credential_unreadable"
        if not isinstance(value, dict) or value.get("registry") != self.expected_registry:
            return "credential_registry_differs_from_expected"
        user, password = value.get("user"), value.get("password")
        if not isinstance(user, str) or not user or ":" in user or not isinstance(password, str) or not password:
            return "credential_incomplete"
        self._secrets = (user, password)
        data = json.dumps({"registry": value["registry"], "user": user, "password": password}).encode()
        try:
            current = self.private_credentials.read_bytes() if self.private_credentials.is_file() else None
            if current != data:
                self.private_credentials.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.private_credentials.with_name(f".{self.private_credentials.name}.part")
                temporary.unlink(missing_ok=True)
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                os.replace(temporary, self.private_credentials)
        except OSError:
            return "credential_copy_failed"
        return None

    def drop_credentials(self) -> None:
        self.private_credentials.unlink(missing_ok=True)

    # ----------------------------------------------------------------- scan

    def scan(self) -> list[tuple[str, int]]:
        by_sha: dict[str, list[str]] = {}
        for key in self.queue.keys("requests"):
            parts = key.split("/")
            if len(parts) == 3 and len(parts[1]) == 64 and all(c in "0123456789abcdef" for c in parts[1]):
                by_sha.setdefault(parts[1], []).append(key)
        work, awaiting = [], 0
        for sha, keys in sorted(by_sha.items()):
            if request_key(sha) not in keys or sha in self.final or sha in self.in_flight:
                continue
            attempt = requested_attempt(keys, sha)
            try:
                result = parse_result(self.queue.get(result_key(sha)), sha)
            except ValueError:
                result = None  # malformed: redo the attempt and rewrite it
            if result is not None and result["state"] in FINAL_STATES:
                self.final.add(sha)
                continue
            if result is not None and result["attempt"] >= attempt:
                awaiting += 1  # transient; the controller decides whether to resubmit
                continue
            work.append((sha, attempt))
        self.queue_stats = {"requests": len(by_sha), "final": len(self.final),
                            "actionable": len(work), "awaiting_resubmission": awaiting,
                            "in_flight": len(self.in_flight)}
        return work

    # -------------------------------------------------------------- process

    def _claim(self, sha: str, attempt: int) -> bool:
        raw = self.queue.get(started_key(sha, attempt))
        starts = 0
        if raw is not None:
            try:
                starts = int(json.loads(raw).get("starts", 0))
            except (ValueError, TypeError, AttributeError):
                starts = MAX_STARTS_PER_ATTEMPT
        if starts >= MAX_STARTS_PER_ATTEMPT:
            return False
        document = {"schema_version": STARTED_SCHEMA, "packet_sha256": sha, "attempt": attempt,
                    "starts": starts + 1, "last_started_utc": utc_now(self.clock), "host": self.host,
                    "source_sha256": self.source_sha256}
        self.queue.put(started_key(sha, attempt), json.dumps(document, sort_keys=True).encode() + b"\n")
        return True

    def process(self, sha: str, attempt: int) -> str:
        """Publish one packet attempt and commit its result; returns the result state."""
        if not self._claim(sha, attempt):
            state, reason, fields = ("transient_failure",
                                     "publisher_interrupted_repeatedly: attempt restarted too often", {})
        else:
            work = self.work_root / sha
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            try:
                state, reason, fields = self._publish(sha, work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
        self.queue.put(result_key(sha), result_document(
            packet_sha=sha, attempt=attempt, state=state, reason=reason, clock=self.clock,
            publisher_source_sha256=self.source_sha256, publisher_host=self.host, **fields))
        if state in FINAL_STATES:
            self.final.add(sha)
        self.counts[state] += 1
        self.log("packet_result", packet_sha=sha, attempt=attempt, state=state, reason=reason,
                 **{key: value for key, value in fields.items() if key in {"rejection_class", "images"}})
        return state

    def _publish(self, sha: str, work: Path) -> tuple[str, str, dict]:
        try:
            size = self.queue.size(request_key(sha))
            if size is None:
                raise Transient("request_absent")
            if size > MAX_PACKET_BYTES:
                raise Transient(f"publisher_capacity_exceeded: handoff is {size} bytes")
            archive = work / "handoff.tar.gz"
            downloaded = self.queue.download(request_key(sha), archive)
            if downloaded is None:
                raise Transient("request_absent")
            digest, received = downloaded
            if received != size:
                raise Transient("request_download_incomplete")
            if digest != sha:
                raise Rejected("packet_digest_mismatch", "request bytes differ from their content address")
            packet = work / "packet"
            try:
                precheck_archive(archive)
                manifest = open_packet(archive, packet)
            except _PACKET_ERRORS as error:
                raise Rejected("packet_invalid", safe_reason(error, self._secrets)) from None
            if not self.private_credentials.is_file():
                raise Transient("credential_unavailable")
            receipts, images = {}, {}
            for role in manifest["roles"]:
                path = self._publish_role(packet, manifest, role, work)
                body = path.read_bytes()
                receipts[role] = {"key": receipt_key(sha, role), "sha256": sha256(body), "bytes": len(body)}
                images[role] = json.loads(body)["publication"]["image"]
            for role in manifest["roles"]:
                self.queue.put(receipt_key(sha, role), (work / f"publication-{role}.json").read_bytes())
            return "published", "published_pending_cold_pull", {
                "roles": manifest["roles"], "receipts": receipts, "images": images,
                "packet_manifest_sha256": sha256((packet / "manifest.json").read_bytes())}
        except Rejected as rejected:
            return "rejected", rejected.reason, {"rejection_class": rejected.rejection_class}
        except Transient as transient:
            return "transient_failure", transient.reason, {}
        except Exception as error:  # noqa: BLE001 - unknown faults never condemn a packet.
            return "transient_failure", "unexpected: " + safe_reason(error, self._secrets), {}

    def _publish_role(self, packet: Path, manifest: dict, role: str, work: Path) -> Path:
        phase = _RegistryPhase(self.registry_client_factory or _real_registry_client)
        try:
            return publish_packet_role(packet, manifest, role, work, self.max_rootfs_bytes,
                                       self.private_credentials, presigner_factory=lambda: self.reader,
                                       registry_client_factory=phase)
        except Exception as error:  # noqa: BLE001 - classified below
            if phase.entered:
                raise Transient(f"{role}: registry_publish_failed: "
                                + safe_reason(error, self._secrets)) from None
            classified = _classify(error, self._secrets)
            if isinstance(classified, Rejected):
                raise Rejected(classified.rejection_class, f"{role}: {classified.reason}") from None
            raise Transient(f"{role}: {classified.reason}") from None

    # ----------------------------------------------------------------- loop

    def heartbeat(self, *, loop: int, state: str, reason: str | None, poll_seconds: float,
                  loop_seconds: float) -> None:
        document = {"schema_version": HEARTBEAT_SCHEMA, "epoch": self.clock(), "utc": utc_now(self.clock),
                    "state": state, "reason": reason, "host": self.host,
                    "source_sha256": self.source_sha256, "loop": loop, "poll_seconds": poll_seconds,
                    "concurrency": self.concurrency, "in_flight": sorted(self.in_flight),
                    "processed": dict(self.counts), "queue": self.queue_stats,
                    "last_loop_seconds": round(loop_seconds, 3)}
        self.queue.put(HEARTBEAT_KEY, json.dumps(document, sort_keys=True, indent=2).encode() + b"\n")

    def _run_one(self, sha: str, attempt: int) -> None:
        try:
            self.log("packet_start", packet_sha=sha, attempt=attempt)
            self.process(sha, attempt)
        except Exception as error:  # noqa: BLE001 - logged; the next scan retries.
            self.log("packet_error", packet_sha=sha, attempt=attempt,
                     error=safe_reason(error, self._secrets))

    def serve(self, *, poll_seconds: float = 15.0, stop: threading.Event | None = None,
              drain: bool = False, liveness_file: Path | None = None) -> None:
        stop = stop or threading.Event()
        loop = 0
        self.log("service_start", source_sha256=self.source_sha256, queue=self.queue.describe(),
                 concurrency=self.concurrency, registry=self.expected_registry)
        with ThreadPoolExecutor(self.concurrency, thread_name_prefix="publish") as pool:
            while not stop.is_set():
                started = self.clock()
                for sha in [sha for sha, future in self.in_flight.items() if future.done()]:
                    del self.in_flight[sha]
                problem = self.refresh_credentials()
                work: list[tuple[str, int]] = []
                if problem is None:
                    try:
                        work = self.scan()
                    except Exception as error:  # noqa: BLE001 - queue outage: report, keep looping
                        problem = "queue_error: " + safe_reason(error, self._secrets)
                for sha, attempt in work[: max(0, self.concurrency - len(self.in_flight))]:
                    self.in_flight[sha] = pool.submit(self._run_one, sha, attempt)
                try:
                    self.heartbeat(loop=loop, state="running" if problem is None else "degraded",
                                   reason=problem, poll_seconds=poll_seconds,
                                   loop_seconds=self.clock() - started)
                except Exception as error:  # noqa: BLE001
                    self.log("heartbeat_error", error=safe_reason(error, self._secrets))
                if problem is not None and loop % 20 == 0:
                    self.log("degraded", reason=problem)
                if liveness_file is not None:
                    liveness_file.parent.mkdir(parents=True, exist_ok=True)
                    liveness_file.write_text(str(self.clock()))
                loop += 1
                if drain and not work and not self.in_flight:
                    break
                if self.in_flight:
                    wait(list(self.in_flight.values()), timeout=None if drain else poll_seconds,
                         return_when=FIRST_COMPLETED)
                elif not drain:
                    stop.wait(poll_seconds)
        self.drop_credentials()
        self.log("service_stop", processed=dict(self.counts))


def _real_registry_client(host, repository, user, password):
    from capability_pipeline.oci_registry import RegistryClient

    return RegistryClient(host, repository, user, password)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True, help="s3://bucket/prefix, or a local directory for dry runs")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, required=True,
                        help="mounted registry Secret JSON (registry, user, password)")
    parser.add_argument("--private-credentials", type=Path, required=True,
                        help="0400 copy location on a memory-backed volume")
    parser.add_argument("--expected-registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--capture-object-prefix", action="append", default=[])
    parser.add_argument("--max-rootfs-bytes", type=int, default=DEFAULT_MAX_ROOTFS_BYTES)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--source-sha256")
    parser.add_argument("--liveness-file", type=Path)
    parser.add_argument("--drain", action="store_true", help="exit once no actionable work remains")
    parser.add_argument("--dry-run-object-root", type=Path)
    parser.add_argument("--dry-run-registry-root", type=Path)
    args = parser.parse_args(argv)
    prefixes = args.capture_object_prefix or [DEFAULT_CAPTURE_PREFIX]
    dry = args.dry_run_object_root is not None or args.dry_run_registry_root is not None
    if dry:
        if args.queue.startswith("s3://") or args.dry_run_object_root is None or args.dry_run_registry_root is None:
            raise SystemExit("dry run needs a local --queue and both --dry-run-* roots")
        queue = LocalQueue(Path(args.queue))
        reader = ObjectStoreReader(LocalObjectClient(args.dry_run_object_root), prefixes)
        factory = LocalDirectoryRegistry.factory(args.dry_run_registry_root)
    else:
        queue = open_queue(args.queue, client_factory=s3_client)
        reader = ObjectStoreReader(s3_client(), prefixes)
        factory = None
    service = PublisherService(
        queue=queue, work_root=args.work_root, credentials_file=args.credentials_file,
        private_credentials=args.private_credentials, expected_registry=args.expected_registry,
        reader=reader, registry_client_factory=factory, max_rootfs_bytes=args.max_rootfs_bytes,
        concurrency=args.concurrency, source_sha256=args.source_sha256,
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    service.serve(poll_seconds=args.poll_seconds, stop=stop, drain=args.drain,
                  liveness_file=args.liveness_file)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - never print credential-bearing provider text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
