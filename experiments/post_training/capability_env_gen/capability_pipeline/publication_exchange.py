"""Credential-free request queue between construction controllers and the
trusted image publisher service.

Layout under the queue root (``s3://bucket/prefix`` or a local directory)::

    requests/<packet_sha>/handoff.tar.gz          credential-free handoff, content-addressed
    requests/<packet_sha>/resubmit-<n>.json       controller asks for attempt n after a transient failure
    returns/<packet_sha>/started-<n>.json         publisher crash accounting for attempt n
    returns/<packet_sha>/publication-<role>.json  publisher receipts (published only)
    returns/<packet_sha>/result.json              commit record: published | rejected | transient_failure
    service/heartbeat.json                        publisher liveness, rewritten every loop

The queue carries data only. Construction controllers hold object-store
credentials but never registry credentials; the publisher holds the registry
credential and treats every packet as untrusted data (it never imports or
executes packet bytes). A controller imports a returned receipt only through
``import_publication_for_item``'s exactness checks against its live inputs.

This module is stdlib-only at import time; the publisher image has botocore
and the construction job has fsspec/rigging, each imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_QUEUE_URI = "s3://marin-us-east-02a/users/muchanem/capability-pipeline/publication"
QUEUE_ENV = "CAPABILITY_PUBLICATION_QUEUE"
DISABLED_VALUES = frozenset({"off", "disabled", "manual"})

RESULT_SCHEMA = "capability-image-publication-result-v1"
REQUEST_SCHEMA = "capability-image-publication-request-v1"
RESUBMIT_SCHEMA = "capability-image-publication-resubmit-v1"
STARTED_SCHEMA = "capability-image-publication-started-v1"
HEARTBEAT_SCHEMA = "capability-image-publisher-heartbeat-v1"
HEARTBEAT_KEY = "service/heartbeat.json"

RESULT_STATES = frozenset({"published", "rejected", "transient_failure"})
FINAL_STATES = frozenset({"published", "rejected"})
# Rejections a builder can fix with a new image-capture-request (a fresh
# attempt, hence a new packet). Everything else is a packet/infrastructure
# verdict. Reported as ``builder_repairable`` on ``failed_terminal``.
BUILDER_REPAIRABLE_REJECTIONS = frozenset({"rootfs_review_failed"})
ROLES = frozenset({"candidate", "private_verifier"})

# Total publisher attempts per packet, including the first. Resubmissions wait
# RESUBMIT_BASE_SECONDS * 2**(attempt-1) after the transient result: 5, 10,
# 20, 40 minutes, so a registry or object-store outage of ~75 minutes is
# absorbed before the item holds for an operator.
MAX_PUBLICATION_ATTEMPTS = 5
RESUBMIT_BASE_SECONDS = 300
HEARTBEAT_STALE_SECONDS = 300
QUEUE_UNREACHABLE_BACKOFF_SECONDS = 120

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_RESUBMIT = re.compile(r"requests/([0-9a-f]{64})/resubmit-([1-9][0-9]{0,3})\.json\Z")


# --------------------------------------------------------------------------- keys


def _checked_sha(value: object) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError("packet identity is not a SHA-256")
    return value


def request_key(packet_sha: str) -> str:
    return f"requests/{_checked_sha(packet_sha)}/handoff.tar.gz"


def resubmit_key(packet_sha: str, attempt: int) -> str:
    if type(attempt) is not int or attempt < 2:
        raise ValueError("resubmission attempt must be at least 2")
    return f"requests/{_checked_sha(packet_sha)}/resubmit-{attempt}.json"


def started_key(packet_sha: str, attempt: int) -> str:
    if type(attempt) is not int or attempt < 1:
        raise ValueError("publication attempt must be positive")
    return f"returns/{_checked_sha(packet_sha)}/started-{attempt}.json"


def result_key(packet_sha: str) -> str:
    return f"returns/{_checked_sha(packet_sha)}/result.json"


def receipt_key(packet_sha: str, role: str) -> str:
    if role not in ROLES:
        raise ValueError("publication role is invalid")
    return f"returns/{_checked_sha(packet_sha)}/publication-{role}.json"


def requested_attempt(keys: list[str], packet_sha: str) -> int:
    """Highest attempt a controller has requested (1 when never resubmitted)."""
    attempts = [int(match.group(2)) for key in keys
                if (match := _RESUBMIT.fullmatch(key)) and match.group(1) == packet_sha]
    return max([1, *attempts])


def _check_key(key: object) -> str:
    if (not isinstance(key, str) or _KEY.fullmatch(key) is None
            or any(part in {"", ".", ".."} for part in key.split("/"))):
        raise ValueError("queue key is unsafe")
    return key


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError("object-store URI is invalid")
    bucket, _, prefix = uri[5:].partition("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,62}", bucket) or (prefix and (
            not re.fullmatch(r"[A-Za-z0-9_./-]+", prefix)
            or any(part in {".", ".."} for part in prefix.split("/")))):
        raise ValueError("object-store URI is unsafe")
    return bucket, prefix.strip("/")


def utc_now(clock=time.time) -> str:
    return datetime.fromtimestamp(clock(), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_hashed(source, target: Path) -> tuple[str, int]:
    """Stream ``source`` into a new file at ``target``; return (sha256, bytes)."""
    checksum, total = hashlib.sha256(), 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        while chunk := source.read(1 << 20):
            checksum.update(chunk)
            total += len(chunk)
            output.write(chunk)
    return checksum.hexdigest(), total


# ------------------------------------------------------------------------ queues


class LocalQueue:
    """Directory-backed queue for dry runs and tests (same keys as S3)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = self.root / _check_key(key)
        if path.is_symlink():
            raise ValueError("queue object is linked")
        return path

    def get(self, key: str) -> bytes | None:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None

    def size(self, key: str) -> int | None:
        try:
            return self._path(key).stat().st_size
        except FileNotFoundError:
            return None

    def download(self, key: str, target: Path) -> tuple[str, int] | None:
        try:
            with self._path(key).open("rb") as source:
                return _copy_hashed(source, target)
        except FileNotFoundError:
            return None

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def keys(self, prefix: str) -> list[str]:
        base = self._path(prefix.rstrip("/"))
        if not base.is_dir():
            return []
        return sorted(path.relative_to(self.root).as_posix() for path in base.rglob("*")
                      if path.is_file() and not path.name.startswith("."))

    def describe(self) -> str:
        return str(self.root)


class FsspecQueue:
    """Construction side: CW object storage through rigging's bucket-routed
    filesystem, from the job's ambient ``CW_KEY_*`` pair.

    ``filesystem_for`` builds the client with explicit credentials and never
    writes ``AWS_*``/``FSSPEC_S3`` into ``os.environ`` (which every builder and
    capture subprocess inherits).
    """

    def __init__(self, uri: str, *, fs=None, root: str | None = None):
        split_s3_uri(uri)
        if fs is None:
            from rigging.filesystem.buckets import filesystem_for

            fs, root = filesystem_for(uri.rstrip("/"))
        self.fs, self.root, self.uri = fs, str(root).rstrip("/"), uri.rstrip("/")

    def _path(self, key: str) -> str:
        return f"{self.root}/{_check_key(key)}"

    def get(self, key: str) -> bytes | None:
        try:
            return self.fs.cat_file(self._path(key))
        except FileNotFoundError:
            return None

    def size(self, key: str) -> int | None:
        path = self._path(key)
        self.fs.invalidate_cache(path.rsplit("/", 1)[0])
        try:
            return int(self.fs.info(path)["size"])
        except FileNotFoundError:
            return None

    def put(self, key: str, data: bytes) -> None:
        self.fs.pipe_file(self._path(key), data)

    def keys(self, prefix: str) -> list[str]:
        path = self._path(prefix.rstrip("/"))
        self.fs.invalidate_cache(path)
        try:
            found = self.fs.find(path)
        except FileNotFoundError:
            return []
        return sorted(str(item).removeprefix(self.root + "/") for item in found)

    def describe(self) -> str:
        return self.uri


def s3_client(*, endpoint: str = "https://cwobject.com"):
    """botocore S3 client for CW object storage (virtual-hosted, region auto).

    Credentials come from ``CW_KEY_ID``/``CW_KEY_SECRET`` when set, else
    botocore's default chain (the publisher pod's ``AWS_*`` from ``cw-s3``).
    """
    import botocore.session
    from botocore.config import Config

    key = os.environ.get("CW_KEY_ID")
    secret = os.environ.get("CW_KEY_SECRET")
    explicit = {"aws_access_key_id": key, "aws_secret_access_key": secret} if key and secret else {}
    return botocore.session.get_session().create_client(
        "s3", region_name="auto", endpoint_url=endpoint,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"},
                      connect_timeout=30, read_timeout=120,
                      retries={"max_attempts": 5, "mode": "standard"}),
        **explicit,
    )


def _missing(error: Exception) -> bool:
    response = getattr(error, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


class BotocoreQueue:
    """Publisher side: botocore only (the service image carries no fsspec)."""

    def __init__(self, uri: str, client):
        self.bucket, self.prefix = split_s3_uri(uri)
        self.client, self.uri = client, uri.rstrip("/")

    def _key(self, key: str) -> str:
        key = _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            if _missing(error):
                return None
            raise
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def size(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=self._key(key))["ContentLength"])
        except ClientError as error:
            if _missing(error):
                return None
            raise

    def download(self, key: str, target: Path) -> tuple[str, int] | None:
        """Stream an object to ``target``; (sha256, bytes) or None if absent."""
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as error:
            if _missing(error):
                return None
            raise
        body = response["Body"]
        try:
            return _copy_hashed(body, target)
        finally:
            body.close()

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def keys(self, prefix: str) -> list[str]:
        full = self._key(prefix.rstrip("/")) + "/"
        strip = self.prefix + "/" if self.prefix else ""
        keys = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
            keys.extend(row["Key"].removeprefix(strip) for row in page.get("Contents", ()))
        return sorted(keys)

    def describe(self) -> str:
        return self.uri


def open_queue(location: str, *, client_factory=None):
    """Queue for an ``s3://`` URI (fsspec on the controller, botocore when a
    client factory is supplied) or a local directory (dry run)."""
    if location.startswith("s3://"):
        if client_factory is not None:
            return BotocoreQueue(location, client_factory())
        return FsspecQueue(location)
    location = location.removeprefix("file://")
    if not location or "://" in location:
        raise ValueError("publication queue location is invalid")
    return LocalQueue(Path(location))


def configured_queue_location() -> str:
    return (os.environ.get(QUEUE_ENV) or DEFAULT_QUEUE_URI).strip()


# ----------------------------------------------------------------- object reads


class ObjectStoreReader:
    """Pinned replacement for a packet's staged ``cw_presign``.

    ``download_captured_layer`` needs ``head(bucket, key)`` and
    ``.c.get_object(Bucket=, Key=)``. Reads are confined to the reviewed
    capture-object prefixes, so a forged packet cannot make the publisher copy
    arbitrary objects into the registry.
    """

    def __init__(self, client, allowed_prefixes: list[str]):
        if not allowed_prefixes:
            raise ValueError("publisher capture-object prefixes are required")
        self._client = client
        self._allowed = [split_s3_uri(prefix) for prefix in allowed_prefixes]
        self.c = self

    def _check(self, bucket: str, key: str) -> None:
        if not any(bucket == allowed_bucket and key.startswith(prefix + "/" if prefix else "")
                   for allowed_bucket, prefix in self._allowed):
            raise ValueError("capture object is outside the publisher's allowed prefixes")

    def head(self, bucket: str, key: str) -> dict:
        self._check(bucket, key)
        return self._client.head_object(Bucket=bucket, Key=key)

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self._check(Bucket, Key)
        return self._client.get_object(Bucket=Bucket, Key=Key)


class LocalObjectClient:
    """Dry-run stand-in for the S3 client: ``s3://b/k`` is ``root/b/k``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, bucket: str, key: str) -> Path:
        path = self.root / bucket / _check_key(key)
        if path.is_symlink() or not path.is_file():
            error = FileNotFoundError("capture object is absent")
            error.response = {"Error": {"Code": "404"}}  # type: ignore[attr-defined]
            raise error
        return path

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        return {"ContentLength": self._path(Bucket, Key).stat().st_size}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        return {"Body": self._path(Bucket, Key).open("rb")}


# ------------------------------------------------------------------ result docs


def parse_result(raw: bytes | None, packet_sha: str) -> dict | None:
    """Validated publisher result, ``None`` when absent; malformed raises."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("publication result is not JSON") from error
    if (not isinstance(value, dict) or value.get("schema_version") != RESULT_SCHEMA
            or value.get("packet_sha256") != packet_sha or value.get("state") not in RESULT_STATES
            or type(value.get("attempt")) is not int or value["attempt"] < 1
            or not isinstance(value.get("reason"), str)
            or parse_utc(value.get("completed_utc")) is None):
        raise ValueError("publication result is malformed")
    if value["state"] == "published":
        roles, receipts = value.get("roles"), value.get("receipts")
        if (not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
                or any(role not in ROLES for role in roles)
                or not isinstance(receipts, dict) or set(receipts) != set(roles)
                or not isinstance(value.get("packet_manifest_sha256"), str)
                or _HEX.fullmatch(value["packet_manifest_sha256"]) is None):
            raise ValueError("published result lacks exact receipt identities")
        for role, record in receipts.items():
            if (not isinstance(record, dict) or record.get("key") != receipt_key(packet_sha, role)
                    or not isinstance(record.get("sha256"), str) or _HEX.fullmatch(record["sha256"]) is None
                    or type(record.get("bytes")) is not int or record["bytes"] <= 0):
                raise ValueError("published result receipt identity is invalid")
    elif value["state"] == "rejected" and not isinstance(value.get("rejection_class"), str):
        raise ValueError("rejected result lacks its class")
    return value


def result_document(*, packet_sha: str, attempt: int, state: str, reason: str,
                    clock=time.time, **fields) -> bytes:
    if state not in RESULT_STATES:
        raise ValueError("publication result state is invalid")
    document = {"schema_version": RESULT_SCHEMA, "packet_sha256": _checked_sha(packet_sha),
                "attempt": attempt, "state": state, "reason": reason,
                "completed_utc": utc_now(clock), **fields}
    data = json.dumps(document, sort_keys=True, indent=2).encode() + b"\n"
    parse_result(data, packet_sha)
    return data


def heartbeat_age(queue, clock=time.time) -> float | None:
    """Seconds since the publisher's last heartbeat, ``None`` if unreadable."""
    raw = queue.get(HEARTBEAT_KEY)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    stamp = value.get("epoch") if isinstance(value, dict) else None
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
        return None
    return max(0.0, clock() - float(stamp))


# ------------------------------------------------------- construction controller


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _read_record(path: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != REQUEST_SCHEMA
            or not isinstance(value.get("packet_sha256"), str) or _HEX.fullmatch(value["packet_sha256"]) is None
            or not isinstance(value.get("manifest_sha256"), str)):
        return None
    return value


def _file_sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inputs_fingerprint(pending: dict) -> str:
    """Plan, approval and capture receipts determine every packet member: the
    plan binds the frozen workspace sources and capture tools by hash, and the
    approval binds the retained review packet."""
    parts = {"plan": _file_sha(pending["plan_path"]), "approval": _file_sha(pending["approval_path"]),
             "captures": {role: _file_sha(path) for role, path in sorted(pending["capture_paths"].items())},
             "sessions": sorted(pending["builder_session_ids"])}
    return sha256(json.dumps(parts, sort_keys=True).encode())


class _PacketUnreproducible(OSError):
    """The recorded packet key can no longer be rebuilt; the record was reset."""


def _write_record(exchange_root: Path, record: dict) -> dict:
    _atomic_write(exchange_root / "request.json", json.dumps(record, sort_keys=True, indent=2).encode() + b"\n")
    return record


def _freeze_packet(item_root: Path, pending: dict, exchange_root: Path, clock) -> tuple[dict, bytes | None]:
    """Deterministic handoff for this attempt; returns (record, fresh bytes or None).

    Built (and fully validated) once. Later polls reuse the small retained
    ``request.json`` while the binding inputs are unchanged, so a multi-GiB
    review packet is neither re-read nor recompressed every pass, and the
    archive itself is not duplicated into the run tree: the queue's
    content-addressed ``requests/<sha>/handoff.tar.gz`` is the retained copy.
    """
    from .image_publication_handoff import handoff_packet

    record = _read_record(exchange_root / "request.json")
    fingerprint = _inputs_fingerprint(pending)
    if record is not None and record.get("inputs_sha256") == fingerprint:
        return record, None
    packet = handoff_packet(item_root, pending)
    if (record is not None and record["manifest_sha256"] == packet["manifest_sha256"]
            and record["packet_sha256"] == packet["packet_sha256"]):
        record = {**record, "inputs_sha256": fingerprint, "packet_bytes": len(packet["archive"])}
    else:
        superseded = ([record["packet_sha256"]] + list(record.get("superseded", []))) if record else []
        record = {"schema_version": REQUEST_SCHEMA, "packet_sha256": packet["packet_sha256"],
                  "manifest_sha256": packet["manifest_sha256"], "roles": packet["roles"],
                  "inputs_sha256": fingerprint, "packet_bytes": len(packet["archive"]),
                  "frozen_at": utc_now(clock), "submitted_at": None, "superseded": superseded}
    return _write_record(exchange_root, record), packet["archive"]


class _HandoffInvalid(Exception):
    """The live inputs no longer validate as a reviewed handoff."""


def _rebuild(item_root: Path, pending: dict, exchange_root: Path, record: dict) -> bytes:
    """Recreate the recorded packet bytes (deterministic), or reset the record."""
    from .image_publication_handoff import handoff_packet

    try:
        packet = handoff_packet(item_root, pending)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise _HandoffInvalid(str(error)[:500]) from error
    if packet["packet_sha256"] != record["packet_sha256"]:
        (exchange_root / "request.json").unlink(missing_ok=True)
        raise _PacketUnreproducible("recorded handoff no longer rebuilds byte-identically")
    return packet["archive"]


def _mark_submitted(exchange_root: Path, record: dict, clock) -> dict:
    if record.get("submitted_at"):
        return record
    return _write_record(exchange_root, {**record, "submitted_at": utc_now(clock)})


def _upload_request(queue, packet_sha: str, data: bytes) -> None:
    """Upload the content-addressed packet unless an equal-length object is there."""
    key = request_key(packet_sha)
    if sha256(data) != packet_sha:
        raise _PacketUnreproducible("handoff bytes differ from their recorded key")
    if queue.size(key) != len(data):
        queue.put(key, data)
        readback = queue.get(key)
        if readback is None or sha256(readback) != packet_sha:
            raise OSError("publication request readback differs")


def _wait_backoff(submitted_at: object, clock) -> int:
    started = parse_utc(submitted_at)
    age = 0.0 if started is None else max(0.0, clock() - started)
    return int(min(600, max(60, age / 4)))


def exchange_publication(*, item_root: Path, pending: dict, exchange_root: Path,
                         queue=None, clock=time.time) -> dict | None:
    """Advance one item's publication through the queue.

    Returns ``None`` once every pending role's receipt has been imported (the
    caller continues to cold pull), otherwise the state the controller must
    report. See ``docs/image_registry.md`` for the contract.
    """
    item_root = Path(item_root)
    exchange_root = Path(exchange_root)
    location = None
    if queue is None:
        location = configured_queue_location()
        if location.lower() in DISABLED_VALUES:
            return {**pending, "retryable": False, "reason": "publication_queue_disabled"}
    try:
        record, fresh = _freeze_packet(item_root, pending, exchange_root, clock)
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        return {**pending, "retryable": False, "reason": "handoff_export_failed",
                "error_type": type(error).__name__, "issues": [str(error)[:500]]}
    packet_sha = record["packet_sha256"]
    base = {**pending, "packet_sha": packet_sha, "submitted_at": record.get("submitted_at"),
            "publication_exchange": str(exchange_root)}
    unreachable = {**base, "retryable": True, "reason": "queue_unreachable",
                   "backoff_seconds": QUEUE_UNREACHABLE_BACKOFF_SECONDS}
    if queue is None:
        try:
            queue = open_queue(location)
        except Exception as error:  # noqa: BLE001 - absent creds/rigging or bad URI: hold, retry.
            return {**unreachable, "error_type": type(error).__name__}
    try:
        keys = queue.keys(f"requests/{packet_sha}")
        attempt_requested = requested_attempt(keys, packet_sha)
        age = heartbeat_age(queue, clock)
        result = parse_result(queue.get(result_key(packet_sha)), packet_sha)
        if result is None or result["state"] == "transient_failure":
            if request_key(packet_sha) not in keys:
                _upload_request(queue, packet_sha, fresh if fresh is not None
                                else _rebuild(item_root, pending, exchange_root, record))
            record = _mark_submitted(exchange_root, record, clock)
            base["submitted_at"] = record["submitted_at"]
        waiting = {**base, "retryable": True, "publication_attempt": attempt_requested,
                   "publisher_heartbeat_age_seconds": None if age is None else int(age)}
        stale = age is None or age > HEARTBEAT_STALE_SECONDS
        if result is None:
            return {**waiting, "reason": "awaiting_publisher_heartbeat_stale" if stale else "awaiting_publisher",
                    "backoff_seconds": _wait_backoff(record["submitted_at"], clock)}
        if result["state"] == "rejected":
            reason = f"publisher_rejected: {result['rejection_class']}: {result['reason']}"[:1000]
            repairable = result["rejection_class"] in BUILDER_REPAIRABLE_REJECTIONS
            issues = [(f"The trusted publisher's rootfs review rejected the captured image "
                       f"({result['reason'][:500]}). In a candidate image every regular file under "
                       "/workspace, /opt/task and /fixtures must be declared with its sha256 in "
                       "required_ready_hashes (or removed), and private assets and credentials must "
                       "be absent; fix the source snapshot and provide an updated "
                       "image-capture-request.json for fresh review.")] if repairable else [reason]
            return {**base, "state": "failed_terminal", "failure_stage": "image_publication",
                    "retryable": False, "reason": reason, "rejection_class": result["rejection_class"],
                    "builder_repairable": repairable,
                    "publication_attempt": result["attempt"], "issues": issues}
        if result["state"] == "transient_failure":
            last = result["attempt"]
            if last < attempt_requested:
                return {**waiting, "reason": "publisher_transient_failure_resubmitted",
                        "last_transient_reason": result["reason"][:500], "backoff_seconds": 60}
            if last >= MAX_PUBLICATION_ATTEMPTS:
                return {**base, "retryable": False, "reason": "publisher_transient_retries_exhausted",
                        "publication_attempt": last, "last_transient_reason": result["reason"][:500]}
            ready_at = parse_utc(result["completed_utc"]) + RESUBMIT_BASE_SECONDS * 2 ** (last - 1)
            if clock() < ready_at:
                return {**waiting, "reason": "publisher_transient_failure_backoff",
                        "last_transient_reason": result["reason"][:500],
                        "backoff_seconds": int(max(30, ready_at - clock()))}
            marker = {"schema_version": RESUBMIT_SCHEMA, "packet_sha256": packet_sha,
                      "attempt": last + 1, "requested_utc": utc_now(clock),
                      "after_transient_reason": result["reason"][:500]}
            queue.put(resubmit_key(packet_sha, last + 1),
                      json.dumps(marker, sort_keys=True, indent=2).encode() + b"\n")
            return {**waiting, "reason": "publisher_transient_failure_resubmitted",
                    "publication_attempt": last + 1,
                    "last_transient_reason": result["reason"][:500], "backoff_seconds": 60}
        if result["packet_manifest_sha256"] != record["manifest_sha256"]:
            raise ValueError("published result names a different packet manifest")
        receipts = _fetch_receipts(queue, packet_sha, result, pending["roles"])
        archive = fresh if fresh is not None else queue.get(request_key(packet_sha))
        if archive is None or sha256(archive) != packet_sha:
            archive = _rebuild(item_root, pending, exchange_root, record)
    except _PacketUnreproducible as error:
        return {**unreachable, "reason": "handoff_rebuilt_next_pass", "error_type": type(error).__name__,
                "backoff_seconds": 30}
    except _HandoffInvalid as error:
        return {**base, "retryable": False, "reason": "handoff_export_failed",
                "error_type": type(error.__cause__).__name__, "issues": [str(error)]}
    except ValueError as error:
        # Queue content that is present but inconsistent (a torn or foreign
        # write). The publisher rewrites non-final or malformed results, so
        # this is a retryable hold rather than a verdict on the item.
        return {**base, "retryable": True, "reason": "publisher_result_invalid",
                "error_type": type(error).__name__, "issues": [str(error)[:500]],
                "backoff_seconds": 300}
    except Exception as error:  # noqa: BLE001 - any transport failure is a retryable hold.
        return {**unreachable, "error_type": type(error).__name__}
    return _import_receipts(item_root, pending, archive, exchange_root, result, receipts, base)


def _fetch_receipts(queue, packet_sha: str, result: dict, roles: list[str]) -> dict[str, bytes]:
    missing = [role for role in roles if role not in result["receipts"]]
    if missing:
        raise ValueError("published result lacks a pending role")
    receipts = {}
    for role in roles:
        record = result["receipts"][role]
        data = queue.get(record["key"])
        if data is None or len(data) != record["bytes"] or sha256(data) != record["sha256"]:
            raise ValueError("returned receipt differs from its result record")
        receipts[role] = data
    return receipts


def _import_receipts(item_root: Path, pending: dict, archive: bytes, exchange_root: Path,
                     result: dict, receipts: dict[str, bytes], base: dict) -> dict | None:
    from .image_publication_handoff import import_publication_for_item

    returned = exchange_root / "returns" / f"attempt-{result['attempt']}"
    imported = []
    try:
        _atomic_write(returned / "result.json", json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")
        with tempfile.TemporaryDirectory(prefix="publication-import-") as temporary:
            packet = Path(temporary) / "handoff.tar.gz"
            packet.write_bytes(archive)
            for role, data in receipts.items():
                staged = returned / f"publication-{role}.json"
                _atomic_write(staged, data)
                imported.append(import_publication_for_item(item_root, pending, packet, role, staged))
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        return {**base, "retryable": False, "reason": "publication_receipt_import_failed",
                "error_type": type(error).__name__, "issues": [str(error)[:500]],
                "imported_roles": [row["role"] for row in imported]}
    return None
