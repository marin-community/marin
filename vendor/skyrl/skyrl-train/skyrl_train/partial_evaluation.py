"""Best-effort snapshots of completed evaluation rollouts."""

from collections.abc import Callable, Mapping, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)

BACKUP_ROLLOUT_INTERVAL = 25
BACKUP_TIME_INTERVAL = 300
BACKUP_TIMEOUT = 30


def _copy_snapshot(
    rows_path: str,
    snapshot_path: str,
    prefix_bytes: int,
    metadata: Mapping[str, Any],
) -> None:
    with open(rows_path, "rb") as source, open(snapshot_path, "wb") as destination:
        destination.write(json.dumps(metadata, ensure_ascii=False).encode() + b"\n")
        remaining = prefix_bytes
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise OSError("evaluation rows ended before the snapshot boundary")
            destination.write(chunk)
            remaining -= len(chunk)


def _publish_snapshot(
    rows_path: str,
    snapshot_path: str,
    target: str,
    prefix_bytes: int,
    metadata: Mapping[str, Any],
    upload: Callable[[str, str], None],
) -> None:
    _copy_snapshot(rows_path, snapshot_path, prefix_bytes, metadata)
    if target.startswith(("s3://", "gs://", "gcs://")):
        upload(snapshot_path, target)
    else:
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}")
        try:
            shutil.copyfile(snapshot_path, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class PartialEvaluationRecorder:
    """Append completed rows locally and back up a scored prefix in the background."""

    def __init__(
        self,
        target_prefix: str,
        expected_rollouts: int,
        source: str | None,
        upload: Callable[[str, str], None],
    ) -> None:
        self._expected_rollouts = expected_rollouts
        self._source = source
        self._upload = upload
        self._attempt_id = uuid4().hex
        self._targets = tuple(f"{target_prefix}-{self._attempt_id}-{slot}.jsonl" for slot in range(2))
        self._directory = tempfile.TemporaryDirectory(prefix="skyrl-eval-")
        self._rows_path = str(Path(self._directory.name) / "rows.jsonl")
        self._rows = open(self._rows_path, "wb")
        self._completed_rollouts = 0
        self._trajectory_rows = 0
        self._last_attempt_count = 0
        self._last_attempt_at = 0.0
        self._last_uploaded_count = 0
        self._slot = 0
        self._backup: threading.Thread | None = None
        self._backup_ok = False
        self._backup_count = 0
        self._disabled = False

    def append(self, entries: Sequence[Mapping[str, Any]], repetition: int) -> None:
        """Record a finished rollout without waiting for remote storage."""
        if self._disabled:
            return
        try:
            rows = [
                json.dumps({**entry, "repetition_id": repetition}, ensure_ascii=False).encode() + b"\n"
                for entry in entries
            ]
            completed = self._completed_rollouts + 1
            progress = json.dumps({"record_type": "progress", "completed_rollouts": completed}).encode() + b"\n"
            self._rows.write(b"".join(rows) + progress)
            self._completed_rollouts = completed
            self._trajectory_rows += len(entries)
            self._maybe_start_backup()
        except Exception:
            self._disabled = True
            logger.exception("Partial evaluation recording failed; evaluation continues")

    def _maybe_start_backup(self) -> None:
        self._finish_backup()
        if self._backup is not None:
            return
        now = time.monotonic()
        if (
            self._last_attempt_count
            and self._completed_rollouts - self._last_attempt_count < BACKUP_ROLLOUT_INTERVAL
            and now - self._last_attempt_at < BACKUP_TIME_INTERVAL
        ):
            return
        self._start_backup(now)

    def _start_backup(self, now: float) -> None:
        self._rows.flush()
        prefix_bytes = self._rows.tell()
        count = self._completed_rollouts
        metadata = {
            "record_type": "partial_evaluation",
            "status": "partial",
            "attempt_id": self._attempt_id,
            "source": self._source,
            "expected_rollouts": self._expected_rollouts,
            "completed_rollouts": count,
            "trajectory_rows": self._trajectory_rows,
            "snapshot_at": time.time(),
        }
        snapshot_path = str(Path(self._directory.name) / f"snapshot-{self._slot}.jsonl")
        target = self._targets[self._slot]
        self._last_attempt_count = count
        self._last_attempt_at = now
        self._backup_count = count
        self._backup_ok = False
        self._backup = threading.Thread(
            target=self._supervise_backup,
            args=(snapshot_path, target, prefix_bytes, metadata),
            daemon=True,
            name="skyrl-eval-backup",
        )
        self._backup.start()

    def _supervise_backup(
        self, snapshot_path: str, target: str, prefix_bytes: int, metadata: Mapping[str, Any]
    ) -> None:
        process = multiprocessing.get_context("spawn").Process(
            target=_publish_snapshot,
            args=(self._rows_path, snapshot_path, target, prefix_bytes, metadata, self._upload),
            daemon=True,
        )
        try:
            process.start()
            process.join(BACKUP_TIMEOUT)
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join(1)
                logger.warning("Partial evaluation backup timed out after %s seconds: %s", BACKUP_TIMEOUT, target)
            else:
                self._backup_ok = process.exitcode == 0
                if not self._backup_ok:
                    logger.warning("Partial evaluation backup failed with exit code %s: %s", process.exitcode, target)
        except Exception:
            logger.exception("Partial evaluation backup failed: %s", target)

    def _finish_backup(self) -> None:
        if self._backup is None or self._backup.is_alive():
            return
        self._backup.join()
        if self._backup_ok:
            self._last_uploaded_count = self._backup_count
            self._slot = 1 - self._slot
        self._backup = None

    def close(self) -> None:
        """Make one final bounded backup without making storage failure fatal."""
        self._rows.flush()
        if self._backup is not None:
            self._backup.join(BACKUP_TIMEOUT + 3)
            self._finish_backup()
        if not self._disabled and self._completed_rollouts > self._last_uploaded_count and self._backup is None:
            self._start_backup(time.monotonic())
            assert self._backup is not None
            self._backup.join(BACKUP_TIMEOUT + 3)
            self._finish_backup()
        self._rows.close()
        if self._backup is None:
            self._directory.cleanup()
        else:
            logger.warning("Partial evaluation backup remains active; retaining local staging files")
