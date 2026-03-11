# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Periodic py-spy capture with ring buffer and cloud upload on exit.

Runs as a background thread alongside the task monitor loop. Each interval,
captures a short py-spy profile via ContainerHandle.profile() and writes it
to a local ring buffer. On exit, uploads the surviving snapshots to cloud
storage at {storage_prefix}/task-profiles/{safe_token}/attempt_{id}/cpu/.
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

from iris.cluster.runtime.types import ContainerHandle
from iris.marin_fs import url_to_fs
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 60.0
_DEFAULT_DURATION = 10
_DEFAULT_MAX_SNAPSHOTS = 10


class ProfileCapture:
    def __init__(
        self,
        container_handle: ContainerHandle,
        safe_token: str,
        attempt_id: int,
        workdir: Path,
        storage_prefix: str,
        *,
        max_snapshots: int = _DEFAULT_MAX_SNAPSHOTS,
        interval: float = _DEFAULT_INTERVAL,
        capture_duration: int = _DEFAULT_DURATION,
    ):
        self._handle = container_handle
        self._safe_token = safe_token
        self._attempt_id = attempt_id
        self._profile_dir = workdir / ".iris" / "profiles" / "cpu"
        self._storage_prefix = storage_prefix
        self._max_snapshots = max_snapshots
        self._interval = interval
        self._capture_duration = capture_duration
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0

    def __enter__(self) -> "ProfileCapture":
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name=f"profile-{self._safe_token}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        assert self._thread is not None
        # Docker/K8s profile() blocks for capture_duration + subprocess timeout (~5-30s).
        self._thread.join(timeout=self._capture_duration + 35)
        if self._thread.is_alive():
            logger.warning(
                "Profile thread for %s did not exit; skipping upload",
                self._safe_token,
            )
            return
        self._upload()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._capture()

    def _capture(self) -> None:
        try:
            profile_type = cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.RAW))
            t0 = time.monotonic()
            data = self._handle.profile(self._capture_duration, profile_type)
            elapsed = time.monotonic() - t0
            if not data:
                logger.info("Empty profile for %s after %.1fs (no frames sampled)", self._safe_token, elapsed)
                return
            logger.info("Captured %d bytes profile for %s in %.1fs", len(data), self._safe_token, elapsed)
            self._write(data)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            logger.warning("Profile capture failed for %s", self._safe_token, exc_info=True)

    def _write(self, data: bytes) -> None:
        epoch = int(time.time())
        path = self._profile_dir / f"snapshot-{self._seq:05d}-{epoch}.txt"
        path.write_bytes(data)
        self._seq += 1
        snapshots = sorted(self._profile_dir.glob("snapshot-*.txt"))
        excess = len(snapshots) - self._max_snapshots
        if excess > 0:
            for old in snapshots[:excess]:
                old.unlink(missing_ok=True)

    def _upload(self) -> None:
        if not self._storage_prefix:
            return
        snapshots = [s for s in sorted(self._profile_dir.glob("snapshot-*.txt")) if s.stat().st_size > 0]
        if not snapshots:
            return
        dest = f"{self._storage_prefix}/task-profiles/{self._safe_token}/attempt_{self._attempt_id}/cpu"
        try:
            fs, base = url_to_fs(dest)
            fs.makedirs(base, exist_ok=True)
            for snap in snapshots:
                remote_path = f"{base}/{snap.name}"
                local_size = snap.stat().st_size
                with fs.open(remote_path, "wb") as f:
                    f.write(snap.read_bytes())
                remote_size = fs.info(remote_path).get("size", -1)
                if remote_size != local_size:
                    logger.warning(
                        "Size mismatch after upload: local=%d remote=%d path=%s",
                        local_size,
                        remote_size,
                        remote_path,
                    )
            logger.info(
                "Uploaded %d profile snapshots for %s to %s",
                len(snapshots),
                self._safe_token,
                dest,
            )
        except OSError:
            logger.warning("Failed to upload profiles for %s", self._safe_token, exc_info=True)
