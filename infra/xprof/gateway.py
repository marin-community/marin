# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve object-store profiles through XProf."""

import gzip
import hashlib
import html
import json
import logging
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode

from fsspec.callbacks import Callback
from rigging.filesystem.storage_path import StoragePath

from infra.xprof.config import HEALTH_PATH

logger = logging.getLogger(__name__)

_SOURCE_MARKER = ".xprof-source"
_REWRITE_SUFFIXES = (".html", ".js")
_XPROF_RUN_PATH = ("plugins", "profile")
_XPROF_TTL_SEGMENT = re.compile(r"ttl=[1-9]\d*d")
_TOOL_QUERY_PARAMETER = "tool"

StartResponse = Callable[[str, list[tuple[str, str]]], Callable[[bytes], object] | None]
WsgiApplication = Callable[[dict, StartResponse], Iterable[bytes]]


class ProfileSourceError(ValueError):
    """Raised when a profile URI is not a supported XProf TTL root."""


@dataclass(frozen=True)
class ProfileStageProgress:
    """Current progress for one profile transfer."""

    downloaded_bytes: int
    files_completed: int
    total_files: int | None
    elapsed_seconds: float
    cache_hit: bool

    @property
    def throughput_bytes_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0
        return self.downloaded_bytes / self.elapsed_seconds


class ProfileDownloadCallback(Callback):
    """Collect byte and file progress from an fsspec recursive transfer."""

    def __init__(self, now: Callable[[], float] = time.monotonic):
        super().__init__()
        self._now = now
        self._started_at = now()
        self._downloaded_bytes = 0
        self._files_completed = 0
        self._total_files: int | None = None
        self._cache_hit = False
        self._lock = threading.Lock()

    def set_size(self, size: int) -> None:
        with self._lock:
            self._total_files = size

    def branched(self, path_1: str, path_2: str, **kwargs) -> Callback:
        del path_1, path_2, kwargs
        return _FileDownloadCallback(self)

    def add_bytes(self, count: int) -> None:
        with self._lock:
            self._downloaded_bytes += count

    def complete_file(self) -> None:
        with self._lock:
            self._files_completed += 1

    def mark_cache_hit(self) -> None:
        with self._lock:
            self._cache_hit = True

    def snapshot(self) -> ProfileStageProgress:
        with self._lock:
            return ProfileStageProgress(
                downloaded_bytes=self._downloaded_bytes,
                files_completed=self._files_completed,
                total_files=self._total_files,
                elapsed_seconds=self._now() - self._started_at,
                cache_hit=self._cache_hit,
            )


class _FileDownloadCallback(Callback):
    def __init__(self, parent: ProfileDownloadCallback):
        super().__init__()
        self._parent = parent

    def relative_update(self, inc: int = 1) -> None:
        self._parent.add_bytes(inc)

    def close(self) -> None:
        self._parent.complete_file()


class ProfileCache:
    """Stage profiles atomically in a local cache."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def validate(self, uri: str) -> str:
        source = StoragePath(uri)
        if source.scheme not in ("gs", "s3") or not source.bucket:
            raise ProfileSourceError("profile URI must use gs:// or s3://")
        # Leave room after the TTL segment for xprof and at least one run segment.
        if not any(
            _XPROF_TTL_SEGMENT.fullmatch(segment) and source.segments[index + 1] == "xprof"
            for index, segment in enumerate(source.segments[:-2])
        ):
            raise ProfileSourceError("profile URI must contain ttl=Nd/xprof/<run>")
        return str(source)

    def stage(self, uri: str, progress: ProfileDownloadCallback | None = None) -> Path:
        """Return the cached XProf run path."""
        progress = progress or ProfileDownloadCallback()
        source_uri = self.validate(uri)
        cache_key = hashlib.sha256(source_uri.encode()).hexdigest()[:24]
        target = self._cache_dir / cache_key
        with self._lock_for(cache_key):
            if self._is_ready(target, source_uri):
                progress.mark_cache_hit()
                return self._xprof_run_path(target)

            temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}-", dir=self._cache_dir))
            downloaded = temporary / "profile"
            try:
                StoragePath(source_uri).download_to(str(downloaded), recursive=True, callback=progress)
                run_path = self._xprof_run_path(downloaded)
                if not any(run_path.glob("*/*.xplane.pb")) and not any(run_path.glob("*/*.xplane.riegeli")):
                    raise FileNotFoundError(f"no XPlane files found under {source_uri}")
                (downloaded / _SOURCE_MARKER).write_text(source_uri)
                downloaded.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return self._xprof_run_path(target)

    def _lock_for(self, cache_key: str) -> threading.Lock:
        with self._locks_lock:
            return self._locks.setdefault(cache_key, threading.Lock())

    @staticmethod
    def _is_ready(target: Path, source_uri: str) -> bool:
        marker = target / _SOURCE_MARKER
        return marker.is_file() and marker.read_text() == source_uri

    @staticmethod
    def _xprof_run_path(cache_root: Path) -> Path:
        return cache_root.joinpath(*_XPROF_RUN_PATH)


class ProfileStager(Protocol):
    """Validate and stage profile trees."""

    def validate(self, uri: str) -> str: ...

    def stage(self, uri: str, progress: ProfileDownloadCallback) -> Path: ...


class ProfileStageManager:
    """Stage profiles outside the Iris request timeout."""

    def __init__(self, stager: ProfileStager, max_workers: int = 4, max_retained: int = 256):
        self._stager = stager
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xprof-stage")
        self._futures: OrderedDict[str, Future[Path]] = OrderedDict()
        self._progress: dict[str, ProfileDownloadCallback] = {}
        self._max_retained = max_retained
        self._lock = threading.Lock()

    def validate(self, uri: str) -> str:
        return self._stager.validate(uri)

    def future(self, uri: str) -> Future[Path]:
        """Return or start the staging task for ``uri``."""
        with self._lock:
            future = self._futures.get(uri)
            if future is None:
                progress = ProfileDownloadCallback()
                future = self._executor.submit(self._stage, uri, progress)
                self._futures[uri] = future
                self._progress[uri] = progress
            self._futures.move_to_end(uri)
            self._discard_old_results()
            return future

    def _stage(self, uri: str, progress: ProfileDownloadCallback) -> Path:
        try:
            path = self._stager.stage(uri, progress)
        except Exception:
            snapshot = progress.snapshot()
            logger.exception(
                "Profile download failed uri=%s bytes=%d files=%d elapsed=%.1fs throughput=%.1fMiB/s",
                uri,
                snapshot.downloaded_bytes,
                snapshot.files_completed,
                snapshot.elapsed_seconds,
                snapshot.throughput_bytes_per_second / (1024 * 1024),
            )
            raise
        snapshot = progress.snapshot()
        if snapshot.cache_hit:
            logger.info("Profile cache hit uri=%s elapsed=%.1fs", uri, snapshot.elapsed_seconds)
            return path
        logger.info(
            "Profile download complete uri=%s bytes=%d files=%d elapsed=%.1fs throughput=%.1fMiB/s",
            uri,
            snapshot.downloaded_bytes,
            snapshot.files_completed,
            snapshot.elapsed_seconds,
            snapshot.throughput_bytes_per_second / (1024 * 1024),
        )
        return path

    def progress(self, uri: str) -> ProfileStageProgress | None:
        with self._lock:
            progress = self._progress.get(uri)
            return progress.snapshot() if progress is not None else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def discard(self, uri: str, future: Future[Path]) -> None:
        with self._lock:
            if self._futures.get(uri) is future:
                self._futures.pop(uri)
                self._progress.pop(uri, None)

    def _discard_old_results(self) -> None:
        while len(self._futures) > self._max_retained:
            oldest_uri, oldest = next(iter(self._futures.items()))
            if not oldest.done():
                return
            self._futures.pop(oldest_uri)
            self._progress.pop(oldest_uri, None)


class XprofGateway:
    """Serve gateway routes and delegate XProf routes."""

    def __init__(self, xprof_app: WsgiApplication, profiles: ProfileStageManager, public_path: str):
        self._xprof_app = xprof_app
        self._profiles = profiles
        self._public_path = public_path.rstrip("/")

    def __call__(self, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        path = environ.get("PATH_INFO", "/")
        if path == HEALTH_PATH:
            return _response(start_response, "200 OK", b"ok\n", "text/plain; charset=utf-8")
        if path == "/open":
            return self._open(environ, start_response)
        if path == "/progress":
            return self._progress(environ, start_response)
        return self._serve_xprof(path, environ, start_response)

    def shutdown(self) -> None:
        self._profiles.shutdown()

    def _open(self, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        if environ.get("REQUEST_METHOD", "GET") != "GET":
            return _response(start_response, "405 Method Not Allowed", b"GET required\n", "text/plain")

        query = parse_qs(environ.get("QUERY_STRING", ""))
        uri = query.get("uri", [""])[0]
        tool = query.get(_TOOL_QUERY_PARAMETER, [""])[0]
        if not uri:
            return _response(start_response, "400 Bad Request", b"missing uri query parameter\n", "text/plain")
        try:
            normalized_uri = self._profiles.validate(uri)
        except ProfileSourceError as exc:
            return _response(start_response, "403 Forbidden", f"{exc}\n".encode(), "text/plain")

        future = self._profiles.future(normalized_uri)
        if not future.done():
            return _response(
                start_response, "202 Accepted", _loading_page(normalized_uri, tool), "text/html; charset=utf-8"
            )
        try:
            local_path = future.result()
        except Exception as exc:
            self._profiles.discard(normalized_uri, future)
            logger.exception("Failed to stage XProf profile %s", normalized_uri)
            return _response(
                start_response, "502 Bad Gateway", f"profile staging failed: {exc}\n".encode(), "text/plain"
            )

        location = _xprof_location(local_path, tool)
        start_response("303 See Other", [("Location", location), ("Content-Length", "0")])
        return [b""]

    def _progress(self, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        query = parse_qs(environ.get("QUERY_STRING", ""))
        uri = query.get("uri", [""])[0]
        tool = query.get(_TOOL_QUERY_PARAMETER, [""])[0]
        try:
            normalized_uri = self._profiles.validate(uri)
        except ProfileSourceError as exc:
            return _json_response(start_response, "403 Forbidden", {"error": str(exc)})

        future = self._profiles.future(normalized_uri)
        progress = self._profiles.progress(normalized_uri)
        if future.done():
            if future.exception() is not None:
                self._profiles.discard(normalized_uri, future)
                return _json_response(start_response, "200 OK", {"state": "failed"})
            local_path = future.result()
            return _json_response(
                start_response,
                "200 OK",
                {"state": "ready", "location": _xprof_location(local_path, tool)},
            )
        if progress is None:
            return _json_response(start_response, "200 OK", {"state": "starting"})
        return _json_response(
            start_response,
            "200 OK",
            {
                "state": "downloading",
                "downloaded_bytes": progress.downloaded_bytes,
                "files_completed": progress.files_completed,
                "total_files": progress.total_files,
                "elapsed_seconds": round(progress.elapsed_seconds, 1),
                "throughput_bytes_per_second": round(progress.throughput_bytes_per_second),
            },
        )

    def _serve_xprof(self, path: str, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        if path != "/" and not path.endswith(_REWRITE_SUFFIXES):
            return self._xprof_app(environ, start_response)

        captured: list[tuple[str, list[tuple[str, str]]]] = []

        def capture_response(status: str, headers: list[tuple[str, str]], _exc_info=None):
            captured.append((status, headers))
            return None

        body_iter = self._xprof_app(environ, capture_response)
        try:
            body = b"".join(body_iter)
        finally:
            close = getattr(body_iter, "close", None)
            if close is not None:
                close()
        status, headers = captured[0]
        content_encoding = next((value for name, value in headers if name.lower() == "content-encoding"), None)
        if content_encoding == "gzip":
            body = gzip.decompress(body)
        body = body.replace(b"/data/plugin/", f"{self._public_path}/data/plugin/".encode())
        if content_encoding == "gzip":
            body = gzip.compress(body)
        rewritten_headers = [
            (name, value) for name, value in headers if name.lower() not in ("content-length", "etag", "content-md5")
        ]
        rewritten_headers.append(("Content-Length", str(len(body))))
        start_response(status, rewritten_headers)
        return [body]


def _response(
    start_response: StartResponse,
    status: str,
    body: bytes,
    content_type: str,
) -> list[bytes]:
    start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(body)))])
    return [body]


def _json_response(start_response: StartResponse, status: str, value: dict) -> list[bytes]:
    return _response(start_response, status, json.dumps(value).encode(), "application/json")


def _xprof_location(run_path: Path, tool: str) -> str:
    params = {"run_path": str(run_path)}
    if tool:
        params[_TOOL_QUERY_PARAMETER] = tool
    return f"./?{urlencode(params)}"


def _loading_page(uri: str, tool: str) -> bytes:
    safe_uri = html.escape(uri)
    params = {"uri": uri}
    if tool:
        params[_TOOL_QUERY_PARAMETER] = tool
    progress_url = f"./progress?{urlencode(params)}"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Loading XProf profile</title>
<style>
body {{ font: 16px system-ui; margin: 4rem auto; max-width: 48rem; padding: 0 1rem; }}
.bar {{ background: #ddd; border-radius: 4px; height: 8px; margin: 2rem 0; overflow: hidden; }}
.bar::after {{
  animation: move 1.2s infinite linear; background: #1976d2; content: "";
  display: block; height: 100%; width: 35%;
}}
@keyframes move {{ from {{ margin-left: -35%; }} to {{ margin-left: 100%; }} }}
code {{ overflow-wrap: anywhere; }}
</style></head>
<body><h1>Loading XProf profile</h1><p><code>{safe_uri}</code></p>
<div class="bar"></div><p id="status">Starting download…</p>
<script>
const status = document.getElementById('status');
const units = value => {{
  if (value < 1024 * 1024) return `${{(value / 1024).toFixed(1)}} KiB`;
  return `${{(value / 1024 / 1024).toFixed(1)}} MiB`;
}};
async function update() {{
  try {{
    const response = await fetch({json.dumps(progress_url)}, {{cache: 'no-store'}});
    const progress = await response.json();
    if (progress.state === 'ready') {{ window.location.replace(progress.location); return; }}
    if (progress.state === 'failed') {{
      status.textContent = 'Download failed. Reload this page to try again.';
      return;
    }}
    if (progress.state === 'downloading') {{
      const files = progress.total_files === null
        ? `${{progress.files_completed}} files`
        : `${{progress.files_completed}} of ${{progress.total_files}} files`;
      status.textContent = `${{units(progress.downloaded_bytes)}} downloaded · ${{files}} · ` +
        `${{units(progress.throughput_bytes_per_second)}}/s · ${{progress.elapsed_seconds.toFixed(1)}} seconds`;
    }}
  }} catch (error) {{ status.textContent = 'Waiting for the XProf service…'; }}
  window.setTimeout(update, 1000);
}}
update();
</script></body></html>
""".encode()
