# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve object-store profiles through XProf."""

import gzip
import hashlib
import html
import logging
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode

from rigging.filesystem import StoragePath

from infra.xprof.config import HEALTH_PATH

logger = logging.getLogger(__name__)

_SOURCE_MARKER = ".xprof-source"
_REWRITE_SUFFIXES = (".html", ".js")
_XPROF_RUN_PATH = ("plugins", "profile")
_XPROF_TTL_SEGMENT = re.compile(r"ttl=[1-9]\d*d")

StartResponse = Callable[[str, list[tuple[str, str]]], Callable[[bytes], object] | None]
WsgiApplication = Callable[[dict, StartResponse], Iterable[bytes]]


class ProfileSourceError(ValueError):
    """Raised when a profile URI is not a supported XProf TTL root."""


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

    def stage(self, uri: str) -> Path:
        """Return the cached XProf run path."""
        source_uri = self.validate(uri)
        cache_key = hashlib.sha256(source_uri.encode()).hexdigest()[:24]
        target = self._cache_dir / cache_key
        with self._lock_for(cache_key):
            if self._is_ready(target, source_uri):
                return self._xprof_run_path(target)

            temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}-", dir=self._cache_dir))
            downloaded = temporary / "profile"
            try:
                StoragePath(source_uri).download_to(str(downloaded), recursive=True)
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

    def stage(self, uri: str) -> Path: ...


class ProfileStageManager:
    """Stage profiles outside the Iris request timeout."""

    def __init__(self, stager: ProfileStager, max_workers: int = 4, max_retained: int = 256):
        self._stager = stager
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xprof-stage")
        self._futures: OrderedDict[str, Future[Path]] = OrderedDict()
        self._max_retained = max_retained
        self._lock = threading.Lock()

    def validate(self, uri: str) -> str:
        return self._stager.validate(uri)

    def future(self, uri: str) -> Future[Path]:
        """Return or start the staging task for ``uri``."""
        with self._lock:
            future = self._futures.get(uri)
            if future is None:
                future = self._executor.submit(self._stager.stage, uri)
                self._futures[uri] = future
            self._futures.move_to_end(uri)
            self._discard_old_results()
            return future

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def discard(self, uri: str, future: Future[Path]) -> None:
        with self._lock:
            if self._futures.get(uri) is future:
                self._futures.pop(uri)

    def _discard_old_results(self) -> None:
        while len(self._futures) > self._max_retained:
            oldest_uri, oldest = next(iter(self._futures.items()))
            if not oldest.done():
                return
            self._futures.pop(oldest_uri)


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
        return self._serve_xprof(path, environ, start_response)

    def shutdown(self) -> None:
        self._profiles.shutdown()

    def _open(self, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        if environ.get("REQUEST_METHOD", "GET") != "GET":
            return _response(start_response, "405 Method Not Allowed", b"GET required\n", "text/plain")

        uri = parse_qs(environ.get("QUERY_STRING", "")).get("uri", [""])[0]
        if not uri:
            return _response(start_response, "400 Bad Request", b"missing uri query parameter\n", "text/plain")
        try:
            normalized_uri = self._profiles.validate(uri)
        except ProfileSourceError as exc:
            return _response(start_response, "403 Forbidden", f"{exc}\n".encode(), "text/plain")

        future = self._profiles.future(normalized_uri)
        if not future.done():
            return _response(start_response, "202 Accepted", _loading_page(normalized_uri), "text/html; charset=utf-8")
        try:
            local_path = future.result()
        except Exception as exc:
            self._profiles.discard(normalized_uri, future)
            logger.exception("Failed to stage XProf profile %s", normalized_uri)
            return _response(
                start_response, "502 Bad Gateway", f"profile staging failed: {exc}\n".encode(), "text/plain"
            )

        location = f"./?{urlencode({'run_path': str(local_path)})}"
        start_response("303 See Other", [("Location", location), ("Content-Length", "0")])
        return [b""]

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


def _loading_page(uri: str) -> bytes:
    safe_uri = html.escape(uri)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<title>Loading XProf profile</title></head>
<body><p>Staging <code>{safe_uri}</code> for XProf…</p></body></html>
""".encode()
