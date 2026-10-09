# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward hosted viewer requests to an xprof-rs process on loopback."""

import http.client
import logging
from collections.abc import Callable, Iterable
from urllib.parse import quote

from infra.xprof.config import BACKEND_HOST, PROXY_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

StartResponse = Callable[[str, list[tuple[str, str]]], Callable[[bytes], object] | None]


class RustProxy:
    """Proxy browser requests to the xprof-rs HTTP server."""

    def __init__(self, port: int):
        self._port = port

    def __call__(self, environ: dict, start_response: StartResponse) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        if method not in ("GET", "HEAD") and not (
            method == "POST" and path.rstrip("/") in ("/generate_cache", "/data/plugin/profile/generate_cache")
        ):
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"Method Not Allowed\n"]

        target = quote(path, safe="/%")
        if query := environ.get("QUERY_STRING"):
            target += f"?{query}"
        connection = http.client.HTTPConnection(BACKEND_HOST, self._port, timeout=PROXY_TIMEOUT_SECONDS)
        headers = {"Accept-Encoding": environ.get("HTTP_ACCEPT_ENCODING", "identity")}
        try:
            connection.request(method, target, headers=headers)
            reply = connection.getresponse()
        except OSError:
            connection.close()
            logger.exception("xprof-rs backend request failed")
            start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
            return [b"xprof-rs backend unavailable\n"]

        omitted = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
        response_headers = [(name, value) for name, value in reply.getheaders() if name.lower() not in omitted]
        start_response(f"{reply.status} {reply.reason}", response_headers)

        def chunks() -> Iterable[bytes]:
            try:
                while chunk := reply.read(256 * 1024):
                    yield chunk
            finally:
                connection.close()

        return chunks()
