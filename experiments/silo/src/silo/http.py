# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A small stdlib-only HTTP client.

stdlib on purpose: the worker-side shim is a single file that the pipeline loads
with ``exec()`` (``daytona_environment.py:93-100``), so it cannot bring a
dependency with it. The broker reuses the same client to call hosts.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from silo.errors import SiloError
from silo.wire import error_from_body


class TransportError(SiloError):
    """The peer could not be reached at all -- as opposed to answering with an error.

    Kept distinct because the two call for different handling: a transport error
    is a reason to re-resolve an address, an HTTP error is an answer.

    ``refused`` is True only when the connection was refused outright, i.e. the
    request provably never reached a server. That is the one case in which a
    non-idempotent request is safe to retry elsewhere.
    """

    status_code = 503

    def __init__(self, message: str, *, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


class HttpClient:
    def __init__(self, base_url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._timeout = timeout

    def _url(self, path: str, query: Mapping[str, Any] | None) -> str:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        return url

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        data: bytes | None = None,
        query: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        raw: bool = False,
    ) -> Any:
        all_headers = {**self._headers, **(headers or {})}
        body = data
        if json_body is not None:
            body = json.dumps(json_body).encode()
            all_headers["Content-Type"] = "application/json"
        elif data is not None:
            all_headers.setdefault("Content-Type", "application/octet-stream")
        request = urllib.request.Request(self._url(path, query), data=body, method=method, headers=all_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self._timeout) as response:
                payload = response.read()
                if raw:
                    return payload
                if not payload:
                    return None
                return json.loads(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError):
                parsed = {"message": payload.decode(errors="replace")[:1000] or error.reason}
            if not isinstance(parsed, dict):
                parsed = {"message": str(parsed)}
            raise error_from_body(parsed, error.code) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            refused = isinstance(reason, ConnectionRefusedError) or isinstance(error, ConnectionRefusedError)
            raise TransportError(f"cannot reach {self.base_url}{path}: {error}", refused=refused) from error

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, body: Any = None, **kwargs: Any) -> Any:
        return self.request("POST", path, json_body=body if body is not None else {}, **kwargs)

    def put(self, path: str, data: bytes, **kwargs: Any) -> Any:
        return self.request("PUT", path, data=data, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)
