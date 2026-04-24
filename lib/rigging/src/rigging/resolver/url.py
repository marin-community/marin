# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""URL parsing for rigging service references.

`ServiceURL` is a small dataclass over `urllib.parse` that exposes the three
pieces our resolver dispatches on: scheme, host (URL authority, with no
userinfo or port), and a flat query-param dict.

Multi-value query parameters are not part of the rigging URL contract — if a
key appears more than once, the first value wins. Validation is intentionally
permissive: only the empty-scheme and empty-host cases are rejected here; the
resolver decides what to do with unknown schemes.
"""

from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class ServiceURL:
    """Parsed `scheme://host?query` reference for the rigging resolver."""

    scheme: str
    host: str
    query: dict[str, str]

    @classmethod
    def parse(cls, ref: str) -> "ServiceURL":
        """Parse a rigging-style URL reference.

        Empty scheme or empty host raise ``ValueError``. Anything else is
        accepted; the caller is responsible for rejecting unknown schemes.
        """
        parts = urlsplit(ref)
        if not parts.scheme:
            raise ValueError(f"missing scheme in URL: {ref!r}")
        # `urlsplit` puts the authority in `netloc` and ignores port-less
        # authorities for some custom schemes (e.g. `iris://marin?...` keeps
        # `netloc='marin'`). Use `hostname` so userinfo / port are stripped
        # if anyone sneaks them in; fall back to `netloc` for raw schemes.
        host = parts.hostname or parts.netloc
        if not host:
            raise ValueError(f"missing host in URL: {ref!r}")
        # `parse_qs` returns `dict[str, list[str]]`; flatten by taking the
        # first occurrence of each key.
        query: dict[str, str] = {k: v[0] for k, v in parse_qs(parts.query).items() if v}
        return cls(scheme=parts.scheme, host=host, query=query)
