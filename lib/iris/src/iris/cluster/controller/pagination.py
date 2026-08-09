# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import base64
import hashlib
import json
from collections.abc import Mapping

from iris.cluster.types import (
    LOCAL_CLUSTER,
)
from iris.resources.errors import (
    InvalidPageToken,
)
from iris.resources.identity import (
    ResourceKey,
    ResourceKind,
)


def _query_fingerprint(kind: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps({"kind": kind, **payload}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_page_token(fingerprint: str, position: Mapping[str, object]) -> str:
    payload = json.dumps(
        {"query": fingerprint, "position": position},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_page_token(token: str | None, fingerprint: str) -> dict[str, object] | None:
    if token is None:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        if payload["query"] != fingerprint or not isinstance(payload["position"], dict):
            raise InvalidPageToken("page token does not match the query")
        return payload["position"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, InvalidPageToken):
            raise
        raise InvalidPageToken("malformed page token") from exc


def _page_size(value: int, maximum: int) -> int:
    if value <= 0 or value > maximum:
        raise ValueError(f"page_size must be between 1 and {maximum}")
    return value


def _require_kind(key: ResourceKey, kind: ResourceKind) -> None:
    if key.kind is not kind:
        raise ValueError(f"expected {kind.value}, got {key.kind.value}")


def _stored_cluster(local_cluster_id: str, execution_cluster_id: str | None) -> str:
    if execution_cluster_id is None:
        return ""
    return LOCAL_CLUSTER if execution_cluster_id == local_cluster_id else execution_cluster_id


def _escaped_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"
