# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test-support doubles for services built on rigging's auth stack.

Shipped alongside the library (not under ``tests/``) so both rigging's own suite
and downstream consumers such as iris import one copy rather than each carrying
its own.
"""

import json
import threading
from collections.abc import Mapping
from typing import Any, cast

from rigging.server_auth import VerifiedIdentity


class MockVerifier:
    """Deterministic :class:`~rigging.server_auth.TokenVerifier` double: maps fixed tokens to identities.

    A stand-in for a real service verifier (which checks JWTs it signed) in tests
    that only need a bearer token to resolve to a known user. Not a production auth
    mechanism.

    Args:
        tokens: Mapping of token string to username. Every user gets role ``"user"``.
    """

    def __init__(self, tokens: dict[str, str]):
        self._tokens = tokens

    def verify(self, token: str) -> VerifiedIdentity:
        user = self._tokens.get(token)
        if user is None:
            raise ValueError("Invalid token")
        return VerifiedIdentity(user_id=user, role="user")


class _TelemetryResponse:
    status_code = 200

    def __init__(self, batch_id: str) -> None:
        self._batch_id = batch_id
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        return {"batch_id": self._batch_id, "status": "accepted"}


class RecordingTelemetryTransport:
    """Capture telemetry records and return valid acknowledgements in tests."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self.condition = threading.Condition()

    def post(self, endpoint: str, body: bytes, batch_id: str, timeout: tuple[float, float]) -> _TelemetryResponse:
        del endpoint, timeout
        payload = json.loads(body)
        with self.condition:
            self.records.extend(payload["records"])
            self.resources.append(payload["resource"])
            self.condition.notify_all()
        return _TelemetryResponse(batch_id)

    def close(self) -> None:
        pass

    def wait_for(self, count: int) -> list[dict[str, Any]]:
        with self.condition:
            assert self.condition.wait_for(lambda: len(self.records) >= count, timeout=5)
            return list(self.records)

    def record(self, name: str, attributes: Mapping[str, str]) -> dict[str, Any]:
        def matches(record: dict[str, Any]) -> bool:
            record_attributes = cast(dict[str, str], record["attributes"])
            return record["name"] == name and all(
                record_attributes.get(key) == value for key, value in attributes.items()
            )

        with self.condition:
            assert self.condition.wait_for(lambda: any(matches(record) for record in self.records), timeout=5)
            return [record for record in self.records if matches(record)][-1]

    def wait_for_value(self, name: str, attributes: Mapping[str, str], expected: float) -> None:
        with self.condition:
            assert self.condition.wait_for(
                lambda: any(
                    record["name"] == name
                    and record["value"] == expected
                    and all(
                        cast(dict[str, str], record["attributes"]).get(key) == value for key, value in attributes.items()
                    )
                    for record in self.records
                ),
                timeout=5,
            )
