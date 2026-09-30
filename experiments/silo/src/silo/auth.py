# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Credentials between the three parties: worker, broker, host.

Two secrets, with different holders:

  SILO_API_TOKEN     worker -> broker, control plane. The direct analogue of the
                     DAYTONA_API_KEY a worker holds today.
  SILO_HOST_SECRET   broker <-> hosts only. Never leaves those jobs.

A worker reaches a host's data plane (exec, sessions, files) with a per-sandbox
capability, ``HMAC(SILO_HOST_SECRET, sandbox_id)``, which the broker hands back
with the sandbox. So a worker holds capabilities only for sandboxes it was issued,
and a leaked capability opens one sandbox, not the fleet.

Neither secret ever enters a sandbox: sandboxes run with ``--network none`` and
receive only the -e flags a caller passes explicitly.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

HEADER_API_TOKEN = "Authorization"
HEADER_SANDBOX_TOKEN = "X-Silo-Sandbox-Token"


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def sandbox_capability(host_secret: str, sandbox_id: str) -> str:
    return hmac.new(host_secret.encode(), f"sandbox:{sandbox_id}".encode(), hashlib.sha256).hexdigest()


def verify_sandbox_capability(host_secret: str, sandbox_id: str, presented: str | None) -> bool:
    if not presented:
        return False
    return hmac.compare_digest(sandbox_capability(host_secret, sandbox_id), presented)


def bearer(token: str) -> str:
    return f"Bearer {token}"


def verify_bearer(expected: str, presented: str | None) -> bool:
    if not presented or not presented.startswith("Bearer "):
        return False
    return hmac.compare_digest(expected, presented.removeprefix("Bearer "))
