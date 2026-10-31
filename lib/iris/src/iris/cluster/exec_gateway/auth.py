# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only execution grants, verified against operator-pinned public keys."""

import json
from dataclasses import asdict, dataclass

from rigging.token_authority import JwksVerifier, JwtSigner

EXEC_SCOPE = "exec-session"
EXEC_GRANT_TTL = 3600
MAX_COMMAND_JSON_BYTES = 64 * 1024


@dataclass(frozen=True)
class ExecPodTarget:
    namespace: str
    pod_name: str
    pod_uid: str
    container_name: str
    attempt_uid: str


@dataclass(frozen=True)
class ExecGrant:
    grant_id: str
    subject: str
    target: ExecPodTarget
    command: tuple[str, ...]
    expires_at_ms: int


def exec_audience(gateway_id: str) -> str:
    if not gateway_id:
        raise ValueError("Exec gateway identity is required")
    return f"iris-exec:{gateway_id}"


def mint_exec_grant(
    signer: JwtSigner,
    *,
    gateway_id: str,
    grant_id: str,
    subject: str,
    target: ExecPodTarget,
    command: list[str],
) -> str:
    return signer.mint(
        {"sub": subject, "jti": grant_id, "scope": EXEC_SCOPE, "target": asdict(target), "command": command},
        audience=exec_audience(gateway_id),
        ttl_seconds=EXEC_GRANT_TTL,
    )


def verify_exec_grant(verifier: JwksVerifier, token: str) -> ExecGrant:
    verified = verifier.verify(token)
    claims = verified.claims
    target = claims.get("target")
    command = claims.get("command")
    grant_id = claims.get("jti")
    if verified.scope != EXEC_SCOPE or not verified.sub or not isinstance(grant_id, str) or not grant_id:
        raise ValueError("Invalid execution grant identity or scope")
    if not isinstance(target, dict) or set(target) != set(ExecPodTarget.__dataclass_fields__):
        raise ValueError("Invalid execution target")
    if not all(isinstance(value, str) and value for value in target.values()):
        raise ValueError("Execution target fields must be nonempty strings")
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("Execution command must be a nonempty string list")
    if len(json.dumps(command).encode()) > MAX_COMMAND_JSON_BYTES:
        raise ValueError("Execution command exceeds grant budget")
    return ExecGrant(grant_id, verified.sub, ExecPodTarget(**target), tuple(command), int(claims["exp"]) * 1000)
