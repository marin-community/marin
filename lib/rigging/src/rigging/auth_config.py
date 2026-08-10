# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative request-auth-stack schema.

An :class:`AuthStackConfig` is an ordered list of internally-tagged layer objects
(``{"type": <layer>, ...}``) describing the request chain — the authenticators
that decide an already-authenticated request. The request chain's verifier is
always the service JWT manager; there is no login-exchange step, since users
authenticate at the IAP edge and machines present a service JWT directly.

The wire shape matches finelog's ``FINELOG_AUTH_POLICY`` list, and the evaluation
order is first-match: the first layer to authenticate admits, the first to
reject denies, and an all-absent walk falls to a deny terminal (a stack whose last
layer is not ``anonymous`` is default-deny). A shared, language-neutral conformance
suite (:mod:`rigging.auth_vectors`) pins the evaluator behaviour across the Python
and Rust implementations rather than a second hand-maintained parser.

This module carries only the schema; :meth:`rigging.server_auth.RequestAuthPolicy.from_config`
compiles a stack into the concrete authenticator chain (binding the injected
verifiers).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class AuthLayerType(StrEnum):
    """The wire ``type`` tag of a request-auth layer."""

    JWT = "jwt"
    IAP_ASSERTION = "iap_assertion"
    CIDR = "cidr"
    LOOPBACK = "loopback"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class JwtLayer:
    """A bearer-token layer verified by the service's policy ``TokenVerifier``.

    Present+valid ⇒ AUTHENTICATED; absent ⇒ ABSENT. A present-but-invalid token is
    REJECTED when ``optional`` is false, else ABSENT (the best-effort case that lets
    a null-auth chain attribute a valid worker JWT but never reject).
    """

    TYPE: ClassVar[AuthLayerType] = AuthLayerType.JWT
    optional: bool = False

    def to_dict(self) -> dict:
        # Emit ``optional`` only when set, matching finelog's minimal ``{"type":"jwt"}``.
        if self.optional:
            return {"type": self.TYPE.value, "optional": True}
        return {"type": self.TYPE.value}

    @classmethod
    def from_dict(cls, data: dict) -> "JwtLayer":
        return cls(optional=bool(data.get("optional", False)))


@dataclass(frozen=True)
class IapAssertionLayer:
    """Verifies IAP's signed ``X-Goog-IAP-JWT-Assertion`` header.

    Forged ⇒ REJECTED; absent ⇒ ABSENT. Binds the injected ``iap_assertion_verifier``.
    """

    TYPE: ClassVar[AuthLayerType] = AuthLayerType.IAP_ASSERTION

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value}

    @classmethod
    def from_dict(cls, data: dict) -> "IapAssertionLayer":
        del data
        return cls()


@dataclass(frozen=True)
class CidrLayer:
    """Trusts a direct transport peer inside one of ``cidrs`` as the anonymous admin.

    A forwarded request (``X-Forwarded-For`` present / port 0) never matches.
    """

    TYPE: ClassVar[AuthLayerType] = AuthLayerType.CIDR
    cidrs: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value, "cidrs": list(self.cidrs)}

    @classmethod
    def from_dict(cls, data: dict) -> "CidrLayer":
        cidrs = data.get("cidrs")
        if not isinstance(cidrs, list) or not cidrs:
            raise ValueError("a 'cidr' auth layer requires a non-empty 'cidrs' list")
        return cls(cidrs=tuple(cidrs))


@dataclass(frozen=True)
class LoopbackLayer:
    """Trusts a genuine loopback transport peer (SSH tunnel / on-host) as admin."""

    TYPE: ClassVar[AuthLayerType] = AuthLayerType.LOOPBACK

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value}

    @classmethod
    def from_dict(cls, data: dict) -> "LoopbackLayer":
        del data
        return cls()


@dataclass(frozen=True)
class AnonymousLayer:
    """Terminal layer: admit any request as the anonymous admin (the permissive tail)."""

    TYPE: ClassVar[AuthLayerType] = AuthLayerType.ANONYMOUS

    def to_dict(self) -> dict:
        return {"type": self.TYPE.value}

    @classmethod
    def from_dict(cls, data: dict) -> "AnonymousLayer":
        del data
        return cls()


AuthLayerSpec = JwtLayer | IapAssertionLayer | CidrLayer | LoopbackLayer | AnonymousLayer

# Parse dispatch by the internally-tagged ``type`` field.
_LAYER_PARSERS: dict[AuthLayerType, Callable[[dict], AuthLayerSpec]] = {
    AuthLayerType.JWT: JwtLayer.from_dict,
    AuthLayerType.IAP_ASSERTION: IapAssertionLayer.from_dict,
    AuthLayerType.CIDR: CidrLayer.from_dict,
    AuthLayerType.LOOPBACK: LoopbackLayer.from_dict,
    AuthLayerType.ANONYMOUS: AnonymousLayer.from_dict,
}


def _layer_from_dict(entry: dict) -> AuthLayerSpec:
    """Parse one internally-tagged layer object; raise ValueError on an unknown ``type``."""
    if not isinstance(entry, dict):
        raise ValueError(f"an auth layer must be a JSON object, got {type(entry).__name__}")
    raw_type = entry.get("type")
    try:
        layer_type = AuthLayerType(raw_type)
    except ValueError as exc:
        raise ValueError(f"unknown auth layer type: {raw_type!r}") from exc
    return _LAYER_PARSERS[layer_type](entry)


@dataclass(frozen=True)
class AuthStackConfig:
    """An ordered, declarative request-auth-layer stack.

    ``layers`` is evaluation order: first-match admits, first-reject denies, and an
    all-absent walk falls to the deny terminal. Compile into a concrete chain with
    :meth:`rigging.server_auth.RequestAuthPolicy.from_config`.
    """

    layers: tuple[AuthLayerSpec, ...]

    @classmethod
    def from_json(cls, data: str | list[dict]) -> "AuthStackConfig":
        """Parse the wire list; raise ValueError on an empty list or an unknown ``type``.

        An empty list is a total lockout: a service passes an explicit
        default stack rather than relying on omission.
        """
        raw = json.loads(data) if isinstance(data, str) else data
        if not isinstance(raw, list):
            raise ValueError(f"an auth stack must be a JSON list, got {type(raw).__name__}")
        if not raw:
            raise ValueError("auth stack is an empty list (a total lockout — pass an explicit default stack)")
        return cls(layers=tuple(_layer_from_dict(entry) for entry in raw))

    def to_json(self) -> list[dict]:
        """Serialize to the ordered wire list of internally-tagged layer objects."""
        return [layer.to_dict() for layer in self.layers]
