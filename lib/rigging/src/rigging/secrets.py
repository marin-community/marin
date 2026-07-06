# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ordered, reference-based secret resolution shared across Marin services.

A secret-bearing config field is a reference (or an ordered list of them)
rather than an inlined value, so plaintext never reaches a rendered ConfigMap or
GCE startup-metadata (#6873). Each reference names where a secret lives:

    env:NAME
    file:/abs/path
    gcp-secret://projects/<p>/secrets/<n>/versions/<v>

`resolve_secret_spec` walks the ordered list first-present-wins with an
absent-vs-failed discipline that mirrors the request-auth chain: an absent
source (unset env / missing file / secret NOT_FOUND) skips to the next; a
configured-but-erroring one (denied IAM, unreachable, unreadable, malformed)
raises immediately and never silently shadows to a staler/weaker source.

There is deliberately no ``k8s-secret://`` scheme: the k8s-native path is a
Secret projected to ``env:`` (``envFrom``) or ``file:`` (CSI volume), so the
controller never needs ``secrets: get`` on its ClusterRole.
"""

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENV_SCHEME = "env:"
FILE_SCHEME = "file:"
GCP_SECRET_SCHEME = "gcp-secret://"

# A scheme-shaped prefix (``foo:``): used to tell an unknown-scheme reference
# (raise) apart from a bare literal secret (also raise, different message).
_SCHEME_SHAPED_RE = re.compile(r"^[a-z0-9+-]+:")

# An ordered list of references; a bare string normalizes to a one-element tuple.
SecretSpec = tuple[str, ...]


class SecretResolutionError(RuntimeError):
    """A configured secret source failed to read (as distinct from being absent)."""


@dataclass(frozen=True)
class ResolvedSecret:
    """A resolved secret plus the reference that produced it (safe to log)."""

    value: str
    source: str


def as_secret_spec(spec: str | Sequence[str]) -> SecretSpec:
    """Normalize a bare reference or an ordered list into a `SecretSpec`.

    This is the single boundary that accepts the `str | Sequence[str]` sugar;
    everything downstream operates on the normalized tuple.
    """
    if isinstance(spec, str):
        return (spec,)
    return tuple(spec)


def is_secret_reference(value: str) -> bool:
    """True if `value` begins with a known secret-source scheme."""
    return value.startswith((ENV_SCHEME, FILE_SCHEME, GCP_SECRET_SCHEME))


def _fetch_env(ref: str) -> str | None:
    name = ref[len(ENV_SCHEME) :]
    if not name:
        raise SecretResolutionError(f"{ref!r}: env reference has an empty variable name")
    value = os.environ.get(name)
    return value.strip() if value is not None else None


def _fetch_file(ref: str) -> str | None:
    path = ref[len(FILE_SCHEME) :]
    if not path:
        raise SecretResolutionError(f"{ref!r}: file reference has an empty path")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return None  # ABSENT — try the next source
    except OSError as exc:  # permission denied, is-a-directory, decode error, … — FAILED
        raise SecretResolutionError(f"{ref!r}: {exc}") from exc


def _fetch_gcp_secret(ref: str) -> str | None:
    resource = ref[len(GCP_SECRET_SCHEME) :]
    if "/versions/" not in resource:
        raise SecretResolutionError(
            f"{ref!r}: gcp-secret reference must pin an explicit version "
            "(…/versions/<n>); an unversioned reference is rejected"
        )
    try:
        from google.cloud import secretmanager  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise SecretResolutionError(
            f"{ref!r}: resolving a gcp-secret reference needs the optional dependency; " "install marin-rigging[secrets]"
        ) from exc
    from google.api_core.exceptions import GoogleAPICallError, NotFound  # noqa: PLC0415  # optional dep

    client = secretmanager.SecretManagerServiceClient()
    try:
        response = client.access_secret_version(name=resource)
    except NotFound:
        return None  # ABSENT — try the next source
    except GoogleAPICallError as exc:  # denied IAM / unreachable / … — FAILED
        raise SecretResolutionError(f"{ref!r}: {exc}") from exc
    return response.payload.data.decode("utf-8").strip()


_FETCHERS: dict[str, Callable[[str], str | None]] = {
    ENV_SCHEME: _fetch_env,
    FILE_SCHEME: _fetch_file,
    GCP_SECRET_SCHEME: _fetch_gcp_secret,
}


def _fetch(ref: str) -> str | None:
    for scheme, fetch in _FETCHERS.items():
        if ref.startswith(scheme):
            return fetch(ref)
    if _SCHEME_SHAPED_RE.match(ref):
        raise SecretResolutionError(f"{ref!r}: unknown secret-source scheme (expected env: / file: / gcp-secret://)")
    raise SecretResolutionError(
        f"{ref!r}: not a secret reference (expected env: / file: / gcp-secret://); "
        "a bare literal secret is not permitted in config"
    )


def _scheme_of(ref: str) -> str:
    """The source-scheme category of a reference (``env`` / ``file`` / ``gcp-secret``).

    Returns one of the fixed scheme constants — a locator category, derived from
    ``_FETCHERS`` and never from the secret value or the reference body — so it is
    safe to log while the reference itself (which may name a sensitive path) is not.
    """
    for scheme in _FETCHERS:
        if ref.startswith(scheme):
            return scheme.rstrip(":/")
    return "unknown"


def resolve_secret_spec(spec: str | Sequence[str]) -> ResolvedSecret:
    """Resolve an ordered secret path, first PRESENT source wins.

    Raises `SecretResolutionError` if a source is configured-but-erroring, if a
    reference has an unknown/absent scheme, or if every source is absent.
    """
    refs = as_secret_spec(spec)
    if not refs:
        raise SecretResolutionError("empty secret spec: no sources to resolve")
    for ref in refs:
        value = _fetch(ref)  # None ⇒ ABSENT (next); raise ⇒ FAILED (propagate)
        if value is not None:
            logger.info("resolved a secret from a %s source", _scheme_of(ref))
            return ResolvedSecret(value=value, source=ref)
    raise SecretResolutionError(f"no secret source produced a value (tried: {', '.join(refs)})")


def default_secret_spec(field_name: str, *, env_prefix: str, secrets_dir: str) -> SecretSpec:
    """The conventional env:/file: path for a field with no explicit reference.

    `gcp-secret://` is intentionally absent — its mandatory version segment
    cannot be conventional — so a Secret-Manager source is always explicit
    config. `env_prefix`/`secrets_dir` are passed by the service so this module
    stays service-generic (e.g. iris passes `IRIS` / `/etc/iris/secrets`).
    """
    return (
        f"{ENV_SCHEME}{env_prefix}_{field_name.upper()}",
        f"{FILE_SCHEME}{secrets_dir.rstrip('/')}/{field_name}",
    )
