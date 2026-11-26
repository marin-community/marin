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

Kubernetes Secrets and CSI providers project files; applications need no
Kubernetes API access. Named lookups prefer files and fall back to env for absent keys.
"""

import io
import logging
import os
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from configparser import ConfigParser
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_SCHEME = "env:"
FILE_SCHEME = "file:"
GCP_SECRET_SCHEME = "gcp-secret://"
SECRETS_DIR = Path("/run/rigging/secrets")
SECRETS_DIR_ENV = "RIGGING_SECRETS_DIR"
COREWEAVE_SECRET_NAMES = ("CW_KEY_ID", "CW_KEY_SECRET")
AWS_CREDENTIALS_FILENAME = ".aws-credentials"
_FORWARDED_SECRET_NAMES = ("HF_TOKEN", "WANDB_API_KEY", *COREWEAVE_SECRET_NAMES)
_SECRET_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")

# A scheme-shaped prefix (``foo:``): used to tell an unknown-scheme reference
# (raise) apart from a bare literal secret (also raise, different message).
_SCHEME_SHAPED_RE = re.compile(r"^[a-z0-9+-]+:")

# An ordered list of references; a bare string normalizes to a one-element tuple.
SecretSpec = tuple[str, ...]


class SecretResolutionError(RuntimeError):
    """A configured secret source failed to read (as distinct from being absent)."""


class SecretBehavior(StrEnum):
    """Inherit available credentials or use only explicitly supplied secrets."""

    INHERIT = "inherit"
    ISOLATE = "isolate"


@dataclass(frozen=True)
class ResolvedSecret:
    """A resolved secret whose value is excluded from its representation."""

    value: str = field(repr=False)
    source: str


def secret_path(name: str, *, directory: Path | None = None) -> Path:
    """Return a named secret's path without reading its value."""
    validate_secret_name(name)
    root = directory if directory is not None else Path(os.environ.get(SECRETS_DIR_ENV, SECRETS_DIR))
    return root / name


def validate_secret_name(name: str) -> None:
    """Reject names that could escape a secret directory."""
    if not _SECRET_NAME_RE.fullmatch(name):
        raise ValueError("secret names must contain only letters, digits, underscores, dots, and hyphens")


def list_secret_names(*, directory: Path | None = None) -> tuple[str, ...]:
    """List mounted secret names without reading values or guessing env names.

    Kubernetes projection metadata and subdirectories are excluded. A missing
    directory has no mounted secrets; other filesystem errors propagate.
    """
    root = directory if directory is not None else Path(os.environ.get(SECRETS_DIR_ENV, SECRETS_DIR))
    try:
        return tuple(
            sorted(path.name for path in root.iterdir() if _SECRET_NAME_RE.fullmatch(path.name) and path.is_file())
        )
    except FileNotFoundError:
        return ()


def get_secret(name: str, *, directory: Path | None = None) -> str | None:
    """Read a file first, falling back to the environment for an absent key."""
    path = secret_path(name, directory=directory)
    try:
        value = _fetch_file(f"{FILE_SCHEME}{path}")
    except SecretResolutionError:
        raise SecretResolutionError(f"secret {name!r} could not be read") from None
    return value if value is not None else _fetch_env(f"{ENV_SCHEME}{name}")


def require_secret(name: str, *, directory: Path | None = None) -> str:
    """Read a named credential, failing if it is missing or empty."""
    value = get_secret(name, directory=directory)
    if not value:
        raise SecretResolutionError(f"required secret {name!r} is missing or empty")
    return value


def forwarded_secrets(*, exclude: Collection[str] = ()) -> dict[str, str]:
    """Read the current mounted bundle and available HF/W&B/CoreWeave defaults."""
    return {
        name: value
        for name in dict.fromkeys((*list_secret_names(), *_FORWARDED_SECRET_NAMES))
        if name not in exclude and (value := get_secret(name)) is not None
    }


def secret_file_environment(names: Collection[str], *, directory: Path = SECRETS_DIR) -> dict[str, str]:
    """Return nonsecret file locations understood by native credential clients."""
    env = {"HF_TOKEN_PATH": str(directory / "HF_TOKEN")} if "HF_TOKEN" in names else {}
    if all(name in names for name in COREWEAVE_SECRET_NAMES):
        env["AWS_SHARED_CREDENTIALS_FILE"] = str(directory / AWS_CREDENTIALS_FILENAME)
    return env


def secret_file_contents(values: Mapping[str, str]) -> dict[str, str]:
    """Prepare named files and native credential formats for a delivered bundle.

    Derived dotfiles are excluded from named-secret discovery and inheritance.
    """
    for name in values:
        validate_secret_name(name)
    files = dict(values)
    if all(values.get(name) for name in COREWEAVE_SECRET_NAMES):
        credentials = ConfigParser(interpolation=None)
        credentials["default"] = {
            "aws_access_key_id": values["CW_KEY_ID"],
            "aws_secret_access_key": values["CW_KEY_SECRET"],
        }
        output = io.StringIO()
        credentials.write(output)
        files[AWS_CREDENTIALS_FILENAME] = output.getvalue()
    return files


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
    except (OSError, UnicodeError):
        # Decode errors can contain payload bytes; do not chain them into logs.
        raise SecretResolutionError("file secret source could not be read") from None


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
    except GoogleAPICallError:
        raise SecretResolutionError("cloud secret source could not be read") from None
    try:
        return response.payload.data.decode("utf-8").strip()
    except UnicodeError:
        raise SecretResolutionError("cloud secret source is not UTF-8") from None


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
        raise SecretResolutionError("unknown secret-source scheme (expected env: / file: / gcp-secret://)")
    raise SecretResolutionError(
        "not a secret reference (expected env: / file: / gcp-secret://); "
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
