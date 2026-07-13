# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One per-cluster credential store, opaque to what mints the tokens.

This is the *mechanism* half of "one login": a single place a Marin client tool
caches the bearer material for a cluster, stored owner-only and written
atomically. rigging deliberately does not know how an ``edge_refresh_token`` was
obtained — orchestrating a login and deciding what goes in the record is the job
of the login orchestration layer above. Keeping the store opaque is what lets it
live at the leaf without dragging in iris/IAP concepts.

It supersedes the two overlapping stores it replaces — iris's
``~/.iris/tokens.sqlite`` and rigging's ``~/.config/marin/iap/<name>.json`` — with
one record per cluster under ``~/.config/marin/credentials/<cluster>.json``.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

_CREDENTIALS_DIR = Path.home() / ".config" / "marin" / "credentials"


def cluster_name_from_url(url: str) -> str:
    """Derive a filesystem-safe cluster name from a controller URL.

    Used to key the store when only a raw ``--controller-url`` is given, with no
    named cluster (``http://localhost:10000`` -> ``localhost-10000``).
    """
    parsed = urlparse(url)
    host = parsed.hostname or "unknown"
    return f"{host}-{parsed.port}" if parsed.port else host


@dataclass(frozen=True)
class CredentialRecord:
    """Cached bearer material for one cluster.

    Attributes:
        cluster: The logical cluster name this record belongs to.
        endpoint: The base URL the tokens authenticate to (e.g. the IAP edge).
        edge_refresh_token: A long-lived refresh token for the network edge (e.g.
            the IAP desktop-OAuth refresh token), used to silently re-mint a
            short-lived edge token. None for clusters with no edge auth.
        metadata: Free-form non-secret annotations (e.g. the user id a token was
            issued to), for display and diagnostics.
    """

    cluster: str
    endpoint: str
    edge_refresh_token: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


def credentials_dir() -> Path:
    """The directory holding per-cluster credential files."""
    return _CREDENTIALS_DIR


def credential_path(cluster: str) -> Path:
    """Path to the credential file for ``cluster``."""
    return _CREDENTIALS_DIR / f"{cluster}.json"


def load_credentials(cluster: str) -> CredentialRecord | None:
    """Load ``cluster``'s credentials, or None if the user has not logged in."""
    path = credential_path(cluster)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return CredentialRecord(
        cluster=data["cluster"],
        endpoint=data["endpoint"],
        edge_refresh_token=data.get("edge_refresh_token"),
        metadata=dict(data.get("metadata") or {}),
    )


def save_credentials(record: CredentialRecord) -> Path:
    """Persist ``record`` atomically with owner-only permissions.

    The directory is created ``0700`` and the file written ``0600`` *before* it is
    moved into place, so the secret is never world-readable even briefly.
    """
    path = credential_path(record.cluster)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(asdict(record), indent=2)
    # Write to a temp file in the same dir (so os.replace is atomic), fix its mode
    # before it holds the secret, then move it over the target in one step.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def delete_credentials(cluster: str) -> bool:
    """Drop ``cluster``'s cached credentials. Returns True if a file was removed."""
    path = credential_path(cluster)
    if not path.is_file():
        return False
    path.unlink()
    return True
