# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cross-lib slice of a cluster manifest (``config/<cluster>.yaml``).

A cluster manifest is the single top-level file describing one logical cluster.
Different layers read different slices of it:

- ``rigging`` models the narrow slice every client must agree on — the
  cluster's identity (name, public origins) and how to authenticate to its edge
  (:class:`ClusterAuth`). This is pure mechanism: a typed view over YAML, with no
  IO, no orchestration, and no knowledge of what mints a token.
- ``data:`` is parsed separately into a :class:`~rigging.filesystem.DataConfig`
  (storage layout); :func:`load_manifest` attaches it but does not redefine it.
- Sections owned by callers above rigging — ``provisioning:`` (one-time GCP
  rollouts) and ``policy:`` (per-user budgets/roles) — are intentionally not
  modeled here. The loader tolerates them and exposes the raw document via
  :attr:`ClusterManifest.document` so a higher layer can parse its own slice.

Keeping this leaf narrow is what lets iris and finelog read identity/auth without
importing the admin tool, while the admin tool layers provisioning/policy on top.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem import MARIN_CLUSTER_CONFIG_DIRS

_MARIN_CLUSTER_ENV = "MARIN_CLUSTER"
_DEFAULT_CLUSTER = "marin"


class AuthProvider(StrEnum):
    """Which edge-auth mechanism a cluster uses.

    The provider is implied by which ``auth:`` sub-block is present; ``NONE`` is a
    loopback / unauthenticated cluster (e.g. a local controller).
    """

    IAP = "iap"
    GCP = "gcp"
    NONE = "none"


@dataclass(frozen=True)
class IapAuth:
    """IAP edge-auth parameters a client needs to authenticate and a backend to verify.

    ``desktop_oauth_client_secret`` is the Google desktop client secret, which is
    non-confidential per RFC 8252 §8.5 and may be committed; confidential material
    (e.g. the GCLB web client secret) belongs in ``provisioning:`` as a reference.
    """

    url: str
    desktop_oauth_client_id: str | None = None
    desktop_oauth_client_secret: str | None = None
    programmatic_audiences: tuple[str, ...] = ()
    signed_header_audience: str | None = None


@dataclass(frozen=True)
class ClusterAuth:
    """How clients authenticate to a cluster's edge, plus the admin allowlist."""

    provider: AuthProvider
    iap: IapAuth | None = None
    admin_users: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterManifest:
    """The identity + auth slice of a cluster manifest, plus the raw document.

    Attributes:
        name: The logical cluster name (``identity.name``, defaulting to the file stem).
        dashboard_url: The cluster's public dashboard origin, if any.
        auth: How to authenticate to the cluster edge.
        document: The full parsed YAML document, for sections this leaf does not
            model (``provisioning:``, ``policy:``, ``services:``, ``data:``).
    """

    name: str
    dashboard_url: str | None
    auth: ClusterAuth
    document: Mapping[str, Any] = field(default_factory=dict)


def _parse_iap(raw: Mapping[str, Any]) -> IapAuth:
    url = raw.get("url")
    if not url:
        raise ValueError("auth.iap requires a 'url'")
    audiences = raw.get("programmatic_audiences") or ()
    return IapAuth(
        url=str(url),
        desktop_oauth_client_id=_opt_str(raw.get("desktop_oauth_client_id")),
        desktop_oauth_client_secret=_opt_str(raw.get("desktop_oauth_client_secret")),
        programmatic_audiences=tuple(str(a) for a in audiences),
        signed_header_audience=_opt_str(raw.get("signed_header_audience")),
    )


def _parse_auth(raw: Mapping[str, Any]) -> ClusterAuth:
    admin_users = tuple(str(u) for u in raw.get("admin_users") or ())
    if "iap" in raw:
        return ClusterAuth(AuthProvider.IAP, iap=_parse_iap(raw["iap"]), admin_users=admin_users)
    if raw.get("gcp"):
        return ClusterAuth(AuthProvider.GCP, admin_users=admin_users)
    return ClusterAuth(AuthProvider.NONE, admin_users=admin_users)


def parse_manifest(document: Mapping[str, Any], *, name: str) -> ClusterManifest:
    """Build a :class:`ClusterManifest` from a parsed manifest ``document``.

    ``name`` is the fallback cluster name used when ``identity.name`` is absent
    (typically the config file stem).
    """
    identity = document.get("identity") or {}
    auth_raw = document.get("auth") or {}
    return ClusterManifest(
        name=str(identity.get("name") or name),
        dashboard_url=_opt_str(identity.get("dashboard_url")),
        auth=_parse_auth(auth_raw),
        document=document,
    )


def load_manifest(cluster: str | None = None, *, dirs: tuple[str, ...] = MARIN_CLUSTER_CONFIG_DIRS) -> ClusterManifest:
    """Resolve and parse the manifest for ``cluster``.

    The cluster name is ``cluster`` arg > ``MARIN_CLUSTER`` env > ``marin``; it is
    resolved to ``config/<cluster>.yaml`` via the standard search dirs.

    Raises:
        FileNotFoundError: when no manifest is found for a named cluster.
    """
    name = cluster or os.environ.get(_MARIN_CLUSTER_ENV) or _DEFAULT_CLUSTER
    path = resolve_cluster_config(name, dirs)
    document = yaml.safe_load(Path(path).read_text()) or {}
    return parse_manifest(document, name=path.stem)


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
