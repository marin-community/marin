# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Iris cluster discovery and client credential resolution."""

from pathlib import Path

from rigging.cluster_manifest import AuthProvider, ClusterAuth, IapAuth
from rigging.credentials import ClientCredentials, credentials_for

from iris.cluster.config import AuthConfig, IrisClusterConfig


def _bundled_iris_config_dir() -> str | None:
    """Find configs in a wheel install or an editable workspace."""
    here = Path(__file__).resolve()
    wheel_path = here.parent.parent / "config"
    if wheel_path.is_dir():
        return str(wheel_path)
    editable_path = here.parents[3] / "config"
    if editable_path.is_dir():
        return str(editable_path)
    return None


# Directories searched (in priority order) to resolve ``--cluster=<name>`` to
# a YAML config file. Relative paths are resolved against the marin project
# root by ``rigging.config_discovery``; absolute paths are used as-is.
IRIS_CLUSTER_CONFIG_DIRS: tuple[str, ...] = tuple(
    p
    for p in (
        "~/.config/marin/clusters",  # user override — checked first
        "lib/iris/config",  # in-tree marin checkout
        _bundled_iris_config_dir(),  # editable install from sibling workspace
    )
    if p is not None
)


def _cluster_auth_from_config(auth: AuthConfig) -> ClusterAuth:
    """Adapt iris's ``AuthConfig`` to rigging's shared credential vocabulary."""
    provider = auth.provider_kind()
    if provider == "iap":
        # `programmatic_audiences` are the service-account edge audiences the
        # client mints against. Empty is fine: rigging's edge resolver falls back
        # to the desktop client id, which IAP registers as a programmatic client.
        return ClusterAuth(
            AuthProvider.IAP,
            iap=IapAuth(
                url=auth.iap.url,
                desktop_oauth_client_id=auth.iap.oauth_client_id or None,
                desktop_oauth_client_secret=auth.iap.oauth_client_secret or None,
                programmatic_audiences=tuple(auth.iap.programmatic_audiences),
                signed_header_audience=auth.iap.signed_header_audience or None,
            ),
        )
    return ClusterAuth(AuthProvider.NONE)


def client_credentials(
    config: IrisClusterConfig | None,
    cluster_name: str,
) -> ClientCredentials:
    """Resolve the cluster's client credentials via the shared rigging resolver."""
    auth = (
        _cluster_auth_from_config(config.auth)
        if config is not None and config.auth is not None
        else ClusterAuth(AuthProvider.NONE)
    )
    return credentials_for(cluster_name, auth)
