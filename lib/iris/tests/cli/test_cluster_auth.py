# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cli.connect import _cluster_auth_from_config
from iris.cluster.config import AuthConfig, IapAuthConfig
from rigging.cluster_manifest import AuthProvider


def test_cluster_auth_from_config_drops_desktop_audience_from_programmatic():
    """The desktop client id is a login audience, not a programmatic edge audience.

    ``audiences`` lists every ``aud`` the controller accepts at login, which
    includes the desktop client id. A service-account edge token minted for the
    desktop ``aud`` is rejected by IAP, so the adapter must keep only the
    IAP-secured audiences in ``programmatic_audiences``.
    """
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
            audiences=["desktop.apps.googleusercontent.com", "iap-secured.apps.googleusercontent.com"],
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.provider is AuthProvider.IAP
    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ("iap-secured.apps.googleusercontent.com",)


def test_cluster_auth_from_config_desktop_only_has_no_programmatic_audience():
    """A user-flow-only config (audiences == the desktop client) exposes no
    programmatic audience, so no service-account edge token is ever minted."""
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
            audiences=["desktop.apps.googleusercontent.com"],
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ()
