# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cli.connect import _cluster_auth_from_config
from iris.cluster.config import AuthConfig, IapAuthConfig
from rigging.cluster_manifest import AuthProvider


def test_cluster_auth_from_config_drops_desktop_audience_from_programmatic():
    """A dedicated programmatic audience takes precedence over the desktop id.

    ``audiences`` lists every ``aud`` the controller accepts at login, which
    includes the desktop client id. When a cluster also configures a dedicated
    IAP-secured audience, the service-account edge path should prefer it, so the
    adapter drops the desktop id from ``programmatic_audiences``. (With no
    dedicated audience, rigging's edge resolver falls back to the desktop client
    id, which IAP also admits -- see ``test_credentials``.)
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
    *dedicated* programmatic audience. The service-account edge path then falls
    back to the desktop client id in rigging's resolver (see ``test_credentials``),
    so a token is still minted -- this only asserts the adapter's output."""
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
