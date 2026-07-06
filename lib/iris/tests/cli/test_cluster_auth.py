# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cli.connect import _cluster_auth_from_config
from iris.cluster.config import AuthConfig, IapAuthConfig
from rigging.cluster_manifest import AuthProvider


def test_cluster_auth_from_config_passes_programmatic_audiences_through():
    """The service-account edge audience is configured explicitly and passed
    straight through -- not derived from the login ``audiences`` list."""
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
            audiences=["desktop.apps.googleusercontent.com"],
            programmatic_audiences=["iap-secured.apps.googleusercontent.com"],
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.provider is AuthProvider.IAP
    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ("iap-secured.apps.googleusercontent.com",)


def test_cluster_auth_from_config_login_audiences_do_not_leak_into_programmatic():
    """``audiences`` (login) and ``programmatic_audiences`` (service-account edge)
    are independent: with no ``programmatic_audiences`` set the adapter exposes
    none, regardless of what ``audiences`` lists. The edge path then falls back to
    the desktop client id in rigging's resolver (see ``test_credentials``)."""
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
            audiences=["desktop.apps.googleusercontent.com", "extra-login.apps.googleusercontent.com"],
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ()
