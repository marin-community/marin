# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from click.testing import CliRunner
from iris.cli import cluster as cluster_cli
from iris.cli.connect import _cluster_auth_from_config
from iris.cluster.config import AuthConfig, IapAuthConfig
from rigging.auth import StaticTokenProvider
from rigging.cluster_manifest import AuthProvider
from rigging.credentials import ClientCredentials


def test_cluster_auth_from_config_passes_programmatic_audiences_through():
    """The service-account edge audience is configured explicitly and passed
    straight through to rigging's credential vocabulary."""
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
            programmatic_audiences=["iap-secured.apps.googleusercontent.com"],
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.provider is AuthProvider.IAP
    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ("iap-secured.apps.googleusercontent.com",)


def test_cluster_auth_from_config_empty_programmatic_audiences():
    """With no ``programmatic_audiences`` set the adapter exposes none; the edge path
    then falls back to the desktop client id in rigging's resolver (see
    ``test_credentials``)."""
    auth = AuthConfig(
        iap=IapAuthConfig(
            url="https://iris.example",
            oauth_client_id="desktop.apps.googleusercontent.com",
        )
    )

    cluster_auth = _cluster_auth_from_config(auth)

    assert cluster_auth.iap is not None
    assert cluster_auth.iap.programmatic_audiences == ()


def test_dashboard_proxy_command_forwards_context_credentials_to_the_proxy(monkeypatch):
    """The browser holds no cluster credentials, so the proxy must send the operator's.

    Regression guard: the command resolved them onto the context and then built the
    proxy without them, so every upstream call hit an IAP-fronted controller
    unauthenticated.
    """
    credentials = ClientCredentials(iap_provider=StaticTokenProvider("iap-token"))
    built = {}

    class StubDashboard:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.app = object()

    monkeypatch.setattr(cluster_cli, "ProxyControllerDashboard", StubDashboard)
    monkeypatch.setattr(cluster_cli, "require_controller_url", lambda ctx: "https://iris.example")
    monkeypatch.setattr(cluster_cli.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(cluster_cli.subprocess, "Popen", lambda *a, **kw: Mock())
    monkeypatch.setattr(cluster_cli.uvicorn, "run", lambda *a, **kw: None)

    CliRunner().invoke(
        cluster_cli.cluster_dashboard_proxy,
        obj={"credentials": credentials},
        catch_exceptions=False,
    )

    assert built["credentials"] is credentials
