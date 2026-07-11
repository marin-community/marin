# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`iris whoami` login-status dump and the email stashed by `iris login`."""

import base64
import importlib
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

from click.testing import CliRunner
from iris.cli import whoami as whoami_cli
from iris.cluster.config import AuthConfig, IapAuthConfig
from rigging import credential_store
from rigging.cluster_manifest import AuthProvider
from rigging.login_status import LoginMethod, LoginState, LoginStatus

# `iris.cli.main` the submodule is shadowed by the `main` entry-point function
# re-exported in `iris/cli/__init__.py`, so import it by name to patch its globals.
main_cli = importlib.import_module("iris.cli.main")


def _id_token(claims: dict) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


def _status(cluster: str, **kw) -> LoginStatus:
    base = dict(
        state=LoginState.ACTIVE,
        method=LoginMethod.IAP_BROWSER,
        endpoint="https://iris.example",
        email=f"{cluster}@marin.io",
        token_expiry=None,
        detail=None,
    )
    base.update(kw)
    return LoginStatus(cluster=cluster, **base)


# --------------------------------------------------------------------------- #
# whoami output
# --------------------------------------------------------------------------- #


def test_whoami_json_is_the_machine_readable_contract(monkeypatch):
    expiry = datetime.fromtimestamp(1_900_000_000, tz=UTC)
    status = _status("marin-eu", token_expiry=expiry, email="alice@marin.io")
    monkeypatch.setattr(whoami_cli, "resolve_login_status", lambda cluster, auth: status)

    result = CliRunner().invoke(
        whoami_cli.whoami, ["--format", "json"], obj={"cluster_name": "marin-eu"}, catch_exceptions=False
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == [
        {
            "cluster": "marin-eu",
            "endpoint": "https://iris.example",
            "method": "iap-browser",
            "state": "active",
            "email": "alice@marin.io",
            "token_expiry": expiry.isoformat(),
            "detail": None,
        }
    ]


def test_whoami_text_dump_names_the_signed_in_email(monkeypatch):
    status = _status("marin-eu", email="alice@marin.io")
    monkeypatch.setattr(whoami_cli, "resolve_login_status", lambda cluster, auth: status)

    result = CliRunner().invoke(whoami_cli.whoami, [], obj={"cluster_name": "marin-eu"}, catch_exceptions=False)

    assert result.exit_code == 0
    # The whole point of the dump is to surface who you are and where.
    assert "alice@marin.io" in result.output
    assert "https://iris.example" in result.output


def test_whoami_offline_status_reports_the_failure_and_cached_email(monkeypatch):
    status = _status(
        "marin-eu",
        state=LoginState.OFFLINE,
        email="alice@marin.io",
        detail="cached IAP credentials are no longer valid",
    )
    monkeypatch.setattr(whoami_cli, "resolve_login_status", lambda cluster, auth: status)

    result = CliRunner().invoke(
        whoami_cli.whoami, ["--format", "json"], obj={"cluster_name": "marin-eu"}, catch_exceptions=False
    )

    row = json.loads(result.output)[0]
    assert row["state"] == "offline"
    assert row["email"] == "alice@marin.io"
    assert row["detail"] == "cached IAP credentials are no longer valid"


def test_whoami_respects_explicit_non_iap_config_over_stale_cache(monkeypatch):
    # A loaded non-IAP config is authoritative: a leftover credential file carrying
    # a refresh token for the same cluster name must not flip it to a synthetic IAP
    # login (which would mislead and trigger a needless edge-token mint).
    captured = {}

    def fake_resolve(cluster, auth):
        captured["provider"] = auth.provider
        return _status(cluster, method=LoginMethod.NETWORK_TRUST, state=LoginState.NO_LOGIN_REQUIRED, email=None)

    monkeypatch.setattr(whoami_cli, "resolve_login_status", fake_resolve)
    monkeypatch.setattr(
        whoami_cli,
        "load_credentials",
        lambda cluster: credential_store.CredentialRecord(
            cluster=cluster, endpoint="https://stale", edge_refresh_token="rt"
        ),
    )

    non_iap_config = SimpleNamespace(auth=None)  # cluster_auth_for -> network trust
    result = CliRunner().invoke(
        whoami_cli.whoami,
        ["--format", "json"],
        obj={"cluster_name": "c", "config": non_iap_config},
        catch_exceptions=False,
    )

    assert captured["provider"] is AuthProvider.NONE
    assert json.loads(result.output)[0]["method"] == "network-trust"


def test_whoami_all_enumerates_configured_and_cached_only_clusters(monkeypatch, tmp_path):
    # One cluster has a config; another was logged into by raw URL (cache only, no
    # config). --all must report both, without duplicating the configured one.
    monkeypatch.setattr(whoami_cli, "list_cluster_configs", lambda dirs: {"configured": tmp_path / "configured.yaml"})
    monkeypatch.setattr(
        whoami_cli, "load_config", lambda path: SimpleNamespace(auth=AuthConfig(iap=IapAuthConfig(url="https://cfg")))
    )
    monkeypatch.setattr(whoami_cli, "list_cached_clusters", lambda: ["configured", "cache-only"])
    monkeypatch.setattr(
        whoami_cli,
        "load_credentials",
        lambda cluster: credential_store.CredentialRecord(
            cluster=cluster, endpoint="https://cached", edge_refresh_token="rt"
        ),
    )

    def fake_resolve(cluster, auth):
        # A cache-only cluster must be resolved as an IAP browser login, proving
        # `_status_for` synthesized IAP auth from the cached refresh token.
        method = LoginMethod.IAP_BROWSER if auth.provider is AuthProvider.IAP else LoginMethod.NETWORK_TRUST
        return _status(cluster, method=method)

    monkeypatch.setattr(whoami_cli, "resolve_login_status", fake_resolve)

    result = CliRunner().invoke(whoami_cli.whoami, ["--all", "--format", "json"], obj={}, catch_exceptions=False)

    rows = json.loads(result.output)
    assert {r["cluster"] for r in rows} == {"configured", "cache-only"}
    assert all(r["method"] == "iap-browser" for r in rows)


# --------------------------------------------------------------------------- #
# expiry formatting
# --------------------------------------------------------------------------- #


def test_format_expiry_relative_labels():
    now = datetime.fromtimestamp(1_000_000, tz=UTC)
    future = datetime.fromtimestamp(1_000_000 + 3660, tz=UTC)
    past = datetime.fromtimestamp(1_000_000 - 120, tz=UTC)

    assert "in 1h1m" in whoami_cli._format_expiry(future, now)
    assert "expired 2m ago" in whoami_cli._format_expiry(past, now)
    assert whoami_cli._format_expiry(None, now) == "—"


# --------------------------------------------------------------------------- #
# iris login persists the signed-in email
# --------------------------------------------------------------------------- #


def test_login_persists_signed_in_email(monkeypatch, tmp_path):
    monkeypatch.setattr(credential_store, "_CREDENTIALS_DIR", tmp_path)
    token = _id_token({"email": "dev@marin.io", "exp": int(time.time()) + 3600})
    monkeypatch.setattr(main_cli, "run_iap_desktop_login", lambda *a, **k: (token, "refresh-token-xyz"))

    config = SimpleNamespace(auth=AuthConfig(iap=IapAuthConfig(url="https://iris.example")))
    obj = {"controller_url": "https://iris.example", "config": config, "cluster_name": "marin-test"}
    result = CliRunner().invoke(main_cli.login, [], obj=obj, catch_exceptions=False)

    assert result.exit_code == 0
    record = credential_store.load_credentials("marin-test")
    assert record is not None
    assert record.edge_refresh_token == "refresh-token-xyz"
    assert record.metadata["email"] == "dev@marin.io"
