# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import stat
import webbrowser

import pytest
from rigging import iap_login_cli
from rigging.iap_login import (
    IapCredentials,
    credentials_path,
    desktop_login,
    load_iap_credentials,
    provider_for,
    save_iap_credentials,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_save_then_load_round_trips(home):
    creds = IapCredentials(client_id="cid", client_secret="secret", refresh_token="rtok")
    path = save_iap_credentials("marin", creds)

    assert path == credentials_path("marin")
    assert load_iap_credentials("marin") == creds


def test_save_is_owner_only(home):
    path = save_iap_credentials("marin", IapCredentials("cid", "secret", "rtok"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_load_missing_returns_none(home):
    assert load_iap_credentials("never-logged-in") is None


def test_provider_for_missing_raises(home):
    with pytest.raises(FileNotFoundError, match="marin-login never-logged-in"):
        provider_for("never-logged-in")


def test_provider_for_threads_cached_credentials(home, monkeypatch):
    save_iap_credentials("marin", IapCredentials("cid", "secret", "rtok"))

    captured = {}
    monkeypatch.setattr(
        "rigging.iap_login.IapRefreshTokenProvider",
        lambda client_id, client_secret, refresh_token: captured.update(
            client_id=client_id, client_secret=client_secret, refresh_token=refresh_token
        ),
    )

    provider_for("marin")

    assert captured == {"client_id": "cid", "client_secret": "secret", "refresh_token": "rtok"}


def test_desktop_login_caches_refresh_token(home, tmp_path, monkeypatch):
    secret_file = tmp_path / "client_secret.json"
    secret_file.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "secret"}}))

    captured = {}

    def fake_login(client_id, client_secret, *, headless):
        captured["args"] = (client_id, client_secret, headless)
        return "id-token", "rtok"

    monkeypatch.setattr("rigging.iap_login.run_iap_desktop_login", fake_login)

    creds = desktop_login("marin", str(secret_file), headless=True)

    assert captured["args"] == ("cid", "secret", True)
    assert creds == IapCredentials("cid", "secret", "rtok")
    # The credentials landed in the cache for later silent re-minting.
    assert load_iap_credentials("marin") == creds


def test_desktop_login_rejects_web_client_secret(home, tmp_path):
    secret_file = tmp_path / "web_secret.json"
    secret_file.write_text(json.dumps({"web": {"client_id": "cid", "client_secret": "secret"}}))

    with pytest.raises(ValueError, match="desktop"):
        desktop_login("marin", str(secret_file))


def test_cli_threads_args_and_detects_headless(home, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "rigging.iap_login_cli.desktop_login",
        lambda name, client_secrets, *, headless: captured.update(
            name=name, client_secrets=client_secrets, headless=headless
        ),
    )
    # No browser on the box -> the CLI should pick the headless paste flow.
    monkeypatch.setattr("rigging.iap_login_cli.webbrowser.get", lambda *a: (_ for _ in ()).throw(webbrowser.Error()))

    rc = iap_login_cli.main(["marin", "--client-secrets", "/tmp/desktop.json"])

    assert rc == 0
    assert captured == {"name": "marin", "client_secrets": "/tmp/desktop.json", "headless": True}
