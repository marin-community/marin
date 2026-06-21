# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import stat

import pytest
from rigging.auth import IapRefreshTokenProvider
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
    with pytest.raises(FileNotFoundError, match="run the desktop login"):
        provider_for("never-logged-in")


def test_provider_for_builds_from_cache(home):
    save_iap_credentials("marin", IapCredentials("cid", "secret", "rtok"))
    assert isinstance(provider_for("marin"), IapRefreshTokenProvider)


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
