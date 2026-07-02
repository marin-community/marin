# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from ducky.config import DuckyConfig

_BASE_ENV = {"DUCKY_SCRATCH_BUCKET": "/tmp/ducky"}
_GCS_ENV = {"DUCKY_GCS_HMAC_KEY_ID": "k", "DUCKY_GCS_HMAC_SECRET": "s"}


@pytest.fixture(autouse=True)
def _clear_ducky_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DUCKY_"):
            monkeypatch.delenv(key, raising=False)


def _set(monkeypatch, env: dict[str, str]) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_requires_scratch_bucket(monkeypatch):
    with pytest.raises(ValueError, match="DUCKY_SCRATCH_BUCKET"):
        DuckyConfig.from_environment()


def test_no_backend_creds_is_allowed(monkeypatch):
    _set(monkeypatch, _BASE_ENV)
    config = DuckyConfig.from_environment()
    assert config.scratch_bucket == "/tmp/ducky"
    assert not config.gcs_enabled
    assert not config.r2_enabled
    assert not config.cw_enabled


def test_full_backend_enables_it(monkeypatch):
    _set(monkeypatch, {**_BASE_ENV, **_GCS_ENV})
    config = DuckyConfig.from_environment()
    assert config.gcs_enabled
    assert not config.r2_enabled


def test_partial_backend_creds_raise(monkeypatch):
    _set(monkeypatch, {**_BASE_ENV, "DUCKY_GCS_HMAC_KEY_ID": "k"})  # secret missing
    with pytest.raises(ValueError, match="partially configured"):
        DuckyConfig.from_environment()
