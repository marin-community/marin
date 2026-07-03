# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from ducky.config import DEFAULT_DATAKIT_ROOT, DEFAULT_FINELOG_ROOT, DuckyConfig

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


def test_catalog_roots_default_to_real_locations(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    _set(monkeypatch, _BASE_ENV)
    config = DuckyConfig.from_environment()
    assert config.finelog_root == DEFAULT_FINELOG_ROOT
    assert config.datakit_root == DEFAULT_DATAKIT_ROOT  # fallback when MARIN_PREFIX unset


def test_datakit_root_derived_from_marin_prefix(monkeypatch):
    _set(monkeypatch, {**_BASE_ENV, "MARIN_PREFIX": "gs://marin-eu-west4"})
    assert DuckyConfig.from_environment().datakit_root == "gs://marin-eu-west4/normalized"


def test_catalog_root_override_and_disable(monkeypatch):
    # explicit DUCKY_DATAKIT_ROOT wins over MARIN_PREFIX; empty disables the source
    _set(
        monkeypatch,
        {
            **_BASE_ENV,
            "MARIN_PREFIX": "gs://marin-eu-west4",
            "DUCKY_FINELOG_ROOT": "gs://elsewhere/finelog",
            "DUCKY_DATAKIT_ROOT": "",
        },
    )
    config = DuckyConfig.from_environment()
    assert config.finelog_root == "gs://elsewhere/finelog"
    assert config.datakit_root is None


def test_directly_constructed_config_has_no_catalog_roots():
    # Unit tests build DuckyConfig directly; roots must default to None so a runner doesn't
    # reach the network binding views.
    config = DuckyConfig(scratch_bucket="/tmp/ducky")
    assert config.finelog_root is None
    assert config.datakit_root is None


def test_effective_allowlist_includes_configured_roots():
    config = DuckyConfig(
        scratch_bucket="/tmp/ducky",
        allowed_buckets=("gs://marin-us-east5",),
        finelog_root="gs://marin-us-central2/finelog/marin",
        datakit_root="gs://marin-us-east5/normalized",
    )
    eff = config.effective_allowed_buckets
    assert "gs://marin-us-east5" in eff
    assert "gs://marin-us-central2/finelog/marin" in eff  # cross-region finelog root is readable


def test_effective_allowlist_empty_stays_allow_all():
    # an empty allowlist means allow-all; adding roots must not turn it into a restrictive list
    config = DuckyConfig(scratch_bucket="/tmp/ducky", allowed_buckets=(), finelog_root="gs://x/finelog")
    assert config.effective_allowed_buckets == ()
