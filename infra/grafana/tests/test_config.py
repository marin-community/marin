# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from config import BridgeConfig

LOOM_ENV = (
    "LOOM_ALERT_URL",
    "LOOM_ALERT_PROFILE",
    "LOOM_ALERT_REPOSITORY",
)


def test_loom_alert_configuration_is_explicit(monkeypatch):
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com/")
    monkeypatch.setenv("LOOM_ALERT_PROFILE", "grafana_alert")
    monkeypatch.setenv("LOOM_ALERT_REPOSITORY", "marin-community/marin")

    config = BridgeConfig.from_environment()

    assert config.loom_alerts is not None
    assert config.loom_alerts.url == "https://loom.example.com"
    assert config.loom_alerts.profile == "grafana_alert"
    assert config.loom_alerts.repository == "marin-community/marin"


def test_partial_loom_alert_configuration_fails_fast(monkeypatch):
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com")

    with pytest.raises(ValueError, match="LOOM_ALERT_PROFILE"):
        BridgeConfig.from_environment()
