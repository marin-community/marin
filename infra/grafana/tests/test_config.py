# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from config import (
    LOCAL_FINELOG_MAX_QUERIES_ENV,
    LOCAL_FINELOG_URL_ENV,
    BridgeConfig,
    LocalFinelog,
    local_finelog_from_environment,
)

LOOM_ENV = (
    "LOOM_ALERT_URL",
    "LOOM_ALERT_PROFILE",
    "LOOM_ALERT_REPOSITORY",
)


def test_loom_alert_configuration_is_explicit(monkeypatch):
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com/")
    monkeypatch.setenv("LOOM_ALERT_PROFILE", "ops")
    monkeypatch.setenv("LOOM_ALERT_REPOSITORY", "marin-community/marin")
    monkeypatch.setenv("SLACK_ALERTS_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_ALERTS_CHANNEL", "C0123ABCD")

    config = BridgeConfig.from_environment()

    assert config.loom_alerts is not None
    assert config.loom_alerts.url == "https://loom.example.com"
    assert config.loom_alerts.profile == "ops"
    assert config.loom_alerts.repository == "marin-community/marin"
    assert config.loom_alerts.slack.bot_token == "xoxb-test"
    assert config.loom_alerts.slack.channel == "C0123ABCD"


def test_partial_loom_alert_configuration_fails_fast(monkeypatch):
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com")

    with pytest.raises(ValueError, match="LOOM_ALERT_PROFILE"):
        BridgeConfig.from_environment()


def test_alert_delivery_without_a_slack_destination_fails_fast(monkeypatch):
    """The bridge is the only thing that announces critical alerts now that
    Grafana's Slack receiver is gone, so a missing token must not boot."""
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com")
    monkeypatch.setenv("LOOM_ALERT_PROFILE", "ops")
    monkeypatch.setenv("LOOM_ALERT_REPOSITORY", "marin-community/marin")

    with pytest.raises(ValueError, match="SLACK_ALERTS_BOT_TOKEN"):
        BridgeConfig.from_environment()


def test_a_secret_payload_with_a_trailing_newline_is_usable(monkeypatch):
    """Secret Manager serves whatever bytes it was given, and a payload created
    from a shell pipeline usually ends in a newline. Unstripped, the token reaches
    an Authorization header and the channel reaches a request body."""
    for name in LOOM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOM_ALERT_URL", "https://loom.example.com")
    monkeypatch.setenv("LOOM_ALERT_PROFILE", "ops")
    monkeypatch.setenv("LOOM_ALERT_REPOSITORY", "marin-community/marin")
    monkeypatch.setenv("SLACK_ALERTS_BOT_TOKEN", "xoxb-test\n")
    monkeypatch.setenv("SLACK_ALERTS_CHANNEL", "C0123ABCD\n")

    config = BridgeConfig.from_environment()

    assert config.loom_alerts is not None
    assert config.loom_alerts.slack.bot_token == "xoxb-test"
    assert config.loom_alerts.slack.channel == "C0123ABCD"


LOCAL_FINELOG_ENV = (LOCAL_FINELOG_URL_ENV, LOCAL_FINELOG_MAX_QUERIES_ENV)
LOCAL_URL = "iap+https://iris.example.com/proxy/system.log-server"


def test_the_deployed_bridge_has_no_local_finelog_route(monkeypatch):
    for name in LOCAL_FINELOG_ENV:
        monkeypatch.delenv(name, raising=False)

    assert local_finelog_from_environment() is None


def test_a_local_bridge_states_its_route_and_concurrency_bound(monkeypatch):
    monkeypatch.setenv(LOCAL_FINELOG_URL_ENV, LOCAL_URL)
    monkeypatch.setenv(LOCAL_FINELOG_MAX_QUERIES_ENV, "2")

    assert local_finelog_from_environment() == LocalFinelog(url=LOCAL_URL, max_concurrent_queries=2)


@pytest.mark.parametrize("present", LOCAL_FINELOG_ENV)
def test_a_half_configured_local_route_fails_fast(monkeypatch, present):
    for name in LOCAL_FINELOG_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(present, LOCAL_URL if present == LOCAL_FINELOG_URL_ENV else "1")

    with pytest.raises(ValueError, match="set together"):
        local_finelog_from_environment()


def test_a_local_route_must_go_through_iap(monkeypatch):
    """A plain URL would send queries with no IAP token."""
    monkeypatch.setenv(LOCAL_FINELOG_URL_ENV, "https://iris.example.com/proxy/system.log-server")
    monkeypatch.setenv(LOCAL_FINELOG_MAX_QUERIES_ENV, "1")

    with pytest.raises(ValueError, match="iap"):
        local_finelog_from_environment()
