# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts.ops.discord — webhook URL resolution and post payload."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from scripts.ops import discord


@pytest.fixture
def clear_env(monkeypatch):
    for ch in discord.CHANNELS:
        monkeypatch.delenv(discord._env_var_for(ch), raising=False)


def test_env_var_takes_precedence_over_gcloud(monkeypatch, clear_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_CODE_REVIEW", "https://discord.test/env-hook")
    with mock.patch.object(discord.subprocess, "check_output") as gcloud:
        url = discord._webhook_url("code-review")
    assert url == "https://discord.test/env-hook"
    gcloud.assert_not_called()


def test_falls_back_to_gcloud_secret(clear_env):
    with mock.patch.object(
        discord.subprocess, "check_output", return_value="https://discord.test/gcloud-hook\n"
    ) as gcloud:
        url = discord._webhook_url("internal-discuss")
    assert url == "https://discord.test/gcloud-hook"
    args, _ = gcloud.call_args
    assert args[0] == [
        "gcloud",
        "secrets",
        "versions",
        "access",
        "latest",
        "--secret=marin-discord-webhook-internal-discuss",
    ]


def test_unknown_channel_rejected():
    with pytest.raises(AssertionError):
        discord._webhook_url("nope")


def test_post_sends_json_payload(monkeypatch, clear_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_CODE_REVIEW", "https://discord.test/hook")
    fake_resp = mock.MagicMock()
    fake_resp.status = 204
    fake_resp.__enter__.return_value = fake_resp
    with mock.patch.object(discord.urllib.request, "urlopen", return_value=fake_resp) as urlopen:
        discord.post("code-review", "hello world", username="marin-bot")
    req = urlopen.call_args[0][0]
    assert req.full_url == "https://discord.test/hook"
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"content": "hello world", "username": "marin-bot"}


def test_post_raises_on_http_error(monkeypatch, clear_env):
    monkeypatch.setenv("DISCORD_WEBHOOK_CODE_REVIEW", "https://discord.test/hook")
    fake_resp = mock.MagicMock()
    fake_resp.status = 500
    fake_resp.read.return_value = b"server boom"
    fake_resp.__enter__.return_value = fake_resp
    with mock.patch.object(discord.urllib.request, "urlopen", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="500"):
            discord.post("code-review", "hi")


def test_post_rejects_empty_message(clear_env):
    with pytest.raises(ValueError):
        discord.post("code-review", "   ")
