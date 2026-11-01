# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Installed agents must use a host model proxy for cloud-provider authentication."""

import pytest

from harbor.agents.installed.claude_code import ClaudeCode


@pytest.mark.parametrize(
    "selector", ["CLAUDE_CODE_USE_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK"]
)
def test_installed_claude_refuses_cloud_credentials_before_agent_setup(
    tmp_path, monkeypatch, selector
):
    monkeypatch.setenv(selector, "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "storage-private-canary")
    with pytest.raises(ValueError, match="host-side model proxy"):
        ClaudeCode(logs_dir=tmp_path, model_name="bedrock/model")
