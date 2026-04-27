# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.diagnostic_logs import sanitize_diagnostic_log_text


def test_sanitize_diagnostic_log_text_redacts_secrets_and_identifiers():
    text = (
        "token=supersecretvalue123 ghp_abcdefghijklmnopqrstuvwxyz123456 "
        "email alice@example.com path=/Users/alice/project user Alice failed"
    )
    redacted = sanitize_diagnostic_log_text(text)
    assert "supersecretvalue123" not in redacted
    assert "alice@example.com" not in redacted
    assert "/Users/alice" not in redacted
    assert "user Alice failed" not in redacted
    assert "<REDACTED_SECRET>" in redacted
    assert "<REDACTED_GITHUB_TOKEN>" in redacted
    assert "<USER_0_EMAIL>" in redacted
    assert "/Users/<USER_0>/project" in redacted
    assert "user <USER_0> failed" in redacted
