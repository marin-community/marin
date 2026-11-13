# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.redaction.

The shared regex/entropy/key-name behavior is covered in
``lib/rigging/tests/test_redaction.py``. These tests focus on iris-specific
adaptation: argv parsing for ``iris job run -e KEY VALUE`` in every Click
syntax variant.
"""

import pytest
from iris.cluster.redaction import REDACTED_VALUE, redact_submit_argv, strip_url_userinfo


def test_redact_submit_argv_redacts_short_flag():
    argv = ["iris", "job", "run", "-e", "HF_TOKEN", "hf_xyz", "--", "python", "t.py"]
    assert redact_submit_argv(argv) == [
        "iris",
        "job",
        "run",
        "-e",
        "HF_TOKEN",
        REDACTED_VALUE,
        "--",
        "python",
        "t.py",
    ]


def test_redact_submit_argv_redacts_long_flag():
    argv = ["iris", "--env-vars", "API_KEY", "sk-xyz"]
    assert redact_submit_argv(argv)[-1] == REDACTED_VALUE


def test_redact_submit_argv_leaves_benign_env_alone():
    argv = ["iris", "job", "run", "-e", "LOG_LEVEL", "info", "--", "python", "t.py"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_redacts_secret_like_value_under_benign_key():
    argv = [
        "iris",
        "job",
        "run",
        "-e",
        "CACHE_BUSTER",
        "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
        "--",
        "python",
        "t.py",
    ]

    out = redact_submit_argv(argv)

    assert out[5] == REDACTED_VALUE


def test_redact_submit_argv_passthrough_without_env_flag():
    argv = ["iris", "job", "run", "--", "python", "-m", "foo"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_handles_multiple_env_pairs():
    argv = [
        "iris",
        "job",
        "run",
        "-e",
        "LOG_LEVEL",
        "info",
        "-e",
        "HF_TOKEN",
        "hf_xyz",
        "-e",
        "NUM_WORKERS",
        "4",
        "--",
        "python",
        "t.py",
    ]
    out = redact_submit_argv(argv)
    assert out[5] == "info"
    assert out[8] == REDACTED_VALUE
    assert out[11] == "4"


def test_redact_submit_argv_does_not_mutate_input():
    argv = ["iris", "-e", "HF_TOKEN", "hf_xyz"]
    original = list(argv)
    redact_submit_argv(argv)
    assert argv == original


def test_redact_submit_argv_trailing_env_flag_without_value_is_passthrough():
    # Malformed input (missing value) should not crash; we bail out cleanly.
    argv = ["iris", "-e", "HF_TOKEN"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_attached_long_form():
    # Click accepts --env-vars=KEY with VALUE as the next token.
    argv = ["iris", "job", "run", "--env-vars=HF_TOKEN", "hf_xyz", "--", "python", "t.py"]
    out = redact_submit_argv(argv)
    assert out[3] == "--env-vars=HF_TOKEN"
    assert out[4] == REDACTED_VALUE


def test_redact_submit_argv_attached_long_form_benign_key():
    argv = ["iris", "--env-vars=LOG_LEVEL", "info", "--"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_attached_short_form():
    # Click accepts -eKEY with VALUE as the next token.
    argv = ["iris", "job", "run", "-eAPI_KEY", "sk-xyz", "--", "python", "t.py"]
    out = redact_submit_argv(argv)
    assert out[3] == "-eAPI_KEY"
    assert out[4] == REDACTED_VALUE


def test_redact_submit_argv_attached_short_form_benign_key():
    argv = ["iris", "-eLOG_LEVEL", "info", "--"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_mixed_forms():
    argv = [
        "iris",
        "job",
        "run",
        "-e",
        "LOG_LEVEL",
        "info",
        "--env-vars=HF_TOKEN",
        "hf_xyz",
        "-eAPI_KEY",
        "sk-abc",
        "--",
        "python",
        "t.py",
    ]
    out = redact_submit_argv(argv)
    assert out[5] == "info"
    assert out[7] == REDACTED_VALUE
    assert out[9] == REDACTED_VALUE


def test_redact_submit_argv_attached_long_form_missing_value():
    argv = ["iris", "--env-vars=HF_TOKEN"]
    assert redact_submit_argv(argv) == argv


def test_redact_submit_argv_masks_the_value_of_a_flag_named_like_a_secret():
    argv = ["iris", "job", "run", "--", "python", "t.py", "--api-key", "made-up-key", "--password=made-up-pw"]
    assert redact_submit_argv(argv) == [*argv[:7], REDACTED_VALUE, f"--password={REDACTED_VALUE}"]


def test_redact_submit_argv_keeps_flags_not_named_like_a_secret():
    argv = ["iris", "job", "run", "--", "python", "t.py", "--max-tokens", "512", "--lr=0.1", "--auth", "--verbose"]
    assert redact_submit_argv(argv) == argv


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://made-up-user:made-up-pw@github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("https://made-up-token@example.com:8443/repo", "https://example.com:8443/repo"),
        ("https://github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
        ("https://made-up-user:made-up-pw@example.com:abc/repo", "https://example.com:abc/repo"),
        ("https://made-up-user:made-up-pw@[::1]:8080/repo", "https://[::1]:8080/repo"),
        ("https://made-up-user:made-up-pw@[::1/repo", "https://[::1/repo"),
        ("https://made-up-host/repo?ref=made-up@tag", "https://made-up-host/repo?ref=made-up@tag"),
    ],
    ids=[
        "user-and-password",
        "token-and-port",
        "no-userinfo",
        "scp-style",
        "bad-port",
        "ipv6",
        "unclosed-ipv6",
        "at-sign-after-host",
    ],
)
def test_strip_url_userinfo(url, expected):
    assert strip_url_userinfo(url) == expected
