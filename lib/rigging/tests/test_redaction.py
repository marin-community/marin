# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from rigging.redaction import REDACTED_VALUE, looks_like_key, redact_json_text, redact_string, redact_value


def test_redact_value_redacts_sensitive_key_names():
    value = {
        "api_key": "short-low-entropy-value",
        "nested": {
            "access-token": "another-low-entropy-value",
            "log_level": "debug",
        },
        "items": [{"password": "password-value"}, {"name": "worker"}],
    }

    assert redact_value(value) == {
        "api_key": REDACTED_VALUE,
        "nested": {
            "access-token": REDACTED_VALUE,
            "log_level": "debug",
        },
        "items": [{"password": REDACTED_VALUE}, {"name": "worker"}],
    }


def test_redact_string_redacts_prefixed_tokens():
    slack_token = "xox" + "b-1234567890-abcdefghijklmnopqrstuvwxyz"
    raw = (
        "aws=AKIAIOSFODNN7EXAMPLE "
        "github=ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ "
        f"slack={slack_token} "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    redacted = redact_string(raw)

    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ" not in redacted
    assert slack_token not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert redacted.count(REDACTED_VALUE) == 4


def test_redact_string_redacts_private_keys():
    raw = """\
before
-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAxV2cX8jVh2cP3m8n5q4r3s2t1u0v9w8x7y6z5a4b3c2d1e0f
-----END RSA PRIVATE KEY-----
after
"""

    assert redact_string(raw) == f"before\n{REDACTED_VALUE}\nafter\n"


def test_redact_string_redacts_high_entropy_key_like_runs():
    secret = "AbCDef1234567890+/MnOpQrStUvWxYz0987654321"
    redacted = redact_string(f"token={secret}")

    assert secret not in redacted
    assert redacted == f"token={REDACTED_VALUE}"


def test_redact_string_preserves_low_entropy_long_strings():
    value = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    assert not looks_like_key(value)
    assert redact_string(value) == value


def test_redact_value_redacts_secret_like_strings_under_benign_keys():
    value = {"cache_buster": "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ", "log_level": "info"}

    assert redact_value(value) == {"cache_buster": REDACTED_VALUE, "log_level": "info"}


def test_redact_json_text_redacts_nested_sensitive_keys():
    raw = json.dumps(
        {
            "name": "train-job",
            "environment": {"env_vars": {"HF_TOKEN": "hf_xyz", "LOG_LEVEL": "info"}},
            "metadata": [{"api_key": "sk-abc"}, {"benign": "ok"}],
        }
    )

    out = json.loads(redact_json_text(raw))

    assert out["environment"]["env_vars"]["HF_TOKEN"] == REDACTED_VALUE
    assert out["environment"]["env_vars"]["LOG_LEVEL"] == "info"
    assert out["metadata"][0]["api_key"] == REDACTED_VALUE
    assert out["metadata"][1]["benign"] == "ok"
    assert out["name"] == "train-job"


def test_redact_json_text_falls_back_to_string_redaction_on_invalid_json():
    # Not valid JSON, but contains a prefixed token — caller still gets protection.
    leaky = "not json{{{ ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
    assert redact_json_text(leaky) == "not json{{{ " + REDACTED_VALUE
    assert redact_json_text("plain not json") == "plain not json"
    assert redact_json_text("") == ""
