# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from rigging.redaction import (
    REDACTED_VALUE,
    is_safe_key_name,
    is_sensitive_key_name,
    looks_like_key,
    redact_json_text,
    redact_string,
    redact_value,
)


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


def test_is_safe_key_name_matches_identifier_suffixes():
    # Generic ``_id``/``_name`` style identifiers are recognized.
    for name in [
        "task_id",
        "job_id",
        "attempt_id",
        "worker_id",
        "vm_id",
        "slice_id",
        "container_id",
        "task_ids",
        "tpu_name",
        "gce_instance_name",
        "task_index",
    ]:
        assert is_safe_key_name(name), name


def test_is_safe_key_name_matches_bare_identifier_fields():
    for name in ["name", "namespace", "hostname", "zone", "region", "status", "state", "url", "shard"]:
        assert is_safe_key_name(name), name


def test_is_safe_key_name_rejects_freeform_keys():
    # Keys without an identifier shape are not safe — random payloads under
    # them should still get the entropy heuristic.
    for name in ["payload", "data", "value", "blob", "config", "args"]:
        assert not is_safe_key_name(name), name


def test_sensitive_key_takes_precedence_over_safe_key():
    # ``auth_id`` matches both regexes; sensitive must win.
    assert is_sensitive_key_name("auth_id")
    assert is_safe_key_name("auth_id")
    assert redact_value({"auth_id": "anything"}) == {"auth_id": REDACTED_VALUE}


def test_redact_value_preserves_high_entropy_task_id():
    # Regression: iris task_id wire format like ``/yonromai/job-name/0`` was
    # being redacted by the entropy heuristic. Identifier-shaped keys should
    # pass through unchanged.
    value = {
        "task_id": "/yonromai/awesome-pipeline-name/some-step-name/0",
        "status_text_summary_md": "**stage0-Map → Write**\nshard 966/3575",
    }
    assert redact_value(value) == value


def test_redact_value_preserves_uuid_under_safe_key():
    # UUID hex (32 chars) has entropy ~4.0 and was being redacted.
    value = {"job_id": "deadbeefcafef00d1234567890abcdef"}
    assert redact_value(value) == value


def test_redact_value_preserves_safe_key_lists():
    # ``task_ids: [...]`` (plural) should also pass identifier values through.
    value = {"task_ids": ["/alice/job/0", "/alice/job/1"]}
    assert redact_value(value) == value


def test_safe_key_still_strips_prefixed_secrets():
    # Even under a "safe" key, prefix-based detection must still catch a real
    # API token someone accidentally stashed in an identifier field.
    value = {"task_id": "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"}
    assert redact_value(value) == {"task_id": REDACTED_VALUE}


def test_safe_key_strips_embedded_prefixed_secrets():
    # A prefixed token embedded in a longer identifier-shaped string still
    # gets stripped, but the surrounding text is preserved.
    raw = "task=/alice/job/0 token=ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
    redacted = redact_value({"reason": raw})
    assert isinstance(redacted, dict)
    assert "/alice/job/0" in redacted["reason"]
    assert "ghp_" not in redacted["reason"]
    assert REDACTED_VALUE in redacted["reason"]


def test_redact_json_text_preserves_task_id_in_status_payload():
    # End-to-end repro of the bug from the issue: SetTaskStatusTextRequest
    # rendered to JSON should keep its task_id intact.
    raw = json.dumps(
        {
            "task_id": "/yonromai/awesome-pipeline-name/0",
            "status_text_detail_md": "**Stage**: stage0-Map → Write\n**Shard**: 966/3575",
            "status_text_summary_md": "**stage0-Map → Write**\nshard 966/3575",
        }
    )
    out = json.loads(redact_json_text(raw))
    assert out["task_id"] == "/yonromai/awesome-pipeline-name/0"
    assert "Map" in out["status_text_detail_md"]
