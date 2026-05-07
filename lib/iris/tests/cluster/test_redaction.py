# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.redaction."""

import json

from iris.cluster.redaction import (
    REDACTED_VALUE,
    is_sensitive_env_key,
    redact_json_preview,
    redact_submit_argv,
)


def test_redact_json_preview_redacts_nested_sensitive_keys():
    raw = json.dumps(
        {
            "name": "train-job",
            "environment": {"env_vars": {"HF_TOKEN": "hf_xyz", "LOG_LEVEL": "info"}},
            "metadata": [{"api_key": "sk-abc"}, {"benign": "ok"}],
        }
    )
    out = json.loads(redact_json_preview(raw))
    assert out["environment"]["env_vars"]["HF_TOKEN"] == REDACTED_VALUE
    assert out["environment"]["env_vars"]["LOG_LEVEL"] == "info"
    assert out["metadata"][0]["api_key"] == REDACTED_VALUE
    assert out["metadata"][1]["benign"] == "ok"
    assert out["name"] == "train-job"


def test_redact_json_preview_passes_through_invalid_json():
    assert redact_json_preview("not json{{{") == "not json{{{"
    assert redact_json_preview("") == ""


def test_is_sensitive_env_key_matches_common_secret_names():
    assert is_sensitive_env_key("WANDB_API_KEY")
    assert is_sensitive_env_key("HF_TOKEN")
    assert is_sensitive_env_key("MY_SECRET")
    assert is_sensitive_env_key("DB_PASSWORD")
    assert is_sensitive_env_key("GCP_CREDENTIAL_PATH")
    assert is_sensitive_env_key("api_key")  # case-insensitive


def test_is_sensitive_env_key_leaves_benign_names():
    assert not is_sensitive_env_key("LOG_LEVEL")
    assert not is_sensitive_env_key("NUM_WORKERS")
    assert not is_sensitive_env_key("HOSTNAME")


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
