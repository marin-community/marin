# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.redaction.

The shared regex/entropy/key-name behavior is covered in
``lib/rigging/tests/test_redaction.py``. These tests focus on iris-specific
adaptation: argv parsing for ``iris job run -e KEY VALUE`` in every Click
syntax variant.
"""

import pytest
from iris.cluster.redaction import (
    REDACTED_VALUE,
    redact_command_argv,
    redact_script,
    redact_stored_submit_argv,
    redact_submit_argv,
)


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


_RUN = ["iris", "job", "run"]
_CMD = ["--", "python", "t.py"]


@pytest.mark.parametrize(
    "flags",
    [
        ["-e", "SERVICE_URL", "made-up-value"],
        ["-eSERVICE_URL", "made-up-value"],
        ["--env-vars", "SERVICE_URL", "made-up-value"],
        ["--env-vars=SERVICE_URL", "made-up-value"],
        ["--secret", "SERVICE_KEY", "made-up-value"],
        ["--secret=SERVICE_KEY", "made-up-value"],
    ],
    ids=["e", "e-attached", "env-vars", "env-vars-attached", "secret", "secret-attached"],
)
def test_stored_submit_argv_drops_every_env_and_secret_value(flags):
    redacted = redact_stored_submit_argv([*_RUN, *flags, *_CMD])

    assert "made-up-value" not in redacted
    assert redacted[: len(_RUN)] == _RUN
    assert redacted[len(_RUN)].split("=")[0] == flags[0].split("=")[0]
    assert redacted[-len(_CMD) :] == _CMD


def test_stored_submit_argv_redacts_a_command_given_without_double_dash():
    redacted = redact_stored_submit_argv([*_RUN, "python", "train.py", "--hf-token", "made-up-token"])

    assert redacted == [*_RUN, "python", "train.py", "--hf-token", REDACTED_VALUE]


def test_command_argv_redacts_values_of_sensitive_flags():
    argv = ["python", "t.py", "--hf-token", "made-up-1", "--api-key=made-up-2", "--epochs", "3"]

    assert redact_command_argv(argv) == [
        "python",
        "t.py",
        "--hf-token",
        REDACTED_VALUE,
        f"--api-key={REDACTED_VALUE}",
        "--epochs",
        "3",
    ]


def test_script_redacts_sensitive_assignments_only():
    script = "export WANDB_API_KEY=made-up-key && export LOG_LEVEL=debug && uv sync"

    assert redact_script(script) == f"export WANDB_API_KEY={REDACTED_VALUE} && export LOG_LEVEL=debug && uv sync"


def test_command_argv_redacts_assignments_inside_a_shell_string():
    argv = ["bash", "-c", "export HF_TOKEN=made-up-token && python x.py"]

    assert redact_command_argv(argv) == ["bash", "-c", f"export HF_TOKEN={REDACTED_VALUE} && python x.py"]
