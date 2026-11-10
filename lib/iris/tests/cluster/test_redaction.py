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
    NAMES_ONLY_REDACTED,
    NAMES_ONLY_UNPARSED,
    REDACTED_VALUE,
    names_only_argv,
    names_only_script,
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


R = NAMES_ONLY_REDACTED


def test_names_only_argv_keeps_the_command_and_flag_names():
    argv = ["python", "train.py", "--hf-token", "made-up-1", "--lr", "3e-4"]

    assert names_only_argv(argv) == ["python", R, "--hf-token", R, "--lr", R]


def test_names_only_argv_cuts_attached_values():
    argv = ["mysql", "-pmade-up-2", "--api-key=made-up-3", "-eSERVICE_KEY", "made-up-4"]

    assert names_only_argv(argv) == ["mysql", "-p", "--api-key", "-e", R]


def test_names_only_argv_redacts_a_shell_body_whole():
    argv = ["bash", "-c", "export HF_TOKEN=made-up-5 && python x.py"]

    assert names_only_argv(argv) == ["bash", "-c", R]


def test_names_only_argv_on_an_iris_submit_line():
    argv = ["iris", "job", "run", "-e", "SERVICE_URL", "made-up-6", "--secret", "SERVICE_KEY", "made-up-7"]
    argv += ["--", "python", "t.py"]

    assert names_only_argv(argv) == ["iris", R, R, "-e", R, R, "--secret", R, R, "--", "python", R]


def test_names_only_script_keeps_command_words_after_separators_and_assignment_names():
    script = "export HF_TOKEN=made-up-8 && pip install --index-url https://u:made-up-9@pypi.example/simple x | tee log"
    script += "\ngit clone https://made-up-10@github.com/o/r.git; huggingface-cli login --token made-up-11"

    assert names_only_script(script) == (
        f"export HF_TOKEN= && pip {R} --index-url {R} {R} | tee {R}\ngit {R} {R} ; huggingface-cli {R} --token {R}"
    )


def test_names_only_script_marks_an_unparsable_line():
    assert names_only_script('uv sync\necho "made-up-12') == f"uv {R}\n{NAMES_ONLY_UNPARSED}"


_MADE_UP_SECRETS = [
    "made-up-hf-token",
    "AIzaSyMADEUPmadeupMADEUPmadeupMADEUP12",
    "ya29.made-up-oauth-token",
    "https://user:made-up-pw@pypi.example.com/simple",
    "https://made-up-gh-token@github.com/org/repo.git",
    "short",
]


@pytest.mark.parametrize("secret", _MADE_UP_SECRETS)
@pytest.mark.parametrize(
    "shape",
    [
        ["python", "t.py", "{s}"],
        ["python", "t.py", "--token", "{s}"],
        ["python", "t.py", "--token={s}"],
        ["tool", "-p{s}"],
        ["env", "HF_TOKEN={s}", "python", "t.py"],
        ["bash", "-c", "login --token {s} && run"],
    ],
)
def test_no_value_survives_names_only_redaction(shape, secret):
    argv = [part.format(s=secret) for part in shape]

    for redacted in (names_only_argv(argv), names_only_script(" ".join(f"'{part}'" for part in argv)).split(" ")):
        assert not any(secret in token for token in redacted)
