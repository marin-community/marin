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
    names_only_submit_argv,
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


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["python", "train.py", "--hf-token", "made-up-1", "--lr", "3e-4"], ["python", R, "--hf-token", R, "--lr", R]),
        (
            ["mysql", "-pmade-up-2", "--api-key=made-up-3", "-eSERVICE_KEY", "made-up-4"],
            ["mysql", "-p", "--api-key", "-e", R],
        ),
        (["bash", "-c", "export HF_TOKEN=made-up-5 && python x.py"], ["bash", "-c", R]),
    ],
    ids=["flag-names", "attached-values", "shell-body"],
)
def test_names_only_argv_keeps_command_and_flag_names_only(argv, expected):
    assert names_only_argv(argv) == expected


def test_names_only_argv_on_an_iris_submit_line():
    argv = ["iris", "job", "run", "-e", "SERVICE_URL", "made-up-6", "--secret", "SERVICE_KEY", "made-up-7"]
    argv += ["--", "python", "t.py"]

    assert names_only_submit_argv(argv) == ["iris", R, R, "-e", R, R, "--secret", R, R, "--", "python", R]


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
        ["login", "--", "{s}"],
        ["-p", "{s}"],
    ],
)
def test_no_value_survives_names_only_redaction(shape, secret):
    argv = [part.format(s=secret) for part in shape]

    for redacted in (names_only_argv(argv), names_only_script(" ".join(f"'{part}'" for part in argv)).split(" ")):
        assert not any(secret in token for token in redacted)


def test_names_only_script_redacts_heredoc_bodies():
    script = (
        "cat > ~/.cache/hf_token <<EOF\nmade-up-heredoc-secret\nEOF\n"
        "cat <<-'PEM' > key\n\tmade-up-pem-line\n\tPEM\nuv sync"
    )

    stored = names_only_script(script)

    assert "made-up" not in stored
    assert stored.splitlines()[1] == R
    assert stored.splitlines()[-1] == f"uv {R}"


def test_names_only_script_joins_continued_lines():
    script = (
        "docker login \\\n  -u made-up-user \\\n  -p made-up-continued-secret\n"
        "export HF_TOKEN=\\\n  made-up-split-secret"
    )

    assert names_only_script(script) == f"docker {R} -u {R} -p {R}\nexport HF_TOKEN= {R}"


def test_names_only_flag_first_and_double_dash_end_the_command_word():
    assert names_only_script("-p made-up-flag-first") == f"-p {R}"
    assert names_only_script("huggingface-cli login -- made-up-after-dashes") == f"huggingface-cli {R} -- {R}"
    assert names_only_argv(["python", "t.py", "--", "made-up-after-dashes"]) == ["python", R, "--", R]


def test_names_only_keeps_a_command_word_only_if_it_looks_like_a_program():
    assert names_only_argv(["https://user:made-up-pw@x.example/run", "a"]) == [R, R]
    assert names_only_argv(["./run.sh", "a"]) == ["./run.sh", R]


_SCRIPT_SHAPES = [
    "cat > f <<EOF\n{s}\nEOF",
    "cat <<-'END'\n\t{s}\n\tEND",
    "docker login \\\n  -u user \\\n  -p {s}",
    "export HF_TOKEN=\\\n  {s}",
    "-p {s}",
    "huggingface-cli login -- {s}",
    "pip install --index-url {s} x",
    "docker login \\\r\n  -p {s}\r\n",
    'python -c "import os\n{s}\nprint(1)"',
]


@pytest.mark.parametrize("secret", _MADE_UP_SECRETS)
@pytest.mark.parametrize("shape", _SCRIPT_SHAPES)
def test_no_value_survives_names_only_script_shapes(shape, secret):
    assert secret not in names_only_script(shape.format(s=secret))


def test_names_only_script_joins_crlf_continuations():
    assert names_only_script("docker login \\\r\n  -p made-up-crlf-secret\r\nuv sync") == f"docker {R} -p {R}\nuv {R}"


def test_names_only_script_reads_a_multiline_quote_as_one_statement():
    script = 'python -c "import os\nmade-up-midline-secret\nprint(1)" && uv sync\necho "never closed\nmade-up-tail'

    assert names_only_script(script) == f"python -c {R} && uv {R}\n{NAMES_ONLY_UNPARSED}"
