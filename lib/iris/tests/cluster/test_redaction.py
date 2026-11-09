# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.cluster.redaction.

The shared regex/entropy/key-name behavior is covered in
``lib/rigging/tests/test_redaction.py``. These tests focus on iris-specific
adaptation: argv parsing for ``iris job run -e KEY VALUE`` in every Click
syntax variant, and capability-token scrubbing of free text.
"""

import base64
import json

from iris.cluster.redaction import (
    REDACTED_VALUE,
    count_capability_tokens,
    redact_submit_argv,
    scrub_capability_tokens,
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


def _jwt(claims: dict, alg: str = "EdDSA", signature_bytes: int = 64) -> str:
    def part(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    header = json.dumps({"alg": alg, "typ": "JWT"}).encode()
    return f"{part(header)}.{part(json.dumps(claims).encode())}.{part(bytes(range(signature_bytes)))}"


_CAPABILITY = _jwt({"aud": "iris-proxy", "sub": "endpoint:/job/serve", "exp": 1})
_OTHER_JWT = _jwt({"aud": "oauthlib-tests", "sub": "user"}, alg="HS256", signature_bytes=32)


def _wrapped(token: str, separator: str, width: int = 40) -> str:
    return separator.join(token[i : i + width] for i in range(0, len(token), width))


def test_scrub_capability_tokens_handles_every_break_shape_and_keeps_other_jwts():
    shapes = [
        _CAPABILITY,
        _wrapped(_CAPABILITY, "\n"),
        _wrapped(_CAPABILITY, "\\n"),
        _wrapped(_CAPABILITY, "\\\\n"),  # a log line in a tool output, JSON inside JSON
        _wrapped(_CAPABILITY, "\\\\\\\\n  "),
        "e\\\\ny" + _CAPABILITY[2:],  # a break inside the eyJ prefix
    ]
    for shape in shapes:
        text = f"model.api_base=https://iris.example/proxy/t/{shape}/serve/v1' and {_OTHER_JWT}"
        assert count_capability_tokens(text) == 1, shape
        scrubbed, replaced = scrub_capability_tokens(text)
        assert replaced == 1 and count_capability_tokens(scrubbed) == 0, shape
        assert scrubbed == f"model.api_base=https://iris.example/proxy/t/{REDACTED_VALUE}/serve/v1' and {_OTHER_JWT}"


def test_scrub_capability_tokens_keeps_the_text_after_a_token():
    second = _jwt({"aud": "iris-proxy", "sub": "endpoint:/job/other", "exp": 1})
    cases = [
        (f"API_BASE={_CAPABILITY}\nHOME=/root\n", f"API_BASE={REDACTED_VALUE}\nHOME=/root\n"),
        (f"url {_CAPABILITY} then more words", f"url {REDACTED_VALUE} then more words"),
        (f"{_CAPABILITY}\n{second}\n", f"{REDACTED_VALUE}\n{REDACTED_VALUE}\n"),
        (f"{_OTHER_JWT}\n{_CAPABILITY}\n", f"{_OTHER_JWT}\n{REDACTED_VALUE}\n"),
    ]
    for text, expected in cases:
        scrubbed, replaced = scrub_capability_tokens(text)
        assert scrubbed == expected
        assert replaced == count_capability_tokens(text) == expected.count(REDACTED_VALUE)
        assert count_capability_tokens(scrubbed) == 0


def test_count_capability_tokens_accepts_a_list_audience():
    assert count_capability_tokens(_jwt({"aud": ["other", "iris-proxy"]})) == 1
    assert count_capability_tokens(_OTHER_JWT) == 0


def test_scrub_capability_tokens_after_an_unrelated_shorter_jwt():
    for separator in ("", "\n", "\\n", "\\\\n  "):
        capability = _wrapped(_CAPABILITY, separator)
        for before, after in ((_OTHER_JWT + " and ", " end"), ("start ", " and " + _OTHER_JWT)):
            text = before + capability + after
            assert count_capability_tokens(text) == 1
            scrubbed, replaced = scrub_capability_tokens(text)
            assert (scrubbed, replaced) == (before + REDACTED_VALUE + after, 1)
            assert count_capability_tokens(scrubbed) == 0
