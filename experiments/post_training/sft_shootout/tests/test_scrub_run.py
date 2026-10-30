# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

from experiments.post_training.sft_shootout.scrub_run import REPLACEMENT, count_tokens, scrub_text


def jwt(claims: dict) -> str:
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part(claims)}.c2lnbmF0dXJlLXNpZ25hdHVyZQ"


TOKEN = jwt({"aud": "iris-proxy", "sub": "endpoint:/glm-harbor/x", "exp": 1})
FIXTURE = jwt({"aud": "oauthlib-tests", "sub": "user"})


def wrapped(token: str, separator: str, width: int = 40) -> str:
    return separator.join(token[i : i + width] for i in range(0, len(token), width))


def url(token: str) -> str:
    return f"https://iris.oa.dev/proxy/t/cluster=cw-us-east-02a/{token}/glm-harbor.s-1cbfc798/v1"


def test_scrubs_every_break_shape_and_keeps_fixtures():
    shapes = [
        TOKEN,
        wrapped(TOKEN, "\n"),
        wrapped(TOKEN, "\\n"),
        wrapped(TOKEN, "\\\\n"),  # a log line in a tool output, JSON inside JSON
        wrapped(TOKEN, "\\\\\\\\n  "),
        "e\\\\ny" + TOKEN[2:],  # a break inside the eyJ prefix
    ]
    for shape in shapes:
        text = f"model.api_base={url(shape)}' and {FIXTURE}"
        assert count_tokens(text) == 1, shape
        scrubbed, replaced = scrub_text(text)
        assert replaced == 1 and count_tokens(scrubbed) == 0, shape
        assert REPLACEMENT in scrubbed and FIXTURE in scrubbed


def test_list_audience_counts():
    assert count_tokens(jwt({"aud": ["other", "iris-proxy"]})) == 1
    assert count_tokens(FIXTURE) == 0
