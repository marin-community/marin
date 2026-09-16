# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP failures retain the model endpoint's diagnostic response."""

from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from taskcompendium.harbor.agents import DirectChatAgent


def test_chat_endpoint_error_includes_response_detail(monkeypatch):
    def fail(*_args, **_kwargs):
        raise HTTPError(
            "http://localhost/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":{"message":"model name mismatch"}}'),
        )

    monkeypatch.setattr("taskcompendium.harbor.agents.urllib.request.urlopen", fail)
    agent = SimpleNamespace(
        model_name="policy",
        max_tokens=32,
        temperature=1.0,
        chat_template_kwargs=None,
        api_key="",
        api_base="http://localhost/v1",
        request_timeout=1,
    )
    with pytest.raises(RuntimeError, match=r"Chat completion HTTP 400:.*model name mismatch"):
        DirectChatAgent._completion(agent, [{"role": "user", "content": "Hello"}])
