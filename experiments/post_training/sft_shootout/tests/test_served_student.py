# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import httpx

from experiments.post_training.sft_shootout import harbor_glm
from experiments.post_training.sft_shootout.relay import Author


def test_the_student_backend_waits_for_v0_and_uses_its_served_model_id(monkeypatch):
    url = "https://iris.example/proxy/t/cluster=c/token/serve.v0/v1"
    replies = [
        httpx.ConnectError("refused"),
        httpx.Response(502),
        httpx.Response(200, json={"data": [{"id": "org/v0"}]}),
    ]
    requested = []

    def get(target, timeout):
        requested.append(target)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setenv(harbor_glm.STUDENT_URL_ENV, url + "/")
    monkeypatch.setattr(harbor_glm.httpx, "get", get)
    monkeypatch.setattr(harbor_glm, "STUDENT_POLL_S", 0)
    monkeypatch.setattr(harbor_glm, "load_model_config", lambda path: SimpleNamespace(tokenizer="org/tok"))
    monkeypatch.setattr(harbor_glm, "load_tokenizer", lambda name, revision: (name, revision))
    student = harbor_glm.served_student()
    assert requested == [f"{url}/models"] * 3
    assert (student.author, student.base_url, student.model) == (Author.STUDENT, url, "org/v0")
    assert student.resolve is not None and student.resolve() == url
