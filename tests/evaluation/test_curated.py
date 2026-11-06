# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated execution binds model admission to the live serving boundary."""

from functools import partial
from types import SimpleNamespace

import httpx
from marin.evaluation.curated import runtime_config
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ServeConfig
from marin.evaluation.records import CuratedSettings
from openai import OpenAI


def test_context_admission_uses_server_limit_and_resume_identity_ignores_server_timestamp(monkeypatch):
    created = [100]

    def endpoint(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "served-alias",
                        "object": "model",
                        "created": created[0],
                        "owned_by": "vllm",
                        "max_model_len": 4096,
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "marin.evaluation.curated.OpenAI",
        partial(
            OpenAI,
            http_client=httpx.Client(transport=httpx.MockTransport(endpoint)),
        ),
    )
    model = ModelConfig(
        name="solver",
        location="example/solver",
        revision="pinned",
        tokenizer_revision="tokenizer-pin",
        serve=ServeConfig(max_model_len=8192),
        generation=GenerationConfig(max_gen_toks=8192),
    )
    settings = CuratedSettings(max_tokens=2048, source_limit=2, seed=42, judge=None)
    session = SimpleNamespace(
        model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://server/v1", api_key=None, model="served-alias"))
    )
    first = runtime_config(model, settings, session)
    assert first.model.context_window == 4096
    assert first.max_tokens == 2048
    assert first.model.tokenizer_revision == "tokenizer-pin"
    created[0] += 100
    # Each request owns a new client, as the production constructor does.
    monkeypatch.setattr(
        "marin.evaluation.curated.OpenAI",
        partial(
            OpenAI,
            http_client=httpx.Client(transport=httpx.MockTransport(endpoint)),
        ),
    )
    assert runtime_config(model, settings, session) == first
