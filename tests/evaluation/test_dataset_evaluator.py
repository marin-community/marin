# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model admission follows the running server's context limit."""

from types import SimpleNamespace

import httpx
from marin.evaluation.dataset_evaluator import runtime_config
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

    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))

    monkeypatch.setattr("marin.evaluation.dataset_evaluator.OpenAI", client)
    model = ModelConfig(
        name="solver",
        location="example/solver",
        revision="pinned",
        tokenizer_revision="tokenizer-pin",
        serve=ServeConfig(max_model_len=8192),
        generation=GenerationConfig(max_gen_toks=8192),
    )
    settings = CuratedSettings(max_tokens=8192, source_limit=2, seed=42, judge=None)
    session = SimpleNamespace(
        model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://server/v1", api_key=None, model="served-alias"))
    )
    first = runtime_config(model, settings, session)
    assert first.model.context_window == 4096
    assert first.max_tokens == 4095
    created[0] += 100
    assert runtime_config(model, settings, session) == first
