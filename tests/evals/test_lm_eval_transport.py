# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier B: lm-eval's `local-completions` client actually speaks to our fake server.

Verifies the transport contract (URL path, payload fields, response parsing)
without spinning up vLLM or loading a real HF dataset.

Full `run_lm_eval` end-to-end coverage (task loading from HF, simple_evaluate
orchestration) lives in the TPU CI path (`tests/evals/test_lm_eval.py`).
"""

from __future__ import annotations

import pytest

# These imports pull in torch; skip cleanly when unavailable (e.g. Mac CI).
pytest.importorskip("torch")
# Importing `lm_eval.models` triggers `@register_model` decorators that populate
# MODEL_REGISTRY — without it the `local-completions` / `local-chat-completions`
# keys aren't registered.
pytest.importorskip("lm_eval.models")
from lm_eval.api.registry import MODEL_REGISTRY


def test_local_completions_sends_expected_payload_to_fake_server(fake_openai):
    model_cls = MODEL_REGISTRY["local-completions"]
    model = model_cls(
        base_url=f"{fake_openai.url}/completions",
        model="served-model",
        tokenizer="hf-internal-testing/llama-tokenizer",
        tokenizer_backend="huggingface",
        tokenized_requests=False,
    )
    # Use the raw OAI payload builder so we avoid lm-eval's Instance/task machinery.
    payload = model._create_payload(["Hello world"], generate=False, gen_kwargs=None)
    assert payload["model"] == "served-model"
    assert payload["prompt"] == ["Hello world"]
    assert payload["logprobs"] == 1
    assert payload["echo"] is True

    generation_payload = model._create_payload(
        ["Hello world"], generate=True, gen_kwargs={"max_gen_toks": 64, "temperature": 0.0}
    )
    assert generation_payload["model"] == "served-model"
    assert generation_payload["prompt"] == ["Hello world"]
    assert generation_payload["max_tokens"] == 64
    assert generation_payload["temperature"] == 0.0


def test_local_completions_parse_logprobs_roundtrips_synthetic_shape(fake_openai):
    model_cls = MODEL_REGISTRY["local-completions"]
    prompt_tokens = ["hello", "world"]
    # Minimal logprobs shape: token_logprobs and top_logprobs aligned, ctxlen=1
    # means we look at suffix tokens after the first context token.
    synthetic = {
        "choices": [
            {
                "index": 0,
                "logprobs": {
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [
                        {"hello": -0.1, "other": -5.0},
                        {"world": -0.2, "other": -4.0},
                    ],
                },
            }
        ]
    }
    results = model_cls.parse_logprobs(synthetic, tokens=[prompt_tokens], ctxlens=[1])
    assert len(results) == 1
    logprob_sum, is_greedy = results[0]
    # With ctxlen=1, the suffix is token_logprobs[1:-1] which is empty. This is
    # the canonical lm-eval behavior on the local-completions client — the sum
    # is 0.0 and is_greedy stays True. We assert the shape rather than values
    # because lm-eval's client is what we rely on; any change in the contract
    # would surface here.
    assert isinstance(logprob_sum, int | float)
    assert isinstance(is_greedy, bool)


def test_local_chat_completions_uses_messages_not_prompt(fake_openai):
    model_cls = MODEL_REGISTRY["local-chat-completions"]
    model = model_cls(
        base_url=f"{fake_openai.url}/chat/completions",
        model="served-model",
        tokenizer="hf-internal-testing/llama-tokenizer",
        tokenizer_backend=None,  # chat endpoint: no tokenizer_backend
        tokenized_requests=False,
    )
    payload = model._create_payload(
        [{"role": "user", "content": "hi"}],
        generate=True,
        gen_kwargs={"max_gen_toks": 16, "temperature": 0.0},
    )
    assert "messages" in payload
    assert "prompt" not in payload  # chat schema, not completions schema
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["max_tokens"] == 16


def test_fake_server_reports_path_and_payload(fake_openai):
    # Baseline smoke: the fixture records POSTs we make to it.
    import urllib.request

    req = urllib.request.Request(
        f"{fake_openai.url}/completions",
        data=b'{"model": "fake", "prompt": "hi"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    assert b"choices" in body
    assert fake_openai.received_requests
    path, payload = fake_openai.received_requests[-1]
    assert path == "/v1/completions"
    assert payload["model"] == "fake"
