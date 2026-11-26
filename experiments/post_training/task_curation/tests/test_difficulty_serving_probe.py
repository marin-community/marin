# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic completions stay bounded without changing graded-trial budgets."""

import json
from types import SimpleNamespace

import httpx
import pytest
from finestore.rl.difficulty import SOLVER_MODELS, Solver

from experiments.post_training.task_curation import difficulty_serving_probe as probe
from experiments.post_training.task_curation.difficulty.state import ActionFailure, GapReason
from experiments.post_training.task_curation.difficulty_sampling import sampling_settings


@pytest.mark.parametrize("solver", list(Solver))
@pytest.mark.parametrize("served_prompt_tokens", [3, 4])
def test_solver_probe_caps_decode_and_rejects_tokenization_drift(monkeypatch, solver, served_prompt_tokens):
    settings = sampling_settings(solver, [])
    original = settings.model_dump()
    requests = []

    def serve(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": SOLVER_MODELS[solver]}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"prompt_tokens": served_prompt_tokens}, "choices": []})

    client = httpx.Client
    monkeypatch.setattr(probe.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(serve), **kwargs))
    monkeypatch.setattr(
        probe.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3]),
    )
    if served_prompt_tokens == 3:
        result = probe.probe_endpoint(solver, "https://fixture/v1", SOLVER_MODELS[solver], "fixture", settings)
        assert result["tools_thinking_probe"]["request"] == requests[0]
    else:
        with pytest.raises(ActionFailure) as failure:
            probe.probe_endpoint(solver, "https://fixture/v1", SOLVER_MODELS[solver], "fixture", settings)
        assert failure.value.reason == GapReason.SERVING_PROBE_MISMATCH
    assert requests[0]["max_tokens"] == 512
    assert requests[0]["model"] == SOLVER_MODELS[solver]
    assert requests[0]["tools"]
    assert settings.model_dump() == original
    assert settings.output_tokens == 16384
