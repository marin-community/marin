# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
from functools import partial
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from experiments.post_training import mochi_dspark_gate as gate


@pytest.mark.parametrize("sampled_logprob_error", [0.0, 1.0])
def test_gate_uses_current_plain_teacher_and_rejects_changed_sampled_logprobs(
    tmp_path, monkeypatch, sampled_logprob_error
):
    def response(request):
        body = json.loads(request.content)
        count = body["max_tokens"]
        routes = np.broadcast_to(np.arange(8, dtype=np.uint16), (count, 48, 8))
        encoded = io.BytesIO()
        np.save(encoded, routes, allow_pickle=False)
        error = sampled_logprob_error if request.url.host == "draft" and count > 1 else 0.0
        choice = {
            "token_ids": [7] * count,
            "finish_reason": "length",
            "routed_experts": base64.b64encode(encoded.getvalue()).decode(),
            "logprobs": {
                "token_logprobs": [-1.0 - error] * count,
                "top_logprobs": [{"token_id:7": -1.0}] * count,
            },
        }
        return httpx.Response(200, json={"choices": [choice]})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(response)))
    bounds = {
        "max_absolute_logprob_error": 0.1,
        "max_top1_logprob_error": 0.1,
        "top1_mismatch_fraction": 0.0,
        "mean_prompt_route_choice_mismatch_fraction": 0.0,
    }
    sampling_bounds = {
        "mean_logprob": {"reference": -20.0, "maximum_mean_shift": 2.0, "standard_error": 0.1},
        "mean_top1_probability": {"reference": 0.0, "maximum_mean_shift": 2.0, "standard_error": 0.1},
    }
    calibration = {
        "probes": [{"identity": f"prompt-{index}@0", "prompt": [index + 1, 7], "token": 7} for index in range(2)],
        "fixed_bounds": bounds,
        "decode_bounds": bounds,
        # A different checkpoint's teacher must not score the current policy.
        "reference": {"wrong-policy": {"logprob": -20.0}},
        "sampling": {"by_prompt": {}, "metrics": sampling_bounds, "standard_errors": 3},
    }
    spec = {
        "output": str(tmp_path),
        "correctness_seeds": [17, 23, 29],
        "calibration_sha256": "frozen-noise-limits",
        "model": {"weights": "s3://regional/current-policy"},
    }
    asyncio.run(gate.qualify(SimpleNamespace(server_url="http://plain/v1"), spec, "plain-check", calibration))
    reference = json.loads((tmp_path / "plain-check/sampling-reference.json").read_text())
    for metric, limits in sampling_bounds.items():
        assert reference["metrics"][metric]["maximum_mean_shift"] == limits["maximum_mean_shift"]
    assert reference["metrics"]["mean_logprob"]["reference"] == -1.0
    asyncio.run(gate.qualify(SimpleNamespace(server_url="http://draft/v1"), spec, "dspark-check", calibration))
    if sampled_logprob_error:
        with pytest.raises(ValueError, match="dspark-check/autoregressive-correctness"):
            asyncio.run(
                gate.qualify(SimpleNamespace(server_url="http://teacher/v1"), spec, "plain-rescore", calibration)
            )
        assert not (tmp_path / "correctness-passed.json").exists()
        rejected = json.loads((tmp_path / "dspark-check/autoregressive-correctness.json").read_text())
        assert rejected["difference"]["max_absolute_logprob_error"] == sampled_logprob_error
        assert not rejected["passed"]
        return
    asyncio.run(gate.qualify(SimpleNamespace(server_url="http://teacher/v1"), spec, "plain-rescore", calibration))
    passed = json.loads((tmp_path / "correctness-passed.json").read_text())
    assert passed["passed"] and passed["teacher_model"] == spec["model"]["weights"]


def test_plain_investigation_retains_failed_controls_and_pins_requested_rank(tmp_path, monkeypatch):
    calls = {}

    def response(request):
        body = json.loads(request.content)
        identity = body["prompt"][0]
        rank = request.headers.get("X-data-parallel-rank")
        count = calls.get((identity, rank), 0)
        calls[identity, rank] = count + 1
        # The external server changes one repeated plain score, then returns a
        # distinguishable score on each explicitly requested DP rank.
        logprob = -1.0 - (int(rank) if rank is not None else count % 2)
        encoded = io.BytesIO()
        np.save(encoded, np.broadcast_to(np.arange(8, dtype=np.uint16), (1, 48, 8)), allow_pickle=False)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [7],
                        "routed_experts": base64.b64encode(encoded.getvalue()).decode(),
                        "logprobs": {"top_logprobs": [{"token_id:7": logprob}]},
                    }
                ]
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(response)))
    calibration = {
        "probes": [{"identity": f"prompt-{i}@0", "prompt": [i + 1, 7], "token": 7} for i in range(2)],
        "fixed_bounds": {"max_absolute_logprob_error": 0.1},
    }
    spec = {"output": str(tmp_path), "calibration_sha256": "frozen", "model": {"weights": "s3://local/V5"}}
    asyncio.run(gate.investigate_plain_batching(SimpleNamespace(server_url="http://plain/v1"), spec, calibration))
    summary = json.loads((tmp_path / "plain-batching/investigation-complete.json").read_text())
    repeated = json.loads((tmp_path / "plain-batching/auto-c16-1.json").read_text())
    assert not repeated["within_frozen_bounds"]
    assert repeated["difference_from_same_condition_first"]["max_absolute_logprob_error"] == 1.0
    rank0 = json.loads((tmp_path / "plain-batching/rank0-c1-1.json").read_text())
    rank1 = json.loads((tmp_path / "plain-batching/rank1-c1-1.json").read_text())
    assert rank0["scores"]["prompt-0@0"]["logprob"] == -1.0
    assert rank1["scores"]["prompt-0@0"]["logprob"] == -2.0
    assert summary["comparisons"]["rank0-c1-vs-rank1-c1"]["max_absolute_logprob_error"] == 1.0
    assert not summary["qualification_passed"]
    assert not (tmp_path / "correctness-passed.json").exists()
