# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import threading
from dataclasses import replace
from hashlib import sha256

import pytest
from marin.inference.backend import ModelSpec
from marin.inference.config import VllmCompilationCacheMode
from marin.inference.vllm_backend import VllmBackend
from marin.testing.inference import snowball
from marin.testing.inference.backend_parity import TokenScore
from marin.testing.inference.fake_vllm_server import FakeVllmLauncher
from marin.testing.inference.snowball import PromptBatch, RepresentativeCase, RepresentativePromptFixture
from marin.testing.inference.snowball_backend_parity_jobs import score_vllm_endpoint, tpu_vllm_engine_config
from rigging.filesystem.storage_path import StoragePath


def fixture():
    cases = tuple(
        RepresentativeCase(
            id="knowledge-longbench-02" if i == 0 else f"case-{i}",
            prompt_token_ids=(i,) * 2050,
            top_logprobs=(TokenScore(token_id=7, logprob=math.log(0.8)), TokenScore(token_id=9, logprob=math.log(0.2))),
        )
        for i in range(8)
    )
    return RepresentativePromptFixture("unused", "unused", cases, (PromptBatch(4096, cases),))


@pytest.mark.parametrize("ranks", [1, 8])
@pytest.mark.parametrize("failure", [None, "repeat", "probability", "missing", "nan"])
def test_endpoint_checks_wire_distributions_and_repeats(monkeypatch, ranks, failure):
    requests = []
    lock = threading.Lock()
    prompts = fixture()

    class Response:
        def __init__(self, request):
            self.request = request

        def raise_for_status(self):
            pass

        def json(self):
            request_id = self.request["headers"]["X-Request-Id"]
            probability = 0.5 if failure == "probability" else 0.8
            if failure == "repeat" and request_id.startswith("same-rank-repeat-1"):
                probability = 0.79
            scores = {"token_id:7": math.log(probability), "token_id:9": math.log(1 - probability)}
            if failure == "missing":
                del scores["token_id:9"]
            if failure == "nan":
                scores["token_id:9"] = float("nan")
            return {
                "choices": [
                    {
                        "prompt_token_ids": self.request["json"]["prompt"],
                        "token_ids": [7],
                        "logprobs": {"top_logprobs": [scores]},
                    }
                ]
            }

    def post(url, **kwargs):
        with lock:
            requests.append(kwargs)
        return Response(kwargs)

    monkeypatch.setattr("marin.testing.inference.snowball_backend_parity_jobs.requests.post", post)
    if failure:
        with pytest.raises(AssertionError):
            score_vllm_endpoint(prompts, "http://unused/completions", "model", data_parallel_size=ranks)
        return
    parities = score_vllm_endpoint(prompts, "http://unused/completions", "model", data_parallel_size=ranks)
    assert {p.case_id for p in parities} == {c.id for c in prompts.cases}
    assert [p.max_probability_error for p in parities] == pytest.approx([0.0] * len(parities), abs=1e-15)
    waves = [r for r in requests if r["headers"]["X-Request-Id"].startswith("wave-")]
    assert sorted(r["headers"]["X-data-parallel-rank"] for r in waves) == sorted(str(i % ranks) for i in range(8))
    assert {tuple(r["json"]["prompt"]) for r in waves} == {c.prompt_token_ids for c in prompts.cases}
    assert all(r["json"]["max_tokens"] == 1 and r["json"]["add_special_tokens"] is False for r in requests)
    assert all(r["json"]["logprobs"] == 50 and "echo" not in r["json"] for r in requests)
    assert len(requests) == 8 + 2 * ranks


def test_tpu_arm_engine_timeout_is_an_integer_in_the_server_process(monkeypatch, tmp_path):
    record = tmp_path / "timeout.json"
    monkeypatch.setattr(
        "marin.inference.vllm_backend.vllm_launcher",
        lambda config: FakeVllmLauncher("record-engine-timeout", str(record)),
    )
    config = replace(tpu_vllm_engine_config(), compilation_cache=VllmCompilationCacheMode.CALLER_MANAGED)
    spec = ModelSpec(
        weights="fake-model",
        api_model="model",
        num_chips=16,
        tensor_parallel_size=16,
        dtype="bfloat16",
        max_model_len=32768,
        chat_template_content="",
    )
    with VllmBackend(config).start(spec) as environment:
        environment.wait_until_ready()
        value = json.loads(record.read_text())["engine_ready_timeout"]
        assert int(value) == config.startup_timeout_seconds


@pytest.mark.parametrize("damage", [None, "bytes", "export"])
def test_durable_manifest_requires_both_pinned_bytes_and_export_identity(monkeypatch, damage):
    model = snowball.SNOWBALL
    manifest = dict(
        verified=True, destination=model.tpu_export_uri, source=model.export_uri, export_sha256=model.export_sha256
    )
    if damage == "export":
        manifest["export_sha256"] = "different-export"
    raw = json.dumps(manifest).encode()
    monkeypatch.setattr(snowball, "SNOWBALL", replace(model, tpu_manifest_sha256=sha256(raw).hexdigest()))
    monkeypatch.setattr(StoragePath, "read_bytes", lambda self: raw + (b" " if damage == "bytes" else b""))
    if damage:
        with pytest.raises(AssertionError):
            snowball.read_tpu_export_manifest()
    else:
        assert snowball.read_tpu_export_manifest() == manifest
