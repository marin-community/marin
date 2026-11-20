import json

import pytest

from marinskyrl.near_window_serving import run_near_window_probe


class RenderedTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return [1, 2, *([3] * messages[-1]["content"].count(" x")), 4, 5]


class CompletionServer:
    def __init__(self, *, drop_prompt_token=False):
        self.requests = []
        self.drop_prompt_token = drop_prompt_token

    async def completion(self, request_payload):
        request = request_payload["json"]
        self.requests.append(request)
        prompt = request["prompt"]
        count = len(prompt) - int(self.drop_prompt_token) if isinstance(prompt, list) else 7
        return {"choices": [{"text": "ok"}], "usage": {
            "prompt_tokens": count, "completion_tokens": request["max_tokens"],
        }}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_prompt_token", [False, True])
async def test_probe_certifies_server_rendered_length_and_persists_failed_gate(tmp_path, monkeypatch, drop_prompt_token):
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    engines = [CompletionServer(drop_prompt_token=drop_prompt_token), CompletionServer()]

    async def memory_reader():
        return [{"gpu_uuid": f"gpu-{rank}", "used_mib": 120000, "total_mib": 180000} for rank in range(2)]

    destination = tmp_path / "published-receipt.json"
    settings = dict(engines=2, minimum_prompt_tokens=20, max_new_tokens=4, context_tokens=64,
                    served_model_name="pinned-model", temperature=0.7, top_p=1.0,
                    request_timeout=10, memory_sample_interval=2, receipt_uri=str(destination))
    if drop_prompt_token:
        with pytest.raises(ExceptionGroup) as failure:
            await run_near_window_probe(engines, RenderedTokenizer(), settings, 3, memory_reader=memory_reader)
        assert any(isinstance(error, ValueError) for error in failure.value.exceptions)
    else:
        await run_near_window_probe(engines, RenderedTokenizer(), settings, 3, memory_reader=memory_reader)
    receipt = json.loads(destination.read_bytes())
    assert receipt == json.loads((tmp_path / "near-window-serving-receipt.json").read_bytes())
    assert receipt["result"] == ("failed" if drop_prompt_token else "passed")
    if drop_prompt_token:
        assert "near_window_request" not in receipt
        return
    near = engines[0].requests[-1]
    assert near["prompt"][0:2] == [1, 2] and near["prompt"][-2:] == [4, 5]
    assert receipt["near_window_request"]["usage"] == {"prompt_tokens": 24, "completion_tokens": 4}
    assert receipt["peak_used_mib_by_gpu"] == {"gpu-0": 120000, "gpu-1": 120000}
    assert len(receipt["rank_readiness_requests"]) == 2
