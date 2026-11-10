"""Numeric chat evidence and request lifetimes at the engine boundary."""

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from skyrl_train.inference_engines.vllm.compact_chat import CompactChatEngineClient


@dataclass
class SamplingParams:
    temperature: float = 1.0
    logprobs: int | None = None
    flat_logprobs: bool = False
    max_tokens: int = 4096

    def clone(self):
        return replace(self)


def _output(top_k, temperature=1.0):
    logits = np.asarray([[0.5, -0.25, 1.3, -2, -0.8, 0.2, -3], [1.2, 0.2, 2, -3, -1, 0.5, -2]])
    scaled = logits / temperature
    normalized = scaled - np.log(np.exp(scaled).sum(-1, keepdims=True))
    chosen = [6, 2]
    ids = np.column_stack((chosen, np.argsort(-normalized, axis=-1)[:, :top_k]))
    scores = np.take_along_axis(normalized, ids, axis=-1).astype(np.float32)
    stride = top_k + 1
    flat = SimpleNamespace(
        start_indices=[0, stride],
        end_indices=[stride, 2 * stride],
        token_ids=ids.reshape(-1).tolist(),
        logprobs=scores.reshape(-1).tolist(),
    )
    return SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(token_ids=chosen, logprobs=flat, text="answer", finish_reason="stop")],
    ), normalized


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [1.0, 0.7])
async def test_compact_chat_preserves_full_vocabulary_probabilities_and_sampling_options(temperature):
    class Engine:
        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            self.received = prompt, sampling_params, request_id, kwargs
            yield _output(sampling_params.logprobs, sampling_params.temperature)[0]

    engine = Engine()
    requests = set()
    client = CompactChatEngineClient(engine, requests)
    params = SamplingParams(temperature=temperature)
    with client.capture(2) as capture:
        outputs = [output async for output in client.generate([1, 2], params, "chat-1", data_parallel_rank=3)]
    token_ids, selected = capture.response
    _, normalized = _output(2, temperature)
    expected_ids = np.argsort(-normalized, axis=-1)[:, :2]
    np.testing.assert_array_equal(selected.topk_indices, expected_ids)
    np.testing.assert_allclose(
        selected.topk_logprobs,
        np.take_along_axis(normalized, expected_ids, axis=-1).astype(np.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        selected.sampled_logprobs, normalized[[0, 1], token_ids].astype(np.float32), rtol=0, atol=0
    )
    assert np.all(np.exp(selected.topk_logprobs).sum(-1) < 1)
    assert token_ids == outputs[0].outputs[0].token_ids == [6, 2]
    assert outputs[0].outputs[0].text == "answer"
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert engine.received == ([1, 2], SamplingParams(temperature, 3, True), "chat-1", {"data_parallel_rank": 3})
    assert params == SamplingParams(temperature=temperature)
    assert selected.topk_indices.dtype == np.int32
    assert selected.topk_logprobs.dtype == np.float32
    assert not requests


@pytest.mark.asyncio
async def test_concurrent_compact_and_ordinary_chats_keep_separate_heads():
    class Engine:
        def __init__(self):
            self.started = set()
            self.ready = asyncio.Event()

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            self.started.add(request_id)
            if len(self.started) == 3:
                self.ready.set()
            await self.ready.wait()
            if sampling_params.logprobs is None:
                yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(logprobs=None)])
            else:
                yield _output(sampling_params.logprobs)[0]

    client = CompactChatEngineClient(Engine(), set())

    async def compact(request_id, width):
        with client.capture(width) as capture:
            async for _ in client.generate([1], SamplingParams(), request_id):
                pass
        return capture.response[1]

    async def ordinary():
        return [output async for output in client.generate([1], SamplingParams(), "ordinary")]

    first, second, plain = await asyncio.gather(compact("first", 1), compact("second", 2), ordinary())
    np.testing.assert_array_equal(first.topk_indices, [[2], [2]])
    np.testing.assert_array_equal(second.topk_indices, [[2, 0], [2, 0]])
    assert plain[0].outputs[0].logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "cancel", "error"])
async def test_incomplete_chat_closes_engine_iterator_and_releases_capture(ending):
    class Engine:
        def __init__(self):
            self.waiting = asyncio.Event()
            self.release = asyncio.Event()
            self.active = set()

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            self.active.add(request_id)
            try:
                yield SimpleNamespace(finished=False)
                self.waiting.set()
                if ending == "error":
                    raise RuntimeError("engine failed")
                await self.release.wait()
            finally:
                self.active.remove(request_id)

    engine = Engine()
    requests = set()
    client = CompactChatEngineClient(engine, requests)
    with client.capture(2) as capture:
        iterator = client.generate([1], SamplingParams(), "interrupted")
        assert not (await anext(iterator)).finished
        if ending == "close":
            await iterator.aclose()
        elif ending == "error":
            with pytest.raises(RuntimeError, match="engine failed"):
                await anext(iterator)
        else:
            pending = asyncio.create_task(anext(iterator))
            await engine.waiting.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    assert capture.response is None
    assert not engine.active
    assert not requests


@pytest.mark.asyncio
async def test_abort_before_first_token_keeps_empty_aligned_evidence():
    class Engine:
        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            yield SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(token_ids=[], logprobs=None, finish_reason="abort")],
            )

    client = CompactChatEngineClient(Engine(), set())
    with client.capture(128) as capture:
        output = [output async for output in client.generate([1], SamplingParams(), "empty")]
    ids, selected = capture.response
    assert ids == selected.sampled_logprobs == []
    assert selected.topk_indices.shape == selected.topk_logprobs.shape == (0, 128)
    assert output[0].outputs[0].finish_reason == "abort"
