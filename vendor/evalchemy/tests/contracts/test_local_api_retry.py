"""Transport retry behavior for Evalchemy's local API adapters."""

import asyncio
import json

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.WildBench import eval_instruct as wildbench_source
from eval.contracts.eval_spec import ModelRequest, RequestKind
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.execution.backend import LMBackend
from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.execution.judge import JudgeBackend
from eval.execution.metrics import SourceMetrics
from eval.graders.answer_equivalence import JudgeConfig
from eval.graders.specs import GRADERS
from eval.runtime import pool as runtime_pool
from eval.sample_logging import canonicalize_samples
from eval.serve_eval.local_api import RetryingLocalChatCompletions, RetryingLocalCompletions
from eval.sources.common import SourceOptions
from lm_eval.models.api_models import JsonChatStr
from tests.contracts.source_harness import FixedEndpoint, MemoryCheckpoints


class _Clock:
    def __init__(self):
        self.now = 0.0

    async def sleep(self, delay):
        self.now += delay
        await asyncio.sleep(0)


async def _endpoint(handler):
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


def _adapter(kind, base_url, budget, *, timeout=5, attempt_timeout=None):
    budget_arg = "" if budget is None else f",transport_retry_budget={budget}"
    attempt_timeout_arg = "" if attempt_timeout is None else f",transport_attempt_timeout={attempt_timeout}"
    model_args = (
        f"model=served,base_url={base_url}/{kind},tokenizer_backend=none,"
        f"tokenized_requests=False,max_length=4096,timeout={timeout}{budget_arg}"
        f"{attempt_timeout_arg}"
    )
    if kind == "chat/completions":
        return RetryingLocalChatCompletions.create_from_arg_string(model_args)
    return RetryingLocalCompletions.create_from_arg_string(model_args)


@pytest.mark.parametrize("budget,successful", [(None, True), (900, False)])
def test_local_api_long_attempt_respects_default_budget_and_explicit_override(monkeypatch, budget, successful):
    async def run():
        adapter = _adapter("completions", "http://unused.invalid/v1", budget, timeout=1800)
        loop = asyncio.get_running_loop()
        original_time = loop.time
        elapsed = 0

        async def slow_http_call(**_kwargs):
            nonlocal elapsed
            elapsed += 1000
            # Let both the request task and its real deadline callbacks run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return ["answer"]

        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: original_time() + elapsed)
            patch.setattr(adapter, "amodel_call", slow_http_call)
            return await _generate(adapter, "question")

    [output] = asyncio.run(run())
    assert (output == "answer") is successful
    if not successful:
        assert output.failure_category == "AgentTimeoutError"


def _prompt(kind):
    if kind == "chat/completions":
        return JsonChatStr(json.dumps([{"role": "user", "content": "question"}]))
    return "question"


def _response(kind):
    if kind == "chat/completions":
        return {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "answer"}}]}
    return {"choices": [{"index": 0, "text": "answer"}]}


async def _generate(adapter, prompt):
    async with LMBackend(adapter) as backend:
        return [await backend.execute(ModelRequest(RequestKind.GENERATE, prompt))]


@pytest.mark.parametrize("status,choice,fallback", [
    (200, {"message": {"content": ""}}, True),
    (200, {"message": {"content": "  \n"}}, True),
    (200, {"message": {"content": None}}, True),
    (200, {"message": {"content": "malformed"}}, True),
    (402, None, False),
    (200, {"finish_reason": "content_filter"}, False),
    (200, {"message": {"content": 99}}, False),
])
def test_mixeval_http_semantic_retries_preserve_empty_judgments_and_payment_errors(status, choice, fallback):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        payload = {"choices": [choice]} if status == 200 else {"error": "prepayment required"}
        return web.json_response(payload, status=status)

    source = MixEvalBenchmark(debug=True)
    spec = next(spec for spec in source.iter_specs(SourceOptions("MixEval", source.SOURCE, limit=1))
                if spec.namespace == "multichoice")
    checkpoint = MemoryCheckpoints()

    async def run():
        runner, base_url = await _endpoint(handler)
        grading = Graders(GRADERS)
        try:
            async with ClientSession() as session:
                judge = JudgeBackend(JudgeConfig("judge", base_url, "test-key"), session)

                async def evaluate():
                    sink = CollectSamples()
                    candidate = FixedEndpoint("candidate")
                    engine = Evaluator(candidate, grading, sink, workers=1, judges={"MixEval": judge},
                                       checkpoints={"MixEval": checkpoint})
                    completed = [item async for item in engine.run([spec])]
                    return completed[0], sink.samples["MixEval"], candidate.requests

                first, samples, candidate_requests = await evaluate()
                assert len(candidate_requests) == 1
                if not fallback:
                    assert first.error is not None
                    assert all("accuracy" not in sample for sample in samples)
                    return
                assert first.error is None
                assert samples[0]["judge_fallback"] == "malformed_response_exhaustion"
                replay, replay_samples, candidate_requests = await evaluate()
                assert replay.error is None
                assert replay_samples == samples
                assert candidate_requests == []
        finally:
            grading.close()
            await runner.cleanup()

    asyncio.run(run())
    assert len(requests) == (10 if fallback else 1)


@pytest.mark.parametrize("available", [0, 1])
@pytest.mark.parametrize("bad_response", ["blocked", "invalid_json"])
def test_wildbench_excludes_unavailable_http_judgments(monkeypatch, available, bad_response):
    rows = [
        {"session_id": "blocked", "conversation_input": [{"role": "user", "content": "blocked-judge"}],
         "checklist": ["be right"], "primary_tag": "Coding & Debugging", "secondary_tags": []},
    ]
    if available:
        rows.append({"session_id": "valid", "conversation_input": [{"role": "user", "content": "valid-judge"}],
                     "checklist": ["be right"], "primary_tag": "Math", "secondary_tags": []})
    monkeypatch.setattr(wildbench_source, "load_dataset", lambda *_args, **_kwargs: rows)
    source = wildbench_source.WildBenchBenchmark()

    async def handler(request):
        payload = await request.json()
        if "blocked-judge" in payload["messages"][0]["content"]:
            if bad_response == "blocked":
                return web.json_response(
                    {"choices": [{"index": 0, "finish_reason": "content_filter: PROHIBITED_CONTENT"}]}
                )
            content = "not a judgment"
        else:
            content = '{"score": "7"}'
        return web.json_response({"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}]})

    async def run():
        runner, base_url = await _endpoint(handler)
        sink = CollectSamples()
        grading = Graders(GRADERS)
        metrics = SourceMetrics(repeated=False, policy=source.REDUCER)
        try:
            async with ClientSession() as session:
                judge = JudgeBackend(JudgeConfig("judge", base_url, "test-key"), session)
                engine = Evaluator(FixedEndpoint("candidate"), grading, sink, workers=2, judges={"WildBench": judge})
                options = SourceOptions("WildBench", source.SOURCE)
                async for completed in engine.run(source.iter_specs(options)):
                    if completed.error is not None:
                        raise completed.error
                    metrics.add(completed.result)
            return metrics.finish("WildBench")["results"]["WildBench"], sink.samples["WildBench"]
        finally:
            grading.close()
            await runner.cleanup()

    metrics, samples = asyncio.run(run())

    assert metrics["score"] == (7.0 if available else None)
    assert metrics["adjusted_score"] == (4.0 if available else None)
    assert metrics["task_macro_score"] == (
        pytest.approx(4 / sum(source.config.task_weights.values())) if available else None
    )
    assert metrics["task_categorized_scores"] == ({"Math & Data Analysis": 4.0} if available else {})
    assert metrics["num_total"] == 1 + available
    assert metrics["num_judged"] == available
    assert metrics["num_judge_failed"] == 1
    assert metrics["judge_coverage"] == pytest.approx(available / (1 + available))
    [failed] = [sample for sample in samples if sample.get("judge_failed")]
    assert failed["failure_category"] == "grader_infrastructure"
    assert "score" not in failed and "adjusted_score" not in failed


@pytest.mark.parametrize("kind", ["completions", "chat/completions"])
def test_local_api_recovers_after_ten_minute_proxy_outage(monkeypatch, kind):
    clock = _Clock()
    attempts = 0
    monkeypatch.setattr(runtime_pool, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime_pool, "sleep", clock.sleep)
    monkeypatch.setattr(runtime_pool.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if clock.now < 600:
            return web.json_response({"error": "proxy unavailable"}, status=502)
        return web.json_response(_response(kind))

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter(kind, base_url, 900)
            return await _generate(adapter, _prompt(kind))
        finally:
            await runner.cleanup()

    output = asyncio.run(run())

    assert output == ["answer"]
    assert attempts > 8
    assert clock.now >= 600


@pytest.mark.parametrize("status,retry_after", [(408, None), (429, 7), (500, None), (503, 12), (504, None)])
def test_local_api_retries_other_transient_http_statuses(monkeypatch, status, retry_after):
    clock = _Clock()
    attempts = 0
    monkeypatch.setattr(runtime_pool, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime_pool, "sleep", clock.sleep)
    monkeypatch.setattr(runtime_pool.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
            return web.json_response({"error": "unavailable"}, status=status, headers=headers)
        if retry_after is not None:
            assert clock.now >= retry_after
        return web.json_response(_response("completions"))

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter("completions", base_url, 20)
            return await _generate(adapter, "question")
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == ["answer"]
    assert attempts == 2


@pytest.mark.parametrize("kind", ["completions", "chat/completions"])
def test_local_api_retries_a_stalled_attempt_within_the_total_budget(monkeypatch, kind):
    attempts = 0
    release_stalled_handler = asyncio.Event()
    monkeypatch.setattr(runtime_pool.random, "uniform", lambda _low, _high: 0)

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await release_stalled_handler.wait()
        else:
            release_stalled_handler.set()
        return web.json_response(_response(kind))

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter(kind, base_url, 1, timeout=1, attempt_timeout=0.05)
            return await _generate(adapter, _prompt(kind))
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == ["answer"]
    assert attempts == 2


@pytest.mark.parametrize("kind", ["completions", "chat/completions"])
def test_local_api_bad_request_fails_after_one_attempt(kind):
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        return web.json_response({"error": "malformed request"}, status=400)

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter(kind, base_url, 900)
            with pytest.raises(ClientResponseError) as error:
                await _generate(adapter, _prompt(kind))
            assert error.value.status == 400
        finally:
            await runner.cleanup()

    asyncio.run(run())

    assert attempts == 1


def test_local_api_budget_exhaustion_remains_infrastructure_error(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(runtime_pool, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime_pool, "sleep", clock.sleep)
    monkeypatch.setattr(runtime_pool.random, "uniform", lambda _low, high: high)

    async def handler(_request):
        return web.json_response({"error": "proxy unavailable"}, status=502)

    async def run():
        runner, base_url = await _endpoint(handler)
        try:
            adapter = _adapter("completions", base_url, 20)
            return await _generate(adapter, "question")
        finally:
            await runner.cleanup()

    output = asyncio.run(run())
    record = canonicalize_samples("task", [{"resps": [output], "metrics": ["accuracy"], "accuracy": 0.0}])[0]
    sample = sample_from_lm_eval(
        "task",
        {
            **record,
            "doc_id": 0,
            "doc": {"question": "question"},
            "target": "answer",
            "arguments": [["question", {}]],
            "filtered_resps": [output[0]],
            "filter": "none",
        },
    )

    assert clock.now == 20
    assert record["failure_category"] == "model_transport"
    assert sample.output == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
