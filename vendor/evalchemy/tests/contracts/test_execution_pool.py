"""Observable admission, retry, cancellation and checkpoint behavior."""

import asyncio
from dataclasses import asdict, replace
from threading import Event

import pytest

from eval.completion_response import FailedGeneration
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.execution.engine import CollectSamples, Evaluator
from eval.runtime.pool import ExecutionPool, RetryPolicy, retry
from eval.runtime.checkpoints import CheckpointRow, input_digest
from tests.contracts.source_harness import MemoryCheckpoints


def test_pool_bounds_source_consumption_and_preserves_positions_when_grouping():
    consumed = []
    started = []

    async def run():
        release = asyncio.Event()
        admitted = asyncio.Event()

        def source():
            for value in range(30):
                consumed.append(value)
                yield value

        async def execute(value):
            started.append(value)
            if len(started) == 2:
                admitted.set()
            await release.wait()
            return value * 10

        stream = ExecutionPool(2, lookahead=4).map(source(), execute, group=lambda value: (-value,))
        first = asyncio.create_task(anext(stream))
        await admitted.wait()
        assert consumed == [0, 1, 2, 3]
        assert started == [3, 2]
        release.set()
        results = [await first] + [item async for item in stream]
        return results

    results = asyncio.run(run())
    assert sorted((item.position, item.result) for item in results) == [(value, value * 10) for value in range(30)]


def test_pool_closing_early_cancels_and_joins_outstanding_work():
    cancelled = []

    async def run():
        both_started = asyncio.Event()

        async def execute(value):
            if value == 0:
                await both_started.wait()
                return value
            both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(value)

        stream = ExecutionPool(2).map(range(2), execute)
        result = await anext(stream)
        await stream.aclose()
        return result.result

    assert asyncio.run(run()) == 0
    assert cancelled == [1]


def test_pool_blocking_source_refill_allows_inflight_response_to_finish():
    release_refill = Event()

    async def run():
        loop = asyncio.get_running_loop()
        refill_started = asyncio.Event()
        second_started = asyncio.Event()
        release_response = asyncio.Event()
        response_finished = asyncio.Event()

        def source():
            yield 0
            yield 1
            loop.call_soon_threadsafe(refill_started.set)
            if not release_refill.wait(timeout=5):
                raise RuntimeError("source refill blocked the event loop")
            yield 2

        async def execute(value):
            if value == 0:
                await second_started.wait()
            elif value == 1:
                second_started.set()
                await release_response.wait()
                response_finished.set()
            return value

        stream = ExecutionPool(2, lookahead=2).map(source(), execute)
        results = [await anext(stream)]
        refill = asyncio.create_task(anext(stream))
        try:
            async with asyncio.timeout(3):
                await refill_started.wait()
                release_response.set()
                await response_finished.wait()
        finally:
            release_refill.set()
        results.append(await refill)
        results.extend([item async for item in stream])
        return results

    results = asyncio.run(run())
    assert sorted((item.position, item.result) for item in results) == [(0, 0), (1, 1), (2, 2)]


def test_pool_cancellation_joins_source_admission_before_checkpoint_cleanup():
    release_refill = Event()
    refill_finished = Event()
    finalized = []

    async def run():
        loop = asyncio.get_running_loop()
        refill_started = asyncio.Event()
        cancelled = asyncio.Event()

        def source():
            for position in range(2):
                yield replace(_spec(), source_id=str(position), ordinal=position, requests=_spec().requests[:1])
            loop.call_soon_threadsafe(refill_started.set)
            if not release_refill.wait(timeout=5):
                raise RuntimeError("cancelled admission was not released")
            refill_finished.set()
            yield replace(_spec(), source_id="2", ordinal=2, requests=_spec().requests[:1])

        class Archive:
            def records(self):
                yield from ()

            def record(self, row):
                pass

            def finalize(self):
                assert refill_finished.is_set()
                finalized.append("checkpoint_closed")

        engine = Evaluator(_Model(), _grade, CollectSamples(), workers=1, checkpoints={"conversation": Archive()})
        engine.pool.lookahead = 2
        stream = engine.run(source())
        await anext(stream)
        await anext(stream)
        refill = asyncio.create_task(anext(stream))
        try:
            async with asyncio.timeout(3):
                await refill_started.wait()
                refill.cancel()
                loop.call_soon(cancelled.set)
                await cancelled.wait()
                assert not refill.done()
                assert finalized == []
        finally:
            release_refill.set()
        with pytest.raises(asyncio.CancelledError):
            await refill
        await stream.aclose()

    asyncio.run(run())
    assert finalized == ["checkpoint_closed"]


def test_pool_reports_item_failure_and_finishes_siblings():
    async def execute(value):
        if value == 1:
            raise ConnectionError("unavailable")
        return value * 2

    async def run():
        return [item async for item in ExecutionPool(2).map(range(3), execute)]

    results = {item.position: item for item in asyncio.run(run())}
    assert results[0].result == 0
    assert isinstance(results[1].error, ConnectionError)
    assert results[2].result == 4


def test_retry_recovers_transient_failure_and_propagates_permanent_failure():
    attempts = []

    async def call():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("temporary outage")
        return "completed"

    policy = RetryPolicy(budget=2, attempt_timeout=1, max_attempts=2)
    assert asyncio.run(retry(call, policy, lambda error: isinstance(error, ConnectionError))) == "completed"
    attempts.clear()
    with pytest.raises(ConnectionError, match="temporary outage"):
        asyncio.run(retry(call, policy, lambda _error: False))
    assert len(attempts) == 1


class _Model:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    async def execute(self, request):
        self.requests.append(request)
        if self.fail:
            return FailedGeneration("model_transport")
        return "answer " + str(len(request.prompt))


async def _grade(_spec, outputs):
    return Grade({"accuracy": float(bool(outputs[-1]))})


def _spec():
    return EvalSpec(
        task="conversation", source=SourceRef("fixture"), source_id="opaque-id", ordinal=0,
        requests=(
            ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "first question"}]),
            ModelRequest(RequestKind.GENERATE, [{"role": "user", "content": "second question"}], previous_turn=0),
        ),
        grader=GraderSpec("exact_match"), target="answer 3",
    )


def _evaluate(specs, model, store):
    samples = CollectSamples()
    evaluator = Evaluator(model, _grade, samples, workers=2, checkpoints={"conversation": store})

    async def run():
        return [item async for item in evaluator.run(specs)]

    return asyncio.run(run()), samples.samples


def test_conversation_checkpoints_each_turn_and_restores_without_aliasing():
    store = MemoryCheckpoints()
    model = _Model()
    result, samples = _evaluate([_spec()], model, store)

    assert result[0].error is None
    assert model.requests[1].prompt == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "second question"},
    ]
    resumed = _Model()
    retried, resumed_samples = _evaluate([_spec()], resumed, store)
    assert retried[0].error is None
    assert resumed.requests == []
    assert resumed_samples == samples
    assert len(store.rows) == 2
    assert store.closed


def test_failed_transport_is_generated_again_on_resume():
    store = MemoryCheckpoints()
    failed = _Model(fail=True)
    spec = replace(_spec(), requests=_spec().requests[:1])
    _evaluate([spec], failed, store)
    assert store.rows == {}

    healthy = _Model()
    result, samples = _evaluate([spec], healthy, store)
    assert len(healthy.requests) == 1
    assert result[0].error is None
    assert samples["conversation"][0]["accuracy"] == 1.0


def test_changed_checkpoint_input_refuses_a_stale_response():
    store = MemoryCheckpoints()
    spec = replace(_spec(), requests=_spec().requests[:1])
    _evaluate([spec], _Model(), store)
    changed = replace(spec, requests=(replace(spec.requests[0], generation={"temperature": 0.7}),))
    model = _Model()
    with pytest.raises(ValueError, match="checkpoint source changed"):
        _evaluate([changed], model, store)
    assert model.requests == []


def test_resume_stream_reads_only_the_admitted_window_and_closes_on_early_exit():
    consumed = []
    closed = []

    def spec_at(position):
        original = _spec()
        return replace(original, source_id=str(position), ordinal=position, requests=original.requests[:1])

    class Archive:
        def records(self):
            try:
                for position in range(10_000):
                    spec = spec_at(position)
                    consumed.append(position)
                    yield CheckpointRow(position, input_digest(asdict(spec)), {"phase": "generate", "request_index": 0},
                                        {"input_digest": input_digest(asdict(spec.requests[0])), "output": "saved answer"})
            finally:
                closed.append("reader")

        def record(self, row):
            raise AssertionError("restored operations must not be written again")

        def finalize(self):
            closed.append("writer")

    model = _Model()
    sink = CollectSamples()
    engine = Evaluator(model, _grade, sink, workers=2, checkpoints={"conversation": Archive()})

    async def run():
        stream = engine.run(spec_at(index) for index in range(10_000))
        result = await anext(stream)
        await stream.aclose()
        return result

    result = asyncio.run(run())
    assert result.error is None
    assert result.result.responses == ("saved answer",)
    assert len(consumed) <= 9
    assert model.requests == []
    assert closed == ["reader", "writer"]


def test_local_backend_batches_only_work_already_admitted_by_the_pool():
    from eval.execution.backend import LMBackend

    class LocalModel:
        world_size = 1

        def apply_chat_template(self, messages):
            return str(messages)

        def __init__(self):
            self.batches = []

        def generate_until(self, instances):
            self.batches.append(len(instances))
            return ["answer" for instance in instances]

    model = LocalModel()
    original = _spec()
    specs = [replace(original, source_id=str(index), ordinal=index, requests=original.requests[:1]) for index in range(3)]

    async def run():
        async with LMBackend(model) as backend:
            engine = Evaluator(backend, _grade, CollectSamples(), workers=3)
            return [item async for item in engine.run(specs)]

    results = asyncio.run(run())
    assert [item.error for item in results] == [None, None, None]
    assert model.batches == [3]


def test_local_backend_applies_trial_rng_without_forwarding_unsupported_seed():
    import numpy as np
    from eval.execution.backend import LMBackend

    class LocalModel:
        world_size = 1

        def apply_chat_template(self, messages):
            return str(messages)

        def generate_until(self, instances):
            for instance in instances:
                if "seed" in instance.args[1]:
                    raise ValueError("unsupported generation keyword: seed")
            return [str(np.random.randint(10000))] * len(instances)

    specs = []
    for index, trial in enumerate((0, 1, 0)):
        original = _spec()
        request = replace(original.requests[0], generation={"do_sample": True, "seed": 17 + trial})
        specs.append(replace(original, source_id=str(index), ordinal=index, trial=trial, requests=(request,)))

    async def run():
        async with LMBackend(LocalModel(), seeds=(17, 23, 29, 31)) as backend:
            engine = Evaluator(backend, _grade, CollectSamples(), workers=3)
            return [item async for item in engine.run(specs)]

    results = asyncio.run(run())
    assert all(item.error is None for item in results)
    actual = {item.job.source_id: item.result.responses[0] for item in results}
    assert actual == {
        str(index): str(np.random.RandomState(23 + trial).randint(10000))
        for index, trial in enumerate((0, 1, 0))
    }
