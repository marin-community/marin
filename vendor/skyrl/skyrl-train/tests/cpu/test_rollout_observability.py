import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from skyrl_train import rollout_observability as rollout
from skyrl_train.telemetry import record_training_metrics
from skyrl_train.batch_metrics import consumed_stop_metrics


@pytest.mark.parametrize(("reasons", "count"), [(["length", "stop", None], 3), (None, 3), ([], 0)])
def test_consumed_stop_fraction_is_absent_without_complete_coverage(reasons, count):
    metrics = consumed_stop_metrics(reasons, count)
    assert "consumed/length_stop_fraction" not in metrics
    assert metrics["consumed/known_stop_count"] + metrics["consumed/unknown_stop_count"] == count


@pytest.mark.asyncio
async def test_cancelled_environment_thread_does_not_mutate_published_call(delivered_telemetry):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finish = Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def env_step():
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)
        return 1.0

    async def produce():
        with rollout.observe_rollout_call(step=3, mode="async", enabled=True):
            return await rollout.run_environment(executor, env_step)

    task = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        published = list(delivered_telemetry.flush())
    finally:
        finish.set()
        await asyncio.to_thread(executor.shutdown, wait=True)

    assert delivered_telemetry.flush() == published
    assert [row["attributes"]["outcome"] for row in delivered_telemetry.select("rollout_call")] == ["cancelled"]
    assert {row["attributes"]["wait"] for row in delivered_telemetry.select("rollout_waits")} == {"env_await"}


def test_training_metrics_keep_selected_finite_values_and_count_nonfinite_ones(delivered_telemetry):
    record_training_metrics(
        {
            "policy/entropy": 1.25,
            "async/staleness_max": 2,
            "eval/all/avg_score": 0.625,
            "environment/exact_n10": 0.75,
            "policy/loss": float("nan"),
            "policy/grad_norm": float("inf"),
            "policy/details": [1, 2],
            "unselected/value": 99.0,
        },
        step=8,
        kind="train",
    )

    values = delivered_telemetry.select("training_metric_value", step="8", payload_kind="train")
    assert {row["attributes"]["metric"]: row["value"] for row in values} == {
        "policy/entropy": 1.25,
        "async/staleness_max": 2.0,
        "eval/all/avg_score": 0.625,
        "environment/exact_n10": 0.75,
    }
    nonfinite = delivered_telemetry.select("training_nonfinite_values", step="8")
    assert {row["attributes"]["metric"] for row in nonfinite} == {"policy/loss", "policy/grad_norm"}
