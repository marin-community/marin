# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import MagicMock

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import optax
import pytest

import levanter.tracker as tracker_mod
from levanter.callbacks import eval_loss_loop
from levanter.callbacks._metrics import compute_instant_throughput, log_step_info
from levanter.metrics import (
    Metric,
    ReductionType,
    auto_metric_from_name,
    fold,
    unwrap_metrics,
)
from levanter.schedule import BatchSchedule, ScheduleStep
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig, WrappedLossFunction

# Use a batch size that remains divisible by the data-parallel axis on multi-device setups.
Embed = hax.Axis("embed", size=8)


@pytest.mark.parametrize(
    "reduction,value,count,expected",
    [
        (ReductionType.MEAN, 30.0, 2.0, 15.0),
        (ReductionType.SUM, 30.0, 0.0, 30.0),
        (ReductionType.MAX, 20.0, 0.0, 20.0),
        (ReductionType.MIN, 10.0, 0.0, 10.0),
        (ReductionType.LAST, 20.0, 0.0, 20.0),
    ],
    ids=["mean", "sum", "max", "min", "last"],
)
def test_metric_value(reduction, value, count, expected):
    """Metric.value() applies correct reduction."""
    m = Metric(_value=value, _count=count, reduction=reduction)
    assert jnp.allclose(m.value(), expected)


def test_metric_fold():
    """fold() combines metrics correctly."""
    m1 = Metric.from_value(10.0, ReductionType.MEAN)
    m2 = Metric.from_value(20.0, ReductionType.MEAN)
    result = fold(m1, m2)

    assert result.reduction == ReductionType.MEAN
    assert jnp.allclose(result.value(), 15.0)  # (10 + 20) / 2 = 15


@pytest.mark.parametrize(
    "reduction",
    [ReductionType.MEAN, ReductionType.SUM, ReductionType.MAX, ReductionType.MIN],
    ids=["mean", "sum", "max", "min"],
)
def test_metric_fold_associativity(reduction):
    """fold is associative for all reduction types."""
    m1 = Metric.from_value(10.0, reduction)
    m2 = Metric.from_value(20.0, reduction)
    m3 = Metric.from_value(30.0, reduction)

    # (m1 + m2) + m3
    result1 = fold(fold(m1, m2), m3)

    # m1 + (m2 + m3)
    result2 = fold(m1, fold(m2, m3))

    assert jnp.allclose(result1.value(), result2.value())


def test_metric_float_conversion():
    """Metrics support float() conversion."""
    m = Metric.from_value(42.0, ReductionType.MEAN)
    assert float(m) == 42.0


def test_metric_pytree():
    """Metrics are JAX pytrees."""
    m = Metric(_value=30.0, _count=2.0, reduction=ReductionType.MEAN)
    flat, treedef = jax.tree_util.tree_flatten(m)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    assert jnp.allclose(reconstructed.value(), 15.0)  # 30 / 2 = 15
    assert reconstructed.reduction == ReductionType.MEAN


def test_metric_jit():
    """Metrics work through JIT."""

    @jax.jit
    def fold_metrics_jit(m1, m2):
        return fold(m1, m2)

    m1 = Metric.from_value(10.0, ReductionType.SUM)
    m2 = Metric.from_value(20.0, ReductionType.SUM)
    result = fold_metrics_jit(m1, m2)

    assert jnp.allclose(result.value(), 30.0)


@pytest.mark.parametrize(
    "name,expected_reduction",
    [
        ("num_tokens", ReductionType.SUM),
        ("token_count", ReductionType.SUM),
        ("total_examples", ReductionType.SUM),
        ("correct_sum", ReductionType.SUM),
        ("accuracy", ReductionType.MEAN),
        ("loss", ReductionType.MEAN),
        ("perplexity", ReductionType.MEAN),
        ("max_logit", ReductionType.MAX),
        ("gradient_max", ReductionType.MAX),
        ("min_loss", ReductionType.MIN),
        ("loss_min", ReductionType.MIN),
        ("learning_rate", ReductionType.LAST),
    ],
    ids=[
        "num_tokens",
        "token_count",
        "total_examples",
        "correct_sum",
        "accuracy",
        "loss",
        "perplexity",
        "max_logit",
        "gradient_max",
        "min_loss",
        "loss_min",
        "learning_rate",
    ],
)
def test_auto_metric_inference(name, expected_reduction):
    """auto_metric_from_name infers correct reduction type."""
    m = auto_metric_from_name(name, 42.0)
    assert m.reduction == expected_reduction


def test_unwrap_metrics():
    """unwrap_metrics extracts values from all Metrics in a pytree."""
    tree = {
        "accuracy": Metric.from_value(0.95, ReductionType.MEAN),
        "num_tokens": Metric.from_value(1024, ReductionType.SUM),
        "nested": {"max_logit": Metric.from_value(5.0, ReductionType.MAX)},
        "plain_value": 42.0,
    }

    result = unwrap_metrics(tree)

    assert jnp.allclose(result["accuracy"], 0.95)
    assert jnp.allclose(result["num_tokens"], 1024)
    assert jnp.allclose(result["nested"]["max_logit"], 5.0)
    assert result["plain_value"] == 42.0


def test_compute_instant_throughput_rates_and_mfu():
    """Rates, model FLOPs/sec, and MFU all derive from one step's duration."""
    t = compute_instant_throughput(
        batch_size=256,
        step_duration=0.5,
        tokens_per_example=4096,
        flops_per_example=1e12,
        theoretical_flops=1e15,
    )
    assert t.examples_per_second == 512.0  # 256 / 0.5
    assert t.tokens_per_second == 4096 * 512.0
    assert t.model_flops_per_second == 1e12 / 0.5 * 256
    # mfu = model_flops_per_second / theoretical_flops * 100
    assert jnp.allclose(t.mfu, (1e12 / 0.5 * 256) / 1e15 * 100.0)


def test_compute_instant_throughput_without_flops_has_no_mfu():
    t = compute_instant_throughput(batch_size=128, step_duration=2.0, tokens_per_example=1024)
    assert t.examples_per_second == 64.0
    assert t.tokens_per_second == 1024 * 64.0
    assert t.model_flops_per_second is None
    assert t.mfu is None


def test_compute_instant_throughput_zero_duration_is_empty():
    """A zero-duration step yields no rates instead of dividing by zero."""
    t = compute_instant_throughput(batch_size=128, step_duration=0.0, tokens_per_example=1024, flops_per_example=1e12)
    assert t.examples_per_second is None
    assert t.tokens_per_second is None
    assert t.model_flops_per_second is None
    assert t.mfu is None


class SimpleModel(eqx.Module):
    weight: hax.NamedArray

    @staticmethod
    def init(key):
        return SimpleModel(hax.random.normal(key, (Embed,)))


def simple_loss_fn(model, batch, key=None):
    """Loss function returning scalar only."""
    return hax.sum(batch * model.weight)


def metrics_loss_fn(model, batch, key=None):
    """Loss function returning (loss, metrics) tuple."""
    loss = hax.sum(batch * model.weight)
    metrics = {"accuracy": jnp.array(0.95), "perplexity": jnp.array(2.5)}
    return loss, metrics


def varied_metrics_loss_fn(model, batch, key=None):
    """Loss function with different metric types for aggregation testing."""
    loss = hax.sum(batch * model.weight)
    metrics = {
        "accuracy": jnp.array(0.95),
        "num_tokens": jnp.array(128.0),
        "max_logit": jnp.array(5.0),
    }
    return loss, metrics


def test_wrapped_loss_function_invalid_metrics():
    """WrappedLossFunction validates metrics must be a dict."""
    Batch = hax.Axis("batch", size=4 * max(1, jax.device_count()))
    model = hax.random.normal(jax.random.PRNGKey(0), (Embed,))
    batch = hax.random.normal(jax.random.PRNGKey(1), (Batch, Embed))

    def bad_loss(model, batch):
        return hax.sum(batch), [1, 2, 3]

    wrapped = WrappedLossFunction(bad_loss, jmp.get_policy("f32"), {})

    with pytest.raises(ValueError, match="Expected metrics to be dict"):
        wrapped(model, batch)


@pytest.mark.parametrize(
    "has_metrics,max_batches",
    [
        (False, None),
        (True, None),
        (True, 2),
    ],
    ids=["no-metrics-all", "with-metrics-all", "with-metrics-limited"],
)
def test_eval_loss_loop(has_metrics, max_batches):
    """eval_loss_loop handles metrics and max_batches correctly."""
    model = hax.random.normal(jax.random.PRNGKey(0), (Embed,))
    Batch = hax.Axis("batch", size=4 * max(1, jax.device_count()))

    def raw_loss_fn(model, batch):
        val = hax.mean(batch["value"])
        if has_metrics:
            return val, {"metric_a": val * 2, "metric_b": val + 10}
        return val

    wrapped_loss_fn = WrappedLossFunction(raw_loss_fn, jmp.get_policy("f32"), {})

    dataset = [
        {"value": hax.full((Batch, Embed), 1.0)},
        {"value": hax.full((Batch, Embed), 2.0)},
        {"value": hax.full((Batch, Embed), 3.0)},
    ]

    avg_loss, avg_metrics = eval_loss_loop(wrapped_loss_fn, model, dataset, max_batches=max_batches)

    n_batches = min(len(dataset), max_batches) if max_batches else len(dataset)
    expected_loss = sum(range(1, n_batches + 1)) / n_batches

    assert jnp.allclose(avg_loss, expected_loss, atol=1e-5)
    assert avg_metrics["eval/timing/load_time"] >= 0.0
    assert avg_metrics["eval/timing/loss_time"] >= 0.0
    assert avg_metrics["eval/timing/num_batches"] == float(n_batches)

    metric_keys = {key for key in avg_metrics if not key.startswith("eval/timing/")}

    if has_metrics:
        assert jnp.allclose(avg_metrics["metric_a"], expected_loss * 2, atol=1e-5)
        assert jnp.allclose(avg_metrics["metric_b"], expected_loss + 10, atol=1e-5)
    else:
        assert metric_keys == set()


@pytest.mark.parametrize(
    "loss_fn,per_device_parallelism,expected_metrics",
    [
        (simple_loss_fn, 1, None),
        (metrics_loss_fn, 1, {"train/accuracy": 0.95, "train/perplexity": 2.5}),
        (metrics_loss_fn, 2, {"train/accuracy": 0.95, "train/perplexity": 2.5}),
    ],
    ids=["scalar-only", "with-metrics", "microbatching"],
)
def test_trainer_train_step(loss_fn, per_device_parallelism, expected_metrics):
    """Trainer.train_step works with various loss functions and microbatching."""
    model = SimpleModel.init(jax.random.PRNGKey(0))
    Batch = hax.Axis("batch", size=4 * max(1, jax.device_count()))

    config = TrainerConfig(
        tracker=NoopConfig(),
        seed=42,
        num_train_steps=10,
        train_batch_size=Batch.size,
        per_device_parallelism=per_device_parallelism,
        id="test_run",
    )

    optimizer = optax.sgd(0.01)
    trainer = Trainer(config, optimizer, loss_fn, add_default_hooks=False)

    logged_metrics = {}
    trainer.tracker.log = lambda metrics, step=None, commit=None: logged_metrics.update(metrics)

    with trainer:
        batch = hax.random.normal(jax.random.PRNGKey(1), (Batch, Embed))
        state = trainer.initial_state(jax.random.PRNGKey(0), model=model)
        trainer.train_step(state, batch)

        if expected_metrics is None:
            train_metrics = {k: v for k, v in logged_metrics.items() if k.startswith("train/")}
            assert len(train_metrics) == 0
        else:
            for key, expected_val in expected_metrics.items():
                assert key in logged_metrics
                assert jnp.allclose(logged_metrics[key], expected_val)


def test_microbatching_metric_aggregation():
    """Microbatching correctly aggregates different metric types."""
    model = SimpleModel.init(jax.random.PRNGKey(0))
    Batch = hax.Axis("batch", size=4 * max(1, jax.device_count()))

    config = TrainerConfig(
        tracker=NoopConfig(),
        seed=42,
        num_train_steps=10,
        train_batch_size=Batch.size,
        per_device_parallelism=2,
        id="test_run",
    )

    optimizer = optax.sgd(0.01)
    trainer = Trainer(config, optimizer, varied_metrics_loss_fn, add_default_hooks=False)

    logged_metrics = {}
    trainer.tracker.log = lambda metrics, step=None, commit=None: logged_metrics.update(metrics)

    with trainer:
        batch = hax.random.normal(jax.random.PRNGKey(1), (Batch, Embed))
        state = trainer.initial_state(jax.random.PRNGKey(0), model=model)
        trainer.train_step(state, batch)

        # With 2 microbatches:
        # - accuracy (MEAN): (0.95 + 0.95) / 2 = 0.95
        # - num_tokens (SUM): 128 + 128 = 256
        # - max_logit (MAX): max(5.0, 5.0) = 5.0
        assert jnp.allclose(logged_metrics["train/accuracy"], 0.95)
        assert jnp.allclose(logged_metrics["train/num_tokens"], 256.0)
        assert jnp.allclose(logged_metrics["train/max_logit"], 5.0)


def _make_step_info(step_number: int, loss: float = 1.0) -> MagicMock:
    """Create a minimal StepInfo mock for testing log_step_info."""
    info = MagicMock()
    info.step = step_number
    info.loss = loss
    info.opt_state = {}
    return info


def _patch_tracker(logged: dict):
    """Context manager that redirects tracker.log into ``logged``."""

    @contextlib.contextmanager
    def _ctx():
        orig_log = tracker_mod.log
        orig_log_optim = tracker_mod.log_optimizer_hyperparams
        tracker_mod.log = lambda m, step=None, commit=None: logged.update(m)
        tracker_mod.log_optimizer_hyperparams = lambda *a, **kw: None
        try:
            yield
        finally:
            tracker_mod.log = orig_log
            tracker_mod.log_optimizer_hyperparams = orig_log_optim

    return _ctx()


def test_log_step_info_token_progress_uniform_batch():
    """With a uniform batch schedule, progress equals completed_steps / total_steps."""
    schedule = BatchSchedule(32)
    total_steps = 100
    logged: dict = {}

    cb = log_step_info(total_steps, schedule)
    with _patch_tracker(logged):
        cb(_make_step_info(49))  # step 49 done → 50 steps completed

    # token-based: global_data_offset(50) / global_data_offset(100) = 50*32 / 100*32 = 0.5
    assert abs(logged["run_progress"] - 0.5) < 1e-9


def test_log_step_info_token_progress_variable_batch():
    """Token-based progress is batch-size independent when the schedule changes mid-run."""
    # Steps 0–49: batch 32  → 50 * 32 = 1600 examples
    # Steps 50–99: batch 64 → 50 * 64 = 3200 examples
    # Total: 4800 examples
    schedule = BatchSchedule([ScheduleStep(0, 32), ScheduleStep(50, 64)])
    total_steps = 100
    logged: dict = {}

    cb = log_step_info(total_steps, schedule)
    with _patch_tracker(logged):
        # After step 74 (0-indexed): examples = 1600 + 25*64 = 3200; total = 4800
        cb(_make_step_info(74))

    expected = 3200 / 4800
    assert abs(logged["run_progress"] - expected) < 1e-9
    # Also confirm it differs from naive step ratio
    assert abs(logged["run_progress"] - 74 / 100) > 0.01


def test_log_step_info_falls_back_to_step_progress_without_schedule():
    """Without a batch schedule, run_progress falls back to step ratio."""
    total_steps = 100
    logged: dict = {}

    cb = log_step_info(total_steps)
    with _patch_tracker(logged):
        cb(_make_step_info(25))

    assert abs(logged["run_progress"] - 0.25) < 1e-9
