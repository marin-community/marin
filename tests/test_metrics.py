# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import optax
import pytest

from levanter.callbacks import eval_loss_loop
from levanter.metrics import (
    Metric,
    ReductionType,
    auto_metric_from_name,
    fold,
    unwrap_metrics,
)
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig, WrappedLossFunction

Batch = hax.Axis("batch", size=4)
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

    if has_metrics:
        assert jnp.allclose(avg_metrics["metric_a"], expected_loss * 2, atol=1e-5)
        assert jnp.allclose(avg_metrics["metric_b"], expected_loss + 10, atol=1e-5)
    else:
        assert len(avg_metrics) == 0


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

    config = TrainerConfig(
        tracker=NoopConfig(),
        seed=42,
        num_train_steps=10,
        train_batch_size=Batch.size,
        per_device_parallelism=per_device_parallelism,
        fsdp_axis=None,
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

    config = TrainerConfig(
        tracker=NoopConfig(),
        seed=42,
        num_train_steps=10,
        train_batch_size=Batch.size,
        per_device_parallelism=2,
        fsdp_axis=None,
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
