# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from levanter.callbacks import StepInfo
from levanter.callbacks.labeled_eval import (
    LabeledLmEvalConfig,
    LabeledLmEvalDatasetConfig,
    add_labeled_lm_eval_callbacks,
)
from levanter.compute_dtype import cast_model_for_compute
from levanter.data.text.datasets import LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.trace_chat import TraceChatEvaluationFormat
from levanter.store.cache import SerialCacheWriter
from levanter.tracker import current_tracker
from levanter.tracker.tracker import DictTracker
from levanter.testing.helpers import use_test_mesh
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig


class Router(eqx.Module):
    router_bias: jax.Array | hax.NamedArray
    weight: jax.Array


def router(named=False):
    bias = jnp.asarray([0.1234567, -0.7654321], dtype=jnp.float32)
    if named:
        bias = hax.named(bias, hax.Axis("expert", 2))
    return Router(bias, jnp.asarray(0.1234567, dtype=jnp.float32))


@pytest.mark.parametrize("named", [False, True])
def test_compute_override_preserves_original_bits(named):
    model = tuple(router(named) for _ in range(26))
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")
    compute = eqx.filter_jit(lambda model: cast_model_for_compute(model, policy, {"router_bias": "float32"}))(model)
    for master, view in zip(model, compute, strict=True):
        original = master.router_bias.array if named else master.router_bias
        preserved = view.router_bias.array if named else view.router_bias
        np.testing.assert_array_equal(original, preserved)
        assert preserved.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        assert np.any(np.asarray(original) != np.asarray(original.astype(jnp.bfloat16).astype(jnp.float32)))


def test_trainer_loss_preserves_router_bias_and_gradient():
    policy = jmp.get_policy("p=f32,c=bf16,o=f32")
    model = router()

    def loss(view):
        assert view.router_bias.dtype == jnp.float32
        assert view.weight.dtype == jnp.bfloat16
        return jnp.sum(view.router_bias) + view.weight.astype(jnp.float32), {}

    config = TrainerConfig(
        id="compute-view-test",
        mp=policy,
        compute_dtype_overrides={"router_bias": "float32"},
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
    )
    trainer = Trainer(config, optax.sgd(0.01), loss, add_default_hooks=False)
    actual, _ = trainer.loss_fn(model)
    expected = jnp.sum(model.router_bias) + model.weight.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_array_equal(actual, expected)
    gradient = eqx.filter_grad(lambda model: trainer.loss_fn(model)[0])(model)
    np.testing.assert_array_equal(gradient.router_bias, jnp.ones_like(model.router_bias))
    np.testing.assert_array_equal(model.router_bias, router().router_bias)


class RouterLossModel(Router):
    Pos: hax.Axis = eqx.field(static=True)

    def compute_next_token_loss(self, batch, *, reduction, reduction_axis):
        value = jnp.sum(self.router_bias) + self.weight.astype(jnp.float32)
        return hax.ones(batch.tokens.axes) * value


def test_labeled_eval_callback_preserves_compute_override(tmp_path, local_gpt2_tokenizer):
    Pos = hax.Axis("position", 4)
    EvalBatch = hax.Axis("batch", len(jax.devices()))
    model = RouterLossModel(router().router_bias, router().weight, Pos)
    cache_dir = str(tmp_path / "cache")
    example = {"input_ids": np.array([1, 2, 3], dtype=np.int32), "loss_labels": np.ones(3, dtype=np.int32)}
    with SerialCacheWriter(cache_dir, example) as writer:
        writer.write_batch([example] * EvalBatch.size)
    source = tmp_path / "trace.jsonl"
    source.write_text("{}\n")
    config = TrainerConfig(
        id="labeled-compute-view-test",
        mp=jmp.get_policy("p=f32,c=bf16,o=f32"),
        compute_dtype_overrides={"router_bias": "float32"},
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
    )
    trainer = Trainer(config, optax.sgd(0.01), lambda model: (jnp.sum(model.router_bias), {}), add_default_hooks=False)
    tracker = DictTracker()
    with use_test_mesh(tensor_parallelism=1) as mesh:
        add_labeled_lm_eval_callbacks(
            trainer,
            labeled_eval_config=LabeledLmEvalConfig(
                datasets={
                    "trace": LabeledLmEvalDatasetConfig(
                        source=UrlDatasetSourceConfig(validation_urls=[str(source)]),
                        split="validation",
                        cache_dir=cache_dir,
                        trace_format=TraceChatEvaluationFormat(
                            pack=False,
                            chat_template="{% for message in messages %}{% generation %}{{ message.content }}{% endgeneration %}{% endfor %}",
                        ),
                    )
                },
                eval_model=False,
            ),
            data_config=LmDataConfig(),
            trainer_config=config,
            EvalBatch=EvalBatch,
            Pos=Pos,
            tokenizer=local_gpt2_tokenizer,
            device_mesh=mesh,
            axis_mapping={EvalBatch.name: "data"},
            max_eval_examples_per_dataset=None,
        )
        with current_tracker(tracker):
            trainer.run_hooks(
                StepInfo(state=SimpleNamespace(model=model, step=1), loss=0.0, step_duration=0.0), force=True
            )
    expected = jnp.sum(model.router_bias) + model.weight.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_allclose(
        tracker.metrics["step_0"]["labeled_eval/trace/assistant_text/loss"], expected, rtol=1e-6
    )
