from typing import cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.tokenizers import MarinTokenizer
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.post_training.identity_context_distill.precision_probe import (
    adam_gradient_witness,
    gated_zero_lr_step,
    synthetic_lr0_batch,
)


def test_synthetic_batch_shifts_native_mask_without_padding_loss(monkeypatch):
    def build_processor(format_config, tokenizer):
        del tokenizer
        assert not format_config.pack and format_config.slice_strategy == "raise"

        def process(rows):
            assert len(rows) == 8
            assert all(row["chat_template_kwargs"]["enable_thinking"] is False for row in rows)
            return [{"input_ids": [128000, 11, 12, 13], "assistant_masks": [0, 0, 1, 1]} for _ in rows]

        return process

    monkeypatch.setattr(ChatLmDatasetFormat, "build_preprocessor", build_processor)
    batch = synthetic_lr0_batch(cast(MarinTokenizer, object()))
    assert batch.tokens.array.shape == (8, 8192)
    np.testing.assert_array_equal(batch.loss_weight.array[:, :4], np.tile([0, 1, 1, 0], (8, 1)))
    assert np.all(np.asarray(batch.loss_weight.array[:, 4:]) == 0)


class TinyModel(eqx.Module):
    weight: jax.Array
    other: jax.Array


def test_lr0_gate_precedes_real_backward_and_update():
    config = TrainerConfig(
        id="lr0-test",
        tracker=(),
        require_accelerator=False,
        mesh=MeshConfig(),
        train_batch_size=1,
        per_device_parallelism=1,
        load_checkpoint=False,
    )
    optimizer = AdamConfig(learning_rate=0.0, warmup=0, weight_decay=0.0).build(1)
    axes = (hax.Axis("batch", 1), hax.Axis("position", 2))
    batch = LmExample(hax.named(jnp.ones((1, 2), dtype=jnp.int32), axes), hax.ones(axes))

    def loss(model, batch, *, key=None):
        del batch, key
        return jnp.square(model.weight) + jnp.square(model.other), {}

    model = TinyModel(jnp.asarray(0.1234567, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32))
    initial_weight = np.asarray(model.weight).copy()
    initial_other = np.asarray(model.other).copy()
    calls = []

    def masters(state, step):
        assert int(state.step) == step
        calls.append(f"masters-{step}")
        np.testing.assert_array_equal(state.model.weight, initial_weight)
        np.testing.assert_array_equal(state.model.other, initial_other)

    def passed_gate(trainer, state):
        del trainer
        assert int(state.step) == 0
        calls.append("gate")

    with Trainer(config, optimizer, loss, add_default_hooks=False) as trainer:
        state = trainer.initial_state(jax.random.PRNGKey(42), model=model)

        def failed_gate(trainer, state):
            del trainer, state
            raise ValueError("parity failed")

        with pytest.raises(ValueError, match="parity failed"):
            gated_zero_lr_step(trainer, state, batch, failed_gate, masters)
        assert int(state.step) == 0
        assert calls == ["masters-0"]
        calls.clear()
        updated, measured_loss, witness = gated_zero_lr_step(trainer, state, batch, passed_gate, masters)
        assert calls == ["masters-0", "gate", "masters-1"]
        assert measured_loss > 0 and witness > 0
        assert int(updated.step) == 1
        np.testing.assert_array_equal(updated.model.weight, initial_weight)
        with pytest.raises(ValueError, match="start fresh"):
            gated_zero_lr_step(trainer, updated, batch, passed_gate, masters)

        moment_paths = [
            (path, leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(updated.opt_state)[0]
            if any(isinstance(key, jax.tree_util.GetAttrKey) and key.name == "mu" for key in path)
        ]
        assert len(moment_paths) == 2
        assert witness == max(float(jnp.max(jnp.abs(moment))) for _, moment in moment_paths)
        for invalid in (float("nan"), float("inf")):
            damaged_optimizer = jax.tree_util.tree_map_with_path(
                lambda path, leaf, value=invalid: jnp.asarray(value) if path == moment_paths[1][0] else leaf,
                updated.opt_state,
            )
            damaged = eqx.tree_at(lambda result: result.opt_state, updated, damaged_optimizer)
            with pytest.raises(ValueError, match="Nonfinite Adam"):
                adam_gradient_witness(damaged)

        def reject_post_update(state, step):
            masters(state, step)
            if step == 1:
                raise ValueError("post-update master drift")

        fresh = trainer.initial_state(
            jax.random.PRNGKey(43),
            model=TinyModel(jnp.asarray(0.1234567, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32)),
        )
        with pytest.raises(ValueError, match="post-update master drift"):
            gated_zero_lr_step(trainer, fresh, batch, passed_gate, reject_post_update)
