# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import optax
import pytest

from levanter.checkpoint import CheckpointerConfig, load_checkpoint
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig


class LinearModel(eqx.Module):
    weight: hax.NamedArray


def loss(model, batch, *, key=None):
    return hax.sum(batch * model.weight)


@pytest.mark.parametrize("save_final", [False, True])
def test_final_checkpoint_policy_preserves_other_final_hooks(tmp_path, save_final):
    batch_axis = hax.Axis("batch", jax.device_count())
    embed_axis = hax.Axis("embed", 4)
    model = LinearModel(hax.ones(embed_axis))
    batch = hax.ones((batch_axis, embed_axis))
    config = TrainerConfig(
        id="final-save",
        train_batch_size=batch_axis.size,
        num_train_steps=2,
        tracker=NoopConfig(),
        checkpointer=CheckpointerConfig(
            base_path=str(tmp_path), append_run_id_to_base_path=False, save_interval=None, keep=None
        ),
        save_final_checkpoint=save_final,
        load_checkpoint=False,
    )
    finalized = []
    with Trainer(config, optax.sgd(0.1), loss) as trainer:
        trainer.add_hook(lambda info: finalized.append(int(info.step)), every=3)
        state = trainer.initial_state(jax.random.PRNGKey(0), model=model)
        result = trainer.train(state, [batch, batch])
    assert finalized == [1]  # Hooks name the completed update with a zero-based index.
    checkpoint = tmp_path / "step-1"
    if save_final:
        restored = load_checkpoint(
            result.state.saveable_state,
            checkpoint,
            mesh=config.device_mesh,
            axis_mapping=config.parameter_axis_mapping,
        )
        assert int(restored.step) == 2
        assert hax.all(restored.model.weight == result.state.model.weight).scalar()
    else:
        assert not list(tmp_path.glob("step-*"))
