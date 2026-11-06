# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
from haliax.nn.scan import ScanCheckpointPolicy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.models.llama_native import NativeLlamaSharding
from levanter.checkpoint import CheckpointerConfig, discover_latest_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import next_token_ids
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.tracker import NoopTracker, current_tracker
from levanter.utils.mesh import MeshConfig
from levanter.models.lm_model import LmExample


pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_native_model_matches_named_hf_weights_loss_and_all_gradients(kv_heads):
    _assert_named_native_loss_and_gradients(kv_heads, gradient_checkpointing=False)


def test_native_gqa_checkpointed_scan_matches_named_loss_and_all_gradients():
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=True)


def test_native_gqa_full_named_checkpoint_policy_matches_loss_and_all_gradients():
    policy = ScanCheckpointPolicy(save_carries=False, save_inputs=True, prevent_cse=True, nested=2)
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=policy)


def _assert_named_native_loss_and_gradients(kv_heads, *, gradient_checkpointing):
    config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=kv_heads,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
        gradient_checkpointing=gradient_checkpointing,
    )
    native_config = NativeLlamaLmConfig(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    )
    Vocab, Batch, Pos = hax.Axis("vocab", 32), hax.Axis("batch", 2), config.max_Pos
    tokens = hax.named((jnp.arange(16).reshape(2, 8) * 3) % 32, (Batch, Pos))
    segments = hax.named(jnp.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]]), (Batch, Pos))
    weights = hax.named(jnp.ones((2, 8)).at[:, -1].set(0).at[0, 2].set(0.5), (Batch, Pos))
    example = LmExample(tokens, weights, AttentionMask.causal().with_segment_ids(segments))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        named = LlamaLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))
        source = named.to_state_dict()
        native = native_config.build(Vocab, key=jax.random.PRNGKey(1)).from_state_dict(source)
        for name, value in native.to_state_dict().items():
            np.testing.assert_array_equal(value, source[name].reshape(value.shape))
        losses = []
        gradients = []
        for model in (named, native):
            loss, gradient = eqx.filter_jit(
                eqx.filter_value_and_grad(lambda m: m.compute_next_token_loss(example).scalar())
            )(model)
            losses.append(loss)
            gradients.append(gradient.to_state_dict())
        np.testing.assert_allclose(losses[1], losses[0], atol=1e-5, rtol=1e-5)
        assert gradients[0].keys() == gradients[1].keys()
        for name, value in gradients[1].items():
            np.testing.assert_allclose(
                value, gradients[0][name].reshape(value.shape), atol=1e-5, rtol=1e-5, err_msg=name
            )


def test_native_public_trainer_eval_step_checkpoint_resume_preserves_explicit_state(tmp_path):
    if len(jax.devices()) < 4:
        pytest.skip("Requires four devices for DP2/TP2 placement")
    config = NativeLlamaLmConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=8,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    mapping = {"batch": "data", "mlp": "model", "heads": "model", "kv_head": "model", "vocab": "model"}
    trainer_config = TrainerConfig(
        id="native-explicit",
        train_batch_size=4,
        num_train_steps=1,
        per_device_parallelism=-1,
        require_accelerator=False,
        load_checkpoint=False,
        mesh=MeshConfig(
            axes={"data": 2, "model": 2},
            param_mapping={k: v for k, v in mapping.items() if k != "batch"},
            compute_mapping=mapping,
        ),
        checkpointer=CheckpointerConfig(
            base_path=str(tmp_path / "checkpoints"), append_run_id_to_base_path=False, save_interval=None, keep=None
        ),
        log_dir=tmp_path / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    sharding = NativeLlamaSharding(batch_axes=("data",), model_axis="model")
    layout = ExplicitTrainingLayout(mesh, P("data", None))
    examples = []
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        for row in range(4):
            tokens = hax.named((jnp.arange(8) + row * 3) % 19, config.max_Pos)
            weight = hax.named(jnp.ones(8).at[-1].set(0), config.max_Pos)
            segments = hax.named(jnp.array([0, 0, 0, 0, 1, 1, 1, 1]), config.max_Pos)
            examples.append(LmExample(tokens, weight, AttentionMask.causal().with_segment_ids(segments)))
    dataset = ListAsyncDataset(examples)

    def loss(model, batch, *, key=None):
        return model.compute_next_token_loss(batch, key=key)

    def eval_loss(model, batch):
        return (
            model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array,
            batch.loss_weight.array,
            next_token_ids(model.Pos, batch.tokens).array,
        )

    evaluator = TaggedEvaluator(
        hax.Axis("batch", 4),
        [(dataset, ["held"])],
        eval_loss,
        tokenizer=None,
        device_mesh=mesh,
        axis_mapping={"batch": "data"},
        shuffle=False,
    )
    with (
        current_tracker(NoopTracker()),
        Trainer(
            trainer_config,
            optax.adam(1e-3),
            loss,
            explicit_layout=layout,
            checkpoint_metadata=lambda model: model.checkpoint_metadata(),
        ) as trainer,
    ):
        initial = trainer.initial_state(
            jax.random.PRNGKey(2),
            model_init=lambda: config.build(hax.Axis("vocab", 32), key=jax.random.PRNGKey(1), sharding=sharding),
        )
        initial_loss = evaluator.evaluate(initial.model).micro_avg_loss
        initial_key = np.array(initial.training_key)
        info = trainer.train(initial, trainer.data_loader(dataset))
        after_loss = evaluator.evaluate(info.state.model).micro_avg_loss
        expected_arrays = [np.array(x) for x in jax.tree.leaves(eqx.filter(info.state, eqx.is_array))]
        assert int(info.state.step) == 1
        assert int(info.state.opt_state[0].count) == 1
        assert not np.array_equal(initial_key, np.asarray(info.state.training_key))
        assert np.isfinite(initial_loss) and np.isfinite(after_loss)
        assert initial_loss != after_loss
        for leaf in jax.tree.leaves(info.state.model):
            if eqx.is_array(leaf):
                assert leaf.sharding.mesh == mesh

    checkpoint = discover_latest_checkpoint(str(tmp_path / "checkpoints"))
    restored_config = dataclasses.replace(trainer_config, load_checkpoint=True, load_checkpoint_path=checkpoint)
    with (
        current_tracker(NoopTracker()),
        Trainer(
            restored_config,
            optax.adam(1e-3),
            loss,
            explicit_layout=layout,
            add_default_hooks=False,
        ) as trainer,
    ):
        vocab_size = config.checkpoint_vocab_size(trainer.explicit_checkpoint_to_load.metadata)
        assert vocab_size == 32
        restored = trainer.initial_state(
            jax.random.PRNGKey(99),
            model_init=lambda: config.build(
                hax.Axis("vocab", vocab_size), key=jax.random.PRNGKey(98), sharding=sharding
            ),
        )
        actual_arrays = jax.tree.leaves(eqx.filter(restored, eqx.is_array))
        assert len(actual_arrays) == len(expected_arrays)
        for actual, expected in zip(actual_arrays, expected_arrays, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert evaluator.evaluate(restored.model).micro_avg_loss == after_loss
        # The checkpoint's full vocabulary survives a smaller tokenizer vocabulary.
        loaded = load_levanter_checkpoint(
            config,
            checkpoint,
            Vocab=hax.Axis("vocab", 19),
            axis_mapping=trainer.parameter_axis_mapping,
            compute_axis_mapping=trainer.compute_axis_mapping,
            batch_axis_name="batch",
            key=jax.random.PRNGKey(3),
        )
        assert loaded.Vocab.size == 32
        assert evaluator.evaluate(loaded).micro_avg_loss == after_loss
