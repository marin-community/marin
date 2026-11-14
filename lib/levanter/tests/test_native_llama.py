# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
from haliax.nn.scan import ScanCheckpointPolicy
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
import optax
from jax.sharding import AxisType, Mesh, PartitionSpec as P

from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.llama_native_lm import NativeLlamaLmConfig
from levanter.models.llama_native import NativeLlamaAttention, NativeLlamaConfig, NativeLlamaSharding
from levanter.checkpoint import CheckpointerConfig, discover_latest_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug
from levanter.eval import TaggedEvaluator
from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageTable
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import next_token_ids
from levanter.trainer import ExplicitTrainingLayout, Trainer, TrainerConfig
from levanter.tracker import NoopTracker, current_tracker
from levanter.utils.mesh import MeshConfig
from levanter.models.lm_model import LmExample


pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("compute_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("upcast_attn", [False, True])
def test_native_model_matches_named_hf_weights_loss_and_all_gradients(kv_heads, compute_dtype, upcast_attn):
    _assert_named_native_loss_and_gradients(
        kv_heads, gradient_checkpointing=False, compute_dtype=compute_dtype, upcast_attn=upcast_attn
    )


def test_native_bfloat16_reference_attention_upcasts_before_query_scaling():
    # An eight-wide head exposes BF16 rounding in the inverse square root scale.
    config = NativeLlamaConfig(vocab_size=32, hidden_dim=8, num_heads=1, num_kv_heads=1, upcast_attn=True)
    layout = NativeLlamaSharding()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        attention = NativeLlamaAttention.init(config, layout, key=jax.random.PRNGKey(0))
        policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)
        attention = policy.cast_to_compute(attention)
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 8, config.hidden_dim), dtype=jnp.bfloat16)
        positions = jnp.arange(x.shape[1])[None, :]
        mask = positions[:, :, None] >= positions[:, None, :]

        q, k, v = attention.project_qkv(x, positions, sharding=layout)
        q, k = q[:, :, 0].astype(jnp.float32), k[:, :, 0].astype(jnp.float32)
        scores = (q @ k.swapaxes(-1, -2)) * (config.actual_head_dim**-0.5)
        weights = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1).astype(v.dtype)
        expected = attention.project_output((weights @ v[:, :, 0])[:, :, None], dtype=x.dtype, sharding=layout)
        actual = eqx.filter_jit(attention, compiler_options={"xla_allow_excess_precision": False})(x, mask, positions)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_native_gqa_checkpointed_scan_matches_named_loss_and_all_gradients():
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=True)


def test_native_gqa_full_named_checkpoint_policy_matches_loss_and_all_gradients():
    policy = ScanCheckpointPolicy(save_carries=False, save_inputs=True, prevent_cse=True, nested=2)
    _assert_named_native_loss_and_gradients(2, gradient_checkpointing=policy)


def test_native_keyed_rematerialized_gqa_raw_and_public_match_named_values_and_gradients():
    config = LlamaConfig(
        hidden_dim=24,
        intermediate_dim=40,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=6,
        attn_backend=AttentionBackend.VANILLA,
        scan_layers=True,
        gradient_checkpointing=True,
    )
    native_config = NativeLlamaLmConfig(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    )
    Vocab, Batch, Pos = hax.Axis("vocab", 48), hax.Axis("batch", 2), config.max_Pos
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        tokens = jnp.array([[3, 13, 27, 6, 1, 19], [4, 1, 9, 25, 3, 17]], dtype=jnp.int32)
        raw_example = jax.vmap(lambda row: GrugLmExample.causal(row, eos_id=1))(tokens)
        example = named_lm_example_from_grug(raw_example, Pos, batch_axis=Batch)
        named = LlamaLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(127))
        source = named.to_state_dict()
        native = native_config.build(Vocab, key=jax.random.PRNGKey(131)).from_state_dict(source)
        for name, value in native.to_state_dict().items():
            np.testing.assert_array_equal(value, source[name].reshape(value.shape))
        forward_key = jax.random.PRNGKey(137)
        named_logits = eqx.filter_jit(lambda model: model(example.tokens, example.attn_mask, key=forward_key))(named)
        native_logits = eqx.filter_jit(lambda model: model(example.tokens, example.attn_mask, key=forward_key))(native)
        raw_logits = eqx.filter_jit(lambda model: model(tokens, raw_example.attn_mask, key=forward_key))(native.native)
        np.testing.assert_allclose(native_logits.array, named_logits.array, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(raw_logits, named_logits.array, atol=1e-5, rtol=1e-4)

        def public_loss(model):
            return model.compute_next_token_loss(example, key=forward_key, logsumexp_weight=0.0).scalar()

        def raw_loss(model):
            return model.next_token_loss(
                tokens,
                raw_example.loss_weight,
                mask=raw_example.attn_mask,
                key=forward_key,
                logsumexp_weight=0.0,
            )

        reference_loss, reference_grad = eqx.filter_jit(eqx.filter_value_and_grad(public_loss))(named)
        reference_gradients = reference_grad.to_state_dict()
        for model, loss_fn in ((native, public_loss), (native.native, raw_loss)):
            loss, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
            np.testing.assert_allclose(loss, reference_loss, atol=1e-5, rtol=1e-4)
            gradients = gradient.to_state_dict()
            assert gradients.keys() == reference_gradients.keys()
            for name, value in gradients.items():
                np.testing.assert_allclose(
                    value, reference_gradients[name].reshape(value.shape), atol=1e-5, rtol=1e-4, err_msg=name
                )


def test_native_keyed_paged_prefill_and_next_decode_match_dense_and_preserve_cache():
    config = LlamaConfig(
        hidden_dim=32,
        intermediate_dim=48,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=6,
        attn_backend=AttentionBackend.VANILLA,
        scan_layers=True,
        gradient_checkpointing=True,
    )
    native_config = NativeLlamaLmConfig(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    )
    Vocab = hax.Axis("vocab", 48)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.default_matmul_precision("highest"), jax.set_mesh(mesh), hax.axis_mapping({}):
        named = LlamaLMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(211))
        native = native_config.build(Vocab, key=jax.random.PRNGKey(223)).from_state_dict(named.to_state_dict())
        tokens = hax.named(jnp.array([5, 17, 29, 11, 7], dtype=jnp.int32), hax.Axis("position", 5))
        forward_key = jax.random.PRNGKey(227)
        dense = eqx.filter_jit(lambda model: model(tokens, AttentionMask.causal(), key=forward_key))(named)
        native_dense = eqx.filter_jit(lambda model: model(tokens, AttentionMask.causal(), key=forward_key))(native)
        np.testing.assert_allclose(native_dense.array, dense.array, atol=1e-5, rtol=1e-4)
        table = PageTable.init(max_pages=4, max_seqs=1, page_size=2, max_pages_per_seq=3)
        sequences = SequenceTable.init(table.max_seqs, table.pages_per_seq, table.page_size)
        sequences, slot = sequences.reserve_slot()
        native_cache = native.initial_cache(table.spec(), dtype=jnp.float32)
        named_cache = named.initial_cache(table.spec(), dtype=jnp.float32)
        prefix_pages = None
        prefix_snapshots = None
        for start, size in ((0, 4), (4, 1)):
            chunk = tokens["position", hax.dslice(start, size)]
            positions = hax.named(jnp.arange(start, start + size, dtype=jnp.int32), chunk.axes)
            slots = hax.full(chunk.axes, slot, dtype=jnp.int32)
            sequences, table, info = sequences.allocate_for_seq(table, slots, positions)
            native_out, native_cache = eqx.filter_jit(
                lambda model, cache: model.decode(chunk, cache, info, positions, key=forward_key)
            )(native, native_cache)
            named_out, named_cache = eqx.filter_jit(
                lambda model, cache: model.decode(chunk, cache, info, positions, key=forward_key)
            )(named, named_cache)
            expected = dense["position", hax.dslice(start, size)].array
            native_values, named_values, dense_values = (
                np.asarray(native_out.array),
                np.asarray(named_out.array),
                np.asarray(expected),
            )
            max_absolute_errors = {
                "native_paged_vs_dense": float(np.max(np.abs(native_values - dense_values))),
                "named_paged_vs_dense": float(np.max(np.abs(named_values - dense_values))),
                "native_paged_vs_named_paged": float(np.max(np.abs(native_values - named_values))),
            }
            diagnostics = f"start={start}, size={size}, max_absolute_errors={max_absolute_errors}"
            np.testing.assert_allclose(native_out.array, expected, atol=1e-5, rtol=1e-4, err_msg=diagnostics)
            np.testing.assert_allclose(named_out.array, expected, atol=1e-5, rtol=1e-4, err_msg=diagnostics)
            for actual, reference in zip(native_cache, named_cache, strict=True):
                np.testing.assert_allclose(actual.kv_pages.array, reference.kv_pages.array, atol=1e-5, rtol=1e-4)
            if start == 0:
                prefix_pages = np.asarray(info.page_indices.array[0, :2])
                prefix_snapshots = [np.asarray(cache.kv_pages.array[prefix_pages]).copy() for cache in native_cache]
            else:
                assert prefix_pages is not None and prefix_snapshots is not None
                for cache, previous in zip(native_cache, prefix_snapshots, strict=True):
                    np.testing.assert_array_equal(cache.kv_pages.array[prefix_pages], previous)


def _assert_named_native_loss_and_gradients(
    kv_heads, *, gradient_checkpointing, compute_dtype=jnp.float32, upcast_attn=False
):
    config = LlamaConfig(
        hidden_dim=16,
        intermediate_dim=24,
        num_layers=2,
        num_heads=4,
        num_kv_heads=kv_heads,
        max_seq_len=8,
        attn_backend=AttentionBackend.VANILLA,
        gradient_checkpointing=gradient_checkpointing,
        upcast_attn=upcast_attn,
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
        policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=compute_dtype, output_dtype=jnp.float32)
        losses = []
        gradients = []
        for model in (named, native):
            loss, gradient = eqx.filter_jit(
                eqx.filter_value_and_grad(
                    lambda m: policy.cast_to_compute(m).compute_next_token_loss(example).scalar()
                ),
                # Preserve the BF16 rounding stages that the training runtime requires.
                compiler_options={"xla_allow_excess_precision": False},
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
    assert checkpoint is not None
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

        batch = next(iter(trainer.data_loader(dataset)))
        uninterrupted = trainer.train_step(info.state, batch)
        resumed = trainer.train_step(restored, batch)
        np.testing.assert_allclose(resumed.loss, uninterrupted.loss, atol=1e-5, rtol=1e-4)
        assert int(resumed.state.step) == int(uninterrupted.state.step) == 2
        assert int(resumed.state.opt_state[0].count) == int(uninterrupted.state.opt_state[0].count) == 2
        np.testing.assert_array_equal(resumed.state.training_key, uninterrupted.state.training_key)
        for actual, expected in zip(
            jax.tree.leaves(eqx.filter(resumed.state, eqx.is_array)),
            jax.tree.leaves(eqx.filter(uninterrupted.state, eqx.is_array)),
            strict=True,
        ):
            assert actual.sharding == expected.sharding
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
