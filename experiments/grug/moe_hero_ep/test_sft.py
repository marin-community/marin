# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.moe_hero_ep.model import LoraProjection
from experiments.grug.moe_hero_ep.sft import _local_cross_entropy
from experiments.grug.moe_hero_ep.sft_launch import PilotInspection, PilotStop, sft_stop_reason


def test_pilot_pauses_on_actual_boundaries_and_prioritizes_strict_total_caps():
    inspection = PilotInspection(3, 40, "policy", "budget")

    def stop(step, tokens, next_tokens=0, inspection=inspection, cap=80):
        return sft_stop_reason(step, tokens, next_tokens, num_steps=10, max_processed_tokens=cap, inspection=inspection)

    # Reused smoke already beyond the early boundary must export before learning.
    assert stop(4, 32) == PilotStop.EARLY_UPDATE
    reviewed = dataclasses.replace(inspection, reviewed_step=4, reviewed_tokens=32)
    assert stop(4, 32, inspection=reviewed) is None
    assert stop(5, 64, inspection=reviewed) == PilotStop.TOKEN_INSPECTION
    # One complete update can cross an inspection threshold, but never the cap.
    assert stop(4, 32, 32, inspection=reviewed) is None
    assert stop(5, 64, 32, inspection=reviewed) == PilotStop.TOKEN_CAP
    assert stop(5, 40, cap=40, inspection=reviewed) == PilotStop.TOKEN_CAP
    assert stop(10, 40, inspection=reviewed) == PilotStop.UPDATE_CAP


def test_chunked_vocab_delta_loss_and_activation_gradients_match_dense_reference():
    """The frozen head remains in the activation VJP, and selected columns learn."""
    hidden = jax.random.normal(jax.random.key(1), (1, 7, 8))
    head = jax.random.normal(jax.random.key(2), (8, 37))
    delta = jax.random.normal(jax.random.key(3), (8, 3)) * 0.1
    labels = jnp.array([[3, 5, 17, 25, 1, 36, 0]])
    weights = jnp.array([[0, 1, 1, 1, 0, 1, 0]], jnp.float32)
    selected = jnp.array([3, 17, 36])

    def tiled(x, d):
        total, count = _local_cross_entropy(
            x, head, labels, weights, d, selected, vocab_block_size=16, token_block_size=4
        )
        return total / count

    def reference(x, d):
        logits = x @ head + (x @ d) @ jax.nn.one_hot(selected, 37)
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, labels) * weights) / jnp.sum(weights)

    tiled_result = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(hidden, delta)
    reference_result = jax.value_and_grad(reference, argnums=(0, 1))(hidden, delta)
    for actual, expected in zip(jax.tree.leaves(tiled_result), jax.tree.leaves(reference_result), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_scanned_lora_outputs_and_gradients_match_dense_reference():
    keys = jax.random.split(jax.random.key(42), 5)
    base = jax.random.normal(keys[0], (2, 8, 8)) * 0.2
    a = jax.random.normal(keys[1], (2, 8, 2)) * 0.2
    b = jax.random.normal(keys[2], (2, 2, 8)) * 0.2
    inputs = jax.random.normal(keys[3], (3, 8))
    targets = jax.random.normal(keys[4], (3, 8))
    scale = 1.5

    def scanned(x, adapter_a, adapter_b):
        def layer(hidden, parameters):
            weight, layer_a, layer_b = parameters
            adapter = LoraProjection(layer_a, layer_b, scale)
            return jnp.tanh(hidden @ weight + adapter(hidden)), None

        output, _ = jax.lax.scan(layer, x, (base, adapter_a, adapter_b))
        return jnp.mean(jnp.square(output - targets)), output

    def dense(x, adapter_a, adapter_b):
        for layer in range(2):
            weight = base[layer] + scale * (adapter_a[layer] @ adapter_b[layer])
            x = jnp.tanh(x @ weight)
        return jnp.mean(jnp.square(x - targets)), x

    with jax.set_mesh(compact_grug_mesh()):
        actual = jax.jit(jax.value_and_grad(scanned, argnums=(0, 1, 2), has_aux=True))(inputs, a, b)
        expected = jax.value_and_grad(dense, argnums=(0, 1, 2), has_aux=True)(inputs, a, b)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, atol=1e-5, rtol=1e-5)
    # Nonzero factors exercise gradients of both LoRA factors in every layer.
    for gradient in actual[1][1:]:
        assert np.all(np.linalg.norm(gradient, axis=(1, 2)) > 0)


@pytest.mark.timeout(120)
def test_current_hero_compact_sft_update_resume_export_and_context_sharding(tmp_path):
    script = r"""
        import dataclasses
        import json
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        import numpy as np
        from jax.sharding import AxisType, Mesh, PartitionSpec
        from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer, apply_qb_betas, LoraProjection
        from levanter.checkpoint import save_checkpoint
        from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
        from pathlib import Path
        from unittest.mock import patch
        import draccus
        import yaml
        from experiments.grug.moe_hero_ep.averaging import (
            AverageConfig, CheckpointPin, TokenizerPin, average_checkpoints, _DONORS,
        )
        from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
        from experiments.grug.moe_hero_ep.weights import restore_weights
        from experiments.grug.moe_hero_ep.format_eval import AcquisitionCase, AcquisitionPanel, AcquisitionDiagnostic
        from experiments.grug.moe_hero_ep.sft_launch import (
            HeroSftCheckpoint, write_sft_handoff, HeroSftInspection, PilotInspection, PilotStop,
            pilot_inspection_due, read_pilot_inspection, save_pilot_inspection,
        )
        from experiments.grug.moe_hero_ep.sft import (
            SftConfig, BaseReference, initialize_sft, sft_logits, sft_loss,
            make_sft_step, save_sft_checkpoint, restore_sft_checkpoint, merge_sft,
            compiled_gradient_report, initialize_from_checkpoint, export_sft,
            make_sft_gradients, make_sft_optimizer_step, compact_optimizer_report,
        )

        config = GrugModelConfig(vocab_size=64, hidden_dim=32, intermediate_dim=16,
            shared_expert_intermediate_dim=16, num_experts=4, num_experts_per_token=2,
            num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, attention_implementation="reference",
            moe_implementation="fixed_all_to_all", capacity_factor=4.)
        sft = SftConfig(rank=2, alpha=2., token_ids=(2, 3, 4, 5, 6),
            adapter_learning_rate=1e-3, vocab_learning_rate=1e-3,
            num_steps=10, warmup_steps=0, seed=1, vocab_block_size=32, token_block_size=4)
        reference = BaseReference("candidate-c", "exact-c-digest", "pt-source", 123, "A", "tokenizer", "pinned-revision")
        ids = np.tile(np.array([[2,4,1,4,2,3,5,7]], dtype=np.int32), (4,1))
        output = {}
        for context, expert in [(1,4), (2,2)]:
            mesh = Mesh(np.asarray(jax.devices()).reshape(1,1,context,expert,1),
                        ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
            with jax.set_mesh(mesh):
                base = Transformer.init(config, key=jax.random.key(0))
                # Nonzero, non-BF16-exact control state must survive runtime/export unchanged.
                base = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, base,
                                  jnp.linspace(-0.013, 0.017, 8).reshape(2,4))
                original_qb = np.asarray(base.stacked_blocks.stacked.mlp.router_bias)
                frozen, initial = initialize_sft(base, sft)
                tokens = jnp.asarray(ids)
                weights = jnp.ones_like(tokens, dtype=jnp.float32).at[:,-1].set(0)
                rounded = jax.tree.map(lambda x: x.astype(jnp.bfloat16), base)
                rounded = eqx.tree_at(lambda m: m.stacked_blocks.stacked.mlp.router_bias, rounded,
                                      base.stacked_blocks.stacked.mlp.router_bias)
                np.testing.assert_array_equal(sft_logits(frozen, initial.params, tokens, sft), rounded.logits(tokens))
                frozen_values = [np.asarray(x).copy() for x in jax.tree.leaves(frozen)]
                step = make_sft_step(sft)
                state, metrics = step(initial, frozen, tokens, weights, jnp.ones_like(tokens, dtype=bool))
                gradient_stage = make_sft_gradients(sft)
                adam_stage = make_sft_optimizer_step(sft)
                compact_grads, split_metrics = gradient_stage(initial.params, frozen, tokens, weights,
                                                               jnp.ones_like(tokens, dtype=bool))
                split_state = adam_stage(initial, compact_grads, split_metrics["processed_tokens"],
                                         split_metrics["supervised_tokens"])
                for split, fused in zip(jax.tree.leaves(split_state), jax.tree.leaves(state), strict=True):
                    np.testing.assert_allclose(split, fused, rtol=1e-6, atol=1e-7)
                assert np.isfinite(float(metrics["loss"])) and float(metrics["gradient_norm"]) > 0
                assert int(state.step) == 1 and int(state.processed_tokens) == tokens.size
                assert int(state.supervised_tokens) == 28
                assert np.linalg.norm(np.asarray(state.params.input_delta)) > 0
                assert np.linalg.norm(np.asarray(state.params.output_delta)) > 0
                assert np.linalg.norm(np.asarray(state.params.adapters.stacked_blocks.stacked.attn.lora_q.b[0])) > 0
                np.testing.assert_array_equal(frozen.stacked_blocks.stacked.mlp.router_bias, original_qb)
                for before, after in zip(frozen_values, jax.tree.leaves(frozen), strict=True):
                    np.testing.assert_array_equal(before, after)
                # Gradient outputs contain only adapters/vocab; no attention, routed expert,
                # router or complete vocabulary tensor exists in the differentiated pytree.
                grad = jax.grad(lambda p: sft_loss(p, frozen, tokens, weights, sft)[0])
                grad_shapes = [tuple(x.shape) for x in jax.tree.leaves(jax.eval_shape(grad, state.params))]
                assert base.token_embed.shape not in grad_shapes and base.output_proj.shape not in grad_shapes
                assert base.stacked_blocks.stacked.attn.w_q.shape not in grad_shapes
                trainable_size = sum(x.size for x in jax.tree.leaves(state.params))
                assert trainable_size < sum(x.size for x in jax.tree.leaves(base)) / 4
                # The compiled state/gradient boundary also excludes full-model buffers.
                compiled = step.lower(initial, frozen, tokens, weights, jnp.ones_like(tokens, dtype=bool)).compile()
                Path(CHECKPOINT_ROOT + "/compiled-hlo.txt").write_text(compiled.as_text())
                report = compiled_gradient_report(compiled, base, sft)
                # Tiny CPU activation shapes collide with frozen projections.
                # Parameter operands alone do not prove the contraction's axes
                # or result flow, so schema 4 must leave those cases unverified.
                assert report["schema_version"] == 4
                assert not report["avoids_full_base_gradients"], report
                assert report["unverified_weight_shaped_results"], report
                assert compiled.memory_analysis().output_size_in_bytes < sum(x.nbytes for x in jax.tree.leaves(base))
                path = CHECKPOINT_ROOT + "/cp-" + str(context)
                save_sft_checkpoint(path, state, reference, sft)
                restored = restore_sft_checkpoint(path, initial, reference, sft, mesh)
                assert compact_optimizer_report(restored)["compact_optimizer_only"]
                resumed, _ = step(restored, frozen, tokens, weights, jnp.ones_like(tokens, dtype=bool))
                uninterrupted, _ = step(state, frozen, tokens, weights, jnp.ones_like(tokens, dtype=bool))
                for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted), strict=True):
                    np.testing.assert_array_equal(actual, expected)
                try:
                    wrong = dataclasses.replace(reference, metadata_digest="wrong-base")
                    restore_sft_checkpoint(path, initial, wrong, sft, mesh)
                except ValueError:
                    pass
                else:
                    raise AssertionError("A compact checkpoint must not resume against a different candidate C")
                merged = merge_sft(frozen, state.params, sft)
                adapter_logits = np.asarray(sft_logits(frozen, state.params, tokens, sft))
                merged_logits = np.asarray(merged.logits(tokens), dtype=np.float32)
                # BF16 merge and separately rounded low-rank GEMMs differ; this bound
                # is below one BF16 ULP for these O(0.1) logits plus accumulation.
                np.testing.assert_allclose(merged_logits, adapter_logits, rtol=0.02, atol=0.002)
                diagnostic_panel = AcquisitionPanel(source_artifact_digest="fixed-tiny-source",
                    tokenizer_revision="pinned-revision", cases=(
                        AcquisitionCase(source_id="tiny-one", prefix_token_ids=(2,4,1), target_token_id=4),
                        AcquisitionCase(source_id="tiny-two", prefix_token_ids=(2,3), target_token_id=5),
                    ))
                acquisition = AcquisitionDiagnostic.from_panel(diagnostic_panel)
                diagnostic = acquisition(frozen, state.params, sft, int(state.step))
                expected_losses = []
                expected_successes = 0
                for case in diagnostic_panel.cases:
                    row = jnp.pad(jnp.array(case.prefix_token_ids, dtype=jnp.int32),
                                  (0, 4 - len(case.prefix_token_ids)))
                    prefix = jnp.tile(row, (4,1))
                    # The independent all-position reference selects the same causal position.
                    values = np.asarray(sft_logits(frozen, state.params, prefix, sft))[0, len(case.prefix_token_ids)-1]
                    expected_losses.append(float(jax.scipy.special.logsumexp(values) - values[case.target_token_id]))
                    expected_successes += int(np.argmax(values) == case.target_token_id)
                assert diagnostic["successes"] == expected_successes and diagnostic["cases"] == 2
                np.testing.assert_allclose(diagnostic["mean_nll"], np.mean(expected_losses), rtol=1e-6, atol=1e-6)
                assert merged.stacked_blocks.stacked.attn.lora_q is None
                np.testing.assert_array_equal(merged.stacked_blocks.stacked.mlp.router_bias, original_qb)
                if context == 1:
                    full_gradient = jax.jit(jax.grad(lambda m: m.next_token_loss(tokens, weights)))
                    full_compiled = full_gradient.lower(rounded).compile()
                    full_report = compiled_gradient_report(full_compiled, base)
                    assert not full_report["avoids_full_base_gradients"], full_report
                    assert full_report["unverified_weight_shaped_results"], full_report
                    # Prepare a real candidate C through the streamed donor path. Only
                    # provider metadata/tokenizer acquisition are external-boundary fakes.
                    donor_config = dataclasses.replace(config, vocab_size=128010, hidden_dim=8,
                        intermediate_dim=4, shared_expert_intermediate_dim=4,
                        num_heads=2, num_kv_heads=1, head_dim=4)
                    donor_base = Transformer.init(donor_config, key=jax.random.key(17))
                    source = CHECKPOINT_ROOT + "/source-pt"
                    pending = jnp.array([[0.0031, -0.0021, 0.007, 0.001],
                                         [-0.0031, 0.0051, 0.0023, -0.007]], dtype=jnp.float32)
                    save_checkpoint({"master_params": donor_base,
                        "params": jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), donor_base),
                        "pending_qb_betas": pending, "step": jnp.array(123), "opt_state": jnp.arange(13)},
                        123, source, is_temporary=False)
                    metadata = json.loads((Path(source) / "metadata.json").read_text())
                    manifest = json.loads((Path(source) / "manifest.json").read_text())
                    revision = "a" * 40
                    run = {"model": draccus.encode(donor_config),
                        "optimizer": draccus.encode(GrugMoeMuonHConfig(learning_rate=0.1, adam_lr=0.02)),
                        "trainer": {"trainer": {"num_train_steps": 1000}}, "data": {"tokenizer": "test-tokenizer"}}
                    run = json.loads(json.dumps(run))
                    config_path = CHECKPOINT_ROOT + "/source.yaml"
                    Path(config_path).write_text(yaml.safe_dump(run))
                    pin = CheckpointPin(source, digest(metadata), config_path, digest(run), revision,
                                        digest(manifest), "local tokenizer fixture")
                    class LocalProvider:
                        worker_identity = {"provider": "local-cpu-fixture"}
                        def worker_region(self): return "local-fixture"
                        def storage_region(self, path): return "local-fixture"
                    class LocalTokenizer:
                        def encode(self, text, *, add_special_tokens):
                            encodings = {text: token for row in _DONORS for text, token in (row[:2], row[2:])}
                            return [encodings[text]]
                    prepared = CHECKPOINT_ROOT + "/prepared-c"
                    preparation = AverageConfig((pin,), prepared, CHECKPOINT_ROOT + "/scratch",
                        TokenizerPin("test-tokenizer", revision), (), "f" * 40, 8,
                        max_chunk_bytes=8 * 1024 * 1024, initialize_donors=True)
                    with patch("experiments.grug.moe_hero_ep.averaging.AutoTokenizer.from_pretrained",
                               return_value=LocalTokenizer()):
                        c_metadata = average_checkpoints(preparation, region_probe=LocalProvider())
                    actual_reference = BaseReference(prepared, digest(c_metadata), source, 123, "A",
                                                     "test-tokenizer", revision)
                    donor_sft = dataclasses.replace(sft, token_ids=tuple(row[1] for row in _DONORS),
                                                      vocab_block_size=4096, token_block_size=4)
                    restored_base, reset = initialize_from_checkpoint(actual_reference, donor_config, mesh, donor_sft)
                    expected = apply_qb_betas(donor_base, pending)
                    np.testing.assert_array_equal(restored_base.stacked_blocks.stacked.mlp.router_bias,
                                                   expected.stacked_blocks.stacked.mlp.router_bias)
                    assert int(reset.step) == 0 and int(reset.processed_tokens) == 0
                    assert np.count_nonzero(np.asarray(reset.params.input_delta)) == 0
                    assert np.count_nonzero(np.asarray(reset.params.output_delta)) == 0
                    for destination, donor in [(row[1], row[3]) for row in _DONORS]:
                        destination_value = restored_base.output_proj.at[:, destination].get(
                            out_sharding=PartitionSpec())
                        donor_value = donor_base.output_proj.at[:, donor].get(out_sharding=PartitionSpec())
                        np.testing.assert_array_equal(destination_value, donor_value.astype(jnp.bfloat16))
                    np.testing.assert_array_equal(restored_base.token_embed, donor_base.token_embed.astype(jnp.bfloat16))
                    donor_ids = jnp.tile(jnp.array([[128006, 3, 128007, 5, 128009, 4, 128006, 2]], jnp.int32), (4,1))
                    donor_weights = jnp.ones_like(donor_ids, dtype=jnp.float32).at[:,-1].set(0)
                    learned, _ = make_sft_step(donor_sft)(reset, restored_base, donor_ids, donor_weights,
                                                         jnp.ones_like(donor_ids, dtype=bool))
                    compact_path = CHECKPOINT_ROOT + "/compact-d"
                    save_sft_checkpoint(compact_path, learned, actual_reference, donor_sft)
                    compact_metadata = json.loads((Path(compact_path) / "metadata.json").read_text())
                    assert compact_metadata["hero_sft_counters"] == {
                        "sft_step": 1,
                        "processed_tokens": donor_ids.size,
                        "supervised_tokens": int(donor_weights.sum()),
                    }
                    export_path = CHECKPOINT_ROOT + "/export-d"
                    export_sft(export_path, restored_base, learned, actual_reference, donor_sft)
                    write_sft_handoff(CHECKPOINT_ROOT, export_path, compact_path, learned, actual_reference)
                    handoff = HeroSftCheckpoint(path=CHECKPOINT_ROOT)
                    assert handoff.native_checkpoint == export_path and handoff.compact_checkpoint == compact_path
                    assert handoff.handoff.step == 1 and handoff.handoff.processed_tokens == donor_ids.size
                    export_metadata = json.loads((Path(export_path) / "metadata.json").read_text())
                    ordinary = restore_weights(export_path, digest(export_metadata), donor_config, mesh)
                    np.testing.assert_array_equal(ordinary.stacked_blocks.stacked.mlp.router_bias,
                                                   expected.stacked_blocks.stacked.mlp.router_bias)
                    expected_merged = merge_sft(restored_base, learned.params, donor_sft)
                    np.testing.assert_array_equal(ordinary.output_proj, expected_merged.output_proj)
                    assert handoff.metadata_digest == digest(export_metadata)
                    # A mandatory early pause exports the exact state. Continuing with
                    # the whole-pilot schedule preserves Adam and the next data update.
                    boundary = PilotInspection(early_update=1, non_padding_tokens=donor_ids.size + 8,
                                               policy_digest="pinned-policy", budget_digest="pinned-budget")
                    assert pilot_inspection_due(int(reset.step), int(reset.processed_tokens), boundary) is None
                    assert (
                        pilot_inspection_due(int(learned.step), int(learned.processed_tokens), boundary)
                        == PilotStop.EARLY_UPDATE
                    )
                    inspection = HeroSftInspection(
                        handoff=handoff.handoff, compact_metadata_digest=digest(compact_metadata),
                        sft=donor_sft, measurement_identity_digest="pinned-model-data-batch",
                        inspection=boundary, max_processed_tokens=donor_ids.size * 4,
                        stop_reason=PilotStop.EARLY_UPDATE,
                    )
                    save_pilot_inspection(CHECKPOINT_ROOT, inspection)
                    inspection_json = json.loads((Path(CHECKPOINT_ROOT) / "inspection.json").read_text())
                    checked = read_pilot_inspection(CHECKPOINT_ROOT + "/inspection.json", digest(inspection_json))
                    assert checked.status == "awaiting_evaluation" and checked.sft == donor_sft
                    continued = restore_sft_checkpoint(compact_path, reset, actual_reference, donor_sft, mesh)
                    next_boundary = dataclasses.replace(boundary, reviewed_step=int(continued.step),
                                                        reviewed_tokens=int(continued.processed_tokens))
                    assert (pilot_inspection_due(
                        int(continued.step), int(continued.processed_tokens), next_boundary
                    ) is None)
                    resumed, _ = make_sft_step(donor_sft)(continued, restored_base, donor_ids, donor_weights,
                                                         jnp.ones_like(donor_ids, dtype=bool))
                    uninterrupted, _ = make_sft_step(donor_sft)(learned, restored_base, donor_ids, donor_weights,
                                                               jnp.ones_like(donor_ids, dtype=bool))
                    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(uninterrupted), strict=True):
                        np.testing.assert_array_equal(actual, expected)
                    assert (
                        pilot_inspection_due(int(resumed.step), int(resumed.processed_tokens), next_boundary)
                        == PilotStop.TOKEN_INSPECTION
                    )
                    try:
                        initialize_from_checkpoint(actual_reference,
                            dataclasses.replace(donor_config, qk_mult=2.), mesh, donor_sft)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Shape-compatible changes to forward semantics must be rejected")
                output[context] = (float(metrics["loss"]), adapter_logits)
        np.testing.assert_allclose(output[1][0], output[2][0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(output[1][1], output[2][1], rtol=0.02, atol=0.002)
    """
    script = "CHECKPOINT_ROOT = " + repr(str(tmp_path)) + "\n" + textwrap.dedent(script)
    # Two real EP/CP meshes exercise forward, activation backprop, trainable sharding,
    # native checkpoint I/O and deterministic optimizer/data-step continuation.
    run_on_cpu_devices(script, device_count=4)
