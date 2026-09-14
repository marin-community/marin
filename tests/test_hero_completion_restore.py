# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from ops.vibe_check import config
from ops.vibe_check.completions import Checkpoint, Prompt, SampleRequest, SamplingSpec, digest
from ops.vibe_check.config import Ancestor, ProductionRun, discover_requests
from ops.vibe_check.sample import COMPUTE_POLICY, next_logits, restore_model


@pytest.fixture
def spec():
    model = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    return SamplingSpec(
        release="test-v1",
        batch_size=1,
        prompts=(Prompt(id="p", text="p", seed=0, source_url="https://example.org"),),
        tokenizer="test",
        tokenizer_revision="a" * 40,
        model=draccus.encode(model),
        temperature=0,
        max_new_tokens=2,
        context_length=8,
    )


@pytest.mark.parametrize(("wrapped", "master"), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("include_manifest", [False, True])
def test_native_restore_preserves_weights_and_applies_pending_router_bias(
    tmp_path, spec, wrapped, master, include_manifest
):
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    path = tmp_path / "step-12"
    with jax.set_mesh(mesh):
        model = Transformer.init(draccus.decode(GrugModelConfig, spec.model), key=jax.random.PRNGKey(7))
        state = {"params": model, "pending_qb_betas": jnp.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])}
        if master:
            state["master_params"] = model
            state["params"] = jax.tree.map(jnp.zeros_like, model)
        save_checkpoint({"train_state": state} if wrapped else state, 12, path, is_temporary=False)
        if not include_manifest:
            (path / "manifest.json").unlink()
        metadata = json.loads((path / "metadata.json").read_text())
        request = SampleRequest(
            checkpoint=Checkpoint(
                uri=str(path), run_id="test", step=12, timestamp=metadata["timestamp"], metadata_digest=digest(metadata)
            ),
            spec=spec,
            source_revision="a" * 40,
            target_cluster="test",
        )
        restored = restore_model(request, mesh)
        expected = eqx.tree_at(
            lambda tree: tree.stacked_blocks.stacked.mlp.router_bias,
            model,
            jnp.array([[1.5, 0.5, -0.5, -1.5], [1.5, 0.5, -0.5, -1.5]]),
        )
        for actual, wanted in zip(jax.tree.leaves(restored), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(wanted))


def test_discovery_excludes_temporary_incomplete_and_non_lineage_checkpoints(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(config, "CHECKPOINT_ROOT", str(tmp_path))
    metadata = {"timestamp": "2026-09-12T10:00:00", "is_temporary": False}
    for run_id, step, temporary in [
        ("old", 6000, False),
        ("old", 12000, False),
        ("active", 12000, True),
        ("active", 18000, False),
        ("trial", 24000, False),
    ]:
        checkpoint = tmp_path / run_id / "v1" / "checkpoints" / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(json.dumps({**metadata, "step": step, "is_temporary": temporary}))
    incomplete = tmp_path / "active/v1/checkpoints/step-24000"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text("{}")
    handoff = tmp_path / "forced-handoff"
    handoff.mkdir()
    (handoff / "metadata.json").write_text(json.dumps({**metadata, "step": 7000}))
    run = ProductionRun(
        run_id="active",
        version="v1",
        target_cluster="test",
        handoff_checkpoint=str(handoff),
        handoff_run_id="old",
        ancestors=(Ancestor(run_id="old", version="v1", max_step=7000),),
    )
    requests = discover_requests(run, spec, "a" * 40)
    assert {(row.checkpoint.run_id, row.checkpoint.step) for row in requests} == {
        ("old", 6000),
        ("old", 7000),
        ("active", 18000),
    }


@eqx.filter_jit
def full_sequence_logits(model, tokens):
    hidden, _ = model(tokens)
    return jnp.einsum("bsh,hv->bsv", hidden, model.output_proj, preferred_element_type=jnp.float32)


def test_logits_select_each_rows_last_input_position(spec):
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = COMPUTE_POLICY.cast_to_compute(
            Transformer.init(draccus.decode(GrugModelConfig, spec.model), key=jax.random.PRNGKey(7))
        )
        tokens = jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]])
        positions = jnp.array([2, 1])
        full_logits = full_sequence_logits(model, tokens)
        expected = np.stack([np.asarray(full_logits)[0, 2], np.asarray(full_logits)[1, 1]])
        np.testing.assert_array_equal(np.asarray(next_logits(model, tokens, positions)), expected)
