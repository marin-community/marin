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
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Prompt,
    SampleRequest,
    SamplingSpec,
    digest,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.config import discover_requests
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import score_expected
from experiments.grug.moe_hero_ep.ops.vibe_check.sample import (
    COMPUTE_POLICY,
    expected_logprobs,
    next_logits,
    restore_model,
)


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
        completions_per_prompt=3,
        prompts=(Prompt(id="p", text="p", seed=0, source_url="https://example.org"),),
        tokenizer="test",
        tokenizer_revision="a" * 40,
        model=draccus.encode(model),
        temperature=0,
        max_new_tokens=2,
        context_length=8,
    )


@pytest.mark.parametrize(
    ("wrapped", "master", "include_manifest"),
    [
        pytest.param(False, True, True, id="master-weights-with-manifest"),
        pytest.param(True, False, False, id="wrapped-weights-without-manifest"),
    ],
)
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


def test_discovery_uses_selected_checkpoint_paths(tmp_path, spec):
    metadata = {"timestamp": "2026-09-12T10:00:00Z", "is_temporary": False}
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
    paths = [str(tmp_path / "old/v1/checkpoints/step-6000"), str(tmp_path / "active/v1/checkpoints/step-18000")]
    requests = discover_requests(paths, spec, "a" * 40, target_cluster="test")
    assert [(row.checkpoint.run_id, row.checkpoint.step) for row in requests] == [
        ("old", 6000),
        ("active", 18000),
    ]
    assert all(row.checkpoint.timestamp == "2026-09-12T10:00:00+00:00" for row in requests)
    assert requests[0].checkpoint.metadata_digest == digest({**metadata, "step": 6000})


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
        # Isolate position selection from rounding in differently shaped matrix products.
        model = eqx.tree_at(
            lambda value: value.output_proj,
            model,
            jnp.eye(*model.output_proj.shape, dtype=model.output_proj.dtype),
        )
        tokens = jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]])
        positions = jnp.array([2, 1])
        full_logits = full_sequence_logits(model, tokens)
        expected = np.stack([np.asarray(full_logits)[0, 2], np.asarray(full_logits)[1, 1]])
        np.testing.assert_array_equal(np.asarray(next_logits(model, tokens, positions)), expected)

        reference_spec = spec.model_copy(update={"context_length": 5})

        def logprobs(tokens, positions, targets):
            return expected_logprobs(model, jnp.asarray(tokens), jnp.asarray(positions), jnp.asarray(targets))

        def decode(ids):
            return "".join(chr(96 + token) for token in ids if token != 0)

        scores = score_expected(reference_spec, [[1, 2]], [[3, 4]], eos_token_id=0, logprobs=logprobs, decode=decode)[0]
        assert [score.token_id for score in scores] == [3, 4, 0]
        assert "".join(score.text for score in scores) == "cd"
        assert scores[-1].text == ""
        reference_logits = full_sequence_logits(model, jnp.array([[1, 2, 3, 4, 0]]))
        reference_logprobs = jax.nn.log_softmax(reference_logits, axis=-1)[0]
        np.testing.assert_allclose(
            [score.logprob for score in scores], np.asarray(reference_logprobs)[[1, 2, 3], [3, 4, 0]], rtol=1e-6
        )
        top_values, top_ids = jax.lax.top_k(reference_logprobs[3], 5)
        assert [token.token_id for token in scores[-1].top_tokens] == top_ids.tolist()
        np.testing.assert_allclose([token.logprob for token in scores[-1].top_tokens], top_values, rtol=1e-6)
        with pytest.raises(ValueError, match="fit in the context with EOS"):
            score_expected(reference_spec, [[1, 2]], [[3, 4, 5]], eos_token_id=0, logprobs=logprobs, decode=decode)
