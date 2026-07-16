# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the grug value-embedding variant.

The variant's claims are that the value-embedding table enters the model only through the
attention value path -- leaving the attention pattern identical to the base model's -- and that
the table and its per-layer gates actually learn. These tests pin both.
"""

import dataclasses
import json
import logging
import uuid
from io import StringIO
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from fray.cluster import ResourceConfig
from jax.sharding import AxisType
from levanter.analysis.backward_flow import BackwardFlowConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.grug.ve.launch import _OPTIMIZER
from experiments.grug.ve.model import GrugModelConfig, Transformer
from experiments.grug.ve.train import GrugRunConfig, GrugTrainerConfig, run_grug

# Grouped-query attention on purpose: v carries num_kv_heads heads, not num_heads, so a value
# table sized to hidden_dim would be shape-wrong here while passing under a MHA config.
GQA_CONFIG = GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=64,
    num_layers=3,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=16,
    value_emb_lambda_init=0.5,
)

TOKENS = jnp.asarray([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=jnp.int32)


@pytest.fixture
def mesh():
    """The model's ``reshard`` calls need a mesh; one device is enough to check numerics.

    Not autouse: the training loop enters its own mesh, and nesting the two would not reflect
    how the model is actually constructed at run time.
    """
    device_mesh = jax.make_mesh(
        (1, 1, 1),
        ("replica_dcn", "data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(device_mesh):
        yield device_mesh


def _drop_value_embeddings(model: Transformer) -> Transformer:
    """The control twin: the same weights, with the value-embedding table removed.

    Every other parameter is shared with ``model``, so a difference in output between the two
    is attributable to the value embeddings and to nothing else -- not to a different init draw.
    """
    return eqx.tree_at(lambda m: m.value_embed, model, None, is_leaf=lambda x: x is None)


def _with_lambdas(model: Transformer, value: float) -> Transformer:
    return eqx.tree_at(
        lambda m: [block.attn.lambda_ve for block in m.blocks],
        model,
        [jnp.asarray(value, dtype=jnp.float32)] * len(model.blocks),
    )


def test_zero_lambda_is_exactly_the_base_model(mesh):
    """At lambda=0 the blend contributes nothing, so the model must reduce to the base model.

    This is the claim that the value embeddings touch only ``v``: if the table leaked into q, k,
    or the residual stream, zeroing the value-path gate would not recover the base model's
    logits bit-for-bit.
    """
    model = Transformer.init(GQA_CONFIG, key=jax.random.PRNGKey(0))
    control = _drop_value_embeddings(model)

    gated_off = _with_lambdas(model, 0.0).logits(TOKENS)
    baseline = control.logits(TOKENS)

    np.testing.assert_array_equal(np.asarray(gated_off), np.asarray(baseline))


def test_open_gate_changes_the_output(mesh):
    """A nonzero gate must actually route the stored memory into the output.

    The companion to the test above: without this, a model that silently ignored ``ve``
    entirely would still pass the lambda=0 equivalence check.
    """
    model = Transformer.init(GQA_CONFIG, key=jax.random.PRNGKey(0))
    control = _drop_value_embeddings(model)

    gated_on = _with_lambdas(model, 0.5).logits(TOKENS)
    baseline = control.logits(TOKENS)

    assert not np.allclose(np.asarray(gated_on), np.asarray(baseline))


def test_value_table_is_sized_to_the_kv_width(mesh):
    """The table is blended into ``v``, so a row must be exactly one value vector wide."""
    model = Transformer.init(GQA_CONFIG, key=jax.random.PRNGKey(0))
    head_dim = GQA_CONFIG.inferred_head_dim

    assert model.value_embed is not None
    assert model.value_embed.shape == (GQA_CONFIG.vocab_size, GQA_CONFIG.num_kv_heads * head_dim)


def test_disabled_variant_carries_no_value_embedding_parameters(mesh):
    """The control arm of the ablation must not pay for a table it does not use."""
    config = dataclasses.replace(GQA_CONFIG, value_emb_lambda_init=None)
    model = Transformer.init(config, key=jax.random.PRNGKey(0))

    assert model.value_embed is None
    assert model.value_emb_lambdas() == ()


def test_training_moves_the_table_and_the_gates(mesh):
    """The table and the per-layer gates must both receive gradient and move under an update.

    A gate stuck at its init, or a table that never updates, would make the whole experiment
    measure nothing -- and both are silent failures rather than crashes.
    """
    model = Transformer.init(GQA_CONFIG, key=jax.random.PRNGKey(0))
    loss_weight = jnp.ones_like(TOKENS, dtype=jnp.float32)

    def loss_fn(m: Transformer) -> jax.Array:
        return m.next_token_loss(TOKENS, loss_weight)

    grads = eqx.filter_grad(loss_fn)(model)

    assert grads.value_embed is not None
    assert jnp.any(grads.value_embed != 0), "value-embedding table received no gradient"
    for i, lam_grad in enumerate(grads.value_emb_lambdas()):
        assert lam_grad != 0, f"lambda gate at layer {i} received no gradient"

    optimizer = optax.adam(1e-2)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, _ = optimizer.update(eqx.filter(grads, eqx.is_inexact_array), optimizer.init(params), params)
    updated = eqx.apply_updates(model, updates)

    for i, (before, after) in enumerate(zip(model.value_emb_lambdas(), updated.value_emb_lambdas(), strict=True)):
        assert not jnp.allclose(before, after), f"lambda gate at layer {i} did not move under an update"


@pytest.mark.parametrize("lambda_init", [0.0, 0.5, 1.0])
def test_forward_pass_runs_across_the_gate_range(mesh, lambda_init: float):
    """The blend must be well-formed at both ends of its range, not just in the middle.

    lambda=1 is the degenerate 'stored memory only' case: the value path stops depending on the
    hidden state at all, and it must still produce finite logits rather than NaNs.
    """
    config = dataclasses.replace(GQA_CONFIG, value_emb_lambda_init=lambda_init)
    model = Transformer.init(config, key=jax.random.PRNGKey(0))

    logits = model.logits(TOKENS)

    assert logits.shape == (*TOKENS.shape, config.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


def test_lambda_gates_are_exempt_from_weight_decay(mesh):
    """The launch optimizer must not decay the per-layer gates.

    Levanter's default mask decays them (it only exempts norms, biases, and ``*Embedding*``
    class names, and grug's parameters are raw arrays). A decayed gate is pulled toward zero
    every step regardless of gradient, which confounds the lambda-vs-depth readout the
    experiment exists to produce. The matrices and both embedding tables must keep their decay:
    the exemption is for the gates, not a change to the rest of the recipe.
    """
    model = Transformer.init(GQA_CONFIG, key=jax.random.PRNGKey(0))
    mask = _OPTIMIZER.build_weight_decay_mask()(model)

    for i, block_mask in enumerate(mask.blocks):
        assert block_mask.attn.lambda_ve is False, f"lambda gate at layer {i} would be decayed"
        assert block_mask.attn.w_q is True
        assert block_mask.rms_attn.weight is False
    assert mask.token_embed is True
    assert mask.value_embed is True
    assert mask.output_proj is True


def _run_one_step(tmp_path: Path, *, logger_name: str, value_emb_lambda_init: float | None) -> dict:
    """Train the variant for one step against an in-memory dataset; return the run's summary."""
    vocab_size, seq_len = 128, 32
    examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + i) % vocab_size) for i in range(8)]
    eval_examples = [GrugLmExample.causal((jnp.arange(seq_len, dtype=jnp.int32) + 100) % vocab_size)]
    data_config = LmDataConfig(
        components={
            "direct": DirectDatasetComponent(
                datasets={"train": ListAsyncDataset(examples), "validation": ListAsyncDataset(eval_examples)}
            )
        },
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        run_grug(
            GrugRunConfig(
                model=GrugModelConfig(
                    vocab_size=vocab_size,
                    hidden_dim=32,
                    intermediate_dim=64,
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    max_seq_len=seq_len,
                    value_emb_lambda_init=value_emb_lambda_init,
                ),
                data=data_config,
                resources=ResourceConfig.with_cpu(),
                trainer=GrugTrainerConfig(
                    trainer=TrainerConfig(
                        id=f"test-{logger_name}",
                        num_train_steps=1,
                        train_batch_size=max(1, len(jax.devices())),
                        tracker=JsonLoggerConfig(logger_name=logger_name),
                        require_accelerator=False,
                        use_explicit_mesh_axes=True,
                        distributed=DistributedConfig(initialize_jax_distributed=False),
                        log_dir=tmp_path / "logs",
                        checkpointer=CheckpointerConfig(base_path=str(tmp_path / "checkpoints")),
                    ),
                    log_every=1,
                    backward_flow=BackwardFlowConfig(interval=0),
                ),
                eval=None,
            )
        )
    finally:
        logger.removeHandler(handler)

    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finish = [record for record in records if record.get("event") == "finish"]
    assert len(finish) == 1
    return finish[0]["summary"]


def test_training_run_reports_a_lambda_per_layer(tmp_path: Path):
    """The per-layer gates must reach the tracker: they are the experiment's readout.

    Drives the real training loop rather than the callback in isolation, so a gate that is never
    logged -- the failure that would leave the run un-analyzable after the GPU time is spent --
    is caught here.
    """
    summary = _run_one_step(tmp_path, logger_name=f"grug_ve_{uuid.uuid4().hex}", value_emb_lambda_init=0.5)

    assert summary["ve/lambda/layer_0"] == pytest.approx(0.5, abs=0.1)
    assert summary["ve/lambda/layer_1"] == pytest.approx(0.5, abs=0.1)
    assert "ve/lambda/mean" in summary
    assert "train/loss" in summary


def test_control_arm_trains_and_reports_no_lambdas(tmp_path: Path):
    """The control arm must run the same loop and simply have no gates to report."""
    summary = _run_one_step(tmp_path, logger_name=f"grug_base_{uuid.uuid4().hex}", value_emb_lambda_init=None)

    assert "train/loss" in summary
    assert not [key for key in summary if key.startswith("ve/lambda")]
