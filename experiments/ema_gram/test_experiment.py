# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from dataclasses import asdict, replace

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.ema_gram import EmaGramConfig
from levanter.optim.util import flatten_linear_layers
from levanter.trainer_state import TrainerState
from marin.execution.lazy import materialized_config
from rigging.filesystem.storage_path import StoragePath
from wandb.util import json_friendly_val

from experiments.ema_gram.plan import ALPHAS, SEEDS, Cell, Result, confirmations, expand, lock_settings, pilots
from experiments.ema_gram.probe import loss_probes, norm_matched_direction
from experiments.ema_gram.report import paired_differences, summarize_attempt, validate_run_config
from experiments.ema_gram.train import AttemptLogHandler, build_training_step


def pilot_results(cells):
    # Each grid has an interior optimum; alpha=1/8 wins and must be promoted.
    return [
        Result(
            c.name,
            "completed",
            2 + (ALPHAS.index(c.alpha) - 1) ** 2 / 10 + np.log2(c.learning_rate / 0.01) ** 2,
            7630,
            7630,
            f"https://wandb.ai/test/runs/{c.name}",
        )
        for c in cells
    ]


def test_pilot_selection_expands_boundaries_before_paired_confirmation():
    cells = pilots([0.01] * 5)
    results = pilot_results(cells)
    lock = lock_settings(cells, results, 2.1)
    runs = confirmations(lock)
    assert {(c.alpha, c.seed) for c in runs} == {(a, s) for a in (0, 0.125, 0.25, 0.5) for s in SEEDS}
    # A late, better endpoint observation invalidates the lock and creates a new trial.
    edge = min((c for c in cells if c.alpha == 0.25 and c.ridge == 0.0001), key=lambda c: c.learning_rate)
    results = [replace(r, validation_loss=0.5) if r.name == edge.name else r for r in results]
    with pytest.raises(ValueError, match="boundary"):
        lock_settings(cells, results, 2.1)
    extended = expand(cells, results)
    assert replace(edge, learning_rate=edge.learning_rate / 2) in extended
    assert {c.name for c in cells}.issubset({c.name for c in extended})


def test_failed_and_partial_pilots_cannot_win_selection():
    cells = pilots([0.01] * 5)
    results = pilot_results(cells)
    results[0] = replace(results[0], status="failed", validation_loss=0)
    lock = lock_settings(cells, results, 2.1)
    assert cells[0] not in lock.selected
    results[0] = replace(results[0], status="completed", completed_steps=12)
    with pytest.raises(ValueError, match="Incomplete"):
        lock_settings(cells, results, 2.1)


def test_delphi_configs_pair_initialization_data_and_all_controls():
    cells = pilots([0.01] * 5)
    lock = lock_settings(cells, pilot_results(cells), 2.1)
    for phase in ("confirmation", "horizon", "scale"):
        configs = [
            materialized_config(build_training_step(c, "test-dev"), "/tmp/ema-gram-test").train_config
            for c in confirmations(lock, phase=phase)
        ]
        for seed in SEEDS:
            paired = [c for c in configs if c.trainer.seed == seed]
            baseline = paired[0]
            roundtrip = draccus.decode(TrainLmConfig, draccus.encode(baseline))
            assert roundtrip.optimizer == baseline.optimizer
            assert roundtrip.model == baseline.model
            assert roundtrip.data == baseline.data
            for cfg in paired:
                assert cfg.data == baseline.data
                assert cfg.model == baseline.model
                assert cfg.data_seed == seed
                a = asdict(cfg.optimizer)
                b = asdict(baseline.optimizer)
                for key in ("alpha", "ridge", "learning_rate", "gram_beta"):
                    a.pop(key)
                    b.pop(key)
                assert a == b


def test_checkpoint_probes_match_each_matrix_and_keep_nonmatrices_fixed():
    cfg = Qwen3Config(
        hidden_dim=8, intermediate_dim=16, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, max_seq_len=8
    )
    model = cfg.build(hax.Axis("vocab", 16), key=jax.random.PRNGKey(1))
    tx = EmaGramConfig(alpha=0, learning_rate=0.001).build(20)
    state = TrainerState.init(tx, model, key=jax.random.PRNGKey(2))
    grads = jax.tree.map(jnp.sin, state.trainable_model)
    state, _ = eqx.filter_jit(lambda s: s.take_step(grads, key=jax.random.PRNGKey(3)))(state)
    reference = norm_matched_direction(state.model, state.opt_state, alpha=0, ridge=0.01, polar_steps=5)
    candidate = norm_matched_direction(state.model, state.opt_state, alpha=0.25, ridge=0.01, polar_steps=5)
    for r, c in zip(
        jax.tree.leaves(flatten_linear_layers(reference)), jax.tree.leaves(flatten_linear_layers(candidate)), strict=True
    ):
        if r.ndim >= 2:
            np.testing.assert_allclose(
                np.linalg.norm(c, axis=(-2, -1)), np.linalg.norm(r, axis=(-2, -1)), rtol=1e-5, atol=1e-5
            )
    np.testing.assert_array_equal(candidate.embeddings.token_embeddings.weight.array, 0)
    np.testing.assert_array_equal(candidate.lm_head.weight.array, 0)

    # A quadratic held-out objective gives distinct batch responses and an exact paired delta oracle.
    def loss_fn(m, offset):
        return sum(jnp.sum((x - offset) ** 2) for x in jax.tree.leaves(m))

    rows = loss_probes(state, loss_fn, [0.0, 0.2], [0.001], polar_steps=5)
    row = next(r for r in rows if r.alpha == 0.25 and r.ridge == 0.01)
    actual = eqx.apply_updates(state.model, jax.tree.map(lambda x: -0.001 * x, candidate))
    expected = [float(loss_fn(actual, b) - loss_fn(state.model, b)) for b in (0.0, 0.2)]
    np.testing.assert_allclose(np.subtract(row.losses, row.baseline_losses), expected, atol=1e-5)


def test_reporting_preserves_paired_curves_and_censors_restart_time():
    base = pilots([0.01] * 5)[0]
    variant = replace(base, alpha=0.25)
    history = [{"_step": i, "global_step": i, "_runtime": i + 1, "eval/paloma/c4_en/loss": 4 - i / 10} for i in range(3)]
    control = summarize_attempt(
        base, history, expected_steps=3, tokens_per_step=32, run_state="finished", run_url="control", loss_target=3.9
    )
    better = [{**r, "eval/paloma/c4_en/loss": r["eval/paloma/c4_en/loss"] - 0.1} for r in history]
    candidate = summarize_attempt(
        variant, better, expected_steps=3, tokens_per_step=32, run_state="finished", run_url="candidate", loss_target=3.9
    )
    np.testing.assert_allclose([r.delta_loss for r in paired_differences([control, candidate])], -0.1)
    assert control.time_to_target == 2
    restart = summarize_attempt(
        base, history[1:], expected_steps=3, tokens_per_step=32, run_state="finished", run_url="restart", loss_target=3.9
    )
    assert restart.time_to_target is None
    assert restart.tokens_to_target is None


@pytest.mark.parametrize("phase,slug", [("horizon", "3e18"), ("scale", "3e19")])
def test_followups_pair_half_against_quarter_and_require_reference(phase, slug):
    history = [{"global_step": 0, "_step": 0, "_runtime": 1, "eval/paloma/c4_en/loss": 3.0}]
    runs = [
        summarize_attempt(
            Cell(a, 1e-4, 0.01, phase=phase, slug=slug),
            history,
            expected_steps=1,
            tokens_per_step=32,
            run_state="finished",
            run_url="test",
            loss_target=None,
        )
        for a in (0.25, 0.5)
    ]
    paired = paired_differences(runs)
    assert len(paired) == 1
    assert paired[0].reference_alpha == 0.25
    assert paired[0].alpha == 0.5
    with pytest.raises(ValueError, match="Missing completed reference"):
        paired_differences(runs[1:])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("trainer", "train_batch_size", 16),
        (None, "train_seq_len", 2048),
        ("model", "num_heads", 4),
        ("optimizer", "weight_decay", 0.2),
        ("optimizer", "adam_beta2", 0.99),
        ("data", "tokenizer", "wrong-tokenizer"),
    ],
)
def test_report_rejects_mismatched_training_controls(section, key, value):
    cfg = materialized_config(build_training_step(Cell(0.25, 1e-4, 0.01), "test-dev"), "/tmp/ema-test").train_config
    logged = json_friendly_val(asdict(cfg))
    validate_run_config(logged, cfg)
    (logged[section] if section else logged)[key] = value
    with pytest.raises(ValueError, match="differs from protocol"):
        validate_run_config(logged, cfg)


def test_report_accepts_wandb_omitted_empty_mappings_only():
    cfg = materialized_config(build_training_step(Cell(0.25, 1e-4, 0.01), "test-dev"), "/tmp/ema-test").train_config
    logged = json_friendly_val(asdict(cfg))
    del logged["adapter"]
    del logged["trainer"]["mesh"]["shared_mapping"]
    validate_run_config(logged, cfg)
    logged["adapter"] = {"unexpected": True}
    with pytest.raises(ValueError, match="differs from protocol: adapter"):
        validate_run_config(logged, cfg)
    del logged["adapter"]
    logged["trainer"]["mesh"]["shared_mapping"] = {"embed": "model"}
    with pytest.raises(ValueError, match=r"differs from protocol: trainer\.mesh"):
        validate_run_config(logged, cfg)
    del logged["trainer"]["mesh"]["shared_mapping"]
    del logged["model"]
    with pytest.raises(KeyError, match="model"):
        validate_run_config(logged, cfg)


def test_attempt_metrics_preserve_incremental_chunks_and_final_tail(tmp_path):
    artifact = StoragePath(str(tmp_path / "artifact"))
    artifact.mkdirs()
    local = tmp_path / "metrics.jsonl"
    handler = AttemptLogHandler(local, artifact)
    try:
        for step in range(105):
            handler.emit(logging.LogRecord("test", logging.INFO, "", 0, json.dumps({"step": step}), (), None))
        assert len(local.read_text().splitlines()) == 105
        assert len((artifact / "metrics-000000.jsonl").read_text().splitlines()) == 100
        assert not (artifact / "metrics-000001.jsonl").exists()
    finally:
        handler.close()
    assert [json.loads(line)["step"] for line in (artifact / "metrics-000001.jsonl").read_text().splitlines()] == list(
        range(100, 105)
    )
