# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import textwrap
import threading

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug_sft import replay_data
from experiments.grug_sft.reclose_eval import RecloseEvalConfig, RecloseEvaluator, pack, trajectory_stats
from experiments.grug_sft.replay_data import batch_from_examples, example_at, read_step, write_step
from experiments.grug_sft.special_token_train import _BATCH_AXES, GrugTrainState, _apply_qb_betas
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def test_microbatched_step_matches_full_batch_step():
    # Device count is process-global, so the two-device mesh runs in a fresh CPU process.
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        import optax
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.data.text.examples import GrugLmExample
        from experiments.grug_sft.special_token_train import GrugTrainState, _make_train_step
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        config = GrugModelConfig(
            vocab_size=16, hidden_dim=8, intermediate_dim=8,
            shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="scatter",
            disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(1), (4, 8), 1, 16, dtype=jnp.int32)
        segment_ids = jnp.array([[0] * 8, [0] * 4 + [1] * 4, [0] * 6 + [-1] * 2, [0] * 3 + [1] * 5], jnp.int32)
        weights = jax.random.bernoulli(jax.random.PRNGKey(2), 0.7, (4, 8)).astype(jnp.float32)
        batch = jax.vmap(GrugLmExample.causal)(tokens, segment_ids=segment_ids, loss_weight=weights)
        optimizer = optax.adam(1e-2)
        mp = jmp.get_policy("float32")
        results = []
        mesh = compact_grug_mesh(replica_axis_size=1)
        with jax.set_mesh(mesh):
            for microbatches in (1, 2):
                params = Transformer.init(config, key=jax.random.PRNGKey(0))
                state = GrugTrainState(
                    step=jnp.array(0, jnp.int32), params=params, opt_state=optimizer.init(params),
                    ema_params=None, pending_qb_betas=jnp.zeros((config.num_layers, config.num_experts)),
                )
                step = _make_train_step(optimizer, mp, z_loss_weight=1e-4, ema_beta=None, microbatches=microbatches)
                state, metrics, _ = step(state, batch)
                results.append((metrics["train/loss"], metrics["train/cross_entropy_loss"], state.params))
        for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_trainable_output_rows_freeze_every_other_parameter():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        import optax
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.data.text.examples import GrugLmExample
        from experiments.grug_sft.special_token_train import GrugTrainState, _make_train_step
        from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

        config = GrugModelConfig(
            vocab_size=16, hidden_dim=8, intermediate_dim=8,
            shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
            num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=8, sliding_window=4,
            attention_implementation="reference", moe_implementation="scatter",
            disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
        )
        tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 1, 16, dtype=jnp.int32)
        batch = jax.vmap(GrugLmExample.causal)(tokens)
        optimizer = optax.adam(1e-2)
        with jax.set_mesh(compact_grug_mesh(replica_axis_size=1)):
            params = Transformer.init(config, key=jax.random.PRNGKey(0))
            before = jax.tree.map(np.asarray, params)
            state = GrugTrainState(
                step=jnp.array(0, jnp.int32), params=params, opt_state=optimizer.init(params),
                ema_params=None, pending_qb_betas=jnp.zeros((config.num_layers, config.num_experts)),
            )
            step = _make_train_step(
                optimizer, jmp.get_policy("float32"), z_loss_weight=0.0, ema_beta=None, trainable_output_rows=(15,)
            )
            state, _, _ = step(state, batch)
        after = jax.tree.map(np.asarray, state.params)
        assert not np.allclose(after.output_proj[:, 15], before.output_proj[:, 15])
        np.testing.assert_array_equal(after.output_proj[:, :15], before.output_proj[:, :15])
        rest = lambda m: [leaf for leaf in jax.tree.leaves(m) if leaf.shape != m.output_proj.shape]
        for a, b in zip(rest(after), rest(before), strict=True):
            np.testing.assert_array_equal(a, b)
        """
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "2"}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_reclose_stats_use_positions_that_predict_the_close():
    # ids: prompt(0..1) reasoning(2..3) close(4) lead-in/continuation(5..7) re-close(8) tail(9)
    close_id = 99
    trajectory = {"ids": [1, 2, 3, 4, close_id, 5, 6, 7, close_id, 8], "reasoning_start": 2, "close": 4, "reclose": 8}
    p = np.zeros(10)
    p[3] = 0.9  # predicts the first close
    p[2] = 0.2  # inside the reasoning
    p[4:8] = [0.1, 0.0, 0.0, 0.5]  # predict tokens 5..8; position 7 predicts the re-close
    p[8] = 1.0  # after the re-close: excluded
    stats = trajectory_stats(trajectory, p)
    assert stats["p_first_close"] == 0.9
    assert stats["p_reclose"] == 0.5
    np.testing.assert_allclose(stats["hazard"], 1 - 0.9 * 0.5)
    np.testing.assert_allclose(stats["p_premature"], np.mean([p[1], p[2]]))


def test_pack_keeps_trajectories_in_separate_segments():
    trajectories = [{"ids": [1, 2, 3]}, {"ids": [4, 5]}, {"ids": [6, 7, 8, 9]}]
    tokens, segment_ids, placements = pack(trajectories, seq_len=5, rows_multiple=2)
    assert tokens.shape == (2, 5)
    for i, (t, place) in enumerate(zip(trajectories, placements, strict=True)):
        span = slice(place.offset, place.offset + len(t["ids"]))
        assert tokens[place.row, span].tolist() == t["ids"]
        assert (segment_ids[place.row, span] == i).all()
    assert (segment_ids[segment_ids >= 0].size) == 9


def test_replay_step_round_trips_through_storage(tmp_path):

    tokens = jnp.array([[5, 6, 7, 8], [1, 2, 3, 4]], jnp.int32)
    segment_ids = jnp.array([[0, 0, 1, 1], [0, 0, 0, -1]], jnp.int32)
    examples = [GrugLmExample.causal(t, segment_ids=s) for t, s in zip(tokens, segment_ids, strict=True)]
    write_step(str(tmp_path), 3, batch_from_examples(examples))
    restored = read_step(str(tmp_path), 3)
    for row, original in enumerate(examples):
        rebuilt = example_at(restored, row)
        np.testing.assert_array_equal(rebuilt.tokens, original.tokens)
        np.testing.assert_array_equal(rebuilt.loss_weight, original.loss_weight)
        np.testing.assert_array_equal(rebuilt.attn_mask.segment_ids[0], original.attn_mask.segment_ids[0])


def test_replay_read_retries_a_stalled_request(tmp_path, monkeypatch):
    tokens = jnp.array([[5, 6, 7, 8]], jnp.int32)
    example = GrugLmExample.causal(tokens[0], segment_ids=jnp.zeros(4, jnp.int32))
    write_step(str(tmp_path), 0, batch_from_examples([example]))
    real_filesystem_for = replay_data.filesystem_for
    release = threading.Event()
    calls = []

    class FirstReadStalls:
        def __init__(self, fs):
            self.fs = fs

        def cat(self, path):
            calls.append(path)
            if len(calls) == 1:
                release.wait()
            return self.fs.cat(path)

    def stalling_filesystem_for(url):
        fs, path = real_filesystem_for(url)
        return FirstReadStalls(fs), path

    monkeypatch.setattr(replay_data, "filesystem_for", stalling_filesystem_for)
    monkeypatch.setattr(replay_data, "READ_TIMEOUT", 0.2)
    try:
        batch = replay_data.read_step(str(tmp_path), 0)
    finally:
        release.set()
    assert len(calls) == 2
    np.testing.assert_array_equal(batch.tokens[0], tokens[0])


def test_reclose_evaluator_matches_direct_softmax(tmp_path):

    close_id = 15
    config = GrugModelConfig(
        vocab_size=16, hidden_dim=8, intermediate_dim=8,
        shared_expert_intermediate_dim=8, num_experts=2, num_experts_per_token=1,
        num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=16, sliding_window=4,
        attention_implementation="reference", moe_implementation="scatter",
        disable_pko=True, disable_long_rope=True, use_array_stacked_blocks=True,
    )  # fmt: skip
    trajectories = [
        {"group": "swe", "id": "a", "sample": 0, "bridge": "", "ids": [1, 2, 3, close_id, 4, 5, close_id, 6],
         "reasoning_start": 1, "close": 3, "reclose": 6},
        {"group": "swe", "id": "b", "sample": 0, "bridge": "Wait, ", "ids": [7, 8, close_id, 9, 10, 11],
         "reasoning_start": 1, "close": 2, "reclose": None},
    ]  # fmt: skip
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in trajectories))
    mesh = compact_grug_mesh(replica_axis_size=1)
    with jax.set_mesh(mesh):
        params = Transformer.init(config, key=jax.random.PRNGKey(0))
        state = GrugTrainState(
            step=jnp.array(0, jnp.int32),
            params=params,
            opt_state=None,
            ema_params=None,
            pending_qb_betas=jnp.zeros((config.num_layers, config.num_experts)),
        )
        evaluator = RecloseEvaluator(
            RecloseEvalConfig(trajectories=str(path), close_token_id=close_id, output=str(tmp_path / "out")),
            mesh=mesh,
            batch_axes=_BATCH_AXES,
            seq_len=16,
            rows_per_pass=1,
            mp=jmp.get_policy("float32"),
            apply_qb_betas=_apply_qb_betas,
        )
        metrics = evaluator.run(state, log_step=0)
        # Score each trajectory alone, without packing, from the full softmax.
        expected = []
        for t in trajectories:
            tokens = jnp.asarray([t["ids"]], jnp.int32)
            logits = params.logits(tokens, mask=AttentionMask.causal())
            p = np.asarray(jax.nn.softmax(logits.astype(jnp.float32), axis=-1)[0, :, close_id])
            expected.append(trajectory_stats(t, p))
    rows = [json.loads(line) for line in (tmp_path / "out" / "step-000000.jsonl").read_text().splitlines()]
    for row, want in zip(rows, expected, strict=True):
        for key in ("hazard", "p_first_close", "p_premature"):
            np.testing.assert_allclose(row[key], want[key], rtol=1e-4)
    np.testing.assert_allclose(metrics["reclose/swe/hazard"], np.mean([e["hazard"] for e in expected]), rtol=1e-4)
