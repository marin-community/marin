# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference import snowball_prefill
from levanter.inference.engine import InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.multihost import MultiHostInferenceEngine
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.snowball import SnowballCache
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.testing.cpu_devices import run_on_cpu_devices


def test_stacked_cache_copy_and_reset_preserve_layer_and_head_order():
    axes = tuple(
        hax.Axis(name, size)
        for name, size in zip(("layer", "page", "slot", "kv_head", "head_size"), (3, 4, 2, 10, 2), strict=True)
    )
    values = np.arange(np.prod([axis.size for axis in axes]), dtype=np.float32).reshape(
        tuple(axis.size for axis in axes)
    )
    cache = SnowballCache(hax.named(jnp.asarray(values), axes), jnp.array([2, 3], jnp.int32))
    copied = jax.jit(lambda c: c.copy_page(1, 3))(cache)
    expected = values.copy()
    expected[:, 3] = values[:, 1]
    np.testing.assert_array_equal(copied.kv_pages.array, expected)
    np.testing.assert_array_equal(cache.kv_pages.array, values)
    reset = jax.jit(lambda c: c.reset())(copied)
    np.testing.assert_array_equal(reset.kv_pages.array, np.zeros_like(values))
    np.testing.assert_array_equal(reset.dropped_assignments, [0, 0])


@pytest.mark.timeout(180)
def test_packed_chunked_snowball_decode_matches_full_forward():
    cfg = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=5,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        sliding_window=4,
        max_seq_len=16,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    mesh = compact_grug_mesh(replica_axis_size=1)
    # A single CPU device avoids imposing the machine's production training mesh on this test.
    if jax.device_count() != 1:
        pytest.skip("single-device reference test")
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), cfg, key=jax.random.key(17))
        rows = [np.arange(10, dtype=np.int32), np.arange(10, 20, dtype=np.int32)]
        reference = [model(hax.named(jnp.asarray(row), "position")).array for row in rows]
        cache = model.initial_cache(PageTableSpec(num_pages=8, page_size=4), dtype=jnp.float32)
        decode = eqx.filter_jit(lambda m, ids, c, info, pos: m.decode(ids, c, info, pos))
        consumed = [0, 0]
        # Unequal packed prefixes, then one-token decode with independent absolute positions.
        for lengths in ([5, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1]):
            ids, positions, dests = [], [], []
            offsets = [0]
            for seq, length in enumerate(lengths):
                start = consumed[seq]
                ids.extend(rows[seq][start : start + length])
                positions.extend(range(start, start + length))
                dests.extend(seq * 16 + p for p in range(start, start + length))
                consumed[seq] += length
                offsets.append(len(ids))
            valid = len(ids)
            ids.extend([0] * (8 - valid))
            positions.extend([0] * (8 - valid))
            dests.extend([-1] * (8 - valid))
            info = PageBatchInfo(
                slot_ids=hax.named(jnp.array([0, 1], jnp.int32), "seq"),
                page_indices=hax.named(jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], jnp.int32), ("seq", "page")),
                seq_lens=hax.named(jnp.array(consumed, jnp.int32), "seq"),
                cu_q_lens=hax.named(jnp.array(offsets, jnp.int32), "seq"),
                num_seqs=jnp.array(2, jnp.int32),
                new_token_dests=hax.named(jnp.array(dests, jnp.int32), "position"),
                page_size=4,
            )
            logits, cache = decode(
                model,
                hax.named(jnp.array(ids, jnp.int32), "position"),
                cache,
                info,
                hax.named(jnp.array(positions, jnp.int32), "position"),
            )
            for seq, length in enumerate(lengths):
                if length:
                    expected = reference[seq][consumed[seq] - length : consumed[seq]]
                    np.testing.assert_allclose(
                        logits.array[offsets[seq] : offsets[seq + 1]], expected, rtol=1e-4, atol=1e-5
                    )
            np.testing.assert_array_equal(cache.dropped_assignments, [0, 0])


@pytest.mark.timeout(180)
def test_expert_parallel_decode_preserves_order_and_does_not_drop_skewed_routes():
    run_on_cpu_devices(
        """
        import equinox as eqx
        import haliax as hax
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
        from levanter.models.snowball import SnowballConfig, SnowballMoEMLP
        from levanter.inference.snowball import _moe

        cfg = SnowballConfig(hidden_dim=8, intermediate_dim=8, num_experts=4,
                             num_experts_per_token=2, moe_implementation="ring")
        axes = ("replica_dcn", "data", "context", "expert", "model")
        devices = np.asarray(jax.devices())
        x = jnp.arange(56, dtype=jnp.float32).reshape(7, 8) / 32
        valid = jnp.array([True, True, True, True, True, False, False])
        outputs = []
        for ep in (1, 2):
            mesh = Mesh(devices[:ep].reshape(1, 1, 1, ep, 1), axes,
                        axis_types=(AxisType.Explicit,) * 5)
            with jax.set_mesh(mesh), hax.axis_mapping({}):
                mlp = SnowballMoEMLP.init(cfg, key=jax.random.key(13))
                # Every valid assignment targets expert shard zero under EP2.
                mlp = eqx.tree_at(lambda m: m.router_bias, mlp,
                                  reshard(jnp.array([1000., 1000., -1000., -1000.]), P("expert")))
                output, drops = eqx.filter_jit(_moe)(mlp, x, valid)
                np.testing.assert_array_equal(drops, [0, 0])
                outputs.append(np.asarray(output))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(outputs[1][5:], 0)
        """,
        device_count=2,
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "chunk_size,prompt_length", [(4, 3), (4, 4), (4, 5), (4, 8), (4, 9), (512, 2047), (512, 2048), (512, 2049)]
)
def test_chunked_generation_matches_two_full_forward_steps(chunk_size, prompt_length, monkeypatch):
    mesh = compact_grug_mesh(replica_axis_size=1)
    if jax.device_count() != 1:
        pytest.skip("single-device reference test")
    cfg = SnowballConfig(
        vocab_size=16,
        hidden_dim=8,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sliding_window=2048,
        max_seq_len=2060,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 16), cfg, key=jax.random.key(23))
        prompt = (np.arange(prompt_length) % 15 + 1).tolist()
        forward = eqx.filter_jit(lambda m, ids: m(hax.named(ids, "position")).array[-1])
        expected, logprobs = [], []
        for _ in range(2):
            logits = forward(model, jnp.array(prompt + expected, jnp.int32))
            token = int(jnp.argmax(logits))
            expected.append(token)
            logprobs.append(float(jax.nn.log_softmax(logits)[token]))
        config = InferenceEngineConfig(
            max_seq_len=2060,
            page_size=4,
            max_pages=515,
            max_seqs=1,
            max_seqs_in_prefill=1,
            max_prefill_size=chunk_size,
            max_queued_tokens=2060,
            max_rounds=1,
            max_tokens_per_round=1,
            max_stop_seqs=1,
            max_stop_tokens=1,
        )
        engine = MultiHostInferenceEngine.from_model_with_config(model, None, config, axis_resources={})
        prefix = snowball_prefill._cache_prefix

        def checked_prefix(state, model, queue):
            counts = np.asarray(state.decode_state.page_table.page_ref_counts.array).copy()
            assert np.count_nonzero(counts) == (prompt_length + 3) // 4
            updated = prefix(state, model, queue)
            np.testing.assert_array_equal(updated.decode_state.page_table.page_ref_counts.array, counts)
            return updated

        monkeypatch.setattr(snowball_prefill, "_cache_prefix", checked_prefix)
        params = SeqDecodingParams(
            max_num_tokens=np.array(prompt_length + 2, np.int32),
            temperature=np.array(0, np.float32),
            top_p=np.array(1, np.float32),
            key=np.array([0, 19311], np.uint32),
            stop_tokens=None,
        )
        result = engine.generate([Request(prompt, 0, params, 1)])
        assert result.tokens == [expected]
        assert result.total_generated == 2
        np.testing.assert_allclose(result.logprobs[0], logprobs, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(engine.gen_state.cache.dropped_assignments, [0, 0])
        if prompt_length == 2049:
            repeated = engine.generate([Request(prompt, 1, params, 1)])
            assert repeated.tokens == [expected]
            np.testing.assert_allclose(repeated.logprobs[0], logprobs, rtol=1e-4, atol=1e-5)
        if prompt_length == 9:
            stops = hax.NamedArray(np.array([[expected[0]]], np.int32), ("stop_seq", "position"))
            stopped = engine.generate([Request(prompt, 0, dataclasses.replace(params, stop_tokens=stops), 1)])
            assert stopped.tokens == [[expected[0]]]
            assert stopped.total_generated == 1
