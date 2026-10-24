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
from levanter.inference.snowball_scoring import score_cached_packed
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


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.slow
@pytest.mark.timeout(240)
def test_batched_chunked_prefill_preserves_sequences_and_page_budget_across_reuse(batch_size):
    if jax.device_count() != 1:
        pytest.skip("single-device reference test")
    cfg = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sliding_window=32,
        max_seq_len=2060,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    mesh = compact_grug_mesh(replica_axis_size=1)
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), cfg, key=jax.random.key(913))
        config = InferenceEngineConfig(
            max_seq_len=2060,
            page_size=16,
            max_pages=batch_size * 130,
            max_seqs=batch_size,
            max_seqs_in_prefill=batch_size,
            max_prefill_size=512,
            max_queued_tokens=batch_size * 2060,
            max_tokens_per_round=batch_size,
            max_rounds=2,
            max_stop_seqs=1,
            max_stop_tokens=1,
        )
        engine = MultiHostInferenceEngine.from_model_with_config(model, None, config, axis_resources={})
        lengths = [3, 2050] if batch_size == 2 else [3, 17, 2047, 2050, 8, 259, 2049, 33][:batch_size]
        prompts = [(1 + (np.arange(n) + idx * 3) % 31).tolist() for idx, n in enumerate(lengths)]
        expected = []
        for prompt in prompts:
            first = int(jnp.argmax(model(hax.named(jnp.array(prompt, jnp.int32), "position")).array[-1]))
            second = int(jnp.argmax(model(hax.named(jnp.array([*prompt, first], jnp.int32), "position")).array[-1]))
            expected.append([first, second])

        def request(idx, stop_first=False):
            stop = expected[idx][0] if stop_first else -1
            return Request(
                request_id=idx,
                prompt_tokens=prompts[idx],
                n_generations=1,
                decode_params=SeqDecodingParams(
                    max_num_tokens=np.array(len(prompts[idx]) + 2, np.int32),
                    temperature=np.array(0, np.float32),
                    top_p=np.array(1, np.float32),
                    key=np.array([0, 19311 + idx], np.uint32),
                    stop_tokens=hax.NamedArray(np.array([[stop]], np.int32), ("stop_seq", "position")),
                ),
            )

        result = engine.generate([request(idx, stop_first=idx == 0) for idx in range(batch_size)])
        assert result.tokens == [expected[0][:1], *expected[1:]]
        np.testing.assert_array_equal(engine.gen_state.cache.dropped_assignments, [0, 0])
        # This engine retains finished pages until the next generation reset.
        # Only the first continuation is cached; the second is the final sample.
        expected_pages = sum((length + (idx != 0) + 15) // 16 for idx, length in enumerate(lengths))
        assert (
            np.count_nonzero(np.asarray(engine.gen_state.decode_state.page_table.page_ref_counts.array))
            == expected_pages
        )
        # Fewer requests than available slots exercises invalid padding after a
        # prior full batch, with a different input order on the same engine.
        reuse = [batch_size - 1] if batch_size == 2 else [batch_size - 1, 1]
        repeated = engine.generate([request(idx) for idx in reuse])
        assert repeated.tokens == [expected[idx] for idx in reuse]
        repeated_pages = sum((lengths[idx] + 1 + 15) // 16 for idx in reuse)
        assert (
            np.count_nonzero(np.asarray(engine.gen_state.decode_state.page_table.page_ref_counts.array))
            == repeated_pages
        )


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_packed_teacher_forcing_matches_serial_and_full_forward_with_independent_pages():
    if jax.device_count() != 1:
        pytest.skip("single-device reference test")
    cfg = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sliding_window=32,
        max_seq_len=112,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    mesh = compact_grug_mesh(replica_axis_size=1)
    with jax.set_mesh(mesh), hax.axis_mapping({}):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 32), cfg, key=jax.random.key(913))
        config = InferenceEngineConfig(
            max_seq_len=112,
            page_size=8,
            max_pages=28,
            max_seqs=2,
            max_seqs_in_prefill=2,
            max_prefill_size=64,
            max_queued_tokens=224,
        )
        engine = MultiHostInferenceEngine.from_model_with_config(model, None, config, axis_resources={})
        sequences = [
            (1 + (np.arange(length) * (idx + 1) + idx * 3) % 31).tolist() for idx, length in enumerate((97, 63))
        ]
        serial = [score_cached_packed(engine, [s], 2)[0] for s in sequences]
        repeat = [score_cached_packed(engine, [s], 2)[0] for s in sequences]
        packed = score_cached_packed(engine, sequences, 2)
        reversed_pack = score_cached_packed(engine, sequences[::-1], 2)[::-1]
        layout = engine.last_packed_score_layout
        assert layout["per_sequence_chunk"] == 32
        assert any(step["starts"][0] == 32 and step["physical_query_width"] == 2 for step in layout["steps"])
        assert any(step["valid_lengths"][0] == 0 and step["valid_lengths"][1] > 0 for step in layout["steps"])
        for step in layout["steps"]:
            expected_active = [i for i, length in enumerate(step["valid_lengths"]) if length > 0]
            assert step["active_slots"] == expected_active
            assert step["num_active_sequences"] == len(expected_active)
            assert step["metadata_slot_ids"][: len(expected_active)] == expected_active
            assert (
                step["metadata_cu_q_lens"]
                == np.cumsum(
                    [0, *[step["valid_lengths"][i] for i in expected_active], *[0] * (2 - len(expected_active))]
                ).tolist()
            )
            if expected_active == [1]:
                assert (
                    step["metadata_page_indices"][0][0] == 8
                )  # preserve pages after the shorter sequence's eight pages
                assert step["metadata_slot_ids"][1] == -1
                assert all(page == -1 for page in step["metadata_page_indices"][1])
        assert any(step["active_slots"] == [1] for step in layout["steps"])
        for i, sequence in enumerate(sequences):
            assert serial[i].token_logprobs == repeat[i].token_logprobs
            np.testing.assert_allclose(packed[i].token_logprobs, serial[i].token_logprobs, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(reversed_pack[i].token_logprobs, serial[i].token_logprobs, atol=1e-5, rtol=1e-5)
            ids = hax.named(jnp.asarray(sequence), "position")
            logits = model(ids)
            full = jax.nn.log_softmax(logits.array[:-1].astype(jnp.float32), axis=-1)
            expected = np.asarray(full[jnp.arange(len(sequence) - 1), jnp.asarray(sequence[1:])])
            np.testing.assert_allclose(packed[i].token_logprobs[1:], expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(engine.gen_state.cache.dropped_assignments, [0, 0])
