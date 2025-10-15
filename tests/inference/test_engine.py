# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from haliax import Axis

from levanter.inference.engine import InferenceEngine, Request, InferenceEngineConfig
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.page_table import PageTable
from levanter.layers.kv_cache import KvPageCache


class DummyModel(eqx.Module):
    """Minimal model stub to drive GenerationService for tests.

    - `initial_cache` returns an empty KvPageCache sized to the PageTable.
    - `decode` returns constant logits that strongly prefer token `EOS`.
    """

    Vocab: Axis = eqx.field(static=True)
    eos: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.eos = eos_id

    def initial_cache(self, page_table: PageTable, *, dtype):
        # Use trivial cache dimensions; the cache is unused by this dummy model
        kv_heads = Axis("kv_head", 1)
        head_size = Axis("embed", 1)
        return KvPageCache.init(page_table, kv_heads, head_size, dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        # Produce logits that prefer `eos` for every sampled position
        Pos = input_ids.resolve_axis("position")
        Vocab = self.Vocab
        # One-hot on vocab axis for eos token, broadcast over positions
        logits = hax.nn.one_hot(self.eos, Vocab, dtype=jnp.float32)
        logits = logits.broadcast_axis(Pos)
        return logits, kv_cache


def _build_service(vocab_size=10):
    model = DummyModel(vocab_size=vocab_size, eos_id=3)
    service = InferenceEngine.from_model_with_config(
        model=model,  # type: ignore
        tokenizer=None,
        config=InferenceEngineConfig(
            max_seq_len=32,
            max_pages=64,
            max_seqs=8,
            page_size=8,
            compute_dtype=jnp.float32,
            max_queued_tokens=64,
            max_seqs_in_prefill=4,
        ),
    )
    return service


def test_release_on_finish_and_reuse_slots(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    svc = _build_service()

    prompts = [[7, 7], [1]]
    stop_tokens = [3]  # Dummy model always emits 3, so immediate stop after first generated token

    # Build Requests for the new API
    stop_ids = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis({"stop_seq": 1})
    reqs = []
    for i, toks in enumerate(prompts):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(toks) + 5, dtype=jnp.int32),
            stop_tokens=stop_ids,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(i),
        )
        reqs.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=1))

    result = svc.generate(reqs)

    # Each sequence should be original prompt + a single eos token
    # TODO: we recently stopped appending prompt to outputs; re-enable these checks if we restore that behavior
    # assert outputs[0] == prompts[0] + [3]
    # assert outputs[1] == prompts[1] + [3]
    assert result.tokens[0] == [3]
    assert result.tokens[1] == [3]
    assert result.total_generated == 2  # one new token per prompt

    # Finished sequences are auto-released; PageTable should have no active seqs
    pt = svc.gen_state.decode_state.page_table
    # All slots should be marked unused and lengths zeroed
    seq_lens = jax.device_get(pt.seq_lens.array)
    used_mask = jax.device_get(pt.used_mask.array)
    assert (used_mask == 0).all()
    assert (seq_lens == 0).all()
    # No pages should be held
    ref_counts = jax.device_get(pt.page_ref_counts.array)
    assert int(ref_counts.sum()) == 0

    # Ensure we logged the release
    assert any("Releasing finished sequences" in rec.message for rec in caplog.records)

    # Now reuse service for another prompt without calling reset()
    prompts2 = [[5, 5, 5]]
    # Build a second set of Requests to ensure reuse works without reset()
    stop_ids2 = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis(
        {"stop_seq": 1}
    )
    reqs2 = []
    for i, toks in enumerate(prompts2):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(toks) + 3, dtype=jnp.int32),
            stop_tokens=stop_ids2,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42 + i),
        )
        reqs2.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=1))

    result2 = svc.generate(reqs2)
    # TODO: re-enable if we restore prompt prepending
    # assert outputs2[0] == prompts2[0] + [3]
    assert result2.tokens[0] == [3]
    assert result2.total_generated == 1


def test_reuse_with_clones_and_slot_reassignment():
    svc = _build_service()

    prompts = [[7, 7], [1, 2]]
    stop_tokens = [3]

    stop_ids = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis({"stop_seq": 1})

    def build_requests(seed_offset: int) -> list[Request]:
        reqs: list[Request] = []
        for i, toks in enumerate(prompts):
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(toks) + 5, dtype=jnp.int32),
                stop_tokens=stop_ids,
                temperature=jnp.array(0.0, dtype=jnp.float32),
                key=jax.random.PRNGKey(seed_offset + i),
            )
            reqs.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=3))
        return reqs

    reqs = build_requests(0)
    outputs = svc.generate(reqs)
    assert all(out == [3] for out in outputs.tokens)

    reqs2 = build_requests(100)
    outputs2 = svc.generate(reqs2)
    assert all(out == [3] for out in outputs2.tokens)
