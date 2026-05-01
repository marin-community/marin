# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
from haliax import Axis

from levanter.inference.page_table import PageTableSpec
from levanter.inference.scoring import ScoringEngine, ScoringEngineConfig
from levanter.layers.kv_cache import KvPageCache


class DummyModel(eqx.Module):
    Vocab: Axis = eqx.field(static=True)
    eos: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.eos = eos_id

    def initial_cache(self, spec: PageTableSpec, *, dtype):
        kv_heads = Axis("kv_head", 1)
        head_size = Axis("embed", 1)
        return KvPageCache.init(spec, kv_heads, head_size, dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        Pos = input_ids.resolve_axis("position")
        logits = hax.nn.one_hot(self.eos, self.Vocab, dtype=jnp.float32).broadcast_axis(Pos)
        return logits, kv_cache


def _build_engine(*, max_seq_len: int = 32, max_completion_len: int = 4, prompt_chunk_size: int = 32) -> ScoringEngine:
    return ScoringEngine.from_model_with_config(
        model=DummyModel(vocab_size=16),  # type: ignore[arg-type]
        tokenizer=None,
        config=ScoringEngineConfig(
            max_seq_len=max_seq_len,
            max_batch_size=4,
            max_completion_len=max_completion_len,
            max_pages=64,
            page_size=4,
            compute_dtype=jnp.float32,
            prompt_chunk_size=prompt_chunk_size,
        ),
    )


def test_accept_releases_candidate_clones():
    engine = _build_engine()

    engine.score([1, 2], [[3], [4]])
    used_after_score = engine.gen_state.decode_state.sequences.used_mask.array
    assert int(used_after_score.sum()) == 3

    engine.accept([1, 2], [3])

    used_after_accept = engine.gen_state.decode_state.sequences.used_mask.array
    assert int(used_after_accept.sum()) == 1


def test_score_chunks_long_prompt_and_accept_chunks_long_extension():
    engine = _build_engine(max_seq_len=48, max_completion_len=12, prompt_chunk_size=8)

    prompt_tokens = list(range(20))
    extension_tokens = list(range(20, 32))

    scores = engine.score(prompt_tokens, [[1]])
    assert len(scores) == 1

    engine.accept(prompt_tokens, extension_tokens)
    anchor_len = int(engine.gen_state.decode_state.seq_lens["seq", 0].scalar())
    assert anchor_len == len(prompt_tokens) + len(extension_tokens)


def test_anchor_tokens_track_canonical_prompt_state():
    engine = _build_engine(max_seq_len=48, max_completion_len=12, prompt_chunk_size=8)

    prompt_tokens = list(range(10, 22))
    extension_tokens = list(range(22, 28))

    engine.score(prompt_tokens, [[1, 2]])
    stored_after_score = engine.gen_state.decode_state.tokens["seq", 0, "position", : len(prompt_tokens)].array
    assert stored_after_score.tolist() == prompt_tokens

    engine.accept(prompt_tokens, extension_tokens)
    combined = prompt_tokens + extension_tokens
    stored_after_accept = engine.gen_state.decode_state.tokens["seq", 0, "position", : len(combined)].array
    assert stored_after_accept.tolist() == combined


def test_short_completions_do_not_advance_clones_through_padding():
    engine = _build_engine(max_seq_len=32, max_completion_len=4, prompt_chunk_size=16)

    prompt_tokens = [5, 6, 7]
    engine.score(prompt_tokens, [[8], [9, 10]])

    seq_lens = engine.gen_state.decode_state.seq_lens.array
    assert int(seq_lens[1]) == len(prompt_tokens) + 1
    assert int(seq_lens[2]) == len(prompt_tokens) + 2
