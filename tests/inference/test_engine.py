# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import haliax as hax
import jax.numpy as jnp
from haliax import Axis

from levanter.inference.engine import InferenceEngine, InferenceEngineConfig
from levanter.inference.page_table import PageTableSpec
from levanter.layers.kv_cache import KvPageCache


class DummyModel(eqx.Module):
    """Minimal model stub to drive GenerationService for tests.

    - `initial_cache` returns an empty KvPageCache sized to the page-table spec.
    - `decode` returns constant logits that strongly prefer token `EOS`.
    """

    Vocab: Axis = eqx.field(static=True)
    eos: int = eqx.field(static=True)

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.eos = eos_id

    def initial_cache(self, spec: PageTableSpec, *, dtype):
        # Use trivial cache dimensions; the cache is unused by this dummy model
        kv_heads = Axis("kv_head", 1)
        head_size = Axis("embed", 1)
        return KvPageCache.init(spec, kv_heads, head_size, dtype=dtype)

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
