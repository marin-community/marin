# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest
from haliax import Axis

import levanter.inference.engine as engine_module
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.page_table import PageTableSpec
from levanter.layers.kv_cache import KvPageCache
from levanter.utils.mesh import create_mesh_from_axis_specs


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
        # The cache contents are unused by this dummy model, so only its shardability matters.
        # `KvPageCache.init` packs K and V into a single `2 * kv_heads` axis, so one kv head per
        # device keeps that axis divisible by a `model` mesh axis sized to the device count.
        kv_heads = Axis("kv_head", jax.device_count())
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


def test_auto_page_sizing_uses_explicit_axis_resources(monkeypatch):
    # Device memory statistics are an accelerator I/O boundary. Supply a deterministic ample
    # budget so this test reaches the useful page-capacity bound on any host.
    monkeypatch.setattr(engine_module, "estimated_free_device_memory", lambda _device: 100.0)
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})

    with hax.partitioning.set_mesh(mesh):
        service = InferenceEngine.from_model_with_config(
            model=DummyModel(vocab_size=10),  # type: ignore
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=32,
                max_seqs=8,
                page_size=8,
                compute_dtype=jnp.float32,
                max_queued_tokens=64,
                max_seqs_in_prefill=4,
            ),
            axis_resources={"kv_head": "model"},
        )

    assert service.config.max_pages is not None
    assert service.config.max_pages == service.config.max_seqs * service.config.max_pages_per_seq


@pytest.mark.parametrize("max_rounds", [6, 32])
def test_generate_decodes_through_short_context_limit(max_rounds):
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(
            model=DummyModel(vocab_size=10),  # type: ignore
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=10,
                max_pages=4,
                page_size=4,
                max_rounds=max_rounds,
                max_seqs=2,
                max_seqs_in_prefill=2,
                max_queued_tokens=4,
                max_prefill_size=4,
            ),
        )
        params = SeqDecodingParams.default()
        result = engine.generate([Request([1], request_id=0, decode_params=params, n_generations=1)])
    assert result.tokens == [[3] * 9]


def test_generate_admits_waiting_requests_with_reused_pages():
    output_lengths = (1, 5, 2, 4, 3)
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(
            model=DummyModel(vocab_size=10),  # type: ignore
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=8,
                max_pages=4,
                page_size=4,
                max_rounds=2,
                max_seqs=2,
                max_seqs_in_prefill=1,
                max_queued_tokens=4,
                max_prefill_size=4,
            ),
        )
        requests = [
            Request(
                [1, 2, 4],
                request_id=i,
                decode_params=replace(SeqDecodingParams.default(), max_num_tokens=jnp.asarray(3 + length)),
                n_generations=1,
            )
            for i, length in enumerate(output_lengths)
        ]
        result = engine.generate(requests)

    assert result.tokens == [[3] * length for length in output_lengths]
    assert result.total_generated == sum(output_lengths)
