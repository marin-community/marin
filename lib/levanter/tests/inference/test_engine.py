# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import haliax as hax
import jax
import jax.numpy as jnp
import pytest

import levanter.inference.engine as engine_module
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import FinishReason, SeqDecodingParams
from levanter.inference.utils import INVALID
from levanter.utils.mesh import create_mesh_from_axis_specs


def test_auto_page_sizing_uses_explicit_axis_resources(monkeypatch, fixed_token_model):
    # Device memory statistics are an accelerator I/O boundary. Supply a deterministic ample
    # budget so this test reaches the useful page-capacity bound on any host.
    monkeypatch.setattr(engine_module, "estimated_free_device_memory", lambda _device: 100.0)
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})

    with hax.partitioning.set_mesh(mesh):
        service = InferenceEngine.from_model_with_config(
            model=fixed_token_model(vocab_size=10, token_id=3),  # type: ignore
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
def test_generate_decodes_through_short_context_limit(max_rounds, fixed_token_model):
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(
            model=fixed_token_model(vocab_size=10, token_id=3),  # type: ignore
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


def test_generate_admits_waiting_requests_with_reused_pages(fixed_token_model):
    output_lengths = (1, 5, 2, 4, 3)
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(
            model=fixed_token_model(vocab_size=10, token_id=3),  # type: ignore
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
        stops = hax.named(jnp.array([[INVALID, 3], [4, 5]]), ("stop_seq", "position"))
        requests = [
            Request(
                [1, 2, 4],
                request_id=i,
                decode_params=replace(
                    SeqDecodingParams.default(),
                    max_num_tokens=jnp.asarray(3 + length),
                    stop_tokens=stops if i == 0 else None,
                ),
                n_generations=2 if i == 0 else 1,
            )
            for i, length in enumerate(output_lengths)
        ]
        result = engine.generate(requests)

    assert result.tokens == [[3], *[[3] * length for length in output_lengths]]
    assert result.finish_reasons == [FinishReason.STOP, FinishReason.STOP, *[FinishReason.LENGTH] * 4]
    assert result.total_generated == 1 + sum(output_lengths)


def test_generate_reports_eos_at_the_token_limit(local_gpt2_tokenizer, fixed_token_model):
    eos = local_gpt2_tokenizer.eos_token_id
    assert eos is not None
    engine = InferenceEngine.from_model_with_config(
        model=fixed_token_model(vocab_size=len(local_gpt2_tokenizer), token_id=eos),  # type: ignore
        tokenizer=local_gpt2_tokenizer,
        config=InferenceEngineConfig(
            max_seq_len=4,
            max_pages=2,
            page_size=4,
            max_seqs=1,
            max_seqs_in_prefill=1,
            max_queued_tokens=4,
        ),
    )
    params = replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(2))
    result = engine.generate([Request([1], request_id=0, decode_params=params, n_generations=1)])

    assert result.tokens == [[eos]]
    assert result.finish_reasons == [FinishReason.STOP]


def test_continuous_generation_finishes_a_new_short_request_while_the_first_is_active(fixed_token_model):
    mesh = create_mesh_from_axis_specs(ici_axes={"model": jax.device_count()}, dcn_axes={})
    with hax.partitioning.set_mesh(mesh):
        engine = InferenceEngine.from_model_with_config(
            model=fixed_token_model(vocab_size=10, token_id=3),  # type: ignore
            tokenizer=None,
            config=InferenceEngineConfig(
                max_seq_len=8,
                max_pages=4,
                page_size=4,
                max_rounds=1,
                max_seqs=2,
                max_seqs_in_prefill=1,
                max_queued_tokens=4,
                max_prefill_size=4,
            ),
        )
        params = SeqDecodingParams.default()
        long_request = Request([1, 2, 4], 10, replace(params, max_num_tokens=jnp.asarray(8)), n_generations=1)
        short_request = Request([3, 2, 1], 11, replace(params, max_num_tokens=jnp.asarray(4)), n_generations=1)
        incoming = [short_request]
        completed = {}

        def poll(free_slots):
            if not incoming:
                return []
            assert free_slots == 1
            assert engine.results[10][0].tokens_decoded > 0 and not engine.results[10][0].done
            return [incoming.pop()]

        engine.generate_continuously([long_request], poll_requests=poll, on_complete=completed.__setitem__)

    assert list(completed) == [11, 10]
    assert completed[11].tokens == [[3]]
    assert completed[10].tokens == [[3] * 5]
