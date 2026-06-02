# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.inference.jit_scheduler import SequenceTable
from levanter.inference.page_table import PageTable
from levanter.inference.tpu_kernels import (
    TpuPagedAttentionBackend,
    TpuPagedAttentionConfig,
    available_tpu_paged_attention_backends,
)
from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from test_utils import skip_if_no_torch, use_test_mesh


def _hf_qwen_config(vocab_size=151936):
    """Return a tiny transformers Qwen2Config tweaked for tests but with qk-norm on."""
    from transformers.models.qwen3 import Qwen3Config

    cfg_dict = {
        "architectures": ["Qwen3LMHeadModel"],
        "hidden_size": 16,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 128,
        "vocab_size": vocab_size,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "no_bias": True,
    }
    return Qwen3Config(**cfg_dict)  # type: ignore


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        scan_layers=True,
    )


def _qwen_decode_backends() -> tuple[TpuPagedAttentionBackend, ...]:
    available = available_tpu_paged_attention_backends()
    return (TpuPagedAttentionBackend.AUTO, *available)


@skip_if_no_torch
def test_qwen3_roundtrip():
    import torch
    from transformers.models.qwen3 import Qwen3ForCausalLM

    Vocab = hax.Axis("vocab", 151936)
    hf_config = _hf_qwen_config(Vocab.size)

    # Levanter config from HF
    config = Qwen3Config.from_hf_config(hf_config)  # type: ignore

    converter = config.hf_checkpoint_converter()  # type: ignore

    # Inputs
    input_ids = hax.random.randint(random.PRNGKey(0), config.max_Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int32).unsqueeze(0)

    # Torch reference
    torch_model = Qwen3ForCausalLM(hf_config)
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(input_torch).logits[0].detach().cpu().numpy()

    # Save HF model then load with levanter
    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            Qwen3LMHeadModel, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        def compute(mdl, inp):
            return mdl(inp, attn_mask=attn_mask).array

        jax_out = compute(model, input_ids)

        assert torch_out.shape == jax_out.shape
        np.testing.assert_allclose(torch_out, jax_out, rtol=1e-4, atol=1e-4)

        # now save the levanter model and load it as hf
        with tempfile.TemporaryDirectory() as save_dir:
            converter.save_pretrained(model, save_dir)
            with open(f"{save_dir}/config.json", "r") as f:
                saved_config = json.load(f)
            assert saved_config["vocab_size"] == Vocab.size

            hf_model = Qwen3ForCausalLM.from_pretrained(save_dir)
            hf_out = hf_model(input_torch).logits[0].detach().cpu().numpy()
            assert hf_out.shape == jax_out.shape
            np.testing.assert_allclose(hf_out, jax_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", _qwen_decode_backends())
def test_qwen3_paged_decode_matches_full_logits_for_available_backends(backend: TpuPagedAttentionBackend):
    Vocab = hax.Axis("vocab", 32)
    Pos = hax.Axis("position", 6)
    config = _tiny_qwen3_config()

    with use_test_mesh():
        model = Qwen3LMHeadModel.init(Vocab, config, key=random.PRNGKey(0))
        input_ids = hax.named(jnp.asarray([1, 7, 5, 9, 3, 11], dtype=jnp.int32), Pos)
        full_logits = model(input_ids, attn_mask=AttentionMask.causal())

        page_table = PageTable.init(max_pages=4, max_seqs=1, page_size=4, max_pages_per_seq=2)
        sequences = SequenceTable.init(page_table.max_seqs, page_table.pages_per_seq, page_table.page_size)
        sequences, seq_id_arr = sequences.reserve_slot(0)
        seq_id = int(seq_id_arr)
        kv_cache = model.initial_cache(page_table.spec(), dtype=jnp.float32)

        logits_chunks = []
        for start, chunk_size in [(0, 2), (2, 1), (3, 3)]:
            ChunkPos = hax.Axis("position", chunk_size)
            slot_ids = hax.named([seq_id] * chunk_size, ChunkPos)
            pos_ids = hax.arange(ChunkPos, start=start, dtype=jnp.int32)
            token_chunk = input_ids[Pos, hax.dslice(start, chunk_size)]

            sequences, page_table, batch_info = sequences.allocate_for_seq(page_table, slot_ids, pos_ids)
            logits_chunk, kv_cache = model.decode(
                token_chunk,
                kv_cache,
                batch_info,
                pos_ids,
                tpu_paged_attention=TpuPagedAttentionConfig(backend=backend),
            )
            logits_chunks.append(logits_chunk)

        decode_logits = hax.concatenate("position", logits_chunks)

    assert decode_logits.axes == full_logits.axes
    np.testing.assert_allclose(decode_logits.array, full_logits.array, rtol=1e-4, atol=1e-4)
