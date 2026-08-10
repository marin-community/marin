# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import haliax as hax
import numpy as np
import pytest
from jax import random
from test_utils import skip_if_module_missing, skip_if_no_torch, use_test_mesh
from transformers import AutoModelForMaskedLM
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig as HfModernBertConfig

from levanter.models.modernbert import ModernBertConfig, ModernBertForMaskedLM

pytestmark = skip_if_module_missing("transformers.models.modernbert.modeling_modernbert")


def _make_hf_config(vocab_size: int, tie_word_embeddings: bool) -> HfModernBertConfig:
    return HfModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=256,
        norm_eps=1e-5,
        norm_bias=False,
        attention_bias=False,
        mlp_bias=False,
        decoder_bias=True,
        classifier_bias=False,
        hidden_activation="gelu",
        classifier_activation="gelu",
        global_attn_every_n_layers=3,
        local_attention=16,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        _attn_implementation="eager",
    )


def _make_levanter_config(tokenizer: str, tie_word_embeddings: bool) -> ModernBertConfig:
    return ModernBertConfig(
        max_seq_len=256,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=4,
        layer_norm_epsilon=1e-5,
        global_attn_every_n_layers=3,
        local_attention=16,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=0,
        tokenizer=tokenizer,
    )


@skip_if_no_torch
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_modernbert_roundtrip(tie_word_embeddings, local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415  # optional dep: torch
    from transformers.models.modernbert import modeling_modernbert  # noqa: PLC0415  # optional dep: torch

    Vocab = hax.Axis("vocab", 4096)
    hf_config = _make_hf_config(Vocab.size, tie_word_embeddings)
    lev_config = _make_levanter_config(local_gpt2_tokenizer_path, tie_word_embeddings)

    input_ids = hax.random.randint(random.PRNGKey(0), lev_config.max_Pos, 0, Vocab.size)
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int64).unsqueeze(0)

    torch.random.manual_seed(0)
    torch_model = modeling_modernbert.ModernBertForMaskedLM(hf_config)
    torch_model.eval()
    torch_out = torch_model(input_ids=input_torch).logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        converter = lev_config.hf_checkpoint_converter(ref_checkpoint=model_path)

        model = converter.load_pretrained(
            ModernBertForMaskedLM,
            ref=model_path,
            resize_vocab_to_match_tokenizer=False,
        )

        @hax.named_jit
        def compute(model, input_ids):
            return model(input_ids)

        jax_out = compute(model, input_ids).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False, save_tokenizer=False)

        torch_model2 = AutoModelForMaskedLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()
        torch_out2 = torch_model2(input_ids=input_torch).logits[0].detach().cpu().numpy()
        np.testing.assert_allclose(torch_out2, jax_out, rtol=1e-4, atol=1e-4)
