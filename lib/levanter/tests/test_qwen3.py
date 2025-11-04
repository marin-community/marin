# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile

import numpy as np
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from test_utils import skip_if_no_torch


def _hf_qwen_config(vocab_size=1000):
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


@skip_if_no_torch
def test_qwen3_roundtrip():
    import torch
    from transformers.models.qwen3 import Qwen3ForCausalLM

    Vocab = hax.Axis("vocab", 1000)
    hf_config = _hf_qwen_config(Vocab.size)

    # Levanter config from HF
    config = Qwen3Config.from_hf_config(hf_config)  # type: ignore

    converter = Qwen3Config().hf_checkpoint_converter()  # type: ignore

    # Inputs
    input_ids = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int32).unsqueeze(0)

    # Torch reference
    torch_model = Qwen3ForCausalLM(hf_config)
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(input_torch).logits[0].detach().cpu().numpy()

    # Save HF model then load with levanter
    with tempfile.TemporaryDirectory() as tmpdir:
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
