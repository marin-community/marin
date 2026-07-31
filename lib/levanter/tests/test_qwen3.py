# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import tempfile

import haliax as hax
import jax.numpy as jnp
import numpy as np
from jax import random
from test_utils import skip_if_no_torch, use_test_mesh
from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3Config as HFQwen3Config

from levanter.compat import hf_checkpoints
from levanter.layers.attention import AttentionMask
from levanter.model_loading import load_hf_checkpoint
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel


def _hf_qwen_config(vocab_size=151936):
    """Return a tiny transformers Qwen2Config tweaked for tests but with qk-norm on."""
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
    return HFQwen3Config(**cfg_dict)  # type: ignore


def test_load_hf_checkpoint_uses_supplied_tokenizer(local_gpt2_tokenizer_path, monkeypatch):
    tokenizer = AutoTokenizer.from_pretrained(local_gpt2_tokenizer_path, local_files_only=True)
    config = Qwen3Config.from_hf_config(_hf_qwen_config(len(tokenizer)))  # type: ignore
    local_config = dataclasses.replace(config, reference_checkpoint=None, tokenizer=local_gpt2_tokenizer_path)
    converter = local_config.hf_checkpoint_converter()
    Vocab = hax.Axis("vocab", len(tokenizer))

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model = Qwen3LMHeadModel.init(Vocab, config, key=random.PRNGKey(0))
        converter.save_pretrained(model, tmpdir, save_reference_code=False, save_tokenizer=False)

        def reject_tokenizer_load(*args, **kwargs):
            raise AssertionError("load_hf_checkpoint must use its supplied tokenizer")

        monkeypatch.setattr(hf_checkpoints, "load_tokenizer", reject_tokenizer_load)
        loaded = load_hf_checkpoint(
            config,
            tmpdir,
            axis_mapping={},
            tokenizer=tokenizer,
            compute_dtype=jnp.float32,
            cache_ttl_days=0,
        )

    assert loaded.config == config
    assert loaded.Vocab == Vocab


@skip_if_no_torch
def test_qwen3_roundtrip(local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415  # optional dep: torch
    from transformers.models.qwen3 import Qwen3ForCausalLM  # noqa: PLC0415  # optional dep: torch

    Vocab = hax.Axis("vocab", 151936)
    hf_config = _hf_qwen_config(Vocab.size)

    # Levanter config from HF
    config = Qwen3Config.from_hf_config(hf_config)  # type: ignore

    # Build the converter against a local tokenizer with no remote reference so the
    # roundtrip never touches the Hub. The tokenizer is incidental here: inputs are
    # random ids and we only assert logit equivalence across the conversion.
    converter = dataclasses.replace(
        config, reference_checkpoint=None, tokenizer=local_gpt2_tokenizer_path
    ).hf_checkpoint_converter()  # type: ignore

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
            converter.save_pretrained(model, save_dir, save_tokenizer=False)
            with open(f"{save_dir}/config.json", "r") as f:
                saved_config = json.load(f)
            assert saved_config["vocab_size"] == Vocab.size

            hf_model = Qwen3ForCausalLM.from_pretrained(save_dir)
            hf_out = hf_model(input_torch).logits[0].detach().cpu().numpy()
            assert hf_out.shape == jax_out.shape
            np.testing.assert_allclose(hf_out, jax_out, rtol=1e-4, atol=1e-4)
