# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import chex
import equinox as eqx
import numpy
import numpy as np
import pytest
import transformers
from jax import random

import haliax as hax
import haliax.nn as hnn

from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import Attention, LlamaConfig, LlamaDecoderLayer, LlamaLMHeadModel
from levanter.utils.jax_utils import parameter_count
from test_utils import (
    check_load_config,
    check_model_works_with_seqlen,
    parameterize_with_configs,
    skip_if_hf_model_not_accessible,
    skip_if_no_torch,
)


@skip_if_no_torch
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_llama_config():
    # load HF config and convert to levanter config
    hf_config = transformers.LlamaConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
    llama_config = LlamaConfig.from_hf_config(hf_config)

    # convert back to HF config
    config_overrides = {
        "_name_or_path": hf_config._name_or_path,
        "architectures": hf_config.architectures,
        "torch_dtype": hf_config.torch_dtype,
        "pad_token_id": hf_config.pad_token_id,
    }
    new_hf_config = llama_config.to_hf_config(
        vocab_size=hf_config.vocab_size,
        config_overrides=config_overrides,
    )

    # assert the content in new_hf_config is the same as hf_config
    for k in new_hf_config.__dict__.keys():
        if k in ["_commit_hash", "transformers_version", "_attn_implementation_internal"]:
            continue
        assert getattr(new_hf_config, k) == getattr(
            hf_config, k
        ), f"{k} {getattr(new_hf_config, k)} != {getattr(hf_config, k)}"


def test_llama_flops():
    # Check that the forward flops is within 10% of the naive calculation
    hf_config = transformers.LlamaConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
    llama_config = LlamaConfig.from_hf_config(hf_config)
    n_params = 6.738415616e9
    ratio = llama_config.flops_per_token(hf_config.vocab_size) / (2 * n_params)
    assert ratio > 1.1, f"ratio {ratio} < 1.1"
    assert ratio < 1.2, f"ratio {ratio} > 1.2"


def test_llama_params():
    # Check that the computed number of trainable params is close to the actual number of params
    hf_config = transformers.LlamaConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
    llama_config = LlamaConfig.from_hf_config(hf_config)
    actual_params = 6.738415616e9
    params = llama_config.total_trainable_params(hf_config.vocab_size)
    assert np.isclose(actual_params, params, rtol=1e-2)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_llama_attention(use_flash, num_kv_heads):
    import torch
    from transformers.models.llama.modeling_llama import LlamaAttention as HFLlamaAttention
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    config = _get_llama_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

    attention_config = config.attention_config()
    attention = Attention.init(config=attention_config, key=random.PRNGKey(0))  # type: ignore

    state = hax.state_dict.to_torch_compatible_state_dict(attention)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = config.to_hf_config(32000)

    hf_rotary_emb = HFLlamaRotaryEmbedding(config=hf_config)
    hf_attention = HFLlamaAttention(hf_config, layer_idx=0)
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(config.Pos, config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    position_ids = torch.arange(config.Pos.size).unsqueeze(0)  # [1, seq_len]
    cos, sin = hf_rotary_emb(x_torch, position_ids)  # Pass x_torch instead of zeros tensor
    hf_out = hf_attention(
        x_torch, position_ids=position_ids, attention_mask=mask_torch, position_embeddings=(cos, sin)
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), out.array, rtol=1e-4, atol=1e-4)


def test_llama_param_counts_dont_change_with_seqlen():
    model = LlamaLMHeadModel.init(hax.Axis("v", 2048), _get_llama_config(seq_len=128), key=random.PRNGKey(0))
    model2 = LlamaLMHeadModel.init(hax.Axis("v", 2048), _get_llama_config(seq_len=256), key=random.PRNGKey(0))
    assert parameter_count(model) == parameter_count(model2)


@skip_if_no_torch
def test_llama_rms_norm():
    import torch
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFLlamaRMSNorm

    config = _get_llama_config()
    ln = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
    hf_ln = HFLlamaRMSNorm(config.Embed.size, eps=config.layer_norm_epsilon)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = ln(x)
    hf_out = hf_ln(x_torch)

    assert np.isclose(
        hf_out.detach().cpu().numpy(), np.array(out.array), rtol=1e-6, atol=1e-6
    ).all(), f"{hf_out} != {out}"


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_llama_decoder_layer(num_kv_heads):
    import torch
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLlamaDecoderLayer
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    llama_config = _get_llama_config(num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)
    llama_decoder_layer = LlamaDecoderLayer.init(config=llama_config, key=key)

    state = hax.state_dict.to_torch_compatible_state_dict(llama_decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = llama_config.to_hf_config(32000)
    hf_decoder_layer = HFLlamaDecoderLayer(hf_config, layer_idx=0)
    hf_decoder_layer.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(llama_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(llama_config.Pos, llama_config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e10

    position_ids = torch.arange(llama_config.Pos.size).unsqueeze(0)
    hf_rotary_emb = HFLlamaRotaryEmbedding(config=hf_config)
    cos, sin = hf_rotary_emb(x_torch, position_ids)

    out = llama_decoder_layer(x, mask)
    hf_out = hf_decoder_layer(
        x_torch, attention_mask=mask_torch, position_ids=position_ids, position_embeddings=(cos, sin)
    )

    # Handle the case where HF returns separate batch elements vs single tensor
    if isinstance(hf_out, torch.Tensor):
        # Single tensor case - should preserve batch dimension
        hf_array = hf_out.detach().cpu().numpy()
    else:
        # Multiple tensors case - need to stack them back together
        hf_stacked = torch.stack(hf_out)
        hf_array = hf_stacked.detach().cpu().numpy()

    chex.assert_trees_all_close(hf_array, out.array, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_llama_lm_head_model(num_kv_heads):
    llama_config = _get_llama_config(num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))
    out = llama_model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_llama_lm_head_model_bwd(use_flash, num_kv_heads):
    llama_config = _get_llama_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))

    def f(llama_model, input_ids, mask):
        out = llama_model(input_ids, mask)
        return hax.sum(out).scalar()

    _, grads = eqx.filter_value_and_grad(f)(llama_model, input_ids, mask)


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_llama_roundtrip(scan_layers, num_kv_heads):
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    converter = LlamaConfig().hf_checkpoint_converter()

    config = LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
    )
    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = LlamaForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    # torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            LlamaLMHeadModel, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        @hax.named_jit
        def compute(model, input):
            model_output = model(input, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        # now we're going to magnify the model parameters enough that differences should actualy show up
        jax_out = compute(model, input).array

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        numpy.testing.assert_allclose(torch_out2, jax_out, rtol=1e-5, atol=1e-5)


def _get_llama_config(use_flash=False, num_kv_heads=4, seq_len=128) -> LlamaConfig:
    return LlamaConfig(
        seq_len=seq_len,
        hidden_dim=32,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        attn_backend=AttentionBackend.DEFAULT if use_flash else AttentionBackend.VANILLA,
        flash_attention_block_size=8 if use_flash else None,
    )


def _get_random_inputs(config: LlamaConfig, override_Pos=None):
    Embed = config.Embed
    if override_Pos is not None:
        Pos = override_Pos
    else:
        Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = AttentionMask.causal()
    return x, mask


@parameterize_with_configs("llama*.yaml")
def test_llama_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pass_different_length_seq(num_kv_heads):
    config = LlamaConfig(
        seq_len=64,
        hidden_dim=64,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=num_kv_heads,
    )
    check_model_works_with_seqlen(LlamaLMHeadModel, config, 16)


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_state_dict_consistency(scan_layers, num_kv_heads):
    from transformers import LlamaForCausalLM

    config = LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_layers=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
    )
    Vocab = hax.Axis("vocab", 1000)
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    hf_config = config.to_hf_config(Vocab.size)
    hf_model = LlamaForCausalLM(hf_config)
    levanter_state_dict = hax.state_dict.to_torch_compatible_state_dict(model)
    assert set(hf_model.state_dict().keys()) == set(levanter_state_dict.keys())


@pytest.mark.parametrize("num_kv_heads", [2, 4])
def test_llama_seq_len_doesnt_change_predictions(num_kv_heads):
    config = LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
    )
    Vocab = hax.Axis("vocab", 1000)

    # Make input and attn_mask
    input_256 = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    input_128 = input_256[config.Pos, :128]
    attn_mask = AttentionMask.causal()

    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))

    @hax.named_jit
    def compute(model, input):
        model_output = model(input, attn_mask=attn_mask)
        return model_output

    jax_out_1 = compute(model, input_128)
    jax_out_2 = compute(model, input_256)[config.Pos, :128]

    assert np.allclose(jax_out_1.array, jax_out_2.array, rtol=1e-6, atol=1e-6)
