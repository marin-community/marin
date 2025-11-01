# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import tempfile

import chex
import jax
import numpy as np
import pytest
import transformers
from jax import random
from transformers import Gemma2Config as HFGemma2Config
from transformers import Gemma3TextConfig as HFGemma3Config

import haliax as hax

from levanter.layers.attention import Attention, AttentionMask
from levanter.models.gemma import (
    Gemma2Config,
    Gemma3Config,
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaLMHeadModel,
    GemmaRMSNorm,
)
from levanter.models.llama import LlamaMlp
from levanter.utils.jax_utils import parameter_count
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs, skip_if_no_torch


# N.B. Gemma uses LLamaAttention directly so we skip tests for attention and rotary embeddings.


@skip_if_no_torch
def test_gemma_config():
    # load HF config and convert to levanter config
    hf_config = transformers.GemmaConfig.from_pretrained("google/gemma-2b")
    gemma_config = GemmaConfig.from_hf_config(hf_config)

    # convert back to HF config
    config_overrides = {
        "_name_or_path": hf_config._name_or_path,
        "architectures": hf_config.architectures,
        "torch_dtype": hf_config.torch_dtype,
    }
    new_hf_config = gemma_config.to_hf_config(
        vocab_size=hf_config.vocab_size,
        config_overrides=config_overrides,
    )

    # Gemma has some weird patched behavior in the HF configuration to deal with the original
    # version not using an approximate gelu layer: the configuration has both `hidden_act` and
    # `hidden_activation` fields. We don't touch the `hidden_act` field, and it is overridden
    # by `hidden_activation`.
    # See https://github.com/huggingface/transformers/pull/29402 for more info.
    assert gemma_config.activation_function == "gelu_new"  # gelu_new is a closer match to gelu_pytorch_tanh
    assert new_hf_config.hidden_activation == "gelu_pytorch_tanh"

    # assert the content in new_hf_config is the same as hf_config
    for k in new_hf_config.__dict__.keys():
        if k in ["_commit_hash", "transformers_version", "_attn_implementation_internal"]:
            continue

        if k in ["hidden_act", "hidden_activation"]:
            continue

        assert getattr(new_hf_config, k) == getattr(
            hf_config, k
        ), f"{k} {getattr(new_hf_config, k)} != {getattr(hf_config, k)}"


def test_gemma_param_counts_dont_change_with_seqlen():
    model = GemmaLMHeadModel.init(hax.Axis("v", 2048), _get_gemma_config(seq_len=128), key=random.PRNGKey(0))
    model2 = GemmaLMHeadModel.init(hax.Axis("v", 2048), _get_gemma_config(seq_len=256), key=random.PRNGKey(0))
    assert parameter_count(model) == parameter_count(model2)


@skip_if_no_torch
@pytest.mark.parametrize("gemma_version", [1, 2])
def test_gemma_rms_norm(gemma_version):
    import torch

    if gemma_version == 1:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm

        config = _get_gemma_config()
    else:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm as HFGemmaRMSNorm

        config = _get_gemma2_config()

    ln = GemmaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
    hf_ln = HFGemmaRMSNorm(config.Embed.size, eps=config.layer_norm_epsilon)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = ln(x)
    hf_out = hf_ln(x_torch)

    assert np.isclose(
        hf_out.detach().cpu().numpy(), np.array(out.array), rtol=1e-6, atol=1e-6
    ).all(), f"{hf_out} != {out}"


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_gemma1_decoder_layer(num_kv_heads):
    """Validate Levanter Gemma-1 decoder layer against HF reference."""
    import torch
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as HFGemmaDecoderLayer
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding as HFGemmaRotaryEmbedding

    gemma_config = _get_gemma_config(num_kv_heads=num_kv_heads)
    LevDecoderLayer = GemmaDecoderLayer

    key = random.PRNGKey(0)
    decoder_layer = LevDecoderLayer.init(config=gemma_config, key=key)

    # copy weights
    state = hax.state_dict.to_torch_compatible_state_dict(decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

    hf_config = gemma_config.to_hf_config(32000)
    hf_decoder = HFGemmaDecoderLayer(hf_config, layer_idx=0)
    hf_decoder.load_state_dict(state, strict=True)

    # run forward
    x, mask = _get_random_inputs(gemma_config)
    x_t = torch.from_numpy(np.array(x.array))
    batch = x_t.shape[0]
    bias = torch.from_numpy(np.array(mask.materialize(gemma_config.Pos, gemma_config.KeyPos).array))
    bias = bias.broadcast_to((batch, 1, -1, -1))
    bias = (bias == 0).float() * -1e10

    position_ids = torch.arange(gemma_config.Pos.size).unsqueeze(0)
    rot = HFGemmaRotaryEmbedding(config=hf_config)
    cos, sin = rot(x_t, position_ids)

    out = decoder_layer(x, mask)
    hf_out = hf_decoder(
        x_t,
        attention_mask=bias,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    # Handle the case where HF returns separate batch elements vs single tensor
    if isinstance(hf_out, torch.Tensor):
        hf_array = hf_out.detach().cpu().numpy()
    else:
        hf_stacked = torch.stack(hf_out)
        hf_array = hf_stacked.detach().cpu().numpy()

    chex.assert_trees_all_close(hf_array, out.array, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_gemma2_decoder_layer(num_kv_heads):
    """Validate Levanter Gemma-2 decoder layer against HF reference."""
    import torch
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer as HFGemmaDecoderLayer
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding as HFGemmaRotaryEmbedding

    from levanter.models.gemma import Gemma2DecoderLayer as LevDecoderLayer  # local to avoid circular import at top

    gemma_config = _get_gemma2_config(num_kv_heads=num_kv_heads)

    key = random.PRNGKey(0)
    decoder_layer = LevDecoderLayer.init(config=gemma_config, key=key)

    state = hax.state_dict.to_torch_compatible_state_dict(decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

    hf_config = gemma_config.to_hf_config(32000)
    hf_decoder = HFGemmaDecoderLayer(hf_config, layer_idx=0)
    hf_decoder.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(gemma_config)
    x_t = torch.from_numpy(np.array(x.array))
    batch = x_t.shape[0]
    bias = torch.from_numpy(np.array(mask.materialize(gemma_config.Pos, gemma_config.KeyPos).array))
    bias = bias.broadcast_to((batch, 1, -1, -1))
    bias = (bias == 0).float() * -1e10

    position_ids = torch.arange(gemma_config.Pos.size).unsqueeze(0)
    rot = HFGemmaRotaryEmbedding(config=hf_config)
    cos, sin = rot(x_t, position_ids)

    lev_out = decoder_layer(x, mask)
    hf_out = hf_decoder(
        x_t,
        attention_mask=bias,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        cache_position=torch.zeros((batch,), dtype=torch.int32),  # HF expects a cache position
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), lev_out.array, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pass_different_length_seq(num_kv_heads):
    config = GemmaConfig(
        seq_len=64,
        hidden_dim=64,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=num_kv_heads,
        use_flash_attention=True,
        head_dim=4,
    )
    check_model_works_with_seqlen(GemmaLMHeadModel, config, 16)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("gemma_version", [1, 2])
def test_gemma_attention(use_flash, num_kv_heads, gemma_version):
    import torch

    if gemma_version == 1:
        from transformers.models.gemma.modeling_gemma import GemmaAttention as HFGemmaAttention
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding as HFGemmaRotaryEmbedding

        config = _get_gemma_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    else:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention as HFGemmaAttention
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding as HFGemmaRotaryEmbedding

        config = _get_gemma2_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

    attention = Attention.init(config.attention_config(), key=random.PRNGKey(0))

    state = hax.state_dict.to_torch_compatible_state_dict(attention)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_config = config.to_hf_config(32000)

    hf_rotary_emb = HFGemmaRotaryEmbedding(config=hf_config)
    hf_attention = HFGemmaAttention(hf_config, layer_idx=0)
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(config.Pos, config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))

    # the torch mask is really a bias, so we need to invert it and make it a big negative number
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    position_ids = torch.arange(config.Pos.size).unsqueeze(0)  # [1, seq_len]
    cos, sin = hf_rotary_emb(x_torch, position_ids)
    hf_out = hf_attention(
        x_torch, position_ids=position_ids, attention_mask=mask_torch, position_embeddings=(cos, sin)
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), out.array, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_gemma_mlp():
    import torch
    from transformers.models.gemma.modeling_gemma import GemmaMLP as HFGemmaMLP

    config = _get_gemma_config()
    mlp = LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=random.PRNGKey(0))

    state = hax.state_dict.to_torch_compatible_state_dict(mlp)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_mlp = HFGemmaMLP(config.to_hf_config(32000))
    hf_mlp.load_state_dict(state, strict=True)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = mlp(x)
    hf_out = hf_mlp(x_torch)

    chex.assert_trees_all_close(hf_out.detach().cpu().numpy(), out.array, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_gemma2_mlp():
    import torch
    from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP as HFGemmaMLP

    config = _get_gemma2_config()
    mlp = LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=random.PRNGKey(0))

    state = hax.state_dict.to_torch_compatible_state_dict(mlp)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_mlp = HFGemmaMLP(config.to_hf_config(32000))
    hf_mlp.load_state_dict(state, strict=True)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = mlp(x)
    hf_out = hf_mlp(x_torch)

    chex.assert_trees_all_close(hf_out.detach().cpu().numpy(), out.array, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_gemma2_roundtrip():
    import torch
    from transformers import AutoModelForCausalLM, Gemma2ForCausalLM

    config = Gemma2Config(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=4,
        gradient_checkpointing=False,
        head_dim=4,
        query_pre_attn_scalar=4,
        num_layers=2,
    )
    converter = config.hf_checkpoint_converter()

    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)
    assert isinstance(hf_config, HFGemma2Config)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = Gemma2ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            converter.default_config.model_type, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        def compute(input):
            model_output = model(input, attn_mask=attn_mask)
            return hax.nn.softmax(model_output, axis=model.Vocab)

        with jax.disable_jit():
            jax_out = compute(input).array

        chex.assert_trees_all_close(jax_out, torch_out, rtol=1e-3, atol=1e-3)

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        assert np.isclose(torch_out2, np.array(jax_out), rtol=1e-3, atol=1e-3).all(), f"{torch_out2} != {jax_out}"


def _get_gemma_config(use_flash=False, num_kv_heads=4, seq_len=128) -> GemmaConfig:
    return GemmaConfig(
        seq_len=seq_len,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
        head_dim=4,
    )


def _get_gemma2_config(use_flash=False, num_kv_heads=4, seq_len=128) -> Gemma2Config:
    from levanter.models.gemma import Gemma2Config

    return Gemma2Config(
        seq_len=seq_len,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
        head_dim=4,
        query_pre_attn_scalar=4,
        attn_logit_softcapping=4,
        final_logit_softcapping=None,
        sliding_window=seq_len,
    )


def _get_gemma3_config(use_flash=False, num_kv_heads=4, seq_len=128) -> Gemma3Config:
    from levanter.models.gemma import Gemma3Config

    return Gemma3Config(
        seq_len=seq_len,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
        head_dim=4,
        query_pre_attn_scalar=4,
        sliding_window=seq_len,
    )


def _get_random_inputs(config: GemmaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = AttentionMask.causal()
    return x, mask


@parameterize_with_configs("gemma*.yaml")
def test_gemma_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


# ---------------- Gemma-3 ----------------

# Reuse Gemma2DecoderLayer implementation under the hood


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_gemma3_decoder_layer(num_kv_heads):
    """Validate Levanter Gemma-3 decoder layer against HF reference."""
    import torch
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer as HFGemmaDecoderLayer
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding as HFGemmaRotaryEmbedding

    from levanter.models.gemma import Gemma2DecoderLayer as LevDecoderLayer  # Gemma3 reuses implementation

    gemma_config = _get_gemma3_config(num_kv_heads=num_kv_heads)

    key = random.PRNGKey(0)
    decoder_layer = LevDecoderLayer.init(config=gemma_config, key=key)

    state = hax.state_dict.to_torch_compatible_state_dict(decoder_layer)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

    hf_config = gemma_config.to_hf_config(32000)
    hf_decoder = HFGemmaDecoderLayer(hf_config, layer_idx=0)
    hf_decoder.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(gemma_config)
    x_t = torch.from_numpy(np.array(x.array))
    batch = x_t.shape[0]
    bias = torch.from_numpy(np.array(mask.materialize(gemma_config.Pos, gemma_config.KeyPos).array))
    bias = bias.broadcast_to((batch, 1, -1, -1))
    bias = (bias == 0).float() * -1e10

    position_ids = torch.arange(gemma_config.Pos.size).unsqueeze(0)
    rot = HFGemmaRotaryEmbedding(config=hf_config)
    cos, sin = rot(x_t, position_ids)

    # HF does this hacky crap so we copy it
    local_hf_config = copy.deepcopy(hf_config)
    local_hf_config.rope_theta = local_hf_config.rope_local_base_freq
    local_hf_config.rope_scaling = {"rope_type": "default"}
    local_rot = HFGemmaRotaryEmbedding(config=local_hf_config)
    local_cos, local_sin = local_rot(x_t, position_ids)

    lev_out = decoder_layer(x, mask)
    hf_out = hf_decoder(
        x_t,
        attention_mask=bias,
        position_ids=position_ids,
        position_embeddings_global=(cos, sin),
        position_embeddings_local=(local_cos, local_sin),
        cache_position=torch.zeros((batch,), dtype=torch.int32),
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), lev_out.array, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_gemma3_roundtrip():
    import torch
    from transformers import AutoModelForCausalLM, Gemma3ForCausalLM

    config = Gemma3Config(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=4,
        gradient_checkpointing=False,
        head_dim=4,
        query_pre_attn_scalar=4,
        num_layers=2,
    )

    converter = config.hf_checkpoint_converter()

    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)
    assert isinstance(hf_config, HFGemma3Config)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = Gemma3ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            converter.default_config.model_type, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        def compute(input):
            model_output = model(input, attn_mask=attn_mask)
            return hax.nn.softmax(model_output, axis=model.Vocab)

        with jax.disable_jit():
            jax_out = compute(input).array

        chex.assert_trees_all_close(jax_out, torch_out, rtol=1e-3, atol=1e-3)

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        assert np.isclose(torch_out2, np.array(jax_out), rtol=1e-3, atol=1e-3).all(), f"{torch_out2} != {jax_out}"


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_gemma3_attention(use_flash, num_kv_heads):
    import torch
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention as HFGemmaAttention
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding as HFGemmaRotaryEmbedding

    gemma_config = _get_gemma3_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

    attention = Attention.init(gemma_config.attention_config(), key=random.PRNGKey(0))

    state = hax.state_dict.to_torch_compatible_state_dict(attention)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

    hf_config = gemma_config.to_hf_config(32000)

    hf_rotary_emb = HFGemmaRotaryEmbedding(config=hf_config)
    hf_attention = HFGemmaAttention(hf_config, layer_idx=0)
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(gemma_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(gemma_config.Pos, gemma_config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))

    # the torch mask is really a bias, so we need to invert it and make it a big negative number
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    position_ids = torch.arange(gemma_config.Pos.size).unsqueeze(0)  # [1, seq_len]
    cos, sin = hf_rotary_emb(x_torch, position_ids)
    hf_out = hf_attention(
        x_torch, position_ids=position_ids, attention_mask=mask_torch, position_embeddings=(cos, sin)
    )

    chex.assert_trees_all_close(hf_out[0].detach().cpu().numpy(), out.array, rtol=1e-5, atol=1e-5)
