import tempfile

import equinox as eqx
import jax
import numpy as np
import pytest
import transformers
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.mistral import MistralConfig, MistralLMHeadModel
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs, skip_if_no_torch


@skip_if_no_torch
def test_mistral_config():
    # load HF config and convert to levanter config
    hf_config = transformers.MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
    mistral_config = MistralConfig.from_hf_config(hf_config)

    # convert back to HF config
    config_overrides = {
        "_name_or_path": hf_config._name_or_path,
        "architectures": hf_config.architectures,
        "torch_dtype": hf_config.torch_dtype,
    }
    new_hf_config = mistral_config.to_hf_config(
        vocab_size=hf_config.vocab_size,
        config_overrides=config_overrides,
    )

    # assert the content in new_hf_config is the same as hf_config
    for k in new_hf_config.__dict__.keys():
        if k in ["_commit_hash", "transformers_version"]:
            continue
        assert getattr(new_hf_config, k) == getattr(
            hf_config, k
        ), f"{k} {getattr(new_hf_config, k)} != {getattr(hf_config, k)}"


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_mistral_lm_head_model(num_kv_heads):
    mistral_config = _get_mistral_config(num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = mistral_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    def fn(input_ids, mask):
        return MistralLMHeadModel.init(Vocab=Vocab, config=mistral_config, key=random.PRNGKey(0))(input_ids, mask)

    out = eqx.filter_eval_shape(fn, input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_mistral_lm_head_model_bwd(use_flash, num_kv_heads):
    llama_config = _get_mistral_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    llama_model = MistralLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))

    def f(llama_model, input_ids, mask):
        out = llama_model(input_ids, mask)
        return hax.sum(out).scalar()

    _, grads = eqx.filter_eval_shape(eqx.filter_value_and_grad(f), llama_model, input_ids, mask)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_mistral_roundtrip(num_kv_heads):
    import torch
    from transformers import AutoModelForCausalLM, MistralForCausalLM

    config = MistralConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
    )
    converter = config.hf_checkpoint_converter()

    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = MistralForCausalLM(hf_config)
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

        compute = jax.jit(compute)
        jax_out = compute(input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        assert np.isclose(torch_out2, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out2} != {jax_out}"


def _get_mistral_config(use_flash=False, num_kv_heads=4) -> MistralConfig:
    return MistralConfig(
        num_layers=2,
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
    )


@parameterize_with_configs("mistral*.yaml")
def test_mistral_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pass_different_length_seq(num_kv_heads):
    config = MistralConfig(
        seq_len=64,
        hidden_dim=32,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=num_kv_heads,
        use_flash_attention=True,
    )
    check_model_works_with_seqlen(MistralLMHeadModel, config, 16)
