# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import random
from transformers import Qwen3_5MoeConfig as HfQwen3_5MoeConfig
from transformers import Qwen3_5MoeTextConfig as HfQwen3_5MoeTextConfig

import haliax as hax
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeLMHeadModel
from levanter.testing.helpers import skip_if_no_torch, use_test_mesh

VOCAB_SIZE = 128
SEQ_LEN = 20
# Shorter than SEQ_LEN so sequences span several delta-rule chunks. HF always uses 64.
GDN_CHUNK_SIZE = 8


def _tiny_hf_text_config() -> HfQwen3_5MoeTextConfig:
    return HfQwen3_5MoeTextConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQ_LEN,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_experts=8,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
    )


def _tiny_config() -> Qwen35MoeConfig:
    # HF's forward adds no router load-balancing loss, so parity checks disable it.
    return dataclasses.replace(
        Qwen35MoeConfig.from_hf_config(_tiny_hf_text_config()),
        gdn_chunk_size=GDN_CHUNK_SIZE,
        router_aux_loss_coef=None,
    )


def _input_ids(batch_size: int = 2) -> hax.NamedArray:
    Batch = hax.Axis("batch", batch_size)
    Pos = hax.Axis("position", SEQ_LEN)
    return hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, VOCAB_SIZE)


@hax.named_jit
def _logits(model: Qwen35MoeLMHeadModel, input_ids: hax.NamedArray) -> jnp.ndarray:
    return model(input_ids, attn_mask=AttentionMask.causal()).rearrange(("batch", "position", "vocab")).array


def _mean_next_token_loss(model: Qwen35MoeLMHeadModel, input_ids: hax.NamedArray) -> jnp.ndarray:
    example = LmExample(
        tokens=input_ids, loss_weight=hax.ones(input_ids.axes, dtype=jnp.float32), attn_mask=AttentionMask.causal()
    )
    return model.compute_next_token_loss(example).array


def _randomized_torch_model(hf_config: HfQwen3_5MoeTextConfig):
    """An HF model whose norm scales, decay parameters and gates are away from their initial values."""
    import torch  # noqa: PLC0415
    from transformers import Qwen3_5MoeForCausalLM  # noqa: PLC0415

    torch.random.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(hf_config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name or name.endswith(("A_log", "dt_bias")):
                parameter.add_(0.3 * torch.randn_like(parameter))
            else:
                parameter.mul_(4.0)
    return model


def test_state_dict_uses_hf_layer_layout_and_round_trips():
    config = _tiny_config()
    with use_test_mesh():
        model = Qwen35MoeLMHeadModel.init(hax.Axis("vocab", VOCAB_SIZE), config, key=random.PRNGKey(0))
        state_dict = to_torch_compatible_state_dict(model)
        reloaded = to_torch_compatible_state_dict(from_torch_compatible_state_dict(model, state_dict))

    # Layers 0, 1, 3 and 4 are linear attention; layers 2 and 5 are full attention.
    assert state_dict["model.layers.1.linear_attn.in_proj_qkv.weight"].shape == (64, 32)
    assert state_dict["model.layers.4.linear_attn.conv1d.weight"].shape == (64, 1, 4)
    assert state_dict["model.layers.4.linear_attn.A_log"].shape == (4,)
    # q_proj also carries the output gate, so it has twice the rows of o_proj's input.
    assert state_dict["model.layers.2.self_attn.q_proj.weight"].shape == (64, 32)
    assert state_dict["model.layers.5.self_attn.o_proj.weight"].shape == (32, 32)
    assert state_dict["model.layers.5.mlp.experts.gate_up_proj"].shape == (8, 32, 32)
    assert state_dict["model.layers.0.mlp.experts.down_proj"].shape == (8, 32, 16)
    assert state_dict["model.layers.0.mlp.shared_expert_gate.weight"].shape == (1, 32)
    assert "model.layers.2.linear_attn.in_proj_qkv.weight" not in state_dict
    assert "model.layers.6.input_layernorm.weight" not in state_dict

    assert state_dict.keys() == reloaded.keys()
    for key, value in state_dict.items():
        np.testing.assert_array_equal(np.asarray(reloaded[key]), np.asarray(value), err_msg=key)


def test_logits_are_causal_and_independent_of_gdn_chunk_size():
    config = _tiny_config()
    input_ids = _input_ids()
    changed_position = SEQ_LEN - 5
    changed_ids = input_ids.at[{"position": changed_position}].set((input_ids["position", changed_position] + 1) % 128)

    with use_test_mesh():
        model = Qwen35MoeLMHeadModel.init(hax.Axis("vocab", VOCAB_SIZE), config, key=random.PRNGKey(0))
        logits = np.asarray(_logits(model, input_ids))
        changed_logits = np.asarray(_logits(model, changed_ids))

        whole_sequence_chunk = dataclasses.replace(config, gdn_chunk_size=64)
        rechunked = Qwen35MoeLMHeadModel.init(
            hax.Axis("vocab", VOCAB_SIZE), whole_sequence_chunk, key=random.PRNGKey(0)
        )
        rechunked_logits = np.asarray(_logits(rechunked, input_ids))

    np.testing.assert_array_equal(changed_logits[:, :changed_position], logits[:, :changed_position])
    assert not np.allclose(changed_logits[:, changed_position:], logits[:, changed_position:])
    np.testing.assert_allclose(rechunked_logits, logits, rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_logits_loss_and_gradients_match_hf_and_export_reloads_in_hf(local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    hf_config = _tiny_hf_text_config()
    config = _tiny_config()
    # Local tokenizer + no remote reference keeps the conversion off the Hub; the
    # tokenizer is incidental (random inputs, logit-equivalence only).
    converter = dataclasses.replace(
        config, reference_checkpoint=None, tokenizer=local_gpt2_tokenizer_path
    ).hf_checkpoint_converter()
    input_ids = _input_ids()
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.long)

    torch_model = _randomized_torch_model(hf_config)
    torch_logits = torch_model(input_torch).logits
    torch_loss = F.cross_entropy(torch_logits[:, :-1].reshape(-1, VOCAB_SIZE), input_torch[:, 1:].reshape(-1))
    torch_loss.backward()
    torch_grads = {name: parameter.grad.numpy() for name, parameter in torch_model.named_parameters()}

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        torch_model.save_pretrained(f"{tmpdir}/torch_model")
        model = converter.load_pretrained(
            Qwen35MoeLMHeadModel, ref=f"{tmpdir}/torch_model", config=config, resize_vocab_to_match_tokenizer=False
        )
        logits = _logits(model, input_ids)
        loss, grads = hax.named_jit(eqx.filter_value_and_grad(_mean_next_token_loss))(model, input_ids)
        grads = to_torch_compatible_state_dict(grads)

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_tokenizer=False)
        reloaded_torch_model = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model").eval()

    np.testing.assert_allclose(np.asarray(logits), torch_logits.detach().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(loss), torch_loss.item(), rtol=1e-4, atol=1e-4)

    assert grads.keys() == torch_grads.keys()
    for name, torch_grad in torch_grads.items():
        np.testing.assert_allclose(np.asarray(grads[name]), torch_grad, rtol=1e-4, atol=1e-4, err_msg=name)

    with torch.no_grad():
        reloaded_logits = reloaded_torch_model(input_torch).logits.numpy()
    np.testing.assert_allclose(reloaded_logits, torch_logits.detach().numpy(), rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_loads_text_decoder_from_vision_language_checkpoint(local_gpt2_tokenizer_path):
    """Released checkpoints are vision-language models with the decoder under `model.language_model`."""
    import torch  # noqa: PLC0415
    from transformers import Qwen3_5MoeForConditionalGeneration  # noqa: PLC0415

    text_config = _tiny_hf_text_config()
    hf_config = HfQwen3_5MoeConfig(
        text_config=text_config.to_dict(),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32),
    )
    config = dataclasses.replace(
        Qwen35MoeConfig.from_hf_config(hf_config), gdn_chunk_size=GDN_CHUNK_SIZE, router_aux_loss_coef=None
    )
    converter = dataclasses.replace(
        config, reference_checkpoint=None, tokenizer=local_gpt2_tokenizer_path
    ).hf_checkpoint_converter()
    input_ids = _input_ids()

    torch.random.manual_seed(0)
    torch_model = Qwen3_5MoeForConditionalGeneration(hf_config).eval()
    with torch.no_grad():
        torch_logits = torch_model(input_ids=torch.from_numpy(np.array(input_ids.array)).to(torch.long)).logits

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        torch_model.save_pretrained(tmpdir)
        model = converter.load_pretrained(
            Qwen35MoeLMHeadModel, ref=tmpdir, config=config, resize_vocab_to_match_tokenizer=False
        )
        logits = _logits(model, input_ids)

    np.testing.assert_allclose(np.asarray(logits), torch_logits.numpy(), rtol=1e-4, atol=1e-4)
