# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from transformers import Qwen3MoeConfig as HfQwen3MoeConfig

import haliax as hax
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeLMHeadModel
from levanter.utils.jax_utils import local_cpu_mesh
from test_utils import skip_if_no_torch, use_test_mesh


def _tiny_hf_config() -> HfQwen3MoeConfig:
    return HfQwen3MoeConfig(
        vocab_size=128,
        max_position_embeddings=16,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_theta=1_000_000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )


def test_qwen3_moe_config_roundtrip():
    hf_config = _tiny_hf_config()
    config = Qwen3MoeConfig.from_hf_config(hf_config)

    assert config.max_seq_len == hf_config.max_position_embeddings
    assert config.hidden_dim == hf_config.hidden_size
    assert config.moe_intermediate_dim == hf_config.moe_intermediate_size
    assert config.num_experts == hf_config.num_experts
    assert config.num_experts_per_tok == hf_config.num_experts_per_tok
    assert config.head_dim == hf_config.head_dim
    assert config.use_qk_norm

    roundtripped = config.to_hf_config(hf_config.vocab_size)
    assert roundtripped.model_type == "qwen3_moe"
    assert roundtripped.hidden_size == hf_config.hidden_size
    assert roundtripped.moe_intermediate_size == hf_config.moe_intermediate_size
    assert roundtripped.num_experts == hf_config.num_experts
    assert roundtripped.num_experts_per_tok == hf_config.num_experts_per_tok
    assert roundtripped.head_dim == hf_config.head_dim


def test_qwen3_moe_state_dict_keys_match_hf_qwen_layout():
    config = Qwen3MoeConfig.from_hf_config(_tiny_hf_config())
    model = Qwen3MoeLMHeadModel.init(hax.Axis("vocab", 128), config, key=random.PRNGKey(0))

    state_dict = to_torch_compatible_state_dict(model)

    assert state_dict["model.layers.0.self_attn.q_norm.weight"].shape == (8,)
    assert state_dict["model.layers.0.self_attn.k_norm.weight"].shape == (8,)
    assert state_dict["model.layers.0.self_attn.q_proj.weight"].shape == (32, 32)
    assert state_dict["model.layers.0.mlp.gate.weight"].shape == (4, 32)
    assert state_dict["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (8, 32)
    assert state_dict["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (8, 32)
    assert state_dict["model.layers.0.mlp.experts.0.down_proj.weight"].shape == (32, 8)
    assert state_dict["lm_head.weight"].shape == (128, 32)


@skip_if_no_torch
def test_qwen3_moe_loads_torch_compatible_state_dict():
    config = Qwen3MoeConfig.from_hf_config(_tiny_hf_config())
    with local_cpu_mesh():
        model = Qwen3MoeLMHeadModel.init(hax.Axis("vocab", 128), config, key=random.PRNGKey(0))
        state_dict = to_torch_compatible_state_dict(model)

        loaded = from_torch_compatible_state_dict(model, state_dict)

        loaded_state_dict = to_torch_compatible_state_dict(loaded)
    assert state_dict.keys() == loaded_state_dict.keys()
    for key, value in state_dict.items():
        np.testing.assert_allclose(np.asarray(loaded_state_dict[key]), np.asarray(value))


def test_qwen3_moe_forward_and_next_token_loss():
    config = Qwen3MoeConfig.from_hf_config(_tiny_hf_config())
    Batch = hax.Axis("batch", 2)
    Pos = config.max_Pos
    Vocab = hax.Axis("vocab", 128)
    input_ids = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)

    with use_test_mesh():
        model = Qwen3MoeLMHeadModel.init(Vocab, config, key=random.PRNGKey(0))

        @jax.jit
        def compute_loss(model, input_ids):
            loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32)
            example = LmExample(tokens=input_ids, loss_weight=loss_weight, attn_mask=AttentionMask.causal())
            return model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array

        losses = compute_loss(model, input_ids)

    assert losses.shape == (Batch.size, Pos.size)
    assert np.isfinite(np.asarray(losses)).all()


@skip_if_no_torch
def test_qwen3_moe_hf_logits_and_loss_match_torch(local_gpt2_tokenizer_path):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from transformers import Qwen3MoeForCausalLM  # noqa: PLC0415

    hf_config = _tiny_hf_config()
    config = Qwen3MoeConfig.from_hf_config(hf_config)
    # Local tokenizer + no remote reference keeps the conversion off the Hub; the
    # tokenizer is incidental (random inputs, logit-equivalence only).
    converter = dataclasses.replace(
        config, reference_checkpoint=None, tokenizer=local_gpt2_tokenizer_path
    ).hf_checkpoint_converter()
    Batch = hax.Axis("batch", 2)
    Pos = config.max_Pos
    Vocab = hax.Axis("vocab", hf_config.vocab_size)
    input_ids = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.long)

    torch.random.manual_seed(0)
    torch_model = Qwen3MoeForCausalLM(hf_config)
    torch_model.eval()

    with torch.no_grad():
        torch_logits = torch_model(input_torch).logits.detach().cpu().numpy()
        torch_loss = F.cross_entropy(
            torch.from_numpy(torch_logits[:, :-1]).reshape(-1, Vocab.size),
            input_torch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(Batch.size, Pos.size - 1)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        torch_model.save_pretrained(f"{tmpdir}/torch_model")
        model = converter.load_pretrained(
            Qwen3MoeLMHeadModel,
            ref=f"{tmpdir}/torch_model",
            resize_vocab_to_match_tokenizer=False,
        )

        @hax.named_jit
        def compute_logits(model, input_ids):
            return model(input_ids, attn_mask=AttentionMask.causal()).array

        @hax.named_jit
        def compute_loss(model, input_ids):
            loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32)
            example = LmExample(tokens=input_ids, loss_weight=loss_weight, attn_mask=AttentionMask.causal())
            return model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array

        jax_logits = compute_logits(model, input_ids)
        jax_loss = compute_loss(model, input_ids)

    np.testing.assert_allclose(np.asarray(jax_logits), torch_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(jax_loss[:, :-1]), torch_loss.numpy(), rtol=1e-4, atol=1e-4)
