# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from transformers import Qwen3MoeConfig as HfQwen3MoeConfig

import haliax as hax
from haliax.state_dict import from_torch_compatible_state_dict, to_torch_compatible_state_dict

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeLMHeadModel, Qwen3MoeSparseMoeBlock
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.tree_utils import inference_mode
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


def _tiny_moe_config(dense_router_gradient: bool) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        moe_intermediate_dim=8,
        num_layers=2,
        num_heads=4,
        head_dim=4,
        num_kv_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        gradient_checkpointing=False,
        dense_router_gradient=dense_router_gradient,
    )


def _gate_grad_row_norms(block_grad: Qwen3MoeSparseMoeBlock, config: Qwen3MoeConfig) -> np.ndarray:
    """L2 norm of the router-gate weight gradient per expert (shape [num_experts])."""
    per_expert = hax.sqrt(hax.sum(block_grad.gate.weight**2, axis=config.Embed))
    return np.asarray(per_expert.rearrange((config.Experts,)).array)


def test_qwen3_moe_dense_router_gradient_leaves_forward_unchanged():
    # The DenseMixer flag is a training-only backward-pass change: the forward value must be
    # bit-identical to the stock sparse forward.
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 64)
    config_off = _tiny_moe_config(dense_router_gradient=False)
    Pos = config_off.max_Pos
    input_ids = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)

    with use_test_mesh():
        # Same key => identical weights; the flag does not touch initialization.
        model_off = Qwen3MoeLMHeadModel.init(Vocab, config_off, key=random.PRNGKey(0))
        model_on = Qwen3MoeLMHeadModel.init(Vocab, _tiny_moe_config(dense_router_gradient=True), key=random.PRNGKey(0))

        @eqx.filter_jit
        def logits(model, ids):
            return model(ids, attn_mask=AttentionMask.causal()).array

        logits_off = logits(model_off, input_ids)
        logits_on = logits(model_on, input_ids)

    np.testing.assert_array_equal(np.asarray(logits_on), np.asarray(logits_off))


def _single_token_block_grad(block: Qwen3MoeSparseMoeBlock, config: Qwen3MoeConfig):
    """Task-loss gradient of a one-token MoE-block forward.

    With a single token, the router-gate weight gradient factors as ``x (x) dL/d(logit_e)``, so a
    zero gate-gradient row for expert ``e`` means expert ``e``'s router logit received no task-loss
    gradient. This isolates the router gradient per expert without reimplementing the routing.
    """
    x = hax.random.normal(random.PRNGKey(2), (hax.Axis(config.max_Pos.name, 1), config.Embed))

    def task_loss(block, x):
        return hax.sum(block(x)).scalar()

    return eqx.filter_jit(eqx.filter_grad(task_loss))(block, x)


def test_qwen3_moe_dense_router_gradient_reaches_unselected_experts():
    config_off = _tiny_moe_config(dense_router_gradient=False)
    config_on = _tiny_moe_config(dense_router_gradient=True)
    n_unselected = config_off.num_experts - config_off.num_experts_per_tok

    with use_test_mesh():
        block_off = Qwen3MoeSparseMoeBlock.init(config_off, key=random.PRNGKey(0))
        block_on = dataclasses.replace(block_off, config=config_on)

        grad_off = _single_token_block_grad(block_off, config_off)
        grad_on = _single_token_block_grad(block_on, config_on)

    rows_off = _gate_grad_row_norms(grad_off, config_off)
    rows_on = _gate_grad_row_norms(grad_on, config_on)

    # Sparse forward: exactly `n_unselected` experts get no task-loss router gradient.
    assert int(np.sum(rows_off < 1e-6)) == n_unselected
    # Dense router gradient: every expert (including the unselected ones) gets a real gradient.
    assert np.all(rows_on > 1e-4)

    # Expert-parameter gradients must be untouched by the flag: only the router gradient changes.
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight_off = getattr(grad_off.experts, projection).weight.array
        weight_on = getattr(grad_on.experts, projection).weight.array
        np.testing.assert_allclose(np.asarray(weight_on), np.asarray(weight_off), rtol=1e-6, atol=1e-6)


def test_qwen3_moe_dense_router_gradient_disabled_in_inference():
    config_off = _tiny_moe_config(dense_router_gradient=False)
    config_on = _tiny_moe_config(dense_router_gradient=True)

    with use_test_mesh():
        block_off = Qwen3MoeSparseMoeBlock.init(config_off, key=random.PRNGKey(0))
        block_on = dataclasses.replace(block_off, config=config_on)
        block_on_eval = inference_mode(block_on, True)

        grad_off = _single_token_block_grad(block_off, config_off)
        grad_eval = _single_token_block_grad(block_on_eval, config_on)

    # inference_mode skips the dense pass, so the router gradient collapses back to the sparse one.
    np.testing.assert_allclose(
        _gate_grad_row_norms(grad_eval, config_on),
        _gate_grad_row_norms(grad_off, config_off),
        rtol=1e-5,
        atol=1e-6,
    )


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
