# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile
from typing import Optional, cast

import equinox
import fsspec
import jax
import numpy as onp
import pytest
from fsspec import AbstractFileSystem
from jax.random import PRNGKey
from numpy.testing import assert_allclose
from transformers import AutoModelForCausalLM
from transformers import GPT2Config as HfGpt2Config
from transformers import GPT2LMHeadModel as HfGpt2LMHeadModel

import haliax as hax

from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.layers.attention import AttentionMask
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.lm_model import LmExample, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig
from levanter.utils.tree_utils import inference_mode
from test_utils import arrays_only, skip_if_no_torch
from tests.test_utils import use_test_mesh


@skip_if_no_torch
def test_hf_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("gpt2", None)


@skip_if_no_torch
def test_hf_gpt2_roundtrip_fa():
    hf_config = HfGpt2Config.from_pretrained("gpt2")
    config = Gpt2Config.from_hf_config(hf_config)
    config = dataclasses.replace(config, use_flash_attention=True, flash_attention_block_size=128)
    _roundtrip_compare_gpt2_checkpoint("gpt2", None, config=config)


# TODO: gotta figure out why this regressed
@pytest.mark.skip
@skip_if_no_torch
def test_mistral_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("stanford-crfm/expanse-gpt2-small-x777", "checkpoint-60000")


def _roundtrip_compare_gpt2_checkpoint(model_id, revision, config: Optional[Gpt2Config] = None):
    import torch

    if config is None:
        converter = Gpt2Config(use_flash_attention=False).hf_checkpoint_converter()
    else:
        converter = config.hf_checkpoint_converter()

    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    torch_model.eval()

    model: Gpt2LMHeadModel = cast(
        Gpt2LMHeadModel,
        converter.load_pretrained(Gpt2LMHeadModel, RepoRef(model_id, revision=revision), config),
    )
    model = inference_mode(model, True)

    lm_head = model.embeddings.token_embeddings
    jax_lm_head = onp.array(lm_head.weight.array)
    torch_lm_head = torch_model.transformer.wte.weight.detach().cpu().numpy()
    assert torch_lm_head.shape == jax_lm_head.shape
    assert_allclose(jax_lm_head, torch_lm_head, rtol=1e-4, atol=1e-4)

    input = hax.random.randint(PRNGKey(0), model.Pos, 0, model.Vocab.size)
    attn_mask = AttentionMask.causal()

    # we compare softmaxes because the numerics are wonky and we usually just care about the softmax
    torch_out = torch_model(torch.from_numpy(onp.array(input.array)).to(torch.int32).unsqueeze(0))
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    def compute(input):
        return hax.nn.softmax(model(input, key=None, attn_mask=attn_mask), axis=model.Vocab)

    compute = jax.jit(compute)
    jax_out = compute(input).array
    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    # get the argmaxes for the two models
    assert_allclose(torch_out, onp.array(jax_out), rtol=1e-2, atol=1e-2)

    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(model, tmpdir)

        torch_model2: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(tmpdir)
        torch_model2.eval()

        torch_out2 = torch_model2(torch.from_numpy(onp.array(input.array)).to(torch.int32).unsqueeze(0))
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)

        assert onp.isclose(torch_out2, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"


# Gradient tests


@skip_if_no_torch
def test_hf_gradient():
    _compare_gpt2_checkpoint_gradients("gpt2", None)


@skip_if_no_torch
def test_hf_gradient_fa():
    hf_config = HfGpt2Config.from_pretrained("gpt2")
    config = Gpt2Config.from_hf_config(hf_config)
    # keep block size small to make sure we test the tiling behavior
    config = dataclasses.replace(config, use_flash_attention=True, flash_attention_block_size=128)
    _compare_gpt2_checkpoint_gradients("gpt2", None, config=config)


def _compare_gpt2_checkpoint_gradients(model_id, revision, config: Optional[Gpt2Config] = None):
    import torch

    config = config or Gpt2Config()
    converter = config.hf_checkpoint_converter()
    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    torch_model.eval()

    model = cast(Gpt2LMHeadModel, converter.load_pretrained(config.model_type, RepoRef(model_id, revision), config))
    model = inference_mode(model, True)

    input = hax.random.randint(PRNGKey(0), model.Pos, 0, model.Vocab.size)

    def torch_loss(model, input_ids) -> torch.Tensor:
        return model(input_ids, labels=input_ids)[0]

    torch_out = torch_loss(torch_model, torch.from_numpy(onp.array(input.array)).to(torch.int64).unsqueeze(0))

    def compute_loss(model: LmHeadModel, input_ids):
        example = LmExample.causal(input_ids, eos_id=converter.tokenizer.eos_token_id)
        return compute_next_token_loss(model, example, key=None).scalar()

    jax_compute_grad = equinox.filter_value_and_grad(compute_loss, has_aux=False)
    jax_grad: Gpt2LMHeadModel
    jax_loss, jax_grad = jax_compute_grad(model, input)

    # gradients are kind of a pain to get at in torch, but we do it anyway
    torch_out.backward()
    state_dict = torch_model.transformer.state_dict(keep_vars=True)
    state_dict = {k: v.grad for k, v in state_dict.items()}

    jax_grad_dict = hax.state_dict.to_torch_compatible_state_dict(jax_grad)

    for jax_key, jax_g in jax_grad_dict.items():
        if jax_key not in state_dict:
            assert jax_key == "token_out_embeddings"
            continue

        torch_g = state_dict[jax_key]
        assert onp.isclose(jax_g, torch_g.detach().cpu().numpy(), rtol=1e-2, atol=1e-2).all(), f"{jax_g} != {torch_g}"

    # now we also want to check that the optimizers do similar things
    optimizer_config = AdamConfig(weight_decay=0.0, learning_rate=1e-3, warmup=0.0, lr_schedule="constant")

    if optimizer_config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), optimizer_config.max_grad_norm)
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.epsilon,
    )

    torch_optimizer.step()

    jax_optimizer = optimizer_config.build(1000)
    state = jax_optimizer.init(arrays_only(model))
    updates, state = jax_optimizer.update(updates=jax_grad, state=state, params=model)
    new_model = equinox.apply_updates(model, updates)

    new_model_dict = hax.state_dict.to_torch_compatible_state_dict(new_model)
    state_dict = torch_model.transformer.state_dict(keep_vars=True)

    # now compare new params
    for key, jax_p in new_model_dict.items():
        if key not in state_dict:
            assert key == "token_out_embeddings"
            continue
        torch_p = state_dict[key]
        assert onp.isclose(
            jax_p, torch_p.detach().cpu().numpy(), rtol=1e-3, atol=2e-3
        ).all(), f"{key}: {onp.linalg.norm(jax_p - torch_p.detach().cpu().numpy(), ord=onp.inf)}"


def test_hf_save_to_fs_spec():
    config = Gpt2Config(hidden_dim=32, num_heads=2, num_layers=2)
    converter = HFCheckpointConverter(Gpt2Config, "gpt2", HfGpt2Config, ignore_prefix="transformer")
    simple_model = Gpt2LMHeadModel.init(converter.Vocab, config, key=PRNGKey(0))

    with use_test_mesh():
        converter.save_pretrained(simple_model, "memory://model")

    with tempfile.TemporaryDirectory() as tmpdir:
        # now copy the model to tmp because loading from memory doesn't work
        fs: AbstractFileSystem = fsspec.filesystem("memory")
        fs.get("model/", f"{tmpdir}/test", recursive=True)

        loaded_model = converter.load_pretrained(Gpt2LMHeadModel, ref=f"{tmpdir}/test")

        simple_dict = hax.state_dict.to_torch_compatible_state_dict(simple_model)
        loaded_dict = hax.state_dict.to_torch_compatible_state_dict(loaded_model)

        assert simple_dict.keys() == loaded_dict.keys()

        for key, simple_p in simple_dict.items():
            loaded_p = loaded_dict[key]
            assert onp.allclose(simple_p, loaded_p), f"{key}: {onp.linalg.norm(simple_p - loaded_p, ord=onp.inf)}"


# TODO: would be nice to have a test that tests hf upload?
