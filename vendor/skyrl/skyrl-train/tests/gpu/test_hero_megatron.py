"""Hero architecture through the actual Megatron worker, with split-expert checkpoints."""

import math
import os
import posixpath
from pathlib import Path
from uuid import uuid4

from omegaconf import open_dict
from cloud.iris.hf_model_cache import stage_model_metadata
from marinskyrl.model_manifest import snapshot_model_manifest
import pytest
import ray
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.io import io
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.test_grug_megatron import (
    _config,
    _init_policy,
    _megatron_response_logprobs,
    _padded_batch,
    _write_tiny_checkpoint,
)


def _train_step(policy, batch):
    outputs = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))
    status = outputs[0].metadata["train_status"]
    assert math.isfinite(status["policy_loss"])
    assert status["policy_update_steps"] == 1
    assert status["raw_grad_norm"] > 0
    return status


def write_tiny_hero_checkpoint(path: Path):
    """Create all Hero parameter families, retaining an ordinary saved tokenizer."""
    shape = dict(
        hidden_size=128,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=16,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        sliding_window=8,
    )
    _write_tiny_checkpoint(path, shape=shape, num_experts_per_tok=8)
    config = GrugMoeConfig.from_pretrained(path)
    state = load_file(str(path / "model.safetensors"))
    torch.manual_seed(23)
    # The borrowed tokenizer has an odd vocabulary; Hero's real vocabulary is
    # divisible by the supported tensor-parallel sizes.
    vocab_size = (config.vocab_size + 127) // 128 * 128
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        state[name] = torch.nn.functional.pad(state[name], (0, 0, 0, vocab_size - config.vocab_size))
    config.vocab_size = vocab_size
    latent = 64
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for projection, dims in (("gate_proj", (64, latent)), ("up_proj", (64, latent)), ("down_proj", (latent, 64))):
            del state[f"{prefix}.mlp.experts.{projection}.weight"]
            for expert in range(config.num_local_experts):
                state[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = torch.randn(*dims) * 0.1
        for projection in ("gate_proj", "up_proj", "down_proj"):
            original = state.pop(f"{prefix}.shared_expert.{projection}.weight")
            for expert in range(2):
                state[f"{prefix}.shared_experts.{expert}.{projection}.weight"] = torch.randn_like(original) * 0.1
        state[f"{prefix}.mlp.latent_down_proj.weight"] = torch.randn(latent, config.hidden_size) * 0.1
        state[f"{prefix}.mlp.latent_up_proj.weight"] = torch.randn(config.hidden_size, latent) * 0.1
        state[f"{prefix}.mlp.latent_norm.weight"] = torch.ones(latent)
        for site, channels in (("self_attn.sconv_k", 64), ("sconv_attn", 128), ("sconv_mlp", 128)):
            weight = torch.randn(4, channels) * 0.1
            weight[0] += 1
            state[f"{prefix}.{site}.weight"] = weight
    config = GrugMoeConfig(
        **{
            **config.to_dict(),
            "latent_dim": latent,
            "num_shared_experts": 2,
            "local_kv_heads": 4,
            "global_kv_heads": 2,
            "rope_fused": True,
            "sconv": True,
            "grugmoe_artifact_schema_version": 2,
        }
    )
    config.save_pretrained(path)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    return state


@pytest.mark.parametrize(
    "tp,pp,ep,cp,packing,optimizer_offload,muonh,remote_read_mode",
    [
        pytest.param(1, 1, 1, 1, False, None, False, "default", id="remote-default-prefetch"),
        (1, 2, 1, 1, False, None, False, None),
        (1, 1, 2, 1, True, None, False, None),
        (1, 1, 1, 2, True, None, False, None),
        (2, 1, 1, 1, True, None, False, None),
        pytest.param(1, 1, 2, 1, True, 0.0, False, None, id="precision-aware-gpu"),
        pytest.param(1, 1, 2, 1, True, 0.5, False, None, id="half-offloaded-adamw"),
        pytest.param(1, 1, 2, 1, True, 1.0, False, None, id="cpu-adamw"),
        pytest.param(1, 2, 2, 1, False, None, True, None, id="pp2-ep2-muonh-cpu-momentum"),
        pytest.param(1, 1, 1, 1, False, None, False, "per_key", id="remote-per-key"),
    ],
)
def test_hero_worker_repeated_updates(
    tmp_path, monkeypatch, request, tp, pp, ep, cp, packing, optimizer_offload, muonh, remote_read_mode
):
    world_size = tp * pp * ep * cp
    require_hoppers(world_size)
    checkpoint_prefix = os.environ.get("MARIN_TEMP_PREFIX", os.environ.get("MARIN_PREFIX", ""))
    if not checkpoint_prefix.startswith("s3://"):
        raise ValueError("Run the Hero checkpoint test on Iris with CoreWeave object storage configured")
    checkpoint = posixpath.join(checkpoint_prefix, "tests", "hero-checkpoint", uuid4().hex)
    model_path = tmp_path / "model"
    model_path.mkdir()
    original = write_tiny_hero_checkpoint(model_path)
    source_uri = (
        posixpath.join(checkpoint_prefix, "tests", "hero-source", uuid4().hex) if remote_read_mode is not None else None
    )
    if source_uri is not None:
        request.addfinalizer(lambda: io.remove(source_uri) if io.exists(source_uri) else None)
    policy_path = model_path
    if source_uri is not None:
        manifest = snapshot_model_manifest(model_path, model_id="tiny-hero", revision="fixture")
        io.upload_directory(str(model_path), source_uri)
        policy_path = tmp_path / "metadata"
        stage_model_metadata(source_uri, manifest, str(policy_path))
    cfg = _config(str(policy_path), world_size=world_size, pp=pp, ep=ep)
    if source_uri is not None:
        cfg.trainer.policy.model.source_uri = source_uri
    if remote_read_mode == "per_key":
        cfg.trainer.policy.model.remote_read_mode = remote_read_mode
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    if muonh:
        cfg.trainer.policy.optimizer_config.optimizer = "MuonH"
        cfg.trainer.policy.optimizer_config.weight_decay = 0.0
        cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.95]
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {"adam_lr": 2.0e-2, "offload_momentum": True}
        cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = "dp_reshardable"
    if optimizer_offload is not None:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        cfg.trainer.flash_attn = True
        megatron = cfg.trainer.policy.megatron_config
        megatron.ddp_config.overlap_param_gather = True
        megatron.ddp_config.grad_reduce_in_fp32 = False
        megatron.optimizer_checkpoint_sharding_type = "dp_reshardable"
        with open_dict(megatron.transformer_config_kwargs), open_dict(megatron.optimizer_config_kwargs):
            megatron.transformer_config_kwargs.deterministic_mode = True
            megatron.optimizer_config_kwargs.use_precision_aware_optimizer = True
            megatron.optimizer_config_kwargs.store_param_remainders = False
            megatron.optimizer_config_kwargs.optimizer_cpu_offload = optimizer_offload > 0
            megatron.optimizer_config_kwargs.optimizer_offload_fraction = optimizer_offload
            megatron.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d = False
    cfg.trainer.use_sample_packing = packing
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id, prompt_length=48, response_length=48, variable_lengths=True)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, world_size)
        if muonh:
            snapshots = ray.get(policy.async_run_ray_method("pass_through", "grug_optimizer_route_snapshot"))
            routes = {name: route for snapshot in snapshots for name, route in snapshot["routes"].items()}
            for name_fragment, expected_route in (
                ("self_attention.linear_qkv.weight", "grug_muonh_qkv"),
                ("mlp.experts.linear_fc1.weight", "grug_muonh_gate_up"),
                ("output_layer.weight", "grug_adamh"),
                ("self_attention.sconv_k.weight", "adam"),
                ("mlp.router.weight", "adam"),
            ):
                assert any(name_fragment in name and route == expected_route for name, route in routes.items()), (
                    name_fragment,
                    expected_route,
                )
        names = list(original)
        before = rank0_validation_snapshot(policy, names)
        for name in names:
            expected = original[name] if name.endswith("router.bias") else original[name].to(torch.bfloat16).float()
            torch.testing.assert_close(before[name], expected, rtol=0, atol=0)
        selected_names = [
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            "model.layers.0.mlp.experts.15.down_proj.weight",
        ]
        selected = rank0_validation_snapshot(policy, selected_names)
        assert set(selected) == set(selected_names)
        for name in selected_names:
            torch.testing.assert_close(selected[name], before[name], rtol=0, atol=0)
        initial_scores = _megatron_response_logprobs(policy, batch)
        assert torch.isfinite(initial_scores).all()
        repeated_scores = _megatron_response_logprobs(policy, batch)
        torch.testing.assert_close(initial_scores, repeated_scores, rtol=0, atol=0)
        for _ in range(3):
            scores = _megatron_response_logprobs(policy, batch)
            batch["action_log_probs"] = (scores * batch["response_mask"]).float()
            status = _train_step(policy, batch)
            assert status["log_ratio_abs_max"] < 1e-3, status
        after = rank0_validation_snapshot(policy, names)
        for name in names:
            if name.endswith("router.bias"):
                torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
            assert torch.isfinite(after[name]).all(), name
        for name in (
            "model.layers.0.mlp.latent_down_proj.weight",
            "model.layers.0.sconv_mlp.weight",
            "model.layers.0.self_attn.sconv_k.weight",
            "model.layers.0.shared_experts.1.up_proj.weight",
        ):
            assert not torch.equal(before[name], after[name]), name
        final_scores = _megatron_response_logprobs(policy, batch)
        assert torch.isfinite(final_scores).all()
        assert not torch.equal(initial_scores, final_scores)
        batch["action_log_probs"] = (final_scores * batch["response_mask"]).float()
        ray.get(
            policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint, tokenizer=tokenizer)
        )
        # The second update catches a restored step counter reset on every step.
        continued = []
        for _ in range(2):
            _train_step(policy, batch)
            continued.append(rank0_validation_snapshot(policy, names))
        ray.get(policy.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint))
        restored = rank0_validation_snapshot(policy, names)
        for name in names:
            torch.testing.assert_close(restored[name], after[name], rtol=0, atol=0)
        restored_scores = _megatron_response_logprobs(policy, batch)
        torch.testing.assert_close(restored_scores, final_scores, rtol=0, atol=0)
        for expected in continued:
            _train_step(policy, batch)
            resumed = rank0_validation_snapshot(policy, names)
            for name in names:
                torch.testing.assert_close(resumed[name], expected[name], rtol=0, atol=0)
    finally:
        ray.shutdown()
        if io.exists(checkpoint):
            io.remove(checkpoint)
