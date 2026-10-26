import os
import posixpath
from pathlib import Path
from uuid import uuid4

import pytest
import ray
import torch
from safetensors.torch import load_file, save_file
from skyrl_train.io import io
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from transformers import AutoTokenizer

from cloud.iris.hf_model_cache import stage_model_metadata
from marinskyrl.model_manifest import snapshot_model_manifest
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.test_grug_megatron import _config, _init_policy, _megatron_response_logprobs, _padded_batch


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
    write_tiny_checkpoint(path, shape=shape, num_experts_per_tok=8)
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


@pytest.mark.parametrize("remote_read_mode", ["default", "per_key"], ids=["remote-default-prefetch", "remote-per-key"])
def test_hero_remote_load_preserves_checkpoint_weights_and_trains(tmp_path, request, remote_read_mode):
    require_hoppers(1)
    checkpoint_prefix = os.environ.get("MARIN_TEMP_PREFIX", os.environ.get("MARIN_PREFIX", ""))
    if not checkpoint_prefix.startswith("s3://"):
        raise ValueError("Run the Hero checkpoint test on Iris with CoreWeave object storage configured")
    model_path = tmp_path / "model"
    model_path.mkdir()
    original = write_tiny_hero_checkpoint(model_path)
    source_uri = posixpath.join(checkpoint_prefix, "tests", "hero-source", uuid4().hex)
    request.addfinalizer(lambda: io.remove(source_uri) if io.exists(source_uri) else None)
    manifest = snapshot_model_manifest(model_path, model_id="tiny-hero", revision="fixture")
    io.upload_directory(str(model_path), source_uri)
    policy_path = tmp_path / "metadata"
    stage_model_metadata(source_uri, manifest, str(policy_path))
    cfg = _config(str(policy_path), world_size=1, pp=1, ep=1)
    cfg.trainer.policy.model.source_uri = source_uri
    if remote_read_mode == "per_key":
        cfg.trainer.policy.model.remote_read_mode = remote_read_mode
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    batch = _padded_batch(tokenizer.pad_token_id, prompt_length=48, response_length=48, variable_lengths=True)
    validate_cfg(cfg)
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, 1)
        before = rank0_validation_snapshot(policy, list(original))
        for name in original:
            expected = original[name] if name.endswith("router.bias") else original[name].to(torch.bfloat16).float()
            torch.testing.assert_close(before[name], expected, rtol=0, atol=0)
        scores = _megatron_response_logprobs(policy, batch)
        assert torch.isfinite(scores).all()
        batch["action_log_probs"] = (scores * batch["response_mask"]).float()
        output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
        status = output.metadata["train_status"]
        assert torch.isfinite(torch.tensor(status["policy_loss"]))
        assert status["policy_update_steps"] == 1 and status["raw_grad_norm"] > 0
    finally:
        ray.shutdown()
