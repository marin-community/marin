# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import click
import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import SNOWBALL_POLICY

MEGATRON_GEOMETRY = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 2,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 8,
    "expert_tensor_parallel_size": 1,
}

# Every setting the launcher decides, by dotted key, with the value every preset writes. A preset
# that stops writing one of these, or leaves it to the base config, fails the drift guard.
HOUSE_SETTINGS = {
    "entrypoint": "fully_async",
    "context_budget.max_turns": 1,
    "trainer.strategy": "megatron",
    "trainer.flash_attn": False,
    "trainer.use_sample_packing": False,
    "trainer.gradient_checkpointing": True,
    "trainer.offload_optimizer_during_rollouts": False,
    "trainer.epochs": 50,
    "trainer.update_epochs_per_batch": 1,
    "trainer.train_batch_size": 128,
    "trainer.policy_mini_batch_size": 128,
    "trainer.micro_train_batch_size_per_gpu": 1,
    "trainer.micro_forward_batch_size_per_gpu": 1,
    "trainer.eval_batch_size": 256,
    "trainer.hf_save_interval": -1,
    "trainer.resume_mode": "latest",
    "trainer.seed": 17,
    "trainer.logger": "wandb",
    "trainer.project_name": "marin-async-rl",
    "trainer.tracker_commit_each_step": True,
    "trainer.training_metrics": True,
    "trainer.async_spans": True,
    "trainer.policy_train_spans": True,
    "trainer.generate_spans": False,
    "trainer.optimizer_state_metrics": True,
    "trainer.algorithm.advantage_estimator": "grpo",
    "trainer.algorithm.policy_loss_type": "regular",
    "trainer.algorithm.use_kl_loss": False,
    "trainer.algorithm.use_kl_in_reward": False,
    "trainer.algorithm.use_tis": False,
    "trainer.algorithm.eps_clip_low": 0.2,
    "trainer.algorithm.eps_clip_high": 0.2,
    "trainer.algorithm.ratio_diagnostics.pooled": True,
    "trainer.algorithm.ratio_diagnostics.exact_quantiles": False,
    "trainer.algorithm.grad_cosine.enabled": False,
    "trainer.policy.optimizer_config.optimizer": "AdamW",
    "trainer.policy.optimizer_config.lr": 1.0e-6,
    "trainer.policy.optimizer_config.weight_decay": 1e-2,
    "trainer.policy.optimizer_config.max_grad_norm": 1.0,
    **{f"trainer.policy.megatron_config.{key}": value for key, value in MEGATRON_GEOMETRY.items()},
    **{f"trainer.ref.megatron_config.{key}": value for key, value in MEGATRON_GEOMETRY.items()},
    "trainer.placement.colocate_all": False,
    "trainer.placement.colocate_policy_ref": True,
    "trainer.placement.policy_num_nodes": 4,
    "trainer.placement.policy_num_gpus_per_node": 8,
    "trainer.placement.ref_num_nodes": 4,
    "trainer.placement.ref_num_gpus_per_node": 8,
    "trainer.fully_async.max_buffered_groups": 32,
    "trainer.fully_async.pause_mode": "abort",
    "trainer.fully_async.clear_kv_cache_on_weight_sync": True,
    "trainer.fully_async.first_token_admission": True,
    "generator.backend": "vllm",
    "generator.model_dtype": "bfloat16",
    "generator.vllm_attention_backend": "FLASH_ATTN",
    "generator.run_engines_locally": True,
    "generator.weight_sync_backend": "nccl",
    "generator.async_engine": True,
    "generator.batched": False,
    "generator.num_inference_engines": 1,
    "generator.inference_engine_tensor_parallel_size": 1,
    "generator.inference_engine_pipeline_parallel_size": 1,
    "generator.inference_engine_data_parallel_size": 8,
    "generator.inference_engine_expert_parallel_size": 8,
    "generator.n_samples_per_prompt": 4,
    "generator.gpu_memory_utilization": 0.75,
    "generator.max_num_seqs": 1024,
    "generator.max_num_batched_tokens": 8192,
    "generator.enable_prefix_caching": True,
    "generator.enable_chunked_prefill": True,
    "generator.enforce_eager": False,
    "generator.enable_http_endpoint": True,
    "generator.use_conversation_multi_turn": True,
    "generator.chat_template.source": "name",
    "generator.chat_template.name_or_path": "marin_tokenizer",
    "generator.engine_init_kwargs.moe_backend": "triton",
    "generator.sampling_params.temperature": 1.0,
    "generator.sampling_params.top_p": 1.0,
    "generator.sampling_params.logprobs": 0,
    "extra_env.PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# The settings that differ between presets, with each preset's value.
PRESET_SETTINGS = {
    "default": {
        "trainer.max_steps": 100,
        "trainer.eval_interval": 5,
        "trainer.eval_before_train": True,
        "trainer.ckpt_interval": 5,
        "context_budget.request_window_tokens": 8192,
        "context_budget.max_new_tokens_per_turn": 4096,
        "trainer.fully_async.max_staleness_steps": 4,
        "trainer.fully_async.num_parallel_generation_workers": 192,
    },
    "smoke": {
        "trainer.max_steps": 2,
        "trainer.eval_interval": -1,
        "trainer.eval_before_train": False,
        "trainer.ckpt_interval": 2,
        "context_budget.request_window_tokens": 2048,
        "context_budget.max_new_tokens_per_turn": 1024,
        "trainer.fully_async.max_staleness_steps": 4,
        "trainer.fully_async.num_parallel_generation_workers": 192,
    },
    "on_policy": {
        "trainer.max_steps": 100,
        "trainer.eval_interval": 5,
        "trainer.eval_before_train": True,
        "trainer.ckpt_interval": 5,
        "context_budget.request_window_tokens": 8192,
        "context_budget.max_new_tokens_per_turn": 4096,
        "trainer.fully_async.max_staleness_steps": 0,
        "trainer.fully_async.num_parallel_generation_workers": 128,
    },
}

# MarinSkyRL derives these from context_budget and refuses a config that declares them.
DERIVED_CONTEXT_KEYS = (
    "trainer.max_prompt_length",
    "generator.max_input_length",
    "generator.max_turns",
    "generator.sampling_params.max_generate_length",
    "generator.engine_init_kwargs.max_model_len",
)


def rendered(preset=async_rl.DEFAULT, policy=SNOWBALL_POLICY, settings=()):
    return yaml.safe_load(async_rl.training_config(policy, preset, async_rl.RECIPES[policy.label], settings))


def value_at(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        node = node[part]
    return node


def declared(config: dict, dotted: str) -> bool:
    try:
        value_at(config, dotted)
    except KeyError:
        return False
    return True


@pytest.mark.parametrize("label", sorted(PRESET_SETTINGS))
def test_every_preset_writes_every_house_setting_explicitly(label):
    config = rendered(async_rl.PRESETS[label])
    expected = {**HOUSE_SETTINGS, **PRESET_SETTINGS[label]}
    missing = [key for key in expected if not declared(config, key)]
    assert missing == []
    wrong = {key: value_at(config, key) for key, value in expected.items() if value_at(config, key) != value}
    assert wrong == {}
    assert [key for key in DERIVED_CONTEXT_KEYS if declared(config, key)] == []


def test_presets_keep_generation_inside_the_staleness_allowance():
    with pytest.raises(ValueError, match="cannot fill an update"):
        replace(async_rl.DEFAULT, generation_workers=64)
    # 480 workers plus the 32-group buffer is exactly four updates of 128: the bound is strict.
    with pytest.raises(ValueError, match="would age out"):
        replace(async_rl.DEFAULT, generation_workers=480)
    with pytest.raises(ValueError, match="do not fit the request window"):
        replace(async_rl.DEFAULT, request_window_tokens=4096, max_new_tokens=4096)
    # At staleness 0 nothing is ever admitted stale, so only the worker floor applies.
    assert replace(async_rl.ON_POLICY, generation_workers=480).generation_workers == 480


def test_settings_change_existing_keys_and_reject_unknown_ones():
    config = rendered(settings=("trainer.fully_async.max_staleness_steps=1", "generator.gpu_memory_utilization=0.8"))
    assert config["trainer"]["fully_async"]["max_staleness_steps"] == 1
    assert config["generator"]["gpu_memory_utilization"] == 0.8
    with pytest.raises(click.BadParameter, match="unknown setting"):
        rendered(settings=("trainer.fully_async.max_stalness_steps=1",))
    with pytest.raises(click.BadParameter, match="entrypoint cannot change"):
        rendered(settings=("entrypoint=standard",))
    with pytest.raises(click.BadParameter, match=r"dotted\.key=value"):
        rendered(settings=("trainer.max_steps",))
    with pytest.raises(click.BadParameter, match="holds a value, not a section"):
        rendered(settings=("+trainer.max_steps.more=1",))
    added = rendered(settings=("+trainer.fully_async.weight_sync_interval=1",))
    assert added["trainer"]["fully_async"]["weight_sync_interval"] == 1


def test_presets_score_the_right_suites_on_five_nodes():
    assert async_rl.DEFAULT.scale(SNOWBALL_POLICY).evals == "math500,gsm8k-0shot"
    assert async_rl.SMOKE_PRESET.scale(SNOWBALL_POLICY).evals == "gsm8k-smoke"
    assert async_rl.DEFAULT.scale(SNOWBALL_POLICY).num_nodes == 5


def test_run_wires_the_curriculum_pool_and_snowball_export_into_one_chain(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    run = async_rl.build_run(policy=SNOWBALL_POLICY, preset=async_rl.SMOKE_PRESET, version="2026.09.18")
    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    changed = async_rl.build_run(
        policy=SNOWBALL_POLICY,
        preset=async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("trainer.fully_async.max_staleness_steps=1",),
    )
    assert changed.rl.name.startswith("users/alice/checkpoints/async-rl/snowball-smoke-set-")
    assert changed.rl.name != run.rl.name


def test_command_plans_without_running(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.18", "--preset", "smoke"])
    assert result.exit_code == 0, result.output
    assert "snowball-smoke" in result.output


def test_set_refuses_keys_the_topology_overwrites():
    config = yaml.safe_load(
        async_rl.training_config(SNOWBALL_POLICY, async_rl.PRESETS["default"], async_rl.RECIPES["snowball"])
    )
    with pytest.raises(click.BadParameter, match="written from the topology"):
        async_rl.apply_setting(config, "trainer.train_batch_size", 64, adds=False)
    with pytest.raises(click.BadParameter, match="written from the topology"):
        async_rl.apply_setting(config, "generator.num_inference_engines", 2, adds=False)
