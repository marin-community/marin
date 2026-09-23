# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, fields, replace

import click
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import BASE_OVERRIDES, QWEN_POLICY, SNOWBALL_POLICY, rl_config_yaml

# Every key the launcher decides beyond the ones its own tables and dataclasses name.
EXPLICIT_KEYS = (
    "entrypoint",
    "context_budget.request_window_tokens",
    "context_budget.max_new_tokens_per_turn",
    "context_budget.max_turns",
    "trainer.strategy",
    "trainer.flash_attn",
    "trainer.use_sample_packing",
    "trainer.gradient_checkpointing",
    "trainer.offload_optimizer_during_rollouts",
    "trainer.epochs",
    "trainer.max_steps",
    "trainer.update_epochs_per_batch",
    "trainer.micro_forward_batch_size_per_gpu",
    "trainer.eval_batch_size",
    "trainer.dump_eval_results",
    "trainer.mismatch_decomposition.enabled",
    "trainer.mismatch_decomposition.sample_rows_per_step",
    "trainer.eval_before_train",
    "trainer.eval_interval",
    "trainer.seed",
    "trainer.ckpt_interval",
    "trainer.hf_save_interval",
    "trainer.resume_mode",
    "trainer.logger",
    "trainer.project_name",
    "trainer.tracker_commit_each_step",
    "trainer.training_metrics",
    "trainer.async_spans",
    "trainer.policy_train_spans",
    "trainer.generate_spans",
    "trainer.optimizer_state_metrics",
    "trainer.algorithm.advantage_estimator",
    "trainer.algorithm.policy_loss_type",
    "trainer.algorithm.use_kl_loss",
    "trainer.algorithm.use_kl_in_reward",
    "trainer.algorithm.use_tis",
    "trainer.algorithm.score_centering_topk",
    "trainer.algorithm.tis_imp_ratio_cap",
    "trainer.algorithm.eps_clip_low",
    "trainer.algorithm.eps_clip_high",
    "trainer.algorithm.ratio_diagnostics.pooled",
    "trainer.algorithm.ratio_diagnostics.exact_quantiles",
    "trainer.algorithm.grad_cosine.enabled",
    "trainer.policy.optimizer_config.optimizer",
    "trainer.policy.optimizer_config.lr",
    "trainer.policy.optimizer_config.weight_decay",
    "trainer.policy.optimizer_config.max_grad_norm",
    "trainer.policy.megatron_config.optimizer_checkpoint_sharding_type",
    "trainer.fully_async.max_staleness_steps",
    "trainer.fully_async.weight_sync_interval_steps",
    "trainer.fully_async.num_parallel_generation_workers",
    "trainer.fully_async.max_buffered_groups",
    "trainer.fully_async.pause_mode",
    "trainer.fully_async.clear_kv_cache_on_weight_sync",
    "trainer.fully_async.first_token_admission",
    "generator.backend",
    "generator.model_dtype",
    "generator.vllm_attention_backend",
    "generator.weight_sync_backend",
    "generator.async_engine",
    "generator.batched",
    "generator.inference_engine_pipeline_parallel_size",
    "generator.gpu_memory_utilization",
    "generator.max_num_seqs",
    "generator.max_num_batched_tokens",
    "generator.enable_prefix_caching",
    "generator.enable_chunked_prefill",
    "generator.enforce_eager",
    "generator.enable_http_endpoint",
    "generator.use_conversation_multi_turn",
    "generator.engine_init_kwargs.moe_backend",
    "generator.sampling_params.temperature",
    "generator.sampling_params.top_p",
    "generator.sampling_params.logprobs",
    "generator.eval_n_samples_per_prompt",
    "generator.eval_sampling_params.max_generate_length",
    "generator.eval_sampling_params.repetition_penalty",
    "generator.eval_sampling_params.temperature",
    "generator.eval_sampling_params.top_p",
    "generator.eval_sampling_params.min_p",
    "generator.eval_sampling_params.top_k",
    "generator.eval_sampling_params.logprobs",
    "generator.eval_sampling_params.stop",
    "extra_env.PYTORCH_CUDA_ALLOC_CONF",
)

# Values that are contracts with MarinSkyRL rather than tuning.
CONTRACT_VALUES = {
    "entrypoint": "fully_async",
    "trainer.strategy": "megatron",
    "trainer.algorithm.advantage_estimator": "grpo",
    "trainer.algorithm.policy_loss_type": "regular",
    "trainer.training_metrics": True,
    "trainer.async_spans": True,
    "trainer.policy_train_spans": True,
    "trainer.generate_spans": False,
    "trainer.optimizer_state_metrics": True,
    "trainer.dump_eval_results": True,
    "trainer.resume_mode": "latest",
    "trainer.mismatch_decomposition.enabled": False,
    "generator.eval_n_samples_per_prompt": 1,
    "generator.eval_sampling_params.temperature": 0.0,
    "trainer.algorithm.ratio_diagnostics.pooled": True,
    "trainer.algorithm.ratio_diagnostics.exact_quantiles": False,
    "trainer.algorithm.grad_cosine.enabled": False,
    "trainer.fully_async.max_buffered_groups": 32,
    "trainer.fully_async.weight_sync_interval_steps": 1,
    "trainer.fully_async.pause_mode": "abort",
    "trainer.fully_async.clear_kv_cache_on_weight_sync": True,
    "trainer.fully_async.first_token_admission": True,
    "trainer.policy.megatron_config.optimizer_checkpoint_sharding_type": "dp_reshardable",
}

PRESET_LOOPS = {
    "default": {
        "trainer.fully_async.max_staleness_steps": 4,
        "trainer.fully_async.num_parallel_generation_workers": 192,
    },
    "smoke": {
        "trainer.fully_async.max_staleness_steps": 4,
        "trainer.fully_async.num_parallel_generation_workers": 192,
    },
    "on_policy": {
        "trainer.fully_async.max_staleness_steps": 0,
        "trainer.fully_async.num_parallel_generation_workers": 128,
    },
}


def ruled_keys() -> set[str]:
    keys = set(EXPLICIT_KEYS) | set(async_rl.TOPOLOGY_OWNED_SETTINGS) | set(async_rl.RECIPE_OWNED_SETTINGS)
    for section in ("trainer.policy.megatron_config", "trainer.ref.megatron_config"):
        keys |= {f"{section}.{field.name}" for field in fields(async_rl.MegatronGeometry)}
    keys |= {f"generator.chat_template.{field.name}" for field in fields(async_rl.ChatTemplate)}
    return keys


def flattened(node: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            out.update(flattened(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def rendered(preset=async_rl.DEFAULT, settings=()) -> dict:
    return async_rl.training_config(preset, settings)


@pytest.fixture
def owner(monkeypatch):
    """Fix the artifact owner so run and evaluation addresses are deterministic."""
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")


@pytest.mark.parametrize("label", sorted(PRESET_LOOPS))
def test_every_preset_writes_every_ruled_key_and_inherits_none(label):
    config = flattened(rendered(async_rl.PRESETS[label]))
    expected = ruled_keys()
    missing = sorted(expected - config.keys())
    assert missing == []
    inherited = sorted(key for key in config if key not in expected and not key.startswith(("data.", "environment.")))
    assert inherited == []
    assert sorted(async_rl.DERIVED_CONTEXT_SETTINGS & config.keys()) == []
    literal = {**CONTRACT_VALUES, **PRESET_LOOPS[label]}
    wrong = {key: config[key] for key, value in literal.items() if config[key] != value}
    assert wrong == {}


def test_settings_change_existing_keys_and_reject_unknown_ones():
    config = rendered(settings=("trainer.fully_async.max_staleness_steps=2", "generator.gpu_memory_utilization=0.8"))
    assert config["trainer"]["fully_async"]["max_staleness_steps"] == 2
    assert config["generator"]["gpu_memory_utilization"] == 0.8
    with pytest.raises(click.BadParameter, match="unknown setting"):
        rendered(settings=("trainer.fully_async.max_stalness_steps=1",))
    with pytest.raises(click.BadParameter, match="entrypoint cannot change"):
        rendered(settings=("entrypoint=standard",))
    with pytest.raises(click.BadParameter, match=r"dotted\.key=value"):
        rendered(settings=("trainer.max_steps",))
    with pytest.raises(click.BadParameter, match="holds a value, not a section"):
        rendered(settings=("+trainer.max_steps.more=1",))
    with pytest.raises(click.BadParameter, match="derives"):
        rendered(settings=("+generator.max_input_length=1024",))
    added = rendered(settings=("+trainer.fully_async.weight_sync_interval=1",))
    assert added["trainer"]["fully_async"]["weight_sync_interval"] == 1


def test_explicit_resume_path_becomes_final_request_overrides(owner):
    checkpoint = "s3://bucket/tmp/ttl=14d/run/checkpoints/global_step_10"
    settings = (
        "trainer.resume_mode=from_path",
        f"+trainer.resume_path={checkpoint}",
    )
    config = rendered(settings=settings)
    assert config["trainer"]["resume_mode"] == "from_path"
    assert config["trainer"]["resume_path"] == checkpoint

    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.21", settings=settings)
    request = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args.keys(), run.rl.deps)).request
    resume_overrides = (
        "++trainer.resume_mode=from_path",
        f'++trainer.resume_path="{checkpoint}"',
    )
    assert all(override in request.overrides for override in resume_overrides)
    assert request.overrides.index(resume_overrides[0]) < request.overrides.index(resume_overrides[1])

    with pytest.raises(click.BadParameter, match="resume_path is required"):
        rendered(settings=("trainer.resume_mode=from_path",))
    with pytest.raises(click.BadParameter, match=r"unknown trainer\.resume_mode"):
        rendered(settings=("trainer.resume_mode=somewhere",))


def test_settings_preserve_core_worker_floor_and_check_prompt_window():
    with pytest.raises(click.BadParameter, match="cannot fill a policy mini-batch"):
        rendered(settings=("trainer.fully_async.num_parallel_generation_workers=8",))
    with pytest.raises(click.BadParameter, match="do not fit the request window"):
        rendered(settings=("context_budget.request_window_tokens=1024",))
    # Submission capacity and stale-group admission are managed by MarinSkyRL at runtime.
    wide_pool = rendered(settings=("trainer.fully_async.num_parallel_generation_workers=480",))
    assert wide_pool["trainer"]["fully_async"]["num_parallel_generation_workers"] == 480
    on_policy = rendered(async_rl.ON_POLICY, settings=("trainer.fully_async.num_parallel_generation_workers=480",))
    assert on_policy["trainer"]["fully_async"]["num_parallel_generation_workers"] == 480


def test_eval_interval_setting_moves_the_checkpoint_cadence():
    config = rendered(settings=("trainer.eval_interval=10",))
    assert config["trainer"]["ckpt_interval"] == 10
    assert config["trainer"]["eval_before_train"] is True
    off = rendered(settings=("trainer.eval_interval=-1",))
    assert off["trainer"]["ckpt_interval"] == off["trainer"]["max_steps"]
    assert off["trainer"]["eval_before_train"] is False
    with pytest.raises(click.BadParameter, match=r"follows trainer\.eval_interval"):
        rendered(settings=("trainer.ckpt_interval=3",))


def test_set_refuses_keys_the_topology_overwrites():
    with pytest.raises(click.BadParameter, match="written from the topology"):
        rendered(settings=("trainer.train_batch_size=64",))
    with pytest.raises(click.BadParameter, match="written from the topology"):
        rendered(settings=("generator.num_inference_engines=2",))
    with pytest.raises(click.BadParameter, match="written from the topology"):
        rendered(settings=("generator.inference_engine_pipeline_parallel_size=2",))
    with pytest.raises(click.BadParameter, match="engine geometry the recipe decides"):
        rendered(settings=("generator.inference_engine_data_parallel_size=4",))


def test_curriculum_template_supplies_only_the_data_and_environment_sections(monkeypatch):
    """No field of the scale point the template renders with reaches the rendered config."""
    template = yaml.safe_load(rl_config_yaml(async_rl.CURRICULUM_TEMPLATE))
    config = rendered()
    assert {section for section in template if config.get(section) == template[section]} == {"data", "environment"}
    plan = replace(
        async_rl.CURRICULUM_TEMPLATE.role_plan,
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=2,
        num_inference_engines=3,
        inference_engine_tensor_parallel_size=4,
        train_batch_size=5,
        policy_mini_batch_size=5,
        micro_train_batch_size_per_gpu=6,
        n_samples_per_prompt=7,
    )
    monkeypatch.setattr(
        async_rl,
        "CURRICULUM_TEMPLATE",
        replace(
            async_rl.CURRICULUM_TEMPLATE,
            label="other",
            num_nodes=9,
            role_plan=plan,
            max_steps=11,
            eval_interval=13,
            ckpt_interval=17,
            request_window_tokens=19,
            max_new_tokens=23,
            micro_forward_batch_size_per_gpu=29,
            evals="none",
        ),
    )
    assert rendered() == config


def test_recipe_engine_parallelism_reaches_the_run_unopposed(owner, monkeypatch):
    """The launcher's engine geometry must beat the curriculum policy's inherited override."""
    inherited = {override.lstrip("+").partition("=")[0] for override in SNOWBALL_POLICY.overrides}
    assert "generator.inference_engine_data_parallel_size" in inherited, SNOWBALL_POLICY.overrides
    monkeypatch.setattr(
        async_rl,
        "SNOWBALL_RECIPE",
        replace(
            async_rl.SNOWBALL_RECIPE,
            role_plan=replace(
                async_rl.SNOWBALL_RECIPE.role_plan,
                inference_engine_tensor_parallel_size=2,
                inference_engine_data_parallel_size=4,
                inference_engine_expert_parallel_size=4,
            ),
        ),
    )
    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")
    request = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args.keys(), run.rl.deps)).request
    generator = yaml.safe_load(request.config_yaml)["generator"]
    assert generator["inference_engine_data_parallel_size"] == 4
    assert generator["inference_engine_expert_parallel_size"] == 4
    assert asdict(request.topology.role_plan)["inference_engine_data_parallel_size"] == 4
    assert asdict(request.topology.role_plan)["inference_engine_expert_parallel_size"] == 4
    keys = {override.lstrip("+").partition("=")[0] for override in request.overrides}
    assert keys.isdisjoint(async_rl.RECIPE_OWNED_SETTINGS), request.overrides
    # An override on a key the rendered config says nothing about still reaches the run.
    assert {override.lstrip("+").partition("=")[0] for override in BASE_OVERRIDES} <= keys


def test_run_wires_the_curriculum_pool_and_snowball_export_into_one_chain(owner):
    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18")
    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    changed = async_rl.build_run(
        SNOWBALL_POLICY,
        async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("context_budget.request_window_tokens=4096",),
    )
    assert changed.rl.name.startswith("users/alice/checkpoints/async-rl/snowball-smoke-set-")
    assert changed.rl.name != run.rl.name
    assert changed.evaluation.name != run.evaluation.name
    evaluation = changed.evaluation
    served = evaluation.build_config(StepContext.for_fingerprint(evaluation.runtime_args.keys(), evaluation.deps))
    assert served.model.serve.max_model_len == 4096 + async_rl.SMOKE_PRESET.max_new_tokens


def test_qwen_smoke_selects_megatron_policy_and_pinned_skyrl_runtime(owner):
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.18", "--policy", "qwen", "--preset", "smoke"])
    assert result.exit_code == 0, result.output
    run = async_rl.build_run(
        QWEN_POLICY,
        async_rl.QWEN_SMOKE,
        version="2026.09.18",
        recipe=async_rl.QWEN_RECIPE,
        chat_template=async_rl.QWEN_CHAT_TEMPLATE,
    )
    built = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args.keys(), run.rl.deps))
    config = yaml.safe_load(built.request.config_yaml)
    assert built.request.runtime.commit == async_rl.SCORE_CENTERING_SKYRL_COMMIT
    assert built.launcher_requirement.endswith(f"@{async_rl.SCORE_CENTERING_SKYRL_COMMIT}")
    assert config["trainer"]["strategy"] == "megatron"
    assert "optimizer_checkpoint_sharding_type" not in config["trainer"]["policy"]["megatron_config"]
    assert config["generator"]["chat_template"]["name_or_path"] == "qwen3_without_thinking"
    assert config["generator"]["num_inference_engines"] == 8
    assert config["generator"]["rollout_num_nodes"] == 1
    assert config["generator"]["inference_engine_data_parallel_size"] == 1
    assert built.request.topology.role_plan.rollout_num_nodes == 1


def test_separate_engine_recipe_rejects_unallocated_node_bundles():
    with pytest.raises(ValueError, match="declared rollout node count"):
        replace(async_rl.QWEN_RECIPE, role_plan=replace(async_rl.QWEN_RECIPE.role_plan, rollout_num_nodes=2))


def test_score_centering_setting_changes_only_the_correction_at_matched_capture():
    matched = ("trainer.algorithm.use_tis=true", "generator.sampling_params.logprobs=32")
    tis = flattened(async_rl.training_config(async_rl.QWEN_SMOKE, matched, recipe=async_rl.QWEN_RECIPE))
    centered = flattened(
        async_rl.training_config(
            async_rl.QWEN_SMOKE,
            (*matched, "trainer.algorithm.score_centering_topk=32"),
            recipe=async_rl.QWEN_RECIPE,
        )
    )
    differences = {key: (tis[key], centered[key]) for key in tis if tis[key] != centered[key]}
    assert differences == {"trainer.algorithm.score_centering_topk": (0, 32)}


def test_seed_setting_reaches_the_config_and_the_request(owner):
    """MarinSkyRL writes the request's seed over the config, so both must carry the --set value."""
    assert rendered()["trainer"]["seed"] == async_rl.SEED
    assert rendered(settings=("trainer.seed=23",))["trainer"]["seed"] == 23
    run = async_rl.build_run(SNOWBALL_POLICY, async_rl.SMOKE_PRESET, version="2026.09.18", settings=("trainer.seed=23",))
    built = run.rl.build_config(StepContext.for_fingerprint(run.rl.runtime_args.keys(), run.rl.deps))
    seeds = {key: value for key, value in flattened(asdict(built)).items() if key.endswith("seed")}
    assert seeds and all(value == 23 for value in seeds.values()), seeds
