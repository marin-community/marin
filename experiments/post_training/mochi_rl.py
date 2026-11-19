# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run reinforcement learning from the retained production Mochi SFT export."""

from dataclasses import replace
from pathlib import Path

from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLRetentionPolicy, SkyRLRolePlan
from marin.training.training import LevanterCheckpoint
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationSampling
from skyrlrecipe.locality import RoleTopologyMode
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import rl
from experiments.post_training.rl_agentic_recipe import AGENTIC_BATCH_MIX, AGENTIC_RECIPE, agentic_limits
from experiments.post_training.rl_presets import Evaluation, Layout, ModelPresets, Preset, Recipe, Training
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.task_curation.selection import VALIDATION_FILENAME

_MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        "models/mochi-sft-step-400407",
        "2026.11.13",
        "s3://marin-us-east-02a/marin/exports/mochi-sft/" "hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
        kind=LevanterCheckpoint,
    ),
    tokenizer_uri="marin-community/marin-tokenizer-tools",
    tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    relative_path="",
)
_STOP_TOKEN_IDS = (128001, 128009)

_ONE_RACK_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=12,
    policy_num_gpus_per_node=4,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=12,
    inference_engine_expert_parallel_size=12,
    train_batch_size=32,
    policy_mini_batch_size=32,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
    entrypoint="taskcompendium",
)
ONE_RACK = Layout(
    name="mochi-one-rack",
    workload=Training(_ONE_RACK_PLAN, epochs_follow_updates=True),
    megatron=MegatronGeometry(1, 12, 4, 4, 1),
    cluster="cw-us-east-08a",
    host_memory="900GB",
    gpu_variant="GB200",
    host_cpu=120,
    host_disk="2TB",
    priority="production",
    coordinator_timeout_hours=96,
    job_timeout_seconds=48 * 3600,
    role_topology=RoleTopologyMode.NVLINK_DOMAIN,
)
THREE_RACK = replace(
    ONE_RACK,
    name="mochi-three-rack",
    megatron=replace(ONE_RACK.megatron, pipeline_model_parallel_size=16),
    workload=replace(
        ONE_RACK.workload,
        plan=replace(
            _ONE_RACK_PLAN,
            policy_num_nodes=16,
            num_inference_engines=8,
            inference_engine_data_parallel_size=16,
            inference_engine_expert_parallel_size=16,
        ),
    ),
)

_SPLITTING_OPS = ("vllm::unified_attention_with_output", "vllm::unified_kv_cache_update", "vllm::grug_moe_short_conv")
_ONE_TURN = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=65_536, max_new_tokens_per_turn=16_384, max_turns=1),
    environment=sections.Environment(task_sessions={"local": {}}),
    data=sections.Data(kind="parquet", shuffle=True),
    trainer=sections.Trainer(
        strategy="megatron",
        flash_attn=True,
        use_sample_packing=True,
        gradient_checkpointing=True,
        offload_optimizer_during_rollouts=False,
        update_epochs_per_batch=1,
        eval_batch_size=120,
        logger=("console", "records", "wandb"),
        project_name="mochi-rl",
        tracker_commit_each_step=True,
        training_metrics=True,
        rollout_spans=True,
        policy_train_spans=True,
        algorithm=sections.Algorithm(
            advantage_estimator="rloo_n",
            policy_loss_type="regular",
            loss_reduction="token_mean",
            use_kl_loss=False,
            off_policy_correction="tis",
            group_advantage_min_size=2,
            advantage_batch_normalize=False,
            rloo_n_filter_zero_reward_groups=True,
            tis_splice=True,
        ),
        policy=sections.Policy(
            optimizer_config=sections.PolicyOptimizerConfig(
                optimizer="MuonH",
                lr=1e-5,
                weight_decay=0.0,
                max_grad_norm=0.0,
                adam_betas=(0.9, 0.95),
                optimizer_kwargs={
                    "adam_lr": 2.307692307692308e-6,
                    "momentum": 0.95,
                    "nesterov": True,
                    "backend_steps": 5,
                    "epsilon": 1e-8,
                    "muon_epsilon": 1e-8,
                    "offload_momentum": True,
                    "expert_normalization": "all_experts",
                },
            ),
            megatron_config=sections.PolicyMegatronConfig(
                optimizer_checkpoint_sharding_type="dp_reshardable", moe_router_replay=True
            ),
        ),
        rollout_buffer=sections.RolloutBuffer(max_staleness_steps=3, batch_policy="full_batch", object_store_root=None),
    ),
    generator=sections.Generator(
        backend="vllm",
        model_dtype="bfloat16",
        vllm_attention_backend="FLASH_ATTN",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_transport="expert_block",
        weight_sync_pause=sections.WeightSyncPause(mode="keep", clear_cache=True),
        expert_block_sync=sections.ExpertBlockSync(timeout_seconds=600, verify=False),
        gpu_memory_utilization=0.75,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        chat_template=None,
        chat_template_kwargs={"enable_thinking": True},
        sampling_params=sections.SamplingParams(
            temperature=0.7, top_p=1.0, logprobs=0, stop_token_ids=_STOP_TOKEN_IDS, skip_special_tokens=False
        ),
        trajectory_reward_shaping=sections.TrajectoryRewardShaping(
            enabled=True,
            overlong=sections.Overlong(penalty_scale=0.0),
            reasoning_span=sections.ReasoningSpan(penalty=0.1, target="marker"),
        ),
        engine_init_kwargs={
            "moe_backend": "triton",
            "enable_auto_tools": True,
            "tool_parser": "hermes",
            "model_loader_extra_config": {"distributed": False, "concurrency": 4, "memory_limit": 8 * 1024**3},
            "compilation_config": {"mode": 3, "cudagraph_mode": "PIECEWISE", "splitting_ops": _SPLITTING_OPS},
            "enable_return_routed_experts": True,
        },
    ),
    trajectory_runner=sections.TrajectoryRunner(
        rollout_workers=sections.RolloutWorkers(num_workers=32, cpus_per_worker=2, start_interval_seconds=0)
    ),
    extra_env={
        "OT_AGENT_RAY_OBJECT_STORE_CAP_GIB": "32",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    },
)

_TRANSFORMER_KWARGS = {"bias_activation_fusion": True, "grug_checkpoint_cpu_offload": True}
_THREE_RACK_PATCH = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=262_144, max_new_tokens_per_turn=16_384, max_turns=1),
    trainer=sections.Trainer(
        project_name="mochi-three-rack",
        rollout_buffer=sections.RolloutBuffer(max_in_flight=128),
        policy=sections.Policy(
            megatron_config=sections.PolicyMegatronConfig(transformer_config_kwargs=_TRANSFORMER_KWARGS)
        ),
        ref=sections.Ref(megatron_config=sections.RefMegatronConfig(transformer_config_kwargs=_TRANSFORMER_KWARGS)),
    ),
    generator=sections.Generator(
        vllm_attention_backend="FLASHINFER",
        max_num_seqs=128,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        expert_block_sync=sections.ExpertBlockSync(verify=True, timeout_seconds=1800),
        engine_init_kwargs={
            "compilation_config": {
                "mode": 3,
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": _SPLITTING_OPS,
                "inductor_compile_config": {
                    "enable_auto_functionalized_v2": False,
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                },
            }
        },
    ),
    extra_env={"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": "ibp"},
)

_EVALUATION = Evaluation(
    name="eval_lite",
    file=Path(VALIDATION_FILENAME).stem,
    source="eval_lite",
    profile=EvaluationSampling(
        sampling_params=EvalSamplingParams(
            temperature=0.7, top_p=1.0, stop_token_ids=_STOP_TOKEN_IDS, skip_special_tokens=False
        ),
        n_samples_per_prompt=1,
    ),
    every=2,
)

TRAIN_LONG = Preset(
    name="train-long",
    revision="2026.11.19.1",
    recipe=Recipe(layout=ONE_RACK, fixed_reply_tokens=None, config=_ONE_TURN),
    updates=10_000,
    measured=range(1, 10_001),
    tasks_input="optional",
    accepts_settings=True,
    accepts_init=True,
    publishes=False,
    controls=None,
    checkpoint_every=2,
    owned_keys=frozenset({"generator.run_engines_locally"}),
    data_settings=frozenset({"shuffle"}),
    tasks="snowball-100k@2026.11.19.1",
    model=_MODEL,
    eval_lite="eval-lite@2026.11.17.1",
    evaluations=(_EVALUATION,),
    seed=20261020,
    sample_namespace="datasets/mochi-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3),
    export="best",
)
TRAIN_LONG_3RACK = replace(
    TRAIN_LONG,
    name="train-long-3rack",
    recipe=replace(TRAIN_LONG.recipe, layout=THREE_RACK, config=_ONE_TURN.merge(_THREE_RACK_PATCH)),
)
TRAIN_MEDIUM = replace(
    TRAIN_LONG.with_updates(2),
    name="train-medium",
    evaluations=(),
    export="none",
)


def _agentic(preset: Preset, name: str) -> Preset:
    config = preset.recipe.config
    assert config.context_budget is not None
    config = config.merge(AGENTIC_RECIPE).merge(
        agentic_limits(attempt_timeout=14_400, model_turn_timeout=10_800, ttl_minutes=300, stall_timeout=30_000)
    )
    config = config.merge(RecipePatch(context_budget=config.context_budget.replace(max_turns=128)))
    return replace(
        preset,
        name=name,
        tasks_input="required",
        batch_mix=AGENTIC_BATCH_MIX,
        harbor=SNOWBALL.by_name["train-long-harbor"].harbor,
        recipe=replace(preset.recipe, config=config),
    )


MOCHI = ModelPresets(
    model="mochi",
    presets=(
        TRAIN_MEDIUM,
        TRAIN_LONG,
        TRAIN_LONG_3RACK,
        _agentic(TRAIN_LONG, "train-long-agentic"),
        _agentic(TRAIN_LONG_3RACK, "train-long-agentic-3rack"),
    ),
    namespace="mochi-rl",
    controls={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(MOCHI))
