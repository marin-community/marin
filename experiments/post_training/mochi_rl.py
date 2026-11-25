# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run reinforcement learning from the production Mochi SFT export."""

from dataclasses import replace
from pathlib import Path

from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLRetentionPolicy, SkyRLRolePlan
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.locality import RoleTopologyMode
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import rl
from experiments.post_training.mochi_assets import (
    MOCHI_STEP_400407_EXPORT,
    MOCHI_TOKENIZER_REVISION,
    MOCHI_TOKENIZER_URI,
)
from experiments.post_training.rl_agentic_recipe import agentic_limits, agentic_preset
from experiments.post_training.rl_presets import (
    RL_TRAINING_BASE,
    SAMPLED_TRAJECTORIES,
    STOP_TOKEN_IDS,
    Evaluation,
    Layout,
    ModelPresets,
    Preset,
    Recipe,
    Training,
)
from experiments.post_training.task_curation.selection import VALIDATION_FILENAME
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

_MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        "models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407",
        "2026.11.24",
        MOCHI_STEP_400407_EXPORT,
    ),
    tokenizer_uri=MOCHI_TOKENIZER_URI,
    tokenizer_revision=MOCHI_TOKENIZER_REVISION,
    relative_path="",
)

_DSPARK = sections.SpeculatorModel(
    source_uri="s3://marin-us-east-08a/marin/checkpoints/dspark/mochi/ablation-interleaved-two-layer/2026.11.25-skyrl",
    source_identity="sha256:e163ef4fb50cbdfd578f1d26a27adf35b7c22e5a597a8da63b4591a2190aa468",
)

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
    workload=Training(_ONE_RACK_PLAN),
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
    workload=Training(
        replace(
            _ONE_RACK_PLAN,
            policy_num_nodes=16,
            num_inference_engines=8,
            inference_engine_data_parallel_size=16,
            inference_engine_expert_parallel_size=16,
        ),
    ),
)

_SPLITTING_OPS = ("vllm::unified_attention_with_output", "vllm::unified_kv_cache_update", "vllm::grug_moe_short_conv")
_ONE_TURN = RL_TRAINING_BASE.merge(
    RecipePatch(
        context_budget=ContextBudget(request_window_tokens=65_536, max_new_tokens_per_turn=16_384, max_turns=1),
        trainer=sections.Trainer(
            flash_attn=True,
            use_sample_packing=True,
            epochs=10_000,
            eval_batch_size=120,
            project_name="marin-mochi-rl",
            algorithm=sections.Algorithm(
                off_policy_correction="tis",
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
            rollout_buffer=sections.RolloutBuffer(max_staleness_steps=3, batch_policy="full_batch"),
        ),
        generator=sections.Generator(
            model_dtype="bfloat16",
            gpu_memory_utilization=0.775,
            max_num_seqs=16,
            chat_template_kwargs={"enable_thinking": True},
            sampling_params=sections.SamplingParams(
                temperature=0.7, top_p=1.0, logprobs=0, stop_token_ids=STOP_TOKEN_IDS, skip_special_tokens=False
            ),
            engine_init_kwargs={
                "model_loader_extra_config": {"distributed": False, "concurrency": 4, "memory_limit": 8 * 1024**3},
                "compilation_config": {"mode": 3, "cudagraph_mode": "PIECEWISE", "splitting_ops": _SPLITTING_OPS},
            },
            trajectory_retention=SAMPLED_TRAJECTORIES.replace(
                max_bytes_per_step=64 * 1024**2,
                max_bytes_per_run=512 * 1024**2,
            ),
        ),
        trajectory_runner=sections.TrajectoryRunner(
            rollout_workers=sections.RolloutWorkers(num_workers=32, cpus_per_worker=2, start_interval_seconds=0)
        ),
        extra_env={
            "OT_AGENT_RAY_OBJECT_STORE_CAP_GIB": "32",
        },
    )
)

_TRANSFORMER_KWARGS = {"bias_activation_fusion": True, "grug_checkpoint_cpu_offload": True}
_THREE_RACK_PATCH = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=262_144, max_new_tokens_per_turn=16_384, max_turns=1),
    trainer=sections.Trainer(
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
                "inductor_compile_config": {
                    "enable_auto_functionalized_v2": False,
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                },
            }
        },
    ),
    extra_env={"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0", "NCCL_SOCKET_FAMILY": "AF_INET"},
)

_EVALUATION = Evaluation(
    name="eval_lite_100",
    file=Path(VALIDATION_FILENAME).stem,
    source="held_out",
    profile=EvaluationProfile(
        sampling_params=EvalSamplingParams(
            temperature=0.7, top_p=1.0, stop_token_ids=STOP_TOKEN_IDS, skip_special_tokens=False
        ),
        n_samples_per_prompt=1,
    ),
    every=2,
)

# Four hours covers the trial; eight hours covers the trial and one retry.
_AGENTIC_LIMITS = agentic_limits(
    trial_timeout=14_400,
    agent_timeout=10_800,
    stall_timeout=28_800,
    memory_mb=4096,
)


_TRAIN_LONG = Preset(
    name="train-long",
    revision="2026.11.25",
    recipe=Recipe(layout=ONE_RACK, fixed_reply_tokens=None, config=_ONE_TURN),
    updates=10_000,
    measured=range(1, 10_001),
    tasks_input="optional",
    accepts_settings=True,
    accepts_init=True,
    publishes=False,
    control_preset=None,
    checkpoints="none",
    data_settings=frozenset({"shuffle"}),
    tasks="snowball-100k@2026.11.19.1",
    model=_MODEL,
    speculative_decoding_model=_DSPARK,
    held_out_tasks="eval-lite@2026.11.17.1",
    evaluations=(_EVALUATION,),
    seed=20261020,
    sample_namespace="datasets/mochi-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3),
    temporary_output=True,
    export="none",
)
_TRAIN_LONG_3RACK = replace(
    _TRAIN_LONG,
    name="train-long-3rack",
    recipe=replace(_TRAIN_LONG.recipe, layout=THREE_RACK, config=_ONE_TURN.merge(_THREE_RACK_PATCH)),
    speculative_decoding_model=None,
)
_TRAIN_MEDIUM = replace(
    _TRAIN_LONG.with_updates(2),
    name="train-medium",
    evaluations=(),
)
_TRAIN_LONG_AGENTIC = agentic_preset(_TRAIN_LONG, name="train-long-agentic", limits=_AGENTIC_LIMITS)

MOCHI = ModelPresets(
    model="mochi",
    presets=(
        _TRAIN_MEDIUM,
        _TRAIN_LONG,
        _TRAIN_LONG_3RACK,
        agentic_preset(_TRAIN_MEDIUM, name="train-medium-agentic", limits=_AGENTIC_LIMITS),
        _TRAIN_LONG_AGENTIC,
        agentic_preset(_TRAIN_LONG_3RACK, name="train-long-agentic-3rack", limits=_AGENTIC_LIMITS),
    ),
    namespace="mochi-rl",
    control_pins={},
    default_export_marker=MODEL_MANIFEST_FILENAME,
)


if __name__ == "__main__":

    def sft_presets():
        # Load the SFT training dependencies only for --sft.
        from experiments.post_training.mochi_sft import MOCHI_SFT_PRESETS  # noqa: PLC0415

        return MOCHI_SFT_PRESETS

    raise SystemExit(rl.main(MOCHI, sft_presets=sft_presets))
