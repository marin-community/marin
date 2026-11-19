# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model, layouts and RL experiment presets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import (
    ArtifactHfModel,
    MegatronGeometry,
    SkyRLGenerationPlan,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
)
from marin.training.training import LevanterCheckpoint
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationSampling
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import rl
from experiments.post_training.rl_agentic_recipe import HARBOR_EVALUATION, agentic, agentic_limits
from experiments.post_training.rl_presets import (
    UPDATE_SECONDS_METRIC,
    ControlPin,
    Evaluation,
    Generation,
    Layout,
    Metric,
    ModelPresets,
    Preset,
    Recipe,
    Training,
    WindowStatistic,
)
from experiments.post_training.rl_recipe import BATCH_SETTINGS
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)

SEED = 20261020

MODEL_ID = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"

MODEL_REVISION = "fe666478454e79fe6027a0bab0388d9abd68f51a"

MODEL_URI = "s3://marin-us-east-02a/models/open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"

DEFAULT_SELECTION_VERSION = "2026.11.19.1"

DEFAULT_EVAL_LITE_NAME = "eval-lite"

DEFAULT_EVAL_LITE_VERSION = "2026.11.17.1"

EVAL_REPLY_TOKENS = 8192

_DEFAULT_TASKS = f"snowball-100k@{DEFAULT_SELECTION_VERSION}"

_DEFAULT_EVAL_LITE = f"{DEFAULT_EVAL_LITE_NAME}@{DEFAULT_EVAL_LITE_VERSION}"

_DEFAULT_MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        f"models/snowball-datakit-sft-{MODEL_REVISION}", "2026.09.21", MODEL_URI, kind=LevanterCheckpoint
    ),
    MODEL_ID,
    MODEL_REVISION,
    relative_path="",
)

_DEFAULT_PATCH = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=24576, max_new_tokens_per_turn=8192, max_turns=1),
    environment=sections.Environment(task_sessions={"local": {}}),
    data=sections.Data(kind="parquet", shuffle=True),
    trainer=sections.Trainer(
        strategy="megatron",
        flash_attn=False,
        use_sample_packing=False,
        gradient_checkpointing=True,
        offload_optimizer_during_rollouts=False,
        epochs=50,
        update_epochs_per_batch=1,
        eval_batch_size=256,
        eval_before_train=False,
        logger=("console", "records", "wandb"),
        project_name="marin-snowball-rl",
        tracker_commit_each_step=True,
        training_metrics=True,
        rollout_spans=True,
        policy_train_spans=True,
        algorithm=sections.Algorithm(
            advantage_estimator="rloo_n",
            off_policy_correction="none",
            policy_loss_type="regular",
            loss_reduction="token_mean",
            use_kl_loss=False,
            kl_estimator_type="k3_unbiased_gradient",
            eps_clip_low=0.2,
            eps_clip_high=0.2,
            group_advantage_min_size=2,
            advantage_batch_normalize=False,
            rloo_n_filter_zero_reward_groups=True,
            use_entropy_loss=False,
        ),
        policy=sections.Policy(
            optimizer_config=sections.PolicyOptimizerConfig(
                optimizer="AdamW",
                lr=1e-6,
                adam_betas=(0.9, 0.999),
                weight_decay=0.0,
                max_grad_norm=0.5,
                scheduler="constant_with_warmup",
                num_warmup_steps=0,
            ),
        ),
        rollout_buffer=sections.RolloutBuffer(object_store_root=None),
    ),
    generator=sections.Generator(
        backend="vllm",
        vllm_attention_backend="FLASH_ATTN",
        run_engines_locally=True,
        weight_sync_backend="nccl",
        weight_sync_pause=sections.WeightSyncPause(mode="keep", clear_cache=True),
        weight_sync_transport="expert_block",
        expert_block_sync=sections.ExpertBlockSync(timeout_seconds=600, verify=False),
        gpu_memory_utilization=0.75,
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        trajectory_reward_shaping=sections.TrajectoryRewardShaping(
            enabled=True,
            overlong=sections.Overlong(penalty_scale=0.0),
            reasoning_span=sections.ReasoningSpan(penalty=0.1, target="marker"),
        ),
        enable_chunked_prefill=True,
        enforce_eager=False,
        chat_template=None,
        apply_overlong_filtering=False,
        engine_init_kwargs={"moe_backend": "triton", "enable_auto_tools": True, "tool_parser": "hermes"},
        sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
    ),
    trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=1024),
    extra_env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    },
)

_MODE_KEYS = (
    "trainer.rollout_buffer.max_staleness_steps",
    "trainer.rollout_buffer.batch_policy",
    "trainer.algorithm.off_policy_correction",
)

_HELD_OUT_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=1,
    sampling_params=EvalSamplingParams(
        temperature=0.0, top_p=1.0, logprobs=None, ignore_eos=False, max_generate_length=EVAL_REPLY_TOKENS
    ),
)

_THROUGHPUT_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=True),
)

_SCREENING_SAMPLING = EvaluationSampling(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=False),
)

EVALUATIONS = MappingProxyType(
    {
        evaluation.name: evaluation
        for evaluation in (
            Evaluation(
                name="eval_lite",
                file=Path(FULL_VALIDATION_FILENAME).stem,
                source="eval_lite",
                profile=_HELD_OUT_SAMPLING,
            ),
            Evaluation(
                name="eval_lite_100",
                file=Path(VALIDATION_FILENAME).stem,
                source="eval_lite",
                profile=_HELD_OUT_SAMPLING,
            ),
            Evaluation(
                name="speed_512x4",
                file=Path(TRAIN_FILENAME).stem,
                sample=512,
                profile=_THROUGHPUT_SAMPLING,
            ),
            Evaluation(
                name="screen_256x4",
                file=Path(TRAIN_FILENAME).stem,
                source="tasks",
                sample=256,
                profile=_SCREENING_SAMPLING,
            ),
        )
    }
)

_P40_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=128,
    policy_mini_batch_size=128,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)

P40 = Layout(
    name="p40",
    workload=Training(_P40_PLAN),
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    ),
    cluster="cw-us-east-02a",
    host_memory="1800GB",
    gpu_variant="H100",
    host_cpu=32,
    host_disk="2TB",
    priority="interactive",
    coordinator_timeout_hours=72,
    job_timeout_seconds=72 * 3600,
)

E8 = replace(
    P40,
    name="e8",
    workload=Generation(
        SkyRLGenerationPlan(
            gpus_per_node=8,
            num_inference_engines=1,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=8,
            inference_engine_expert_parallel_size=8,
            n_samples_per_prompt=4,
        ),
    ),
)

_ASYNC_P40 = Recipe(
    layout=P40,
    fixed_reply_tokens=None,
    config=_DEFAULT_PATCH.merge(
        RecipePatch(
            trainer=sections.Trainer(
                rollout_buffer=sections.RolloutBuffer(max_staleness_steps=2, batch_policy="full_batch"),
                algorithm=sections.Algorithm(off_policy_correction="tis"),
            )
        )
    ),
)

_GENERATING = Recipe(layout=E8, fixed_reply_tokens=None, config=_DEFAULT_PATCH, training_plan=_P40_PLAN)

_GAPS = (
    "aaii-coverage",
    "no-agentic-tasks",
    "no-image-sandbox",
)

_TASK_SET = EVALUATIONS["eval_lite_100"]

_BATCH_KEYS = frozenset(BATCH_SETTINGS)

_OWNED_KEYS = frozenset({"generator.run_engines_locally", "generator.trajectory_retention", *_MODE_KEYS})

_GATED_KEYS = frozenset(
    {
        "trainer.max_steps",
        "trainer.logger",
        "trainer.project_name",
        "context_budget",
        "generator.sampling_params.ignore_eos",
        "generator.sampling_params.min_tokens",
    }
)

_SPEED_KEYS = frozenset(
    {"generator.sampling_params.logprobs", "trainer.training_metrics", "generator.apply_overlong_filtering"}
)

_LEARNING_MEDIUM = Preset(
    tasks=_DEFAULT_TASKS,
    model=_DEFAULT_MODEL,
    eval_lite=_DEFAULT_EVAL_LITE,
    seed=SEED,
    sample_namespace="datasets/snowball-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
    export="latest",
    name="learning-medium",
    revision="2026.11.19.1",
    recipe=_ASYNC_P40,
    updates=25,
    measured=range(21, 26),
    tasks_input="forbidden",
    accepts_settings=True,
    accepts_init=True,
    publishes=True,
    controls="learning-medium",
    checkpoint_every=None,
    owned_keys=_OWNED_KEYS | _GATED_KEYS,
    data_settings=frozenset({"shuffle", "sampling"}),
    evaluations=(replace(EVALUATIONS["eval_lite"], at_start=True),),
    guards=(Metric("eval_lite/family_mean", True),),
    objectives=(
        Metric("eval_lite/family_mean", True),
        Metric("eval_lite/family_mean/gain_per_h100_hour", True, WindowStatistic.GAIN_PER_GPU_HOUR),
    ),
    gaps=_GAPS,
    budget="asynchronous runtime unmeasured",
)


_TRAIN_LONG = Preset(
    tasks=_DEFAULT_TASKS,
    model=_DEFAULT_MODEL,
    eval_lite=_DEFAULT_EVAL_LITE,
    seed=SEED,
    sample_namespace="datasets/snowball-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=5),
    export="latest",
    name="train-long",
    revision="2026.11.19.1",
    recipe=_ASYNC_P40,
    updates=100,
    measured=range(1, 101),
    tasks_input="optional",
    accepts_settings=True,
    accepts_init=True,
    publishes=True,
    controls=None,
    checkpoint_every=10,
    owned_keys=_OWNED_KEYS,
    data_settings=frozenset({"shuffle", "sampling"}),
    evaluations=(replace(_TASK_SET, every=10, at_start=True),),
    gaps=_GAPS,
)

_CAMPAIGN_PLAN = replace(
    _P40_PLAN,
    policy_num_nodes=10,
    num_inference_engines=22,
    train_batch_size=64,
    policy_mini_batch_size=64,
    n_samples_per_prompt=8,
    entrypoint="taskcompendium",
)
_CAMPAIGN_LAYOUT = replace(
    P40,
    name="rno2a-256",
    workload=Training(_CAMPAIGN_PLAN),
    megatron=replace(P40.megatron, context_parallel_size=5),
    cluster="cw-rno2a",
    priority="production",
    # Each campaign job has a 12-hour allocation and resumes from its committed checkpoint.
    job_timeout_seconds=12 * 3600,
)
_CAMPAIGN_PATCH = RecipePatch(
    context_budget=ContextBudget(request_window_tokens=102_400, max_new_tokens_per_turn=16_384, max_turns=128),
    trainer=sections.Trainer(
        use_sample_packing=True,
        rollout_buffer=sections.RolloutBuffer(max_in_flight=32),
        policy=sections.Policy(
            megatron_config=sections.PolicyMegatronConfig(
                transformer_config_kwargs={"cp_comm_type": "a2a", "moe_router_fusion": False},
                moe_router_replay=True,
            )
        ),
        ref=sections.Ref(megatron_config=sections.RefMegatronConfig(transformer_config_kwargs={"cp_comm_type": "a2a"})),
        algorithm=sections.Algorithm(tis_splice=True),
    ),
    generator=sections.Generator(
        max_num_seqs=64,
        chat_template_kwargs={"enable_thinking": True},
        engine_init_kwargs={"enable_auto_tools": True, "tool_parser": "hermes"},
        trajectory_retention=sections.TrajectoryRetention(
            phases=("train", "eval"),
            sample_fraction=0.1,
            sample_count_per_step=0,
            required=True,
            always_retain_failures=False,
            always_retain_non_terminating=False,
            always_retain_loops=False,
        ),
    ),
)

_TRAIN_LONG_AGENTIC = agentic(
    replace(
        _TRAIN_LONG.with_updates(10_000),
        data_settings=frozenset({"shuffle"}),
        export="best",
        evaluations=(replace(_TASK_SET, every=2),),
        checkpoint_every=2,
        gaps=(*(gap for gap in _TRAIN_LONG.gaps if gap != "no-agentic-tasks"), "harbor-on-daytona"),
    ),
    name="train-long-agentic",
    layout=_CAMPAIGN_LAYOUT,
    # The sandbox covers one 90-minute trial; admission covers the trial and one retry.
    limits=agentic_limits(
        attempt_timeout=5400,
        session_attempt_timeout=2520,
        model_turn_timeout=1800,
        ttl_minutes=90,
        stall_timeout=10_800,
    ).merge(_CAMPAIGN_PATCH),
)
_TRAIN_MEDIUM_AGENTIC = replace(
    _TRAIN_LONG_AGENTIC.with_updates(10),
    name="train-medium-agentic",
    recipe=replace(
        _TRAIN_LONG_AGENTIC.recipe,
        layout=replace(
            _CAMPAIGN_LAYOUT,
            name="rno2a-88",
            workload=Training(replace(_CAMPAIGN_PLAN, num_inference_engines=1)),
            priority="interactive",
        ),
    ),
)

_PRESETS = (
    Preset(
        tasks=_DEFAULT_TASKS,
        model=_DEFAULT_MODEL,
        eval_lite=_DEFAULT_EVAL_LITE,
        seed=SEED,
        sample_namespace="datasets/snowball-rl",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        export="none",
        name="speed-short",
        revision="2026.11.19.1",
        recipe=replace(_GENERATING, fixed_reply_tokens=4096),
        updates=0,
        measured=range(1),
        tasks_input="forbidden",
        accepts_settings=True,
        accepts_init=False,
        publishes=False,
        controls="speed-short",
        checkpoint_every=None,
        owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        data_settings=frozenset(),
        evaluations=(EVALUATIONS["speed_512x4"], EVALUATIONS["eval_lite"]),
        guards=(Metric("generate/tokens_per_second_per_gpu", True), Metric("eval_lite/family_mean", True)),
        objectives=(Metric("generate/tokens_per_second_per_gpu", True),),
        gaps=_GAPS,
        budget="14-17 min; 2 H100-h",
    ),
    Preset(
        tasks=_DEFAULT_TASKS,
        model=_DEFAULT_MODEL,
        eval_lite=_DEFAULT_EVAL_LITE,
        seed=SEED,
        sample_namespace="datasets/snowball-rl",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        export="none",
        name="data-short",
        revision="2026.11.19.1",
        recipe=_GENERATING,
        updates=0,
        measured=range(1),
        tasks_input="required",
        accepts_settings=False,
        accepts_init=False,
        publishes=False,
        controls="data-short",
        checkpoint_every=None,
        owned_keys=_OWNED_KEYS | _GATED_KEYS | _BATCH_KEYS,
        data_settings=frozenset(),
        evaluations=(EVALUATIONS["screen_256x4"],),
        guards=(Metric("screen_256x4/mixed_fraction", True),),
        objectives=(Metric("screen_256x4/mixed_fraction", True),),
        gaps=_GAPS,
        budget="10-15 min; 2 H100-h",
    ),
    Preset(
        tasks=_DEFAULT_TASKS,
        model=_DEFAULT_MODEL,
        eval_lite=_DEFAULT_EVAL_LITE,
        seed=SEED,
        sample_namespace="datasets/snowball-rl",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        export="none",
        evaluations=(),
        name="speed-medium",
        revision="2026.11.19.1",
        recipe=replace(
            _ASYNC_P40,
            fixed_reply_tokens=4096,
            config=_ASYNC_P40.config.merge(
                RecipePatch(
                    trainer=sections.Trainer(algorithm=sections.Algorithm(rloo_n_filter_zero_reward_groups=False))
                )
            ),
        ),
        updates=10,
        measured=range(2, 11),
        tasks_input="forbidden",
        accepts_settings=True,
        accepts_init=False,
        publishes=False,
        controls="speed-medium",
        checkpoint_every=None,
        owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        data_settings=frozenset(),
        guards=(
            Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),
            Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
        ),
        objectives=(Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),),
        gaps=(*_GAPS, "mismatch-is-agreement", "layout-p40-only"),
        budget="asynchronous runtime unmeasured",
    ),
    Preset(
        tasks=_DEFAULT_TASKS,
        model=_DEFAULT_MODEL,
        eval_lite=_DEFAULT_EVAL_LITE,
        seed=SEED,
        sample_namespace="datasets/snowball-rl",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
        export="none",
        evaluations=(),
        name="speed-long",
        revision="2026.11.19.1",
        recipe=_ASYNC_P40,
        updates=20,
        measured=range(3, 21),
        tasks_input="forbidden",
        accepts_settings=True,
        accepts_init=False,
        publishes=False,
        controls="speed-long",
        checkpoint_every=None,
        owned_keys=_OWNED_KEYS | _GATED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        data_settings=frozenset(),
        guards=(
            Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),
            Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
        ),
        objectives=(Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),),
        gaps=(*_GAPS, "layout-p40-only"),
        budget="45-75 min; 30-50 H100-h",
    ),
    _LEARNING_MEDIUM,
    replace(
        _LEARNING_MEDIUM,
        name="data-medium",
        tasks_input="required",
        accepts_settings=False,
        accepts_init=False,
        controls=_LEARNING_MEDIUM.name,
        data_settings=frozenset(),
    ),
    _TRAIN_LONG,
    _TRAIN_MEDIUM_AGENTIC,
    _TRAIN_LONG_AGENTIC,
    replace(
        _TRAIN_LONG,
        name="train-long-harbor",
        revision="2026.11.19.1",
        harbor=HARBOR_EVALUATION,
        gaps=(*_TRAIN_LONG.gaps, "harbor-on-daytona"),
    ),
)

SNOWBALL = ModelPresets(
    model="snowball",
    presets=_PRESETS,
    namespace="snowball-rl",
    controls={
        name: ControlPin(name, "648b5687f0", "2026.11.06.1", count)
        for name, count in (
            ("speed-short", 5),
            ("data-short", 4),
            ("speed-medium", 4),
            ("learning-medium", 4),
            ("speed-long", 4),
        )
    },
)

if __name__ == "__main__":
    raise SystemExit(rl.main(SNOWBALL))
