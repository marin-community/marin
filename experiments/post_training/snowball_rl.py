# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model, layouts and RL experiment presets."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
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
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.documents import set_path
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import rl
from experiments.post_training.rl_compose import BATCH_SETTINGS
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
        ckpt_interval=0,
        hf_save_interval=-1,
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
        enable_http_endpoint=False,
        chat_template=None,
        apply_overlong_filtering=False,
        engine_init_kwargs={"moe_backend": "triton", "enable_auto_tools": True, "tool_parser": "hermes"},
        sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
    ),
    trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=1024),
    extra_env={
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # The privileged profile below exposes the front-end Ethernet NIC's RoCE functions
        # (mlx5_*) beside the InfiniBand rails. NCCL routes some channels over them, which fails
        # with IBV_WC_RETRY_EXC_ERR(12). Pin NCCL to the rails (ibp0-ibp7).
        "NCCL_IB_HCA": "ibp",
    },
)

_MODE_KEYS = (
    "trainer.rollout_buffer.max_staleness_steps",
    "trainer.rollout_buffer.batch_policy",
    "trainer.algorithm.off_policy_correction",
)

_HELD_OUT_SAMPLING = EvaluationProfile(
    n_samples_per_prompt=1,
    sampling_params=EvalSamplingParams(
        temperature=0.0, top_p=1.0, logprobs=None, ignore_eos=False, max_generate_length=EVAL_REPLY_TOKENS
    ),
)

_THROUGHPUT_SAMPLING = EvaluationProfile(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=True),
)

_SCREENING_SAMPLING = EvaluationProfile(
    n_samples_per_prompt=4,
    sampling_params=EvalSamplingParams(temperature=1.0, top_p=1.0, logprobs=None, ignore_eos=False),
)

EVALUATIONS = MappingProxyType(
    {
        evaluation.name: evaluation
        for evaluation in (
            Evaluation("eval_lite", Path(FULL_VALIDATION_FILENAME).stem, _HELD_OUT_SAMPLING, source="held_out"),
            Evaluation("eval_lite_100", Path(VALIDATION_FILENAME).stem, _HELD_OUT_SAMPLING, source="held_out"),
            Evaluation("speed_512x4", Path(TRAIN_FILENAME).stem, _THROUGHPUT_SAMPLING, sample=512),
            Evaluation("screen_256x4", Path(TRAIN_FILENAME).stem, _SCREENING_SAMPLING, source="tasks", sample=256),
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
        )
    ),
)

_GENERATION_DOCUMENT = {}

for _key, _value in _P40_PLAN.config_values().items():
    if _key.startswith("trainer."):
        set_path(_GENERATION_DOCUMENT, _key, _value)

_GENERATION_NATIVE_DEFAULTS = RecipePatch.from_document(_GENERATION_DOCUMENT)

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

_EXPORT_BEST = replace(
    _ASYNC_P40,
    config=_ASYNC_P40.config.merge(
        RecipePatch(trainer=sections.Trainer(hf_export_checkpoint="best", dump_grader_report=True))
    ),
)

_GENERATING = Recipe(layout=E8, fixed_reply_tokens=None, config=_DEFAULT_PATCH.merge(_GENERATION_NATIVE_DEFAULTS))

_GAPS = ("aaii-coverage", "no-agentic-tasks", "no-image-sandbox")

_TASK_SET = EVALUATIONS["eval_lite_100"]

_BATCH_KEYS = frozenset(BATCH_SETTINGS)

_OWNED_KEYS = frozenset(_MODE_KEYS)
_BEST_EXPORT_KEYS = _OWNED_KEYS | {"trainer.hf_export_checkpoint"}

_SPEED_KEYS = frozenset(
    {"generator.sampling_params.logprobs", "trainer.training_metrics", "generator.apply_overlong_filtering"}
)

_SPEED_OBJECTIVES = (Metric(UPDATE_SECONDS_METRIC, False, WindowStatistic.PHASE_SUM_MEDIAN),)
_SPEED_GUARDS = (
    *_SPEED_OBJECTIVES,
    Metric("policy/mismatch/staleness0/kl_k3", False, WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN),
)

_snowball = partial(
    Preset,
    revision="2026.11.20.3",
    tasks=_DEFAULT_TASKS,
    model=_DEFAULT_MODEL,
    seed=SEED,
    sample_namespace="datasets/snowball-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
    owned_keys=_OWNED_KEYS,
    gaps=_GAPS,
)

_LEARNING_MEDIUM = _snowball(
    held_out_tasks=_DEFAULT_EVAL_LITE,
    export_hf=True,
    name="learning-medium",
    recipe=_EXPORT_BEST,
    owned_keys=_BEST_EXPORT_KEYS,
    updates=25,
    measured=range(21, 26),
    accepts_init=True,
    control_preset="learning-medium",
    data_settings=frozenset({"shuffle", "sampling"}),
    evaluations=(replace(EVALUATIONS["eval_lite"], at_start=True),),
    guards=(Metric("eval_lite/family_mean", True),),
    objectives=(
        Metric("eval_lite/family_mean", True),
        Metric("eval_lite/family_mean/gain_per_gpu_hour", True, WindowStatistic.GAIN_PER_GPU_HOUR),
    ),
    budget="asynchronous runtime unmeasured",
)

_PRESETS = (
    _snowball(
        held_out_tasks=_DEFAULT_EVAL_LITE,
        name="speed-short",
        recipe=replace(_GENERATING, fixed_reply_tokens=4096),
        updates=0,
        measured=range(1),
        control_preset="speed-short",
        owned_keys=_OWNED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        evaluations=(EVALUATIONS["speed_512x4"], EVALUATIONS["eval_lite"]),
        guards=(Metric("generate/tokens_per_second_per_gpu", True), Metric("eval_lite/family_mean", True)),
        objectives=(Metric("generate/tokens_per_second_per_gpu", True),),
        budget="14-17 min; 2 H100-h",
    ),
    _snowball(
        held_out_tasks=_DEFAULT_EVAL_LITE,
        name="data-short",
        recipe=_GENERATING,
        updates=0,
        measured=range(1),
        tasks_input="required",
        accepts_settings=False,
        control_preset="data-short",
        owned_keys=_OWNED_KEYS | _BATCH_KEYS,
        evaluations=(EVALUATIONS["screen_256x4"],),
        guards=(Metric("screen_256x4/mixed_fraction", True),),
        objectives=(Metric("screen_256x4/mixed_fraction", True),),
        budget="10-15 min; 2 H100-h",
    ),
    _snowball(
        evaluations=(),
        name="speed-medium",
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
        control_preset="speed-medium",
        owned_keys=_OWNED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        guards=_SPEED_GUARDS,
        objectives=_SPEED_OBJECTIVES,
        gaps=(*_GAPS, "mismatch-is-agreement", "layout-p40-only"),
        budget="asynchronous runtime unmeasured",
    ),
    _snowball(
        evaluations=(),
        name="speed-long",
        recipe=_ASYNC_P40,
        updates=20,
        measured=range(3, 21),
        control_preset="speed-long",
        owned_keys=_OWNED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        guards=_SPEED_GUARDS,
        objectives=_SPEED_OBJECTIVES,
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
        control_preset=_LEARNING_MEDIUM.name,
        data_settings=frozenset(),
    ),
    _snowball(
        held_out_tasks=_DEFAULT_EVAL_LITE,
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=5),
        export_hf=True,
        name="train-long",
        recipe=_EXPORT_BEST,
        owned_keys=_BEST_EXPORT_KEYS,
        updates=100,
        measured=range(1, 101),
        tasks_input="optional",
        accepts_init=True,
        checkpoint_every=10,
        data_settings=frozenset({"shuffle", "sampling"}),
        evaluations=(replace(_TASK_SET, every=10, at_start=True),),
    ),
)

SNOWBALL = ModelPresets(
    model="snowball",
    presets=_PRESETS,
    namespace="snowball-rl",
    control_pins={
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
