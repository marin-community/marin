# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model, layouts and RL experiment presets."""

from __future__ import annotations

import sys
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
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.evaluation.micro210 import micro210_after_rl, micro210_command
from experiments.post_training import rl
from experiments.post_training.rl_agentic_recipe import (
    NO_AGENTIC_TASKS,
    NO_IMAGE_SANDBOX,
    agentic_limits,
    agentic_preset,
    shell_tool_preset,
)
from experiments.post_training.rl_compose import BATCH_SETTINGS
from experiments.post_training.rl_presets import (
    CHECKPOINT_EVALUATION,
    RL_TRAINING_BASE,
    SAMPLED_TRAJECTORIES,
    STOP_TOKEN_IDS,
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
from experiments.post_training.task_curation.task_health import task_health_step

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

_DSPARK = sections.SpeculativeDecoding(
    method="dspark",
    model=sections.SpeculatorModel(
        source_uri="hf://atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020",
        source_identity="d22f4f21ce5cc0968dfd8e2c0d06aaf4adacbbf2",
    ),
    num_speculative_tokens=4,
)

_DEFAULT_MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        f"models/snowball-datakit-sft-{MODEL_REVISION}", "2026.09.21", MODEL_URI, kind=LevanterCheckpoint
    ),
    MODEL_ID,
    MODEL_REVISION,
    relative_path="",
)

_DEFAULT_PATCH = RL_TRAINING_BASE.merge(
    RecipePatch(
        context_budget=ContextBudget(request_window_tokens=24576, max_new_tokens_per_turn=8192, max_turns=1),
        trainer=sections.Trainer(
            flash_attn=False,
            use_sample_packing=False,
            epochs=50,
            eval_batch_size=256,
            eval_before_train=False,
            project_name="marin-snowball-rl",
            algorithm=sections.Algorithm(
                off_policy_correction="none",
                kl_estimator_type="k3_unbiased_gradient",
                eps_clip_low=0.2,
                eps_clip_high=0.2,
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
        ),
        generator=sections.Generator(
            max_num_seqs=1024,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=False,
            apply_overlong_filtering=False,
            sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
        ),
        trajectory_runner=sections.TrajectoryRunner(max_concurrent_tasks=1024),
    )
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

_REPORTING_P40 = replace(
    _ASYNC_P40,
    config=_ASYNC_P40.config.merge(RecipePatch(trainer=sections.Trainer(dump_grader_report=True))),
)

_GENERATING = Recipe(layout=E8, fixed_reply_tokens=None, config=_DEFAULT_PATCH, training_plan=_P40_PLAN)

MICRO210_RECIPE = Recipe(
    layout=replace(
        E8,
        workload=Generation(replace(E8.workload.plan, num_inference_engines=16, n_samples_per_prompt=1)),
        host_memory="512GB",
        host_disk="256GB",
        priority="production",
    ),
    fixed_reply_tokens=None,
    config=RecipePatch(generator=sections.Generator(max_num_seqs=8, gpu_memory_utilization=0.85)),
)

_GAPS = ("aaii-coverage", NO_AGENTIC_TASKS, NO_IMAGE_SANDBOX)

_TASK_SET = EVALUATIONS["eval_lite_100"]

_BATCH_KEYS = frozenset(BATCH_SETTINGS)

_OWNED_KEYS = frozenset(_MODE_KEYS)

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
    revision="2026.11.24.1",
    tasks=_DEFAULT_TASKS,
    model=_DEFAULT_MODEL,
    default_speculative_decoding=_DSPARK,
    seed=SEED,
    sample_namespace="datasets/snowball-rl",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
    owned_keys=_OWNED_KEYS,
    gaps=_GAPS,
)

_LEARNING_MEDIUM = _snowball(
    held_out_tasks=_DEFAULT_EVAL_LITE,
    checkpoints="best",
    export="best",
    name="learning-medium",
    recipe=_REPORTING_P40,
    owned_keys=_OWNED_KEYS,
    updates=25,
    measured=range(21, 26),
    checkpoint_every=26,
    accepts_init=True,
    publishes=True,
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


_TRAIN_LONG = _snowball(
    held_out_tasks=_DEFAULT_EVAL_LITE,
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=5),
    checkpoints="best",
    export="best",
    name="train-long",
    recipe=_REPORTING_P40,
    owned_keys=_OWNED_KEYS,
    updates=100,
    measured=range(1, 101),
    tasks_input="optional",
    accepts_init=True,
    publishes=True,
    checkpoint_every=10,
    data_settings=frozenset({"shuffle", "sampling"}),
    evaluations=(replace(_TASK_SET, every=10, at_start=True),),
)

_AGENTIC_PLAN = replace(
    _P40_PLAN,
    policy_num_nodes=10,
    num_inference_engines=22,
    train_batch_size=64,
    policy_mini_batch_size=64,
    n_samples_per_prompt=8,
    entrypoint="taskcompendium",
)
_AGENTIC_LAYOUT = replace(
    P40,
    name="h100x256",
    workload=Training(_AGENTIC_PLAN),
    megatron=replace(P40.megatron, context_parallel_size=5),
    cluster="cw-rno2a",
    priority="production",
    # Each job has a 12-hour allocation and resumes from its committed checkpoint.
    job_timeout_seconds=12 * 3600,
)
_AGENTIC_PATCH = RecipePatch(
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
    ),
    generator=sections.Generator(
        max_num_seqs=64,
        chat_template_kwargs={"enable_thinking": True},
        sampling_params=sections.SamplingParams(
            temperature=0.7, stop_token_ids=STOP_TOKEN_IDS, skip_special_tokens=False
        ),
        trajectory_retention=SAMPLED_TRAJECTORIES,
    ),
)

_TRAIN_LONG_AGENTIC = agentic_preset(
    replace(
        _TRAIN_LONG.with_updates(10_000),
        recipe=replace(_TRAIN_LONG.recipe, layout=_AGENTIC_LAYOUT),
        data_settings=frozenset({"shuffle"}),
        evaluations=(replace(_TASK_SET, every=2),),
        checkpoint_every=2,
        gaps=tuple(gap for gap in _TRAIN_LONG.gaps if gap != NO_AGENTIC_TASKS),
    ),
    name="train-long-agentic",
    # The sandbox covers one 90-minute trial; admission covers the trial and one retry.
    limits=agentic_limits(
        trial_timeout=5400,
        agent_timeout=1800,
        stall_timeout=10_800,
    ).merge(_AGENTIC_PATCH),
)
_TRAIN_MEDIUM_AGENTIC = replace(
    _TRAIN_LONG_AGENTIC.with_updates(10),
    name="train-medium-agentic",
    recipe=replace(
        _TRAIN_LONG_AGENTIC.recipe,
        layout=replace(
            _AGENTIC_LAYOUT,
            name="h100x88",
            workload=Training(replace(_AGENTIC_PLAN, num_inference_engines=1)),
            priority="interactive",
        ),
    ),
)

_TRAIN_MEDIUM_TOOLS = shell_tool_preset(
    replace(
        _TRAIN_LONG.with_updates(10),
        recipe=replace(
            _TRAIN_LONG.recipe,
            config=_TRAIN_LONG.recipe.config.merge(
                RecipePatch(
                    trainer=sections.Trainer(
                        policy=sections.Policy(
                            # A full-window trajectory needs the GPU memory for the policy update, so the
                            # optimizer state stays on the CPU.
                            megatron_config=sections.PolicyMegatronConfig(
                                optimizer_config_kwargs=sections.OptimizerConfigKwargs(
                                    optimizer_cpu_offload=True, optimizer_offload_fraction=1.0
                                )
                            )
                        )
                    )
                )
            ),
        ),
        data_settings=frozenset({"shuffle"}),
        checkpoints="none",
        checkpoint_every=None,
        export="none",
        budget="40 H100s plus Daytona sandboxes; wall time and cost are unmeasured",
    ),
    name="train-medium-tools",
)

_TRAIN_LONG_MICRO210 = replace(
    _TRAIN_LONG_AGENTIC,
    name="train-long-agentic-micro210",
    checkpoints="best_and_final",
    export="best_and_final",
    temporary_output=False,
    retention=replace(_TRAIN_LONG_AGENTIC.retention, checkpoints_in_run_output=True),
    recipe=replace(
        _TRAIN_LONG_AGENTIC.recipe, layout=replace(_TRAIN_LONG_AGENTIC.recipe.layout, coordinator_timeout_hours=146)
    ),
    then=lambda run: (task_health_step(run), micro210_after_rl(run, serving=MICRO210_RECIPE)),
)

_PRESETS = (
    _snowball(
        held_out_tasks=_DEFAULT_EVAL_LITE,
        name="speed-short",
        checkpoints="none",
        recipe=replace(
            _GENERATING,
            fixed_reply_tokens=4096,
            config=_GENERATING.config.merge(RecipePatch(generator=sections.Generator(gpu_memory_utilization=0.80))),
        ),
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
        checkpoints="none",
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
        checkpoints="none",
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
        budget="20m12s and 13.5 H100-h in the four-proposal comparison",
    ),
    _snowball(
        evaluations=(),
        name="speed-long",
        checkpoints="none",
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
    _TRAIN_LONG,
    _TRAIN_MEDIUM_AGENTIC,
    _TRAIN_LONG_AGENTIC,
    _TRAIN_LONG_MICRO210,
    _TRAIN_MEDIUM_TOOLS,
    replace(
        _TRAIN_LONG,
        name="train-long-checkpoint-eval",
        checkpoint_evaluation=CHECKPOINT_EVALUATION,
        gaps=(*_TRAIN_LONG.gaps, "agentic-eval-on-daytona"),
    ),
    _snowball(
        held_out_tasks=_DEFAULT_EVAL_LITE,
        name="eval-short",
        checkpoints="none",
        recipe=_GENERATING,
        updates=0,
        measured=range(1),
        accepts_init=True,
        owned_keys=_OWNED_KEYS | _SPEED_KEYS | _BATCH_KEYS,
        evaluations=(EVALUATIONS["eval_lite"],),
        budget="8 min; 1.1 H100-h",
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
    if sys.argv[1:2] == ["micro210"]:
        micro210_command(MICRO210_RECIPE, entry_module="experiments.post_training.snowball_rl").main(args=sys.argv[2:])

    def sft_presets():
        # Load the SFT training dependencies only for --sft.
        from experiments.post_training.snowball_sft import SNOWBALL_SFT  # noqa: PLC0415

        return SNOWBALL_SFT

    raise SystemExit(rl.main(SNOWBALL, sft_presets=sft_presets))
