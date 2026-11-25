# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline and staged FTPO presets for the Mochi doom-loop campaign."""

import os
from dataclasses import replace

from marin.execution.lazy import ArtifactStep
from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLGenerationPlan, SkyRLRetentionPolicy, SkyRLRolePlan
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationProfile, FtpoPilotCallback
from skyrlrecipe.locality import RoleTopologyMode
from skyrlrecipe.sampling import EvalSamplingParams

from experiments.post_training import mochi_rl, rl
from experiments.post_training.mochi_doom_ftpo.constants import (
    REGIONAL_ROOT,
    SELECTION_NAME,
    TOKENIZER,
    TOKENIZER_REVISION,
    VERSION,
)
from experiments.post_training.rl_presets import (
    STOP_TOKEN_IDS,
    Evaluation,
    Generation,
    Layout,
    ModelPresets,
    Preset,
    Recipe,
    Training,
)

MODEL = ArtifactHfModel(
    ArtifactStep.adopt(
        "models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407",
        "2026.11.13",
        "s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
    ),
    tokenizer_uri=TOKENIZER,
    tokenizer_revision=TOKENIZER_REVISION,
    relative_path="",
)
TRAIN_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=12,
    policy_num_gpus_per_node=4,
    num_inference_engines=4,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=16,
    inference_engine_expert_parallel_size=16,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
    entrypoint="taskcompendium",
)
TRAIN_LAYOUT = Layout(
    name="doom-112",
    workload=Training(TRAIN_PLAN),
    megatron=MegatronGeometry(1, 12, 1, 4, 1),
    cluster="cw-us-east-08a",
    host_memory="900GB",
    gpu_variant="GB200",
    host_cpu=120,
    host_disk="2TB",
    priority="production",
    coordinator_timeout_hours=72,
    job_timeout_seconds=72 * 3600,
    role_topology=RoleTopologyMode.NVLINK_DOMAIN,
)
PP16_LAYOUT = replace(
    TRAIN_LAYOUT,
    name="doom-pp16-112",
    megatron=MegatronGeometry(1, 16, 1, 4, 1),
    workload=Training(replace(TRAIN_PLAN, policy_num_nodes=16, num_inference_engines=3)),
)
BASELINE_LAYOUT = replace(
    TRAIN_LAYOUT,
    name="doom-baseline-128",
    workload=Generation(
        SkyRLGenerationPlan(
            gpus_per_node=4,
            num_inference_engines=8,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=16,
            inference_engine_expert_parallel_size=16,
            n_samples_per_prompt=1,
        )
    ),
)
EVAL_LAYOUT = replace(
    BASELINE_LAYOUT,
    name="doom-eval-16",
    workload=Generation(replace(BASELINE_LAYOUT.workload.plan, num_inference_engines=1)),
)


def evaluation(
    name: str, file: str, budget: int, temperature: float = 0.0, *, every: int | None = None, at_start: bool = False
) -> Evaluation:
    return Evaluation(
        name=name,
        file=file,
        source="held_out",
        every=every,
        at_start=at_start,
        profile=EvaluationProfile(
            n_samples_per_prompt=1,
            sampling_params=EvalSamplingParams(
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                max_generate_length=budget,
                stop_token_ids=STOP_TOKEN_IDS,
                skip_special_tokens=False,
            ),
        ),
    )


def config(budget: int, lr: float, *, training: bool) -> RecipePatch:
    base = mochi_rl._ONE_TURN.merge(
        RecipePatch(
            context_budget=ContextBudget(
                request_window_tokens=budget + 2048, max_new_tokens_per_turn=budget, max_turns=1
            ),
            trainer=sections.Trainer(
                project_name="mochi-doom-ftpo",
                use_sample_packing=False,
                rollout_buffer=sections.RolloutBuffer(
                    max_staleness_steps=0, batch_policy="full_batch", max_in_flight=64
                ),
            ),
            generator=sections.Generator(
                max_num_seqs=64,
                engine_init_timeout_seconds=7200,
                sampling_params=sections.SamplingParams(
                    temperature=0.0, top_p=1.0, top_k=-1, logprobs=32 if training else 0
                ),
                trajectory_reward_shaping=sections.TrajectoryRewardShaping(
                    enabled=False,
                    loop=sections.Loop(max_period_tokens=budget // 8, tail_tokens=budget, minimum_occurrences=8),
                ),
                trajectory_retention=mochi_rl.SAMPLED_TRAJECTORIES.replace(
                    sample_fraction=0.0,
                    sample_count_per_step=512,
                    always_retain_loops=True,
                    max_bytes_per_step=512 * 1024**2,
                    max_bytes_per_run=4 * 1024**3,
                ),
                engine_init_kwargs={"logprobs_mode": "raw_logprobs"},
            ),
            extra_env={"NCCL_MAX_CTAS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
    )
    if not training:
        return base.merge(
            RecipePatch(
                trainer=sections.Trainer(
                    logger=("console", "records"),
                    policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(moe_router_replay=False)),
                    ref=sections.Ref(megatron_config=sections.RefMegatronConfig(moe_router_replay=False)),
                )
            )
        )
    transformer = {
        "bias_activation_fusion": True,
        "grug_checkpoint_cpu_offload": True,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
    return base.merge(
        RecipePatch(
            config_groups=sections.ConfigGroups(algorithm_recipe="ftpo"),
            trainer=sections.Trainer(
                update_epochs_per_batch=1,
                algorithm=sections.Algorithm(
                    policy_loss_type="ftpo",
                    advantage_estimator="uniform",
                    group_advantage_min_size=None,
                    rloo_n_filter_zero_reward_groups=False,
                    advantage_batch_normalize=False,
                    loss_reduction="token_mean",
                    think_token_weight=1.0,
                    use_kl_loss=False,
                    use_kl_in_reward=False,
                    use_entropy_loss=False,
                    off_policy_correction="none",
                    dynamic_sampling=sections.DynamicSampling(type=None),
                    ftpo=sections.Ftpo(),
                ),
                policy=sections.Policy(
                    optimizer_config=sections.PolicyOptimizerConfig(
                        optimizer="MuonH",
                        lr=lr,
                        weight_decay=0.0,
                        max_grad_norm=0.5,
                        num_warmup_steps=0,
                        optimizer_kwargs={
                            "adam_lr": lr * 0.230769,
                            "momentum": 0.95,
                            "nesterov": True,
                            "backend_steps": 5,
                            "offload_momentum": True,
                            "expert_normalization": "all_experts",
                        },
                    ),
                    megatron_config=sections.PolicyMegatronConfig(
                        moe_router_replay=True, transformer_config_kwargs=transformer
                    ),
                ),
                ref=sections.Ref(
                    megatron_config=sections.RefMegatronConfig(
                        moe_router_replay=True, transformer_config_kwargs=transformer
                    )
                ),
            ),
            generator=sections.Generator(engine_init_kwargs={"enable_return_routed_experts": True}),
        )
    )


BASELINE = Preset(
    name="doom-baseline",
    revision="2026.11.24.4",
    recipe=Recipe(
        layout=BASELINE_LAYOUT,
        fixed_reply_tokens=None,
        config=config(32768, 3e-6, training=False),
        training_plan=TRAIN_PLAN,
    ),
    updates=0,
    measured=range(1),
    checkpoints="none",
    tasks=f"{SELECTION_NAME}@{VERSION}",
    held_out_tasks=f"{SELECTION_NAME}@{VERSION}",
    model=MODEL,
    evaluations=(
        evaluation("math16k", "math16k", 16384),
        evaluation("panel32k", "panel32k", 32768),
        evaluation("sampled32k", "panel32k", 32768, 0.7),
    ),
    accepts_init=True,
    seed=20261124,
    sample_namespace="datasets/mochi-doom-ftpo",
    wandb_entity="marin-community",
    retention=SkyRLRetentionPolicy(resume_checkpoint_count=4, temporary_storage_ttl_days=3),
    temporary_output=False,
    budget="128 GB200, estimated 2-2.5 hours for baseline and sampled panel",
)
SAMPLED = replace(
    BASELINE, name="doom-baseline-sampled", evaluations=(evaluation("sampled32k", "panel32k", 32768, 0.7),)
)
RESERVE = replace(
    BASELINE,
    name="doom-reserve",
    evaluations=(
        evaluation("reserve32k", "reserve32k", 32768),
        evaluation("sampled_reserve32k", "reserve32k", 32768, 0.7),
    ),
)
MATH = replace(
    BASELINE,
    name="doom-math",
    tasks=f"{SELECTION_NAME}-math@{VERSION}",
    recipe=Recipe(layout=TRAIN_LAYOUT, fixed_reply_tokens=None, config=config(16384, 3e-6, training=True)),
    updates=12,
    measured=range(1, 13),
    checkpoints="best",
    checkpoint_every=3,
    checkpoint_selection_metric="eval/all/ftpo_selection_score",
    checkpoint_preservation_max_drop={"eval/all/fail_closed_pass_rate": 0.05},
    additional_callbacks=(FtpoPilotCallback(type="ftpo_pilot", patience_evaluations=2, max_pass_rate_drop=0.05),),
    export="best",
    evaluations=(evaluation("pilot", "validation", 16384, every=3, at_start=True),),
)
SMOKE = replace(
    MATH,
    name="doom-smoke",
    updates=2,
    measured=range(1, 3),
    checkpoint_every=1,
    checkpoints="latest",
    export="latest",
    additional_callbacks=(),
    recipe=replace(MATH.recipe, config=config(32768, 3e-6, training=True)),
    evaluations=(evaluation("pilot", "validation", 32768, every=1, at_start=True),),
)
MIXED = replace(MATH, name="doom-mixed", tasks=f"{SELECTION_NAME}@{VERSION}", tasks_input="required")
LONG = replace(
    SMOKE,
    name="doom-long",
    checkpoints="best",
    export="best",
    additional_callbacks=MATH.additional_callbacks,
    tasks=f"{SELECTION_NAME}@{VERSION}",
    tasks_input="required",
    updates=4,
    measured=range(1, 5),
    recipe=replace(SMOKE.recipe, config=config(32768, 6e-7, training=True)),
)
LONG_PP16 = replace(LONG, name="doom-long-pp16", recipe=replace(LONG.recipe, layout=PP16_LAYOUT))
LONG_16K = replace(
    LONG,
    name="doom-long-16k",
    recipe=replace(LONG.recipe, config=config(16384, 6e-7, training=True)),
    evaluations=(evaluation("pilot", "validation", 16384, every=1, at_start=True),),
)
EVAL = replace(BASELINE, name="doom-eval", recipe=replace(BASELINE.recipe, layout=EVAL_LAYOUT))
EVAL_GREEDY = replace(EVAL, name="doom-eval-greedy", evaluations=BASELINE.evaluations[:2])
MOCHI_DOOM = ModelPresets(
    model="mochi",
    namespace="mochi-doom-ftpo",
    control_pins={},
    presets=(BASELINE, SAMPLED, RESERVE, SMOKE, MATH, MIXED, LONG, LONG_PP16, LONG_16K, EVAL, EVAL_GREEDY),
    needs_completed_default_export=False,
)


if __name__ == "__main__":
    os.environ.setdefault("MARIN_PREFIX", REGIONAL_ROOT)
    os.environ.setdefault("MARIN_TEMP_PREFIX", "s3://marin-us-east-08a/tmp/ttl=3d")
    for variable in ("MARIN_PREFIX", "MARIN_TEMP_PREFIX"):
        if not os.environ[variable].startswith("s3://marin-us-east-08a/"):
            raise ValueError(f"{variable} must stay in the campaign's cw-us-east-08a region")
    raise SystemExit(rl.main(MOCHI_DOOM, forward_env=("MARIN_PREFIX", "MARIN_TEMP_PREFIX")))
