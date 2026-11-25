# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare task samplers with short Qwen3-0.6B updates and ShellSim graders."""

from dataclasses import replace

from iris.rpc import job_pb2
from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLRetentionPolicy
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.sampling import EvalSamplingParams
from taskcompendium.models import CommandSemantics

from experiments.models import qwen3_0_6b
from experiments.post_training import rl
from experiments.post_training.cat_count_canary.launcher import role_plan
from experiments.post_training.rl_agentic_recipe import TASK_MACHINES
from experiments.post_training.rl_presets import Evaluation, Layout, ModelPresets, Preset, Recipe, Training
from experiments.post_training.task_curation import qwen_sampler_lab

TASKS = f"{qwen_sampler_lab.NAME}@{qwen_sampler_lab.VERSION}"
QWEN = ArtifactHfModel(qwen3_0_6b, relative_path="")
RESPONSE_TOKENS = 4096
PROMPT_EXPLORATION = 0.20
LOW_EXPLORATION = 0.05
ROLE_PLAN = replace(role_plan(batch_size=16, group_size=4, micro_train_batch_size=4), colocate_all=True)
LAYOUT = Layout(
    name="qwen-h100x2",
    workload=Training(ROLE_PLAN),
    megatron=MegatronGeometry(1, 1, 1, 1, 1),
    cluster="cw-us-east-02a",
    host_memory="64GB",
    gpu_variant="H100",
    host_cpu=16,
    host_disk="200GB",
    priority="interactive",
    coordinator_timeout_hours=2,
    job_timeout_seconds=1800,
    container_profile=job_pb2.CONTAINER_PROFILE_DEFAULT,
)
RNO_PAIR_LAYOUT = replace(LAYOUT, name="qwen-h100x2-rno", cluster="cw-rno2a")
RNO_LAYOUT = replace(
    RNO_PAIR_LAYOUT,
    name="qwen-h100x8-rno",
    workload=Training(replace(ROLE_PLAN, policy_num_gpus_per_node=8, num_inference_engines=8)),
    host_cpu=80,
    host_memory="512GB",
    container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
)
EVAL = Evaluation(
    "qwen_lab",
    "validation",
    EvaluationProfile(
        n_samples_per_prompt=1,
        sampling_params=EvalSamplingParams(temperature=0.0, max_generate_length=RESPONSE_TOKENS),
    ),
    source="held_out",
    at_start=True,
)


def sampler_preset(
    name: str,
    updates: int,
    kind: str,
    epsilon: float,
    seed: int,
    *,
    layout: Layout,
    revision: str,
    evaluation_interval: int | None,
    additional_evaluations: tuple[Evaluation, ...] = (),
) -> Preset:
    verifier = TASK_MACHINES[CommandSemantics.SHELL_SIMULATOR]
    config = RecipePatch(
        environment=sections.Environment(
            task_sessions={"task_spec": {"package_verifier": verifier.model_dump(mode="json")}}
        ),
        context_budget=ContextBudget(
            request_window_tokens=512 + RESPONSE_TOKENS, max_new_tokens_per_turn=RESPONSE_TOKENS, max_turns=1
        ),
        data=sections.Data(
            kind="parquet", sampling=sections.Sampling(kind=kind, source_column="family", epsilon=epsilon)
        ),
        trainer=sections.Trainer(
            strategy="megatron",
            flash_attn=False,
            use_sample_packing=False,
            epochs=50,
            update_epochs_per_batch=1,
            eval_batch_size=32,
            training_metrics=True,
            rollout_spans=True,
            project_name="marin-qwen-sampler",
            logger=("console", "records", "wandb"),
            algorithm=sections.Algorithm(advantage_estimator="grpo", use_kl_loss=False),
            rollout_buffer=sections.RolloutBuffer(max_staleness_steps=0, max_in_flight=16),
            policy=sections.Policy(optimizer_config=sections.PolicyOptimizerConfig(lr=2e-6, max_grad_norm=1.0)),
        ),
        generator=sections.Generator(
            backend="vllm",
            weight_sync_backend="nccl",
            model_dtype="bfloat16",
            run_engines_locally=True,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
            chat_template_kwargs={"enable_thinking": True},
            sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
            trajectory_retention=sections.TrajectoryRetention(enabled=False),
        ),
        trajectory_runner=sections.TrajectoryRunner(
            max_concurrent_tasks=64,
            rollout_workers=sections.RolloutWorkers(num_workers=2, cpus_per_worker=4),
        ),
    )
    return Preset(
        name=name,
        revision=revision,
        tasks=TASKS,
        model=QWEN,
        recipe=Recipe(
            layout,
            fixed_reply_tokens=None,
            config=config,
        ),
        updates=updates,
        measured=range(1, updates + 1),
        seed=seed,
        evaluations=(replace(EVAL, every=evaluation_interval), *additional_evaluations),
        held_out_tasks=TASKS,
        checkpoints="none",
        sample_namespace="datasets/qwen-sampler",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
        budget=f"{layout.workload.plan.gpus_per_node} H100s; wall time unmeasured",
    )


PROMPT_ARMS = (
    ("source-only", "prompt-thompson-source"),
    ("contextual", "prompt-thompson-contextual"),
    ("cost-aware", "prompt-thompson-cost-aware"),
)
ARMS = (
    ("fixed", "adaptive-source", 1.0),
    ("thompson", "adaptive-source-thompson", LOW_EXPLORATION),
    *((arm, kind, PROMPT_EXPLORATION) for arm, kind in PROMPT_ARMS),
)
RNO_PAIR_PILOT_ARMS = (
    *ARMS,
    *((f"{arm}-eps05", kind, LOW_EXPLORATION) for arm, kind in PROMPT_ARMS),
)
FULL_VALIDATION = replace(EVAL, name="qwen_lab_full", file="full_validation", at_start=False)
CONFIRMATION_ARMS = ("fixed", "thompson", "cost-aware-eps05")
WITHIN_SOURCE = "prompt-thompson-within-source-cost-aware"
SOURCE_ARMS = (
    ("within-source", WITHIN_SOURCE, LOW_EXPLORATION),
    ("within-source-random", WITHIN_SOURCE, 1.0),
)
SOURCE_EVAL_TASKS = f"{qwen_sampler_lab.SOURCE_EVAL_NAME}@{qwen_sampler_lab.SOURCE_EVAL_VERSION}"
SOURCE_EVAL = replace(EVAL, name="qwen_source_eval")


def with_rollout_token_budget(preset: Preset, tokens: int) -> Preset:
    config = preset.recipe.config.merge(RecipePatch(trainer=sections.Trainer(rollout_token_budget=tokens)))
    return replace(preset, recipe=replace(preset.recipe, config=config))


def token_budget_smoke(arm: str, kind: str, epsilon: float) -> Preset:
    preset = sampler_preset(
        f"rno-pair-budget-smoke-{arm}",
        16,
        kind,
        epsilon,
        41,
        layout=RNO_PAIR_LAYOUT,
        revision="2026.11.25.8",
        evaluation_interval=None,
    )
    return with_rollout_token_budget(replace(preset, evaluations=(replace(EVAL, at_start=False),)), 200_000)


def budget_preset(
    name: str,
    kind: str,
    epsilon: float,
    seed: int,
    *,
    revision: str,
    max_updates: int,
    tokens: int,
    evaluation: Evaluation,
    held_out_tasks: str,
) -> Preset:
    """Evaluate at initialization and the first complete update that reaches the token budget."""
    preset = sampler_preset(
        name,
        max_updates,
        kind,
        epsilon,
        seed,
        layout=replace(RNO_PAIR_LAYOUT, coordinator_timeout_hours=3, job_timeout_seconds=7200),
        revision=revision,
        evaluation_interval=None,
    )
    return with_rollout_token_budget(replace(preset, evaluations=(evaluation,), held_out_tasks=held_out_tasks), tokens)


def layout_presets(
    prefix: str,
    layout: Layout,
    *,
    smoke_revision: str,
    pilot_revision: str,
    pilot_arms: tuple[tuple[str, str, float], ...] = ARMS,
    confirmation: tuple[Evaluation, ...] = (),
) -> tuple[Preset, ...]:
    return tuple(
        sampler_preset(
            f"{prefix}{stage}-{arm}{suffix}",
            updates,
            kind,
            epsilon,
            seed,
            layout=replace(layout, job_timeout_seconds=2700) if stage == "pilot" else layout,
            revision=revision,
            evaluation_interval=8 if stage == "pilot" else updates,
            additional_evaluations=additional,
        )
        for stage, updates, seed, suffix, revision, arms, additional in (
            ("smoke", 4, 17, "", smoke_revision, ARMS, ()),
            ("pilot", 32, 17, "", pilot_revision, pilot_arms, ()),
            ("pilot", 32, 29, "-seed29", pilot_revision, pilot_arms, confirmation),
        )
        for arm, kind, epsilon in arms
    )


QWEN_SAMPLER = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-sampler",
    presets=(
        *layout_presets("", LAYOUT, smoke_revision="2026.11.25.6", pilot_revision="2026.11.25.6"),
        *layout_presets(
            "rno-pair-",
            RNO_PAIR_LAYOUT,
            smoke_revision="2026.11.25.6",
            pilot_revision="2026.11.25.6",
            pilot_arms=RNO_PAIR_PILOT_ARMS,
            confirmation=(FULL_VALIDATION,),
        ),
        *layout_presets("rno-", RNO_LAYOUT, smoke_revision="2026.11.25.3", pilot_revision="2026.11.25.3"),
        *(
            budget_preset(
                f"rno-pair-budget-confirm-{arm}-seed{seed}",
                kind,
                epsilon,
                seed,
                revision="2026.11.25.9",
                max_updates=160,
                tokens=3_000_000,
                evaluation=replace(FULL_VALIDATION, at_start=True),
                held_out_tasks=TASKS,
            )
            for arm, kind, epsilon in RNO_PAIR_PILOT_ARMS
            if arm in CONFIRMATION_ARMS
            for seed in (41, 53)
        ),
        *(
            sampler_preset(
                f"rno-pair-source-smoke-{arm}",
                4,
                kind,
                epsilon,
                67,
                layout=RNO_PAIR_LAYOUT,
                revision="2026.11.25.10",
                evaluation_interval=4,
            )
            for arm, kind, epsilon in SOURCE_ARMS
        ),
        *(
            budget_preset(
                f"rno-pair-source-confirm-{arm}-seed{seed}",
                kind,
                epsilon,
                seed,
                revision="2026.11.25.11",
                max_updates=220,
                tokens=5_242_880,
                evaluation=SOURCE_EVAL,
                held_out_tasks=SOURCE_EVAL_TASKS,
            )
            for arm, kind, epsilon in (*RNO_PAIR_PILOT_ARMS, *SOURCE_ARMS)
            if arm in {"thompson", "cost-aware-eps05", "within-source", "within-source-random"}
            for seed in (67, 79)
        ),
        *(
            token_budget_smoke(arm, kind, epsilon)
            for arm, kind, epsilon in RNO_PAIR_PILOT_ARMS
            if arm in {"fixed", "cost-aware-eps05"}
        ),
        # Evaluate every update to compare controls at the contextual arms' observed token budgets.
        *(
            sampler_preset(
                f"rno-pair-calibrate-{arm}",
                26,
                kind,
                epsilon,
                17,
                layout=replace(RNO_PAIR_LAYOUT, job_timeout_seconds=3600),
                revision="2026.11.25.7",
                evaluation_interval=1,
            )
            for arm, kind, epsilon in RNO_PAIR_PILOT_ARMS
            if arm in {"fixed", "thompson", "source-only", "source-only-eps05"}
        ),
    ),
    control_pins={},
)


if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_SAMPLER))
