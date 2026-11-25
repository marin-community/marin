# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare task samplers with short Qwen3-0.6B updates and ShellSim graders."""

from dataclasses import replace

from marin.rl.skyrl import ArtifactHfModel, MegatronGeometry, SkyRLRetentionPolicy
from skyrlrecipe import ContextBudget, RecipePatch, sections
from skyrlrecipe.declared import EvaluationProfile
from skyrlrecipe.sampling import EvalSamplingParams
from taskcompendium.models import CommandSemantics

from experiments.models import qwen3_0_6b
from experiments.post_training import rl
from experiments.post_training.cat_count_canary.launcher import MODELS, role_plan
from experiments.post_training.rl_agentic_recipe import TASK_MACHINES
from experiments.post_training.rl_presets import Evaluation, Layout, ModelPresets, Preset, Recipe, Training
from experiments.post_training.task_curation import qwen_sampler_lab

TASKS = f"{qwen_sampler_lab.NAME}@{qwen_sampler_lab.VERSION}"
QWEN = ArtifactHfModel(qwen3_0_6b, relative_path="")
CHAT = MODELS["qwen3-0.6b"]
LAYOUT = Layout(
    name="qwen-h100x2",
    workload=Training(replace(role_plan(batch_size=16, group_size=4, micro_train_batch_size=4), colocate_all=True)),
    megatron=MegatronGeometry(1, 1, 1, 1, 1),
    cluster="cw-us-east-02a",
    host_memory="128GB",
    gpu_variant="H100",
    host_cpu=20,
    host_disk="200GB",
    priority="interactive",
    coordinator_timeout_hours=2,
    job_timeout_seconds=1800,
)
EVAL = Evaluation(
    "qwen_lab",
    "validation",
    EvaluationProfile(
        n_samples_per_prompt=1,
        sampling_params=EvalSamplingParams(temperature=0.0, max_generate_length=64),
    ),
    source="held_out",
    at_start=True,
)


def sampler_preset(name: str, updates: int, kind: str, epsilon: float) -> Preset:
    verifier = TASK_MACHINES[CommandSemantics.SHELL_SIMULATOR]
    config = RecipePatch(
        environment=sections.Environment(
            task_sessions={"task_spec": {"package_verifier": verifier.model_dump(mode="json")}}
        ),
        context_budget=ContextBudget(request_window_tokens=384, max_new_tokens_per_turn=64, max_turns=1),
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
            chat_template=sections.ChatTemplate(source="name", name_or_path=CHAT.chat_template),
            chat_template_kwargs=CHAT.chat_template_kwargs,
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
        revision="2026.11.25.1",
        tasks=TASKS,
        model=QWEN,
        recipe=Recipe(LAYOUT, fixed_reply_tokens=None, config=config),
        updates=updates,
        measured=range(1, updates + 1),
        seed=17,
        evaluations=(replace(EVAL, every=updates),),
        held_out_tasks=TASKS,
        checkpoints="none",
        sample_namespace="datasets/qwen-sampler",
        wandb_entity="marin-community",
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
        budget="2 H100s; wall time unmeasured",
    )


QWEN_SAMPLER = ModelPresets(
    model="qwen3-0.6b",
    namespace="qwen-sampler",
    presets=tuple(
        sampler_preset(f"{stage}-{arm}", updates, kind, epsilon)
        for stage, updates in (("smoke", 4), ("pilot", 32))
        for arm, kind, epsilon in (("fixed", "adaptive-source", 1.0), ("thompson", "adaptive-source-thompson", 0.05))
    ),
    control_pins={},
)

if __name__ == "__main__":
    raise SystemExit(rl.main(QWEN_SAMPLER))
