# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Megatron RL on the curriculum pool with a rolling rollout buffer.

The launcher trains the 67B-A2B Snowball policy using the curriculum-RL pool and evaluation.
It writes every setting it decides into the rendered
config, including values that MarinSkyRL's base config or the curriculum template already hold.
Presets bundle the loop settings, and ``--set`` changes one key; a run's address carries a hash of
its ``--set`` changes.

Plan or run::

    python -m experiments.post_training.async_rl --version 2026.09.18 --preset smoke
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default --run
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default \\
        --set trainer.rollout_buffer.max_staleness_steps=2 --run

The default preset runs on 40 GPUs: 128 prompts per update with four answers each,
192 concurrent prompt groups, staleness 4, and an 8192-token request window with a
4096-token response cap. To grow the batch, add prompts; more answers per prompt changes the group
each advantage is computed over. Rollouts use the tokenizer's built-in chat template and preserve
sampled completion token IDs for the behavior-policy loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import lcm
from types import MappingProxyType

import click
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLHardware,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    skyrl_step,
)
from marin.skyrl_recipe import (
    Algorithm,
    ChatTemplate,
    ContextBudget,
    Data,
    Environment,
    ExpertBlockSync,
    Generator,
    Placement,
    Policy,
    PolicyMegatronConfig,
    PolicyOptimizerConfig,
    RecipePatch,
    Ref,
    RefMegatronConfig,
    RLEntrypoint,
    RolloutBuffer,
    SamplingParams,
    SkyRLRecipe,
    Trainer,
)
from rigging.provenance import username_segment

from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.curriculum_rl.launch import (
    GPU_VARIANT,
    GPUS_PER_NODE,
    MODEL_ARTIFACT_NAME,
    POOL_ARTIFACT_NAME,
    SEED,
    SNOWBALL_POLICY,
    SNOWBALL_SMOKE,
    PolicySpec,
    evaluation_model_config,
    model_step,
)
from experiments.post_training.curriculum_rl.pool import (
    MAX_PROMPT_TOKENS,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)
from experiments.post_training.skyrl_evaluation import skyrl_eval_step

EXPERIMENT_NAME = "async-rl"
WANDB_PROJECT = f"marin-{EXPERIMENT_NAME}"
# Prompts per optimizer update. MarinSkyRL's train_batch_size counts prompts.
PROMPTS_PER_UPDATE = 128
# Answers sampled per prompt, so one update trains on 512 sequences.
ANSWERS_PER_PROMPT = 4
# Keep two resumable checkpoints in the temporary bucket, which deletes objects after 14 days. The
# terminal export does not expire.
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2)


SNOWBALL_RESOURCES = RecipePatch(
    trainer=Trainer(
        placement=Placement(
            colocate_all=False,
            colocate_policy_ref=True,
            policy_num_nodes=4,
            policy_num_gpus_per_node=GPUS_PER_NODE,
            ref_num_nodes=4,
            ref_num_gpus_per_node=GPUS_PER_NODE,
        ),
        train_batch_size=PROMPTS_PER_UPDATE,
        policy_mini_batch_size=PROMPTS_PER_UPDATE,
        micro_train_batch_size_per_gpu=1,
    ),
    generator=Generator(
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        n_samples_per_prompt=ANSWERS_PER_PROMPT,
    ),
)


SNOWBALL_RECIPE = RecipePatch.combine(
    resources=SNOWBALL_RESOURCES,
    policy=RecipePatch(
        entrypoint=RLEntrypoint.STANDARD,
        trainer=Trainer(
            strategy="megatron",
            resume_mode="latest",
            policy=Policy(
                optimizer_config=PolicyOptimizerConfig(
                    optimizer="AdamW", lr=1e-06, weight_decay=0.01, max_grad_norm=1.0
                ),
                megatron_config=PolicyMegatronConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_model_parallel_size=8,
                    expert_tensor_parallel_size=1,
                ),
            ),
            ref=Ref(
                megatron_config=RefMegatronConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    context_parallel_size=1,
                    expert_model_parallel_size=8,
                    expert_tensor_parallel_size=1,
                )
            ),
        ),
        generator=Generator(backend="vllm", run_engines_locally=True, engine_init_kwargs={"moe_backend": "triton"}),
        data=Data(kind="parquet", train_data=(), val_data=()),
    ),
)
HOST_MEMORY = "1800GB"


@dataclass(frozen=True)
class AsyncPreset:
    """Async-loop settings.

    ``default`` is the 40-GPU configuration, ``smoke`` runs two short updates to check the wiring, and
    ``on_policy`` admits only groups that the current weights sampled.
    """

    label: str
    # Optimizer updates the run performs before it stops; the epoch bound never fires first.
    max_steps: int
    # Evaluate every this many updates, after the weight sync so the scored weights are the trained
    # ones; -1 turns evaluation off, including the pass at the end of training.
    eval_interval: int
    # How many updates old a group may be when the trainer consumes it; 0 admits only groups
    # sampled by the current weights.
    max_staleness_steps: int
    # Maximum prompt groups admitted to the rollout buffer at once.
    max_in_flight_groups: int
    # Prompt-plus-response budget one request may occupy in the engine, in tokens.
    request_window_tokens: int
    # Longest response the policy may generate, in tokens; it also caps the in-run evaluation.
    max_new_tokens: int
    # Evaluation suites the ``evaluation`` stage scores the terminal export on, comma separated.
    evals: str
    # Export the telemetry the async RL dashboard reads.
    telemetry: bool = True


# An evaluation pauses generation for 256 prompts, about 13 minutes against a 55 to 97 second
# update, so the default evaluates every 10 updates.
DEFAULT = AsyncPreset(
    label="default",
    max_steps=100,
    eval_interval=10,
    max_staleness_steps=4,
    max_in_flight_groups=192,
    request_window_tokens=8192,
    max_new_tokens=4096,
    evals="math500,gsm8k-0shot",
)
# Two updates with short responses on the default geometry and batch, to check the wiring.
SMOKE_PRESET = replace(
    DEFAULT,
    label="smoke",
    max_steps=2,
    eval_interval=-1,
    request_window_tokens=2048,
    max_new_tokens=1024,
    evals="gsm8k-smoke",
)
# Every consumed group was sampled by the current weights.
ON_POLICY = replace(DEFAULT, label="on_policy", max_staleness_steps=0)
PRESETS: Mapping[str, AsyncPreset] = MappingProxyType(
    {preset.label: preset for preset in (SMOKE_PRESET, DEFAULT, ON_POLICY)}
)


def checkpoint_interval(max_steps: int, eval_interval: int) -> int:
    """A resumable checkpoint at every evaluation, or at the end when there is no evaluation."""
    return max_steps if eval_interval <= 0 else eval_interval


CURRICULUM_TEMPLATE = SNOWBALL_SMOKE


def training_recipe(preset: AsyncPreset, settings: tuple[str, ...] = ()) -> SkyRLRecipe:
    """Combine the loop preset, Snowball policy and checkpoint schedule."""
    recipe = SkyRLRecipe.combine(
        preset=RecipePatch(
            context_budget=ContextBudget(
                request_window_tokens=preset.request_window_tokens,
                max_new_tokens_per_turn=preset.max_new_tokens,
                max_turns=1,
            ),
            trainer=Trainer(
                flash_attn=False,
                use_sample_packing=False,
                gradient_checkpointing=True,
                offload_optimizer_during_rollouts=False,
                epochs=50,
                max_steps=preset.max_steps,
                update_epochs_per_batch=1,
                eval_batch_size=256,
                eval_interval=preset.eval_interval,
                logger="wandb",
                project_name=WANDB_PROJECT,
                tracker_commit_each_step=True,
                training_metrics=preset.telemetry,
                rollout_spans=preset.telemetry,
                policy_train_spans=preset.telemetry,
                algorithm=Algorithm(
                    advantage_estimator="grpo",
                    policy_loss_type="behavior_clip",
                    use_kl_loss=False,
                    use_kl_in_reward=False,
                    off_policy_correction="none",
                    eps_clip_low=0.2,
                    eps_clip_high=0.2,
                ),
                rollout_buffer=RolloutBuffer(
                    max_staleness_steps=preset.max_staleness_steps,
                    max_in_flight=preset.max_in_flight_groups,
                    batch_policy="rolling",
                ),
            ),
            generator=Generator(
                model_dtype="bfloat16",
                vllm_attention_backend="FLASH_ATTN",
                weight_sync_backend="nccl",
                weight_sync_transport="expert_block",
                expert_block_sync=ExpertBlockSync(timeout_seconds=600, verify=False),
                gpu_memory_utilization=0.75,
                max_num_seqs=1024,
                max_num_batched_tokens=8192,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                enforce_eager=False,
                enable_http_endpoint=True,
                use_conversation_multi_turn=True,
                chat_template=ChatTemplate(source="name", name_or_path=None),
                sampling_params=SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
            ),
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            environment=Environment(env_class="gsm8k"),
        ),
        policy=SNOWBALL_RECIPE,
    ).with_settings(settings)
    trainer = recipe.trainer
    interval = checkpoint_interval(trainer.max_steps, trainer.eval_interval)
    computed = RecipePatch(
        trainer=Trainer(
            eval_before_train=trainer.eval_interval > 0,
            ckpt_interval=interval,
            hf_save_interval=lcm(trainer.max_steps, interval),
        )
    )
    recipe = SkyRLRecipe.combine(
        recipe=recipe,
        resources=SNOWBALL_RESOURCES,
        launch_policy=RecipePatch(
            entrypoint=RLEntrypoint.STANDARD,
            trainer=Trainer(resume_mode="latest"),
            generator=Generator(run_engines_locally=True),
            data=Data(kind="parquet", train_data=(), val_data=()),
        ),
        checkpoint_schedule=computed,
    )
    budget = recipe.context_budget
    if MAX_PROMPT_TOKENS > budget.request_window_tokens - budget.max_new_tokens_per_turn:
        raise click.BadParameter("pool prompts do not fit the request window beside the response cap")
    return recipe


@dataclass(frozen=True)
class AsyncRun:
    rl: ArtifactStep[SkyRLRun]
    evaluation: ArtifactStep[EvaluationResult]


def build_run(policy: PolicySpec, preset: AsyncPreset, version: str | None, settings: tuple[str, ...] = ()) -> AsyncRun:
    """Assemble the RL step and its evaluation for one policy and preset."""
    config = training_recipe(preset, settings)
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    model = policy.adopted_model or model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    # A --set run gets its own address: the name carries a hash of its settings.
    changes = "\n".join(settings)
    suffix = f"-set-{fingerprint_hash(changes)}" if settings else ""
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy.label}-{preset.label}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            recipe=config,
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            hardware=SkyRLHardware(gpus_per_node=GPUS_PER_NODE, gpu_variant=GPU_VARIANT),
            retention=RETENTION,
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=f"lib/iris/config/{policy.cluster}.yaml",
            cpu=16,
            memory=HOST_MEMORY,
            disk="2TB",
            priority="interactive",
            # One automatic retry, then fail; a healthy run resumes from its latest checkpoint on resubmission.
            max_retries=1,
            target_cluster=policy.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=72,
            # The W&B key decides the entity; a hard-coded one fails at runtime for a key that
            # cannot write there.
            wandb_entity=None,
        ),
        export_hf=True,
    )
    # The evaluation serves the rendered window, so a --set on the budget reaches the server.
    budget = config.context_budget
    # The eval artifact is keyed on the model name; the owner keeps two users at one version apart.
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy.label}-{preset.label}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{preset.evals}"
    evaluation_version = version or resolve_version(evaluation_base_name, None)
    evaluation_model = evaluation_model_config(policy, CURRICULUM_TEMPLATE, evaluation_model_name)
    evaluation_model = replace(
        evaluation_model,
        serve=replace(evaluation_model.serve, max_model_len=budget.request_window_tokens),
        generation=replace(evaluation_model.generation, max_gen_toks=budget.max_new_tokens_per_turn),
    )
    evaluation = skyrl_eval_step(
        rl,
        evaluation_model,
        preset.evals,
        version=evaluation_version,
        accelerator=f"{GPU_VARIANT}x{policy.serve_gpus}",
        submission_cluster=policy.cluster,
        federated_cluster=policy.cluster,
    )
    return AsyncRun(rl=rl, evaluation=evaluation)


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default=SMOKE_PRESET.label, show_default=True)
@click.option(
    "--target-cluster",
    type=click.Choice(("cw-us-east-02a", "cw-rno2a")),
    default=None,
    help="H100 cluster that runs the job; defaults to the policy's cluster.",
)
@click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Change one setting of the rendered RL config (typed dotted key).",
)
@click.option(
    "--stage",
    type=click.Choice(tuple(field.name for field in fields(AsyncRun))),
    default="rl",
    show_default=True,
    help="Terminal stage; evaluation includes the RL run automatically.",
)
@rl_build_options
def main(preset: str, target_cluster: str | None, settings: tuple[str, ...], stage: str) -> dict[str, ArtifactStep]:
    policy = replace(SNOWBALL_POLICY, cluster=target_cluster) if target_cluster is not None else SNOWBALL_POLICY
    run = build_run(policy, PRESETS[preset], version=None, settings=settings)
    return {f"{SNOWBALL_POLICY.label}-{preset}": getattr(run, stage)}


if __name__ == "__main__":
    main()
