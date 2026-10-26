# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch asynchronous RL through the unified rollout buffer on the curriculum pool.

The launcher trains the 67B-A2B Snowball policy with MarinSkyRL's rollout buffer, using the
curriculum-RL pool, policy and evaluation. It writes every setting it decides into the rendered
config, including values that MarinSkyRL's base config or the curriculum template already hold.
Presets bundle the loop settings, and ``--set`` changes one key; a run's address carries a hash of
its ``--set`` changes.

Plan or run::

    python -m experiments.post_training.async_rl --version 2026.09.18 --preset smoke
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default --run
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default \\
        --set trainer.rollout_buffer.max_staleness_steps=2 --run

The default preset runs on 40 GPUs: 128 prompts per update with four answers each, 192 generation
groups generating at once, staleness 4, and an 8192-token request window with a
4096-token response cap. To grow the batch, add prompts; more answers per prompt changes the group
each advantage is computed over. The selected MarinSkyRL revision validates the rollout and
telemetry settings before the job takes a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from types import MappingProxyType

import click
import yaml
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
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _role_plan_config_values,
    skyrl_step,
)
from rigging.provenance import username_segment
from skyrlrecipe import ContextBudget, RecipePatch, SkyRLRecipe
from skyrlrecipe.sections import (
    Algorithm,
    ExpertBlockSync,
    Generator,
    Placement,
    Policy,
    PolicyOptimizerConfig,
    Ref,
    RolloutBuffer,
    SamplingParams,
    Trainer,
    WeightSyncPause,
)

from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.curriculum_rl.launch import (
    ARMS,
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
    rl_config_yaml,
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
# Freeze requests at weight sync and re-prefill their prefix under the new weights.
PAUSE_MODE = "keep"
# Keep two resumable checkpoints in the temporary bucket, which deletes objects after 14 days. The
# terminal export does not expire.
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2)


@dataclass(frozen=True)
class TrainingRecipe:
    """Topology, parallel geometry, and optimizer settings for one policy."""

    # Placement and batch shape; Marin writes these into the SkyRL config from the role plan.
    role_plan: SkyRLRolePlan
    # AdamW learning rate; MarinSkyRL's Snowball Megatron configs use 1e-6.
    learning_rate: float
    # AdamW decoupled weight decay; MarinSkyRL's base config also uses 1e-2.
    weight_decay: float
    # Gradient-norm clip applied before each optimizer step.
    max_grad_norm: float
    # Megatron parallelism for the policy and the reference model.
    megatron: MegatronGeometry
    # Host memory includes Megatron checkpoint staging buffers.
    host_memory: str
    # vLLM engine settings the model needs beyond the ones the launcher writes itself.
    engine_init_kwargs: Mapping[str, object]


# Snowball 67B-A2B on the 40-GPU topology: four policy nodes with the reference colocated,
# pipeline depth 2 with 16-way data parallelism and experts sharded eight ways, and one engine node
# sharding experts across its eight ranks (DP8/EP8).
SNOWBALL_RECIPE = TrainingRecipe(
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=PROMPTS_PER_UPDATE,
        policy_mini_batch_size=PROMPTS_PER_UPDATE,
        # One sequence per GPU per micro-step: 32 micro-steps over 16 data-parallel ranks.
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=ANSWERS_PER_PROMPT,
    ),
    learning_rate=1.0e-6,
    weight_decay=1e-2,
    max_grad_norm=1.0,
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
    ),
    host_memory="1800GB",
    # The Triton MoE kernels; the fused defaults do not cover this expert layout.
    engine_init_kwargs=MappingProxyType({"moe_backend": "triton"}),
)


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
    # Maximum prompt groups generating at once. Smaller caps fill a batch in several waves.
    generation_workers: int
    # Prompt-plus-response budget one request may occupy in the engine, in tokens.
    request_window_tokens: int
    # Longest response the policy may generate, in tokens; it also caps the in-run evaluation.
    max_new_tokens: int
    # Evaluation suites the ``evaluation`` stage scores the terminal export on, comma separated.
    evals: str
    # Drop the engines' KV cache at the pause so nothing computed by the old weights is reused.
    clear_kv_cache_on_weight_sync: bool = True
    # Preserve the former fully_async loop's commit-order batching.
    batch_policy: str = "rolling"
    # Export the telemetry the async RL dashboard reads.
    telemetry: bool = True


# An evaluation pauses generation for 256 prompts, about 13 minutes against a 55 to 97 second
# update, so the default evaluates every 10 updates.
DEFAULT = AsyncPreset(
    label="default",
    max_steps=100,
    eval_interval=10,
    max_staleness_steps=4,
    generation_workers=192,
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
# Every consumed group was sampled by the current weights; workers sit at the trainer's floor of
# one update's prompts.
ON_POLICY = replace(DEFAULT, label="on_policy", max_staleness_steps=0, generation_workers=PROMPTS_PER_UPDATE)
PRESETS: Mapping[str, AsyncPreset] = MappingProxyType(
    {preset.label: preset for preset in (SMOKE_PRESET, DEFAULT, ON_POLICY)}
)


def checkpoint_interval(max_steps: int, eval_interval: int) -> int:
    """A resumable checkpoint at every evaluation, or at the end when there is no evaluation."""
    return max_steps if eval_interval <= 0 else eval_interval


# The curriculum scale point the template renders through. The launcher keeps only the template's
# data and environment sections, which every scale point renders the same way, and passes the
# rendered context budget to ``evaluation_model_config``.
CURRICULUM_TEMPLATE = SNOWBALL_SMOKE


ROLE_PLAN_SETTINGS = frozenset(_role_plan_config_values(SNOWBALL_RECIPE.role_plan))
EVAL_DERIVED_SETTINGS = frozenset({"trainer.eval_before_train", "trainer.ckpt_interval", "trainer.hf_save_interval"})


def training_config(preset: AsyncPreset, settings: tuple[str, ...] = ()) -> dict:
    """Render the RL config: the curriculum data wiring plus every setting this launcher decides."""
    recipe = SNOWBALL_RECIPE
    plan = recipe.role_plan
    # The curriculum template supplies data and environment; the typed sections define the training settings.
    config = yaml.safe_load(rl_config_yaml(CURRICULUM_TEMPLATE, ARMS["naive"], SNOWBALL_POLICY))
    config["entrypoint"] = "standard"
    # The public context declaration supplies the prompt, generation and engine lengths.
    config["context_budget"] = ContextBudget(
        request_window_tokens=preset.request_window_tokens,
        max_new_tokens_per_turn=preset.max_new_tokens,
        # Single-turn math ends after one answer; each of the prompt's four answers is its own rollout.
        max_turns=1,
    ).to_skyrl()
    config["trainer"] = Trainer(
        strategy="megatron",
        # Keep Transformer Engine's attention backend; true forces the flash-attn package
        # (NVTE_FUSED_ATTN=0).
        flash_attn=False,
        # One sequence per row; packing changes the micro-step shape.
        use_sample_packing=False,
        # Recompute activations in the backward pass; the 67B-A2B forward does not fit otherwise.
        gradient_checkpointing=True,
        # Keep the optimizer state resident; the engines run on their own node.
        offload_optimizer_during_rollouts=False,
        # Passes over the pool the dataloader may make; 100 updates of 128 prompts need several.
        epochs=50,
        max_steps=preset.max_steps,
        # One optimizer pass over each batch; the mini batch equals the batch, so one update per batch.
        update_epochs_per_batch=1,
        train_batch_size=plan.train_batch_size,
        policy_mini_batch_size=plan.policy_mini_batch_size,
        micro_train_batch_size_per_gpu=plan.micro_train_batch_size_per_gpu,
        micro_forward_batch_size_per_gpu=1,
        # Validation prompts scored per in-run evaluation.
        eval_batch_size=256,
        eval_interval=preset.eval_interval,
        resume_mode="latest",
        logger="wandb",
        project_name=WANDB_PROJECT,
        # Commit each step's metrics as they are logged, so a killed run keeps its curve.
        tracker_commit_each_step=True,
        training_metrics=preset.telemetry,
        rollout_spans=preset.telemetry,
        policy_train_spans=preset.telemetry,
        algorithm=Algorithm(
            # Group-relative advantages over each prompt's answers.
            advantage_estimator="grpo",
            # The plain clipped loss: every sampled token contributes, with no off-policy mask or
            # reweighting of the tokens the current weights did not sample.
            policy_loss_type="regular",
            loss_reduction="token_mean",
            # No KL term against the reference, in the loss or in the reward.
            use_kl_loss=False,
            use_kl_in_reward=False,
            kl_estimator_type="k3_unbiased_gradient",
            # The policy loss uses unit correction weights.
            off_policy_correction="none",
            # Symmetric PPO clip.
            eps_clip_low=0.2,
            eps_clip_high=0.2,
        ),
        policy=Policy(
            optimizer_config=PolicyOptimizerConfig(
                optimizer="AdamW",
                lr=recipe.learning_rate,
                weight_decay=recipe.weight_decay,
                max_grad_norm=recipe.max_grad_norm,
            ),
            megatron_config=asdict(recipe.megatron),
        ),
        # The reference model shares the policy's geometry so it can share the policy's GPUs.
        ref=Ref(megatron_config=asdict(recipe.megatron)),
        placement=Placement(
            colocate_all=plan.colocate_all,
            # The reference model lives on the policy's nodes; only the engine has its own node.
            colocate_policy_ref=True,
            policy_num_nodes=plan.policy_num_nodes,
            policy_num_gpus_per_node=plan.policy_num_gpus_per_node,
            ref_num_nodes=plan.policy_num_nodes,
            ref_num_gpus_per_node=plan.policy_num_gpus_per_node,
        ),
        # Leases bound untrained groups to max_staleness_steps + 1 batches.
        rollout_buffer=RolloutBuffer(
            max_staleness_steps=preset.max_staleness_steps,
            batch_policy=preset.batch_policy,
            max_in_flight=preset.generation_workers,
            object_store_root=None,
        ),
    ).to_skyrl()
    config["generator"] = Generator(
        backend="vllm",
        model_dtype="bfloat16",
        vllm_attention_backend="FLASH_ATTN",
        # vLLM runs inside the job, on the engine node.
        run_engines_locally=True,
        # Weights reach the engine over NCCL from the policy ranks at each sync.
        weight_sync_backend="nccl",
        weight_sync_pause=WeightSyncPause(mode=PAUSE_MODE, clear_cache=preset.clear_kv_cache_on_weight_sync),
        # Each expert matrix goes from a Megatron rank that holds it straight to the vLLM workers that
        # serve it; the recipe meets its requirements (TP=1, ETP=1, local vLLM engines at TP=1 with
        # EP=DP=8, and the Triton MoE backend).
        weight_sync_transport="expert_block",
        expert_block_sync=ExpertBlockSync(timeout_seconds=600, verify=False),
        num_inference_engines=plan.num_inference_engines,
        inference_engine_tensor_parallel_size=plan.inference_engine_tensor_parallel_size,
        inference_engine_pipeline_parallel_size=plan.inference_engine_pipeline_parallel_size,
        inference_engine_data_parallel_size=plan.inference_engine_data_parallel_size,
        inference_engine_expert_parallel_size=plan.inference_engine_expert_parallel_size,
        n_samples_per_prompt=plan.n_samples_per_prompt,
        # Fraction of each engine GPU vLLM may occupy, weights and KV cache together; the rest
        # leaves room for the NCCL weight-sync buffers.
        gpu_memory_utilization=0.75,
        # Concurrent sequences per engine rank, above the 512 answers one update needs.
        max_num_seqs=1024,
        # Tokens one engine scheduling step may prefill or decode.
        max_num_batched_tokens=8192,
        # Share each prompt's prefill across its four answers; the cache is cleared at each pause.
        enable_prefix_caching=True,
        # Split long prefills across scheduling steps so decodes keep flowing.
        enable_chunked_prefill=True,
        # Capture CUDA graphs for decode.
        enforce_eager=False,
        # Task sessions send token requests directly to model engines.
        enable_http_endpoint=False,
        chat_template=None,
        engine_init_kwargs=dict(recipe.engine_init_kwargs),
        # Return the sampled token's logprob so the ratio diagnostics can compare it with the learner's.
        sampling_params=SamplingParams(temperature=1.0, top_p=1.0, logprobs=0),
    ).to_skyrl()
    # Let the allocator grow segments instead of fragmenting at the memory ceiling.
    config["extra_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for setting in settings:
        key = setting.partition("=")[0]
        if (
            key in ROLE_PLAN_SETTINGS | EVAL_DERIVED_SETTINGS
            or key
            in {
                "entrypoint",
                "trainer.placement.colocate_policy_ref",
                "trainer.resume_mode",
                "generator.run_engines_locally",
            }
            or key.startswith("data.")
        ):
            raise click.BadParameter(f"{key!r} is fixed by this launcher's run inputs")
    try:
        config = RecipePatch.from_document(config).with_settings(settings).to_skyrl()
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    trainer = config["trainer"]
    # Score the starting weights once when evaluation is on, so the curves have a step-0 point.
    trainer["eval_before_train"] = trainer["eval_interval"] > 0
    trainer["ckpt_interval"] = checkpoint_interval(trainer["max_steps"], trainer["eval_interval"])
    # Terminal export aligns with checkpoints and has no earlier periodic export.
    trainer["hf_save_interval"] = math.ceil(trainer["max_steps"] / trainer["ckpt_interval"]) * trainer["ckpt_interval"]
    # Pool prompts fit beside a complete response in the request window.
    budget = config["context_budget"]
    if MAX_PROMPT_TOKENS > budget["request_window_tokens"] - budget["max_new_tokens_per_turn"]:
        raise click.BadParameter("pool prompts do not fit the request window beside the response cap")
    authored = SkyRLRecipe.from_document(config)
    return authored.to_skyrl()


@dataclass(frozen=True)
class AsyncRun:
    rl: ArtifactStep[SkyRLRun]
    evaluation: ArtifactStep[EvaluationResult]


def build_run(
    policy: PolicySpec, preset: AsyncPreset, version: str | None, settings: tuple[str, ...] = (), *, seed: int = SEED
) -> AsyncRun:
    """Assemble the RL step and its evaluation for one policy and preset."""
    recipe = SNOWBALL_RECIPE
    config = training_config(preset, settings)
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    model = policy.adopted_model or model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    # A --set run gets its own address: the name carries a hash of its settings.
    identity_settings = settings if seed == SEED else (*settings, f"seed={seed}")
    changes = "\n".join(identity_settings)
    suffix = f"-set-{fingerprint_hash(changes)}" if identity_settings else ""
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy.label}-{preset.label}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=recipe.role_plan.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=recipe.role_plan,
            ),
            retention=RETENTION,
            seed=seed,  # Sampling and shuffling use the run seed.
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=f"lib/iris/config/{policy.cluster}.yaml",
            cpu=16,
            memory=recipe.host_memory,
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
    # evaluation_model_config adds max_new_tokens to request_window_tokens, so it gets the prompt share.
    budget = config["context_budget"]
    served = replace(
        CURRICULUM_TEMPLATE,
        request_window_tokens=budget["request_window_tokens"] - budget["max_new_tokens_per_turn"],
        max_new_tokens=budget["max_new_tokens_per_turn"],
    )
    # The eval artifact is keyed on the model name; the owner keeps two users at one version apart.
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy.label}-{preset.label}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{preset.evals}"
    evaluation_version = version or resolve_version(evaluation_base_name, None)
    evaluation_model = evaluation_model_config(policy, served, evaluation_model_name)
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
@click.option("--seed", type=int, default=SEED, show_default=True)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default=SMOKE_PRESET.label, show_default=True)
@click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Change one setting of the rendered RL config (dotted key).",
)
@click.option(
    "--stage",
    type=click.Choice(tuple(field.name for field in fields(AsyncRun))),
    default="rl",
    show_default=True,
    help="Terminal stage; evaluation includes the RL run automatically.",
)
@rl_build_options
def main(preset: str, settings: tuple[str, ...], stage: str, seed: int) -> dict[str, ArtifactStep]:
    run = build_run(SNOWBALL_POLICY, PRESETS[preset], version=None, settings=settings, seed=seed)
    return {f"{SNOWBALL_POLICY.label}-{preset}": getattr(run, stage)}


if __name__ == "__main__":
    main()
