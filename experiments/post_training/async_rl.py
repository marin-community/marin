# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch fully asynchronous RL on the curriculum pool at the async programme's house settings.

One launcher for asynchronous RL experiments. It reuses the curriculum-RL policy, pool and
evaluation wiring and adds the fully asynchronous training loop, the Megatron geometry the 67B-A2B
Snowball policy trains with, and the loop settings the async programme settled on. Every setting
below carries one sentence saying what it does and why it has its value; presets bundle them,
``--set`` changes one. The launcher writes every setting it decides into the rendered config,
including values MarinSkyRL's base config or the curriculum template already hold, so the run
never depends on a default changing underneath it.

Plan or run::

    python -m experiments.post_training.async_rl --version 2026.09.18 --preset smoke
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default --run
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default \\
        --set trainer.fully_async.max_staleness_steps=1 --run

Two parts of the programme's frozen recipe are not here. The off-policy correction
(``regular_mask``) lives in marin-community/MarinSkyRL#628, so this launcher trains the plain
``regular`` loss. The stopping package (``parser_only``) exists only on the research stack, so runs
use MarinSkyRL's stock answer parsing and stopping. The loop settings, the telemetry gates and the
``marin_tokenizer`` chat template are declared only by the MarinSkyRL revisions that carry the
async-knobs and telemetry changes; an older pinned revision rejects them when Hydra parses the
config, before any GPU is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, replace
from enum import StrEnum

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLEvaluationModel,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.provenance import username_segment

from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.curriculum_rl.launch import (
    BASE_OVERRIDES,
    GPU_VARIANT,
    GPUS_PER_NODE,
    MODEL_ARTIFACT_NAME,
    POLICIES,
    POOL_ARTIFACT_NAME,
    SEED,
    SNOWBALL_SMOKE,
    PolicySpec,
    ScalePreset,
    evaluation_serving,
    model_step,
    rl_config_yaml,
)
from experiments.post_training.curriculum_rl.pool import (
    MAX_PROMPT_TOKENS,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)

EXPERIMENT_NAME = "async-rl"
WANDB_PROJECT = f"marin-{EXPERIMENT_NAME}"
# Prompts sampled per optimizer update; the trainer's batch sizes count prompts, not answers. 128
# is where the simulated loop saturates on the house topology: four times the programme's 32 for
# +184% consumed tokens per second, zero discards and the trainer idle 8% instead of 50%; 4,096
# sequences per update would buy 16% more and fill the KV cache. The batch grows in prompts, not
# answers: 128 x 4 and 32 x 16 sample the same tokens per second, and more answers per prompt
# would change the advantage estimate.
PROMPTS_PER_UPDATE = 128
# Answers sampled per prompt, so one update trains on 512 sequences.
ANSWERS_PER_PROMPT = 4
# The named chat template the rollout runner renders conversations with; it is the marin-tokenizer's
# own template registered in MarinSkyRL, so the runner tokenizes exactly what the tokenizer would.
CHAT_TEMPLATE = {"source": "name", "name_or_path": "marin_tokenizer"}


class PauseMode(StrEnum):
    """What the vLLM engines do with requests still generating when new weights arrive."""

    # Cancel them; the client resubmits each prompt with the tokens it already generated, so the
    # answer continues under the new weights after one prefill. The programme's measured default.
    ABORT = "abort"
    # Freeze them in the scheduler and resume them after the reload; they keep the cache the old
    # weights built only when clear_kv_cache_on_weight_sync is false.
    KEEP = "keep"


@dataclass(frozen=True)
class TrainingRecipe:
    """How one policy trains: backend, parallel geometry, engine geometry and optimizer."""

    # MarinSkyRL runtime profile: the frozen dependency set the run installs; the trainer backend
    # (megatron) follows from it.
    profile: SkyRLRuntimeProfile
    # AdamW step size for the policy; every measured programme run and the checked-in Snowball
    # Megatron configs use 1e-6.
    learning_rate: float
    # AdamW decoupled weight decay; the base config's 1e-2, written here so it cannot drift.
    weight_decay: float
    # Gradient-norm clip applied before each optimizer step.
    max_grad_norm: float
    # Megatron parallelism for the policy and the reference model.
    megatron: dict[str, int]
    # Data and expert parallelism of the one vLLM engine, which spans one node.
    engine_data_parallel_size: int
    engine_expert_parallel_size: int
    # Host memory per training task; the programme's Megatron runs asked for 1800GB per node so
    # checkpoint staging never ran out.
    task_memory: str
    # vLLM engine settings the model needs beyond the ones the launcher writes itself.
    engine_init_kwargs: dict[str, object]

    @property
    def strategy(self) -> str:
        return "megatron" if self.profile is SkyRLRuntimeProfile.MEGATRON else "fsdp2"


# Snowball 67B-A2B on the 40-GPU house topology: pipeline depth 2 with 16-way data parallelism
# across four policy nodes, experts sharded eight ways, no tensor or context parallelism; the one
# engine shards experts across its node's eight ranks (DP8/EP8). The shape of every measured run.
SNOWBALL_RECIPE = TrainingRecipe(
    profile=SkyRLRuntimeProfile.MEGATRON,
    learning_rate=1.0e-6,
    weight_decay=1e-2,
    max_grad_norm=1.0,
    megatron={
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    },
    engine_data_parallel_size=GPUS_PER_NODE,
    engine_expert_parallel_size=GPUS_PER_NODE,
    task_memory="1800GB",
    # The Triton MoE kernels; the fused defaults do not cover this expert layout.
    engine_init_kwargs={"moe_backend": "triton"},
)
RECIPES = {"snowball": SNOWBALL_RECIPE}


@dataclass(frozen=True)
class AsyncPreset:
    """One bundle of async-loop settings; ``smoke`` proves wiring, ``default`` is the house loop."""

    label: str
    # Optimizer updates the run performs before it stops; the epoch bound never fires first.
    max_steps: int
    # Evaluate every this many updates; -1 turns evaluation off entirely, including the pass at the
    # end of training. Evaluation waits until the weight sync completes, so it scores the weights
    # the run just trained.
    eval_interval: int
    # How many updates old a group may be when the trainer consumes it; 0 admits only groups
    # sampled by the current weights.
    max_staleness_steps: int
    # Groups generating at once across the engines: at least one update's prompts, or the trainer
    # refuses to start, and few enough that every group in flight or buffered is consumed inside
    # the staleness allowance.
    generation_workers: int
    # Finished groups the completed buffer holds before a worker waits with its group in hand. The
    # depth moves throughput nowhere; a deeper buffer lengthens residence and re-pays the prompt at
    # every abort (554 re-prefills at 32 against 3,704 at 128), and 32 keeps a full update of
    # staleness headroom when responses lengthen by half, where 64 has none. Always a literal:
    # null would mean the worker count, an invisible buffer nobody chose.
    max_buffered_groups: int
    # Prompt-plus-response budget one request may occupy in the engine, in tokens.
    request_window_tokens: int
    # Longest response the policy may generate, in tokens; it also caps the in-run evaluation.
    max_new_tokens: int
    # Evaluation suites the ``evaluation`` stage scores the terminal export on, comma separated.
    evals: str
    # What engines do with in-flight requests at a weight sync; see PauseMode.
    pause_mode: PauseMode = PauseMode.ABORT
    # Drop the engines' KV cache at the pause so nothing computed by the old weights is reused.
    clear_kv_cache_on_weight_sync: bool = True
    # Count a group's staleness from the policy version that sampled its first token rather than
    # from the trainer's step when the group was submitted.
    first_token_admission: bool = True
    # Export the telemetry the async dashboards read: trainer scalars, async-loop windows and
    # generation-worker waits, the per-rank Megatron policy-update spans, the Megatron optimizer
    # inventory and the pooled learner-versus-engine ratio diagnostics. Each is a separate gate.
    telemetry: bool = True
    # Record exact quantiles of the log-ratio on top of the pooled diagnostics; no dashboard panel
    # reads them and they gather up to 4M values per rank.
    exact_ratio_quantiles: bool = False
    # Track the cosine between successive gradients; costs one fp32 gradient copy per rank at the
    # micro-batch-1 memory ceiling and one all-reduce per update.
    grad_cosine: bool = False

    def __post_init__(self) -> None:
        # Every retained prompt must fit the request window beside the response budget, or rows skip
        # generation and their groups fail admission.
        if MAX_PROMPT_TOKENS > self.request_window_tokens - self.max_new_tokens:
            raise ValueError(f"{self.label}: pool prompts do not fit the request window beside the response cap")
        if self.generation_workers < PROMPTS_PER_UPDATE:
            raise ValueError(
                f"{self.label}: {self.generation_workers} workers cannot fill an update of {PROMPTS_PER_UPDATE}"
            )
        # At staleness 0 nothing stale is ever admitted: every group that crosses a weight sync is
        # discarded and regenerated, so the allowance bounds no queue and the ratio check is moot.
        if self.max_staleness_steps == 0:
            return
        in_flight = self.generation_workers + self.max_buffered_groups
        if in_flight / PROMPTS_PER_UPDATE >= self.max_staleness_steps:
            raise ValueError(
                f"{self.label}: {in_flight} groups in flight or buffered exceed "
                f"{self.max_staleness_steps} updates of {PROMPTS_PER_UPDATE}, so the last of them would age out"
            )

    def scale(self, policy: PolicySpec) -> ScalePreset:
        """The curriculum scale point this preset trains at for ``policy``."""
        # 128 prompts of 4 answers per update, one sequence per GPU per micro-step (32 micro-steps
        # over 16 data-parallel ranks), no packing: the house shape.
        plan = replace(
            SNOWBALL_SMOKE.role_plan,
            train_batch_size=PROMPTS_PER_UPDATE,
            policy_mini_batch_size=PROMPTS_PER_UPDATE,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=ANSWERS_PER_PROMPT,
        )
        return replace(
            SNOWBALL_SMOKE,
            label=f"{policy.label}-{self.label}",
            role_plan=plan,
            max_steps=self.max_steps,
            eval_interval=self.eval_interval,
            # A resumable checkpoint at every evaluation, or at the end when there is no evaluation.
            ckpt_interval=self.max_steps if self.eval_interval <= 0 else self.eval_interval,
            request_window_tokens=self.request_window_tokens,
            max_new_tokens=self.max_new_tokens,
            micro_forward_batch_size_per_gpu=1,
            evals=self.evals,
        )


# The house loop: staleness 4 with 192 workers (1.5 updates of prompts generating, where the
# simulated throughput plateaus: 160 gives 1% less, 224 gives 0.5% more for seven points of KV)
# and 32 groups buffered, 224 groups against an allowance of 512 so uneven finish times never age
# a group out; abort at the copy; 100 updates evaluated every 5. The window is the programme's
# 8192 with a 4096-token cap: mean generated length is about 1,060 tokens, so a 16k window adds
# no capacity and only lets the long tail run on and age out (discards 0% to 3%), and the 32k
# window and max_num_seqs 16 of the 64-GPU Snowball Ultra campaign are that topology's numbers,
# which here would collapse concurrency from about 8,192 slots to about 124.
DEFAULT = AsyncPreset(
    label="default",
    max_steps=100,
    eval_interval=5,
    max_staleness_steps=4,
    generation_workers=192,
    max_buffered_groups=32,
    request_window_tokens=8192,
    max_new_tokens=4096,
    evals="math500,gsm8k-0shot",
)
# Two updates on the house geometry and batch with short responses: proves the wiring end to end
# and measures policy_train at its 32 micro-steps before the first long run.
SMOKE_PRESET = replace(
    DEFAULT,
    label="smoke",
    max_steps=2,
    eval_interval=-1,
    request_window_tokens=2048,
    max_new_tokens=1024,
    evals="gsm8k-smoke",
)
# Every consumed group was sampled by the current weights, so importance ratios are exactly one up
# to engine mismatch; generation that crosses a weight sync is discarded rather than trained on.
# Workers sit at the trainer's floor of one update's prompts.
ON_POLICY = replace(DEFAULT, label="on_policy", max_staleness_steps=0, generation_workers=PROMPTS_PER_UPDATE)
PRESETS = {preset.label: preset for preset in (SMOKE_PRESET, DEFAULT, ON_POLICY)}


def parse_setting(text: str) -> tuple[str, object, bool]:
    """Split ``[+]dotted.key=value`` into a path, a YAML-parsed value and whether it may add a key."""
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"a setting must look like dotted.key=value, got {text!r}")
    adds = key.startswith("+")
    return key.lstrip("+"), yaml.safe_load(value), adds


# Keys MarinSkyRL's launcher writes from the topology and the run request as ++ overrides after the
# config, so a --set on them would be silently discarded; they change through the preset only.
TOPOLOGY_OWNED_SETTINGS = frozenset(
    {
        "trainer.placement.colocate_all",
        "trainer.placement.colocate_policy_ref",
        "trainer.placement.policy_num_nodes",
        "trainer.placement.policy_num_gpus_per_node",
        "trainer.placement.ref_num_nodes",
        "trainer.placement.ref_num_gpus_per_node",
        "generator.run_engines_locally",
        "generator.num_inference_engines",
        "generator.inference_engine_tensor_parallel_size",
        "generator.inference_engine_data_parallel_size",
        "generator.inference_engine_expert_parallel_size",
        "trainer.train_batch_size",
        "trainer.policy_mini_batch_size",
        "trainer.micro_train_batch_size_per_gpu",
        "generator.n_samples_per_prompt",
        "trainer.seed",
        "trainer.resume_mode",
        "trainer.max_ckpts_to_keep",
    }
)


def apply_setting(config: dict, key: str, value: object, *, adds: bool) -> None:
    """Set one dotted key in the assembled config; unknown keys fail unless the override adds."""
    if key == "entrypoint":
        raise click.BadParameter("this launcher is the fully asynchronous loop; entrypoint cannot change")
    if key in TOPOLOGY_OWNED_SETTINGS:
        raise click.BadParameter(f"{key!r} is written from the topology after the config; change the preset instead")
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if isinstance(node.get(part), dict):
            node = node[part]
        elif part in node:
            raise click.BadParameter(f"{key!r} descends into a setting that holds a value, not a section")
        elif adds:
            node = node.setdefault(part, {})
        else:
            raise click.BadParameter(f"unknown setting {key!r}; prefix with + to add a new key")
    if parts[-1] not in node and not adds:
        raise click.BadParameter(f"unknown setting {key!r}; prefix with + to add a new key")
    node[parts[-1]] = value


def training_config(
    policy: PolicySpec, preset: AsyncPreset, recipe: TrainingRecipe, settings: tuple[str, ...] = ()
) -> str:
    """Render the RL config: the curriculum data wiring plus every setting this launcher decides."""
    scale = preset.scale(policy)
    plan = scale.role_plan
    # The curriculum template supplies the data and environment sections; every other section is
    # written below in full, so nothing is inherited from the template or the base config.
    config = yaml.safe_load(rl_config_yaml(scale))
    config["entrypoint"] = "fully_async"
    # The one public context declaration; MarinSkyRL derives the prompt, generation and engine
    # lengths from it and rejects those keys as direct YAML.
    config["context_budget"] = {
        "request_window_tokens": preset.request_window_tokens,
        "max_new_tokens_per_turn": preset.max_new_tokens,
        # Single-turn math: one answer per prompt.
        "max_turns": 1,
    }
    config["trainer"] = {
        "strategy": recipe.strategy,
        # Megatron brings its own fused attention; flash_attn is the FSDP2 switch.
        "flash_attn": False,
        # One sequence per row, as every measured run trained; packing changes the micro-step shape.
        "use_sample_packing": False,
        # Recompute activations in the backward pass; the 67B-A2B forward does not fit otherwise.
        "gradient_checkpointing": True,
        # Keep the optimizer state resident; the engines run on their own node, so nothing needs
        # the policy's memory while it waits.
        "offload_optimizer_during_rollouts": False,
        # Passes over the pool the dataloader may make; 100 updates of 128 prompts need several.
        "epochs": 50,
        "max_steps": scale.max_steps,
        # One optimizer pass over each batch; the mini batch equals the batch, so one update per batch.
        "update_epochs_per_batch": 1,
        "train_batch_size": plan.train_batch_size,
        "policy_mini_batch_size": plan.policy_mini_batch_size,
        "micro_train_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
        "micro_forward_batch_size_per_gpu": scale.micro_forward_batch_size_per_gpu,
        # Validation prompts scored per in-run evaluation.
        "eval_batch_size": 256,
        # Score the starting weights once when evaluation is on, so the curves have a step-0 point.
        "eval_before_train": scale.eval_interval > 0,
        "eval_interval": scale.eval_interval,
        "ckpt_interval": scale.ckpt_interval,
        # No periodic HF export; the terminal export the launcher performs after training stays.
        "hf_save_interval": -1,
        # Resume from the latest resumable checkpoint on resubmission.
        "resume_mode": "latest",
        "seed": SEED,
        "logger": "wandb",
        "project_name": WANDB_PROJECT,
        # Commit each step's metrics as they are logged, so a killed run keeps its curve.
        "tracker_commit_each_step": True,
        "training_metrics": preset.telemetry,
        "async_spans": preset.telemetry,
        "policy_train_spans": preset.telemetry,
        # Per-request generation spans; no async dashboard panel reads them.
        "generate_spans": False,
        "optimizer_state_metrics": preset.telemetry,
        "algorithm": {
            # Group-relative advantages over each prompt's answers.
            "advantage_estimator": "grpo",
            # The plain clipped loss; the off-policy mask is MarinSkyRL#628.
            "policy_loss_type": "regular",
            # No KL term against the reference, in the loss or in the reward.
            "use_kl_loss": False,
            "use_kl_in_reward": False,
            # No truncated importance sampling on top of the clip.
            "use_tis": False,
            # Symmetric PPO clip.
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "ratio_diagnostics": {"pooled": preset.telemetry, "exact_quantiles": preset.exact_ratio_quantiles},
            "grad_cosine": {"enabled": preset.grad_cosine},
        },
        "policy": {
            "optimizer_config": {
                "optimizer": "AdamW",
                "lr": recipe.learning_rate,
                "weight_decay": recipe.weight_decay,
                "max_grad_norm": recipe.max_grad_norm,
            },
            "megatron_config": dict(recipe.megatron),
        },
        # The reference model shares the policy's geometry so it can share the policy's GPUs.
        "ref": {"megatron_config": dict(recipe.megatron)},
        "placement": {
            "colocate_all": plan.colocate_all,
            # The reference model lives on the policy's nodes; only the engine has its own node.
            "colocate_policy_ref": True,
            "policy_num_nodes": plan.policy_num_nodes,
            "policy_num_gpus_per_node": plan.policy_num_gpus_per_node,
            "ref_num_nodes": plan.policy_num_nodes,
            "ref_num_gpus_per_node": plan.policy_num_gpus_per_node,
        },
        # Group-scale knobs of the fully asynchronous loop; see AsyncPreset for each.
        "fully_async": {
            "max_staleness_steps": preset.max_staleness_steps,
            "num_parallel_generation_workers": preset.generation_workers,
            "max_buffered_groups": preset.max_buffered_groups,
            "pause_mode": preset.pause_mode.value,
            "clear_kv_cache_on_weight_sync": preset.clear_kv_cache_on_weight_sync,
            "first_token_admission": preset.first_token_admission,
        },
    }
    config["generator"] = {
        "backend": "vllm",
        "model_dtype": "bfloat16",
        "vllm_attention_backend": "FLASH_ATTN",
        # The engine runs inside the job on its own node rather than as a remote server.
        "run_engines_locally": True,
        # Weights reach the engine over NCCL from the policy ranks at each sync.
        "weight_sync_backend": "nccl",
        # The asynchronous engine API, which the abort pause and the HTTP route need.
        "async_engine": True,
        # Requests go one prompt at a time, as the chat route submits them.
        "batched": False,
        "num_inference_engines": plan.num_inference_engines,
        "inference_engine_tensor_parallel_size": plan.inference_engine_tensor_parallel_size,
        "inference_engine_pipeline_parallel_size": 1,
        "inference_engine_data_parallel_size": recipe.engine_data_parallel_size,
        "inference_engine_expert_parallel_size": recipe.engine_expert_parallel_size,
        "n_samples_per_prompt": plan.n_samples_per_prompt,
        # KV-cache share of each engine GPU; the rest holds the weights and the sync buffers.
        "gpu_memory_utilization": 0.75,
        # Concurrent sequences per engine rank, above the 512 answers one update needs.
        "max_num_seqs": 1024,
        # Tokens one engine scheduling step may prefill or decode.
        "max_num_batched_tokens": 8192,
        # Reuse prefilled prefixes across an answer's resubmissions after an abort.
        "enable_prefix_caching": True,
        # Split long prefills across scheduling steps so decodes keep flowing.
        "enable_chunked_prefill": True,
        # CUDA graphs on; eager mode costs decode throughput.
        "enforce_eager": False,
        # The fully asynchronous entrypoint samples through the OpenAI-compatible chat route.
        "enable_http_endpoint": True,
        # Each turn re-renders the conversation through the chat template; the entrypoint requires it.
        "use_conversation_multi_turn": True,
        "chat_template": dict(CHAT_TEMPLATE),
        "engine_init_kwargs": dict(recipe.engine_init_kwargs),
        "sampling_params": {
            # Full-distribution sampling; the ratio diagnostics assume no truncation.
            "temperature": 1.0,
            "top_p": 1.0,
            # Return the sampled token's logprob so the ratio diagnostics can compare it with the learner's.
            "logprobs": 0,
        },
    }
    # Let the allocator grow segments instead of fragmenting at the memory ceiling.
    config["extra_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for text in settings:
        key, value, adds = parse_setting(text)
        apply_setting(config, key, value, adds=adds)
    return yaml.safe_dump(config, sort_keys=False)


def engine_geometry_overrides(recipe: TrainingRecipe) -> tuple[str, ...]:
    """Hydra overrides restating the engine's data and expert parallelism.

    MarinSkyRL writes the engine geometry it derives from the role plan as overrides after reading
    the config, and the role plan marin sends carries no data or expert parallelism; the request
    overrides are applied last, so these keep the values the config declares.
    """
    return (
        f"generator.inference_engine_data_parallel_size={recipe.engine_data_parallel_size}",
        f"generator.inference_engine_expert_parallel_size={recipe.engine_expert_parallel_size}",
    )


@dataclass(frozen=True)
class AsyncRun:
    rl: ArtifactStep[SkyRLModel]
    evaluation: ArtifactStep[EvaluationResult]


def build_run(
    *, policy: PolicySpec, preset: AsyncPreset, version: str | None, settings: tuple[str, ...] = ()
) -> AsyncRun:
    """Assemble the RL step and its evaluation for one policy and preset."""
    recipe = RECIPES[policy.label]
    scale = preset.scale(policy)
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    model = policy.adopted_model or model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    # Settings change what the run is, so they change its address: two --set runs never share one.
    suffix = f"-set-{hashlib.sha256(chr(10).join(settings).encode()).hexdigest()[:8]}" if settings else ""
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy.label}-{preset.label}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=training_config(policy, preset, recipe, settings),
            runtime=SkyRLRuntime(profile=recipe.profile),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=scale.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=scale.role_plan,
            ),
            # Two resumable checkpoints in the TTL-14-day temporary bucket; the terminal export is durable.
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
            overrides=(*BASE_OVERRIDES, *engine_geometry_overrides(recipe)),
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=f"lib/iris/config/{policy.cluster}.yaml",
            cpu=16,
            memory=recipe.task_memory,
            disk="2TB",
            priority="interactive",
            # One automatic retry, then fail; a healthy run resumes from its latest checkpoint on resubmission.
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )
    # The eval artifact is keyed on the model name; the owner keeps two users at one version apart.
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy.label}-{preset.label}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{scale.evals}"
    evaluation = eval_step(
        SkyRLEvaluationModel(step=rl, model=evaluation_serving(policy, scale, evaluation_model_name)),
        scale.evals,
        version=version or resolve_version(evaluation_base_name, None),
        accelerator=f"{GPU_VARIANT}x{policy.serve_gpus}",
        submission_cluster=policy.cluster,
        federated_cluster=policy.cluster,
    )
    return AsyncRun(rl=rl, evaluation=evaluation)


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default="smoke", show_default=True)
@click.option("--model", "model_label", type=click.Choice(sorted(RECIPES)), default="snowball", show_default=True)
@click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Change one setting of the rendered RL config (dotted key; prefix + to add a new key).",
)
@click.option(
    "--stage",
    type=click.Choice(tuple(field.name for field in fields(AsyncRun))),
    default="rl",
    show_default=True,
    help="Terminal stage; evaluation includes the RL run automatically.",
)
@build_options
def main(preset: str, model_label: str, settings: tuple[str, ...], stage: str) -> dict[str, ArtifactStep]:
    run = build_run(policy=POLICIES[model_label], preset=PRESETS[preset], version=None, settings=settings)
    return {f"{model_label}-{preset}": getattr(run, stage)}


if __name__ == "__main__":
    main()
