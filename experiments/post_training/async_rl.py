# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Qwen or Snowball fully asynchronous RL on the curriculum pool.

One launcher for asynchronous RL experiments. It reuses the curriculum-RL policies, pool and
evaluation wiring and adds the fully asynchronous training loop and Megatron geometries. Every
setting below carries one sentence saying what it does;
presets bundle them and ``--set`` changes one. The launcher writes every setting it decides into
the rendered config, including values MarinSkyRL's base config or the curriculum template already
hold, so a run never depends on a default changing underneath it, and two runs that differ in any
setting never share an address. A Hydra override inherited from the curriculum policy is dropped
where the launcher writes its key, since Hydra applies overrides after the config.

Plan or run::

    python -m experiments.post_training.async_rl --version 2026.09.18 --preset smoke
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default --run
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default \\
        --set trainer.fully_async.max_staleness_steps=2 --run
    python -m experiments.post_training.async_rl --version 2026.09.19.3 --policy qwen --preset smoke \\
        --override documents/curriculum-rl-pool=2026.08.29.1 \\
        --override models/curriculum-rl-qwen3-0.6b=2026.08.29 \\
        --wandb-entity marin-community --set trainer.algorithm.use_tis=true \\
        --set trainer.algorithm.score_centering_topk=32 \\
        --set generator.sampling_params.logprobs=32 --run

The Snowball defaults are sized for the 40-GPU topology: 128 prompts per update at four answers each, 192
generation workers, a buffer of 32 finished groups, staleness 4, and an 8192-token request window
with a 4096-token response cap. A larger batch grows in prompts rather than in answers per prompt,
which would change the advantage estimate. The loop settings, the telemetry gates and the
``marin_tokenizer`` chat template need a MarinSkyRL revision carrying
marin-community/MarinSkyRL#654 and marin-community/MarinSkyRL#685; an older pin rejects the keys
when Hydra parses the config, before any GPU is used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import NamedTuple

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    _STRATEGY_FOR_PROFILE,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLEvaluationModel,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
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
    POOL_ARTIFACT_NAME,
    QWEN_POLICY,
    SEED,
    SNOWBALL_POLICY,
    SNOWBALL_SMOKE,
    PolicySpec,
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
# Prompts sampled per optimizer update; the trainer's batch sizes count prompts, not answers.
PROMPTS_PER_UPDATE = 128
# Answers sampled per prompt, so one update trains on 512 sequences.
ANSWERS_PER_PROMPT = 4
# Engines cancel requests still generating when new weights arrive, and the client resubmits each
# prompt with the tokens it already generated.
PAUSE_MODE = "abort"
# Two resumable checkpoints in the TTL-14-day temporary bucket; the terminal export is durable.
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2)


@dataclass(frozen=True)
class ChatTemplate:
    """The chat template the rollout runner renders conversations with."""

    source: str
    name_or_path: str


# The marin-tokenizer's own template registered in MarinSkyRL, so the runner tokenizes exactly what
# the tokenizer would.
CHAT_TEMPLATE = ChatTemplate(source="name", name_or_path="marin_tokenizer")
QWEN_CHAT_TEMPLATE = ChatTemplate(source="name", name_or_path="qwen3_without_thinking")
# This revision includes the score-centering learner and exact behavior top-k capture.
SCORE_CENTERING_SKYRL_COMMIT = "22a37adc7135a54995cfb4f3b5cd07504af796b3"


@dataclass(frozen=True)
class MegatronGeometry:
    """Megatron parallelism of one model, keyed as its ``megatron_config`` section."""

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    expert_tensor_parallel_size: int


@dataclass(frozen=True)
class TrainingRecipe:
    """How one policy trains: runtime, topology, parallel geometry and optimizer."""

    # MarinSkyRL runtime profile: the frozen dependency set the run installs.
    profile: SkyRLRuntimeProfile
    # Nodes the job holds: policy nodes plus rollout nodes.
    num_nodes: int
    # Placement and batch shape; MarinSkyRL writes these over the config from the topology.
    role_plan: SkyRLRolePlan
    # AdamW step size; the checked-in Snowball Megatron configs and the measured runs use 1e-6.
    learning_rate: float
    # AdamW decoupled weight decay; the base config's 1e-2, written here so it cannot drift.
    weight_decay: float
    # Gradient-norm clip applied before each optimizer step.
    max_grad_norm: float
    # Megatron parallelism for the policy and the reference model.
    megatron: MegatronGeometry
    # Host memory per training task; the resumed Megatron step-3 save reached Ray's
    # 95% kill threshold at 1800GB, so checkpoint staging needs more headroom.
    host_memory: str
    # vLLM engine settings the model needs beyond the ones the launcher writes itself.
    engine_init_kwargs: dict[str, object]
    # Optional policy optimizer checkpoint format; DP-local shards avoid a host-memory-heavy
    # gather on Snowball and require the same non-DP Megatron geometry on resume.
    policy_optimizer_checkpoint_sharding_type: str | None = None

    def __post_init__(self) -> None:
        plan = self.role_plan
        if not plan.colocate_all and self.num_nodes != plan.policy_num_nodes + plan.effective_rollout_num_nodes:
            raise ValueError("separate policy and engine roles require the declared rollout node count")
        rollout_gpus = (
            plan.num_inference_engines
            * plan.inference_engine_tensor_parallel_size
            * plan.inference_engine_pipeline_parallel_size
            * plan.inference_engine_data_parallel_size
        )
        if not plan.colocate_all and rollout_gpus != plan.effective_rollout_num_nodes * plan.policy_num_gpus_per_node:
            raise ValueError("separate rollout engines must use every allocated GPU")

    @property
    def strategy(self) -> str:
        return _STRATEGY_FOR_PROFILE[self.profile]


# Snowball 67B-A2B on the 40-GPU topology: four policy nodes with the reference colocated,
# pipeline depth 2 with 16-way data parallelism and experts sharded eight ways, and one engine node
# sharding experts across its eight ranks (DP8/EP8).
SNOWBALL_RECIPE = TrainingRecipe(
    profile=SkyRLRuntimeProfile.MEGATRON,
    num_nodes=5,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=4,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=PROMPTS_PER_UPDATE,
        policy_mini_batch_size=PROMPTS_PER_UPDATE,
        # One sequence per GPU per micro-step: 32 micro-steps over 16 data-parallel ranks.
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=ANSWERS_PER_PROMPT,
        inference_engine_data_parallel_size=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
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
    host_memory="1980GB",
    # The Triton MoE kernels; the fused defaults do not cover this expert layout.
    engine_init_kwargs={"moe_backend": "triton"},
    policy_optimizer_checkpoint_sharding_type="dp_reshardable",
)


# Keep the learner on one H100 node and place eight independent Qwen3 engines on another.
QWEN_RECIPE = TrainingRecipe(
    profile=SkyRLRuntimeProfile.MEGATRON,
    num_nodes=2,
    role_plan=SkyRLRolePlan(
        colocate_all=False,
        policy_num_nodes=1,
        policy_num_gpus_per_node=GPUS_PER_NODE,
        num_inference_engines=8,
        inference_engine_tensor_parallel_size=1,
        train_batch_size=32,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=ANSWERS_PER_PROMPT,
        rollout_num_nodes=1,
    ),
    learning_rate=1.0e-6,
    weight_decay=1e-2,
    max_grad_norm=1.0,
    megatron=MegatronGeometry(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    ),
    # Restoring a full Megatron optimizer checkpoint exceeded 128 GB on the learner pod.
    host_memory="256GB",
    engine_init_kwargs={},
)


@dataclass(frozen=True)
class AsyncPreset:
    """One bundle of async-loop settings: ``default`` is the house loop, ``smoke`` proves the wiring
    in two short updates, and ``on_policy`` admits only groups the current weights sampled."""

    label: str
    # Optimizer updates the run performs before it stops; the epoch bound never fires first.
    max_steps: int
    # Evaluate every this many updates, after the weight sync so the scored weights are the trained
    # ones; -1 turns evaluation off, including the pass at the end of training.
    eval_interval: int
    # How many updates old a group may be when the trainer consumes it; 0 admits only groups
    # sampled by the current weights.
    max_staleness_steps: int
    # Groups generating at once across the engines: at least one update's prompts, or the trainer
    # refuses to start.
    generation_workers: int
    # Finished groups held before a worker waits with its group in hand; always a literal, since
    # null would mean the worker count.
    max_buffered_groups: int
    # Prompt-plus-response budget one request may occupy in the engine, in tokens.
    request_window_tokens: int
    # Longest response the policy may generate, in tokens; it also caps the in-run evaluation.
    max_new_tokens: int
    # Evaluation suites the ``evaluation`` stage scores the terminal export on, comma separated.
    evals: str
    # Drop the engines' KV cache at the pause so nothing computed by the old weights is reused.
    clear_kv_cache_on_weight_sync: bool = True
    # Count a group's staleness from the policy version that sampled its first token rather than
    # from the trainer's step when the group was submitted.
    first_token_admission: bool = True
    # Export the telemetry the async dashboards read; each gate is written separately below.
    telemetry: bool = True
    # Record exact quantiles of the log-ratio; no dashboard panel reads them and they gather up to
    # 4M values per rank.
    exact_ratio_quantiles: bool = False
    # Track the cosine between successive gradients; costs one fp32 gradient copy per rank and one
    # all-reduce per update.
    grad_cosine: bool = False


# The house loop the module docstring sizes. An evaluation pauses generation for its 256 prompts,
# and at every 5 updates a run spent longer evaluating than training, so it evaluates every 10.
DEFAULT = AsyncPreset(
    label="default",
    max_steps=100,
    eval_interval=10,
    max_staleness_steps=4,
    generation_workers=192,
    max_buffered_groups=32,
    request_window_tokens=8192,
    max_new_tokens=4096,
    evals="math500,gsm8k-0shot",
)
# Two updates on the default geometry and batch with short responses: proves the wiring end to end.
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
PRESETS = {preset.label: preset for preset in (SMOKE_PRESET, DEFAULT, ON_POLICY)}
QWEN_DEFAULT = replace(DEFAULT, generation_workers=64, max_buffered_groups=16, max_steps=60, eval_interval=5)
QWEN_SMOKE = replace(SMOKE_PRESET, generation_workers=32, max_buffered_groups=8)
QWEN_ON_POLICY = replace(QWEN_DEFAULT, label="on_policy", max_staleness_steps=0, generation_workers=32)
QWEN_PRESETS = {preset.label: preset for preset in (QWEN_SMOKE, QWEN_DEFAULT, QWEN_ON_POLICY)}


def checkpoint_interval(max_steps: int, eval_interval: int) -> int:
    """A resumable checkpoint at every evaluation, or at the end when there is no evaluation."""
    return max_steps if eval_interval <= 0 else eval_interval


# The curriculum scale point this launcher renders the template through. Only the template's data
# and environment sections survive: entrypoint, context_budget, trainer and generator are rewritten
# in full below, so no field of the scale point reaches a run, and the two fields
# ``evaluation_serving`` reads are replaced with the rendered context budget before it sees them.
# Every scale point renders the same two sections; this is the curriculum's Snowball smoke point.
CURRICULUM_TEMPLATE = SNOWBALL_SMOKE


class Setting(NamedTuple):
    key: str
    value: object
    adds: bool


def parse_setting(text: str) -> Setting:
    """Split ``[+]dotted.key=value`` into a path, a YAML-parsed value and whether it may add a key."""
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"a setting must look like dotted.key=value, got {text!r}")
    return Setting(key.lstrip("+"), yaml.safe_load(value), key.startswith("+"))


# Keys MarinSkyRL's launcher writes from the topology and the run request as ++ overrides after the
# config, so a --set on them would be silently discarded; they change through the recipe only.
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
        "generator.rollout_num_nodes",
        "generator.inference_engine_tensor_parallel_size",
        "generator.inference_engine_pipeline_parallel_size",
        "trainer.train_batch_size",
        "trainer.policy_mini_batch_size",
        "trainer.micro_train_batch_size_per_gpu",
        "generator.n_samples_per_prompt",
        "trainer.max_ckpts_to_keep",
    }
)
# The role plan carries engine geometry to MarinSkyRL, which writes it over the config. Drop
# inherited overrides so the config and role plan agree.
RECIPE_OWNED_SETTINGS = frozenset(
    {
        "generator.inference_engine_data_parallel_size",
        "generator.inference_engine_expert_parallel_size",
    }
)
# Keys MarinSkyRL derives from context_budget and rejects as direct YAML.
DERIVED_CONTEXT_SETTINGS = frozenset(
    {
        "generator.max_input_length",
        "trainer.max_prompt_length",
        "generator.max_turns",
        "generator.sampling_params.max_generate_length",
        "generator.engine_init_kwargs.max_model_len",
    }
)
# Keys this launcher derives from the rendered trainer.eval_interval and trainer.max_steps.
EVAL_DERIVED_SETTINGS = frozenset({"trainer.eval_before_train", "trainer.ckpt_interval"})


def apply_setting(config: dict, setting: Setting) -> None:
    """Set one dotted key in the assembled config; unknown keys fail unless the setting adds."""
    key, value, adds = setting
    if key == "entrypoint":
        raise click.BadParameter("this launcher is the fully asynchronous loop; entrypoint cannot change")
    if key in TOPOLOGY_OWNED_SETTINGS:
        raise click.BadParameter(f"{key!r} is written from the topology after the config; change the recipe instead")
    if key in RECIPE_OWNED_SETTINGS:
        raise click.BadParameter(f"{key!r} is the engine geometry the recipe decides; change SNOWBALL_RECIPE instead")
    if key in DERIVED_CONTEXT_SETTINGS:
        raise click.BadParameter(f"MarinSkyRL derives {key!r} from context_budget; set context_budget instead")
    if key in EVAL_DERIVED_SETTINGS:
        raise click.BadParameter(f"{key!r} follows trainer.eval_interval; set that instead")
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


def check_loop_preflight(config: dict) -> None:
    """Reject request budgets or worker counts the downstream trainer would reject."""
    budget = config["context_budget"]
    trainer = config["trainer"]
    resume_mode = trainer["resume_mode"]
    if resume_mode not in ("latest", "none", "from_path"):
        raise click.BadParameter(f"unknown trainer.resume_mode {resume_mode!r}")
    if resume_mode == "from_path" and not trainer.get("resume_path"):
        raise click.BadParameter("trainer.resume_path is required when trainer.resume_mode=from_path")
    workers = trainer["fully_async"]["num_parallel_generation_workers"]
    mini_batch = trainer["policy_mini_batch_size"]
    # Every retained prompt must fit the request window beside the response budget, or rows skip
    # generation and their groups fail admission.
    if MAX_PROMPT_TOKENS > budget["request_window_tokens"] - budget["max_new_tokens_per_turn"]:
        raise click.BadParameter("pool prompts do not fit the request window beside the response cap")
    if workers < mini_batch:
        raise click.BadParameter(f"{workers} generation workers cannot fill a policy mini-batch of {mini_batch}")


def training_config(
    preset: AsyncPreset,
    settings: tuple[str, ...] = (),
    *,
    recipe: TrainingRecipe | None = None,
    chat_template: ChatTemplate = CHAT_TEMPLATE,
) -> dict:
    """Render the RL config: the curriculum data wiring plus every setting this launcher decides."""
    recipe = recipe or SNOWBALL_RECIPE
    plan = recipe.role_plan
    # The curriculum template supplies the data and environment sections; every other section is
    # written below in full.
    config = yaml.safe_load(rl_config_yaml(CURRICULUM_TEMPLATE))
    policy_megatron_config = asdict(recipe.megatron)
    if recipe.policy_optimizer_checkpoint_sharding_type is not None:
        policy_megatron_config["optimizer_checkpoint_sharding_type"] = recipe.policy_optimizer_checkpoint_sharding_type
    config["entrypoint"] = "fully_async"
    # The one public context declaration; MarinSkyRL derives the prompt, generation and engine
    # lengths from it.
    config["context_budget"] = {
        "request_window_tokens": preset.request_window_tokens,
        "max_new_tokens_per_turn": preset.max_new_tokens,
        # Single-turn math: the policy answers once and the episode ends. The four answers per
        # prompt are separate rollouts of it, not turns.
        "max_turns": 1,
    }
    config["trainer"] = {
        "strategy": recipe.strategy,
        # MarinSkyRL uses this flag to choose the Megatron attention backend too. Recipes default
        # to TransformerEngine fused attention; experiments may explicitly select FlashAttention.
        "flash_attn": False,
        # One sequence per row, as every measured run trained; packing changes the micro-step shape.
        "use_sample_packing": False,
        # Recompute activations in the backward pass; the 67B-A2B forward does not fit otherwise.
        "gradient_checkpointing": True,
        # Keep the optimizer state resident; the engines run on their own node.
        "offload_optimizer_during_rollouts": False,
        # Passes over the pool the dataloader may make; 100 updates of 128 prompts need several.
        "epochs": 50,
        "max_steps": preset.max_steps,
        # One optimizer pass over each batch; the mini batch equals the batch, so one update per batch.
        "update_epochs_per_batch": 1,
        "train_batch_size": plan.train_batch_size,
        "policy_mini_batch_size": plan.policy_mini_batch_size,
        "micro_train_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
        "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
        # Validation prompts scored per in-run evaluation.
        "eval_batch_size": 256,
        "eval_interval": preset.eval_interval,
        # Keep per-answer dumps so completion, correctness, and truncation can be re-scored.
        "dump_eval_results": True,
        # Optional token-level A/B/C probe; it adds a frozen reference scorer without
        # changing the training loss.
        "mismatch_decomposition": {"enabled": False, "sample_rows_per_step": 8},
        # No periodic HF export; the terminal export the launcher performs after training stays.
        "hf_save_interval": -1,
        # Resume from the latest resumable checkpoint on resubmission.
        "resume_mode": "latest",
        "max_ckpts_to_keep": RETENTION.resume_checkpoint_count,
        # Sampling and shuffling seed; --set trainer.seed=N changes it and the run's address with it.
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
            # The plain clipped loss: every sampled token contributes, with no off-policy mask or
            # reweighting of the tokens the current weights did not sample.
            "policy_loss_type": "regular",
            # No KL term against the reference, in the loss or in the reward.
            "use_kl_loss": False,
            "use_kl_in_reward": False,
            # No truncated importance sampling on top of the clip.
            "use_tis": False,
            # A positive score-centering width requires TIS and matching behavior top-k capture.
            "score_centering_topk": 0,
            # Truncate old-trainer / behavior ratios at two when TIS is enabled.
            "tis_imp_ratio_cap": 2.0,
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
            "megatron_config": policy_megatron_config,
        },
        # The reference model shares the policy's geometry so it can share the policy's GPUs.
        "ref": {"megatron_config": asdict(recipe.megatron)},
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
            "weight_sync_interval_steps": 1,
            "num_parallel_generation_workers": preset.generation_workers,
            "max_buffered_groups": preset.max_buffered_groups,
            "pause_mode": PAUSE_MODE,
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
        "rollout_num_nodes": plan.effective_rollout_num_nodes,
        "inference_engine_tensor_parallel_size": plan.inference_engine_tensor_parallel_size,
        "inference_engine_pipeline_parallel_size": plan.inference_engine_pipeline_parallel_size,
        "inference_engine_data_parallel_size": plan.inference_engine_data_parallel_size,
        "inference_engine_expert_parallel_size": plan.inference_engine_expert_parallel_size,
        "n_samples_per_prompt": plan.n_samples_per_prompt,
        # Fraction of each engine GPU vLLM may occupy, weights and KV cache together; the rest
        # leaves room for the NCCL weight-sync buffers.
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
        "chat_template": asdict(chat_template),
        "engine_init_kwargs": dict(recipe.engine_init_kwargs),
        "sampling_params": {
            # Full-distribution sampling; the ratio diagnostics assume no truncation.
            "temperature": 1.0,
            "top_p": 1.0,
            # Return the sampled token's logprob so the ratio diagnostics can compare it with the learner's.
            "logprobs": 0,
        },
        # One deterministic held-out answer per prompt, with the same response cap as training.
        "eval_n_samples_per_prompt": 1,
        "eval_sampling_params": {
            "max_generate_length": "${generator.sampling_params.max_generate_length}",
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": -1,
            "logprobs": None,
            "stop": None,
        },
    }
    # Let the allocator grow segments instead of fragmenting at the memory ceiling.
    config["extra_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for text in settings:
        apply_setting(config, parse_setting(text))
    trainer = config["trainer"]
    # Score the starting weights once when evaluation is on, so the curves have a step-0 point.
    trainer["eval_before_train"] = trainer["eval_interval"] > 0
    trainer["ckpt_interval"] = checkpoint_interval(trainer["max_steps"], trainer["eval_interval"])
    check_loop_preflight(config)
    return config


def config_keys(node: dict, prefix: str = "") -> set[str]:
    """Every dotted key the rendered config sets, sections included."""
    keys: set[str] = set()
    for key, value in node.items():
        keys.add(f"{prefix}{key}")
        if isinstance(value, dict):
            keys |= config_keys(value, f"{prefix}{key}.")
    return keys


def request_overrides(policy: PolicySpec, config: dict) -> tuple[str, ...]:
    """Keep inherited Hydra overrides only for keys absent from the rendered config."""
    written = config_keys(config)
    overrides = tuple(
        override
        for override in (*BASE_OVERRIDES, *policy.overrides)
        if override.lstrip("+").partition("=")[0] not in written
    )
    # The Iris backend supplies ``latest`` after the source config. An explicit
    # stage-boundary resume must therefore travel as a final request override.
    resume_mode = config["trainer"]["resume_mode"]
    if resume_mode == "latest":
        return overrides
    resume_overrides = (f"++trainer.resume_mode={resume_mode}",)
    if resume_mode == "from_path":
        resume_overrides += (f"++trainer.resume_path={json.dumps(str(config['trainer']['resume_path']))}",)
    return (*overrides, *resume_overrides)


@dataclass(frozen=True)
class AsyncRun:
    rl: ArtifactStep[SkyRLModel]
    evaluation: ArtifactStep[EvaluationResult]


def build_run(
    policy: PolicySpec,
    preset: AsyncPreset,
    version: str | None,
    settings: tuple[str, ...] = (),
    *,
    recipe: TrainingRecipe | None = None,
    chat_template: ChatTemplate = CHAT_TEMPLATE,
    wandb_entity: str | None = None,
) -> AsyncRun:
    """Assemble the RL step and its evaluation for one policy and preset."""
    recipe = recipe or SNOWBALL_RECIPE
    config = training_config(preset, settings, recipe=recipe, chat_template=chat_template)
    pool = pool_step(POOL_ARTIFACT_NAME, version or resolve_version(POOL_ARTIFACT_NAME, None))
    model = policy.adopted_model or model_step(version or resolve_version(MODEL_ARTIFACT_NAME, None))
    # Settings change what the run is, so they change its address: two --set runs never share one.
    changes = "\n".join(settings)
    suffix = f"-set-{hashlib.sha256(changes.encode()).hexdigest()[:8]}" if settings else ""
    base_name = f"checkpoints/{EXPERIMENT_NAME}/{policy.label}-{preset.label}{suffix}"
    rl = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(base_name),
            version=version or resolve_version(base_name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=recipe.profile, commit=SCORE_CENTERING_SKYRL_COMMIT),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=policy.tokenizer_uri,
                tokenizer_revision=policy.tokenizer_revision,
                relative_path=policy.model_relative_path,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=recipe.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=recipe.role_plan,
            ),
            retention=RETENTION,
            # The request's seed is what MarinSkyRL writes over the config, so it follows a --set.
            seed=config["trainer"]["seed"],
            overrides=request_overrides(policy, config),
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
            # The launch's W&B credential must have write access to this entity.
            wandb_entity=wandb_entity,
        ),
    )
    # The evaluation serves the rendered window, so a --set on the budget reaches the server; these
    # two fields are the only ones evaluation_serving reads.
    budget = config["context_budget"]
    served = replace(
        CURRICULUM_TEMPLATE,
        request_window_tokens=budget["request_window_tokens"],
        max_new_tokens=budget["max_new_tokens_per_turn"],
    )
    # The eval artifact is keyed on the model name; the owner keeps two users at one version apart.
    evaluation_model_name = f"{username_segment()}-{EXPERIMENT_NAME}-{policy.label}-{preset.label}{suffix}"
    evaluation_base_name = f"evals/{evaluation_model_name}/{preset.evals}"
    evaluation = eval_step(
        SkyRLEvaluationModel(step=rl, model=evaluation_serving(policy, served, evaluation_model_name)),
        preset.evals,
        version=version or resolve_version(evaluation_base_name, None),
        accelerator=f"{GPU_VARIANT}x{policy.serve_gpus}",
        submission_cluster=policy.cluster,
        federated_cluster=policy.cluster,
    )
    return AsyncRun(rl=rl, evaluation=evaluation)


@click.command(help=__doc__)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default="smoke", show_default=True)
@click.option("--policy", "policy_name", type=click.Choice(("qwen", "snowball")), default="snowball", show_default=True)
@click.option("--wandb-entity", help="W&B entity that the launch credential can write to.")
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
def main(
    preset: str,
    settings: tuple[str, ...],
    stage: str,
    policy_name: str = "snowball",
    wandb_entity: str | None = None,
) -> dict[str, ArtifactStep]:
    policy = QWEN_POLICY if policy_name == "qwen" else SNOWBALL_POLICY
    recipe = QWEN_RECIPE if policy_name == "qwen" else SNOWBALL_RECIPE
    chat_template = QWEN_CHAT_TEMPLATE if policy_name == "qwen" else CHAT_TEMPLATE
    presets = QWEN_PRESETS if policy_name == "qwen" else PRESETS
    run = build_run(
        policy,
        presets[preset],
        version=None,
        settings=settings,
        recipe=recipe,
        chat_template=chat_template,
        wandb_entity=wandb_entity,
    )
    return {f"{policy.label}-{preset}": getattr(run, stage)}


if __name__ == "__main__":
    main()
