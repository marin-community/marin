# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch fully asynchronous RL on the curriculum pool with the async programme's defaults.

One launcher for asynchronous RL experiments. It reuses the curriculum-RL policies, pool and
scale presets and adds the fully asynchronous training loop, the Megatron geometry the 67B-A2B
Snowball policy trains with, and the loop settings the async programme measured. Every setting
below carries one sentence saying what it does; presets bundle them, ``--set`` changes one.

Plan or run::

    python -m experiments.post_training.async_rl --version 2026.09.18 --preset smoke
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default --run
    python -m experiments.post_training.async_rl --version 2026.09.18 --preset default \\
        --set trainer.fully_async.max_staleness_steps=1 --run

Two parts of the programme's frozen recipe are not here. The off-policy correction
(``regular_mask``) lives in marin-community/MarinSkyRL#628, so this launcher trains the plain
``regular`` loss. The stopping package (``parser_only``) exists only on the research stack, so runs
use MarinSkyRL's stock answer parsing and stopping. The loop settings and the telemetry gates
are declared only by the MarinSkyRL revisions that carry the async-knobs and telemetry changes;
an older pinned revision rejects them when Hydra parses the config, before any GPU is used.
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
    SMOKE,
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
    """How one policy trains: backend, parallel geometry and optimizer step size."""

    # MarinSkyRL runtime profile: the frozen dependency set the run installs; this launcher derives
    # the trainer backend (fsdp2 or megatron) from it.
    profile: SkyRLRuntimeProfile
    # Adam step size for the policy; the async programme's Snowball runs used 1e-6.
    learning_rate: float
    # Megatron parallelism for the policy and the reference model, or None for FSDP2.
    megatron: dict[str, int] | None = None
    # Host memory per training task; the programme's Megatron runs asked for 1800GB per node so
    # checkpoint staging never ran out.
    task_memory: str | None = None
    # vLLM engine settings the model needs beyond the curriculum defaults.
    engine_init_kwargs: dict[str, object] | None = None

    @property
    def strategy(self) -> str:
        return "megatron" if self.profile is SkyRLRuntimeProfile.MEGATRON else "fsdp2"


# Snowball 67B-A2B: pipeline depth 2 with 16-way data parallelism across four policy nodes, experts
# sharded eight ways, no tensor or context parallelism, as in every measured programme run.
SNOWBALL_RECIPE = TrainingRecipe(
    profile=SkyRLRuntimeProfile.MEGATRON,
    learning_rate=1.0e-6,
    megatron={
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    },
    task_memory="1800GB",
    engine_init_kwargs={"moe_backend": "triton"},
)
QWEN_RECIPE = TrainingRecipe(profile=SkyRLRuntimeProfile.FSDP, learning_rate=2.0e-6)
RECIPES = {"snowball": SNOWBALL_RECIPE, "qwen": QWEN_RECIPE}


@dataclass(frozen=True)
class AsyncPreset:
    """One bundle of async-loop settings; ``smoke`` proves wiring, ``default`` is the measured loop."""

    label: str
    # Optimizer updates the run performs before it stops.
    max_steps: int
    # Evaluate every this many updates; -1 turns evaluation off entirely, including the pass at the
    # end of training. Evaluation waits until the weight sync completes, so it scores the weights
    # the run just trained.
    eval_interval: int
    # How many updates old a group may be when the trainer consumes it; 0 admits only groups
    # sampled by the current weights.
    max_staleness_steps: int
    # Groups generating at once across the engines. The programme's arms used
    # (max_staleness_steps + 1) times the groups per update; its later derivation puts the count
    # actually needed near 128 at staleness 4, so 160 is what the quality arm ran, not a minimum.
    generation_workers: int
    # Finished groups the completed buffer holds before a worker waits with its group in hand;
    # None means one update's worth (the mini batch), which bounds the head-node backlog.
    max_buffered_groups: int | None
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
    # generation-worker waits, the per-rank Megatron policy-update spans, and the Megatron optimizer
    # inventory. Each family is a separate MarinSkyRL gate.
    telemetry: bool = True
    # Track the cosine between successive gradients; costs one gradient-sized buffer per rank and
    # one all-reduce per update.
    grad_cosine: bool = False

    def scale(self, policy: PolicySpec) -> ScalePreset:
        """The curriculum scale point this preset trains at for ``policy``."""
        base = SNOWBALL_SMOKE if policy.label == "snowball" else SMOKE
        # 32 groups of 4 answers per update, one sequence per GPU per micro-step, no packing: the
        # measured shape.
        plan = replace(
            base.role_plan,
            train_batch_size=32,
            policy_mini_batch_size=32,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=4,
        )
        return replace(
            base,
            label=f"{base.label}-{self.label}",
            role_plan=plan,
            max_steps=self.max_steps,
            eval_interval=self.eval_interval,
            ckpt_interval=self.max_steps if self.eval_interval <= 0 else self.eval_interval,
            request_window_tokens=self.request_window_tokens,
            max_new_tokens=self.max_new_tokens,
            micro_forward_batch_size_per_gpu=1,
            evals=self.evals,
        )


# The programme's measured loop settings: staleness 4 with 160 workers (5 updates of 32 groups in
# flight), one update's cohort buffered, abort at the copy, 100 updates evaluated every 20. The
# off-policy correction the measured runs added on top is not in this launcher (see the module doc).
DEFAULT = AsyncPreset(
    label="default",
    max_steps=100,
    eval_interval=20,
    max_staleness_steps=4,
    generation_workers=160,
    max_buffered_groups=None,
    request_window_tokens=8192,
    max_new_tokens=4096,
    evals="math500,gsm8k-0shot",
)
# Two updates on the measured geometry with short responses: proves the wiring end to end.
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
ON_POLICY = replace(DEFAULT, label="on_policy", max_staleness_steps=0, generation_workers=32)
PRESETS = {preset.label: preset for preset in (SMOKE_PRESET, DEFAULT, ON_POLICY)}

# Every retained prompt must fit the request window beside the response budget, or rows skip
# generation and their groups fail admission.
for _preset in PRESETS.values():
    assert MAX_PROMPT_TOKENS <= _preset.request_window_tokens - _preset.max_new_tokens, _preset.label


def parse_setting(text: str) -> tuple[str, object, bool]:
    """Split ``[+]dotted.key=value`` into a path, a YAML-parsed value and whether it may add a key."""
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise click.BadParameter(f"a setting must look like dotted.key=value, got {text!r}")
    adds = key.startswith("+")
    return key.lstrip("+"), yaml.safe_load(value), adds


def apply_setting(config: dict, key: str, value: object, *, adds: bool) -> None:
    """Set one dotted key in the assembled config; unknown keys fail unless the override adds."""
    if key == "entrypoint":
        raise click.BadParameter("this launcher is the fully asynchronous loop; entrypoint cannot change")
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
    """Render the RL config: the curriculum config plus the async loop, geometry and telemetry."""
    scale = preset.scale(policy)
    plan = scale.role_plan
    config = yaml.safe_load(rl_config_yaml(scale))
    config["entrypoint"] = "fully_async"
    trainer = config["trainer"]
    trainer.update(
        strategy=recipe.strategy,
        flash_attn=recipe.profile is SkyRLRuntimeProfile.FSDP,
        gradient_checkpointing=True,
        offload_optimizer_during_rollouts=False,
        eval_before_train=preset.eval_interval > 0,
        logger="wandb",
        tracker_commit_each_step=True,
        project_name=WANDB_PROJECT,
        training_metrics=preset.telemetry,
        async_spans=preset.telemetry,
        policy_train_spans=preset.telemetry,
        optimizer_state_metrics=preset.telemetry and recipe.profile is SkyRLRuntimeProfile.MEGATRON,
    )
    trainer["algorithm"].update(use_kl_loss=False, use_kl_in_reward=False, policy_loss_type="regular", use_tis=False)
    trainer["algorithm"]["ratio_diagnostics"] = {
        "pooled": preset.telemetry and recipe.profile is SkyRLRuntimeProfile.MEGATRON
    }
    trainer["algorithm"]["grad_cosine"] = {"enabled": preset.grad_cosine}
    trainer["fully_async"] = {
        "max_staleness_steps": preset.max_staleness_steps,
        "num_parallel_generation_workers": preset.generation_workers,
        "max_buffered_groups": preset.max_buffered_groups or plan.policy_mini_batch_size,
        "pause_mode": preset.pause_mode.value,
        "clear_kv_cache_on_weight_sync": preset.clear_kv_cache_on_weight_sync,
        "first_token_admission": preset.first_token_admission,
    }
    trainer["policy"]["optimizer_config"]["lr"] = recipe.learning_rate
    if recipe.megatron is not None:
        trainer["policy"].pop("fsdp_config")
        trainer["policy"]["megatron_config"] = dict(recipe.megatron)
        trainer["ref"] = {"megatron_config": dict(recipe.megatron)}
    generator = config["generator"]
    generator["enforce_eager"] = False
    # The ratio diagnostics compare the learner with the engine's recorded logprobs.
    generator["sampling_params"]["logprobs"] = 0
    if recipe.engine_init_kwargs:
        generator.setdefault("engine_init_kwargs", {}).update(recipe.engine_init_kwargs)
    config["extra_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for text in settings:
        key, value, adds = parse_setting(text)
        apply_setting(config, key, value, adds=adds)
    return yaml.safe_dump(config, sort_keys=False)


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
            overrides=(*BASE_OVERRIDES, *policy.overrides, *scale.extra_overrides),
        ),
        IrisSkyRLExecution(
            cluster=policy.cluster,
            cluster_config=f"lib/iris/config/{policy.cluster}.yaml",
            cpu=16,
            memory=recipe.task_memory or policy.task_memory,
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
