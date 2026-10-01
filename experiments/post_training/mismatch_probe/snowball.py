# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare pretrained Snowball routing and measure fixed-weight RL step time."""

from dataclasses import asdict, dataclass, replace
from enum import StrEnum

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from mergedeep import merge

from experiments.post_training.async_rl import GPUS_PER_NODE, SNOWBALL_RECIPE
from experiments.post_training.curriculum_rl.launch import SNOWBALL_MODEL, SNOWBALL_POLICY, SNOWBALL_SFT_EXPORT_URI
from experiments.post_training.mismatch_probe.launch import NUMERICS_MODES, REPLAY_MODES, ProbeSettings, probe_block

TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
HELDOUT_DATA_URI = (
    "s3://marin-us-east-02a/marin/users/ahmad/documents/math-eval-pool/"
    "1.0.0-candidate1/batteries/e9-heldout-v1/snowball"
)
HELDOUT_FILENAME = "heldout-math-l345.parquet"


# A Snowball export cut to its first layers runs on this many GPUs of one node: two pipeline stages that the colocated
# vLLM engines share as two data-parallel ranks with expert parallelism across them.
SLICE_GPUS = 2
SLICE_HOST_MEMORY = "256GB"
SLICE_GPU_MEMORY_UTILIZATION = 0.35


@dataclass(frozen=True)
class SnowballSlice:
    """A Snowball export that keeps the first decoder layers (same layer shapes), with its artifact version."""

    model_uri: str
    version: str


class Campaign(StrEnum):
    MISMATCH = "mismatch"
    STEP_TIME = "step-time"
    # One real update at the production learning rate: how far a single step moves the policy.
    DRIFT = "drift"


class Routing(StrEnum):
    NATIVE = "native"
    REPLAY = "router_replay"
    FILTERED = "router_replay_filtered"


def snowball_recipe(
    settings: ProbeSettings,
    *,
    campaign: Campaign,
    routing: Routing,
    request_window_tokens: int,
    response_tokens: int,
    capture_layers: tuple[int, ...] = (),
    timing_modes: tuple[str, ...] = (),
    extra_modes: tuple[str, ...] = (),
    train_numerics: str | None = None,
    batch_invariant: bool = False,
    decode_invariant: bool = False,
    model_slice: bool = False,
) -> str:
    replay = campaign is Campaign.MISMATCH or routing is not Routing.NATIVE
    # A slice keeps each layer's experts on its pipeline stage (no expert parallelism) and shares its GPUs with vLLM.
    megatron = (
        replace(SNOWBALL_RECIPE.megatron, expert_model_parallel_size=1) if model_slice else SNOWBALL_RECIPE.megatron
    )
    if campaign is Campaign.MISMATCH:
        modes = (*NUMERICS_MODES, *extra_modes)
    elif campaign is Campaign.DRIFT:
        modes = ("native_again", *extra_modes)
    else:
        modes = (*(REPLAY_MODES if replay else ()), *extra_modes)
    probe = probe_block(
        settings,
        router_replay=replay,
        trainer_modes=modes,
        capture_layers=capture_layers,
        timing_modes=timing_modes,
        train_numerics=train_numerics,
    )
    probe["generator"]["sampling_params"]["seed"] = settings.seed
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": request_window_tokens,
            "max_new_tokens_per_turn": response_tokens,
            "max_turns": 1,
        },
        "environment": {"env_class": "aime"},
        "trainer": {
            "strategy": SNOWBALL_RECIPE.strategy,
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": False,
            "epochs": 50,
            "max_steps": max(settings.updates),
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "ckpt_interval": -1,
            "hf_save_interval": -1,
            "eval_before_train": False,
            "eval_interval": -1,
            "resume_mode": "none",
            "logger": "console",
            "project_name": "marin-mismatch-probe",
            "algorithm": {
                "advantage_estimator": "grpo",
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "batch_invariant": batch_invariant,
            },
            "policy": {
                # Timing and mismatch campaigns hold the weights; the drift campaign trains as production does.
                **({} if campaign is Campaign.DRIFT else {"grug_query_bias_update_mode": "frozen"}),
                "optimizer_config": {
                    "optimizer": "AdamW",
                    "lr": SNOWBALL_RECIPE.learning_rate if campaign is Campaign.DRIFT else 0.0,
                    "weight_decay": SNOWBALL_RECIPE.weight_decay,
                    "max_grad_norm": SNOWBALL_RECIPE.max_grad_norm,
                },
                "megatron_config": {
                    **asdict(megatron),
                    "moe_router_replay_keep_fraction": settings.keep_fraction if routing is Routing.FILTERED else None,
                },
            },
            "ref": {"megatron_config": asdict(megatron)},
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "decode_invariant": decode_invariant,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "gpu_memory_utilization": SLICE_GPU_MEMORY_UTILIZATION if model_slice else 0.75,
            "chat_template": {
                "source": "file",
                "name_or_path": f"{SNOWBALL_SFT_EXPORT_URI.rstrip('/')}/chat_template.jinja",
            },
            "engine_init_kwargs": dict(SNOWBALL_RECIPE.engine_init_kwargs),
            "trajectory_retention": {
                "enabled": campaign is Campaign.STEP_TIME,
                "sample_fraction": 1.0,
                "required": True,
            },
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": []},
    }
    return yaml.safe_dump(merge({}, config, probe), sort_keys=False)


def build_spec(
    *,
    settings: ProbeSettings,
    campaign: Campaign,
    routing: Routing,
    data_uri: str,
    data_version: str,
    train_filename: str,
    validation_filename: str,
    cluster: str,
    batch_size: int,
    request_window_tokens: int,
    response_tokens: int,
    capture_layers: tuple[int, ...] = (),
    timing_modes: tuple[str, ...] = (),
    extra_modes: tuple[str, ...] = (),
    train_numerics: str | None = None,
    batch_invariant: bool = False,
    decode_invariant: bool = False,
    model_slice: SnowballSlice | None = None,
) -> tuple[SkyRLSpec, IrisSkyRLExecution]:
    """``model_slice`` runs a Snowball export cut to its first layers on ``SLICE_GPUS`` GPUs of one node."""
    if campaign is Campaign.MISMATCH and (settings.updates != (0,) or routing is not Routing.NATIVE):
        raise ValueError("the mismatch campaign scores starting weights in all modes without training updates")
    plan = replace(SNOWBALL_RECIPE.role_plan, train_batch_size=batch_size, policy_mini_batch_size=batch_size)
    model, num_nodes, gpus_per_node, host_memory = SNOWBALL_MODEL, SNOWBALL_RECIPE.num_nodes, GPUS_PER_NODE, None
    if model_slice is not None:
        model = ArtifactStep.adopt(
            "models/snowball-layer-slice", model_slice.version, model_slice.model_uri, kind=LevanterCheckpoint
        )
        plan = replace(
            plan,
            colocate_all=True,
            policy_num_nodes=1,
            policy_num_gpus_per_node=SLICE_GPUS,
            inference_engine_data_parallel_size=SLICE_GPUS,
            inference_engine_expert_parallel_size=SLICE_GPUS,
        )
        num_nodes, gpus_per_node, host_memory = 1, SLICE_GPUS, SLICE_HOST_MEMORY
    data: ArtifactStep[Artifact] = ArtifactStep.adopt("documents/mismatch-probe/snowball-inputs", data_version, data_uri)
    model_name = "snowball" if model_slice is None else "snowball-slice"
    name = user_owned_name(f"checkpoints/mismatch-probe/{model_name}/{campaign}/{routing}/seed-{settings.seed}")
    return (
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=snowball_recipe(
                settings,
                campaign=campaign,
                routing=routing,
                request_window_tokens=request_window_tokens,
                response_tokens=response_tokens,
                capture_layers=capture_layers,
                timing_modes=timing_modes,
                extra_modes=extra_modes,
                train_numerics=train_numerics,
                batch_invariant=batch_invariant,
                decode_invariant=decode_invariant,
                model_slice=model_slice is not None,
            ),
            runtime=SkyRLRuntime(profile=SNOWBALL_RECIPE.profile),
            model=ArtifactHfModel(
                step=model,
                tokenizer_uri=SNOWBALL_POLICY.tokenizer_uri,
                tokenizer_revision=TOKENIZER_REVISION,
                relative_path=SNOWBALL_POLICY.model_relative_path,
            ),
            train_data=(ArtifactDataSource(data, relative_path=train_filename),),
            validation_data=(ArtifactDataSource(data, relative_path=validation_filename),),
            topology=SkyRLTopology(
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                gpu_variant="H100",
                role_plan=plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=30),
            seed=settings.seed,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=16,
            memory=host_memory or SNOWBALL_RECIPE.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity="marin-community",
        ),
    )


def build_run(**kwargs) -> ArtifactStep[SkyRLRun]:
    spec, execution = build_spec(**kwargs)
    return skyrl_step(spec, execution, export_hf=False)


@click.command(help=__doc__)
@click.option(
    "--campaign",
    type=click.Choice([mode.value for mode in Campaign]),
    default=Campaign.MISMATCH.value,
    show_default=True,
)
@click.option(
    "--routing", type=click.Choice([mode.value for mode in Routing]), default=Routing.NATIVE.value, show_default=True
)
@click.option("--cluster", type=click.Choice(("cw-us-east-02a", "cw-rno2a")), default=SNOWBALL_POLICY.cluster)
@click.option("--data-uri", default=HELDOUT_DATA_URI, show_default=True)
@click.option("--data-version", default="2026.09.14", show_default=True)
@click.option("--train-filename", default=HELDOUT_FILENAME, show_default=True)
@click.option("--validation-filename", default=HELDOUT_FILENAME, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=17, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--prompt-count", type=click.IntRange(min=1), default=16, show_default=True)
@click.option("--samples-per-prompt", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--prompt-offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--steps", type=click.IntRange(min=2), default=2, show_default=True)
@click.option("--request-window-tokens", type=click.IntRange(min=1), default=2048, show_default=True)
@click.option("--response-tokens", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--keep-fraction", type=click.FloatRange(min=0, max=1), default=0.5, show_default=True)
@click.option("--rescore-prefix-cache", "cache_mode", type=click.Choice(("off", "on", "both")), default="off")
@click.option("--reuse-probe")
@click.option(
    "--capture-layer",
    "capture_layers",
    multiple=True,
    type=click.IntRange(min=0),
    help="Record this decoder layer's regions on the trainer, and its attention inputs in the first re-read prefix.",
)
@click.option("--timing-mode", "timing_modes", multiple=True, help="Time forward + backward for this mode.")
@click.option("--extra-mode", "extra_modes", multiple=True, help="Score this trainer mode as well (a candidate).")
@click.option("--train-numerics", help="MarinSkyRL numerics set the policy trains and scores under by default.")
@click.option("--batch-invariant/--no-batch-invariant", default=False, help="vLLM and trainer batch-invariant kernels.")
@click.option(
    "--decode-invariant/--no-decode-invariant",
    default=False,
    help="Decode-invariant vLLM engines: a token's bytes are the same in decode and prefill steps of any composition.",
)
@click.option("--drift-updates", type=click.IntRange(min=1), default=1, show_default=True, help="Drift updates scored.")
@click.option("--slice-model-uri", help="A Snowball export cut to its first layers, run on two GPUs of one node.")
@click.option("--slice-model-version", help="Artifact version of --slice-model-uri.")
@rl_build_options
def main(
    campaign: str,
    routing: str,
    cluster: str,
    data_uri: str,
    data_version: str,
    train_filename: str,
    validation_filename: str,
    seed: int,
    batch_size: int,
    prompt_count: int,
    samples_per_prompt: int,
    prompt_offset: int,
    steps: int,
    request_window_tokens: int,
    response_tokens: int,
    keep_fraction: float,
    cache_mode: str,
    reuse_probe: str | None,
    capture_layers: tuple[int, ...],
    timing_modes: tuple[str, ...],
    extra_modes: tuple[str, ...],
    train_numerics: str | None,
    batch_invariant: bool,
    decode_invariant: bool,
    drift_updates: int,
    slice_model_uri: str | None,
    slice_model_version: str | None,
) -> ArtifactStep[SkyRLRun]:
    selected = Campaign(campaign)
    if (slice_model_uri is None) != (slice_model_version is None):
        raise click.UsageError("--slice-model-uri and --slice-model-version go together")
    return build_run(
        settings=ProbeSettings(
            seed,
            prompt_count,
            samples_per_prompt,
            {Campaign.MISMATCH: (0,), Campaign.DRIFT: tuple(range(drift_updates + 1))}.get(selected, (0, steps)),
            keep_fraction,
            cache_mode,
            reuse_probe,
            None,
            prompt_offset,
        ),
        campaign=selected,
        routing=Routing(routing),
        data_uri=data_uri,
        data_version=data_version,
        train_filename=train_filename,
        validation_filename=validation_filename,
        cluster=cluster,
        batch_size=batch_size,
        request_window_tokens=request_window_tokens,
        response_tokens=response_tokens,
        capture_layers=capture_layers,
        timing_modes=timing_modes,
        extra_modes=extra_modes,
        train_numerics=train_numerics,
        batch_invariant=batch_invariant,
        decode_invariant=decode_invariant,
        model_slice=None if slice_model_uri is None else SnowballSlice(slice_model_uri, slice_model_version),
    )


if __name__ == "__main__":
    main()
