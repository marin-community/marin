# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add frozen-token mismatch probes to existing synchronous Megatron recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IrisSkyRLExecution, SkyRLRun, SkyRLSpec, skyrl_step
from marin.training.training import LevanterCheckpoint
from mergedeep import merge

from experiments.post_training.iceball_micro import iceball_rl_execution, iceball_rl_spec

REPLAY_MODES = ("router_replay", "router_replay_filtered")


@dataclass(frozen=True)
class ProbeSettings:
    seed: int
    prompt_count: int
    samples_per_prompt: int
    updates: int
    keep_fraction: float
    cache_mode: str
    reuse_probe: str | None
    resume_path: str | None
    extra_trainer_modes: tuple[str, ...] = ()


def probe_block(settings: ProbeSettings) -> dict:
    """Render collection and scoring controls for a synchronous Megatron recipe."""
    block = {
        "trainer": {
            "mismatch_probe": {
                "enabled": True,
                "prompts": {"count": settings.prompt_count, "samples_per_prompt": settings.samples_per_prompt},
                "seed": settings.seed,
                "archive_uri": None,
                "reuse_probe": settings.reuse_probe,
                "updates": settings.updates,
                "extra_trainer_modes": list(settings.extra_trainer_modes),
                "filtered_replay": {"keep_fraction": settings.keep_fraction},
                "rescore_prefix_cache": settings.cache_mode,
            },
        },
        "generator": {
            "enable_prefix_caching": settings.cache_mode != "off",
            "engine_init_kwargs": {
                "logprobs_mode": "processed_logprobs",
                "generation_config": "vllm",
            },
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "logprobs": 0,
            },
        },
    }
    if settings.updates == 0:
        block["trainer"].update(ckpt_interval=-1, hf_save_interval=-1)
    if settings.extra_trainer_modes:
        block["trainer"]["policy"] = {"megatron_config": {"moe_router_replay": True}}
        block["generator"]["engine_init_kwargs"]["enable_return_routed_experts"] = True
    if settings.resume_path is not None:
        block["trainer"].update(
            resume_mode="from_path", resume_path=settings.resume_path, reset_global_step_on_resume=False
        )
    return block


# Exact-token GSM8K transport renders Iceball's prompts with this template.
ICEBALL_PROBE_RECIPE = {"generator": {"chat_template": {"source": "name", "name_or_path": "qwen3_with_thinking"}}}


def probe_step(spec: SkyRLSpec, execution: IrisSkyRLExecution, settings: ProbeSettings) -> ArtifactStep[SkyRLRun]:
    """Launch a probe using the model, data, topology and execution of an RL recipe."""
    name = user_owned_name(f"checkpoints/mismatch-probe/{spec.name.rsplit('/', 1)[-1]}")
    probe_spec = replace(
        spec,
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(merge({}, yaml.safe_load(spec.config_yaml), probe_block(settings)), sort_keys=False),
        retention=replace(spec.retention, temporary_storage_ttl_days=30),
        seed=settings.seed,
    )
    return skyrl_step(probe_spec, execution, export_hf=False)


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Existing Iceball SFT artifact root containing its HF exports.")
@click.option("--data-uri", required=True, help="Existing Iceball GSM8K artifact root.")
@click.option("--input-version", required=True, help="Version identifying the adopted model and data artifacts.")
@click.option("--resume-path")
@click.option("--reuse-probe")
@click.option("--seed", type=int, default=17, show_default=True)
@click.option("--prompt-count", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--samples-per-prompt", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--updates", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--keep-fraction", type=click.FloatRange(min=0, max=1, min_open=True), default=0.5, show_default=True)
@click.option("--rescore-prefix-cache", "cache_mode", type=click.Choice(("off", "on", "both")), default="off")
@click.option("--extra-trainer-mode", "extra_trainer_modes", multiple=True, type=click.Choice(REPLAY_MODES))
@rl_build_options
def main(
    model_uri: str,
    data_uri: str,
    input_version: str,
    resume_path: str | None,
    reuse_probe: str | None,
    seed: int,
    prompt_count: int,
    samples_per_prompt: int,
    updates: int,
    keep_fraction: float,
    cache_mode: str,
    extra_trainer_modes: tuple[str, ...],
) -> ArtifactStep[SkyRLRun]:
    model = ArtifactStep.adopt(
        user_owned_name("checkpoints/iceball-micro-sft"), input_version, model_uri, kind=LevanterCheckpoint
    )
    data: ArtifactStep[Artifact] = ArtifactStep.adopt(
        user_owned_name("documents/iceball-micro-gsm8k-skyrl"), input_version, data_uri
    )
    settings = ProbeSettings(
        seed=seed,
        prompt_count=prompt_count,
        samples_per_prompt=samples_per_prompt,
        updates=updates,
        keep_fraction=keep_fraction,
        cache_mode=cache_mode,
        reuse_probe=reuse_probe,
        resume_path=resume_path,
        extra_trainer_modes=extra_trainer_modes,
    )
    spec = iceball_rl_spec(model, data)
    spec = replace(
        spec, config_yaml=yaml.safe_dump(merge(yaml.safe_load(spec.config_yaml), ICEBALL_PROBE_RECIPE), sort_keys=False)
    )
    return probe_step(spec, iceball_rl_execution(), settings)


if __name__ == "__main__":
    main()
