# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add frozen-token mismatch probes to existing synchronous Megatron recipes."""

from __future__ import annotations

from dataclasses import replace
from typing import get_args

import click
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import IrisSkyRLExecution, SkyRLRun, SkyRLSpec, skyrl_step
from marin.skyrl_recipe import (
    ChatTemplate,
    Environment,
    Generator,
    Gsm8k,
    MismatchProbe,
    Policy,
    PolicyMegatronConfig,
    RecipePatch,
    SamplingParams,
    SkyrlGym,
    Trainer,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.iceball_micro import iceball_rl_execution, iceball_rl_spec

REPLAY_MODES = get_args(get_args(MismatchProbe.model_fields["extra_trainer_modes"].annotation)[0])


def probe_part(settings: MismatchProbe, *, resume_path: str | None = None) -> RecipePatch:
    """Collection, scoring and replay controls for a synchronous Megatron recipe."""
    part = RecipePatch(
        environment=Environment(skyrl_gym=SkyrlGym(gsm8k=Gsm8k(structured_chat=True))),
        trainer=Trainer(
            mismatch_probe=settings,
        ),
        generator=Generator(
            require_exact_chat_transport=True,
            enable_prefix_caching=settings.rescore_prefix_cache != "off",
            engine_init_kwargs={"logprobs_mode": "processed_logprobs", "generation_config": "vllm"},
            sampling_params=SamplingParams(
                temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, logprobs=0
            ),
        ),
    )
    if settings.updates == 0:
        part = part.merge(RecipePatch(trainer=Trainer(ckpt_interval=-1, hf_save_interval=-1)))
    if settings.extra_trainer_modes:
        part = part.merge(
            RecipePatch(
                trainer=Trainer(policy=Policy(megatron_config=PolicyMegatronConfig(moe_router_replay=True))),
                generator=Generator(engine_init_kwargs={"enable_return_routed_experts": True}),
            )
        )
    if resume_path is not None:
        part = part.merge(
            RecipePatch(
                trainer=Trainer(resume_mode="from_path", resume_path=resume_path, reset_global_step_on_resume=False)
            )
        )
    return part


# Exact-token GSM8K transport renders Iceball's prompts with this template.
ICEBALL_PROBE_RECIPE = RecipePatch(
    generator=Generator(chat_template=ChatTemplate(source="name", name_or_path="qwen3_with_thinking"))
)


def probe_step(
    spec: SkyRLSpec, execution: IrisSkyRLExecution, settings: MismatchProbe, *, resume_path: str | None = None
) -> ArtifactStep[SkyRLRun]:
    """Launch a probe using the model, data, topology and execution of an RL recipe."""
    name = user_owned_name(f"checkpoints/mismatch-probe/{spec.name.rsplit('/', 1)[-1]}")
    probe_spec = replace(
        spec,
        name=name,
        version=resolve_version(name, None),
        recipe=spec.recipe.merge(probe_part(settings, resume_path=resume_path)),
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
    settings = MismatchProbe.from_document(
        {
            "enabled": True,
            "prompts": {"count": prompt_count, "samples_per_prompt": samples_per_prompt},
            "seed": seed,
            "archive_uri": None,
            "reuse_probe": reuse_probe,
            "updates": updates,
            "extra_trainer_modes": extra_trainer_modes,
            "filtered_replay": {"keep_fraction": keep_fraction},
            "rescore_prefix_cache": cache_mode,
        }
    )
    spec = iceball_rl_spec(model, data)
    spec = replace(spec, recipe=spec.recipe.merge(ICEBALL_PROBE_RECIPE))
    return probe_step(spec, iceball_rl_execution(), settings, resume_path=resume_path)


if __name__ == "__main__":
    main()
