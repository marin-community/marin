# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch one Phase 1b single-seed SFT cell from B and fixed teacher rows.

The optimizer, placement, batch size, and 125-step export schedule come from
Phase 1a's M1 recipe. ``task_J`` uses the same 16,384-token response cap as R.
"""

from __future__ import annotations

import os
from dataclasses import replace
from enum import StrEnum

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRetentionPolicy, SkyRLRun, SkyRLSpec, skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.phase1a.launch import CLUSTER, POLICY_CHAT_TEMPLATE, execution
from experiments.post_training.phase1a.matrix import (
    STAGE_C_STEPS,
    WARMUP_STEPS,
    Condition,
    Method,
    RunPlan,
    skyrl_recipe,
    stage_c_schedule,
    topology,
)

LEARNING_RATE = 2e-6
J_SOURCE = "task_J"
R_SOURCE = "task_R"
RESPONSE_CAP = 16_384
PROMPT_CAP = 4_096
OUTPUT_BUCKETS = ("marin-rno2a", "marin-us-east-08a")


def output_env(bucket: str) -> dict[str, str]:
    return {"MARIN_PREFIX": f"s3://{bucket}/marin", "MARIN_TEMP_PREFIX": f"s3://{bucket}"}


class SftCondition(StrEnum):
    R = "theta_R"
    J = "theta_J"
    RJ = "theta_RJ"
    R_PRIME = "theta_R_prime"


_BASE_CONDITIONS = {
    SftCondition.R: Condition.R,
    SftCondition.J: Condition.X,
    SftCondition.RJ: Condition.XR,
    SftCondition.R_PRIME: Condition.R_PRIME,
}
_SOURCES = {
    SftCondition.R: (R_SOURCE,),
    SftCondition.J: (J_SOURCE,),
    SftCondition.RJ: (R_SOURCE, J_SOURCE),
    SftCondition.R_PRIME: (R_SOURCE,),
}


def sft_recipe(
    condition: SftCondition,
    *,
    output_bucket: str,
    resume_from: str | None = None,
    seed: int = 0,
    restore_r_mean: bool = False,
    smoke: bool = False,
) -> tuple[dict, RunPlan]:
    """Render the Phase 1a M1 control with Phase 1b's task names and retry-safe settings."""
    if restore_r_mean and condition is not SftCondition.RJ:
        raise ValueError("R mean restoration applies only to the joint SFT cell")
    control = RunPlan(Method.M1, _BASE_CONDITIONS[condition], seed, LEARNING_RATE)
    recipe = skyrl_recipe(control, {})
    trainer = recipe["trainer"]
    schedule = stage_c_schedule(Method.M1, STAGE_C_STEPS)
    if restore_r_mean or condition is SftCondition.R_PRIME or seed in (1, 2):
        schedule["hf_export_steps"] = [10, 30, 50, 125]
    smoke_steps = WARMUP_STEPS + 1
    if smoke:
        schedule = {
            "checkpoint_steps": [smoke_steps],
            "optimizer_checkpoint_steps": [smoke_steps],
            "hf_export_steps": [],
        }
    trainer.update(
        max_steps=smoke_steps if smoke else STAGE_C_STEPS,
        resume_mode="latest",
        resume_path=None,
        **schedule,
    )
    if restore_r_mean:
        trainer["algorithm"]["sft_restore_source_mean"] = R_SOURCE
    if resume_from is not None:
        trainer["resume_mode"] = "from_path"
        trainer["resume_path"] = resume_from
        trainer["hf_export_steps"] = [STAGE_C_STEPS]
    if smoke:
        trainer["ckpt_interval"] = smoke_steps
    recipe["context_budget"].update(
        request_window_tokens=PROMPT_CAP + RESPONSE_CAP,
        max_new_tokens_per_turn=RESPONSE_CAP,
    )
    recipe["generator"]["max_generate_length_by_data_source"] = {source: RESPONSE_CAP for source in _SOURCES[condition]}
    recipe["extra_env"].update(output_env(output_bucket))
    if "load_checkpoint" in yaml.safe_dump(recipe):
        raise ValueError("fresh Phase 1b SFT recipe unexpectedly sets load_checkpoint")
    return recipe, control


def run_spec(
    condition: SftCondition,
    *,
    model: ArtifactStep[LevanterCheckpoint],
    run_data: ArtifactStep,
    output_bucket: str,
    resume_from: str | None = None,
    seed: int = 0,
    run_label: str | None = None,
    data_relative_path: str | None = None,
    restore_r_mean: bool = False,
    smoke: bool = False,
) -> SkyRLSpec:
    recipe, control = sft_recipe(
        condition,
        output_bucket=output_bucket,
        resume_from=resume_from,
        seed=seed,
        restore_r_mean=restore_r_mean,
        smoke=smoke,
    )
    repo, revision = FAMILY_TOKENIZER_REVISIONS["grug"]
    label = run_label or str(condition)
    name = user_owned_name(f"checkpoints/phase1b/sft-{label}-lr{LEARNING_RATE:g}-seed{seed}")
    plan = topology(control)
    return SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(
            step=model,
            tokenizer_uri=repo,
            tokenizer_revision=revision,
            chat_template=POLICY_CHAT_TEMPLATE,
        ),
        train_data=(
            ArtifactDataSource(run_data, relative_path=data_relative_path or f"{condition}/seed{seed}/m1.parquet"),
        ),
        validation_data=(),
        topology=plan,
        retention=SkyRLRetentionPolicy(
            resume_checkpoint_count=len(recipe["trainer"]["checkpoint_steps"]),
            temporary_storage_ttl_days=90,
        ),
        seed=seed,
    )


@click.command(help=__doc__)
@click.option("--condition", type=click.Choice([str(value) for value in SftCondition]), required=True)
@click.option("--b-uri", required=True, help="Stage 0 B HF export in RNO2A.")
@click.option("--b-version", required=True)
@click.option("--run-data-uri", required=True, help="Phase 1b fixed-completion parquet root in RNO2A.")
@click.option("--run-data-version", required=True)
@click.option("--output-bucket", type=click.Choice(OUTPUT_BUCKETS), required=True)
@click.option("--resume-from", help="Existing optimizer checkpoint to read when moving a run to a new output root.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--run-label", help="Distinct artifact name for a control run.")
@click.option("--data-relative-path", help="Parquet path relative to the run-data artifact root.")
@click.option("--restore-r-mean", is_flag=True, help="Restore R's solo token-mean coefficient in the joint SFT loss.")
@click.option("--smoke", is_flag=True, help="Run through warmup with production model geometry.")
@rl_build_options
def main(
    condition: str,
    b_uri: str,
    b_version: str,
    run_data_uri: str,
    run_data_version: str,
    output_bucket: str,
    resume_from: str | None,
    seed: int,
    run_label: str | None,
    data_relative_path: str | None,
    restore_r_mean: bool,
    smoke: bool,
) -> list[ArtifactStep[SkyRLRun]]:
    if not b_uri.startswith("s3://marin-rno2a/") or not run_data_uri.startswith("s3://marin-rno2a/"):
        raise click.UsageError("B and run data must both be in marin-rno2a")
    for name, expected in output_env(output_bucket).items():
        if os.environ.get(name) != expected:
            raise click.UsageError(f"{name} must be {expected} for Phase 1b SFT")
    model = ArtifactStep.adopt(user_owned_name("models/phase1b-snowball-b"), b_version, b_uri, kind=LevanterCheckpoint)
    run_data = ArtifactStep.adopt(user_owned_name("documents/phase1b-run-data"), run_data_version, run_data_uri)
    if seed not in (0, 1, 2):
        raise click.UsageError("Phase 1b rate controls use seeds 0, 1, or 2")
    if smoke and not restore_r_mean:
        raise click.UsageError("the one-node smoke is for the restored-R objective")
    if resume_from is not None and not resume_from.startswith("s3://marin-rno2a/"):
        raise click.UsageError("the recovery checkpoint must be in marin-rno2a")
    spec = run_spec(
        SftCondition(condition),
        model=model,
        run_data=run_data,
        output_bucket=output_bucket,
        resume_from=resume_from,
        seed=seed,
        run_label=run_label,
        data_relative_path=data_relative_path,
        restore_r_mean=restore_r_mean,
        smoke=smoke,
    )
    regional_execution = replace(
        execution(CLUSTER),
        priority="production",
        max_retries=0 if smoke else 1,
        required_env_vars=tuple(output_env(output_bucket)),
    )
    return [skyrl_step(spec, regional_execution, export_hf=not smoke)]


if __name__ == "__main__":
    main()
