# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or run Mochi RL at 65K context on 48 trainer and 12 inference GPUs."""

from dataclasses import asdict, replace
from pathlib import Path

import click
import yaml
from iris.cluster.client.job_info import get_job_info
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
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
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import marin_prefix
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections

RECIPE_PATH = Path(__file__).with_name("recipe.yaml")
SEED = 20261020
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=12,
    policy_num_gpus_per_node=4,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=12,
    inference_engine_expert_parallel_size=12,
    train_batch_size=2,
    policy_mini_batch_size=2,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)
TOPOLOGY = SkyRLTopology(
    num_nodes=ROLE_PLAN.num_nodes,
    gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
    gpu_variant="GB200",
    role_plan=ROLE_PLAN,
)
MEGATRON = MegatronGeometry(1, 12, 4, 4, 1)
EXECUTION = IrisSkyRLExecution(
    cluster="cw-us-east-08a",
    cluster_config="lib/iris/config/cw-us-east-08a.yaml",
    cpu=120,
    memory="900GB",
    disk="2TB",
    priority="production",
    max_retries=0,
    target_cluster="cw-us-east-08a",
    parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
    coordinator_timeout_hours=72,
    job_timeout_seconds=18000,
    wandb_entity="marin-community",
    required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY", "MARIN_PREFIX"),
    container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
)
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=3)


def build_run(
    *,
    name: str,
    version: str,
    model: ArtifactHfModel,
    train_data: tuple[ArtifactDataSource, ...],
    max_steps: int,
    resume: ArtifactStep[Artifact] | None = None,
    recipe_path: Path = RECIPE_PATH,
    topology: SkyRLTopology = TOPOLOGY,
    megatron: MegatronGeometry = MEGATRON,
    execution: IrisSkyRLExecution = EXECUTION,
    retention: SkyRLRetentionPolicy = RETENTION,
) -> ArtifactStep[SkyRLRun]:
    """Bind model, mixed tasks and optional native state to one RL artifact."""
    recipe = SkyRLRecipe.from_document(yaml.safe_load(recipe_path.read_text())).merge(
        RecipePatch(
            trainer=sections.Trainer(
                # Leave enough epochs for small task sets to reach the global-step target.
                epochs=max_steps,
                max_steps=max_steps,
                micro_forward_batch_size_per_gpu=topology.role_plan.micro_train_batch_size_per_gpu,
                policy=sections.Policy(megatron_config=sections.PolicyMegatronConfig(**asdict(megatron))),
                ref=sections.Ref(megatron_config=sections.RefMegatronConfig(**asdict(megatron))),
            )
        )
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(recipe.to_skyrl(), sort_keys=False),
            model=model,
            train_data=train_data,
            validation_data=(),
            topology=topology,
            retention=retention,
            seed=SEED,
        ),
        execution,
        export_hf=False,
    )
    if resume is None:
        return step
    build_config = step.build_config

    def resume_config(ctx: StepContext) -> SkyRLRunConfig:
        config = build_config(ctx)
        launch = yaml.safe_load(config.launch_config_yaml)
        launch["skyrl"] = (
            SkyRLRecipe.from_document(launch["skyrl"])
            .merge(RecipePatch(trainer=sections.Trainer(resume_mode="from_path", resume_path=ctx.artifact_path(resume))))
            .to_skyrl()
        )
        return replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))

    return replace(step, build_config=resume_config, deps=(*step.deps, resume))


@click.command(help=__doc__)
@click.option("--model-uri", required=True, help="Completed schema-3 BF16 HF export with its portable model manifest.")
@click.option("--tokenizer-uri", help="Pinned tokenizer artifact; defaults to the model export.")
@click.option("--tokenizer-revision", required=True)
@click.option(
    "--data-uri", multiple=True, required=True, help="Mixed-task artifact directory. Repeat for several inputs."
)
@click.option("--data-file", default="train.parquet", show_default=True)
@click.option("--input-version", required=True, help="Immutable version identifying the adopted input set.")
@click.option(
    "--name", default="checkpoints/mochi-rl-65k", show_default=True, help="Output name within your user namespace."
)
@click.option("--max-steps", type=click.IntRange(min=1), required=True, help="Stop at this absolute optimizer step.")
@click.option("--resume-uri", help="Exact native RL checkpoint directory, including model and optimizer state.")
@click.option(
    "--recipe", "recipe_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=RECIPE_PATH
)
@click.option("--checkpoint-count", type=click.IntRange(1, 10), default=RETENTION.resume_checkpoint_count)
@click.option("--ttl-days", type=click.IntRange(min=1), default=RETENTION.temporary_storage_ttl_days)
@click.option(
    "--config-output", type=click.Path(dir_okay=False, path_type=Path), help="Write the source launch YAML locally."
)
@rl_build_options
def main(
    model_uri: str,
    tokenizer_uri: str | None,
    tokenizer_revision: str,
    data_uri: tuple[str, ...],
    data_file: str,
    input_version: str,
    name: str,
    max_steps: int,
    resume_uri: str | None,
    recipe_path: Path,
    checkpoint_count: int,
    ttl_days: int,
    config_output: Path | None,
) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(name)
    model = ArtifactHfModel(
        step=ArtifactStep.adopt(f"{name}/inputs/model", input_version, source=model_uri, kind=LevanterCheckpoint),
        relative_path="",
        tokenizer_uri=tokenizer_uri or model_uri,
        tokenizer_revision=tokenizer_revision,
    )
    train_data = tuple(
        ArtifactDataSource(
            ArtifactStep.adopt(f"{name}/inputs/data-{index}", input_version, source=uri), relative_path=data_file
        )
        for index, uri in enumerate(data_uri)
    )
    resume = (
        ArtifactStep.adopt(f"{name}/inputs/resume", input_version, source=resume_uri) if resume_uri is not None else None
    )
    step = build_run(
        name=name,
        version=resolve_version(name, None),
        model=model,
        train_data=train_data,
        max_steps=max_steps,
        resume=resume,
        recipe_path=recipe_path,
        retention=SkyRLRetentionPolicy(resume_checkpoint_count=checkpoint_count, temporary_storage_ttl_days=ttl_days),
    )
    if config_output is not None and get_job_info() is None:
        ctx = StepContext.for_run(step.path(), marin_prefix(), runtime_args=step.runtime_args, deps=step.deps)
        config_output.write_text(step.build_config(ctx).launch_config_yaml)
    return step


if __name__ == "__main__":
    main()
