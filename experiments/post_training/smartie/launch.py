# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Smartie training from an explicit saved launch configuration.

Files in configs/ are immutable records of submitted runs. For a new runtime or
training change, copy a config to a fresh run version and update its runtime pin;
do not rewrite the historical config. Pass the same version through --version.
"""

from pathlib import Path

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.curriculum_rl.launch import HfSnapshotConfig, mirror_hf_model


def build_configured_run(config_path: Path) -> ArtifactStep:
    """Bind the saved launch settings without campaign-specific training overrides."""
    source = yaml.safe_load(config_path.read_text())
    recipe = {**source["skyrl"], "extra_env": source["runtime"]["task_env"]}
    trainer = recipe["trainer"]
    generator = recipe["generator"]
    placement = trainer["placement"]
    allocation = source["iris"]["allocation"]
    runtime = SkyRLRuntime(profile=SkyRLRuntimeProfile(source["runtime"]["profile"]))
    if runtime.commit != source["runtime"]["launcher_commit"]:
        raise ValueError("Saved runtime revision disagrees with the checkout pin")
    name = user_owned_name("models/snowball-smartie")
    version = resolve_version(name, None)
    if source["run"]["id"] != f"{name}-{version}":
        raise ValueError("Saved run identity disagrees with --version")
    model_input = source["inputs"]["model"]
    model_name, model_version = model_input["identity"].split("@", 1)
    model = ArtifactStep(
        name=model_name,
        version=model_version.split(":", 1)[0],
        artifact_type=LevanterCheckpoint,
        run=remote(mirror_hf_model, resources=ResourceConfig.with_cpu(cpu=8, ram="64g", disk="512g")),
        build_config=lambda ctx: HfSnapshotConfig(
            output_path=ctx.output_path,
            repo_id=model_input["tokenizer_uri"],
            revision=model_input["tokenizer_revision"],
        ),
        override_path=model_input["uri"].removesuffix("/hf"),
    )

    def data_source(item: dict) -> ArtifactDataSource:
        data_name, data_version = item["identity"].split("@", 1)
        if item["kind"] != "directory":
            raise ValueError("Smartie requires directory-backed parquet inputs")
        return ArtifactDataSource(
            ArtifactStep.adopt(data_name, data_version.split(":", 1)[0], item["uri"]),
            relative_path=item["relative_path"],
        )

    role_plan = SkyRLRolePlan(
        colocate_all=placement["colocate_all"],
        colocate_policy_ref=placement["colocate_policy_ref"],
        policy_num_nodes=placement["policy_num_nodes"],
        policy_num_gpus_per_node=placement["policy_num_gpus_per_node"],
        reference_num_nodes=placement["ref_num_nodes"],
        num_inference_engines=generator["num_inference_engines"],
        inference_engine_tensor_parallel_size=generator["inference_engine_tensor_parallel_size"],
        inference_engine_pipeline_parallel_size=generator["inference_engine_pipeline_parallel_size"],
        inference_engine_data_parallel_size=generator["inference_engine_data_parallel_size"],
        inference_engine_expert_parallel_size=generator["inference_engine_expert_parallel_size"],
        train_batch_size=trainer["train_batch_size"],
        policy_mini_batch_size=trainer["policy_mini_batch_size"],
        micro_train_batch_size_per_gpu=trainer["micro_train_batch_size_per_gpu"],
        n_samples_per_prompt=generator["n_samples_per_prompt"],
    )
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            runtime=runtime,
            model=ArtifactHfModel(
                model, model_input["tokenizer_uri"], model_input["tokenizer_revision"], relative_path="hf"
            ),
            train_data=tuple(data_source(item) for item in source["inputs"]["train_data"]),
            validation_data=tuple(data_source(item) for item in source["inputs"]["validation_data"]),
            topology=SkyRLTopology(
                allocation["num_nodes"], allocation["gpus_per_node"], allocation["gpu_variant"], role_plan
            ),
            retention=SkyRLRetentionPolicy(source["artifacts"]["resume_checkpoint_count"]),
            seed=source["run"]["seed"],
        ),
        IrisSkyRLExecution(
            cluster=source["iris"]["cluster"],
            cluster_config=source["iris"]["cluster_config"],
            cpu=allocation["cpu"],
            memory=allocation["memory"],
            disk=allocation["disk"],
            priority=source["iris"]["priority"],
            max_retries=source["iris"]["max_retries"],
            target_cluster=source["iris"]["target_cluster"],
            parent_cluster_config=source["iris"]["parent_cluster_config"],
            coordinator_timeout_hours=72,
            job_timeout_seconds=source["iris"]["timeout"],
            required_env_vars=("DAYTONA_RL_API_KEY", "DAYTONA_EVAL_API_KEY"),
        ),
        export_hf=source["run"]["export_hf"],
    )


@click.command(help=__doc__)
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@rl_build_options
def main(config_path: Path) -> ArtifactStep:
    return build_configured_run(config_path)


if __name__ == "__main__":
    main()
