# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

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
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from mergedeep import merge

from experiments.post_training.async_rl import GPUS_PER_NODE, SNOWBALL_RECIPE
from experiments.post_training.curriculum_rl.launch import SNOWBALL_MODEL, SNOWBALL_POLICY
from experiments.post_training.mismatch_probe.launch import ProbeSettings, probe_block

INPUTS = "s3://marin-us-east-02a/marin/users/ahmad/experiments/rl-startup-latency/snowball-inputs-exact-chat-v1"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
FILENAME = "heldout-math-l345.parquet"


@click.command()
@click.option("--cluster", type=click.Choice(("cw-rno2a", "cw-us-east-02a")), default="cw-rno2a")
@click.option("--batch-size", type=int, default=128)
@click.option("--prompt-count", type=int, default=64)
@click.option("--samples-per-prompt", type=int, default=2)
@rl_build_options
def main(cluster, batch_size, prompt_count, samples_per_prompt):
    settings = ProbeSettings(
        seed=17,
        prompt_count=prompt_count,
        samples_per_prompt=samples_per_prompt,
        updates=0,
        keep_fraction=0.5,
        cache_mode="off",
        reuse_probe=None,
        resume_path=None,
        extra_trainer_modes=("router_replay",),
    )
    config = yaml.safe_load(Path(__file__).with_name("startup_snowball_recipe.yaml").read_text())
    config = merge(config, probe_block(settings))
    plan = replace(SNOWBALL_RECIPE.role_plan, train_batch_size=batch_size, policy_mini_batch_size=batch_size)
    name = user_owned_name("checkpoints/startup-refresh/snowball")
    data = ArtifactStep.adopt("documents/startup-refresh/snowball-inputs", "2026.09.14", INPUTS, kind=Artifact)
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            runtime=SkyRLRuntime(profile=SNOWBALL_RECIPE.profile),
            model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=SNOWBALL_POLICY.tokenizer_uri,
                tokenizer_revision=TOKENIZER_REVISION,
                relative_path=SNOWBALL_POLICY.model_relative_path,
            ),
            train_data=(ArtifactDataSource(data, relative_path=FILENAME),),
            validation_data=(ArtifactDataSource(data, relative_path=FILENAME),),
            topology=SkyRLTopology(
                num_nodes=SNOWBALL_RECIPE.num_nodes, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=30),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster=cluster,
            cluster_config=f"lib/iris/config/{cluster}.yaml",
            cpu=16,
            memory=SNOWBALL_RECIPE.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=2,
            wandb_entity=None,
        ),
        export_hf=False,
    )
    return {"snowball-startup": step}


if __name__ == "__main__":
    main()
