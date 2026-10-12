# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the explicit-closure prompt cohort without allocating a learner."""

from dataclasses import replace

import click
import yaml
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
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.structure_closure.launch import CLUSTER, MODEL_REPOSITORY, MODEL_REVISION, MODEL_URI
from experiments.post_training.structure_closure.recipe import ROLE_PLAN, training_config

EVAL_ROLE_PLAN = replace(ROLE_PLAN, colocate_all=True, policy_num_nodes=1)
DATA_URI = "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-explicit-tasks-v3"


def evaluation_config() -> dict:
    config = training_config(max_steps=1, eval_batch_size=64, learning_rate=4e-6)
    config["entrypoint"] = "generate"
    config["trainer"]["hf_save_interval"] = -1
    # No weights are synchronized in evaluation; expert-block requires a learner layout.
    config["generator"]["weight_sync_transport"] = "broadcast"
    config["trainer"]["placement"]["policy_strict_spread_pg"] = False
    # The launcher validates this unused training geometry against the allocation.
    for role in ("policy", "ref"):
        geometry = config["trainer"][role]["megatron_config"]
        geometry["pipeline_model_parallel_size"] = 1
        geometry["expert_model_parallel_size"] = EVAL_ROLE_PLAN.policy_num_gpus_per_node
    return config


def build_evaluation(*, name: str, data: ArtifactStep) -> ArtifactStep:
    """Evaluate an immutable closure cohort with the policy initializer."""
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-initializer"),
        "2026.10.12.1",
        MODEL_URI,
        kind=LevanterCheckpoint,
        config={"hf_repository": MODEL_REPOSITORY, "hf_revision": MODEL_REVISION},
    )
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(evaluation_config(), sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
            ),
            train_data=(),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=EVAL_ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=42,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=48,
            memory="1800GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity=None,
        ),
        export_hf=False,
    )


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep:
    data = ArtifactStep.adopt(
        user_owned_name("datasets/structure-closure-explicit"),
        "2026.10.12.3",
        DATA_URI,
        config={"validation_sha256": "f64e9ae5724ab081c3e119ee39efc850f12eeaf39f0c5a05ceaf017454a7b011"},
    )
    return build_evaluation(name=user_owned_name("evals/structure-closure-explicit"), data=data)


if __name__ == "__main__":
    main()
