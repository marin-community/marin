# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch joint structure-closure training with closure and BFCL-Parity validation."""

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

from experiments.post_training.structure_closure.recipe import ROLE_PLAN, training_config

MODEL_REPOSITORY = "open-athena/Grug-67B-A2B-BFCL-Protocol-FTPO"
MODEL_REVISION = "d761ea0a94500c2aa24aa4cd71b5a149d962e5c8"
MODEL_URI = (
    "s3://marin-us-east-02a/marin/users/benfeuer/models/online-skill-acquisition-protocol-ftpo/"
    "2026.10.12.2/exports/global_step_4/policy"
)
DATA_URI = "s3://marin-us-east-02a/marin/users/benfeuer/structure-closure/data/closure-copy-tasks-v1"


def build_run(*, parity_uri: str, parity_version: str, max_steps: int, learning_rate: float) -> ArtifactStep:
    """Bind immutable inputs and an explicit 40-H100 role plan to the trainer."""
    name = user_owned_name("models/structure-closure")
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-initializer"),
        "2026.10.12.1",
        MODEL_URI,
        kind=LevanterCheckpoint,
        config={"hf_repository": MODEL_REPOSITORY, "hf_revision": MODEL_REVISION},
    )
    data = ArtifactStep.adopt(
        user_owned_name("datasets/structure-closure"),
        "2026.10.12.1",
        DATA_URI,
        config={
            "train_sha256": "83f5c3007f065ae6d50ec3892e942c644327f204f3333962b5420748478386d9",
            "validation_sha256": "5813e728fab3ffc17fb290c86260ae29719df744acc2db33b5aff86854a3be1c",
        },
    )
    parity = ArtifactStep.adopt(user_owned_name("datasets/structure-closure-parity"), parity_version, parity_uri)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(
                training_config(max_steps=max_steps, eval_batch_size=64, learning_rate=learning_rate), sort_keys=False
            ),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path="."
            ),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(
                ArtifactDataSource(data, relative_path="validation.parquet"),
                ArtifactDataSource(parity, relative_path="validation.parquet"),
            ),
            topology=SkyRLTopology(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=42,
        ),
        IrisSkyRLExecution(
            cluster="cw-us-east-02a",
            cluster_config="lib/iris/config/cw-us-east-02a.yaml",
            cpu=48,
            memory="1800GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster="cw-us-east-02a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity=None,
        ),
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option("--parity-uri", required=True, help="Immutable native BFCL-Parity dataset directory.")
@click.option("--parity-version", required=True)
@click.option("--max-steps", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), default=4e-6, show_default=True)
@rl_build_options
def main(parity_uri: str, parity_version: str, max_steps: int, learning_rate: float) -> ArtifactStep:
    return build_run(
        parity_uri=parity_uri, parity_version=parity_version, max_steps=max_steps, learning_rate=learning_rate
    )


if __name__ == "__main__":
    main()
