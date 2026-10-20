# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch joint structure-closure training with closure and BFCL-Parity validation."""

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
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.datasets.structure_closure import ClosureCohort, structure_closure_dataset
from experiments.post_training.structure_closure.recipe import ROLE_PLAN, ClosureSupervision, training_config

MODEL_REPOSITORY = "open-athena/Grug-67B-A2B-BFCL-Protocol-FTPO"
MODEL_REVISION = "d761ea0a94500c2aa24aa4cd71b5a149d962e5c8"
MODEL_URI = (
    "s3://marin-us-east-02a/marin/users/benfeuer/models/online-skill-acquisition-protocol-ftpo/"
    "2026.10.12.2/exports/global_step_4/policy"
)
CLUSTER = "cw-us-east-02a"


def build_run(
    *,
    cohort: ClosureCohort,
    parity_uri: str,
    parity_version: str,
    max_steps: int,
    learning_rate: float,
    supervision: ClosureSupervision,
    candidate_logprobs: int,
    train_batch_size: int,
) -> ArtifactStep:
    """Bind immutable inputs and an explicit 40-H100 role plan to the trainer."""
    name = user_owned_name("models/structure-closure")
    role_plan = replace(ROLE_PLAN, train_batch_size=train_batch_size, policy_mini_batch_size=train_batch_size)
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-initializer"),
        "2026.10.12.1",
        MODEL_URI,
        kind=LevanterCheckpoint,
        config={"hf_repository": MODEL_REPOSITORY, "hf_revision": MODEL_REVISION},
    )
    data = structure_closure_dataset(cohort)
    parity = ArtifactStep.adopt(user_owned_name("datasets/structure-closure-parity"), parity_version, parity_uri)
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(
                training_config(
                    role_plan=role_plan,
                    max_steps=max_steps,
                    eval_batch_size=64,
                    learning_rate=learning_rate,
                    supervision=supervision,
                    candidate_logprobs=candidate_logprobs,
                ),
                sort_keys=False,
            ),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(
                step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
            ),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(
                ArtifactDataSource(data, relative_path="validation.parquet"),
                ArtifactDataSource(parity, relative_path="validation.parquet"),
            ),
            topology=SkyRLTopology(
                num_nodes=role_plan.num_nodes,
                gpus_per_node=role_plan.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=role_plan,
            ),
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
        export_hf=True,
    )


@click.command(help=__doc__)
@click.option(
    "--cohort", type=click.Choice([cohort.value for cohort in ClosureCohort]), default="copy-v1", show_default=True
)
@click.option(
    "--parity-uri", required=True, help="Immutable BFCL-Parity directory with current task_rollout_spec Parquet rows."
)
@click.option("--parity-version", required=True)
@click.option("--max-steps", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), default=4e-6, show_default=True)
@click.option("--candidate-logprobs", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--train-batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option(
    "--supervision",
    type=click.Choice([mode.value for mode in ClosureSupervision]),
    default=ClosureSupervision.CLOSURE.value,
    show_default=True,
)
@rl_build_options
def main(
    cohort: str,
    parity_uri: str,
    parity_version: str,
    max_steps: int,
    learning_rate: float,
    candidate_logprobs: int,
    train_batch_size: int,
    supervision: str,
) -> ArtifactStep:
    return build_run(
        cohort=ClosureCohort(cohort),
        parity_uri=parity_uri,
        parity_version=parity_version,
        max_steps=max_steps,
        learning_rate=learning_rate,
        supervision=ClosureSupervision(supervision),
        candidate_logprobs=candidate_logprobs,
        train_batch_size=train_batch_size,
    )


if __name__ == "__main__":
    main()
