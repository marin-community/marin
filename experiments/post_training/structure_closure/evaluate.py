# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate registered closure cohorts or versioned data without a learner."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLRetentionPolicy,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.datasets.structure_closure import ClosureCohort, structure_closure_dataset, tool_prompt_probe_dataset
from experiments.post_training.structure_closure.launch import EXECUTION, MODEL_REPOSITORY, MODEL_REVISION, MODEL_URI
from experiments.post_training.structure_closure.recipe import ROLE_PLAN, ClosureSupervision, training_config

EVAL_ROLE_PLAN = replace(ROLE_PLAN, colocate_all=True, policy_num_nodes=1, reference_num_nodes=1)


def evaluation_config() -> dict:
    config = training_config(
        role_plan=EVAL_ROLE_PLAN,
        max_steps=1,
        eval_batch_size=64,
        learning_rate=4e-6,
        supervision=ClosureSupervision.CLOSURE,
        candidate_logprobs=32,
    )
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


def initializer_model() -> ArtifactHfModel:
    """Bind the campaign initializer and its pinned tokenizer."""
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-initializer"),
        "2026.10.12.1",
        MODEL_URI,
        kind=LevanterCheckpoint,
        config={"hf_repository": MODEL_REPOSITORY, "hf_revision": MODEL_REVISION},
    )
    return ArtifactHfModel(
        step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
    )


def exported_model(*, checkpoint_uri: str, checkpoint_version: str, manifest_identity: str) -> ArtifactHfModel:
    """Bind an immutable campaign export and the initializer's pinned tokenizer."""
    model = ArtifactStep.adopt(
        user_owned_name("models/structure-closure-export"),
        checkpoint_version,
        checkpoint_uri,
        kind=LevanterCheckpoint,
        config={"manifest_identity": manifest_identity},
    )
    return ArtifactHfModel(
        step=model, tokenizer_uri=MODEL_REPOSITORY, tokenizer_revision=MODEL_REVISION, relative_path=""
    )


def build_evaluation(*, name: str, data: ArtifactStep, model: ArtifactHfModel) -> ArtifactStep:
    """Evaluate an immutable closure cohort with an explicit model artifact."""
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(evaluation_config(), sort_keys=False),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=model,
            train_data=(),
            validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=EVAL_ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=42,
        ),
        EXECUTION,
        export_hf=False,
    )


@click.command(help=__doc__)
@click.option("--data", required=True, help="Closure cohort name, tool-prompt-probe, or immutable URI@version.")
@click.option("--data-name", default=None, help="Required artifact name for URI datasets; preserve the original name.")
@click.option("--model", required=True, help="base or immutable checkpoint URI@version.")
@click.option("--manifest-identity", default=None, help="Required for checkpoint models.")
@click.option("--name", required=True, help="User-owned evaluation artifact name.")
@rl_build_options
def main(data: str, data_name: str | None, model: str, manifest_identity: str | None, name: str) -> ArtifactStep:
    if data_name is not None and (data in ClosureCohort or data == "tool-prompt-probe"):
        raise click.BadParameter("Registered cohorts already define their artifact name", param_hint="--data-name")
    if data in ClosureCohort:
        dataset = structure_closure_dataset(ClosureCohort(data))
    elif data == "tool-prompt-probe":
        dataset = tool_prompt_probe_dataset()
    else:
        uri, separator, version = data.rpartition("@")
        if not separator or not uri or not version or not data_name:
            raise click.BadParameter("Expected a cohort name or URI@version with --data-name", param_hint="--data")
        dataset = ArtifactStep.adopt(user_owned_name(data_name), version, uri)
    if model == "base":
        if manifest_identity is not None:
            raise click.BadParameter("Only checkpoint models accept a manifest identity")
        selected_model = initializer_model()
    else:
        uri, separator, version = model.rpartition("@")
        if not separator or not uri or not version or not manifest_identity:
            raise click.BadParameter("Expected checkpoint URI@version and --manifest-identity", param_hint="--model")
        selected_model = exported_model(
            checkpoint_uri=uri, checkpoint_version=version, manifest_identity=manifest_identity
        )
    return build_evaluation(name=user_owned_name(name), data=dataset, model=selected_model)


if __name__ == "__main__":
    main()
