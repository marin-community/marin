# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace

import click
import draccus
import yaml
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.inference.iris import remote_inference
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210_policy import (
    MODEL_IDENTITY_PREFIX,
    Micro210Evaluation,
    Micro210Model,
    micro210_evaluation,
)
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

PREPARED_INPUTS = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/" "documents/mochi-exp3-final-eval-inputs/2026.11.18.61"
)


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str


def validate_inputs(evaluation: Micro210Evaluation, inputs: str) -> None:
    """Reject changed prepared tasks or export metadata before GPU allocation."""
    policy = evaluation.policy()
    table = (StoragePath(inputs) / policy["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(table).hexdigest() != policy["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    manifest = (StoragePath(evaluation.model.uri) / MODEL_MANIFEST_FILENAME).read_bytes()
    if MODEL_IDENTITY_PREFIX + hashlib.sha256(manifest).hexdigest() != evaluation.model.identity:
        raise ValueError("Micro210 model export manifest differs from the model binding")


def run_micro210(config: Micro210Run) -> SkyRLRun:
    counterpart_api_key = os.environ["TOGETHER_API_KEY"]
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS)
    (StoragePath(config.evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps(asdict(config.evaluation.record()), indent=2)
    )
    policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(ModelConfig, policy["judge"])
    launch = yaml.safe_load(config.skyrl.launch_config_yaml)
    inference = inference_config_for_model(
        judge,
        AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type=policy["gpu_variant"],
            gpu_count=policy["judge_gpu_count"],
            target_cluster=None,
        ),
        env_vars={},
        priority=priority_band_value(launch["iris"]["priority"]),
        capability_origin=policy["capability_origin"],
        api_model=judge.location,
    )
    with remote_inference(inference) as served:
        launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] = served.model.endpoint.base_url
        for section in (launch["skyrl"]["extra_env"], launch["runtime"]["task_env"]):
            section["JUDGE_BASE_URL"] = served.model.endpoint.base_url
        return run_skyrl(
            replace(config.skyrl, launch_config_yaml=yaml.safe_dump(launch)),
            env_vars={
                "JUDGE_API_KEY": served.model.endpoint.api_key or "EMPTY",
                "OPENAI_API_KEY": counterpart_api_key,
            },
        )


def micro210_step(model: Micro210Model, *, name: str, version: str, **overrides) -> ArtifactStep[SkyRLRun]:
    """Build the canonical evaluation graph; reject all policy override fields."""
    binding = micro210_evaluation(model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>", **overrides)
    frozen = binding.policy()
    policy = frozen["evaluation"]
    model_artifact = ArtifactStep.adopt(
        f"{name}/model", version, model.uri, kind=LevanterCheckpoint, config=asdict(model)
    )
    inputs = ArtifactStep.adopt(
        "evaluation/micro210/prepared",
        "2026.11.18",
        PREPARED_INPUTS,
        config={"sha256": frozen["prepared_rollouts_sha256"]},
    )
    roles = draccus.decode(SkyRLRolePlan, policy["role_plan"])
    execution = IrisSkyRLExecution(
        **{**policy["execution"], "required_env_vars": tuple(policy["execution"]["required_env_vars"])},
        coordinator_timeout_hours=policy["execution"]["job_timeout_seconds"] // 3600 + 1,
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(policy["config"]),
            model=ArtifactHfModel(model_artifact, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=policy["data_relative_path"]),),
            topology=SkyRLTopology(roles.num_nodes, roles.policy_num_gpus_per_node, policy["gpu_variant"], roles),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=policy["seed"],
        ),
        execution,
        export_hf=False,
    )
    builder = step.build_config

    def build(context: StepContext) -> Micro210Run:
        run = builder(context)
        document = yaml.safe_load(run.launch_config_yaml)
        document["ray"].update(policy["ray_timeouts"])
        rendered = yaml.safe_dump(document)
        return Micro210Run(
            replace(run, launch_config_yaml=rendered),
            replace(binding, output_uri=context.output_path),
            hashlib.sha256(rendered.encode()).hexdigest(),
        )

    return replace(step, run=run_micro210, build_config=build)


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-manifest-sha256", required=True)
@click.option("--tokenizer-uri", required=True)
@click.option("--tokenizer-revision", required=True)
@click.option("--name", required=True)
@rl_build_options
def main(
    model_uri: str, model_manifest_sha256: str, tokenizer_uri: str, tokenizer_revision: str, name: str
) -> ArtifactStep:
    configure_coreweave_s3()
    model = Micro210Model(model_uri, MODEL_IDENTITY_PREFIX + model_manifest_sha256, tokenizer_uri, tokenizer_revision)
    return micro210_step(model, name=name, version=resolve_version(name, None))


if __name__ == "__main__":
    main()
