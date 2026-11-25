# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import click
import draccus
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
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
from rigging.secrets import resolve_secret_spec
from transformers import AutoTokenizer

from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210_policy import (
    MODEL_IDENTITY_PREFIX,
    PREPARED_COMMIT,
    PREPARED_REPO,
    Micro210Evaluation,
    Micro210Model,
    micro210_evaluation,
)
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

# The prepared table re-hosts benchmark items that must not be public; tasks read it
# with the standard HF_TOKEN.
PREPARED_INPUTS = f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}"

# Relocation needs a new adoption identity even though the prepared bytes are unchanged.
PREPARED_INPUTS_VERSION = "2026.11.20.1"

STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str


def judge_runtime_config(config: dict, judge: JudgeConfig) -> dict:
    """Bind all runtime judge routes from the typed policy without changing other settings."""
    config = deepcopy(config)
    config["environment"]["task_sessions"]["remote_judge"].update(base_url=judge.base_url, api_key_env="JUDGE_API_KEY")
    config["extra_env"]["JUDGE_BASE_URL"] = judge.base_url
    config["terminal_bench"]["installed_agent"]["judge_model"] = judge.model
    return config


def validate_inputs(evaluation: Micro210Evaluation, inputs: str) -> None:
    """Reject changed prepared tasks or export metadata before GPU allocation."""
    protocol = evaluation.policy()
    table = (StoragePath(inputs) / protocol["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(table).hexdigest() != protocol["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    manifest = (StoragePath(evaluation.model.uri) / MODEL_MANIFEST_FILENAME).read_bytes()
    if MODEL_IDENTITY_PREFIX + hashlib.sha256(manifest).hexdigest() != evaluation.model.identity:
        raise ValueError("Micro210 model export manifest differs from the model binding")
    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.model.tokenizer_uri, revision=evaluation.model.tokenizer_revision, trust_remote_code=False
    )
    generator = protocol["evaluation"]["config"]["generator"]
    for sampling in ("sampling_params", "eval_sampling_params"):
        for token_id in generator[sampling]["stop_token_ids"]:
            if tokenizer.convert_ids_to_tokens(token_id) != STOP_TOKEN_CONTENT[token_id]:
                raise ValueError(f"Canonical Micro210 tokenizer has incompatible stop token ID: {token_id}")


def run_micro210(config: Micro210Run) -> SkyRLRun:
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS)
    (StoragePath(config.evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps(asdict(config.evaluation.record()), indent=2)
    )
    evaluation_policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(JudgeConfig, evaluation_policy["judge"])
    api_key = resolve_secret_spec((judge.credential,)).value
    return run_skyrl(config.skyrl, env_vars={"JUDGE_API_KEY": api_key, "OPENAI_API_KEY": api_key})


def micro210_step(model: Micro210Model, *, name: str, version: str) -> ArtifactStep[SkyRLRun]:
    """Build the canonical evaluation graph for a model and run identity."""
    binding = micro210_evaluation(model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>")
    frozen = binding.policy()
    evaluation_policy = frozen["evaluation"]
    model_artifact = ArtifactStep.adopt(
        f"{name}/model", version, model.uri, kind=LevanterCheckpoint, config=asdict(model)
    )
    inputs = ArtifactStep.adopt(
        "evaluation/micro210/prepared",
        PREPARED_INPUTS_VERSION,
        PREPARED_INPUTS,
        config={"sha256": frozen["prepared_rollouts_sha256"]},
    )
    roles = draccus.decode(SkyRLRolePlan, evaluation_policy["role_plan"])
    execution = IrisSkyRLExecution(
        **{
            **evaluation_policy["execution"],
            "required_env_vars": tuple(evaluation_policy["execution"]["required_env_vars"]),
        },
        coordinator_timeout_hours=evaluation_policy["execution"]["job_timeout_seconds"] // 3600 + 1,
    )
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(
                judge_runtime_config(
                    evaluation_policy["config"], draccus.decode(JudgeConfig, evaluation_policy["judge"])
                )
            ),
            model=ArtifactHfModel(model_artifact, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=evaluation_policy["data_relative_path"]),),
            topology=SkyRLTopology(
                roles.num_nodes, roles.policy_num_gpus_per_node, evaluation_policy["gpu_variant"], roles
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=evaluation_policy["seed"],
        ),
        execution,
        export_hf=False,
    )
    builder = step.build_config

    def build(context: StepContext) -> Micro210Run:
        run = builder(context)
        document = yaml.safe_load(run.launch_config_yaml)
        document["ray"].update(evaluation_policy["ray_timeouts"])
        document["skyrl"]["generator"]["trajectory_retention"]["run_id"] = binding.run_id
        rendered = yaml.safe_dump(document)
        return Micro210Run(
            replace(run, launch_config_yaml=rendered),
            replace(binding, output_uri=context.output_path),
            hashlib.sha256(rendered.encode()).hexdigest(),
        )

    configured = replace(step, run=run_micro210, build_config=build, expected_fingerprint=None)
    return replace(configured, expected_fingerprint=configured.fingerprint())


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
