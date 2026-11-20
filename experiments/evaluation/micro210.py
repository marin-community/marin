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
    ArtifactModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
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
    Micro210Evaluation,
    Micro210Model,
    micro210_evaluation,
    runtime_policy_config,
)
from experiments.post_training.long_loop.launch_validation import validate_before_allocation
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

PREPARED_INPUTS = (
    "gs://marin-us-east1/evaluation-policies/micro210/2026.11.20-together/"
    "ba099cbc869af0dd1c14f14586557e38a812d6b956442e8b1f406b36cc42c7a0"
)

STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
SERVING_GENERATOR_FIELDS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "vllm_v1_disable_multiproc",
    "vllm_attention_backend",
}
SERVING_ENGINE_FIELDS = {"compilation_config", "model_loader_extra_config", "moe_backend"}


@dataclass(frozen=True)
class Micro210ServingProfile:
    """Model-specific serving resources; evaluation protocol fields stay frozen."""

    gpu_variant: str
    role_plan: SkyRLRolePlan
    execution: IrisSkyRLExecution
    generator: dict
    engine_init_kwargs: dict
    cuda_allocator_config: str


def serving_config(frozen: dict, profile: Micro210ServingProfile) -> dict:
    config = runtime_policy_config(frozen).config
    unknown = set(profile.generator) - SERVING_GENERATOR_FIELDS
    unknown_engine = set(profile.engine_init_kwargs) - SERVING_ENGINE_FIELDS
    if unknown or unknown_engine:
        raise ValueError(f"Serving profile cannot override protocol fields: {sorted(unknown | unknown_engine)}")
    if profile.role_plan.n_samples_per_prompt != frozen["evaluation"]["role_plan"]["n_samples_per_prompt"]:
        raise ValueError("Serving profile cannot change protocol samples per prompt")
    compilation = profile.engine_init_kwargs["compilation_config"]
    if compilation["mode"] != 3 or compilation["cudagraph_mode"] != "PIECEWISE":
        raise ValueError("Serving profiles require compilation mode 3 and PIECEWISE CUDA graphs")
    _materialize_role_plan_config(config, profile.role_plan)
    config["generator"].update(profile.generator)
    config["generator"]["enforce_eager"] = False
    config["generator"]["engine_init_kwargs"] = {
        **config["generator"].get("engine_init_kwargs", {}),
        **profile.engine_init_kwargs,
    }
    config["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"] = profile.cuda_allocator_config
    return config


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str
    serving_profile: Micro210ServingProfile | None = None
    model_manifest_filename: str = MODEL_MANIFEST_FILENAME


def judge_runtime_config(config: dict, judge: JudgeConfig) -> dict:
    """Bind all runtime judge routes from the typed policy without changing other settings."""
    config = deepcopy(config)
    config["environment"]["task_sessions"]["remote_judge"].update(base_url=judge.base_url, api_key_env="JUDGE_API_KEY")
    config["extra_env"]["JUDGE_BASE_URL"] = judge.base_url
    config["terminal_bench"]["installed_agent"]["judge_model"] = judge.model
    return config


def validate_inputs(
    evaluation: Micro210Evaluation, inputs: str, *, model_manifest_filename: str = MODEL_MANIFEST_FILENAME
) -> None:
    """Reject changed prepared tasks or export metadata before GPU allocation."""
    policy = evaluation.policy()
    table = (StoragePath(inputs) / policy["evaluation"]["data_relative_path"]).read_bytes()
    if hashlib.sha256(table).hexdigest() != policy["prepared_rollouts_sha256"]:
        raise ValueError("Canonical Micro210 prepared task bytes changed")
    manifest = (StoragePath(evaluation.model.uri) / model_manifest_filename).read_bytes()
    if MODEL_IDENTITY_PREFIX + hashlib.sha256(manifest).hexdigest() != evaluation.model.identity:
        raise ValueError("Micro210 model export manifest differs from the model binding")
    tokenizer = AutoTokenizer.from_pretrained(
        evaluation.model.tokenizer_uri, revision=evaluation.model.tokenizer_revision, trust_remote_code=False
    )
    generator = policy["evaluation"]["config"]["generator"]
    for sampling in ("sampling_params", "eval_sampling_params"):
        for token_id in generator[sampling]["stop_token_ids"]:
            if tokenizer.convert_ids_to_tokens(token_id) != STOP_TOKEN_CONTENT[token_id]:
                raise ValueError(f"Canonical Micro210 tokenizer has incompatible stop token ID: {token_id}")


def run_micro210(config: Micro210Run) -> SkyRLRun:
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS, model_manifest_filename=config.model_manifest_filename)
    validate_before_allocation(config.skyrl.launch_config_yaml)
    (StoragePath(config.evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps(
            {
                **asdict(config.evaluation.record()),
                "runtime_field_renames": runtime_policy_config(config.evaluation.policy()).field_renames,
                **(
                    {
                        "serving_profile": asdict(config.serving_profile),
                        "model_manifest_filename": config.model_manifest_filename,
                    }
                    if config.serving_profile
                    else {}
                ),
            },
            indent=2,
        )
    )
    policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(JudgeConfig, policy["judge"])
    api_key = resolve_secret_spec((judge.credential,)).value
    return run_skyrl(config.skyrl, env_vars={"JUDGE_API_KEY": api_key, "OPENAI_API_KEY": api_key})


def micro210_step(model: Micro210Model, *, name: str, version: str, **overrides) -> ArtifactStep[SkyRLRun]:
    """Build the canonical evaluation graph; reject all policy override fields."""
    binding = micro210_evaluation(model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>", **overrides)
    binding.policy()
    model_artifact = ArtifactStep.adopt(
        f"{name}/model", version, model.uri, kind=LevanterCheckpoint, config=asdict(model)
    )
    return micro210_artifact_step(
        ArtifactHfModel(model_artifact, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
        name=name,
        version=version,
        model_binding=model,
    )


def micro210_artifact_step(
    model: ArtifactModel,
    *,
    name: str,
    version: str,
    serving_profile: Micro210ServingProfile | None = None,
    model_binding: Micro210Model | None = None,
    model_manifest_filename: str = MODEL_MANIFEST_FILENAME,
) -> ArtifactStep[SkyRLRun]:
    """Bind a future committed model to the same frozen evaluator as a static export."""
    fingerprint_model = Micro210Model(
        "s3://unresolved/terminal-export",
        MODEL_IDENTITY_PREFIX + "0" * 64,
        "marin-community/marin-tokenizer-tools",
        "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    )
    binding = micro210_evaluation(fingerprint_model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>")
    frozen = binding.policy()
    policy = frozen["evaluation"]
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
    config = runtime_policy_config(frozen).config
    gpu_variant = policy["gpu_variant"]
    if serving_profile:
        roles = serving_profile.role_plan
        execution = serving_profile.execution
        gpu_variant = serving_profile.gpu_variant
        config = serving_config(frozen, serving_profile)
    config = judge_runtime_config(config, draccus.decode(JudgeConfig, policy["judge"]))
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=model,
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=policy["data_relative_path"]),),
            topology=SkyRLTopology(roles.num_nodes, roles.policy_num_gpus_per_node, gpu_variant, roles),
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
        document["skyrl"]["generator"]["trajectory_retention"]["run_id"] = binding.run_id
        rendered = yaml.safe_dump(document)
        if model_binding is not None:
            bound_model = model_binding
            if not context.is_fingerprint and run.model.uri != bound_model.uri:
                raise ValueError("Adopted model resolved to a different URI")
        elif context.is_fingerprint:
            bound_model = fingerprint_model
        else:
            manifest = (StoragePath(run.model.uri) / model_manifest_filename).read_bytes()
            bound_model = Micro210Model(
                run.model.uri,
                MODEL_IDENTITY_PREFIX + hashlib.sha256(manifest).hexdigest(),
                run.model.tokenizer_uri,
                run.model.tokenizer_revision,
            )
        return Micro210Run(
            replace(run, launch_config_yaml=rendered),
            replace(binding, model=bound_model, output_uri=context.output_path),
            hashlib.sha256(rendered.encode()).hexdigest(),
            serving_profile,
            model_manifest_filename,
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
