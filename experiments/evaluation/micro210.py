# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
import draccus
import yaml
from iris.cluster.config import load_config
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform, serving_geometry
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.inference.config import RemoteInferenceConfig
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint
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
from rigging.secrets import SecretSpec, resolve_secret_spec
from rigging.timing import Duration, ExponentialBackoff
from transformers import AutoTokenizer

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
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/" "documents/mochi-exp3-final-eval-inputs/2026.11.18.61"
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
SERVING_NETWORK_FIELDS = {"NCCL_NET", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS"}


@dataclass(frozen=True)
class Micro210ServingProfile:
    """Model-specific serving resources; evaluation protocol fields stay frozen."""

    gpu_variant: str
    role_plan: SkyRLRolePlan
    execution: IrisSkyRLExecution
    generator: dict
    engine_init_kwargs: dict
    cuda_allocator_config: str
    judge_gpu_count: int
    nccl_buffer_size_bytes: int
    network_env: dict[str, str]
    judge_gpu_variant: str
    judge_target_cluster: str
    judge_endpoint_receipt: str


def serving_config(frozen: dict, profile: Micro210ServingProfile) -> dict:
    config = runtime_policy_config(frozen).config
    unknown = set(profile.generator) - SERVING_GENERATOR_FIELDS
    unknown_engine = set(profile.engine_init_kwargs) - SERVING_ENGINE_FIELDS
    unknown_network = set(profile.network_env) - SERVING_NETWORK_FIELDS
    if unknown or unknown_engine or unknown_network:
        raise ValueError(
            f"Serving profile cannot override protocol fields: {sorted(unknown | unknown_engine | unknown_network)}"
        )
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
    config["extra_env"].update(profile.network_env)
    config["trainer"].setdefault("policy", {})["nccl_buffer_size_bytes"] = profile.nccl_buffer_size_bytes
    return config


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str
    counterpart_api_key: SecretSpec
    serving_profile: Micro210ServingProfile | None = None
    model_manifest_filename: str = MODEL_MANIFEST_FILENAME


@dataclass(frozen=True)
class Micro210JudgeEndpoint:
    canonical_spec_sha256: str
    judge_sha256: str
    tensor_parallel_size: int
    endpoint: OpenAIEndpoint
    jobs: tuple[str, ...]


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


def micro210_judge_inference(config: Micro210Run) -> RemoteInferenceConfig:
    """Lower the frozen judge for both CPU preflight and runtime allocation."""
    policy = config.evaluation.policy()["evaluation"]
    judge = draccus.decode(ModelConfig, policy["judge"])
    gpu_variant = policy["gpu_variant"]
    judge_gpu_count = policy["judge_gpu_count"]
    if config.serving_profile:
        gpu_variant = config.serving_profile.judge_gpu_variant
        judge_gpu_count = config.serving_profile.judge_gpu_count
        judge = replace(judge, resource_hint=replace(judge.resource_hint, gpu={gpu_variant: judge_gpu_count}))
    accelerator = AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type=gpu_variant,
        gpu_count=judge_gpu_count,
        target_cluster=None,
    )
    serving_geometry(judge.serve, accelerator)
    priority = config.serving_profile.execution.priority if config.serving_profile else policy["execution"]["priority"]
    inference = inference_config_for_model(
        judge,
        accelerator,
        env_vars={},
        priority=priority_band_value(priority),
        capability_origin=policy["capability_origin"],
        api_model=judge.location,
    )
    if config.serving_profile:
        cluster = load_config(
            Path(__file__).resolve().parents[2]
            / "lib/iris/config"
            / f"{config.serving_profile.judge_target_cluster}.yaml"
        )
        if cluster.platform.coreweave is None:
            raise ValueError("The Mochi judge requires a regional CoreWeave controller")
        inference = replace(
            inference,
            iris=replace(
                inference.iris,
                worker_resources=replace(
                    inference.iris.worker_resources, regions=[cluster.platform.coreweave.region], target_cluster=None
                ),
            ),
        )
    return inference


def micro210_judge_sha256(config: Micro210Run) -> str:
    return hashlib.sha256(
        json.dumps(config.evaluation.policy()["evaluation"]["judge"], sort_keys=True).encode()
    ).hexdigest()


@contextmanager
def micro210_judge_endpoint(config: Micro210Run, inference: RemoteInferenceConfig) -> Iterator[OpenAIEndpoint]:
    """Use the co-hosted judge or a receipt verified against the frozen judge."""
    if config.serving_profile is None:
        with remote_inference(inference) as served:
            yield served.model.endpoint
        return
    receipt = StoragePath(config.serving_profile.judge_endpoint_receipt)
    ExponentialBackoff(initial=1, maximum=30).wait_until_or_raise(
        receipt.exists,
        Duration.from_seconds(config.evaluation.policy()["evaluation"]["execution"]["job_timeout_seconds"]),
        "Remote Micro210 judge did not publish its endpoint receipt",
    )
    endpoint = draccus.decode(Micro210JudgeEndpoint, json.loads(receipt.read_bytes()))
    if (
        endpoint.canonical_spec_sha256 != config.evaluation.record().canonical_spec_sha256
        or endpoint.judge_sha256 != micro210_judge_sha256(config)
        or endpoint.tensor_parallel_size != inference.model.tensor_parallel_size
        or endpoint.endpoint.model != inference.model.api_model
    ):
        raise ValueError("Remote Micro210 judge receipt differs from the frozen judge")
    yield endpoint.endpoint


def run_micro210(config: Micro210Run) -> SkyRLRun:
    counterpart_api_key = resolve_secret_spec(config.counterpart_api_key).value
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS, model_manifest_filename=config.model_manifest_filename)
    validate_before_allocation(config.skyrl.launch_config_yaml)
    inference = micro210_judge_inference(config)
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
    launch = yaml.safe_load(config.skyrl.launch_config_yaml)
    with micro210_judge_endpoint(config, inference) as endpoint:
        launch["skyrl"]["environment"]["task_sessions"]["remote_judge"]["base_url"] = endpoint.base_url
        for section in (launch["skyrl"]["extra_env"], launch["runtime"]["task_env"]):
            section["JUDGE_BASE_URL"] = endpoint.base_url
        return run_skyrl(
            replace(config.skyrl, launch_config_yaml=yaml.safe_dump(launch)),
            env_vars={
                "JUDGE_API_KEY": endpoint.api_key or "EMPTY",
                "OPENAI_API_KEY": counterpart_api_key,
            },
        )


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
            ("env:TOGETHER_API_KEY",),
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
