# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen Micro210 evaluation with only a model binding and run identity."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import click
import draccus
import httpx
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    ArtifactModel,
    IrisSkyRLExecution,
    SkyRLGenerationPlan,
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
from rigging.secrets import require_secret
from transformers import AutoTokenizer

from experiments.evaluation import micro210_policy
from experiments.evaluation.judge_config import JudgeConfig
from experiments.evaluation.micro210_policy import (
    MODEL_IDENTITY_PREFIX,
    PREPARED_COMMIT,
    PREPARED_REPO,
    Micro210Evaluation,
    Micro210Model,
    Micro210Route,
    micro210_evaluation,
    read_micro210_policy,
    runtime_policy_config,
)
from experiments.evaluation.micro210_subset import Micro210Selection, Micro210Subset, prepare_subset, validate_subset
from experiments.post_training.long_loop.launch_validation import validate_before_allocation
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

# The prepared table re-hosts benchmark items that must not be public; tasks read it
# with the standard HF_TOKEN.
PREPARED_INPUTS = f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}"

# Relocation needs a new adoption identity even though the prepared bytes are unchanged.
PREPARED_INPUTS_VERSION = "2026.11.23.2"

STOP_TOKEN_CONTENT = {128001: "<|end_of_text|>", 128009: "<|eot_id|>"}
JUDGE_API_KEY = "JUDGE_API_KEY"
SERVING_GENERATOR_FIELDS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "enable_prefix_caching",
    "vllm_v1_disable_multiproc",
    "vllm_attention_backend",
}
SERVING_ENGINE_FIELDS = {"compilation_config", "load_format", "model_loader_extra_config", "moe_backend"}
SERVING_NCCL_ENV_FIELDS = {"NCCL_NET", "NCCL_IB_DISABLE", "NCCL_IB_HCA"}
RUNAI_LOADER_FIELDS = frozenset({"distributed", "concurrency", "memory_limit"})


@dataclass(frozen=True)
class Micro210ServingProfile:
    """Model-specific serving resources; evaluation protocol fields stay frozen."""

    gpu_variant: str
    role_plan: SkyRLRolePlan | SkyRLGenerationPlan
    execution: IrisSkyRLExecution
    generator: dict
    engine_init_kwargs: dict
    cuda_allocator_config: str
    nccl_env: dict[str, str]
    nccl_buffer_size_bytes: int
    model_manifest_filename: str


class ServingProfileName(StrEnum):
    POLICY = "policy"
    SNOWBALL_H100_EAST02A = "snowball-h100-cw-us-east-02a"
    MOCHI_GB200 = "mochi-gb200-cw-us-east-08a"


def registered_serving_profile(name: ServingProfileName) -> Micro210ServingProfile | None:
    """Select a reviewed serving binding; the default uses the frozen policy."""
    if name is ServingProfileName.POLICY:
        return None
    profiles = yaml.safe_load(Path(__file__).with_name("micro210-serving-profiles.yaml").read_text())
    profile = profiles[name.value]
    execution = profile["execution"]
    plan = profile["role_plan"]
    role_plan = (
        draccus.decode(SkyRLGenerationPlan, plan)
        if "policy_num_nodes" not in plan
        else draccus.decode(SkyRLRolePlan, plan)
    )
    return Micro210ServingProfile(
        **{
            **profile,
            "role_plan": role_plan,
            "execution": IrisSkyRLExecution(
                **{
                    **execution,
                    "required_env_vars": tuple(execution["required_env_vars"]),
                    "excluded_nvlink_domains": tuple(execution.get("excluded_nvlink_domains", ())),
                }
            ),
        }
    )


def serving_config(frozen: dict, profile: Micro210ServingProfile) -> dict:
    config = runtime_policy_config(frozen).config
    unknown = set(profile.generator) - SERVING_GENERATOR_FIELDS
    unknown_engine = set(profile.engine_init_kwargs) - SERVING_ENGINE_FIELDS
    unknown_network = set(profile.nccl_env) - SERVING_NCCL_ENV_FIELDS
    if unknown or unknown_engine or unknown_network:
        raise ValueError(
            f"Serving profile cannot override protocol fields: {sorted(unknown | unknown_engine | unknown_network)}"
        )
    if profile.role_plan.n_samples_per_prompt != frozen["evaluation"]["role_plan"]["n_samples_per_prompt"]:
        raise ValueError("Serving profile cannot change protocol samples per prompt")
    compilation = profile.engine_init_kwargs["compilation_config"]
    if compilation["mode"] != 3 or compilation["cudagraph_mode"] != "PIECEWISE":
        raise ValueError("Serving profiles require compilation mode 3 and PIECEWISE CUDA graphs")
    loader_options = profile.engine_init_kwargs.get("model_loader_extra_config", {})
    if RUNAI_LOADER_FIELDS.intersection(loader_options) and profile.engine_init_kwargs.get("load_format") not in {
        "runai_streamer",
        "runai_streamer_sharded",
    }:
        raise ValueError("RunAI loader options require an explicit RunAI load_format")
    _materialize_role_plan_config(config, profile.role_plan)
    config["generator"].update(profile.generator)
    config["generator"]["enforce_eager"] = False
    config["generator"]["engine_init_kwargs"] = {
        **config["generator"].get("engine_init_kwargs", {}),
        **profile.engine_init_kwargs,
    }
    config["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"] = profile.cuda_allocator_config
    config["extra_env"].update(profile.nccl_env)
    config["trainer"].setdefault("policy", {})["nccl_buffer_size_bytes"] = profile.nccl_buffer_size_bytes
    return config


@dataclass(frozen=True)
class Micro210Run:
    skyrl: SkyRLRunConfig
    evaluation: Micro210Evaluation
    canonical_launch_sha256: str
    serving_profile: Micro210ServingProfile | None = None
    model_manifest_filename: str = MODEL_MANIFEST_FILENAME
    selection: Micro210Selection | None = None


def judge_runtime_config(config: dict, judge: JudgeConfig) -> dict:
    """Bind the remote judge connection without changing installed-agent settings."""
    config = deepcopy(config)
    config["environment"]["task_sessions"]["remote_judge"].update(base_url=judge.base_url, api_key_env="JUDGE_API_KEY")
    config["extra_env"]["JUDGE_BASE_URL"] = judge.base_url
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


def validate_judge_availability(judge: JudgeConfig, api_key: str) -> None:
    """Reject an unavailable API judge before allocating rollout GPUs."""
    try:
        response = httpx.get(
            f"{judge.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        raise ValueError("Micro210 API judge is unreachable before rollout allocation") from None
    if judge.model not in {model["id"] for model in response.json()}:
        raise ValueError("Micro210 API endpoint does not serve the frozen judge model")


def run_micro210(config: Micro210Run) -> SkyRLRun:
    if hashlib.sha256(config.skyrl.launch_config_yaml.encode()).hexdigest() != config.canonical_launch_sha256:
        raise ValueError("Canonical Micro210 refuses modified field: launch_config_yaml")
    validate_inputs(config.evaluation, PREPARED_INPUTS, model_manifest_filename=config.model_manifest_filename)
    selection_record = None
    if config.selection is not None:
        policy = config.evaluation.policy()
        selection_record = validate_subset(
            prepared_inputs=PREPARED_INPUTS,
            selected_inputs=config.selection.inputs,
            source_sha256=policy["prepared_rollouts_sha256"],
            data_relative_path=policy["evaluation"]["data_relative_path"],
            subset=config.selection.subset,
        )
    validate_before_allocation(
        config.skyrl.launch_config_yaml, readable_uris=(config.evaluation.record().policy_reference.uri,)
    )
    (StoragePath(config.evaluation.output_uri) / "canonical-evaluation.json").write_text(
        json.dumps(
            {
                **asdict(config.evaluation.record()),
                **({"selection": selection_record} if selection_record is not None else {}),
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
    api_key = require_secret(JUDGE_API_KEY)
    validate_judge_availability(judge, api_key)
    return run_skyrl(config.skyrl, env_vars={JUDGE_API_KEY: api_key, "OPENAI_API_KEY": api_key})


def micro210_step(
    model: Micro210Model,
    *,
    name: str,
    version: str,
    subset: Micro210Subset | None = None,
    serving_profile: ServingProfileName = ServingProfileName.POLICY,
    model_manifest_filename: str | None = None,
    **overrides,
) -> ArtifactStep[SkyRLRun]:
    """Build the canonical evaluation graph; reject all policy override fields."""
    binding = micro210_evaluation(model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>", **overrides)
    binding.policy()
    model_artifact = ArtifactStep.adopt(
        f"{name}/model", version, model.uri, kind=LevanterCheckpoint, config=asdict(model)
    )
    profile = registered_serving_profile(serving_profile)
    return micro210_artifact_step(
        ArtifactHfModel(model_artifact, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
        name=name,
        version=version,
        model_binding=model,
        serving_profile=profile,
        model_manifest_filename=model_manifest_filename,
        subset=subset,
        route=overrides.get("route"),
    )


def micro210_artifact_step(
    model: ArtifactModel,
    *,
    name: str,
    version: str,
    serving_profile: Micro210ServingProfile | None = None,
    model_binding: Micro210Model | None = None,
    subset: Micro210Subset | None = None,
    model_manifest_filename: str | None = None,
    route: Micro210Route | None = None,
) -> ArtifactStep[SkyRLRun]:
    """Bind a future committed model to the same frozen evaluator as a static export."""
    if model_manifest_filename is None:
        model_manifest_filename = serving_profile.model_manifest_filename if serving_profile else MODEL_MANIFEST_FILENAME
    if model_manifest_filename not in {MODEL_MANIFEST_FILENAME, "export-manifest.json"}:
        raise ValueError("Micro210 requires a native HF or SFT export completion manifest")
    fingerprint_model = Micro210Model(
        "s3://unresolved/terminal-export",
        MODEL_IDENTITY_PREFIX + "0" * 64,
        "marin-community/marin-tokenizer-tools",
        "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    )
    binding = micro210_evaluation(
        fingerprint_model, run_id=f"{name}@{version}", output_uri="<resolved-at-build>", route=route
    )
    frozen = binding.policy()
    policy = frozen["evaluation"]
    inputs = ArtifactStep.adopt(
        "evaluation/micro210/prepared",
        PREPARED_INPUTS_VERSION,
        PREPARED_INPUTS,
        config={"sha256": frozen["prepared_rollouts_sha256"]},
    )
    if subset is not None:
        inputs = apply(
            f"{name}/inputs",
            prepare_subset,
            version=version,
            prepared_inputs=inputs,
            source_sha256=frozen["prepared_rollouts_sha256"],
            data_relative_path=policy["data_relative_path"],
            subset=subset,
            output=OUT,
        )
    roles = draccus.decode(SkyRLRolePlan, policy["role_plan"])
    # The route is the binding's execution choice; every other protocol field stays frozen.
    resolved_route = route or Micro210Route.from_policy_execution(policy["execution"])
    execution = IrisSkyRLExecution(
        **{
            **policy["execution"],
            "cluster": resolved_route.target_cluster,
            "parent_cluster_config": resolved_route.parent_cluster_config,
            "cluster_config": resolved_route.cluster_config,
            "target_cluster": resolved_route.target_cluster,
            "container_profile": resolved_route.container_profile,
            "required_env_vars": tuple(policy["execution"]["required_env_vars"]),
        },
        coordinator_timeout_hours=policy["execution"]["job_timeout_seconds"] // 3600 + 1,
    )
    config = runtime_policy_config(frozen).config
    gpu_variant = policy["gpu_variant"]
    if serving_profile:
        roles = serving_profile.role_plan
        execution = serving_profile.execution
        gpu_variant = serving_profile.gpu_variant
        config = serving_config(frozen, serving_profile)
    judge = draccus.decode(JudgeConfig, policy["judge"])
    execution = replace(
        execution,
        coordinator_secret_sources=((JUDGE_API_KEY, judge.credential), ("HF_TOKEN", "env:HF_TOKEN")),
    )
    config = judge_runtime_config(config, judge)
    if isinstance(roles, SkyRLRolePlan):
        # The judged evaluation is a generation-only launch: carry the policy's
        # inference geometry as a generation plan so the entrypoint is plan-derived.
        roles = SkyRLGenerationPlan(
            gpus_per_node=roles.policy_num_gpus_per_node,
            num_inference_engines=roles.num_inference_engines,
            inference_engine_tensor_parallel_size=roles.inference_engine_tensor_parallel_size,
            inference_engine_pipeline_parallel_size=roles.inference_engine_pipeline_parallel_size,
            inference_engine_data_parallel_size=roles.inference_engine_data_parallel_size,
            inference_engine_expert_parallel_size=roles.inference_engine_expert_parallel_size,
            n_samples_per_prompt=roles.n_samples_per_prompt,
        )
    if "entrypoint" in config:
        if config["entrypoint"] != roles.entrypoint:
            raise ValueError("Frozen Micro210 entrypoint disagrees with its generation plan")
        del config["entrypoint"]
    _materialize_role_plan_config(config, roles)
    step = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(config),
            model=model,
            train_data=(),
            validation_data=(ArtifactDataSource(inputs, relative_path=policy["data_relative_path"]),),
            topology=SkyRLTopology(roles.num_nodes, roles.gpus_per_node, gpu_variant, roles),
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
            Micro210Selection(subset, context.artifact_path(inputs)) if subset is not None else None,
        )

    configured = replace(step, run=run_micro210, build_config=build, expected_fingerprint=None)
    return replace(configured, expected_fingerprint=configured.fingerprint())


def route_override(cluster: str) -> Micro210Route:
    """The policy's route re-targeted to one cluster; the protocol stays frozen."""
    policy_route = Micro210Route.from_policy_execution(
        read_micro210_policy(micro210_policy.POLICY_REFERENCE)["evaluation"]["execution"]
    )
    return replace(
        policy_route,
        cluster_config=f"lib/iris/config/{cluster}.yaml",
        target_cluster=cluster,
    )


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-manifest-sha256", required=True)
@click.option(
    "--model-manifest-filename",
    type=click.Choice([MODEL_MANIFEST_FILENAME, "export-manifest.json"]),
    default=None,
    help="Model completion manifest; defaults to the serving profile's format.",
)
@click.option("--tokenizer-uri", required=True)
@click.option("--tokenizer-revision", required=True)
@click.option("--name", required=True)
@click.option(
    "--route-cluster",
    default=None,
    help="Run on this cluster (lib/iris/config/<cluster>.yaml) instead of the policy's route.",
)
@click.option(
    "--serving-profile", type=click.Choice([profile.value for profile in ServingProfileName]), default="policy"
)
@click.option("--benchmark", multiple=True, help="Evaluate only these benchmark rows from the frozen prepared inputs.")
@click.option(
    "--row-index", multiple=True, type=int, help="Recover exact zero-based rows from the frozen prepared inputs."
)
@rl_build_options
def main(
    model_uri: str,
    model_manifest_sha256: str,
    model_manifest_filename: str | None,
    tokenizer_uri: str,
    tokenizer_revision: str,
    name: str,
    route_cluster: str | None,
    benchmark: tuple[str, ...],
    row_index: tuple[int, ...],
    serving_profile: str,
) -> ArtifactStep:
    configure_coreweave_s3()
    model = Micro210Model(model_uri, MODEL_IDENTITY_PREFIX + model_manifest_sha256, tokenizer_uri, tokenizer_revision)
    route = None
    if route_cluster is not None:
        # The per-model route rides on the binding; the protocol stays frozen.
        route = route_override(route_cluster)
    return micro210_step(
        model,
        name=name,
        version=resolve_version(name, None),
        subset=Micro210Subset(tuple(sorted(benchmark)), tuple(sorted(row_index))) if benchmark or row_index else None,
        serving_profile=ServingProfileName(serving_profile),
        model_manifest_filename=model_manifest_filename,
        route=route,
    )


if __name__ == "__main__":
    main()
