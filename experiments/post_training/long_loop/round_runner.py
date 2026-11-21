# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a long-loop round from its YAML spec.

A round YAML carries every value that has differed across the long-loop rounds
plus the plausible knobs: identity (name, version, slug, prefixes), the data
release and curriculum pools it binds, the evaluation policy, and the SFT
knobs (updates, batch, learning rates, min LR ratio, special-token multiplier,
mesh and resources). Code holds the mechanism and the invariants: the plan
template, reference resolution, whole-run annealing, the special-token rate
default, and the final-SFT-checkpoint retention rule.

The runner re-derives the resolved plan through the same pipeline exp4.py uses
so environment-derived hashes (spec_sha256, resolved_sha256) are always fresh;
rounds/exp4.yaml must reproduce exp4.py's plan byte for byte, which the proof
test asserts.
"""

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import draccus
import jmp
import yaml
from iris.cluster.client.job_info import get_job_info
from levanter.training_budget import WallClockCooldown
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, MegatronGeometry, SkyRLRolePlan
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.evaluation.micro210 import Micro210ServingProfile, micro210_artifact_step
from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.adopted_sft import pinned_json
from experiments.post_training.long_loop.archive import PERMANENT_TMP_SEGMENT
from experiments.post_training.long_loop.exp3_production import (
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    REFERENCE_INPUTS,
    baseline_files,
    production_plan,
)
from experiments.post_training.long_loop.exp3_recipe import (
    SftLrSchedule,
    exp3_controls,
    verified_initializer_model,
    whole_run_lr_receipt,
)
from experiments.post_training.long_loop.exp3_rl_restart import restart_recipe
from experiments.post_training.long_loop.exp4 import (
    REFERENCE_SHA256,
    REFERENCE_URI,
    release_artifact,
)
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE, pool_steps
from experiments.post_training.long_loop.nvl72 import MochiProfile, build_run, validate_model_geometry
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.run_spec import FilePin, read_pin
from experiments.post_training.long_loop.shared_sft import native_sft_step, shared_sft_spec
from experiments.post_training.long_loop.spec_pipeline import CompletedSft, build_spec_graph, runtime_execution
from experiments.post_training.task_curation.exp4_curriculum import SOURCE_ROUTES, exp4_rl_curriculum
from experiments.sft.launcher import SFTSpec

MICRO210_PLACEHOLDER_URI = "s3://unresolved/terminal"
MICRO210_PLACEHOLDER_SHA = "sha256:" + "0" * 64
MICRO210_OUTPUT_PLACEHOLDER = "<resolved-at-build>"
RL_WALL_CLOCK_BUDGET = {"seconds": 43200, "drain_margin": 1800}
EVALUATION_EXPECTED_ROLLOUTS = 440
NCCL_BUFFSIZE = "4194304"


@dataclass(frozen=True)
class RoundStorage:
    output_prefix: str
    temporary_prefix: str


@dataclass(frozen=True)
class RoundArchive:
    hf_upload_repo_prefix: str
    evaldash_run_prefix: str


@dataclass(frozen=True)
class RoundSft:
    updates: int
    batch_size: int
    min_lr_ratio: float
    learning_rate: float
    adam_learning_rate: float
    mesh: dict
    resources: dict
    special_token_lr_multiplier: float | None = None
    warmup_steps: int = 0
    lr_schedule: str = "linear"
    mp: str | None = None
    expert_weight_sharding: dict = dataclasses.field(default_factory=dict)
    compiler_dumps: bool = False
    wall_clock_cooldown_fraction: float | None = None
    first_collective_call_timeout: int | None = None


@dataclass(frozen=True)
class RoundSpec:
    schema: int
    name: str
    version: str
    slug: str
    cluster: str
    storage: RoundStorage
    archive: RoundArchive
    data_release: str
    pools: str
    evaluation: str
    initializer: str
    sft: RoundSft


@dataclass(frozen=True)
class RoundAdoptedHf:
    name: str
    version: str
    uri: str
    export_manifest_sha256: str
    config_sha256: str
    checkpoint_step: int
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class RoundServing:
    gpu_variant: str
    role_plan: dict
    execution: dict
    generator: dict
    engine_init_kwargs: dict
    cuda_allocator_config: str
    nccl_env: dict
    nccl_buffer_size_bytes: int


@dataclass(frozen=True)
class AdoptedHfRound:
    schema: int
    name: str
    version: str
    slug: str
    cluster: str
    storage: RoundStorage
    evaluation: str
    adopted_model: RoundAdoptedHf
    serving: RoundServing


@dataclass(frozen=True)
class RoundRl:
    role_plan: dict
    megatron: dict
    execution: dict
    recipe: dict
    train_data: dict
    validation_data: dict
    max_steps: int
    selection_eligibility: str
    native_checkpoint: str | None = None


@dataclass(frozen=True)
class AdoptedRlRound:
    schema: int
    name: str
    version: str
    slug: str
    cluster: str
    storage: RoundStorage
    adopted_model: RoundAdoptedHf
    rl: RoundRl


@dataclass(frozen=True)
class RecoveryRound:
    schema: int
    name: str
    version: str
    slug: str
    cluster: str
    storage: RoundStorage
    resolved_plan: FilePin


@dataclass(frozen=True)
class DataReleaseBinding:
    record: str
    record_sha256: str


@dataclass(frozen=True)
class PoolBinding:
    prefix: str
    version: str


@dataclass(frozen=True)
class InitializerBinding:
    receipt: str
    receipt_sha256: str


DATA_RELEASES = {
    "v7-option-g": DataReleaseBinding(
        record=(
            "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.18-v7-option-g/"
            "artifact-graph/sft-mixes/exp3-v7/release/2026.11.18/.artifact.json"
        ),
        record_sha256="d4e3c7ebe798253d94d67588fc8b8f69211d25bad914d40cbbd7bc2ca7eed061",
    ),
}

POOLS = {
    "exp4-regional": PoolBinding(
        prefix="s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4",
        version="2026.11.18.71",
    ),
}

EVALUATION_POLICIES = {
    "frozen-micro210": micro210_evaluation,
}

INITIALIZERS = {
    "exp3-staged": InitializerBinding(receipt=INITIALIZER_RECEIPT, receipt_sha256=INITIALIZER_SHA256),
}

_ROUND_FIELDS = {field.name for field in dataclasses.fields(RoundSpec)}


def _decode_strict(section: dict, target: type, label: str):
    fields = {field.name: field for field in dataclasses.fields(target)}
    unknown = sorted(set(section) - set(fields))
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")
    missing = sorted(
        name
        for name, field in fields.items()
        if name not in section and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    values = {}
    for name, field in fields.items():
        if name not in section:
            continue
        value = section[name]
        nested = isinstance(field.type, type) and dataclasses.is_dataclass(field.type)
        if nested:
            if not isinstance(value, dict):
                raise ValueError(f"{label}.{name} must be a mapping")
            value = _decode_strict(value, field.type, f"{label}.{name}")
        elif field.type is not bool and isinstance(value, bool):
            raise ValueError(f"{label}.{name} must be {field.type.__name__}, got bool")
        elif not isinstance(value, field.type):
            raise ValueError(f"{label}.{name} must be {field.type.__name__}, got {type(value).__name__}")
        values[name] = value
    return target(**values)


def load_round(path: Path) -> RoundSpec | AdoptedHfRound | AdoptedRlRound | RecoveryRound:
    """Load a round YAML into the strict schema; unknown or mistyped keys fail."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping")
    target = (
        RecoveryRound
        if "resolved_plan" in document
        else (
            AdoptedRlRound
            if "adopted_model" in document and "rl" in document
            else AdoptedHfRound if "adopted_model" in document else RoundSpec
        )
    )
    spec = _decode_strict(document, target, str(path))
    if spec.schema != 1:
        raise ValueError(f"{path} has unsupported schema {spec.schema}")
    return spec


def recovery_round_graph(spec: RecoveryRound, *, through: str, source_revision: str) -> tuple[dict, ArtifactStep]:
    """Rebuild a recorded downstream round while retaining its completed SFT inputs."""
    plan = read_pin(spec.resolved_plan, Path.cwd())
    if plan["name"] != spec.name or plan["storage"] != dataclasses.asdict(spec.storage):
        raise ValueError("Recovery round identity differs from the pinned plan")
    if plan["rl"]["execution"]["cluster"] != spec.cluster:
        raise ValueError("Recovery round cluster differs from the pinned plan")
    # The pinned source plan stays immutable; only this active recipe omits the retired key.
    plan["rl"]["config"]["trainer"]["algorithm"].pop("tis_splice")
    plan["marin_revision"] = source_revision
    plan["runtime_source"] = runtime_source_manifest(Path(__file__).resolve().parents[3], source_revision)
    plan["resolved_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in plan.items() if key != "resolved_sha256"}, sort_keys=True).encode()
    ).hexdigest()
    expected = plan["continuation"]["expected_completed_inputs"]
    adopted = {}
    for label, suffix in (("sft", "sft"), ("handoff", "sft-handoff")):
        pin = expected[label]
        record = json.loads((StoragePath(pin["uri"]) / ".artifact.json").read_bytes())
        if record["fingerprint"] != pin["fingerprint"] or record["version"] != pin["version"]:
            raise ValueError(f"Recovery input completion changed: {label}")
        adopted[label] = ArtifactStep.adopt(f"{spec.name}/{suffix}", spec.version, pin["uri"], config=pin)
    pools = {}
    for route, producer in pool_steps(POOLS["exp4-regional"].version).items():
        uri = plan["datasets"][route]["uri"]
        raw = (StoragePath(uri) / ".artifact.json").read_bytes()
        record = json.loads(raw)
        if record["fingerprint"] != producer.fingerprint():
            raise ValueError(f"Recovery data producer changed: {route}")
        pools[route] = ArtifactStep.adopt(
            producer.name,
            producer.version,
            uri,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(raw).hexdigest(),
                "files": plan["datasets"][route]["files"],
            },
        )
    return plan, build_spec_graph(
        plan,
        spec.version,
        through=through,
        data_artifacts=pools,
        curriculum=exp4_rl_curriculum(),
        completed_sft=CompletedSft(adopted["sft"], adopted["handoff"]),
    )


def adopted_round_rl(spec: AdoptedRlRound, *, smoke: bool = False) -> ArtifactStep:
    """Adopt a schema-3 HF export or a native RL checkpoint without an SFT stage."""
    model = spec.adopted_model
    manifest = pinned_json(prefix_join(model.uri, "export-manifest.json"), model.export_manifest_sha256)
    config = pinned_json(prefix_join(model.uri, "config.json"), model.config_sha256)
    if config["grugmoe_artifact_schema_version"] != 3:
        raise ValueError("RL adoption requires a schema-3 HF export")
    if not manifest["request"]["config"]["checkpoint"].endswith(f"/step-{model.checkpoint_step}"):
        raise ValueError("Adopted HF export checkpoint step changed")
    rl = spec.rl
    roles = draccus.decode(SkyRLRolePlan, rl.role_plan)
    execution = runtime_execution(rl.execution)
    job_limit = 2 * 3600 if smoke else 48 * 3600
    if execution.cluster != spec.cluster or execution.job_timeout_seconds != job_limit:
        raise ValueError("Adopted RL requires the round cluster and a 48-hour job limit (two hours for smoke)")
    recipe = json.loads(json.dumps(rl.recipe))
    if rl.selection_eligibility not in {"covered", "complete_window"}:
        raise ValueError("Unknown checkpoint selection eligibility")
    for callback in recipe["trainer"]["callbacks"]:
        if callback["type"] == "best_checkpoint":
            callback["minimum_scored_tasks"] = (
                callback["expected_tasks"]
                if rl.selection_eligibility == "complete_window"
                else math.ceil(0.9 * callback["expected_tasks"])
            )
            callback["selection_eligibility"] = rl.selection_eligibility
    adopted = ArtifactStep.adopt(
        model.name, model.version, model.uri, kind=LevanterCheckpoint, config=dataclasses.asdict(model)
    )
    profile = MochiProfile(spec.name, roles, draccus.decode(MegatronGeometry, rl.megatron), execution)
    validate_model_geometry(config, profile)
    training = []
    heldout = None
    for label, pin in (*rl.train_data.items(), ("heldout", rl.validation_data)):
        if len(pin["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in pin["sha256"]):
            raise ValueError(f"Round data requires a SHA-256 pin: {label}")
        source = ArtifactStep.adopt(f"{spec.name}/{label}", spec.version, pin["uri"], config=pin)
        data = ArtifactDataSource(source, relative_path=pin["file"])
        if label == "heldout":
            heldout = data
        else:
            training.append(data)
    if not training or heldout is None:
        raise ValueError("Adopted RL requires training and held-out data")
    resume = (
        ArtifactStep.adopt(f"{spec.name}/native-input", spec.version, rl.native_checkpoint)
        if rl.native_checkpoint
        else None
    )
    step = build_run(
        profile=profile,
        name=spec.name,
        version=spec.version,
        model=ArtifactHfModel(adopted, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
        train_data=tuple(training),
        validation_data=(heldout,),
        max_steps=rl.max_steps,
        resume=resume,
        smoke=smoke,
        recipe_document=recipe,
    )
    original_build = step.build_config

    def verified_build(context):
        if not context.is_fingerprint:
            for pin in (*rl.train_data.values(), rl.validation_data):
                fs, key = filesystem_for(prefix_join(pin["uri"], pin["file"]))
                with fs.open(key, "rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != pin["sha256"]:
                    raise ValueError(f"Round data changed: {pin['uri']}/{pin['file']}")
        return original_build(context)

    return dataclasses.replace(step, build_config=verified_build)


def adopted_round_evaluation(spec: AdoptedHfRound) -> ArtifactStep:
    """Bind a pinned HF export directly to evaluation, without training stages."""
    if spec.evaluation != "frozen-micro210":
        raise ValueError("Adopted evaluation rounds require frozen-micro210")
    model = spec.adopted_model
    manifest = pinned_json(prefix_join(model.uri, "export-manifest.json"), model.export_manifest_sha256)
    config = pinned_json(prefix_join(model.uri, "config.json"), model.config_sha256)
    if not manifest["request"]["config"]["checkpoint"].endswith(f"/step-{model.checkpoint_step}"):
        raise ValueError("Adopted HF export does not match the requested checkpoint step")
    serving = spec.serving
    roles = draccus.decode(SkyRLRolePlan, serving.role_plan)
    execution = runtime_execution(serving.execution)
    if execution.cluster != spec.cluster:
        raise ValueError("Round cluster and serving execution cluster disagree")
    if config["num_experts"] % roles.inference_engine_expert_parallel_size:
        raise ValueError("Adopted model experts do not divide the serving expert parallelism")
    binding = Micro210Model(
        model.uri, "sha256:" + model.export_manifest_sha256, model.tokenizer_uri, model.tokenizer_revision
    )
    adopted = ArtifactStep.adopt(
        model.name, model.version, model.uri, kind=LevanterCheckpoint, config=dataclasses.asdict(model)
    )
    return micro210_artifact_step(
        ArtifactHfModel(adopted, model.tokenizer_uri, model.tokenizer_revision, relative_path=""),
        name=f"{spec.name}/eval",
        version=spec.version,
        model_binding=binding,
        serving_profile=Micro210ServingProfile(
            gpu_variant=serving.gpu_variant,
            role_plan=roles,
            execution=execution,
            generator=serving.generator,
            engine_init_kwargs=serving.engine_init_kwargs,
            cuda_allocator_config=serving.cuda_allocator_config,
            nccl_env=serving.nccl_env,
            nccl_buffer_size_bytes=serving.nccl_buffer_size_bytes,
            model_manifest_filename="export-manifest.json",
        ),
    )


def round_sft_spec(round_spec: RoundSpec, name: str, version: str, recipe: dict) -> SFTSpec:
    """Bind and verify the round's declared precision, token rates and LR controls."""
    spec = shared_sft_spec(name, version, recipe)
    timeout = round_spec.sft.first_collective_call_timeout
    if timeout is not None:
        if timeout <= 0:
            raise ValueError("first_collective_call_timeout must be positive seconds")
        spec = dataclasses.replace(
            spec,
            xla_flags=(*spec.xla_flags, f"--xla_gpu_first_collective_call_terminate_timeout_seconds={timeout}"),
        )
    section_value = recipe.get("special_token_lr_multiplier")
    round_value = round_spec.sft.special_token_lr_multiplier
    if section_value is not None and round_value is not None and section_value != round_value:
        raise ValueError("The round YAML sets two different special-token LR multipliers")
    multiplier = section_value if section_value is not None else round_value
    if multiplier is None:
        multiplier = 1.0
    optimizer = dataclasses.replace(
        spec.optimizer,
        learning_rate=round_spec.sft.learning_rate,
        adam_lr=round_spec.sft.adam_learning_rate,
    )
    spec = dataclasses.replace(
        spec,
        optimizer=optimizer,
        snowball_sft=dataclasses.replace(spec.snowball_sft, special_token_lr_multiplier=multiplier),
    )
    if spec.snowball_sft.special_token_lr_multiplier != multiplier:
        raise ValueError("Exp round special-token LR multiplier disagrees with its resolved recipe")
    schedule = draccus.decode(SftLrSchedule, recipe["optimizer_schedule"])
    if (schedule.warmup, schedule.min_lr_ratio, schedule.lr_schedule) != (
        round_spec.sft.warmup_steps,
        round_spec.sft.min_lr_ratio,
        round_spec.sft.lr_schedule,
    ):
        raise ValueError("Resolved schedule differs from the round declarations")
    if round_spec.sft.mp is not None:
        if spec.trainer_config is None or spec.trainer_config.mp != jmp.get_policy(round_spec.sft.mp):
            raise ValueError("Built trainer precision differs from the round declaration")
    budget = spec.trainer_config.training_budget if spec.trainer_config is not None else None
    expected_cooldown = (
        WallClockCooldown(round_spec.sft.wall_clock_cooldown_fraction, round_spec.sft.min_lr_ratio)
        if round_spec.sft.wall_clock_cooldown_fraction is not None
        else None
    )
    if (budget.cooldown if budget is not None else None) != expected_cooldown:
        raise ValueError("Built wall-clock cooldown differs from the round declaration")
    whole_run_lr_receipt(spec, schedule=schedule)
    return spec


def round_plan(round_spec: RoundSpec, release, source_revision: str, version: str, *, initial_eval_inputs: str) -> dict:
    """Derive the round's resolved plan: the shared template plus the round's fields."""
    pools_binding = POOLS[round_spec.pools]
    resolved = production_plan(release, source_revision, baseline_files(), initial_eval_inputs=initial_eval_inputs)
    sft = resolved["sft"]
    sft["batch_size"] = round_spec.sft.batch_size
    sft["mesh"] = round_spec.sft.mesh
    sft["trainer"]["mesh"] = round_spec.sft.mesh
    sft["resources"] = round_spec.sft.resources
    if round_spec.sft.first_collective_call_timeout is not None:
        sft["first_collective_call_timeout"] = round_spec.sft.first_collective_call_timeout
    if round_spec.sft.compiler_dumps:
        sft["trainer"]["log_jaxprs"] = True
        sft["trainer"]["log_xla_hlo"] = True
        sft["trainer"]["xla_dump_upload"] = {"enabled": True, "ttl_days": 30}
        sft["xla_flags"] = [*sft["xla_flags"], "--xla_dump_to=/tmp/long-loop-xla", "--xla_dump_hlo_as_text"]
    if round_spec.sft.expert_weight_sharding:
        sft["model"]["expert_weight_sharding"] = round_spec.sft.expert_weight_sharding
    updates = round_spec.sft.updates
    sft["stop_after_steps"] = updates
    sft["trainer"]["num_train_steps"] = updates
    sft["target_positions"] = updates * sft["batch_size"] * sft["seq_len"]
    sft["optimizer_schedule"] = {
        "horizon": updates,
        "warmup": round_spec.sft.warmup_steps,
        "decay": updates - round_spec.sft.warmup_steps,
        "min_lr_ratio": round_spec.sft.min_lr_ratio,
        "lr_schedule": round_spec.sft.lr_schedule,
    }
    if round_spec.sft.mp is not None:
        sft["trainer"]["mp"] = round_spec.sft.mp
    if round_spec.sft.wall_clock_cooldown_fraction is not None:
        budget = sft["trainer"]["training_budget"]
        if budget is None:
            raise ValueError("Declared wall-clock cooldown requires a training budget")
        budget["cooldown"] = draccus.encode(
            WallClockCooldown(
                fraction=round_spec.sft.wall_clock_cooldown_fraction, min_lr_ratio=round_spec.sft.min_lr_ratio
            )
        )
    # Record the effective special-token control in the resolved plan only when the
    # round departs from the qualified natural rate; a round at the natural rate keeps
    # the plan identical to the qualified recipe.
    multiplier = (
        round_spec.sft.special_token_lr_multiplier if round_spec.sft.special_token_lr_multiplier is not None else 1.0
    )
    if multiplier != exp3_controls().special_token_lr_multiplier:
        sft["special_token_lr_multiplier"] = multiplier
    raw = StoragePath(REFERENCE_URI).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Exp3 comparison reference changed")
    reference = json.loads(raw)
    rl = reference["plan"]["rl"]
    recipe = restart_recipe(reference)
    recipe["extra_env"]["NCCL_BUFFSIZE"] = NCCL_BUFFSIZE
    recipe["terminal_bench"]["harbor"].pop("eval_override_storage_mb", None)
    recipe["trainer"]["rollout_buffer"]["consumed_route_quotas"] = {"agentic": 22, "nonagentic": 42}
    recipe["trainer"]["rollout_buffer"]["source_routes"] = {
        **SOURCE_ROUTES,
        NEMOTRON_SOURCE: "nonagentic",
    }
    recipe["trainer"]["callbacks"] = [
        callback for callback in recipe["trainer"]["callbacks"] if callback["type"] != "wall_time_budget"
    ]
    resolved["rl"] = {
        **rl,
        "name": f"{round_spec.slug}-rl",
        "config": recipe,
        "wall_clock_budget": dict(RL_WALL_CLOCK_BUDGET),
    }
    resolved["name"] = round_spec.name
    resolved["storage"]["output_prefix"] = round_spec.storage.output_prefix
    resolved["storage"]["temporary_prefix"] = round_spec.storage.temporary_prefix
    coordinator = resolved["execution"]["coordinator"]
    coordinator["cluster"] = round_spec.cluster
    coordinator["cluster_config"] = f"lib/iris/config/{round_spec.cluster}.yaml"
    coordinator["target_cluster"] = round_spec.cluster
    coordinator["coordinator_name"] = round_spec.slug
    policy = EVALUATION_POLICIES[round_spec.evaluation]
    canonical = policy(
        Micro210Model(
            MICRO210_PLACEHOLDER_URI,
            MICRO210_PLACEHOLDER_SHA,
            resolved["tokenizer"]["repo"],
            resolved["tokenizer"]["revision"],
        ),
        run_id=f"{round_spec.slug}@{version}",
        output_uri=MICRO210_OUTPUT_PLACEHOLDER,
    )
    resolved["evaluation"] = {
        **canonical.policy()["evaluation"],
        "name": f"users/benfeuer/checkpoints/{round_spec.slug}-micro210",
        "expected_rollouts": EVALUATION_EXPECTED_ROLLOUTS,
    }
    for route, producer in pool_steps(pools_binding.version).items():
        resolved["datasets"][route]["uri"] = producer.path(pools_binding.prefix)
        resolved["datasets"][route]["data_file"] = "train.parquet"
    resolved["datasets"]["nonagentic"]["source_names"] = [
        *resolved["datasets"]["nonagentic"]["source_names"],
        NEMOTRON_SOURCE,
    ]
    resolved["archive"]["hf_upload"]["repo_prefix"] = round_spec.archive.hf_upload_repo_prefix
    resolved["archive"]["evaldash"]["run_prefix"] = round_spec.archive.evaldash_run_prefix
    curriculum = exp4_rl_curriculum()
    resolved["curriculum"] = {**dataclasses.asdict(curriculum), "identity": curriculum.identity}
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return resolved


def resolve_round_pools(round_spec: RoundSpec, resolved: dict) -> dict[str, ArtifactStep]:
    """Verify the bound regional pools and adopt them as graph artifacts."""
    binding = POOLS[round_spec.pools]
    pools = {}
    for route, producer in pool_steps(binding.version).items():
        path = producer.path(binding.prefix)
        record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
        producer_record = json.loads(record_bytes)
        if producer_record["fingerprint"] != producer.fingerprint():
            raise ValueError(f"Regional pool producer differs: {route}")
        manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest["curriculum"] != exp4_rl_curriculum().identity:
            raise ValueError("Regional pool was prepared for another curriculum")
        pins = [
            {"uri": path + "/train.parquet", "sha256": manifest["sha256"]},
            {"uri": path + "/manifest.json", "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        ]
        pools[route] = ArtifactStep.adopt(
            producer.name,
            producer.version,
            path,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "files": pins,
            },
        )
        resolved["datasets"][route].update(
            uri=path,
            rows=manifest["rows"],
            files=pins,
            source_names=[NEMOTRON_SOURCE if label == "nemotron_if" else label for label in manifest["source_counts"]],
            teacher_routes=manifest["source_counts"],
        )
    return pools


def require_permanent_round_storage(round_spec: RoundSpec | AdoptedHfRound | AdoptedRlRound | RecoveryRound) -> None:
    """A round's outputs live at a permanent path, never under a scratch TTL prefix."""
    path = StoragePath(round_spec.storage.output_prefix)
    if PERMANENT_TMP_SEGMENT in path.segments:
        raise ValueError(
            f"Round storage.output_prefix must be a permanent path, not scratch storage: "
            f"{round_spec.storage.output_prefix}"
        )


def require_archive_binds_final_sft_checkpoint(resolved: dict) -> None:
    """The archive stage must bind the SFT selection so it can protect and verify it."""
    stage_files = resolved["archive"].get("stage_files", {})
    sft_files = stage_files.get("sft", [])
    if not any(entry.get("path") == "selected.json" for entry in sft_files):
        raise ValueError("Round archive settings must bind the SFT stage's selected.json")


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--through", type=click.Choice(["sft", "rl", "eval", "archive"]), default="archive", show_default=True)
@click.option("--smoke", is_flag=True, help="Use two rollout groups for an adopted RL round with max_steps=2.")
@rl_build_options
def main(round_path: Path, source_revision: str, validation_output: Path, through: str, smoke: bool) -> ArtifactStep:
    configure_coreweave_s3()
    round_spec = load_round(round_path)
    if click.get_current_context().params["default_version"] != round_spec.version:
        raise ValueError(f"--version must equal the round's version {round_spec.version}")
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    require_permanent_round_storage(round_spec)
    if smoke and (not isinstance(round_spec, AdoptedRlRound) or round_spec.rl.max_steps != 2):
        raise click.UsageError("--smoke requires an adopted RL round with rl.max_steps=2")
    if isinstance(round_spec, RecoveryRound):
        if through not in {"rl", "archive"}:
            raise click.UsageError("A recovery round requires --through rl or archive")
        os.environ.update(
            MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
        )
        resolved, terminal = recovery_round_graph(round_spec, through=through, source_revision=source_revision)
        validation_output.mkdir(parents=True, exist_ok=True)
        (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
        model = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
        validate_graph(
            terminal,
            validation_output,
            round_spec.storage.output_prefix,
            model_config=(StoragePath(model) / "config.json").read_bytes(),
        )
        return terminal
    if isinstance(round_spec, AdoptedRlRound):
        if through != "rl":
            raise click.UsageError("An adopted RL round requires --through rl")
        os.environ.update(
            MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
        )
        terminal = adopted_round_rl(round_spec, smoke=smoke)
        validation_output.mkdir(parents=True, exist_ok=True)
        (validation_output / "resolved-plan.json").write_text(
            json.dumps(
                {
                    **dataclasses.asdict(round_spec),
                    "marin_revision": source_revision,
                    "runtime_source": runtime_source_manifest(workspace, source_revision),
                },
                indent=2,
            )
        )
        validate_graph(
            terminal,
            validation_output,
            round_spec.storage.output_prefix,
            model_config=(StoragePath(round_spec.adopted_model.uri) / "config.json").read_bytes(),
        )
        return terminal
    if isinstance(round_spec, AdoptedHfRound):
        if through != "eval":
            raise click.UsageError("An adopted HF round requires --through eval")
        os.environ.update(
            MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
        )
        terminal = adopted_round_evaluation(round_spec)
        validation_output.mkdir(parents=True, exist_ok=True)
        resolved = {
            **dataclasses.asdict(round_spec),
            "through": through,
            "runtime_source": runtime_source_manifest(workspace, source_revision),
        }
        (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
        validate_graph(
            terminal,
            validation_output,
            round_spec.storage.output_prefix,
            model_config=(StoragePath(round_spec.adopted_model.uri) / "config.json").read_bytes(),
        )
        return terminal
    binding = DATA_RELEASES[round_spec.data_release]
    release, handle = release_artifact(
        binding.record, binding.record_sha256, require_ready=click.get_current_context().params["do_run"]
    )
    version = resolve_version(round_spec.slug, round_spec.version)
    resolved = round_plan(round_spec, release, source_revision, version, initial_eval_inputs=REFERENCE_INPUTS)
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    os.environ.update(
        MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
    )
    pools = resolve_round_pools(round_spec, resolved)
    require_archive_binds_final_sft_checkpoint(resolved)
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    terminal = build_spec_graph(
        resolved,
        version,
        through=through,
        data_artifacts=pools,
        curriculum=exp4_rl_curriculum(),
        sft_release=handle,
        sft_step_builder=partial(native_sft_step, spec_builder=partial(round_sft_spec, round_spec)),
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
    (validation_output / "optimizer-groups.json").write_text(
        json.dumps(optimizer_group_receipt(resolved["sft"]), indent=2)
    )
    (validation_output / "sft-lr-schedule.json").write_text(
        json.dumps(
            whole_run_lr_receipt(
                round_sft_spec(round_spec, f"{round_spec.name}/sft", version, resolved["sft"]),
                schedule=draccus.decode(SftLrSchedule, resolved["sft"]["optimizer_schedule"]),
            ),
            indent=2,
        )
    )
    initializer = INITIALIZERS[round_spec.initializer]
    model = verified_initializer_model(initializer.receipt, initializer.receipt_sha256)
    validate_graph(
        terminal,
        validation_output,
        round_spec.storage.output_prefix,
        model_config=(StoragePath(model) / "config.json").read_bytes(),
    )
    return terminal


if __name__ == "__main__":
    main()
