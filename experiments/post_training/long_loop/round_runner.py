# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a long-loop round from its YAML spec.

A round YAML carries every value that has differed across the long-loop rounds
plus the plausible knobs: identity (name, version, slug, prefixes), the data
release and curriculum pools it binds, the evaluation policy, and the SFT
knobs (stopping, batch, learning rates, min LR ratio, special-token multiplier,
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
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin

import click
import draccus
import jmp
import yaml
from iris.cluster.client.job_info import get_job_info
from levanter.training_budget import TrainingBudgetConfig, WallClockCooldown
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.experiment.cli import build_options_with_runner
from marin.rl.cli import submit_or_run
from marin.rl.curriculum import RlCurriculum, consumed_route_quotas
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, MegatronGeometry, SkyRLPolicyModel, SkyRLRolePlan
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from skyrlrecipe import RecipePatch
from skyrlrecipe.documents import merge_mappings, set_path

from experiments.evaluation.micro210 import (
    Micro210ServingProfile,
    ServingProfileName,
    micro210_artifact_step,
    registered_serving_profile,
)
from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.adopted_sft import pinned_json
from experiments.post_training.long_loop.archive import PERMANENT_TMP_SEGMENT
from experiments.post_training.long_loop.execution_policy import (
    RL_TRAINING_BUDGET,
    SFT_FINALIZATION_RESERVE,
    SFT_TRAINING_DURATION,
    SFT_UPDATE_RESERVE,
    rl_training_callbacks,
)
from experiments.post_training.long_loop.exp3_production import (
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    REFERENCE_INPUTS,
    StoragePin,
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
    QUALIFIED_V7,
    REFERENCE_SHA256,
    REFERENCE_URI,
    V7_EVIDENCE_NAMES,
    release_artifact,
)
from experiments.post_training.long_loop.exp4_data import NEMOTRON_SOURCE, pool_steps
from experiments.post_training.long_loop.exp5_data import pool_steps as exp5_pool_steps
from experiments.post_training.long_loop.mochi_sft import (
    MochiSftRound,
    mochi_sft_step,
    resolve_mochi_sft,
    run_mochi_sft,
)
from experiments.post_training.long_loop.nvl72 import RETENTION, MochiProfile, build_run, validate_model_geometry
from experiments.post_training.long_loop.openresearcher_service import (
    OpenResearcherService,
    retrieval_launch,
    run_with_retrieval,
    service_execution,
    service_recipe,
)
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.rl import curriculum_validation_callbacks
from experiments.post_training.long_loop.rl_terminal_evaluation import best_and_final_evaluations
from experiments.post_training.long_loop.run_spec import FilePin, read_pin
from experiments.post_training.long_loop.sft_epoch import MixtureEpochs, SftStopping, bind_sft_stopping
from experiments.post_training.long_loop.shared_sft import native_sft_step, shared_sft_spec
from experiments.post_training.long_loop.spec_pipeline import CompletedSft, build_spec_graph, runtime_execution
from experiments.post_training.long_loop.stage_guard import StageGuard
from experiments.post_training.task_curation.curriculum_manifest import curriculum_from_manifest, rollout_source_routes
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.exp4_curriculum import (
    SOURCE_ROUTES,
)
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.sft.launcher import SFTSpec
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME

MICRO210_PLACEHOLDER_URI = "s3://unresolved/terminal"
MICRO210_PLACEHOLDER_SHA = "sha256:" + "0" * 64
MICRO210_OUTPUT_PLACEHOLDER = "<resolved-at-build>"
EVALUATION_EXPECTED_ROLLOUTS = 440
NCCL_BUFFSIZE = "4194304"


@dataclass(frozen=True)
class RoundPools:
    artifacts: dict[str, ArtifactStep]
    curriculum: RlCurriculum
    dataset: dict


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
    batch_size: int
    min_lr_ratio: float
    learning_rate: float
    adam_learning_rate: float
    mesh: dict
    resources: dict
    stopping: SftStopping = dataclasses.field(default_factory=MixtureEpochs)
    epoch_capacity: FilePin | None = None
    training_budget: TrainingBudgetConfig = dataclasses.field(
        default_factory=partial(
            TrainingBudgetConfig,
            duration=SFT_TRAINING_DURATION,
            finalization_reserve=SFT_FINALIZATION_RESERVE,
            update_reserve=SFT_UPDATE_RESERVE,
        )
    )
    stage_guard: StageGuard | None = None
    task_timeout_seconds: int = int(SFT_TRAINING_DURATION.total_seconds())
    data_seed: int = 0
    special_token_lr_multiplier: float | None = None
    warmup_steps: int = 0
    lr_schedule: str = "linear"
    mp: str | None = None
    expert_weight_sharding: dict = dataclasses.field(default_factory=dict)
    compiler_dumps: bool = False
    wall_clock_cooldown_fraction: float | None = None
    first_collective_call_timeout: int | None = None
    xla_flags: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.training_budget is None or self.training_budget.duration != SFT_TRAINING_DURATION:
            raise ValueError("Long-loop SFT requires the canonical training budget")


@dataclass(frozen=True)
class RoundRlLayout:
    role_plan: SkyRLRolePlan
    megatron: MegatronGeometry


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
    openresearcher_service: OpenResearcherService | None = None
    rl_recipe_patch: FilePin | None = None
    rl_layout: RoundRlLayout | None = None


@dataclass(frozen=True)
class RoundAdoptedHf:
    name: str
    version: str
    uri: str
    config_sha256: str
    checkpoint_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    export_manifest_sha256: str | None = None
    model_manifest_sha256: str | None = None
    certification_seal: FilePin | None = None


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
    pools: dict
    """The regional pool artifact the round trains on: ``{uri}``. Its manifest carries
    the fixed curriculum's identity and holdout, the per-source train files, their
    digests, the rollout routes, and the holdout's validation.parquet — the round's
    validation data is that materialization, never an adopted parquet."""
    max_steps: int
    selection_eligibility: str
    export_hf: bool = False
    retained_checkpoints: int = -1
    native_checkpoint: str | None = None
    gpu_variant: str = "GB200"
    recipe_patch: FilePin | None = None
    draft: str | None = None
    """Registered speculative-decoding draft to serve during rollout; ``None`` keeps plain decoding."""
    openresearcher_service: OpenResearcherService | None = None
    """Required when the round's pools carry the OpenResearcher route; binds its retrieval service."""


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
    evaluation: str | None = None
    evaluation_profile: str | None = None
    evaluation_checkpoints: str = "best"


@dataclass(frozen=True)
class RoundDraft:
    """A registered speculative-decoding draft and the initializer it was trained against."""

    name: str
    trained_for_initializer: str
    source_uri: str
    source_identity: str
    num_speculative_tokens: int

    def speculative_decoding(self) -> dict:
        return {
            "method": "dspark",
            "model": {"source_uri": self.source_uri, "source_identity": self.source_identity},
            "num_speculative_tokens": self.num_speculative_tokens,
        }


# Drafts are registered with the initializer their weights were trained against. Long-loop
# rounds start from round-specific SFT checkpoints, so a draft never binds by default: each
# round must name one explicitly and the resolved plan records that provenance. A draft
# trained against a different initializer is unmeasured on that round and stays behind its
# gate (median update time, rollout-group p95, acceptance length, fresh-policy KL).
DRAFTS = {
    "dspark-dr-doom-datakit-sft-0921": RoundDraft(
        name="dspark-dr-doom-datakit-sft-0921",
        trained_for_initializer=(
            "snowball-datakit-sft 2026.09.21 initializer "
            "(models/snowball-datakit-sft-fe666478454e79fe6027a0bab0388d9abd68f51a, "
            "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21)"
        ),
        source_uri="hf://atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020",
        source_identity="d22f4f21ce5cc0968dfd8e2c0d06aaf4adacbbf2",
        num_speculative_tokens=4,
    ),
}


def round_draft(spec: AdoptedRlRound) -> RoundDraft | None:
    """Resolve the round's explicit draft binding; unknown names fail loudly."""
    if spec.rl.draft is None:
        return None
    if spec.rl.draft not in DRAFTS:
        raise ValueError(f"Unknown round draft binding: {spec.rl.draft}")
    return DRAFTS[spec.rl.draft]


@dataclass(frozen=True)
class RecoveryRound:
    schema: int
    name: str
    version: str
    slug: str
    cluster: str
    storage: RoundStorage
    resolved_plan: FilePin
    pools: str
    openresearcher_service: OpenResearcherService | None = None


@dataclass(frozen=True)
class DataReleaseBinding:
    record: str
    record_sha256: str
    qualification: StoragePin
    evidence_names: tuple[str, ...]


@dataclass(frozen=True)
class PoolBinding:
    prefix: str
    version: str
    producer: Callable[..., dict[str, ArtifactStep]]


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
        qualification=QUALIFIED_V7,
        evidence_names=V7_EVIDENCE_NAMES,
    ),
    "v9-overthinking": DataReleaseBinding(
        record="s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.22-v9-overthinking/artifact-graph/sft-mixes/exp3-v9/release/2026.11.22/.artifact.json",
        record_sha256="811471acd16af6a6d52b2ad2dae0ef597304615047f8c15e32ebda77297e838d",
        qualification=StoragePin(
            "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.22-v9-overthinking/release-v9/release-complete-v9.json",
            "2a7d231a3b79cc722929c76f6c03441c9d7d475ad79b46284eb35b75fd01c898",
        ),
        evidence_names=("native_consumer_receipt", "native_measurement_job", "v9_transform_receipt", "supersedes"),
    ),
}

POOLS = {
    "exp4-regional": PoolBinding(
        prefix="s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4",
        version="2026.11.18.71",
        producer=pool_steps,
    ),
    "exp5-prepared": PoolBinding(
        prefix="s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5",
        version="2026.11.22.1",
        producer=exp5_pool_steps,
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
        value_type = get_origin(field.type)
        value_type = field.type if value_type in (None, UnionType) else value_type
        nested = isinstance(field.type, type) and dataclasses.is_dataclass(field.type)
        if isinstance(field.type, type) and issubclass(field.type, draccus.ChoiceRegistry):
            if not isinstance(value, dict):
                raise ValueError(f"{label}.{name} must be a mapping")
            choice = value.get("type", field.type.default_choice_name())
            target_choice = field.type.get_choice_class(choice)
            value = _decode_strict(
                {key: item for key, item in value.items() if key != "type"}, target_choice, f"{label}.{name}"
            )
        elif get_origin(field.type) is UnionType and any(dataclasses.is_dataclass(arg) for arg in get_args(field.type)):
            value = draccus.decode(field.type, value)
        elif nested:
            if not isinstance(value, dict):
                raise ValueError(f"{label}.{name} must be a mapping")
            value = _decode_strict(value, field.type, f"{label}.{name}")
        elif field.type is timedelta:
            value = draccus.decode(timedelta, value)
        elif field.type is not bool and isinstance(value, bool):
            raise ValueError(f"{label}.{name} must be {field.type.__name__}, got bool")
        elif not isinstance(value, value_type):
            raise ValueError(f"{label}.{name} must be {field.type.__name__}, got {type(value).__name__}")
        values[name] = value
    return target(**values)


def load_round(path: Path) -> RoundSpec | AdoptedHfRound | AdoptedRlRound | RecoveryRound | MochiSftRound:
    """Load a round YAML into the strict schema; unknown or mistyped keys fail."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping")
    if "sft" in document and "training_budget" in document["sft"] and document["sft"]["training_budget"] is None:
        raise ValueError("Long-loop SFT cannot disable its training budget; omit training_budget to use the default")
    if "sft" in document and document["sft"].get("training_budget") is not None:
        budget = document["sft"]["training_budget"]
        if "duration" in budget and draccus.decode(timedelta, budget["duration"]) != SFT_TRAINING_DURATION:
            raise ValueError("Long-loop SFT duration differs from the canonical training budget")
        budget["duration"] = draccus.encode(SFT_TRAINING_DURATION)
        document["sft"].setdefault("task_timeout_seconds", int(SFT_TRAINING_DURATION.total_seconds()))
    target = (
        MochiSftRound
        if document.get("format") == "mochi_sft"
        else (
            RecoveryRound
            if "resolved_plan" in document
            else (
                AdoptedRlRound
                if "adopted_model" in document and "rl" in document
                else AdoptedHfRound if "adopted_model" in document else RoundSpec
            )
        )
    )
    spec = _decode_strict(document, target, str(path))
    if spec.schema != 1:
        raise ValueError(f"{path} has unsupported schema {spec.schema}")
    if isinstance(spec, AdoptedRlRound):
        trainer = spec.rl.recipe["trainer"]
        recipe = {
            **spec.rl.recipe,
            "trainer": {
                **trainer,
                "callbacks": rl_training_callbacks(
                    trainer.get("callbacks", []), groups_per_step=spec.rl.role_plan["train_batch_size"]
                ),
            },
        }
        spec = dataclasses.replace(spec, rl=dataclasses.replace(spec.rl, recipe=recipe))
    return spec


def recovery_round_graph(spec: RecoveryRound, *, through: str, source_revision: str) -> tuple[dict, ArtifactStep]:
    """Rebuild a recorded downstream round while retaining its completed SFT inputs."""
    plan = read_pin(spec.resolved_plan, Path.cwd())
    if plan["name"] != spec.name or plan["storage"] != dataclasses.asdict(spec.storage):
        raise ValueError("Recovery round identity differs from the pinned plan")
    if plan["rl"]["execution"]["cluster"] != spec.cluster:
        raise ValueError("Recovery round cluster differs from the pinned plan")
    # The pinned source plan stays immutable; only this active recipe omits the retired key.
    plan["rl"]["config"]["trainer"]["algorithm"].pop("tis_splice", None)
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
    binding = POOLS[spec.pools]
    pools = round_pools(binding)
    pinned = plan["datasets"]["pools"]
    if pinned["uri"] != pools.dataset["uri"] or pinned["composition"] != pools.dataset["composition"]:
        raise ValueError("Recovery data differs from the pinned plan's pools")
    return plan, build_spec_graph(
        plan,
        spec.version,
        through=through,
        data_artifacts=pools.artifacts,
        curriculum=pools.curriculum,
        completed_sft=CompletedSft(adopted["sft"], adopted["handoff"]),
        openresearcher_service=spec.openresearcher_service,
    )


def adopted_round_resolved_plan(spec: AdoptedRlRound, *, source_revision: str, workspace: Path) -> dict:
    """The adopted round's resolved plan: the spec plus runtime identity and any draft binding."""
    plan = {
        **dataclasses.asdict(spec),
        "marin_revision": source_revision,
        "runtime_source": runtime_source_manifest(workspace, source_revision),
    }
    draft = round_draft(spec)
    if draft is not None:
        plan["draft_binding"] = dataclasses.asdict(draft)
    return plan


def verified_certified_export(model: RoundAdoptedHf) -> None:
    """Adopt a certified SFT export: the seal proves the checkpoint step, the staging manifest binds the payload.

    The export manifest of a certified export may predate the certified weights (the Exp5
    step-527 export carries the exp4 handoff's manifest), so the step evidence is the
    hash-pinned certification seal and its native/HF/completion receipts — never the
    export manifest.
    """
    seal = read_pin(model.certification_seal, Path.cwd())
    if seal["status"] != "COMPLETE":
        raise ValueError("The SFT certification is not complete")
    step = seal["step"]
    if step != model.checkpoint_step or seal["completed_updates"] != step + 1:
        raise ValueError("The certification seal does not certify the adopted checkpoint step")
    checkpoint = StoragePath(seal["native_checkpoint"])
    root = checkpoint.parent.parent
    if (
        checkpoint != root / f"checkpoints/step-{step}"
        or seal["hf_export"] != str(root / f"hf/step-{step}")
        or seal["hf_export"] != model.uri
        or (root.scheme == "s3" and "tmp" in root.segments)
    ):
        raise ValueError("The certification seal identifies inconsistent or temporary outputs")
    receipts = StoragePath(model.certification_seal.path).parent
    completion = read_pin(FilePin(seal["completion_receipt"], seal["completion_sha256"]), Path.cwd())
    native = read_pin(FilePin(str(receipts / "native-checkpoint.json"), seal["native_receipt_sha256"]), Path.cwd())
    hf = read_pin(FilePin(str(receipts / "hf-export.json"), seal["hf_receipt_sha256"]), Path.cwd())
    if (
        completion["checkpoint"] != str(checkpoint)
        or completion["step"] != step
        or native["checkpoint"] != str(checkpoint)
        or native["step"] != step
        or native["is_temporary"]
        or native["verdict"] != "full"
        or hf["native_checkpoint"] != str(checkpoint)
        or hf["source"] != seal["hf_export"]
        or hf["step"] != step
        or hf["verdict"] != "full_hash_and_index"
    ):
        raise ValueError("The native, HF and completion receipts do not certify the same SFT checkpoint")
    manifest = pinned_json(prefix_join(model.uri, MODEL_MANIFEST_FILENAME), model.model_manifest_sha256)
    if (
        manifest.get("format_version") != 1
        or not manifest.get("files")
        or not str(manifest.get("identity", "")).startswith("sha256:")
    ):
        raise ValueError("The certified export's model manifest does not bind its payload files")


def patched_rl_recipe(recipe: dict, patch: FilePin | None) -> dict:
    """Merge a content-pinned typed patch before deriving curriculum-owned settings."""
    if patch is None:
        return recipe
    authored = RecipePatch.from_document(read_pin(patch, Path.cwd()))
    return merge_mappings(recipe, authored.to_skyrl())


def adopted_round_rl(spec: AdoptedRlRound, *, smoke: bool = False) -> ArtifactStep:
    """Adopt a pinned HF export or a native RL checkpoint without an SFT stage."""
    model = spec.adopted_model
    config = pinned_json(prefix_join(model.uri, "config.json"), model.config_sha256)
    if config["grugmoe_artifact_schema_version"] not in (1, 3):
        raise ValueError("RL adoption requires a schema-1 or schema-3 HF export")
    if (model.export_manifest_sha256 is None) == (model.certification_seal is None):
        raise ValueError(
            "Adopt an HF export through its export manifest or through its certification seal with the model manifest"
        )
    if model.certification_seal is not None:
        if model.export_manifest_sha256 is not None or model.model_manifest_sha256 is None:
            raise ValueError("Certified-export adoption binds the model manifest, not the export manifest")
        verified_certified_export(model)
    else:
        manifest = pinned_json(prefix_join(model.uri, "export-manifest.json"), model.export_manifest_sha256)
        if "request" in manifest:
            recorded_step = int(manifest["request"]["config"]["checkpoint"].rsplit("/step-", 1)[-1])
        else:
            recorded_step = int(manifest["step"])
        if recorded_step != model.checkpoint_step:
            raise ValueError("Adopted HF export checkpoint step changed")
    rl = spec.rl
    roles = draccus.decode(SkyRLRolePlan, rl.role_plan)
    execution = runtime_execution(rl.execution)
    if rl.openresearcher_service is not None:
        execution = service_execution(execution)
    job_limit = 2 * 3600 if smoke else 48 * 3600
    if execution.cluster != spec.cluster or execution.job_timeout_seconds != job_limit:
        raise ValueError("Adopted RL requires the round cluster and a 48-hour job limit (two hours for smoke)")
    recipe = patched_rl_recipe(json.loads(json.dumps(rl.recipe)), rl.recipe_patch)
    if "eval_drain" not in recipe["trainer"]["rollout_buffer"]:
        raise ValueError("Every adopted RL round must explicitly choose rollout_buffer.eval_drain: wait or abort")
    draft = round_draft(spec)
    if draft is not None:
        generator = recipe.setdefault("generator", {})
        if generator.get("speculative_decoding") is not None:
            raise ValueError(
                "A round cannot bind both a registered draft and an explicit generator.speculative_decoding block"
            )
        generator["speculative_decoding"] = draft.speculative_decoding()
    if rl.selection_eligibility not in {"covered", "complete_window"}:
        raise ValueError("Unknown checkpoint selection eligibility")
    adopted = ArtifactStep.adopt(
        model.name, model.version, model.uri, kind=LevanterCheckpoint, config=dataclasses.asdict(model)
    )
    profile = MochiProfile(spec.name, roles, draccus.decode(MegatronGeometry, rl.megatron), execution)
    validate_model_geometry(config, profile)
    pools_manifest_bytes = (StoragePath(prefix_join(rl.pools["uri"], "manifest.json"))).read_bytes()
    pools_manifest = json.loads(pools_manifest_bytes)
    curriculum = curriculum_from_manifest(pools_manifest)
    files = {
        **pools_manifest["files"],
        "manifest.json": hashlib.sha256(pools_manifest_bytes).hexdigest(),
    }
    for name, sha256 in files.items():
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"Round pool manifest digest is not a SHA-256 pin: {name}")
    pools = ArtifactStep.adopt(
        f"{spec.name}/pools",
        spec.version,
        rl.pools["uri"],
        config={"files": files, "curriculum": curriculum.identity},
    )
    training = tuple(
        ArtifactDataSource(pools, relative_path=f"{label}/train.parquet")
        for label in sorted(pools_manifest["composition"])
    )
    heldout = ArtifactDataSource(pools, relative_path="validation.parquet")
    # The curriculum derives the rollout buffer's per-route window quotas at the
    # run's batch size; a declared value must match it exactly or name nothing.
    derived_quotas = consumed_route_quotas(curriculum, pools_manifest["routes"], roles.train_batch_size)
    declared_quotas = recipe["trainer"]["rollout_buffer"].get("consumed_route_quotas")
    if declared_quotas is not None and declared_quotas != derived_quotas:
        raise ValueError(
            f"trainer.rollout_buffer.consumed_route_quotas={declared_quotas!r} disagrees with the quotas "
            f"the round curriculum derives at batch {roles.train_batch_size}: {derived_quotas!r}"
        )
    recipe["trainer"]["rollout_buffer"].pop("consumed_route_quotas", None)
    routes = rollout_source_routes(pools_manifest["routes"])
    declared_routes = recipe["trainer"]["rollout_buffer"].get("source_routes") or {}
    if set(declared_routes) != set(pools_manifest["routes"]):
        raise ValueError("Round source routes differ from the pools manifest's teacher-route map")
    recipe["trainer"]["rollout_buffer"]["source_routes"] = routes
    service = rl.openresearcher_service
    admission = pools_manifest.get("openresearcher")
    if (service is None) != (admission is None):
        raise ValueError(
            "The round's OpenResearcher service binding must match the pools manifest's openresearcher admission"
        )
    if service is not None:
        for field in ("dataset_id", "dataset_revision", "parquet_sha256"):
            if admission[field] != getattr(service, field):
                raise ValueError(f"OpenResearcher service {field} differs from the published pools")
        machines = service_recipe().to_skyrl()["environment"]["task_sessions"]["task_spec"]["task_machine_by_dataset"]
        recipe["environment"]["task_sessions"]["task_spec"]["task_machine_by_dataset"] = machines
    for callback in recipe["trainer"]["callbacks"]:
        if callback["type"] == "best_checkpoint":
            callback["expected_tasks"] = curriculum.validation.size
            callback["minimum_scored_tasks"] = (
                callback["expected_tasks"]
                if rl.selection_eligibility == "complete_window"
                else math.ceil(0.9 * callback["expected_tasks"])
            )
            callback["selection_eligibility"] = rl.selection_eligibility
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
        train_data=training,
        validation_data=(heldout,),
        max_steps=rl.max_steps,
        resume=resume,
        smoke=smoke,
        recipe_document=recipe,
        max_new_tokens=recipe["context_budget"]["max_new_tokens_per_turn"],
        gpu_variant=rl.gpu_variant,
        curriculum=curriculum,
        export_hf=rl.export_hf,
        retention=dataclasses.replace(RETENTION, resume_checkpoint_count=rl.retained_checkpoints),
    )
    original_build = step.build_config

    def verified_build(context):
        if not context.is_fingerprint:
            for name, sha256 in files.items():
                fs, key = filesystem_for(prefix_join(rl.pools["uri"], name))
                with fs.open(key, "rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != sha256:
                    raise ValueError(f"Round data changed: {rl.pools['uri']}/{name}")
        return original_build(context)

    step = dataclasses.replace(step, build_config=verified_build)
    if rl.openresearcher_service is not None:
        relative_path = "OpenResearcher:OpenResearcher-Dataset/train.parquet"
        step = dataclasses.replace(
            step,
            run=partial(
                run_with_retrieval,
                launch=retrieval_launch(rl.openresearcher_service, execution),
                parquet_relative_path=relative_path,
                parquet_sha256=pools_manifest["files"][relative_path],
            ),
        )
        step = dataclasses.replace(step, expected_fingerprint=step.fingerprint())
    return step


def adopted_rl_evaluation(spec: AdoptedRlRound, training: ArtifactStep) -> ArtifactStep:
    """Evaluate the RL-selected HF export through the frozen Micro210 pipeline."""
    if spec.evaluation != "frozen-micro210" or spec.evaluation_profile is None or not spec.rl.export_hf:
        raise ValueError("An adopted RL evaluation requires frozen-micro210, a serving profile and HF export")
    profile = registered_serving_profile(ServingProfileName(spec.evaluation_profile))
    best = micro210_artifact_step(
        SkyRLPolicyModel(training),
        name=f"{spec.name}/micro210",
        version=spec.version,
        serving_profile=profile,
        model_manifest_filename=MODEL_MANIFEST_FILENAME,
    )
    if spec.evaluation_checkpoints == "best_and_final":
        return best_and_final_evaluations(training, best, serving_profile=profile)
    if spec.evaluation_checkpoints != "best":
        raise ValueError("Unknown adopted RL evaluation checkpoint selection")
    return best


def adopted_round_evaluation(spec: AdoptedHfRound) -> ArtifactStep:
    """Bind a pinned HF export directly to evaluation, without training stages."""
    if spec.evaluation != "frozen-micro210":
        raise ValueError("Adopted evaluation rounds require frozen-micro210")
    model = spec.adopted_model
    if model.export_manifest_sha256 is None:
        raise ValueError("Adopted evaluation rounds bind the export manifest")
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
    spec = dataclasses.replace(spec, data_seed=round_spec.sft.data_seed)
    spec = dataclasses.replace(spec, xla_flags=tuple(dict.fromkeys((*spec.xla_flags, *round_spec.sft.xla_flags))))
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
    expected_budget = dataclasses.replace(round_spec.sft.training_budget, cooldown=expected_cooldown)
    if budget != expected_budget:
        raise ValueError("Built training budget differs from the round declaration")
    if (budget.cooldown if budget is not None else None) != expected_cooldown:
        raise ValueError("Built wall-clock cooldown differs from the round declaration")
    whole_run_lr_receipt(spec, schedule=schedule)
    return spec


def round_plan(
    round_spec: RoundSpec, release, source_revision: str, version: str, *, initial_eval_inputs: str, pools: RoundPools
) -> dict:
    """Derive the round's resolved plan: the shared template plus the round's fields."""
    resolved = production_plan(release, source_revision, baseline_files(), initial_eval_inputs=initial_eval_inputs)
    sft = resolved["sft"]
    sft["batch_size"] = round_spec.sft.batch_size
    sft["mesh"] = round_spec.sft.mesh
    sft["trainer"]["mesh"] = round_spec.sft.mesh
    sft["resources"] = round_spec.sft.resources
    sft["xla_flags"] = list(dict.fromkeys((*sft["xla_flags"], *round_spec.sft.xla_flags)))
    if round_spec.sft.first_collective_call_timeout is not None:
        sft["first_collective_call_timeout"] = round_spec.sft.first_collective_call_timeout
    if round_spec.sft.compiler_dumps:
        # Python re-tracing can run fused-CE host collectives; compiler dumps need no extra trace.
        sft["trainer"]["jax_config"].update(jax_explain_cache_misses=True, jax_log_compiles=True)
        sft["trainer"]["log_jaxprs"] = False
        sft["trainer"]["log_xla_hlo"] = False
        sft["trainer"]["xla_dump_upload"] = {"enabled": True, "ttl_days": 30}
        sft["xla_flags"] = [*sft["xla_flags"], "--xla_dump_to=/tmp/long-loop-xla", "--xla_dump_hlo_as_text"]
    if round_spec.sft.expert_weight_sharding:
        sft["model"]["expert_weight_sharding"] = round_spec.sft.expert_weight_sharding
    sft["trainer"]["training_budget"] = draccus.encode(round_spec.sft.training_budget)
    sft["stage_guard"] = draccus.encode(round_spec.sft.stage_guard)
    sft["task_timeout_seconds"] = round_spec.sft.task_timeout_seconds
    resolved["execution"]["task_timeouts"]["sft"] = round_spec.sft.task_timeout_seconds
    sft["data_seed"] = round_spec.sft.data_seed
    updates = bind_sft_stopping(sft, round_spec.sft.stopping, round_spec.sft.epoch_capacity)
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
    recipe = patched_rl_recipe(recipe, round_spec.rl_recipe_patch)
    recipe["trainer"]["callbacks"] = curriculum_validation_callbacks(
        recipe["trainer"]["callbacks"], pools.curriculum.validation
    )
    resolved["rl"] = {
        **rl,
        "name": f"{round_spec.slug}-rl",
        "config": recipe,
        "wall_clock_budget": dict(RL_TRAINING_BUDGET),
    }
    if round_spec.rl_layout is not None:
        layout = round_spec.rl_layout
        resolved["rl"]["role_plan"] = dataclasses.asdict(layout.role_plan)
        for path, value in layout.role_plan.config_values().items():
            set_path(recipe, path, value)
        geometry = dataclasses.asdict(layout.megatron)
        for role in ("policy", "ref"):
            recipe["trainer"][role]["megatron_config"].update(geometry)
        recipe["trainer"]["algorithm"].pop("tis_splice")
        harbor = recipe["terminal_bench"]["harbor"]
        harbor["daytona_image_build"] = harbor.pop("daytona_provisioning")
        harbor["eval_daytona_image_build"] = harbor.pop("eval_daytona_provisioning")
        recipe["extra_env"]["VERIFYIT_JUDGE_MODEL"] = "openai/gpt-oss-120b"
        routes = pools.dataset["routes"]
        buffer = recipe["trainer"]["rollout_buffer"]
        buffer["consumed_route_quotas"] = consumed_route_quotas(
            pools.curriculum, routes, layout.role_plan.train_batch_size
        )
        buffer["source_routes"] = routes
        resolved["gpu_peaks"]["rl"] = layout.role_plan.num_nodes * layout.role_plan.policy_num_gpus_per_node
        service = round_spec.openresearcher_service
        if service is not None:
            manifest = json.loads((StoragePath(pools.dataset["uri"]) / "manifest.json").read_bytes())
            admission = manifest["openresearcher"]
            for field in ("dataset_id", "dataset_revision", "parquet_sha256"):
                if admission[field] != getattr(service, field):
                    raise ValueError(f"OpenResearcher service {field} differs from the pools")
            machines = service_recipe().to_skyrl()["environment"]["task_sessions"]["task_spec"][
                "task_machine_by_dataset"
            ]
            recipe["environment"]["task_sessions"]["task_spec"]["task_machine_by_dataset"] = machines
            resolved["rl"]["openresearcher_service"] = dataclasses.asdict(service)
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
    resolved["datasets"] = {
        **{key: value for key, value in resolved["datasets"].items() if key not in ("agentic", "nonagentic", "heldout")},
        "pools": pools.dataset,
    }
    resolved["archive"]["hf_upload"]["repo_prefix"] = round_spec.archive.hf_upload_repo_prefix
    resolved["archive"]["evaldash"]["run_prefix"] = round_spec.archive.evaldash_run_prefix
    resolved["curriculum"] = {**dataclasses.asdict(pools.curriculum), "identity": pools.curriculum.identity}
    resolved["resolved_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in resolved.items() if key != "resolved_sha256"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return resolved


def round_pools(binding: PoolBinding) -> RoundPools:
    """Bind the published pools artifact, curriculum and dataset record for a round.

    The pool step drew the curriculum's holdout and recorded the split and the
    fixed curriculum's identity in its manifest; the curriculum reconstructed
    from that record must match it."""
    producer = binding.producer(binding.version, grader_environment=environment_artifact(GRADER_PACKAGES))["pools"]
    path = producer.path(binding.prefix)
    record_bytes = (StoragePath(path) / ".artifact.json").read_bytes()
    if json.loads(record_bytes)["fingerprint"] != producer.fingerprint():
        raise ValueError("Regional pool producer differs")
    manifest_bytes = (StoragePath(path) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    curriculum = curriculum_from_manifest(manifest)
    if "training_ready" in manifest and manifest["training_ready"] is not True:
        raise ValueError("Regional pools are not training-ready")
    pins = [
        {"uri": prefix_join(path, name), "sha256": sha256}
        for name, sha256 in {**manifest["files"], "manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}.items()
    ]
    pools = {
        "pools": ArtifactStep.adopt(
            producer.name,
            producer.version,
            path,
            config={
                "producer_fingerprint": producer.fingerprint(),
                "producer_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "files": pins,
            },
        )
    }
    datasets = {
        "uri": path,
        "rows": manifest["rows"],
        "files": pins,
        "composition": manifest["composition"],
        "routes": manifest["routes"],
    }
    return RoundPools(pools, curriculum, datasets)


def require_permanent_round_storage(
    round_spec: RoundSpec | AdoptedHfRound | AdoptedRlRound | RecoveryRound | MochiSftRound,
) -> None:
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


def run_round(handles: list[ArtifactStep], max_concurrent: int) -> None:
    if any(handle.run is run_mochi_sft for handle in handles):
        if get_job_info() is None:
            raise click.UsageError("Submit native Mochi SFT through an Iris CPU coordinator on cw-us-east-08a")
        run(*handles, max_concurrent=max_concurrent)
        return
    submit_or_run(
        "experiments.post_training.long_loop.round_runner",
        handles,
        max_concurrent,
        os.environ,
        argv=tuple(sys.argv[1:]),
    )


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--through", type=click.Choice(["sft", "rl", "eval", "archive"]), default="archive", show_default=True)
@click.option("--smoke", is_flag=True, help="Use two rollout groups for an adopted RL round with max_steps=2.")
@partial(build_options_with_runner, runner=run_round)
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
    if isinstance(round_spec, MochiSftRound):
        if through != "sft":
            raise click.UsageError("A Mochi native SFT round requires --through sft")
        os.environ.update(
            MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
        )
        resolved = resolve_mochi_sft(round_spec)
        resolved.receipt["source_revision"] = source_revision
        terminal = mochi_sft_step(round_spec, resolved)
        context = StepContext.for_run(terminal.path(round_spec.storage.output_prefix), round_spec.storage.output_prefix)
        config = terminal.build_config(context)
        validation_output.mkdir(parents=True, exist_ok=True)
        (validation_output / "resolved-config.json").write_text(json.dumps(draccus.encode(config.config), indent=2))
        (validation_output / "receipt.json").write_text(json.dumps(resolved.receipt, indent=2))
        validate_graph(terminal, validation_output, round_spec.storage.output_prefix)
        return terminal
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
        if through not in {"rl", "eval"}:
            raise click.UsageError("An adopted RL round requires --through rl or eval")
        os.environ.update(
            MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
        )
        terminal = adopted_round_rl(round_spec, smoke=smoke)
        plan = adopted_round_resolved_plan(round_spec, source_revision=source_revision, workspace=workspace)
        validation_output.mkdir(parents=True, exist_ok=True)
        (validation_output / "resolved-plan.json").write_text(json.dumps(plan, indent=2))
        validate_graph(
            terminal,
            validation_output,
            round_spec.storage.output_prefix,
            model_config=(StoragePath(round_spec.adopted_model.uri) / "config.json").read_bytes(),
            check_vocabulary=True,
        )
        if through == "eval":
            terminal = adopted_rl_evaluation(round_spec, terminal)
            # The selected model does not exist before RL. Its vocabulary is
            # verified by the evaluator's regional CPU gate after HF export.
            validate_graph(terminal, validation_output / "evaluation-plan", round_spec.storage.output_prefix)
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
        binding.record,
        binding.record_sha256,
        qualification=binding.qualification,
        evidence_names=binding.evidence_names,
        require_ready=click.get_current_context().params["do_run"],
    )
    version = resolve_version(round_spec.slug, round_spec.version)
    pool_binding = POOLS[round_spec.pools]
    pools = round_pools(pool_binding)
    resolved = round_plan(
        round_spec, release, source_revision, version, initial_eval_inputs=REFERENCE_INPUTS, pools=pools
    )
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    os.environ.update(
        MARIN_PREFIX=round_spec.storage.output_prefix, MARIN_TEMP_PREFIX=round_spec.storage.temporary_prefix
    )
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
        data_artifacts=pools.artifacts,
        curriculum=pools.curriculum,
        sft_release=handle,
        openresearcher_service=round_spec.openresearcher_service,
        sft_step_builder=partial(native_sft_step, spec_builder=partial(round_sft_spec, round_spec)),
    )
    validation_output.mkdir(parents=True, exist_ok=True)
    for key in ("sft_capacity", "sft_exposure"):
        (validation_output / f"{key}.json").write_text(json.dumps(resolved["sft"][key], indent=2) + "\n")
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
