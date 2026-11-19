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
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
import yaml
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210_policy import Micro210Model, micro210_evaluation
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.archive import PERMANENT_TMP_SEGMENT
from experiments.post_training.long_loop.exp3_production import (
    INITIALIZER_RECEIPT,
    INITIALIZER_SHA256,
    baseline_files,
    production_plan,
)
from experiments.post_training.long_loop.exp3_recipe import (
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
from experiments.post_training.long_loop.optimizer_admission import optimizer_group_receipt
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.shared_sft import native_sft_step, shared_sft_spec
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph
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


def load_round(path: Path) -> RoundSpec:
    """Load a round YAML into the strict schema; unknown or mistyped keys fail."""
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping")
    spec = _decode_strict(document, RoundSpec, str(path))
    if spec.schema != 1:
        raise ValueError(f"{path} has unsupported schema {spec.schema}")
    return spec


def round_sft_spec(round_spec: RoundSpec, name: str, version: str, recipe: dict) -> SFTSpec:
    """Build the round's SFT spec, carrying the shared learning-rate invariants.

    The special-token rate defaults to 1.0 (R2); a round-level or resolved-section
    override must agree with the built spec. Whole-run annealing is checked here.
    """
    spec = shared_sft_spec(name, version, recipe)
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
    whole_run_lr_receipt(spec, min_lr_ratio=round_spec.sft.min_lr_ratio)
    return spec


def round_plan(round_spec: RoundSpec, release, source_revision: str, version: str) -> dict:
    """Derive the round's resolved plan: the shared template plus the round's fields."""
    pools_binding = POOLS[round_spec.pools]
    resolved = production_plan(release, source_revision, baseline_files())
    sft = resolved["sft"]
    sft["batch_size"] = round_spec.sft.batch_size
    sft["mesh"] = round_spec.sft.mesh
    sft["resources"] = round_spec.sft.resources
    updates = round_spec.sft.updates
    sft["stop_after_steps"] = updates
    sft["trainer"]["num_train_steps"] = updates
    sft["target_positions"] = updates * sft["batch_size"] * sft["seq_len"]
    sft["optimizer_schedule"] = {
        "horizon": updates,
        "warmup": 0,
        "decay": updates,
        "min_lr_ratio": round_spec.sft.min_lr_ratio,
        "lr_schedule": "linear",
    }
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


def require_permanent_round_storage(round_spec: RoundSpec) -> None:
    """R1: a round's outputs live at a permanent path, never under a scratch TTL prefix."""
    path = StoragePath(round_spec.storage.output_prefix)
    if PERMANENT_TMP_SEGMENT in path.segments:
        raise ValueError(
            f"Round storage.output_prefix must be a permanent path, not scratch storage: "
            f"{round_spec.storage.output_prefix}"
        )


def require_archive_binds_final_sft_checkpoint(resolved: dict) -> None:
    """R1: the archive stage must bind the SFT selection so it can protect and verify it."""
    stage_files = resolved["archive"].get("stage_files", {})
    sft_files = stage_files.get("sft", [])
    if not any(entry.get("path") == "selected.json" for entry in sft_files):
        raise ValueError("Round archive settings must bind the SFT stage's selected.json")


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@rl_build_options
def main(round_path: Path, source_revision: str, validation_output: Path) -> ArtifactStep:
    configure_coreweave_s3()
    round_spec = load_round(round_path)
    if click.get_current_context().params["default_version"] != round_spec.version:
        raise ValueError(f"--version must equal the round's version {round_spec.version}")
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    binding = DATA_RELEASES[round_spec.data_release]
    release, handle = release_artifact(
        binding.record, binding.record_sha256, require_ready=click.get_current_context().params["do_run"]
    )
    version = resolve_version(round_spec.slug, round_spec.version)
    resolved = round_plan(round_spec, release, source_revision, version)
    require_permanent_round_storage(round_spec)
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
        through="archive",
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
                min_lr_ratio=round_spec.sft.min_lr_ratio,
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
