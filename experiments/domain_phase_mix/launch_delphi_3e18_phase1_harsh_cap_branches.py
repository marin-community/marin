# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue one selected harsh-cap Delphi prefix over its frozen phase-1 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import fsspec
import jax
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.tracker.wandb import WandbConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, VersionedValue, this_output_path, versioned
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_harsh_cap_candidates as harsh
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_PREFIX = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_harsh_cap_branches_v6e8_20260825"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_CANDIDATE_WEIGHTS = harsh.DEFAULT_CANDIDATE_WEIGHTS
DEFAULT_CONTINUATION_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_branches_20260825"
DEFAULT_CONTINUATION_SUMMARY = DEFAULT_CONTINUATION_DIR / "continuation_summary.csv"
DEFAULT_CONTINUATION_WEIGHTS = DEFAULT_CONTINUATION_DIR / "continuation_weights.csv"
DEFAULT_DESIGN_MANIFEST = DEFAULT_CONTINUATION_DIR / "manifest.json"
LOCAL_DRY_RUN_DIR = DEFAULT_CONTINUATION_DIR / "launch_dry_run"
BRANCH_PROVENANCE_FILENAME = "branch_provenance.json"
EXPECTED_TPU_DEVICE_COUNT = 8
FIT_ROWS_PER_PREFIX = 80
REFEREE_ROWS_PER_PREFIX = 8
CONTROL_ROWS_PER_PREFIX = 8
ROWS_PER_PREFIX = FIT_ROWS_PER_PREFIX + REFEREE_ROWS_PER_PREFIX + CONTROL_ROWS_PER_PREFIX
DEFAULT_MAX_CONCURRENT = ROWS_PER_PREFIX
TOTAL_MATERIALIZED_EPOCH_CAP = 10.0
RUN_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_.-]+")
WANDB_TAG_MAX_LENGTH = 64
WANDB_HASH_TAG_LENGTH = 12
PANEL_HARDWARE_STATUS = "v6e_only_exact_prefix_continuation"


@dataclass(frozen=True)
class TpuHardware:
    tpu_type: str
    region: str
    zone: str


@dataclass(frozen=True)
class PrefixCheckpoint:
    candidate_id: str
    repeat_seed: int
    checkpoint_uri: str
    provenance_sha256: str


@dataclass(frozen=True)
class ObservedTpuHardware:
    platform: str
    device_kind: str
    global_device_count: int
    local_device_count: int


TPU_HARDWARE = TpuHardware(tpu_type="v6e-8", region="us-east5", zone="us-east5-b")


@dataclass(frozen=True)
class HarshBranchTrainingConfig:
    experiment_name: str
    analysis_output_path: str
    output_path: str
    run_spec: base.DelphiSwarmRunSpec
    validation_configs: dict[str, DatasetComponent] | None
    prefix_checkpoint: PrefixCheckpoint
    prefix_replay_code_commit: str
    candidate_weights_sha256: str
    candidate_aliases_sha256: str
    continuation_weights_sha256: str
    design_manifest_sha256: str
    continuation_id: str
    code_commit: str


@dataclass(frozen=True)
class SaveManifestConfig:
    experiment_name: str
    output_path: str
    selected_prefixes_json: str
    selected_prefixes_sha256: str
    candidate_weights_sha256: str
    candidate_aliases_sha256: str
    continuation_summary_sha256: str
    continuation_weights_sha256: str
    design_manifest_sha256: str
    prefix_replay_code_commit: str
    code_commit: str
    branch_run_id_base: int
    full_design_rows: int
    branch_rows_json: str
    manifest_identity: VersionedValue[str]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def phase_weights_sha256(phase_weights: dict[str, dict[str, float]]) -> str:
    return hashlib.sha256(json.dumps(phase_weights, sort_keys=True).encode()).hexdigest()


def hardware_from_run_spec(run_spec: base.DelphiSwarmRunSpec) -> TpuHardware:
    return TpuHardware(tpu_type=run_spec.tpu_type, region=run_spec.tpu_region, zone=run_spec.tpu_zone)


def observe_tpu_hardware() -> ObservedTpuHardware:
    devices = jax.devices()
    platforms = {device.platform for device in devices}
    kinds = {device.device_kind for device in devices}
    if platforms != {"tpu"} or len(kinds) != 1:
        raise ValueError(f"Expected one v6 TPU device kind, observed platforms={platforms}, kinds={kinds}")
    if len(devices) != EXPECTED_TPU_DEVICE_COUNT or jax.local_device_count() != EXPECTED_TPU_DEVICE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TPU_DEVICE_COUNT} v6e devices, observed "
            f"global={len(devices)}, local={jax.local_device_count()}"
        )
    kind = next(iter(kinds))
    if "v6" not in kind.lower():
        raise ValueError(f"Expected v6e hardware, observed {kind!r}")
    return ObservedTpuHardware(
        platform="tpu",
        device_kind=kind,
        global_device_count=len(devices),
        local_device_count=jax.local_device_count(),
    )


def selected_prefixes(
    uri: str,
    expected_sha256: str,
    expected_candidate_weights_sha256: str,
    expected_prefix_replay_code_commit: str,
) -> tuple[list[PrefixCheckpoint], dict[str, object]]:
    payload_bytes = read_uri_bytes(uri)
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Selected-prefix manifest changed: {actual_sha256} != {expected_sha256}")
    payload = json.loads(payload_bytes)
    if payload.get("candidate_weights_sha256") != expected_candidate_weights_sha256:
        raise ValueError("Selected-prefix manifest references different candidate weights")
    if payload.get("prefix_replay_code_commit") != expected_prefix_replay_code_commit:
        raise ValueError("Selected-prefix manifest references different prefix code")
    aliases = payload.get("selected_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("Selected-prefix manifest has no selected aliases")
    candidate_ids = tuple(str(row["canonical_candidate_id"]) for row in aliases)
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_ids) > 2:
        raise ValueError(f"Expected one or two distinct selected candidates, got {candidate_ids}")
    rows = [
        PrefixCheckpoint(
            candidate_id=str(row["canonical_candidate_id"]),
            repeat_seed=int(row["repeat_seed"]),
            checkpoint_uri=str(row["checkpoint_uri"]),
            provenance_sha256=str(row["provenance_sha256"]),
        )
        for row in payload.get("prefixes", [])
    ]
    if len(rows) != 2 * len(candidate_ids):
        raise ValueError("Selected prefixes must include primary and stability checkpoints")
    for candidate_id in candidate_ids:
        seeds = {row.repeat_seed for row in rows if row.candidate_id == candidate_id}
        if seeds != {0, 1}:
            raise ValueError(f"Prefix {candidate_id} has checkpoint seeds {sorted(seeds)}")
    return rows, payload


def source_prefix_specs(
    candidate_weights_path: Path,
    expected_candidate_weights_sha256: str,
    analysis_output_path: str,
) -> dict[tuple[str, int], base.DelphiSwarmRunSpec]:
    specs, _ = harsh.candidate_specs(
        candidate_weights_path=candidate_weights_path,
        expected_sha256=expected_candidate_weights_sha256,
        analysis_output_path=analysis_output_path,
        tpu_type=TPU_HARDWARE.tpu_type,
        tpu_region=TPU_HARDWARE.region,
        tpu_zone=TPU_HARDWARE.zone,
    )
    return {(harsh.candidate_id_for_spec(spec), spec.trainer_seed): spec for spec in specs}


def validate_selected_prefixes(
    prefixes: list[PrefixCheckpoint],
    specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
    candidate_weights_sha256: str,
    candidate_aliases_sha256: str,
    prefix_replay_code_commit: str,
) -> None:
    for prefix in prefixes:
        identity = (prefix.candidate_id, prefix.repeat_seed)
        if identity not in specs:
            raise ValueError(f"Unknown selected prefix {identity}")
        spec = specs[identity]
        if not prefix.checkpoint_uri.startswith("gs://marin-us-east5/"):
            raise ValueError(f"Prefix checkpoint is not east5-local: {prefix.checkpoint_uri}")
        if f"/{harsh.EXPERIMENT_NAME}/" not in prefix.checkpoint_uri:
            raise ValueError(f"Prefix checkpoint is outside the harsh-cap experiment: {prefix.checkpoint_uri}")
        expected_suffix = f"/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}"
        if not prefix.checkpoint_uri.endswith(expected_suffix):
            raise ValueError(f"Prefix checkpoint is not the exact boundary state: {prefix.checkpoint_uri}")
        fs, checkpoint_path = fsspec.core.url_to_fs(prefix.checkpoint_uri)
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not fs.exists(metadata_path):
            raise FileNotFoundError(f"Prefix checkpoint metadata is missing: {prefix.checkpoint_uri}")
        with fs.open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
            raise ValueError(f"Prefix checkpoint is not permanent: {metadata}")
        output_root = prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
        provenance_bytes = read_uri_bytes(f"{output_root}/{harsh.CANDIDATE_PROVENANCE_FILENAME}")
        if hashlib.sha256(provenance_bytes).hexdigest() != prefix.provenance_sha256:
            raise ValueError(f"Prefix provenance changed for {identity}")
        expected = {
            "experiment_name": harsh.EXPERIMENT_NAME,
            "candidate_id": prefix.candidate_id,
            "candidate_weights_sha256": candidate_weights_sha256,
            "candidate_aliases_sha256": candidate_aliases_sha256,
            "phase_weights_sha256": harsh.phase_weights_sha256(spec.phase_weights),
            "replay_code_commit": prefix_replay_code_commit,
            "run_name": spec.run_name,
            "run_order": spec.run_order,
            "run_id": spec.run_id,
            "data_seed": spec.data_seed,
            "trainer_seed": spec.trainer_seed,
            "checkpoint_uri": prefix.checkpoint_uri,
            "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
            "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
            "tpu_type": harsh.TPU_TYPE,
            "tpu_region": harsh.TPU_REGION,
            "tpu_zone": harsh.TPU_ZONE,
            "observed_device_kinds": json.loads(provenance_bytes)["observed_device_kinds"],
            "observed_global_device_count": harsh.EXPECTED_GLOBAL_DEVICE_COUNT,
            "observed_local_device_count": harsh.EXPECTED_GLOBAL_DEVICE_COUNT,
            "panel_hardware_status": harsh.PANEL_HARDWARE_STATUS,
        }
        provenance = json.loads(provenance_bytes)
        if provenance != expected or any("v6" not in kind.lower() for kind in provenance["observed_device_kinds"]):
            raise ValueError(f"Prefix provenance does not match frozen state for {identity}: {provenance}")


def load_design(
    summary_path: Path,
    expected_summary_sha256: str,
    weights_path: Path,
    expected_weights_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    candidate_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    for path, expected in (
        (summary_path, expected_summary_sha256),
        (weights_path, expected_weights_sha256),
        (manifest_path, expected_manifest_sha256),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen design artifact changed: {path} has {actual}, expected {expected}")
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("selected_candidate_ids", [])) != candidate_ids:
        raise ValueError("Design and selected-prefix identities disagree")
    if manifest.get("rows") != {
        "controls_per_prefix": CONTROL_ROWS_PER_PREFIX,
        "fit_per_prefix": FIT_ROWS_PER_PREFIX,
        "sealed_referees_per_prefix": REFEREE_ROWS_PER_PREFIX,
        "total": len(candidate_ids) * ROWS_PER_PREFIX,
    }:
        raise ValueError("Frozen branch allocation changed")
    summary = pd.read_csv(summary_path)
    weights = pd.read_csv(weights_path)
    if len(summary) != len(candidate_ids) * ROWS_PER_PREFIX:
        raise ValueError("Branch summary row count changed")
    rows: list[dict[str, object]] = []
    for summary_row in summary.itertuples(index=False):
        candidate_id = str(summary_row.prefix_candidate_id)
        continuation_id = str(summary_row.continuation_id)
        group = weights[weights.prefix_candidate_id.eq(candidate_id) & weights.continuation_id.eq(continuation_id)]
        counts = group.phase_1_count.to_numpy(dtype=int)
        phase1 = group.phase_1_weight.to_numpy(dtype=float)
        if len(group) != 39 or counts.sum() != replay.MIXTURE_BLOCK_SIZE:
            raise ValueError(f"Invalid continuation lattice for {candidate_id}/{continuation_id}")
        if not (phase1 == counts / replay.MIXTURE_BLOCK_SIZE).all():
            raise ValueError(f"Continuation weights are not runtime-exact for {candidate_id}/{continuation_id}")
        if float(group.total_materialized_epochs.max()) > TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12:
            raise ValueError(f"Total materialized epoch cap exceeded by {candidate_id}/{continuation_id}")
        for column in ("role", "fit_budget", "prefix_repeat_seed", "data_seed", "source"):
            if group[column].nunique(dropna=False) != 1 or group[column].iloc[0] != getattr(summary_row, column):
                raise ValueError(f"Summary/weight design mismatch for {candidate_id}/{continuation_id}: {column}")
        rows.append(
            {
                "prefix_candidate_id": candidate_id,
                "continuation_id": continuation_id,
                "role": str(summary_row.role),
                "fit_budget": bool(summary_row.fit_budget),
                "prefix_repeat_seed": int(summary_row.prefix_repeat_seed),
                "data_seed": int(summary_row.data_seed),
                "source": str(summary_row.source),
                "weights": dict(zip(group.bucket, phase1, strict=True)),
            }
        )
    for candidate_id in candidate_ids:
        candidate_rows = [row for row in rows if row["prefix_candidate_id"] == candidate_id]
        roles = pd.Series([row["role"] for row in candidate_rows]).value_counts().to_dict()
        if roles != {
            "fixed_prefix_response_fit": FIT_ROWS_PER_PREFIX,
            "sealed_geometry_referee": REFEREE_ROWS_PER_PREFIX,
            "prefix_state_tied_control": 4,
            "fresh_tied_control": 3,
            "common_random_tied_control": 1,
        }:
            raise ValueError(f"Branch roles changed for {candidate_id}: {roles}")
    return rows


def enrich_rows(
    rows: list[dict[str, object]],
    prefixes: list[PrefixCheckpoint],
    specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
    branch_run_id_base: int,
) -> list[dict[str, object]]:
    checkpoints = {(row.candidate_id, row.repeat_seed): row for row in prefixes}
    enriched: list[dict[str, object]] = []
    for run_order, row in enumerate(rows):
        identity = (str(row["prefix_candidate_id"]), int(row["prefix_repeat_seed"]))
        source = specs[identity]
        prefix = checkpoints[identity]
        continuation_id = str(row["continuation_id"])
        if RUN_NAME_PATTERN.fullmatch(continuation_id) is None:
            raise ValueError(f"Continuation identity is not run-name safe: {continuation_id}")
        phase_weights = {
            "phase_0": source.phase_weights["phase_0"],
            "phase_1": cast(dict[str, float], row["weights"]),
        }
        max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
        run_name = f"branch_{identity[0]}_seed{identity[1]}_{continuation_id}"
        enriched.append(
            {
                **row,
                "run_order": run_order,
                "run_id": branch_run_id_base + run_order,
                "run_name": run_name,
                "prefix": prefix,
                "phase_weights": phase_weights,
                "trainer_seed": source.trainer_seed,
                "max_simulated_epoch": max_epoch,
                "q95_simulated_epoch": q95_epoch,
                "mean_phase_tv_to_proportional": phase_tv,
            }
        )
    run_names = [str(row["run_name"]) for row in enriched]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Branch run identities are not unique")
    return enriched


def wandb_tags(config: HarshBranchTrainingConfig) -> list[str]:
    tags = [
        "issue-6611",
        "delphi-3e18-harsh-cap-branches",
        f"prefix={config.prefix_checkpoint.candidate_id}",
        f"prefix_seed={config.prefix_checkpoint.repeat_seed}",
        f"prefix_commit={config.prefix_replay_code_commit[:WANDB_HASH_TAG_LENGTH]}",
        f"branch_commit={config.code_commit[:WANDB_HASH_TAG_LENGTH]}",
        f"continuation={config.continuation_id}",
        f"design={config.design_manifest_sha256[:WANDB_HASH_TAG_LENGTH]}",
        f"data_seed={config.run_spec.data_seed}",
    ]
    oversized = [tag for tag in tags if len(tag) > WANDB_TAG_MAX_LENGTH]
    if oversized:
        raise ValueError(f"W&B tags exceed {WANDB_TAG_MAX_LENGTH} characters: {oversized}")
    return tags


def verify_prefix_on_worker(config: HarshBranchTrainingConfig) -> None:
    run_spec = config.run_spec
    if hardware_from_run_spec(run_spec) != TPU_HARDWARE:
        raise ValueError(f"Branch run hardware changed: {hardware_from_run_spec(run_spec)}")
    prefix = config.prefix_checkpoint
    fs, checkpoint_path = fsspec.core.url_to_fs(prefix.checkpoint_uri)
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not fs.exists(metadata_path):
        raise FileNotFoundError(f"Prefix checkpoint metadata is missing: {prefix.checkpoint_uri}")
    with fs.open(metadata_path) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
        raise ValueError(f"Worker received a non-permanent prefix checkpoint: {metadata}")
    output_root = prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    provenance_bytes = read_uri_bytes(f"{output_root}/{harsh.CANDIDATE_PROVENANCE_FILENAME}")
    if hashlib.sha256(provenance_bytes).hexdigest() != prefix.provenance_sha256:
        raise ValueError("Worker prefix provenance changed")


def run_phase_1_branch(config: HarshBranchTrainingConfig) -> None:
    """Restore one exact v6e prefix trainer state and continue through update 3007."""
    verify_prefix_on_worker(config)
    observed_hardware = observe_tpu_hardware()
    run_spec = config.run_spec
    expected = replay.phase_0_boundary(run_spec.train_steps, run_spec.batch_size)
    if expected != (replay.EXPECTED_PREFIX_TRAIN_STEPS, replay.EXPECTED_PREFIX_HF_STEP):
        raise ValueError(f"Phase boundary changed: {expected}")
    if run_spec.train_steps != replay.EXPECTED_FULL_TRAIN_STEPS:
        raise ValueError(f"Full training horizon changed: {run_spec.train_steps}")
    scaling_fits = base._read_scaling_fits(config.analysis_output_path)
    candidate = base._candidate_for_budget(scaling_fits=scaling_fits)
    if candidate.train_steps != run_spec.train_steps:
        raise ValueError(f"Resolved training horizon changed: {candidate.train_steps} != {run_spec.train_steps}")
    parameters = candidate.model_config.total_trainable_params(base.completed_adamh_heuristic.vocab_size)
    if int(parameters) != run_spec.total_trainable_params:
        raise ValueError(f"Resolved parameter count changed: {int(parameters)} != {run_spec.total_trainable_params}")
    inner = replay._prefix_train_config(
        run_spec=run_spec,
        candidate=candidate,
        validation_configs=config.validation_configs,
        replay_code_commit=config.code_commit,
    )
    tracker = inner.trainer.tracker
    if not isinstance(tracker, WandbConfig):
        raise ValueError(f"Expected a W&B tracker, got {type(tracker).__name__}")
    trainer = replace(
        inner.trainer,
        tracker=replace(tracker, tags=wandb_tags(config)),
        num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=config.prefix_checkpoint.checkpoint_uri,
    )
    inner = replace(
        inner,
        trainer=trainer,
        optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
        minimum_initial_step=replay.EXPECTED_PREFIX_TRAIN_STEPS,
    )
    resources = ResourceConfig.with_tpu(TPU_HARDWARE.tpu_type, regions=[TPU_HARDWARE.region], zone=TPU_HARDWARE.zone)
    run_levanter_train_lm(
        TrainLmOnPodConfig(
            train_config=inner,
            resources=resources,
            output_path=config.output_path,
            env_vars={
                "GIT_COMMIT": config.code_commit,
                "MARIN_PREFIX": marin_prefix_for_region(TPU_HARDWARE.region),
                base.SKIP_EVAL_HARNESS_ENV_VAR: "1",
            },
        )
    )
    terminal_uri = os.path.join(config.output_path, "checkpoints", f"step-{replay.EXPECTED_FULL_TRAIN_STEPS - 1}")
    fs, terminal_path = fsspec.core.url_to_fs(terminal_uri)
    metadata_path = os.path.join(terminal_path, "metadata.json")
    if not fs.exists(metadata_path):
        raise FileNotFoundError(f"Terminal branch checkpoint metadata is missing: {terminal_uri}")
    with fs.open(metadata_path) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_FULL_TRAIN_STEPS - 1 or metadata.get("is_temporary") is not False:
        raise ValueError(f"Branch terminal checkpoint is not permanent: {metadata}")
    provenance = {
        "experiment_name": config.experiment_name,
        "run_name": run_spec.run_name,
        "run_order": run_spec.run_order,
        "run_id": run_spec.run_id,
        "data_seed": run_spec.data_seed,
        "trainer_seed": run_spec.trainer_seed,
        "prefix_candidate_id": config.prefix_checkpoint.candidate_id,
        "prefix_repeat_seed": config.prefix_checkpoint.repeat_seed,
        "prefix_checkpoint_uri": config.prefix_checkpoint.checkpoint_uri,
        "prefix_provenance_sha256": config.prefix_checkpoint.provenance_sha256,
        "prefix_replay_code_commit": config.prefix_replay_code_commit,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_aliases_sha256": config.candidate_aliases_sha256,
        "continuation_weights_sha256": config.continuation_weights_sha256,
        "design_manifest_sha256": config.design_manifest_sha256,
        "continuation_id": config.continuation_id,
        "phase_weights_sha256": phase_weights_sha256(run_spec.phase_weights),
        "branch_code_commit": config.code_commit,
        "prefix_hardware": asdict(TPU_HARDWARE),
        "continuation_hardware": asdict(TPU_HARDWARE),
        "observed_continuation_hardware": asdict(observed_hardware),
        "minimum_initial_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "panel_hardware_status": PANEL_HARDWARE_STATUS,
        "terminal_checkpoint_uri": terminal_uri,
        "terminal_checkpoint_step": replay.EXPECTED_FULL_TRAIN_STEPS - 1,
    }
    output_fs, output_path = fsspec.core.url_to_fs(config.output_path)
    provenance_path = os.path.join(output_path, BRANCH_PROVENANCE_FILENAME)
    payload = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    if output_fs.exists(provenance_path):
        with output_fs.open(provenance_path, "rb") as handle:
            if handle.read() != payload:
                raise ValueError(f"Refusing to replace different branch provenance: {provenance_path}")
        return
    with output_fs.open(provenance_path, "wb") as handle:
        handle.write(payload)


def save_manifest(config: SaveManifestConfig) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    rows = json.loads(config.branch_rows_json)
    payload = {
        "experiment_name": config.experiment_name,
        "selected_prefixes": json.loads(config.selected_prefixes_json),
        "selected_prefixes_sha256": config.selected_prefixes_sha256,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "candidate_aliases_sha256": config.candidate_aliases_sha256,
        "continuation_summary_sha256": config.continuation_summary_sha256,
        "continuation_weights_sha256": config.continuation_weights_sha256,
        "design_manifest_sha256": config.design_manifest_sha256,
        "prefix_replay_code_commit": config.prefix_replay_code_commit,
        "code_commit": config.code_commit,
        "branch_run_id_base": config.branch_run_id_base,
        "prefix_hardware": asdict(TPU_HARDWARE),
        "continuation_hardware": asdict(TPU_HARDWARE),
        "panel_hardware_status": PANEL_HARDWARE_STATUS,
        "prefix_completed_updates": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "terminal_completed_updates": replay.EXPECTED_FULL_TRAIN_STEPS,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "full_design_rows": config.full_design_rows,
        "selected_design_rows": len(rows),
        "fit_budget_rows": sum(bool(row["fit_budget"]) for row in rows),
        "sealed_referee_rows": sum(row["role"] == "sealed_geometry_referee" for row in rows),
        "control_rows": sum("control" in row["role"] for row in rows),
        "branch_rows": rows,
        "manifest_identity": config.manifest_identity.value,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = os.path.join(config.output_path, "manifest.json")
    if fs.exists(path):
        with fs.open(path, "rb") as handle:
            if handle.read() != encoded:
                raise ValueError(f"Refusing to replace a different branch manifest: {path}")
        return
    with fs.open(path, "wb") as handle:
        handle.write(encoded)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-candidate-aliases-sha256", required=True)
    parser.add_argument("--continuation-summary", type=Path, default=DEFAULT_CONTINUATION_SUMMARY)
    parser.add_argument("--expected-continuation-summary-sha256", required=True)
    parser.add_argument("--continuation-weights", type=Path, default=DEFAULT_CONTINUATION_WEIGHTS)
    parser.add_argument("--expected-continuation-weights-sha256", required=True)
    parser.add_argument("--design-manifest", type=Path, default=DEFAULT_DESIGN_MANIFEST)
    parser.add_argument("--expected-design-manifest-sha256", required=True)
    parser.add_argument("--selected-prefixes", required=True)
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--branch-run-id-base", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output-dir", type=Path)
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if args.branch_run_id_base < 0:
        raise ValueError("--branch-run-id-base must be nonnegative")
    expected_prefix = marin_prefix_for_region(TPU_HARDWARE.region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())
    prefixes, selected_payload = selected_prefixes(
        args.selected_prefixes,
        args.expected_selected_prefixes_sha256,
        args.expected_candidate_sha256,
        args.prefix_replay_code_commit,
    )
    selected_aliases = selected_payload.get("selected_aliases")
    if not isinstance(selected_aliases, list) or not all(isinstance(row, dict) for row in selected_aliases):
        raise ValueError("Selected-prefix aliases are malformed")
    candidate_ids = tuple(str(row["canonical_candidate_id"]) for row in selected_aliases)
    panel_label = "_".join(candidate_id.split("_", maxsplit=1)[0] for candidate_id in candidate_ids)
    experiment_name = f"{EXPERIMENT_PREFIX}_{panel_label}"
    specs = source_prefix_specs(args.candidate_weights, args.expected_candidate_sha256, args.analysis_output_path)
    validate_selected_prefixes(
        prefixes,
        specs,
        args.expected_candidate_sha256,
        args.expected_candidate_aliases_sha256,
        args.prefix_replay_code_commit,
    )
    design = load_design(
        args.continuation_summary,
        args.expected_continuation_summary_sha256,
        args.continuation_weights,
        args.expected_continuation_weights_sha256,
        args.design_manifest,
        args.expected_design_manifest_sha256,
        candidate_ids,
    )
    all_rows = enrich_rows(design, prefixes, specs, args.branch_run_id_base)
    full_design_rows = len(all_rows)
    rows = all_rows
    if args.run_orders is not None:
        selected_orders = tuple(dict.fromkeys(args.run_orders))
        unknown = sorted(set(selected_orders) - {int(row["run_order"]) for row in rows})
        if unknown:
            raise ValueError(f"Unknown --run-order values: {unknown}")
        rows = [row for row in rows if int(row["run_order"]) in selected_orders]
    serializable_rows = [{**row, "prefix": asdict(row["prefix"])} for row in all_rows]
    manifest_identity = hashlib.sha256(
        json.dumps(
            {
                "selected_prefixes_sha256": args.expected_selected_prefixes_sha256,
                "candidate_weights_sha256": args.expected_candidate_sha256,
                "candidate_aliases_sha256": args.expected_candidate_aliases_sha256,
                "continuation_summary_sha256": args.expected_continuation_summary_sha256,
                "continuation_weights_sha256": args.expected_continuation_weights_sha256,
                "design_manifest_sha256": args.expected_design_manifest_sha256,
                "prefix_replay_code_commit": args.prefix_replay_code_commit,
                "branch_code_commit": code_commit,
                "branch_run_id_base": args.branch_run_id_base,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_config = SaveManifestConfig(
        experiment_name=experiment_name,
        output_path=str(args.dry_run_output_dir or LOCAL_DRY_RUN_DIR),
        selected_prefixes_json=json.dumps([asdict(row) for row in prefixes], sort_keys=True),
        selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
        candidate_weights_sha256=args.expected_candidate_sha256,
        candidate_aliases_sha256=args.expected_candidate_aliases_sha256,
        continuation_summary_sha256=args.expected_continuation_summary_sha256,
        continuation_weights_sha256=args.expected_continuation_weights_sha256,
        design_manifest_sha256=args.expected_design_manifest_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
        code_commit=code_commit,
        branch_run_id_base=args.branch_run_id_base,
        full_design_rows=full_design_rows,
        branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
        manifest_identity=versioned(manifest_identity),
    )
    if args.dry_run:
        save_manifest(manifest_config)
        logger.info(
            "Wrote the %d-row full manifest and selected %d phase-1 branch specs under %s",
            len(all_rows),
            len(rows),
            manifest_config.output_path,
        )
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for row in rows:
            prefix = cast(PrefixCheckpoint, row["prefix"])
            source = specs[(prefix.candidate_id, prefix.repeat_seed)]
            run_spec = replace(
                source,
                run_order=int(row["run_order"]),
                run_id=int(row["run_id"]),
                run_name=str(row["run_name"]),
                source_run_name=str(row["run_name"]),
                source_experiment=experiment_name,
                panel_source="sequential_phase1_harsh_cap_branch",
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
                max_simulated_epoch=float(row["max_simulated_epoch"]),
                q95_simulated_epoch=float(row["q95_simulated_epoch"]),
                mean_phase_tv_to_proportional=float(row["mean_phase_tv_to_proportional"]),
                phase_weights=cast(dict[str, dict[str, float]], row["phase_weights"]),
            )
            resources = ResourceConfig.with_tpu(
                TPU_HARDWARE.tpu_type, regions=[TPU_HARDWARE.region], zone=TPU_HARDWARE.zone
            )
            steps.append(
                ExecutorStep(
                    name=f"{experiment_name}/{run_spec.run_name}",
                    fn=remote(run_phase_1_branch, resources=resources, env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"}),
                    resources=resources,
                    config=HarshBranchTrainingConfig(
                        experiment_name=experiment_name,
                        analysis_output_path=args.analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_checkpoint=prefix,
                        prefix_replay_code_commit=args.prefix_replay_code_commit,
                        candidate_weights_sha256=args.expected_candidate_sha256,
                        candidate_aliases_sha256=args.expected_candidate_aliases_sha256,
                        continuation_weights_sha256=args.expected_continuation_weights_sha256,
                        design_manifest_sha256=args.expected_design_manifest_sha256,
                        continuation_id=str(row["continuation_id"]),
                        code_commit=code_commit,
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{experiment_name}/manifest",
                fn=save_manifest,
                config=replace(manifest_config, output_path=this_output_path()),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d branch steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{experiment_name}: fixed-prefix state-conditioned phase-1 response panel",
    )


if __name__ == "__main__":
    main()
