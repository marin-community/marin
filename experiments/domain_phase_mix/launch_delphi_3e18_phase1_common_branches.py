# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue selected Delphi prefixes over a frozen fully crossed phase-1 panel."""

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
import numpy as np
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.tracker.wandb import WandbConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, VersionedValue, this_output_path, versioned
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_candidates as candidates
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_20260824"
DEFAULT_CANDIDATE_WEIGHTS = candidates.DEFAULT_CANDIDATE_WEIGHTS
DEFAULT_CONTINUATION_WEIGHTS = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_common_branches_20260824"
    / "continuation_weights.csv"
)
LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_common_branches_20260824"
    / "launch_dry_run"
)
SELECTED_PREFIX_COUNT = 4
COMMON_FIT_CONTINUATION_COUNT = 50
COMMON_CONTROL_CONTINUATION_COUNT = 3
COMMON_CONTINUATION_COUNT = COMMON_FIT_CONTINUATION_COUNT + COMMON_CONTROL_CONTINUATION_COUNT
PRIMARY_BRANCH_SEED = 0
STABILITY_BRANCH_SEED = 1
STABILITY_CONTINUATION_COUNT = 3
STABILITY_CONTROL_IDS = {"control_proportional", "control_incumbent_planned"}
BRANCH_NOISE_REPEAT_COUNT = 4
BRANCH_NOISE_PREFIX_CANDIDATE = "observed_cap10_best"
BRANCH_NOISE_CONTINUATION_ID = "control_proportional"
BRANCH_NOISE_DATA_SEED_BASE = 960_000
REQUIRED_SELECTED_CANDIDATES = {"observed_cap10_best"}
HISTORICAL_PHASE_1_EPOCH_CAP = 62.28165425173962
HISTORICAL_TOTAL_EPOCH_CAP = 255.8246349460757
BRANCH_RUN_ID_BASE = 950_000
TOTAL_BRANCH_ROWS = (
    SELECTED_PREFIX_COUNT * (COMMON_CONTINUATION_COUNT + 1 + STABILITY_CONTINUATION_COUNT) + BRANCH_NOISE_REPEAT_COUNT
)
DEFAULT_MAX_CONCURRENT = 56
SUPPORT_TOLERANCE = 1e-9
RUN_NAME_PATTERN = re.compile(r"[a-zA-Z0-9_.-]+")
BRANCH_PROVENANCE_FILENAME = "branch_provenance.json"
WANDB_TAG_MAX_LENGTH = 64
WANDB_HASH_TAG_LENGTH = 12


@dataclass(frozen=True)
class PrefixCheckpoint:
    candidate_id: str
    repeat_seed: int
    checkpoint_uri: str
    provenance_sha256: str


@dataclass(frozen=True)
class BranchTrainingConfig:
    analysis_output_path: str
    output_path: str
    run_spec: base.DelphiSwarmRunSpec
    validation_configs: dict[str, DatasetComponent] | None
    prefix_checkpoint: PrefixCheckpoint
    prefix_replay_code_commit: str
    candidate_weights_sha256: str
    continuation_weights_sha256: str
    continuation_id: str
    code_commit: str


@dataclass(frozen=True)
class SaveBranchManifestConfig:
    output_path: str
    selected_prefixes_json: str
    selected_prefixes_sha256: str
    candidate_weights_sha256: str
    continuation_weights_sha256: str
    prefix_replay_code_commit: str
    code_commit: str
    branch_rows_json: str
    selected_run_orders: VersionedValue[tuple[int, ...]]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def phase_weights_sha256(phase_weights: dict[str, dict[str, float]]) -> str:
    return hashlib.sha256(json.dumps(phase_weights, sort_keys=True).encode()).hexdigest()


def branch_wandb_tags(config: BranchTrainingConfig) -> list[str]:
    run_spec = config.run_spec
    tags = [
        "issue-6611",
        "delphi-3e18-phase1-common-branches",
        f"prefix_candidate={config.prefix_checkpoint.candidate_id}",
        f"prefix_repeat_seed={config.prefix_checkpoint.repeat_seed}",
        f"prefix_replay_commit={config.prefix_replay_code_commit[:WANDB_HASH_TAG_LENGTH]}",
        f"branch_commit={config.code_commit[:WANDB_HASH_TAG_LENGTH]}",
        f"continuation_id={config.continuation_id}",
        f"continuation_sha={config.continuation_weights_sha256[:WANDB_HASH_TAG_LENGTH]}",
        f"data_seed={run_spec.data_seed}",
        f"trainer_seed={run_spec.trainer_seed}",
    ]
    oversized = [tag for tag in tags if len(tag) > WANDB_TAG_MAX_LENGTH]
    if oversized:
        raise ValueError(f"W&B tags exceed {WANDB_TAG_MAX_LENGTH} characters: {oversized}")
    return tags


def verify_prefix_checkpoint_on_worker(config: BranchTrainingConfig) -> None:
    """Re-verify the exact prefix state inside the TPU task before training."""
    prefix = config.prefix_checkpoint
    fs, checkpoint_path = fsspec.core.url_to_fs(prefix.checkpoint_uri)
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not fs.exists(metadata_path):
        raise FileNotFoundError(f"Prefix checkpoint metadata is missing on worker: {prefix.checkpoint_uri}")
    with fs.open(metadata_path) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
        raise ValueError(f"Worker received a non-permanent prefix checkpoint: {metadata}")

    output_root = prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    provenance_uri = f"{output_root}/{candidates.CANDIDATE_PROVENANCE_FILENAME}"
    provenance_bytes = read_uri_bytes(provenance_uri)
    actual_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    if actual_sha256 != prefix.provenance_sha256:
        raise ValueError(f"Prefix provenance changed on worker: {actual_sha256} != {prefix.provenance_sha256}")

    candidate_position = candidates.CANDIDATE_IDS.index(prefix.candidate_id)
    seed_position = candidates.REPEAT_SEEDS.index(prefix.repeat_seed)
    prefix_weights = config.run_spec.phase_weights["phase_0"]
    expected = {
        "experiment_name": candidates.EXPERIMENT_NAME,
        "candidate_id": prefix.candidate_id,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "phase_weights_sha256": phase_weights_sha256({"phase_0": prefix_weights, "phase_1": prefix_weights}),
        "replay_code_commit": config.prefix_replay_code_commit,
        "run_name": f"prefix_{prefix.candidate_id}_seed{prefix.repeat_seed}",
        "run_order": candidate_position * len(candidates.REPEAT_SEEDS) + seed_position,
        "run_id": candidates.RUN_ID_BASE + candidate_position * len(candidates.REPEAT_SEEDS) + seed_position,
        "data_seed": candidates.DATA_SEED_BASE + prefix.repeat_seed,
        "trainer_seed": prefix.repeat_seed,
        "checkpoint_uri": prefix.checkpoint_uri,
        "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
    }
    provenance = json.loads(provenance_bytes)
    if provenance != expected:
        raise ValueError(f"Worker prefix provenance does not match the branch state: {provenance}")


def load_selected_prefixes(
    path: str,
    expected_sha256: str,
    expected_candidate_weights_sha256: str,
    expected_prefix_replay_code_commit: str,
    expected_phase_weights_sha256: dict[tuple[str, int], str],
) -> list[PrefixCheckpoint]:
    payload_bytes = read_uri_bytes(path)
    actual = hashlib.sha256(payload_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Selected-prefix manifest changed: {actual} != {expected_sha256}")
    payload = json.loads(payload_bytes)
    if payload.get("candidate_weights_sha256") != expected_candidate_weights_sha256:
        raise ValueError("Selected-prefix manifest references different candidate weights")
    if payload.get("prefix_replay_code_commit") != expected_prefix_replay_code_commit:
        raise ValueError("Selected-prefix checkpoint code differs from the continuation code")
    rows = [PrefixCheckpoint(**item) for item in payload["prefixes"]]
    candidate_ids = {row.candidate_id for row in rows}
    if len(candidate_ids) != SELECTED_PREFIX_COUNT:
        raise ValueError(f"Expected {SELECTED_PREFIX_COUNT} distinct selected prefixes")
    if not REQUIRED_SELECTED_CANDIDATES.issubset(candidate_ids):
        raise ValueError(f"Selected prefixes must include {sorted(REQUIRED_SELECTED_CANDIDATES)}")
    expected_rows = SELECTED_PREFIX_COUNT * 2
    if len(rows) != expected_rows:
        raise ValueError(f"Expected primary and stability checkpoints for each prefix ({expected_rows} rows)")
    for candidate_id in candidate_ids:
        seeds = {row.repeat_seed for row in rows if row.candidate_id == candidate_id}
        if seeds != {PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED}:
            raise ValueError(f"Prefix {candidate_id} has checkpoint seeds {sorted(seeds)}")
    for row in rows:
        if row.candidate_id not in candidates.CANDIDATE_IDS:
            raise ValueError(f"Unknown candidate prefix: {row.candidate_id}")
        if not row.checkpoint_uri.startswith("gs://marin-us-east5/"):
            raise ValueError(f"Prefix checkpoint is not east5-local: {row.checkpoint_uri}")
        experiment_fragment = f"/{candidates.EXPERIMENT_NAME}/"
        if experiment_fragment not in row.checkpoint_uri:
            raise ValueError(f"Prefix checkpoint is outside the frozen candidate experiment: {row.checkpoint_uri}")
        if not row.checkpoint_uri.endswith(f"/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}"):
            raise ValueError(f"Prefix checkpoint is not the exact post-update-2400 state: {row.checkpoint_uri}")
        expected_run_fragment = f"prefix_{row.candidate_id}_seed{row.repeat_seed}"
        if expected_run_fragment not in row.checkpoint_uri:
            raise ValueError(f"Prefix checkpoint URI does not match its candidate identity: {row.checkpoint_uri}")
        fs, checkpoint_path = fsspec.core.url_to_fs(row.checkpoint_uri)
        if not fs.exists(checkpoint_path):
            raise FileNotFoundError(f"Prefix checkpoint does not exist: {row.checkpoint_uri}")
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not fs.exists(metadata_path):
            raise FileNotFoundError(f"Prefix checkpoint metadata does not exist: {row.checkpoint_uri}")
        with fs.open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
            raise ValueError(f"Prefix checkpoint metadata is not the permanent boundary state: {metadata}")
        output_root = row.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
        provenance_uri = f"{output_root}/{candidates.CANDIDATE_PROVENANCE_FILENAME}"
        provenance_bytes = read_uri_bytes(provenance_uri)
        provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        if provenance_sha256 != row.provenance_sha256:
            raise ValueError(f"Prefix provenance changed: {provenance_sha256} != {row.provenance_sha256}")
        provenance = json.loads(provenance_bytes)
        expected_provenance = {
            "experiment_name": candidates.EXPERIMENT_NAME,
            "candidate_id": row.candidate_id,
            "candidate_weights_sha256": expected_candidate_weights_sha256,
            "phase_weights_sha256": expected_phase_weights_sha256[(row.candidate_id, row.repeat_seed)],
            "replay_code_commit": expected_prefix_replay_code_commit,
            "run_name": expected_run_fragment,
            "run_order": (
                candidates.CANDIDATE_IDS.index(row.candidate_id) * len(candidates.REPEAT_SEEDS)
                + candidates.REPEAT_SEEDS.index(row.repeat_seed)
            ),
            "run_id": (
                candidates.RUN_ID_BASE
                + candidates.CANDIDATE_IDS.index(row.candidate_id) * len(candidates.REPEAT_SEEDS)
                + candidates.REPEAT_SEEDS.index(row.repeat_seed)
            ),
            "data_seed": candidates.DATA_SEED_BASE + row.repeat_seed,
            "trainer_seed": row.repeat_seed,
            "checkpoint_uri": row.checkpoint_uri,
            "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
            "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        }
        if provenance != expected_provenance:
            raise ValueError(f"Prefix provenance does not match the selected checkpoint: {provenance}")
    return rows


def recover_epoch_scales(weights: pd.DataFrame, exposure_column: str, weight_column: str) -> dict[str, float]:
    scales = {}
    for bucket, rows in weights.groupby("bucket", sort=False):
        nonzero = rows[rows[weight_column].gt(0.0)]
        if nonzero.empty:
            raise ValueError(f"Cannot recover materialized-epoch scale for {bucket}")
        ratios = nonzero[exposure_column].to_numpy(dtype=float) / nonzero[weight_column].to_numpy(dtype=float)
        if not pd.Series(ratios).sub(ratios[0]).abs().le(1e-9).all():
            raise ValueError(f"Materialized-epoch scale changes for {bucket}")
        scales[str(bucket)] = float(ratios[0])
    return scales


def load_continuations(
    path: Path,
    expected_sha256: str,
    candidate_weights_path: Path,
    expected_candidate_weights_sha256: str,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Continuation weights changed: {actual} != {expected_sha256}")
    frame = pd.read_csv(path)
    required = {
        "continuation_id",
        "role",
        "fit_budget",
        "bucket",
        "phase_1_count",
        "phase_1_weight",
        "phase_1_materialized_epochs",
        "historical_phase_1_bucket_epoch_cap",
        "historical_total_bucket_epoch_cap",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Continuation weights are missing columns: {sorted(required - set(frame.columns))}")
    candidate_actual = file_sha256(candidate_weights_path)
    if candidate_actual != expected_candidate_weights_sha256:
        raise ValueError(f"Candidate weights changed: {candidate_actual} != {expected_candidate_weights_sha256}")
    candidate_frame = pd.read_csv(candidate_weights_path)
    candidate_required = {
        "candidate_id",
        "bucket",
        "phase_0_weight",
        "phase_0_materialized_epochs",
    }
    if not candidate_required.issubset(candidate_frame.columns):
        raise ValueError(
            f"Candidate weights are missing columns: {sorted(candidate_required - set(candidate_frame.columns))}"
        )
    phase_0_scales = recover_epoch_scales(candidate_frame, "phase_0_materialized_epochs", "phase_0_weight")
    phase_1_scales = recover_epoch_scales(frame, "phase_1_materialized_epochs", "phase_1_weight")
    continuation_ids = tuple(frame.continuation_id.drop_duplicates())
    if len(continuation_ids) != COMMON_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_CONTINUATION_COUNT} common continuations")
    buckets = tuple(frame.loc[frame.continuation_id.eq(continuation_ids[0]), "bucket"])
    rows = []
    for continuation_id in continuation_ids:
        group = frame[frame.continuation_id.eq(continuation_id)]
        if tuple(group.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {continuation_id}")
        counts = group.phase_1_count.to_numpy(dtype=int)
        weights = group.phase_1_weight.to_numpy(dtype=float)
        if counts.sum() != replay.MIXTURE_BLOCK_SIZE or not (counts >= 0).all():
            raise ValueError(f"Invalid runtime counts for {continuation_id}")
        if not (weights == counts / replay.MIXTURE_BLOCK_SIZE).all():
            raise ValueError(f"Weights are not runtime-exact for {continuation_id}")
        if float(group.phase_1_materialized_epochs.max()) > HISTORICAL_PHASE_1_EPOCH_CAP + 1e-12:
            raise ValueError(f"Historical phase-1 support exceeded by {continuation_id}")
        expected_exposure = pd.Series(
            [weights[position] * phase_1_scales[str(bucket)] for position, bucket in enumerate(group.bucket)],
            index=group.index,
        )
        if not group.phase_1_materialized_epochs.sub(expected_exposure).abs().le(1e-9).all():
            raise ValueError(f"Stored phase-1 exposure changed for {continuation_id}")
        phase_1_bucket_caps = group.historical_phase_1_bucket_epoch_cap.to_numpy(dtype=float)
        total_bucket_caps = group.historical_total_bucket_epoch_cap.to_numpy(dtype=float)
        if not np.isclose(float(phase_1_bucket_caps.max()), HISTORICAL_PHASE_1_EPOCH_CAP, atol=SUPPORT_TOLERANCE):
            raise ValueError("Frozen phase-1 support envelope changed")
        if not np.isclose(float(total_bucket_caps.max()), HISTORICAL_TOTAL_EPOCH_CAP, atol=SUPPORT_TOLERANCE):
            raise ValueError("Frozen total-exposure support envelope changed")
        if np.any(expected_exposure.to_numpy(dtype=float) > phase_1_bucket_caps + 1e-12):
            raise ValueError(f"Per-bucket phase-1 support exceeded by {continuation_id}")
        maximum_total_exposure = 0.0
        for _, prefix_rows in candidate_frame.groupby("candidate_id", sort=False):
            if tuple(prefix_rows.bucket) != tuple(group.bucket):
                raise ValueError("Candidate and continuation bucket orders disagree")
            phase_0_exposure = prefix_rows.phase_0_weight.to_numpy(dtype=float) * pd.Series(
                [phase_0_scales[str(bucket)] for bucket in prefix_rows.bucket]
            ).to_numpy(dtype=float)
            maximum_total_exposure = max(
                maximum_total_exposure,
                float((phase_0_exposure + expected_exposure.to_numpy(dtype=float)).max()),
            )
            if np.any(phase_0_exposure + expected_exposure.to_numpy(dtype=float) > total_bucket_caps + 1e-12):
                raise ValueError(f"Per-bucket total-exposure support exceeded by {continuation_id}")
        if maximum_total_exposure > HISTORICAL_TOTAL_EPOCH_CAP + SUPPORT_TOLERANCE:
            raise ValueError(f"Historical total-exposure support exceeded by {continuation_id}")
        fit_budget_values = set(group.fit_budget)
        if len(fit_budget_values) != 1:
            raise ValueError(f"Fit-budget flag changes within {continuation_id}")
        rows.append(
            {
                "continuation_id": continuation_id,
                "role": str(group.role.iloc[0]),
                "fit_budget": bool(next(iter(fit_budget_values))),
                "max_phase_1_materialized_epoch": float(group.phase_1_materialized_epochs.max()),
                "max_total_materialized_epoch_across_candidate_prefixes": maximum_total_exposure,
                "weights": dict(zip(buckets, weights, strict=True)),
            }
        )
    if sum(bool(row["fit_budget"]) for row in rows) != COMMON_FIT_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_FIT_CONTINUATION_COUNT} fit-budget continuations")
    if sum(not bool(row["fit_budget"]) for row in rows) != COMMON_CONTROL_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_CONTROL_CONTINUATION_COUNT} common controls")
    return buckets, rows


def source_prefix_specs(
    *,
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> dict[tuple[str, int], base.DelphiSwarmRunSpec]:
    specs, _ = candidates.candidate_specs(
        candidate_weights_path=candidate_weights_path,
        expected_sha256=candidate_weights_sha256,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    by_identity = {}
    for spec in specs:
        suffix = f"_seed{spec.trainer_seed}"
        candidate_id = spec.run_name.removeprefix("prefix_").removesuffix(suffix)
        by_identity[(candidate_id, spec.trainer_seed)] = spec
    expected = {
        (candidate_id, seed)
        for candidate_id in candidates.CANDIDATE_IDS
        for seed in (PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED)
    }
    if not expected.issubset(by_identity):
        raise ValueError("Primary or stability candidate prefix specs are incomplete")
    return by_identity


def branch_rows(
    *,
    prefixes: list[PrefixCheckpoint],
    prefix_specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
    continuations: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    run_order = 0
    primary_prefixes = [row for row in prefixes if row.repeat_seed == PRIMARY_BRANCH_SEED]
    stability_prefixes = {row.candidate_id: row for row in prefixes if row.repeat_seed == STABILITY_BRANCH_SEED}
    stability_continuations = [row for row in continuations if row["continuation_id"] in STABILITY_CONTROL_IDS]
    if {row["continuation_id"] for row in stability_continuations} != STABILITY_CONTROL_IDS:
        raise ValueError("Stability controls are incomplete")
    highest_exposure = max(
        (row for row in continuations if bool(row["fit_budget"])),
        key=lambda row: float(row["max_phase_1_materialized_epoch"]),
    )
    stability_continuations.append(highest_exposure)
    if len(stability_continuations) != STABILITY_CONTINUATION_COUNT:
        raise ValueError("Stability-sentinel continuation count changed")
    for prefix in primary_prefixes:
        source = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
        phase_0_weights = source.phase_weights["phase_0"]
        for continuation in continuations:
            rows.append(
                {
                    "run_order": run_order,
                    "fit_budget": bool(continuation["fit_budget"]),
                    "branch_role": "primary_cross",
                    "prefix": prefix,
                    "continuation_id": continuation["continuation_id"],
                    "continuation_role": continuation["role"],
                    "phase_weights": {
                        "phase_0": phase_0_weights,
                        "phase_1": continuation["weights"],
                    },
                }
            )
            run_order += 1
        rows.append(
            {
                "run_order": run_order,
                "fit_budget": False,
                "branch_role": "prefix_tied_control",
                "prefix": prefix,
                "continuation_id": "tied_control",
                "continuation_role": "tied_control",
                "phase_weights": {"phase_0": phase_0_weights, "phase_1": phase_0_weights},
            }
        )
        run_order += 1

        stability_prefix = stability_prefixes[prefix.candidate_id]
        stability_source = prefix_specs[(stability_prefix.candidate_id, stability_prefix.repeat_seed)]
        stability_phase_0 = stability_source.phase_weights["phase_0"]
        for continuation in stability_continuations:
            rows.append(
                {
                    "run_order": run_order,
                    "fit_budget": False,
                    "branch_role": "prefix_seed_stability_sentinel",
                    "prefix": stability_prefix,
                    "continuation_id": continuation["continuation_id"],
                    "continuation_role": continuation["role"],
                    "phase_weights": {
                        "phase_0": stability_phase_0,
                        "phase_1": continuation["weights"],
                    },
                }
            )
            run_order += 1

    noise_prefix = next(prefix for prefix in primary_prefixes if prefix.candidate_id == BRANCH_NOISE_PREFIX_CANDIDATE)
    noise_source = prefix_specs[(noise_prefix.candidate_id, noise_prefix.repeat_seed)]
    noise_continuation = next(
        continuation for continuation in continuations if continuation["continuation_id"] == BRANCH_NOISE_CONTINUATION_ID
    )
    for repeat_index in range(BRANCH_NOISE_REPEAT_COUNT):
        rows.append(
            {
                "run_order": run_order,
                "fit_budget": False,
                "branch_role": "same_prefix_branch_noise",
                "prefix": noise_prefix,
                "continuation_id": f"{BRANCH_NOISE_CONTINUATION_ID}_noise{repeat_index + 1}",
                "continuation_role": "same_prefix_branch_noise",
                "noise_group_id": f"{BRANCH_NOISE_PREFIX_CANDIDATE}/{BRANCH_NOISE_CONTINUATION_ID}",
                "branch_noise_repeat_index": repeat_index + 1,
                "data_seed": BRANCH_NOISE_DATA_SEED_BASE + repeat_index,
                "phase_weights": {
                    "phase_0": noise_source.phase_weights["phase_0"],
                    "phase_1": noise_continuation["weights"],
                },
            }
        )
        run_order += 1
    if sum(bool(row["fit_budget"]) for row in rows) != SELECTED_PREFIX_COUNT * COMMON_FIT_CONTINUATION_COUNT:
        raise ValueError("Round-1 fit budget changed")
    if len(rows) != TOTAL_BRANCH_ROWS:
        raise ValueError("Round-1 branch count changed")
    return rows


def enrich_branch_rows(
    rows: list[dict[str, object]],
    prefix_specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
) -> list[dict[str, object]]:
    enriched = []
    for row in rows:
        prefix = row["prefix"]
        source = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
        continuation_id = str(row["continuation_id"])
        if RUN_NAME_PATTERN.fullmatch(continuation_id) is None:
            raise ValueError(f"Continuation identity is not run-name safe: {continuation_id!r}")
        run_order = int(row["run_order"])
        run_name = f"branch_{prefix.candidate_id}_seed{prefix.repeat_seed}_{continuation_id}"
        max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(row["phase_weights"])
        enriched.append(
            {
                **row,
                "run_id": BRANCH_RUN_ID_BASE + run_order,
                "run_name": run_name,
                "data_seed": int(row.get("data_seed", source.data_seed)),
                "trainer_seed": source.trainer_seed,
                "max_simulated_epoch": max_epoch,
                "q95_simulated_epoch": q95_epoch,
                "mean_phase_tv_to_proportional": phase_tv,
            }
        )
    identities = [(row["run_order"], row["run_id"], row["run_name"]) for row in enriched]
    if len(identities) != len(set(identities)):
        raise ValueError("Branch run identities are not unique")
    return enriched


def run_phase_1_branch(config: BranchTrainingConfig) -> None:
    """Restore one exact prefix trainer state and continue through update 3007."""
    run_spec = config.run_spec
    verify_prefix_checkpoint_on_worker(config)
    expected_prefix_steps, expected_prefix_hf_step = replay.phase_0_boundary(run_spec.train_steps, run_spec.batch_size)
    if (expected_prefix_steps, expected_prefix_hf_step) != (
        replay.EXPECTED_PREFIX_TRAIN_STEPS,
        replay.EXPECTED_PREFIX_HF_STEP,
    ):
        raise ValueError(
            "Phase boundary changed: "
            f"{(expected_prefix_steps, expected_prefix_hf_step)} != "
            f"{(replay.EXPECTED_PREFIX_TRAIN_STEPS, replay.EXPECTED_PREFIX_HF_STEP)}"
        )
    if run_spec.train_steps != replay.EXPECTED_FULL_TRAIN_STEPS:
        raise ValueError(f"Full training horizon changed: {run_spec.train_steps}")
    scaling_fits = base._read_scaling_fits(config.analysis_output_path)
    candidate = base._candidate_for_budget(scaling_fits=scaling_fits)
    if candidate.train_steps != run_spec.train_steps:
        raise ValueError(f"Resolved training horizon changed: {candidate.train_steps} != {run_spec.train_steps}")
    params = candidate.model_config.total_trainable_params(base.completed_adamh_heuristic.vocab_size)
    if int(params) != run_spec.total_trainable_params:
        raise ValueError(f"Resolved parameter count changed: {int(params)} != {run_spec.total_trainable_params}")
    inner = replay._prefix_train_config(
        run_spec=run_spec,
        candidate=candidate,
        validation_configs=config.validation_configs,
        replay_code_commit=config.code_commit,
    )
    tracker_config = inner.trainer.tracker
    if not isinstance(tracker_config, WandbConfig):
        raise ValueError(f"Expected a W&B tracker for branch training, got {type(tracker_config).__name__}")
    tracker = replace(
        tracker_config,
        tags=branch_wandb_tags(config),
    )
    trainer = replace(
        inner.trainer,
        tracker=tracker,
        num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
        # None first resumes a partial checkpoint from this branch output. If no
        # local checkpoint exists, Levanter initializes from the exact prefix.
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=config.prefix_checkpoint.checkpoint_uri,
    )
    if trainer.num_train_steps != replay.EXPECTED_FULL_TRAIN_STEPS:
        raise ValueError(f"Branch trainer horizon changed: {trainer.num_train_steps}")
    inner = replace(
        inner,
        trainer=trainer,
        optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
        minimum_initial_step=replay.EXPECTED_PREFIX_TRAIN_STEPS,
    )
    if inner.optimizer_schedule_num_train_steps != run_spec.train_steps:
        raise ValueError("Branch optimizer schedule no longer matches the full training horizon")
    resources = ResourceConfig.with_tpu(
        run_spec.tpu_type,
        regions=[run_spec.tpu_region],
        zone=run_spec.tpu_zone,
    )
    run_levanter_train_lm(
        TrainLmOnPodConfig(
            train_config=inner,
            resources=resources,
            output_path=config.output_path,
            env_vars={
                "GIT_COMMIT": config.code_commit,
                "MARIN_PREFIX": marin_prefix_for_region(run_spec.tpu_region),
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
        "experiment_name": EXPERIMENT_NAME,
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
        "continuation_weights_sha256": config.continuation_weights_sha256,
        "continuation_id": config.continuation_id,
        "phase_weights_sha256": phase_weights_sha256(run_spec.phase_weights),
        "branch_code_commit": config.code_commit,
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


def save_branch_manifest(config: SaveBranchManifestConfig) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    branch_rows = json.loads(config.branch_rows_json)
    fit_budget_rows = sum(bool(row["fit_budget"]) for row in branch_rows)
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "selected_prefixes": json.loads(config.selected_prefixes_json),
        "selected_prefixes_sha256": config.selected_prefixes_sha256,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "continuation_weights_sha256": config.continuation_weights_sha256,
        "prefix_replay_code_commit": config.prefix_replay_code_commit,
        "code_commit": config.code_commit,
        "prefix_completed_updates": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "terminal_completed_updates": replay.EXPECTED_FULL_TRAIN_STEPS,
        "terminal_checkpoint_step": replay.EXPECTED_FULL_TRAIN_STEPS - 1,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "expected_full_design_rows": TOTAL_BRANCH_ROWS,
        "selected_design_rows": len(branch_rows),
        "selected_run_orders": [row["run_order"] for row in branch_rows],
        "fit_budget_rows": fit_budget_rows,
        "control_rows": len(branch_rows) - fit_budget_rows,
        "same_prefix_branch_noise_rows": sum(row["branch_role"] == "same_prefix_branch_noise" for row in branch_rows),
        "noise_estimation": (
            "four phase-1 data-seed repeats hold the observed-incumbent seed-0 prefix checkpoint and "
            "proportional continuation fixed, but changing the data seed also changes the document permutation and "
            "therefore phase-0/phase-1 overlap; compare them with the seed-930000 proportional control rather than "
            "treating them as pure operating-stream noise. Whole-run prefix-seed changes remain confounded"
        ),
        "prefix_selection_caveat": (
            "observed_cap10_best is outcome-selected and protected as an incumbent; frontier claims at that prefix "
            "must be checked against the three KL-materialized prefixes and prefix-seed stability sentinels"
        ),
        "branch_rows": branch_rows,
    }
    with fs.open(os.path.join(config.output_path, "manifest.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--continuation-weights", type=Path, default=DEFAULT_CONTINUATION_WEIGHTS)
    parser.add_argument("--expected-continuation-sha256", required=True)
    parser.add_argument("--selected-prefixes", required=True)
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {base.DEFAULT_TPU_REGION}/{base.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())

    buckets, continuations = load_continuations(
        args.continuation_weights,
        args.expected_continuation_sha256,
        args.candidate_weights,
        args.expected_candidate_sha256,
    )
    prefix_specs = source_prefix_specs(
        candidate_weights_path=args.candidate_weights,
        candidate_weights_sha256=args.expected_candidate_sha256,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    expected_phase_hashes = {
        identity: phase_weights_sha256(spec.phase_weights) for identity, spec in prefix_specs.items()
    }
    prefixes = load_selected_prefixes(
        args.selected_prefixes,
        args.expected_selected_prefixes_sha256,
        args.expected_candidate_sha256,
        args.prefix_replay_code_commit,
        expected_phase_hashes,
    )
    runtime_buckets = tuple(next(iter(prefix_specs.values())).phase_weights["phase_0"])
    if set(runtime_buckets) != set(buckets):
        raise ValueError("Prefix and continuation bucket sets disagree")
    for continuation in continuations:
        weights = cast(dict[str, float], continuation["weights"])
        continuation["weights"] = {bucket: weights[bucket] for bucket in runtime_buckets}
    rows = enrich_branch_rows(
        branch_rows(prefixes=prefixes, prefix_specs=prefix_specs, continuations=continuations),
        prefix_specs,
    )
    if args.run_orders is not None:
        selected_orders = tuple(dict.fromkeys(args.run_orders))
        unknown_orders = sorted(set(selected_orders) - {int(row["run_order"]) for row in rows})
        if unknown_orders:
            raise ValueError(f"Unknown --run-order values: {unknown_orders}")
        rows = [row for row in rows if int(row["run_order"]) in selected_orders]
    serializable_rows = []
    for row in rows:
        prefix = row["prefix"]
        serializable_rows.append({**row, "prefix": asdict(prefix)})

    if args.dry_run:
        save_branch_manifest(
            SaveBranchManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                selected_prefixes_json=json.dumps([asdict(row) for row in prefixes], sort_keys=True),
                selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
                candidate_weights_sha256=args.expected_candidate_sha256,
                continuation_weights_sha256=args.expected_continuation_sha256,
                prefix_replay_code_commit=args.prefix_replay_code_commit,
                code_commit=code_commit,
                branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
                selected_run_orders=versioned(tuple(int(row["run_order"]) for row in serializable_rows)),
            )
        )
        logger.info("Wrote %d phase-1 branch specs under %s", len(rows), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for row in rows:
            prefix = row["prefix"]
            source = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
            run_order = int(row["run_order"])
            run_name = str(row["run_name"])
            run_spec = replace(
                source,
                run_order=run_order,
                run_id=int(row["run_id"]),
                run_name=run_name,
                source_run_name=run_name,
                source_experiment=EXPERIMENT_NAME,
                panel_source="sequential_phase1_common_branch",
                data_seed=int(row["data_seed"]),
                trainer_seed=int(row["trainer_seed"]),
                max_simulated_epoch=float(row["max_simulated_epoch"]),
                q95_simulated_epoch=float(row["q95_simulated_epoch"]),
                mean_phase_tv_to_proportional=float(row["mean_phase_tv_to_proportional"]),
                phase_weights=row["phase_weights"],
            )
            resources = ResourceConfig.with_tpu(
                run_spec.tpu_type,
                regions=[run_spec.tpu_region],
                zone=run_spec.tpu_zone,
            )
            steps.append(
                ExecutorStep(
                    name=f"{EXPERIMENT_NAME}/{run_name}",
                    fn=remote(
                        run_phase_1_branch,
                        resources=resources,
                        env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                    ),
                    resources=resources,
                    config=BranchTrainingConfig(
                        analysis_output_path=args.analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_checkpoint=prefix,
                        prefix_replay_code_commit=args.prefix_replay_code_commit,
                        candidate_weights_sha256=args.expected_candidate_sha256,
                        continuation_weights_sha256=args.expected_continuation_sha256,
                        continuation_id=str(row["continuation_id"]),
                        code_commit=code_commit,
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/manifest",
                fn=save_branch_manifest,
                config=SaveBranchManifestConfig(
                    output_path=this_output_path(),
                    selected_prefixes_json=json.dumps([asdict(row) for row in prefixes], sort_keys=True),
                    selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
                    candidate_weights_sha256=args.expected_candidate_sha256,
                    continuation_weights_sha256=args.expected_continuation_sha256,
                    prefix_replay_code_commit=args.prefix_replay_code_commit,
                    code_commit=code_commit,
                    branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
                    selected_run_orders=versioned(tuple(int(row["run_order"]) for row in serializable_rows)),
                ),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d branch steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{EXPERIMENT_NAME}: fully crossed state-conditioned phase-1 continuation panel",
    )


if __name__ == "__main__":
    main()
