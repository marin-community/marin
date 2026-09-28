# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen review-v9 StarCoder WSD80 gradient-conflict training panel.

The scientific design and operational release are separate artifacts. Manifest
and runtime audits do not require a release, while execution is impossible
until a hash-pinned release records that every training gate passed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import logging
import os
import urllib.parse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import fsspec
import numpy as np
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.store.cache import CACHE_LAYOUT_CONSOLIDATED, CacheLedger, TreeCache
from levanter.tracker.wandb import WandbConfig
from marin.execution.artifact import STEP_RUNNER_EXECUTOR_VERSION
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, apply_output_path
from marin.utilities.json_encoder import CustomJsonEncoder
from rigging.filesystem import prefix_join

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as dense
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_starcoder_wsd80_gradient_conflict_outputs_20260811 as output_inventory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_starcoder_wsd80_gradient_conflict_support_20260811 as support_audit,
)
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_gradient_conflict_review_v9_20260811"
VERSION = "2026.08.11.9"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_gradient_conflict_review_v9_20260811"
PANEL_TAG = "gradient_conflict_review_v9"
WANDB_RUN_ID_PREFIX = "gcfv9"
CANARY_WANDB_RUN_IDS = frozenset(
    {
        "gcf_p1_r3d28260_m100a_common-tied-035_s2026081000",
        "gcf_p1_r3d28260_m100a_common-tied-035_s2026081001",
    }
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = (
    Path(__file__).parent
    / "exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9"
)
DESIGN_MANIFEST = DESIGN_DIR / "design_manifest.json"
TRAJECTORY_MANIFEST = DESIGN_DIR / "trajectory_manifest.csv"
CHECKPOINTER_MANIFEST = DESIGN_DIR / "checkpointer_manifest.csv"
CHECKPOINT_MANIFEST = DESIGN_DIR / "checkpoint_manifest.csv"
SOURCE_DESIGN = Path(__file__).with_name("starcoder_wsd80_dense_support_surface_design_20260808.json")
RELEASE_MANIFEST = Path(__file__).with_name("starcoder_wsd80_gradient_conflict_training_release_v9_20260811.json")

EXPECTED_DESIGN_VERSION = "2026-08-11-review-v9"
EXPECTED_DESIGN_SHA256 = "d1087945919ff8a670308b44a067392657187a727591fc9abccc0b9d6fc06e8d"
EXPECTED_DESIGN_MANIFEST_SHA256 = "4dccc079180c07d2052cdb18713b9e5f6030e2adfe531101d2f4eaf6be603224"
EXPECTED_SOURCE_DESIGN_SHA256 = "ca06420ec7c46379463091bdd55c5f720910ac38b46a0f37f08545ea9966ecbe"
EXPECTED_RELEASE_VERSION = "2026-08-11-training-v9-full"
EXPECTED_RELEASE_MANIFEST_SHA256: str | None = "4231744322e38ed30c22242d5df6f8686e8e47791dff402f175f08d55fbb2227"
EXPECTED_TRAJECTORY_COUNT = 256
EXPECTED_CHECKPOINT_COUNT = 2_768
EXPECTED_TRAIN_HOLDOUT_SEED = 2_026_081_102
EXPECTED_TRAIN_HOLDOUT_PARTITION = "random_sparse_swap"
EXPECTED_SUPPORT_PARTITION_AUDIT_SHA256 = "822cf9d2d6b6130b330947021c3a24f96f75b9bafd594730ad3a9dcb93f67b0c"
EXPECTED_STARCODER_SOURCE_TOKENS = 216_567_300_822
EXPECTED_STARCODER_SOURCE_SEQUENCES = 105_745_752
EXPECTED_STARCODER_TRAILING_TOKENS = 726
EXPECTED_STARCODER_SUPPORT_DIGESTS = {
    "m100a": "f44bf12ef0da5f401689655cca9ce16c0ca30097e5e9e84123cc62cc7e8d7cd7",
    "m100b": "9dee546086dcd39111c0fad824696d5cbc895bae714d3bca86ba16d8e7ef415c",
}
EXPECTED_STARCODER_HOLDOUT_DIGEST = "b40f0c563f181fc8a113728a2223f0ec583aa15b4e74ee2628fc9c76be3f62f9"
EXPECTED_STARCODER_SUPPORT_PHYSICAL_BLOCKS = {"m100a": 1_027, "m100b": 1_026}
MAX_RELEASE_CONCURRENCY = 64
REQUIRED_RECOVERY_CONCURRENCIES = (6, 64)
DECOUPLED_SWITCH_REPORT_SHA256 = "df70c0d3ace52ca00372e59158948a61de8e7a17a60dc69b31decfbca10282c7"
LONG_GATE_REPORT_SHA256 = "37561085c14934dcb8ca54ccf178b2b9720cfbdf4dbee0ea6a453ef139335395"
OUTPUT_INVENTORY_REPORT_SHA256 = "cd587c4cdd8f5faca85f447a0230584738c4a1b38ae79b63d458e54da53a73e9"
GEN19_RECOVERY_GENERATION = 19
GEN19_RECOVERY_PREREGISTRATION_SHA256 = "1c3a125f2521116db8a10ebb87a7e52e3bfe6128539af55da1ee9308f45b418f"
GEN19_RECOVERY_ANALYZER_REVISION_SHA256 = "2f8b030deeb632fac55e275b996e216243de67a80e84906ff41148fe576b334b"
GEN19_RECOVERY_REPORT_SHA256 = "ae4b6dd5a47321518b40bd558bdfc2972432364d2e316391af945d6d3e402405"
GEN19_RECOVERY_REVIEW_VERDICT = "ACCEPT_GEN19_SEMANTIC_ERRATUM"
GEN19_RECOVERY_REVIEW_SESSION_ID = "f3b38b65-58b1-4657-9ac3-22a2eb0bca99"
GEN24_RECOVERY_GENERATION = 24
GEN24_RECOVERY_PREREGISTRATION_SHA256 = "1ffcda8c9a5fce7095bca4c8108969aaa293abc7698ecda5e07e7b66bcb0e1a8"
GEN24_RECOVERY_REPORT_SHA256 = "21fd3c38e3baf0f843a40d441d35ed73f9d32291fd517314f7a145b194cdda1c"
GEN24_RECOVERY_REVIEW_VERDICT = "PASS_GEN24_RUNTIME_REPORT"
GEN24_RECOVERY_REVIEW_SESSION_ID = "9bb988bf-d2e9-4f92-9afa-35f4d49af014"
TPU_HOST_CPU = 16
TPU_HOST_RAM = "128g"
EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS = 559_939_584
EXPECTED_SUPPORT_AUDIT_SOURCE_SHA256 = "1429bc44d950543aef418776f046fae51ccd94e0c94be2c727959e425867ca94"
RUNTIME_SOURCE_PATHS = (
    "experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict_full.py",
    "experiments/domain_phase_mix/launch_starcoder_wsd80_dense_support_surfaces.py",
    "experiments/domain_phase_mix/launch_starcoder_wsd_80_20_surface.py",
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "audit_starcoder_wsd80_gradient_conflict_outputs_20260811.py",
    "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "audit_starcoder_wsd80_gradient_conflict_support_20260811.py",
    "lib/iris/src/iris/cli/job.py",
    "lib/iris/src/iris/client/client.py",
    "lib/levanter/src/levanter/checkpoint.py",
    "lib/levanter/src/levanter/data/dataset.py",
    "lib/levanter/src/levanter/data/text/datasets.py",
    "lib/levanter/src/levanter/main/train_lm.py",
    "lib/fray/src/fray/client.py",
    "lib/fray/src/fray/current_client.py",
    "lib/fray/src/fray/iris_backend.py",
    "lib/fray/src/fray/types.py",
    "lib/marin/src/marin/execution/artifact.py",
    "lib/marin/src/marin/execution/lazy.py",
    "lib/marin/src/marin/execution/remote.py",
    "lib/marin/src/marin/execution/step_runner.py",
    "lib/marin/src/marin/execution/step_spec.py",
    "lib/marin/src/marin/execution/step_status.py",
    "lib/marin/src/marin/run/iris_run.py",
    "lib/marin/src/marin/training/training.py",
    "lib/marin/src/marin/utilities/json_encoder.py",
)
H5_CELL_ID = "h5_fixed_aggregate_h0640_s28160"
H5_TOTAL_STEPS = 28_160
H5_DECAY_STEP = 22_528
EXPECTED_TRAINING_COMPONENT_NAMES = (
    "nemotron_cc/hq_actual-llama3",
    "nemotron_cc/hq_synth-llama3",
    "nemotron_cc/medium_high-llama3",
    "nemotron_cc/medium-llama3",
    "nemotron_cc/medium_low-llama3",
    "nemotron_cc/low_actual-llama3",
    "dolma/starcoder",
)
EXPECTED_NEMOTRON_LEDGERS = {
    "nemotron_cc/hq_actual-llama3": (
        "37e6979de8a88a2b961dea31c956a3463b620790f8ab152b8162f7b9930976d1",
        746_497_814,
        2_755,
    ),
    "nemotron_cc/hq_synth-llama3": (
        "223d1457819974a9e557aa22c11b46909f462fbbfb58c3fcb2b261a1eab548ad",
        3_687_977_948,
        8_353,
    ),
    "nemotron_cc/medium_high-llama3": (
        "42cab981b2c699adadec2b862c01d9182f28840683b1b057e46eca2f05e5a057",
        558_672_867,
        2_454,
    ),
    "nemotron_cc/medium-llama3": (
        "edf0cb25de6f670d700e1a0b1dad5d1ef341afe999ae0cc170343370c51ab098",
        2_284_127_609,
        9_678,
    ),
    "nemotron_cc/medium_low-llama3": (
        "7c4cf9892124947ee5529d2859af296550cbee72e0c83022b7ccf8da881f35ea",
        1_304_151_015,
        4_287,
    ),
    "nemotron_cc/low_actual-llama3": (
        "8b2e53e8744734cd6de90b07414c75378d674a18b4de539a3aefc6da2e8dbe4f",
        886_051_765,
        1_964,
    ),
}
EXPECTED_TOKENIZER_METADATA = {
    "append_bos": False,
    "append_eos": True,
    "max_length": 131_072,
    "padding": False,
    "return_attention_mask": False,
    "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
    "vocab_size": 128_256,
}
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=5)


@dataclass(frozen=True)
class Cell:
    """Frozen model and schedule metadata for one token-horizon cell."""

    cell_id: str
    hidden_size: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int


@dataclass(frozen=True)
class Trajectory:
    """One training row from the review-v9 trajectory manifest."""

    trajectory_id: str
    arm: str
    cell_id: str
    support_id: str
    support_pool_seed: int | None
    training_seed: int
    policy_role: str
    phase_0_fraction: float
    phase_1_fraction: float
    phase_0_starcoder: float
    phase_1_starcoder: float
    aggregate_starcoder: float
    phase_contrast_p0_minus_p1: float
    upstream_phase_contrast_p1_minus_p0: float
    coordinate_selection_rule: str
    total_steps: int
    boundary_step: int
    optimizer_decay_step: int
    primary_inference: bool
    support_start_batches: int | None
    support_batches: int | None
    train_holdout_sequences_per_component: int
    train_holdout_seed: int
    train_holdout_partition: str
    starcoder_phase_0_sequences: int
    starcoder_phase_1_sequences: int
    starcoder_total_sequences: int
    realized_aggregate_starcoder: float
    realized_phase_0_starcoder_per_block: int
    realized_phase_1_starcoder_per_block: int


@dataclass(frozen=True)
class _SourceValidationRequest:
    starcoder_support_batches: int | None
    starcoder_realized_support_tokens: int


def _wandb_run_id(trajectory: Trajectory) -> str:
    if not WANDB_RUN_ID_PREFIX:
        raise ValueError("Review-v9 W&B identities require a nonempty prefix to avoid canary collisions")
    run_id = f"{WANDB_RUN_ID_PREFIX}_{trajectory.trajectory_id}"
    if run_id in CANARY_WANDB_RUN_IDS:
        raise ValueError(f"{trajectory.trajectory_id}: review-v9 W&B identity collides with a canary run")
    return run_id


def _realized_component_count(weights: dict[str, float], component: str, block_size: int) -> int:
    names = tuple(weights)
    counts = [int(weights[name] * block_size) for name in names]
    largest_index = max(range(len(counts)), key=counts.__getitem__)
    counts[largest_index] += block_size - sum(counts)
    return counts[names.index(component)]


def _enumerate_permanent_checkpoint_steps(
    keep: list[dict[str, int | None]],
    *,
    total_steps: int,
) -> list[int]:
    steps: list[int] = []
    for step in range(1, total_steps):
        policy = next(row for row in keep if row["until"] is None or row["until"] >= step)
        if step % cast(int, policy["every"]) == 0:
            steps.append(step)
    return steps


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with fsspec.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_launcher_release_pin(source: str) -> str:
    prefix = "EXPECTED_RELEASE_MANIFEST_SHA256: str | None = "
    matching_lines = [line for line in source.splitlines() if line.startswith(prefix)]
    if len(matching_lines) != 1:
        raise ValueError("Launcher release-pin declaration drifted")
    return source.replace(matching_lines[0], f'{prefix}"<release-manifest-sha256>"')


def _canonical_launcher_source_sha256() -> str:
    source = Path(__file__).read_text(encoding="utf-8")
    return hashlib.sha256(_normalize_launcher_release_pin(source).encode()).hexdigest()


def _runtime_source_sha256() -> dict[str, str]:
    hashes = {relative_path: _file_sha256(REPO_ROOT / relative_path) for relative_path in RUNTIME_SOURCE_PATHS}
    hashes["experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict_full.py"] = (
        _canonical_launcher_source_sha256()
    )
    return hashes


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid frozen boolean: {value!r}")


def _artifact_path(relative_path: str) -> Path:
    if relative_path.startswith("experiments/"):
        return REPO_ROOT / relative_path
    return DESIGN_DIR / relative_path


def _validate_starcoder_token_domain(observed_tokens: int) -> tuple[int, int]:
    observed_sequences, trailing_tokens = divmod(observed_tokens, base.SEQ_LEN)
    expected = (
        EXPECTED_STARCODER_SOURCE_TOKENS,
        EXPECTED_STARCODER_SOURCE_SEQUENCES,
        EXPECTED_STARCODER_TRAILING_TOKENS,
    )
    observed = (observed_tokens, observed_sequences, trailing_tokens)
    if observed != expected:
        raise ValueError(f"StarCoder packed sequence domain drifted: {observed} != {expected}")
    return observed_sequences, trailing_tokens


def _validate_starcoder_support_identity() -> dict[str, Any]:
    """Recompute the finite-support mapping with the live loader implementation."""
    payload, _ = asyncio.run(support_audit.materialize())
    supports = cast(dict[str, dict[str, Any]], payload["supports"])
    holdout = cast(dict[str, Any], payload["holdout"])
    cross_support = cast(dict[str, Any], payload["cross_support"])
    observed = {
        "source_sequence_count": payload["source_sequence_count"],
        "support_sequence_counts": {name: row["sequence_count"] for name, row in supports.items()},
        "support_ordered_sequence_sha256": {name: row["ordered_sequence_sha256"] for name, row in supports.items()},
        "support_distinct_physical_blocks": {name: row["distinct_physical_blocks"] for name, row in supports.items()},
        "holdout_sorted_sequence_sha256": holdout["sorted_sequence_sha256"],
        "shared_sequence_count": cross_support["shared_sequence_count"],
        "shared_physical_block_count": cross_support["shared_physical_block_count"],
        "holdout_overlap_sequence_count": cross_support["holdout_overlap_sequence_count"],
    }
    expected = {
        "source_sequence_count": EXPECTED_STARCODER_SOURCE_SEQUENCES,
        "support_sequence_counts": {"m100a": 136_704, "m100b": 136_704},
        "support_ordered_sequence_sha256": EXPECTED_STARCODER_SUPPORT_DIGESTS,
        "support_distinct_physical_blocks": EXPECTED_STARCODER_SUPPORT_PHYSICAL_BLOCKS,
        "holdout_sorted_sequence_sha256": EXPECTED_STARCODER_HOLDOUT_DIGEST,
        "shared_sequence_count": 0,
        "shared_physical_block_count": 512,
        "holdout_overlap_sequence_count": {"m100a": 0, "m100b": 0},
    }
    if observed != expected:
        raise ValueError(f"StarCoder finite-support identity drifted: {observed} != {expected}")
    return observed


def _load_design_manifest() -> dict[str, Any]:
    observed_file_hash = _file_sha256(DESIGN_MANIFEST)
    if observed_file_hash != EXPECTED_DESIGN_MANIFEST_SHA256:
        raise ValueError(f"Review-v9 manifest drifted: {observed_file_hash} != {EXPECTED_DESIGN_MANIFEST_SHA256}")
    manifest = json.loads(DESIGN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("design_version") != EXPECTED_DESIGN_VERSION:
        raise ValueError(f"Unexpected design version: {manifest.get('design_version')!r}")
    claimed_hash = manifest.get("design_sha256")
    observed_hash = _canonical_sha256({**manifest, "design_sha256": ""})
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Review-v9 self-hash drifted: {observed_hash} != {claimed_hash}")
    expected_placement = {
        "required_region": base.DEFAULT_TPU_REGION,
        "required_zone": base.DEFAULT_TPU_ZONE,
        "required_bucket_prefix": base.DEFAULT_MARIN_PREFIX,
    }
    if {key: manifest.get(key) for key in expected_placement} != expected_placement:
        raise ValueError("Review-v9 placement contract drifted")
    if manifest.get("trajectory_count") != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("Review-v9 trajectory count drifted")
    if manifest.get("checkpoint_count") != EXPECTED_CHECKPOINT_COUNT:
        raise ValueError("Review-v9 checkpoint count drifted")
    if manifest.get("train_holdout_partition") != EXPECTED_TRAIN_HOLDOUT_PARTITION:
        raise ValueError("Review-v9 holdout implementation drifted")
    support_audit = manifest.get("support_partition_audit")
    if not isinstance(support_audit, dict) or support_audit.get("cross_support") != {
        "holdout_overlap_sequence_count": {"m100a": 0, "m100b": 0},
        "shared_physical_block_count": 512,
        "shared_sequence_count": 0,
    }:
        raise ValueError("Review-v9 support partition audit drifted")
    for relative_path, expected_hash in {
        **manifest["input_artifact_sha256"],
        **manifest["artifact_sha256"],
    }.items():
        path = _artifact_path(relative_path)
        observed_artifact_hash = _file_sha256(path)
        if observed_artifact_hash != expected_hash:
            raise ValueError(f"Review-v9 artifact drifted: {relative_path}: {observed_artifact_hash} != {expected_hash}")
    return manifest


def _load_training_release() -> dict[str, Any]:
    if EXPECTED_RELEASE_MANIFEST_SHA256 is None:
        raise RuntimeError("Training fanout is not released: expected release-manifest hash is unset")
    observed_file_hash = _file_sha256(RELEASE_MANIFEST)
    if observed_file_hash != EXPECTED_RELEASE_MANIFEST_SHA256:
        raise ValueError(
            f"Training release manifest drifted: {observed_file_hash} != {EXPECTED_RELEASE_MANIFEST_SHA256}"
        )
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    return _validate_training_release(release)


def _validate_training_release(release: dict[str, Any]) -> dict[str, Any]:
    """Validate the signed release payload independently of its file pin."""
    claimed_hash = release.get("release_sha256")
    observed_hash = _canonical_sha256({**release, "release_sha256": ""})
    if claimed_hash != observed_hash:
        raise ValueError(f"Training release self-hash drifted: {observed_hash} != {claimed_hash}")
    if release.get("release_version") != EXPECTED_RELEASE_VERSION:
        raise ValueError(f"Unexpected release version: {release.get('release_version')!r}")
    expected_design = {
        "design_version": EXPECTED_DESIGN_VERSION,
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "design_manifest_sha256": EXPECTED_DESIGN_MANIFEST_SHA256,
    }
    if {key: release.get(key) for key in expected_design} != expected_design:
        raise ValueError("Training release no longer identifies the frozen review-v9 design")
    if release.get("training_fanout_allowed") is not True or release.get("probe_fanout_allowed") is not False:
        raise RuntimeError("Release must authorize training only; probe fanout remains separately gated")
    maximum_trajectory_count = release.get("maximum_trajectory_count")
    if maximum_trajectory_count != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError("Full training release must authorize the exact v9 trajectory count")
    allowed_trajectory_ids = release.get("allowed_trajectory_ids")
    if (
        not isinstance(allowed_trajectory_ids, list)
        or not allowed_trajectory_ids
        or any(not isinstance(trajectory_id, str) or not trajectory_id for trajectory_id in allowed_trajectory_ids)
        or len(set(allowed_trajectory_ids)) != len(allowed_trajectory_ids)
        or len(allowed_trajectory_ids) != maximum_trajectory_count
    ):
        raise ValueError("Training release has an invalid allowed trajectory set")
    with TRAJECTORY_MANIFEST.open(newline="") as handle:
        expected_trajectory_ids = {row["trajectory_id"] for row in csv.DictReader(handle)}
    if set(allowed_trajectory_ids) != expected_trajectory_ids:
        raise ValueError("Training release does not authorize exactly the frozen v9 trajectories")
    expected_release_contract = {
        "required_region": base.DEFAULT_TPU_REGION,
        "required_zone": base.DEFAULT_TPU_ZONE,
        "required_bucket_prefix": base.DEFAULT_MARIN_PREFIX,
        "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
        "checkpoint_count": EXPECTED_CHECKPOINT_COUNT,
        "maximum_concurrent_trajectories": MAX_RELEASE_CONCURRENCY,
        "train_holdout_seed": EXPECTED_TRAIN_HOLDOUT_SEED,
        "train_holdout_partition": EXPECTED_TRAIN_HOLDOUT_PARTITION,
        "support_partition_audit_sha256": EXPECTED_SUPPORT_PARTITION_AUDIT_SHA256,
    }
    if {key: release.get(key) for key in expected_release_contract} != expected_release_contract:
        raise ValueError("Training release operational contract drifted")
    if release.get("runtime_source_sha256") != _runtime_source_sha256():
        raise ValueError("Training release runtime source hashes drifted")
    evidence = release.get("validated_evidence")
    expected_source_evidence = {
        "starcoder_flat_field_token_count": EXPECTED_STARCODER_SOURCE_TOKENS,
        "starcoder_packed_sequence_count": EXPECTED_STARCODER_SOURCE_SEQUENCES,
        "starcoder_trailing_token_count": EXPECTED_STARCODER_TRAILING_TOKENS,
        "finite_support_required_tokens": EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS,
        "runtime_config_count": EXPECTED_TRAJECTORY_COUNT,
        "support_audit_source_sha256": EXPECTED_SUPPORT_AUDIT_SOURCE_SHA256,
    }
    if not isinstance(evidence, dict) or {key: evidence.get(key) for key in expected_source_evidence} != (
        expected_source_evidence
    ):
        raise ValueError("Training release StarCoder source evidence drifted")
    long_gate = evidence.get("long_gate")
    if (
        not isinstance(long_gate, dict)
        or long_gate.get("status") != "pass"
        or long_gate.get("endpoint_metrics_read") is not False
    ):
        raise ValueError("Training release lacks a sealed passing full-length switch gate")
    switch_canary = evidence.get("decoupled_switch_canary")
    if (
        not isinstance(switch_canary, dict)
        or switch_canary.get("status") != "pass"
        or switch_canary.get("data_switch_step") == switch_canary.get("optimizer_decay_step")
    ):
        raise ValueError("Training release lacks a passing decoupled-switch canary")
    recovery_gates = evidence.get("recovery_gates")
    expected_recovery_gates = [
        {
            "maximum_concurrent": REQUIRED_RECOVERY_CONCURRENCIES[0],
            "generation": GEN19_RECOVERY_GENERATION,
            "status": "pass",
            "report_sha256": GEN19_RECOVERY_REPORT_SHA256,
            "preregistration_sha256": GEN19_RECOVERY_PREREGISTRATION_SHA256,
            "analyzer_revision_sha256": GEN19_RECOVERY_ANALYZER_REVISION_SHA256,
            "independent_review_verdict": GEN19_RECOVERY_REVIEW_VERDICT,
            "independent_review_session_id": GEN19_RECOVERY_REVIEW_SESSION_ID,
            "endpoint_metrics_read": False,
        },
        {
            "maximum_concurrent": REQUIRED_RECOVERY_CONCURRENCIES[1],
            "generation": GEN24_RECOVERY_GENERATION,
            "status": "pass",
            "report_sha256": GEN24_RECOVERY_REPORT_SHA256,
            "preregistration_sha256": GEN24_RECOVERY_PREREGISTRATION_SHA256,
            "prior_gate_report_sha256": GEN19_RECOVERY_REPORT_SHA256,
            "independent_review_verdict": GEN24_RECOVERY_REVIEW_VERDICT,
            "independent_review_session_id": GEN24_RECOVERY_REVIEW_SESSION_ID,
            "endpoint_metrics_read": False,
        },
    ]
    if recovery_gates != expected_recovery_gates:
        raise ValueError("Training release recovery-gate evidence drifted")
    expected_orchestration_scope = {
        "c64_gate_scope": "Levanter run-local checkpoint recovery under 64-way child preemption",
        "c64_gate_exercises_step_runner_fanout": False,
        "production_fanout": "StepRunner dispatches one independent Fray/Iris child per trajectory",
        "production_fanout_live_gate": False,
        "production_fanout_source_hash_pinned": True,
        "parent_failure_policy": "fail closed; resubmit the exact command against owned resumable roots",
        "child_application_failure_retries": 0,
        "child_preemption_retries": 100,
        "child_wall_timeout_seconds": None,
    }
    if evidence.get("orchestration_scope") != expected_orchestration_scope:
        raise ValueError("Training release orchestration-scope disclosure drifted")
    inventory = evidence.get("output_inventory")
    if not isinstance(inventory, dict) or inventory != {
        "expected_root_count": EXPECTED_TRAJECTORY_COUNT,
        "empty_root_count": EXPECTED_TRAJECTORY_COUNT,
        "bookkeeping_root_count": 0,
        "resumable_root_count": 0,
        "completed_root_count": 0,
        "partial_root_count": 0,
        "unexpected_root_count": 0,
    }:
        raise ValueError("Training release output-root inventory drifted")
    expected_report_hashes = {
        "long_gate_report_sha256": LONG_GATE_REPORT_SHA256,
        "decoupled_switch_canary_report_sha256": DECOUPLED_SWITCH_REPORT_SHA256,
        "checkpoint_recovery_report_sha256": GEN19_RECOVERY_REPORT_SHA256,
        "operational_threshold_report_sha256": GEN24_RECOVERY_REPORT_SHA256,
    }
    if {field: evidence.get(field) for field in expected_report_hashes} != expected_report_hashes:
        raise ValueError("Training release gate-report hashes drifted")
    if evidence.get("output_inventory_report_sha256") != OUTPUT_INVENTORY_REPORT_SHA256:
        raise ValueError("Training release output-inventory report hash drifted")
    review = release.get("independent_review")
    if not isinstance(review, dict) or review.get("verdict") != "PASS_FULL_TRAINING":
        raise ValueError("Training release lacks independent full-training approval")
    return release


def _validate_training_selection(trajectories: tuple[Trajectory, ...], release: dict[str, Any]) -> None:
    maximum_trajectory_count = cast(int, release["maximum_trajectory_count"])
    if len(trajectories) > maximum_trajectory_count:
        raise RuntimeError(
            f"Training release authorizes at most {maximum_trajectory_count} trajectories; "
            f"launch selected {len(trajectories)}"
        )
    allowed_trajectory_ids = cast(list[str], release["allowed_trajectory_ids"])
    disallowed = {trajectory.trajectory_id for trajectory in trajectories} - set(allowed_trajectory_ids)
    if disallowed:
        raise RuntimeError(f"Training release does not authorize trajectories: {sorted(disallowed)}")


def _load_cells() -> dict[str, Cell]:
    if _file_sha256(SOURCE_DESIGN) != EXPECTED_SOURCE_DESIGN_SHA256:
        raise ValueError("Frozen source cell design drifted")
    payload = json.loads(SOURCE_DESIGN.read_text(encoding="utf-8"))
    cells = {
        row["cell_id"]: Cell(
            cell_id=row["cell_id"],
            hidden_size=int(row["hidden_size"]),
            total_steps=int(row["total_steps"]),
            boundary_step=int(row["boundary_step"]),
            materialized_tokens=int(row["materialized_tokens"]),
            total_parameters=int(row["total_parameters"]),
            non_embedding_parameters=int(row["non_embedding_parameters"]),
        )
        for row in payload["cells"]
    }
    r3 = cells["r3_increase_d_h0640_s28260"]
    cells[H5_CELL_ID] = replace(
        r3,
        cell_id=H5_CELL_ID,
        total_steps=H5_TOTAL_STEPS,
        boundary_step=H5_DECAY_STEP,
        materialized_tokens=H5_TOTAL_STEPS * base.BATCH_SIZE * base.SEQ_LEN,
    )
    expected_cells = {
        "r0_shared_h0640_s03820",
        "r1_increase_d_h0640_s07320",
        "r2_increase_d_h0640_s14960",
        "r3_increase_d_h0640_s28260",
        H5_CELL_ID,
    }
    if set(cells) != expected_cells:
        raise ValueError(f"Frozen review-v9 cell identities drifted: {sorted(cells)}")
    return cells


def _trajectory_from_csv(row: dict[str, str]) -> Trajectory:
    return Trajectory(
        trajectory_id=row["trajectory_id"],
        arm=row["arm"],
        cell_id=row["cell_id"],
        support_id=row["support_id"],
        support_pool_seed=_optional_int(row["support_pool_seed"]),
        training_seed=int(row["training_seed"]),
        policy_role=row["policy_role"],
        phase_0_fraction=float(row["phase_0_fraction"]),
        phase_1_fraction=float(row["phase_1_fraction"]),
        phase_0_starcoder=float(row["phase_0_starcoder"]),
        phase_1_starcoder=float(row["phase_1_starcoder"]),
        aggregate_starcoder=float(row["aggregate_starcoder"]),
        phase_contrast_p0_minus_p1=float(row["phase_contrast_p0_minus_p1"]),
        upstream_phase_contrast_p1_minus_p0=float(row["upstream_phase_contrast_p1_minus_p0"]),
        coordinate_selection_rule=row["coordinate_selection_rule"],
        total_steps=int(row["total_steps"]),
        boundary_step=int(row["boundary_step"]),
        optimizer_decay_step=int(row["optimizer_decay_step"]),
        primary_inference=_parse_bool(row["primary_inference"]),
        support_start_batches=_optional_int(row["support_start_batches"]),
        support_batches=_optional_int(row["support_batches"]),
        train_holdout_sequences_per_component=int(row["train_holdout_sequences_per_component"]),
        train_holdout_seed=int(row["train_holdout_seed"]),
        train_holdout_partition=row["train_holdout_partition"],
        starcoder_phase_0_sequences=int(row["starcoder_phase_0_sequences"]),
        starcoder_phase_1_sequences=int(row["starcoder_phase_1_sequences"]),
        starcoder_total_sequences=int(row["starcoder_total_sequences"]),
        realized_aggregate_starcoder=float(row["realized_aggregate_starcoder"]),
        realized_phase_0_starcoder_per_block=int(row["realized_phase_0_starcoder_per_block"]),
        realized_phase_1_starcoder_per_block=int(row["realized_phase_1_starcoder_per_block"]),
    )


def load_design(
    *,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, Cell], tuple[Trajectory, ...], dict[str, list[dict[str, int | None]]]]:
    """Load and structurally audit every immutable review-v9 training row."""
    _load_design_manifest()
    cells = _load_cells()
    with TRAJECTORY_MANIFEST.open(newline="") as handle:
        trajectories = tuple(_trajectory_from_csv(row) for row in csv.DictReader(handle))
    with CHECKPOINTER_MANIFEST.open(newline="") as handle:
        checkpointer_rows = list(csv.DictReader(handle))
    checkpoint_steps_by_id: dict[str, list[int]] = {}
    with CHECKPOINT_MANIFEST.open(newline="") as handle:
        for checkpoint_row in csv.DictReader(handle):
            checkpoint_steps_by_id.setdefault(checkpoint_row["trajectory_id"], []).append(
                int(checkpoint_row["checkpoint_step"])
            )

    if len(trajectories) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError(f"Review-v9 trajectory count drifted: {len(trajectories)}")
    if len({row.trajectory_id for row in trajectories}) != len(trajectories):
        raise ValueError("Review-v9 trajectory IDs are not unique")
    keep_by_id = {row["trajectory_id"]: json.loads(row["keep_json"]) for row in checkpointer_rows}
    if set(keep_by_id) != {row.trajectory_id for row in trajectories}:
        raise ValueError("Checkpointer manifest and trajectory manifest identities differ")
    if set(checkpoint_steps_by_id) != set(keep_by_id):
        raise ValueError("Checkpoint and checkpointer manifest identities differ")
    total_steps_by_id = {row.trajectory_id: row.total_steps for row in trajectories}
    for row in checkpointer_rows:
        trajectory_id = row["trajectory_id"]
        expected_steps = [int(step) for step in row["expected_checkpoint_steps"].split("|")]
        translated_steps = [int(item["every"]) for item in keep_by_id[trajectory_id]]
        if checkpoint_steps_by_id[trajectory_id] != expected_steps or translated_steps != expected_steps:
            raise ValueError(f"{trajectory_id}: checkpoint-to-checkpointer translation drifted")
        realized_steps = _enumerate_permanent_checkpoint_steps(
            keep_by_id[trajectory_id],
            total_steps=total_steps_by_id[trajectory_id],
        )
        if realized_steps != expected_steps:
            raise ValueError(f"{trajectory_id}: checkpointer keep policy realizes unexpected steps")
        if int(row["expected_checkpoint_count"]) != len(expected_steps):
            raise ValueError(f"{trajectory_id}: expected checkpoint count drifted")
    if sum(len(keep) for keep in keep_by_id.values()) != EXPECTED_CHECKPOINT_COUNT:
        raise ValueError("Checkpointer manifest checkpoint count drifted")

    finite_required_tokens = max(
        ((row.support_start_batches or 0) + (row.support_batches or 0)) * base.BATCH_SIZE * base.SEQ_LEN
        for row in trajectories
        if row.support_batches is not None
    )
    if finite_required_tokens != EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS:
        raise ValueError(
            f"Finite-support token requirement drifted: "
            f"{finite_required_tokens} != {EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS}"
        )

    for row in trajectories:
        cell = cells[row.cell_id]
        if row.arm == "b":
            expected_boundary = round(row.phase_0_fraction * row.total_steps)
            if (
                row.cell_id != H5_CELL_ID
                or row.total_steps != H5_TOTAL_STEPS
                or row.optimizer_decay_step != H5_DECAY_STEP
                or row.boundary_step != expected_boundary
            ):
                raise ValueError(f"{row.trajectory_id}: decoupled boundary/decay schedule drifted")
        elif row.total_steps != cell.total_steps or row.optimizer_decay_step != cell.boundary_step:
            raise ValueError(f"{row.trajectory_id}: cell schedule drifted")
        if row.total_steps * base.BATCH_SIZE * base.SEQ_LEN <= 0:
            raise ValueError(f"{row.trajectory_id}: invalid token accounting")
        if row.train_holdout_seed != EXPECTED_TRAIN_HOLDOUT_SEED:
            raise ValueError(f"{row.trajectory_id}: global holdout seed drifted")
        if row.train_holdout_partition != EXPECTED_TRAIN_HOLDOUT_PARTITION:
            raise ValueError(f"{row.trajectory_id}: holdout partition implementation drifted")
        if row.support_id == "full":
            if row.support_batches is not None or row.support_pool_seed is not None:
                raise ValueError(f"{row.trajectory_id}: full support is unexpectedly capped")
        else:
            if row.support_batches is None or row.support_pool_seed is None or row.support_start_batches is None:
                raise ValueError(f"{row.trajectory_id}: finite support is underspecified")
        if row.support_id == "m100a" and row.support_start_batches != 0:
            raise ValueError(f"{row.trajectory_id}: m100a support offset drifted")
        if row.support_id == "m100b" and row.support_start_batches != row.support_batches:
            raise ValueError(f"{row.trajectory_id}: m100b is not the adjacent disjoint support slice")
        if not 0.0 <= row.phase_0_starcoder <= 1.0 or not 0.0 <= row.phase_1_starcoder <= 1.0:
            raise ValueError(f"{row.trajectory_id}: infeasible policy")
        if row.starcoder_total_sequences != row.starcoder_phase_0_sequences + row.starcoder_phase_1_sequences:
            raise ValueError(f"{row.trajectory_id}: realized StarCoder sequence accounting drifted")
        expected_decay = row.total_steps - row.optimizer_decay_step
        optimizer = base._optimizer(row.total_steps * base.BATCH_SIZE * base.SEQ_LEN)
        if asdict(optimizer)["decay"] != expected_decay:
            raise ValueError(f"{row.trajectory_id}: optimizer-decay onset drifted")

    available_arms = {row.arm for row in trajectories}
    if selected_arms is not None:
        unknown = selected_arms - available_arms
        if unknown:
            raise ValueError(f"Unknown arms: {sorted(unknown)}")
        trajectories = tuple(row for row in trajectories if row.arm in selected_arms)
    if selected_runs is not None:
        available = {row.trajectory_id for row in trajectories}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs after arm filtering: {sorted(unknown)}")
        trajectories = tuple(row for row in trajectories if row.trajectory_id in selected_runs)
    if not trajectories:
        raise ValueError("Launch filters selected no review-v9 trajectories")
    return cells, trajectories, keep_by_id


def _training_data() -> tuple[
    dict[str, ArtifactStep[TokenizedCache]],
    ArtifactStep[TokenizedCache],
    tuple[ArtifactStep[TokenizedCache], ...],
]:
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    validation = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    return nemotron, starcoder, validation


def _configure_training(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    trajectory: Trajectory,
    phase_weights: list[tuple[int, dict[str, float]]],
    training_component_names: tuple[str, ...],
    starcoder_name: str,
    keep: list[dict[str, int | None]],
) -> ArtifactStep[LevanterCheckpoint]:
    """Install the frozen holdout, support, phase, seed, and checkpoint contract."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        train_config = cast(TrainLmConfig, pod_config.train_config)
        support_cap = None if trajectory.support_batches is None else {starcoder_name: trajectory.support_batches}
        support_start = (
            None if trajectory.support_start_batches is None else {starcoder_name: trajectory.support_start_batches}
        )
        data_config = replace(
            train_config.data,
            train_weights=phase_weights,
            mixture_block_size=base.MIXTURE_BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches=support_cap,
            max_train_batches_subset_seed=trajectory.support_pool_seed,
            max_train_batches_start=support_start,
            train_holdout_sequences={
                name: trajectory.train_holdout_sequences_per_component for name in training_component_names
            },
            train_holdout_seed=trajectory.train_holdout_seed,
            train_holdout_partition=EXPECTED_TRAIN_HOLDOUT_PARTITION,
        )
        trainer = replace(
            train_config.trainer,
            seed=trajectory.training_seed,
            checkpointer=replace(
                train_config.trainer.checkpointer,
                save_interval=CHECKPOINT_INTERVAL,
                keep=keep,
                keep_last_temporary_checkpoints=1,
            ),
        )
        train_config = replace(
            train_config,
            data=data_config,
            data_seed=trajectory.training_seed,
            trainer=trainer,
        )
        return replace(pod_config, train_config=train_config)

    return replace(training, build_config=build_config)


def build_training_steps(
    *,
    marin_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[tuple[Trajectory, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build resumable training handles for the selected frozen rows."""
    cells, trajectories, keep_by_id = load_design(selected_arms=selected_arms, selected_runs=selected_runs)
    nemotron, starcoder, validation = _training_data()
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    component_names = tuple(handle.name for handle in training_handles)
    if component_names != EXPECTED_TRAINING_COMPONENT_NAMES:
        raise ValueError(f"Training-cache identities drifted: {component_names}")
    if len({id(handle) for handle in training_handles}) != len(training_handles):
        raise ValueError("Training-cache handles contain duplicates")
    resources = ResourceConfig.with_tpu(
        tpu_type,
        cpu=TPU_HOST_CPU,
        ram=TPU_HOST_RAM,
        regions=(tpu_region,),
        zone=tpu_zone,
    )
    models: dict[int, Any] = {}
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for trajectory in trajectories:
        cell = cells[trajectory.cell_id]
        model = models.setdefault(
            cell.hidden_size,
            CompletedAdamHHeuristic()._build_model_config(cell.hidden_size, seq_len=base.SEQ_LEN),
        )
        total_parameters = model.total_trainable_params(llama3_tokenizer_vocab_size)
        non_embedding_parameters = model.total_trainable_params(0)
        if total_parameters != cell.total_parameters or non_embedding_parameters != cell.non_embedding_parameters:
            raise ValueError(f"{trajectory.trajectory_id}: model shape drifted")
        phase_0_weights = base._phase_leaf_weights(
            trajectory.phase_0_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        phase_1_weights = base._phase_leaf_weights(
            trajectory.phase_1_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        if set(phase_0_weights) != set(component_names) or set(phase_1_weights) != set(component_names):
            raise ValueError(f"{trajectory.trajectory_id}: phase-weight identities drifted")
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        materialized_tokens = trajectory.total_steps * base.BATCH_SIZE * base.SEQ_LEN
        wandb_run_id = _wandb_run_id(trajectory)
        training = train_lm(
            name=f"checkpoints/{NAME}/trajectories/{trajectory.trajectory_id}",
            version=VERSION,
            model=model,
            optimizer=base._optimizer(materialized_tokens),
            datasets=static_weights,
            validation=validation,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=trajectory.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=NAME,
            run_id=wandb_run_id,
            tags=(
                WANDB_EXPERIMENT_TAG,
                PANEL_TAG,
                trajectory.arm,
                trajectory.cell_id,
                trajectory.support_id,
                trajectory.policy_role,
                "starcoder",
                "wsd80_20",
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            _configure_training(
                training,
                trajectory=trajectory,
                phase_weights=[(0, phase_0_weights), (trajectory.boundary_step, phase_1_weights)],
                training_component_names=component_names,
                starcoder_name=starcoder.name,
                keep=keep_by_id[trajectory.trajectory_id],
            )
        )
    return trajectories, tuple(steps)


def expected_output_owners(
    trajectories: tuple[Trajectory, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
) -> dict[str, dict[str, Any]]:
    """Return the exact StepRunner and artifact identities allowed at each output root."""
    if len(trajectories) != len(steps):
        raise ValueError("Trajectory/step cardinality mismatch")
    owners: dict[str, dict[str, Any]] = {}
    for trajectory, handle in zip(trajectories, steps, strict=True):
        step = lower(handle)
        root = f"{trajectory.trajectory_id}/{VERSION}"
        executor_info = {
            "executor_version": STEP_RUNNER_EXECUTOR_VERSION,
            "name": step.name,
            "config": step.hash_attrs,
            "override_output_path": step.override_output_path,
            "dependencies": list(step.dep_paths),
        }
        normalized_executor_info = json.loads(json.dumps(executor_info, cls=CustomJsonEncoder))
        owners[root] = {
            "executor_info": normalized_executor_info,
            "artifact_record": {
                "name": handle.name,
                "version": handle.version,
                "fingerprint": step.hash_attrs["fingerprint"],
                "result_type": step.hash_attrs["result_type"],
                "output_path": handle.path(marin_prefix),
                "deps": step.hash_attrs["deps"],
                "dep_paths": [],
                "source": None,
                "result": None,
                "fingerprint_payload": step.fingerprint_payload,
            },
        }
    return owners


def audit_runtime_configs(
    trajectories: tuple[Trajectory, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
) -> int:
    """Materialize and verify every selected training config."""
    if len(trajectories) != len(steps):
        raise ValueError("Trajectory/step cardinality mismatch")
    _, _, keep_by_id = load_design(selected_runs=frozenset(row.trajectory_id for row in trajectories))
    nemotron, starcoder, _ = _training_data()
    training_component_names = set(EXPECTED_TRAINING_COMPONENT_NAMES)
    loaded_artifacts: dict[int, Any] = {}

    for trajectory, step in zip(trajectories, steps, strict=True):
        # Every trajectory uses the same immutable cache records. Reuse those
        # sidecars while still materializing and checking every row's config;
        # otherwise this audit performs about 7,200 redundant GCS reads.
        pod_config = materialized_config(step, marin_prefix, artifact_cache=loaded_artifacts)
        if not isinstance(pod_config, TrainLmOnPodConfig):
            raise TypeError(f"{trajectory.trajectory_id}: unexpected runtime config {type(pod_config)}")
        if pod_config.resources.cpu != TPU_HOST_CPU:
            raise ValueError(f"{trajectory.trajectory_id}: TPU host CPU request drifted")
        if pod_config.resources.ram != TPU_HOST_RAM:
            raise ValueError(f"{trajectory.trajectory_id}: TPU host RAM request drifted")
        train_config = cast(TrainLmConfig, pod_config.train_config)
        expected_output = prefix_join(
            marin_prefix,
            f"checkpoints/{NAME}/trajectories/{trajectory.trajectory_id}/{VERSION}",
        )
        if pod_config.output_path != expected_output:
            raise ValueError(f"{trajectory.trajectory_id}: output path drifted")
        if train_config.trainer.num_train_steps != trajectory.total_steps:
            raise ValueError(f"{trajectory.trajectory_id}: training horizon drifted")
        if train_config.trainer.seed != trajectory.training_seed or train_config.data_seed != trajectory.training_seed:
            raise ValueError(f"{trajectory.trajectory_id}: model/data seed drifted")
        expected_wandb_run_id = _wandb_run_id(trajectory)
        if train_config.trainer.id != expected_wandb_run_id:
            raise ValueError(f"{trajectory.trajectory_id}: W&B run identity drifted")
        tracker = train_config.trainer.tracker
        if not isinstance(tracker, WandbConfig) or tracker.name != expected_wandb_run_id or tracker.id is not None:
            raise ValueError(f"{trajectory.trajectory_id}: W&B tracker identity drifted")
        if train_config.optimizer_schedule_num_train_steps is not None:
            raise ValueError(f"{trajectory.trajectory_id}: unexpected optimizer-horizon override")
        expected_optimizer = asdict(base._optimizer(trajectory.total_steps * base.BATCH_SIZE * base.SEQ_LEN))
        if asdict(train_config.optimizer) != expected_optimizer:
            raise ValueError(f"{trajectory.trajectory_id}: optimizer config drifted")
        if train_config.data.train_holdout_sequences != {
            name: trajectory.train_holdout_sequences_per_component for name in EXPECTED_TRAINING_COMPONENT_NAMES
        }:
            raise ValueError(f"{trajectory.trajectory_id}: global holdout contract drifted")
        if train_config.data.train_holdout_seed != trajectory.train_holdout_seed:
            raise ValueError(f"{trajectory.trajectory_id}: holdout seed drifted")
        if train_config.data.train_holdout_partition != EXPECTED_TRAIN_HOLDOUT_PARTITION:
            raise ValueError(f"{trajectory.trajectory_id}: holdout partition implementation drifted")
        if train_config.data.permutation_type != "feistel":
            raise ValueError(f"{trajectory.trajectory_id}: holdout permutation type drifted")
        expected_cap = None if trajectory.support_batches is None else {"dolma/starcoder": trajectory.support_batches}
        expected_start = (
            None if trajectory.support_start_batches is None else {"dolma/starcoder": trajectory.support_start_batches}
        )
        if train_config.data.max_train_batches != expected_cap:
            raise ValueError(f"{trajectory.trajectory_id}: finite-support cap drifted")
        if train_config.data.max_train_batches_start != expected_start:
            raise ValueError(f"{trajectory.trajectory_id}: finite-support offset drifted")
        if train_config.data.max_train_batches_subset_seed != trajectory.support_pool_seed:
            raise ValueError(f"{trajectory.trajectory_id}: support-pool seed drifted")
        if train_config.data.experiment_budget is not None or train_config.data.target_budget is not None:
            raise ValueError(f"{trajectory.trajectory_id}: global simulated budget leaked into training")
        if train_config.data.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{trajectory.trajectory_id}: global simulated subset leaked into training")
        if not training_component_names.issubset(train_config.data.components):
            raise ValueError(f"{trajectory.trajectory_id}: one or more training components are missing")
        phase_weights = train_config.data.train_weights
        if not isinstance(phase_weights, list) or [boundary for boundary, _ in phase_weights] != [
            0,
            trajectory.boundary_step,
        ]:
            raise ValueError(f"{trajectory.trajectory_id}: phase schedule drifted")
        if any(set(weights) != training_component_names for _, weights in phase_weights):
            raise ValueError(f"{trajectory.trajectory_id}: phase-weight component identities drifted")
        expected_phase_weights = [
            base._phase_leaf_weights(trajectory.phase_0_starcoder, nemotron=nemotron, starcoder=starcoder),
            base._phase_leaf_weights(trajectory.phase_1_starcoder, nemotron=nemotron, starcoder=starcoder),
        ]
        for phase_index, ((_, observed_weights), expected_weights) in enumerate(
            zip(phase_weights, expected_phase_weights, strict=True)
        ):
            if observed_weights != expected_weights:
                raise ValueError(f"{trajectory.trajectory_id}: phase-{phase_index} weights drifted")
        if train_config.data.mixture_block_size != base.MIXTURE_BLOCK_SIZE:
            raise ValueError(f"{trajectory.trajectory_id}: mixture block size drifted")
        expected_realized_counts = (
            trajectory.realized_phase_0_starcoder_per_block,
            trajectory.realized_phase_1_starcoder_per_block,
        )
        observed_realized_counts = tuple(
            _realized_component_count(weights, starcoder.name, base.MIXTURE_BLOCK_SIZE)
            for weights in expected_phase_weights
        )
        if observed_realized_counts != expected_realized_counts:
            raise ValueError(
                f"{trajectory.trajectory_id}: realized StarCoder counts drifted: "
                f"{observed_realized_counts} != {expected_realized_counts}"
            )
        checkpointer = train_config.trainer.checkpointer
        if checkpointer.keep != keep_by_id[trajectory.trajectory_id]:
            raise ValueError(f"{trajectory.trajectory_id}: permanent checkpoint policy drifted")
        if checkpointer.save_interval != CHECKPOINT_INTERVAL:
            raise ValueError(f"{trajectory.trajectory_id}: temporary checkpoint interval drifted")
        if checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{trajectory.trajectory_id}: temporary checkpoint retention drifted")
        if train_config.trainer.initialize_from is not None:
            raise ValueError(f"{trajectory.trajectory_id}: trajectory unexpectedly initializes from another run")
        if train_config.trainer.load_checkpoint is not None or train_config.trainer.load_checkpoint_path is not None:
            raise ValueError(f"{trajectory.trajectory_id}: automatic run-local resumption contract drifted")

        runtime_train_config = apply_output_path(train_config, expected_output)
        runtime_checkpointer = runtime_train_config.trainer.checkpointer
        expected_permanent_path = prefix_join(expected_output, "checkpoints")
        if runtime_checkpointer.base_path != expected_permanent_path:
            raise ValueError(f"{trajectory.trajectory_id}: runtime permanent checkpoint path drifted")
        temporary_path = runtime_checkpointer.temporary_base_path
        parsed_output = urllib.parse.urlparse(expected_output)
        output_component = f"{parsed_output.netloc}{parsed_output.path}".strip("/")
        expected_temporary_path = prefix_join(
            marin_prefix,
            f"tmp/ttl=14d/checkpoints-temp/{output_component}/checkpoints",
        )
        if temporary_path != expected_temporary_path:
            raise ValueError(f"{trajectory.trajectory_id}: runtime temporary checkpoint path is not central1-local")
        if runtime_checkpointer.append_run_id_to_base_path:
            raise ValueError(f"{trajectory.trajectory_id}: runtime checkpointer would append a second run identity")
        if runtime_checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{trajectory.trajectory_id}: runtime temporary retention drifted")
    logger.info("Materialized and audited all %d selected review-v9 runtime configs", len(trajectories))
    return len(trajectories)


def audit_sources(marin_prefix: str, trajectories: tuple[Trajectory, ...]) -> dict[str, str]:
    """Validate every training/evaluation cache and its central1-local identity."""
    nemotron, starcoder, validation = _training_data()
    training = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    required_support_batches = [
        (row.support_start_batches or 0) + row.support_batches for row in trajectories if row.support_batches is not None
    ]
    largest_support_batches = max(required_support_batches, default=0)
    dense._validate_starcoder_source(
        marin_prefix,
        cast(
            tuple[dense.SurfaceRun, ...],
            (
                _SourceValidationRequest(
                    starcoder_support_batches=largest_support_batches or None,
                    starcoder_realized_support_tokens=largest_support_batches * base.BATCH_SIZE * base.SEQ_LEN,
                ),
            ),
        ),
    )

    paths: dict[str, str] = {}
    for handle in training[:-1]:
        root = handle.path(marin_prefix)
        if not root.startswith(f"{marin_prefix}/"):
            raise ValueError(f"Cross-region cache path: {root}")
        expected_hash, expected_rows, expected_shards = EXPECTED_NEMOTRON_LEDGERS[handle.name]
        ledger_root = prefix_join(root, "train")
        ledger_path = prefix_join(ledger_root, "shard_ledger.json")
        observed_hash = _remote_sha256(ledger_path)
        if observed_hash != expected_hash:
            raise ValueError(f"Legacy Nemotron ledger drifted: {handle.name}: {observed_hash} != {expected_hash}")
        ledger = CacheLedger.load(ledger_root)
        if not ledger.is_finished or ledger.layout != CACHE_LAYOUT_CONSOLIDATED:
            raise ValueError(f"Legacy Nemotron cache is incomplete: {handle.name}")
        if ledger.total_num_rows != expected_rows or len(ledger.finished_shards) != expected_shards:
            raise ValueError(f"Legacy Nemotron cache shape drifted: {handle.name}")
        if ledger.metadata.preprocessor_metadata != EXPECTED_TOKENIZER_METADATA:
            raise ValueError(f"Legacy Nemotron tokenizer metadata drifted: {handle.name}")
        paths[f"{handle.name}:train"] = ledger_path

    for handle in validation:
        root = handle.path(marin_prefix)
        if not root.startswith(f"{marin_prefix}/"):
            raise ValueError(f"Cross-region cache path: {root}")
        stats_path = prefix_join(root, "validation/.stats.json")
        if not fsspec.open(stats_path).fs.exists(stats_path):
            raise FileNotFoundError(f"Required cache is incomplete: {stats_path}")
        paths[f"{handle.name}:validation"] = stats_path
    starcoder_path = starcoder.path(marin_prefix)
    if not starcoder_path.startswith(f"{marin_prefix}/"):
        raise ValueError(f"Cross-region StarCoder cache path: {starcoder_path}")
    starcoder_cache = TreeCache.load(
        prefix_join(starcoder_path, "train"),
        {"input_ids": np.zeros((0,), dtype=np.int32)},
    )
    observed_tokens = starcoder_cache.flat_field_length("input_ids")
    observed_sequences, trailing_tokens = _validate_starcoder_token_domain(observed_tokens)
    support_identity = _validate_starcoder_support_identity()
    logger.info(
        "Validated exact StarCoder flat field: %d tokens, %d packed sequences, %d trailing tokens",
        observed_tokens,
        observed_sequences,
        trailing_tokens,
    )
    logger.info(
        "Validated live StarCoder support identities: %s",
        support_identity["support_ordered_sequence_sha256"],
    )
    paths[f"{starcoder.name}:train"] = starcoder_path
    logger.info("Validated %d central1-local training/evaluation cache identities", len(paths))
    return paths


def _parse_set(value: str | None, option: str) -> frozenset[str] | None:
    if value is None:
        return None
    values = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{option} must contain at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, required=True)
    parser.add_argument("--arms", help="Comma-separated arm IDs for a staged launch or retry")
    parser.add_argument("--runs", help="Comma-separated exact trajectory IDs for a partial retry")
    parser.add_argument("--audit-manifest", action="store_true")
    parser.add_argument("--audit-runtime-configs", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping WSD80 gradient-conflict full panel in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child placement must remain central1-local: "
            f"region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    if args.max_concurrent > MAX_RELEASE_CONCURRENCY:
        raise ValueError(f"max_concurrent must not exceed {MAX_RELEASE_CONCURRENCY}")

    dense._validate_runtime_scientific_environment()
    selected_arms = _parse_set(args.arms, "--arms")
    selected_runs = _parse_set(args.runs, "--runs")
    if args.audit_manifest:
        _, trajectories, _ = load_design(selected_arms=selected_arms, selected_runs=selected_runs)
        logger.info("Audited %d frozen review-v9 manifest rows", len(trajectories))
        return

    trajectories, steps = build_training_steps(
        marin_prefix=args.marin_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        selected_arms=selected_arms,
        selected_runs=selected_runs,
    )
    audit_runtime_configs(trajectories, steps, marin_prefix=args.marin_prefix)
    if args.audit_runtime_configs:
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Lowered %d frozen review-v9 training graphs", len(steps))
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    if args.audit_source:
        audit_sources(args.marin_prefix, trajectories)
        return
    release = _load_training_release()
    _validate_training_selection(trajectories, release)
    owners = expected_output_owners(trajectories, steps, marin_prefix=args.marin_prefix)
    if selected_arms is not None or selected_runs is not None:
        all_trajectories, all_steps = build_training_steps(
            marin_prefix=args.marin_prefix,
            tpu_type=args.tpu_type,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
        )
        owners = expected_output_owners(all_trajectories, all_steps, marin_prefix=args.marin_prefix)
    inventory, trajectory_parent, temporary_parent = output_inventory.audit_outputs(
        args.marin_prefix,
        expected_root_owners=owners,
    )
    if inventory.partial_root_count or inventory.unexpected_root_count:
        raise RuntimeError(
            "Refusing to launch with partial or unexpected review-v9 output roots: "
            f"partial={inventory.partial_expected_roots}, unexpected={inventory.unexpected_roots}"
        )
    if inventory.bookkeeping_root_count or inventory.resumable_root_count:
        logger.info(
            "Resuming exact owned review-v9 state: bookkeeping=%d, checkpointed=%d, temporary_parent=%s",
            inventory.bookkeeping_root_count,
            inventory.resumable_root_count,
            temporary_parent,
        )
    completed_ids = {root.split("/", 1)[0] for root in inventory.completed_expected_roots}
    pending = tuple(
        (trajectory, step)
        for trajectory, step in zip(trajectories, steps, strict=True)
        if trajectory.trajectory_id not in completed_ids
    )
    if not pending:
        logger.info("All selected trajectories are complete under %s; nothing to launch", trajectory_parent)
        return
    trajectories = tuple(trajectory for trajectory, _ in pending)
    steps = tuple(step for _, step in pending)
    if completed_ids:
        logger.info("Skipping %d completed trajectories after live output audit", len(completed_ids))

    audit_sources(args.marin_prefix, trajectories)
    if args.max_concurrent > release["maximum_concurrent_trajectories"]:
        raise RuntimeError(
            f"Training release authorizes concurrency at most {release['maximum_concurrent_trajectories']}; "
            f"launch requested {args.max_concurrent}"
        )
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)))


if __name__ == "__main__":
    main()
