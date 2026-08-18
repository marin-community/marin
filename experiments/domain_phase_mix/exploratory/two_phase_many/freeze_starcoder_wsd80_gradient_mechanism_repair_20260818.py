# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the post-outcome gradient-mechanism repair over sealed v6 checkpoints."""

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from itertools import pairwise
from pathlib import Path
from typing import Any

import gcsfs
from marin.utilities.json_encoder import CustomJsonEncoder

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as parent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_gradient_mechanism_repair_v8_20260818"
RUNTIME_PATH = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_gradient_mechanism_repair.py"
ANALYZER_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/"
    "analyze_starcoder_wsd80_gradient_mechanism_repair_20260818.py"
)
PARENT_RELEASE_PATH = parent.OUTPUT_DIR / "release.json"
PARENT_FULL_MANIFEST_PATH = parent.OUTPUT_DIR / "full_probe_manifest.csv"
PARENT_CANARY_MANIFEST_PATH = parent.OUTPUT_DIR / "canary_probe_manifest.csv"
TRAJECTORY_MANIFEST_PATH = parent.DESIGN_DIR / "trajectory_manifest.csv"
ANALYSIS_CONTRACT_PATH = OUTPUT_DIR / "analysis_contract.json"
CANARY_MANIFEST_PATH = OUTPUT_DIR / "canary_mechanism_manifest.csv"
FULL_MANIFEST_PATH = OUTPUT_DIR / "full_mechanism_manifest.csv"
CHECKPOINT_PROVENANCE_PATH = OUTPUT_DIR / "checkpoint_object_provenance.csv"
PARENT_RESULT_PROVENANCE_PATH = OUTPUT_DIR / "parent_result_object_provenance.csv"
RELEASE_PATH = OUTPUT_DIR / "release.json"

MARIN_PREFIX = "gs://marin-us-central1"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_mechanism_repair_v8_20260818"
)
RELEASE_VERSION = "2026-08-18-gradient-mechanism-repair-v8"
PARENT_ARTIFACT_VERSION = "2026.08.16.6"
SCIENTIFIC_STATUS = "post_outcome_development_mechanism_repair_not_untouched_confirmation"

PRIMARY_ROLES = frozenset({"h2_primary", "h3_full_support_pair", "h5_preregistered_profile"})
SENSITIVITY_ROLES = frozenset({"h3_second_pool_sensitivity"})
MECHANISM_ROLES = PRIMARY_ROLES | SENSITIVITY_ROLES
TARGET_DISTRIBUTIONS = tuple(sorted(parent.TARGET_COMPONENTS))
GLOBAL_STARCODER = "starcoder_excluded_global"
SUPPORT_STARCODER = "starcoder_support_reference"
ON_POLICY_STARCODER = "starcoder_on_policy"
NEMOTRON = "nemotron_aggregate"
PRIMARY_UPDATE_CONTRAST = (GLOBAL_STARCODER, NEMOTRON)
FULL_STAGE_COUNTS = {1: 28, 2: 56}
H1_STATES = ("fraction_0p10", "fraction_0p25", "fraction_0p70", "decay_onset", "final")
EXECUTION_ACCEPTANCE = {
    "canary_max_concurrent": 14,
    "canary_min_max_distribution_block_count": 64,
    "canary_min_max_distribution_count": 7,
    "probe_batch_size": 64,
    "max_group_wall_seconds": 2_700,
    "max_peak_host_rss_bytes": 120 * 1024**3,
    "required_backend": "tpu",
    "required_device_kind_substring": "TPU",
    "minimum_device_count": 4,
    "minimum_local_device_count": 4,
    "max_no_data_update_abs_diff": 1e-6,
    "max_no_data_update_relative_diff": 1e-6,
    "allowed_missing_rows": 0,
    "allowed_unexpected_rows": 0,
    "allowed_invalid_documents": 0,
    "allowed_nonfinite_documents": 0,
    "allowed_resource_exhaustion_failures": 0,
    "stage_promotion_requires_exact_prior_stage_audit": True,
}

ANALYSIS_CONTRACT: dict[str, Any] = {
    "contract_version": "2026-08-18-gradient-mechanism-repair-analysis-v8",
    "scientific_status": SCIENTIFIC_STATUS,
    "outcomes_inspected_before_contract": True,
    "purpose": (
        "Repair missing persisted cross-statistics required by the already frozen H1, H2, H3, and H5 mechanism "
        "definitions. This release cannot restore untouched confirmatory status."
    ),
    "source_update_definition": (
        "Delta(q,t)-Delta(0,t) under the exactly restored optimizer state. Global-heldout StarCoder, Nemotron, "
        "and included-support StarCoder updates reuse the original v6 probe-row identities at every checkpoint. "
        "Every group must demonstrate that the no-data update is invariant to the source loss and RNG key before "
        "corrected updates are compared."
    ),
    "primary_source_contrast": list(PRIMARY_UPDATE_CONTRAST),
    "estimands": {
        "h1": {
            "status": "restricted_descriptive_subset",
            "selection_rule": (
                "Only trajectories already selected for H2, H3, or H5 are included; this is not complete H1 "
                "coverage of the original trajectory panel."
            ),
            "states": list(H1_STATES),
            "trajectory_inventory": {"m100a": 24, "full": 24, "m100b": 8, "total": 56},
            "row_count": 280,
            "statistic": (
                "projected raw-gradient cosine between global-heldout StarCoder and Nemotron, plus projected "
                "optimizer-update cosine between the same global-heldout StarCoder and Nemotron probe batches"
            ),
        },
        "h2": {
            "status": "development_repair_of_frozen_estimand",
            "states": {
                "mid": ["fraction_0p40", "fraction_0p55"],
                "late_pre_decay": ["decay_minus_256", "decay_minus_64"],
                "late_post_decay": ["decay_plus_64", "decay_plus_256"],
            },
            "alignment": "A_y=-<g_y,Delta(S)-Delta(N)>/(||g_y|| ||Delta(S)-Delta(N)||)",
            "seed_statistic": "(mean A_PL(late)-mean A_PL(mid))-(mean A_C4(late)-mean A_C4(mid))",
            "primary_support": "m100a",
            "primary_alternative": "greater",
            "inferential_unit": "training_seed",
        },
        "h3": {
            "status": "development_repair_of_frozen_estimand",
            "statistic": "paired m100a-minus-full difference in the H2 seed statistic",
            "support_separation": (
                "U_y(starcoder_support_reference)-U_y(starcoder_excluded_global), with "
                "U_y(q)=-g_y^T[Delta(q)-Delta(0)]"
            ),
            "unseen_utility_decline": (
                "mean U_y(starcoder_excluded_global, late_pre_decay)-" "mean U_y(starcoder_excluded_global, mid)"
            ),
            "support_separation_growth": (
                "mean [U_y(starcoder_support_reference)-U_y(starcoder_excluded_global)](late_pre_decay)-"
                "mean [U_y(starcoder_support_reference)-U_y(starcoder_excluded_global)](mid)"
            ),
            "primary_alternative": "two_sided",
            "inferential_unit": "training_seed",
            "m100b_role": "sensitivity_only",
        },
        "h5_profile": {
            "status": "secondary_development_repair_of_frozen_profile",
            "policies": ["boundary_beta_0p60", "boundary_beta_0p85"],
            "periods": {
                "mid": ["fraction_0p40", "fraction_0p55"],
                "pre": ["optimizer_decay_minus_256", "optimizer_decay_minus_64"],
                "post": ["optimizer_decay_onset", "optimizer_decay_plus_64"],
            },
            "primary_profile": "D_pre-D_mid",
            "secondary_profile": "D_post-D_pre",
        },
    },
    "h4_exclusion": (
        "H4 is excluded. Its calibration map, reliability threshold, and family-level test were not numerically "
        "frozen before outcomes; a new untouched trajectory panel is required."
    ),
    "multiplicity": (
        "No new confirmatory familywise claim is permitted. H2/H3/H5-profile results are development evidence and "
        "must be reported with effect sizes, seed bootstrap intervals, and exact sign-flip p-values labeled unadjusted."
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, cls=CustomJsonEncoder, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _write_create_only(path, buffer.getvalue().encode())


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen release artifact already exists with different content: {path}") from error


def _write_json(path: Path, value: Any) -> None:
    _write_create_only(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _manifest_groups(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        groups[row["group_id"]].append(row)
    return groups


def _trajectory_inventory() -> dict[str, dict[str, str]]:
    return {row["trajectory_id"]: row for row in _read_csv(TRAJECTORY_MANIFEST_PATH)}


def _repair_identity(payload: dict[str, Any], *, scope: str) -> tuple[str, str]:
    identity = {"scope": scope, **payload, "parent_release_sha256": _parent_release()["release_sha256"]}
    digest = canonical_sha256(identity)[:24]
    return f"mechanism_{digest}", f"mechanism_group_{digest}"


def _parent_release() -> dict[str, Any]:
    release = json.loads(PARENT_RELEASE_PATH.read_text())
    expected = parent.canonical_sha256({**release, "release_sha256": ""})
    if release["release_sha256"] != expected:
        raise ValueError("Parent v6 release hash is invalid")
    for summary in release["manifests"].values():
        path = REPO_ROOT / summary["path"]
        if file_sha256(path) != summary["sha256"]:
            raise ValueError(f"Parent v6 manifest drifted: {summary['path']}")
    for name, sha256 in release["source_design_files"].items():
        path = parent.DESIGN_DIR / name
        if file_sha256(path) != sha256:
            raise ValueError(f"Parent v6 design input drifted: {name}")
    return release


def _distribution_contract(
    group: list[dict[str, str]],
    *,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int], dict[str, str], dict[str, str]]:
    by_distribution = {row["distribution_id"]: row for row in group}
    requested = (*source_ids, *target_ids)
    missing = set(requested) - set(by_distribution)
    if missing:
        raise ValueError(f"Parent probe group omits required distributions: {sorted(missing)}")
    block_counts = {name: int(by_distribution[name]["replicate_blocks"]) for name in requested}
    update_draws = {
        name: (
            min(int(by_distribution[name]["optimizer_update_draw_count"]), block_counts[name] // 2)
            if name in source_ids
            else 0
        )
        for name in requested
    }
    sequence_sets = {name: by_distribution[name]["probe_sequence_set_id"] for name in requested}
    probe_row_ids = {name: by_distribution[name]["row_id"] for name in requested}
    return block_counts, update_draws, sequence_sets, probe_row_ids


def _repair_row(
    group: list[dict[str, str]],
    *,
    scope: str,
    trajectory: dict[str, str] | None,
    source_ids: tuple[str, ...],
    target_ids: tuple[str, ...],
    analysis_role: str,
) -> dict[str, Any]:
    representative = group[0]
    block_counts, update_draws, sequence_sets, probe_row_ids = _distribution_contract(
        group,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    payload = {
        "trajectory_id": representative["trajectory_id"],
        "checkpoint_label": representative["checkpoint_label"],
        "checkpoint_step": int(representative["checkpoint_step"]),
        "expected_restored_state_step": int(representative["expected_restored_state_step"]),
        "checkpoint_uri": representative["checkpoint_uri"],
        "parent_probe_group_id": representative["group_id"],
        "train_config_sha256": representative["train_config_sha256"],
        "analysis_role": analysis_role,
        "source_distribution_ids_json": canonical_json(source_ids),
        "target_distribution_ids_json": canonical_json(target_ids),
        "distribution_block_counts_json": canonical_json(block_counts),
        "distribution_update_draw_counts_json": canonical_json(update_draws),
        "distribution_sequence_set_ids_json": canonical_json(sequence_sets),
        "distribution_probe_row_ids_json": canonical_json(probe_row_ids),
        "scientific_status": SCIENTIFIC_STATUS,
        "endpoint_metrics_read_by_runner": False,
    }
    if trajectory is not None:
        payload.update(
            {
                "arm": trajectory["arm"],
                "support_id": trajectory["support_id"],
                "training_seed": int(trajectory["training_seed"]),
                "policy_role": trajectory["policy_role"],
            }
        )
    else:
        payload.update({"arm": "canary", "support_id": "pipeline_only", "training_seed": -1, "policy_role": "canary"})
    row_id, group_id = _repair_identity(payload, scope=scope)
    return {**payload, "scope": scope, "row_id": row_id, "group_id": group_id}


def _canary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _manifest_groups(PARENT_CANARY_MANIFEST_PATH).values():
        rows.append(
            _repair_row(
                group,
                scope="canary",
                trajectory=None,
                source_ids=(GLOBAL_STARCODER, SUPPORT_STARCODER, NEMOTRON),
                target_ids=TARGET_DISTRIBUTIONS,
                analysis_role="pipeline_schema_preflight_only",
            )
        )
    return sorted(rows, key=lambda row: row["group_id"])


def _full_rows() -> list[dict[str, Any]]:
    groups = _manifest_groups(PARENT_FULL_MANIFEST_PATH)
    trajectories = _trajectory_inventory()
    mechanism_trajectories = {
        group[0]["trajectory_id"] for group in groups.values() if group[0]["analysis_role"] in MECHANISM_ROLES
    }
    rows: list[dict[str, Any]] = []
    for group in groups.values():
        representative = group[0]
        role = representative["analysis_role"]
        trajectory_id = representative["trajectory_id"]
        if role in MECHANISM_ROLES:
            source_ids = (GLOBAL_STARCODER, NEMOTRON)
            if role in {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}:
                source_ids = (GLOBAL_STARCODER, SUPPORT_STARCODER, NEMOTRON)
            target_ids = TARGET_DISTRIBUTIONS
        elif role == "descriptive_trajectory" and trajectory_id in mechanism_trajectories:
            source_ids = (GLOBAL_STARCODER, NEMOTRON)
            target_ids = ()
            role = "h1_trajectory_extension"
        else:
            continue
        rows.append(
            _repair_row(
                group,
                scope="full",
                trajectory=trajectories[trajectory_id],
                source_ids=source_ids,
                target_ids=target_ids,
                analysis_role=role,
            )
        )
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["group_id"]):
        buckets[(row["analysis_role"], row["support_id"], row["policy_role"])].append(row)
    interleaved: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                interleaved.append(buckets[key].pop(0))
    for index, row in enumerate(interleaved):
        if index < FULL_STAGE_COUNTS[1]:
            row["launch_stage"] = 1
        elif index < FULL_STAGE_COUNTS[1] + FULL_STAGE_COUNTS[2]:
            row["launch_stage"] = 2
        else:
            row["launch_stage"] = 3
    return sorted(interleaved, key=lambda row: row["group_id"])


def _checkpoint_inventory(rows_by_scope: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    checkpoints = sorted({(scope, row["checkpoint_uri"]) for scope, rows in rows_by_scope.items() for row in rows})
    fs = gcsfs.GCSFileSystem()

    def inventory_one(item: tuple[str, str]) -> list[dict[str, Any]]:
        scope, uri = item
        objects = fs.find(uri.removeprefix("gs://"), detail=True)
        if not objects:
            raise ValueError(f"Checkpoint inventory is empty: {uri}")
        result = []
        for path, info in sorted(objects.items()):
            result.append(
                {
                    "scope": scope,
                    "checkpoint_uri": uri,
                    "object_path": f"gs://{path}",
                    "size": int(info["size"]),
                    "generation": str(info["generation"]),
                    "md5_hash": str(info.get("md5Hash", "")),
                    "crc32c": str(info.get("crc32c", "")),
                    "etag": str(info.get("etag", "")),
                }
            )
        return result

    with ThreadPoolExecutor(max_workers=64) as executor:
        inventories = executor.map(inventory_one, checkpoints)
        return [row for inventory in inventories for row in inventory]


def _parent_result_inventory(rows_by_scope: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    parent_release = _parent_release()
    references: dict[str, dict[str, Any]] = {}
    for scope, rows in rows_by_scope.items():
        for row in rows:
            probe_row_ids = json.loads(row["distribution_probe_row_ids_json"])
            for source in json.loads(row["source_distribution_ids_json"]):
                uri = "/".join(
                    (
                        parent_release["result_root"].rstrip("/"),
                        scope,
                        "probe",
                        row["parent_probe_group_id"],
                        PARENT_ARTIFACT_VERSION,
                        "rows",
                        f"{probe_row_ids[source]}.json",
                    )
                )
                references[uri] = {
                    "scope": scope,
                    "parent_probe_group_id": row["parent_probe_group_id"],
                    "parent_probe_row_id": probe_row_ids[source],
                    "distribution_id": source,
                    "object_uri": uri,
                }
    fs = gcsfs.GCSFileSystem()

    def inventory_one(reference: dict[str, Any]) -> dict[str, Any]:
        plain_path = reference["object_uri"].removeprefix("gs://")
        info = fs.info(plain_path)
        with fs.open(plain_path, "rb") as handle:
            payload = handle.read()
        document = json.loads(payload)
        if (
            document.get("release_sha256") != parent_release["release_sha256"]
            or document.get("row", {}).get("row_id") != reference["parent_probe_row_id"]
            or document.get("row", {}).get("distribution_id") != reference["distribution_id"]
            or document.get("endpoint_metrics_read") is not False
        ):
            raise ValueError(f"Parent probe result identity drifted: {reference['object_uri']}")
        return {
            **reference,
            "size": int(info["size"]),
            "generation": str(info["generation"]),
            "md5_hash": str(info.get("md5Hash", "")),
            "crc32c": str(info.get("crc32c", "")),
            "etag": str(info.get("etag", "")),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "parent_identity_sha256": str(document["identity_sha256"]),
        }

    with ThreadPoolExecutor(max_workers=64) as executor:
        return sorted(executor.map(inventory_one, references.values()), key=lambda row: row["object_uri"])


def _workload_shape_sha256(row: dict[str, Any]) -> str:
    blocks = json.loads(row["distribution_block_counts_json"])
    sources = json.loads(row["source_distribution_ids_json"])
    targets = json.loads(row["target_distribution_ids_json"])
    return canonical_sha256(
        {
            "source_block_counts": {name: int(blocks[name]) for name in sources},
            "target_block_counts": {name: int(blocks[name]) for name in targets},
            "probe_batch_size": EXECUTION_ACCEPTANCE["probe_batch_size"],
        }
    )


def _validate_release_design(canary_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_concurrency = (28, 56, 64)
    if any(current > 2 * prior for prior, current in pairwise(stage_concurrency)):
        raise ValueError("A full launch stage increases concurrency by more than 2x")
    h1_rows = [row for row in full_rows if row["analysis_role"] == "h1_trajectory_extension"]
    h1_trajectories = {row["trajectory_id"] for row in h1_rows}
    h1_support_counts = {
        support: len({row["trajectory_id"] for row in h1_rows if row["support_id"] == support})
        for support in ("m100a", "full", "m100b")
    }
    expected_h1 = ANALYSIS_CONTRACT["estimands"]["h1"]
    if (
        len(h1_rows) != expected_h1["row_count"]
        or len(h1_trajectories) != expected_h1["trajectory_inventory"]["total"]
        or h1_support_counts != {key: expected_h1["trajectory_inventory"][key] for key in h1_support_counts}
        or {row["checkpoint_label"] for row in h1_rows} != set(expected_h1["states"])
    ):
        raise ValueError("Restricted H1 trajectory/state inventory drifted")

    full_shapes = {_workload_shape_sha256(row) for row in full_rows}
    stage1_shapes = {_workload_shape_sha256(row) for row in full_rows if int(row["launch_stage"]) == 1}
    if stage1_shapes != full_shapes:
        raise ValueError("Stage 1 must exercise every frozen full-panel workload shape")
    canary_blocks = [max(json.loads(row["distribution_block_counts_json"]).values()) for row in canary_rows]
    canary_distributions = [len(json.loads(row["distribution_block_counts_json"])) for row in canary_rows]
    if max(canary_blocks) < EXECUTION_ACCEPTANCE["canary_min_max_distribution_block_count"]:
        raise ValueError("Canary omits the frozen maximum block-count workload")
    if max(canary_distributions) < EXECUTION_ACCEPTANCE["canary_min_max_distribution_count"]:
        raise ValueError("Canary omits the frozen maximum distribution-count workload")
    return {
        "h1_trajectory_count": len(h1_trajectories),
        "h1_row_count": len(h1_rows),
        "full_workload_shape_sha256": sorted(full_shapes),
        "stage1_workload_shape_sha256": sorted(stage1_shapes),
        "canary_workload_shape_sha256": sorted({_workload_shape_sha256(row) for row in canary_rows}),
    }


def freeze() -> dict[str, Any]:
    """Materialize a hash-pinned repair release without reading endpoint outcomes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(ANALYSIS_CONTRACT_PATH, ANALYSIS_CONTRACT)
    canary_rows = _canary_rows()
    full_rows = _full_rows()
    design_validation = _validate_release_design(canary_rows, full_rows)
    rows_by_scope = {"canary": canary_rows, "full": full_rows}
    checkpoint_inventory = _checkpoint_inventory(rows_by_scope)
    parent_result_inventory = _parent_result_inventory(rows_by_scope)
    _write_csv(CANARY_MANIFEST_PATH, canary_rows)
    _write_csv(FULL_MANIFEST_PATH, full_rows)
    _write_csv(CHECKPOINT_PROVENANCE_PATH, checkpoint_inventory)
    _write_csv(PARENT_RESULT_PROVENANCE_PATH, parent_result_inventory)
    parent_release = _parent_release()
    implementation_paths = (Path(__file__).resolve(), RUNTIME_PATH, ANALYZER_PATH)
    for path in implementation_paths:
        if not path.exists():
            raise ValueError(f"Repair implementation is missing: {path}")
    release = {
        "release_version": RELEASE_VERSION,
        "release_sha256": "",
        "scientific_status": SCIENTIFIC_STATUS,
        "outcomes_inspected_before_repair": True,
        "endpoint_metrics_read_by_runner": False,
        "parent_release_path": str(PARENT_RELEASE_PATH.relative_to(REPO_ROOT)),
        "parent_release_file_sha256": file_sha256(PARENT_RELEASE_PATH),
        "parent_release_sha256": parent_release["release_sha256"],
        "result_root": RESULT_ROOT,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": MARIN_PREFIX,
        "analysis_contract": {
            "path": str(ANALYSIS_CONTRACT_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(ANALYSIS_CONTRACT_PATH),
        },
        "manifests": {
            "canary": {
                "path": str(CANARY_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(CANARY_MANIFEST_PATH),
                "row_count": len(canary_rows),
            },
            "full": {
                "path": str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(FULL_MANIFEST_PATH),
                "row_count": len(full_rows),
            },
            "checkpoint_provenance": {
                "path": str(CHECKPOINT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(CHECKPOINT_PROVENANCE_PATH),
                "row_count": len(checkpoint_inventory),
                "checkpoint_count": len({row["checkpoint_uri"] for row in checkpoint_inventory}),
            },
            "parent_result_provenance": {
                "path": str(PARENT_RESULT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(PARENT_RESULT_PROVENANCE_PATH),
                "row_count": len(parent_result_inventory),
                "object_count": len({row["object_uri"] for row in parent_result_inventory}),
            },
        },
        "implementation_files": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in implementation_paths},
        "parent_implementation_files": parent_release["implementation_files"],
        "parent_result_artifact_version": PARENT_ARTIFACT_VERSION,
        "h4_included": False,
        "h4_exclusion": ANALYSIS_CONTRACT["h4_exclusion"],
        "execution_acceptance": EXECUTION_ACCEPTANCE,
        "design_validation": design_validation,
        "full_launch_stages": {
            "1": {"row_count": FULL_STAGE_COUNTS[1], "max_concurrent": 28},
            "2": {"row_count": FULL_STAGE_COUNTS[2], "max_concurrent": 56},
            "3": {"row_count": len(full_rows) - sum(FULL_STAGE_COUNTS.values()), "max_concurrent": 64},
        },
    }
    release["release_sha256"] = canonical_sha256({**release, "release_sha256": ""})
    _write_json(RELEASE_PATH, release)
    return release


def main() -> None:
    print(json.dumps(freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
