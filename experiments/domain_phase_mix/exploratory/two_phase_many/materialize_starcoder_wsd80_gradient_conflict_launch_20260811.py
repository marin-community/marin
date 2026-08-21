# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Resolve the frozen P2/P3 coordinates for the WSD80 gradient-conflict launch."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DESIGN_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260810"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_launch_20260811"
SOURCE_DESIGN = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
FITTED_SURFACES = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_matched_nd_stage1_20260731/"
    "stage3_dense_surface_results_20260802/fitted_surface_candidates.csv"
)
INPUT_TRAJECTORIES = DESIGN_DIR / "trajectory_manifest.csv"
INPUT_DESIGN_MANIFEST = DESIGN_DIR / "design_manifest.json"
OUTPUT_TRAJECTORIES = OUTPUT_DIR / "trajectory_manifest.csv"
OUTPUT_SELECTIONS = OUTPUT_DIR / "frozen_policy_selections.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "launch_manifest.json"

LAUNCH_DESIGN_VERSION = "2026-08-11-launch-v1"
EXPECTED_DESIGN_MANIFEST_SHA256 = "8ca0c9f433ef6fccf02fb7ed597d90e3b0ea3b663c58d12bb63b2a4a61bec0dc"
EXPECTED_SOURCE_DESIGN_SHA256 = "ca06420ec7c46379463091bdd55c5f720910ac38b46a0f37f08545ea9966ecbe"
EXPECTED_FITTED_SURFACES_SHA256 = "c8fce328418c6e2607244895420376b8842554a79512e7a14ffcb303c2fa3986"
FIXED_D_CELLS = (
    "r0_shared_h0640_s03820",
    "r1_increase_d_h0640_s07320",
    "r2_increase_d_h0640_s14960",
    "r3_increase_d_h0640_s28260",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verify_frozen_inputs() -> None:
    expected = {
        INPUT_DESIGN_MANIFEST: EXPECTED_DESIGN_MANIFEST_SHA256,
        SOURCE_DESIGN: EXPECTED_SOURCE_DESIGN_SHA256,
        FITTED_SURFACES: EXPECTED_FITTED_SURFACES_SHA256,
    }
    for path, expected_hash in expected.items():
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Frozen input drifted: {path}: {observed_hash} != {expected_hash}")


def _nearest_coordinate(
    coordinates: list[dict[str, Any]],
    *,
    p0: float,
    p1: float,
    tied: bool,
) -> tuple[dict[str, Any], float]:
    candidates = coordinates
    if tied:
        candidates = [
            row
            for row in coordinates
            if math.isclose(float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"]), abs_tol=1e-12)
        ]
    if not candidates:
        raise ValueError("Frozen source design has no eligible coordinates")
    selected = min(
        candidates,
        key=lambda row: (
            (float(row["phase_0_starcoder"]) - p0) ** 2 + (float(row["phase_1_starcoder"]) - p1) ** 2,
            row["coordinate_id"],
        ),
    )
    distance = math.dist(
        (p0, p1),
        (float(selected["phase_0_starcoder"]), float(selected["phase_1_starcoder"])),
    )
    return selected, distance


def materialize() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _verify_frozen_inputs()
    source_design = json.loads(SOURCE_DESIGN.read_text())
    coordinates = source_design["coordinates"]
    surfaces = {row["cell_id"]: row for row in read_csv(FITTED_SURFACES)}
    if not set(FIXED_D_CELLS).issubset(surfaces):
        raise ValueError("Frozen surface artifact omits one or more fixed-D cells")

    selections: dict[tuple[str, str], dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for cell_id in FIXED_D_CELLS:
        surface = surfaces[cell_id]
        targets = (
            (
                "fitted_surface_best_two_phase",
                float(surface["fitted_untied_p0"]),
                float(surface["fitted_untied_p1"]),
                False,
            ),
            (
                "fitted_surface_best_tied",
                float(surface["fitted_tied_weight"]),
                float(surface["fitted_tied_weight"]),
                True,
            ),
        )
        for policy_role, fitted_p0, fitted_p1, tied in targets:
            selected, distance = _nearest_coordinate(
                coordinates,
                p0=fitted_p0,
                p1=fitted_p1,
                tied=tied,
            )
            selection = {
                "cell_id": cell_id,
                "policy_role": policy_role,
                "selected_coordinate_id": selected["coordinate_id"],
                "fitted_phase_0_starcoder": fitted_p0,
                "fitted_phase_1_starcoder": fitted_p1,
                "selected_phase_0_starcoder": float(selected["phase_0_starcoder"]),
                "selected_phase_1_starcoder": float(selected["phase_1_starcoder"]),
                "snap_l2_distance": distance,
                "selected_coordinate_sources": "|".join(selected["sources"]),
                "fitted_surface_selected_ridge": float(surface["selected_ridge"]),
                "fitted_surface_spatial_cv_rmse": float(surface["spatial_cv_rmse"]),
            }
            selections[(cell_id, policy_role)] = selection
            selection_rows.append(selection)

    rows: list[dict[str, Any]] = []
    resolved_count = 0
    for row in read_csv(INPUT_TRAJECTORIES):
        output: dict[str, Any] = dict(row)
        if row["phase_0_starcoder"] == "" or row["phase_1_starcoder"] == "":
            selection = selections[(row["cell_id"], row["policy_role"])]
            p0 = float(selection["selected_phase_0_starcoder"])
            p1 = float(selection["selected_phase_1_starcoder"])
            beta = float(row["phase_0_fraction"])
            output.update(
                {
                    "phase_0_starcoder": p0,
                    "phase_1_starcoder": p1,
                    "aggregate_starcoder": beta * p0 + (1.0 - beta) * p1,
                    "phase_contrast_p0_minus_p1": p0 - p1,
                    "upstream_phase_contrast_p1_minus_p0": p1 - p0,
                }
            )
            resolved_count += 1
        rows.append(output)

    if len(rows) != 268 or resolved_count != 60:
        raise ValueError(f"Launch trajectory materialization drifted: rows={len(rows)}, resolved={resolved_count}")
    if any(row["phase_0_starcoder"] == "" or row["phase_1_starcoder"] == "" for row in rows):
        raise ValueError("Launch manifest still contains unresolved policy coordinates")
    return rows, selection_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, selections = materialize()
    write_csv(OUTPUT_TRAJECTORIES, rows)
    write_csv(OUTPUT_SELECTIONS, selections)
    manifest = {
        "launch_design_version": LAUNCH_DESIGN_VERSION,
        "description": "Hash-pinned resolution of the review-v5 P2/P3 policy placeholders.",
        "launch_allowed": False,
        "launch_gate": "Explicit release follows runtime materialization, regional safety, and independent review.",
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "trajectory_count": len(rows),
        "resolved_placeholder_count": 60,
        "selection_count": len(selections),
        "source_sha256": {
            str(INPUT_DESIGN_MANIFEST.relative_to(REPO_ROOT)): EXPECTED_DESIGN_MANIFEST_SHA256,
            str(SOURCE_DESIGN.relative_to(REPO_ROOT)): EXPECTED_SOURCE_DESIGN_SHA256,
            str(FITTED_SURFACES.relative_to(REPO_ROOT)): EXPECTED_FITTED_SURFACES_SHA256,
        },
        "artifact_sha256": {
            "trajectory_manifest.csv": file_sha256(OUTPUT_TRAJECTORIES),
            "frozen_policy_selections.csv": file_sha256(OUTPUT_SELECTIONS),
        },
        "launch_manifest_sha256": "",
    }
    manifest["launch_manifest_sha256"] = canonical_sha256(manifest)
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
