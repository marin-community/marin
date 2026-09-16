# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.domain_phase_mix.audit_delphi_tpp40_multiregion_resolved_paths import (
    audit_resolved_training_paths,
)
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 import (
    EXPECTED_FINAL_CHECKPOINT_STEP,
    EXPECTED_PHASE0_CHECKPOINT_STEP,
)


def _run_path(root: Path, run_order: int) -> Path:
    path = root / f"fit_{run_order:03d}_run_{run_order:05d}-hash"
    path.mkdir(parents=True)
    return path


def _write_phase0_checkpoint(path: Path) -> None:
    checkpoint = path / "checkpoints" / f"step-{EXPECTED_PHASE0_CHECKPOINT_STEP}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(
        json.dumps({"step": EXPECTED_PHASE0_CHECKPOINT_STEP, "is_temporary": False})
    )
    (checkpoint / "manifest.ocdbt").write_text("manifest")


def test_resolved_path_audit_accepts_completed_resumable_and_fresh(tmp_path: Path) -> None:
    completed = _run_path(tmp_path, 0)
    (completed / "hf" / f"step-{EXPECTED_FINAL_CHECKPOINT_STEP}").mkdir(parents=True)
    (completed / "hf" / f"step-{EXPECTED_FINAL_CHECKPOINT_STEP}" / "model.safetensors").write_text("model")
    (completed / ".executor_status").write_text("SUCCESS\n")
    resumable = _run_path(tmp_path, 1)
    _write_phase0_checkpoint(resumable)
    fresh = _run_path(tmp_path, 2)
    assignment = {
        "assignments": {
            "completed": [0],
            "east5": [1, 2],
            "europe": [],
            "resumable_east5": [1],
        }
    }

    result = audit_resolved_training_paths(
        assignment=assignment,
        paths_by_order={0: str(completed), 1: str(resumable), 2: str(fresh)},
        expected_root=str(tmp_path),
    )

    assert result["passed"]
    assert result["completed_count"] == 1
    assert result["resumable_count"] == 1
    assert result["fresh_count"] == 1


def test_resolved_path_audit_rejects_version_drift_for_completed_row(tmp_path: Path) -> None:
    drifted = _run_path(tmp_path, 0)
    assignment = {
        "assignments": {
            "completed": [0],
            "east5": [],
            "europe": [],
            "resumable_east5": [],
        }
    }

    with pytest.raises(ValueError, match="lacks its final marker"):
        audit_resolved_training_paths(
            assignment=assignment,
            paths_by_order={0: str(drifted)},
            expected_root=str(tmp_path),
        )


def test_resolved_path_audit_rejects_checkpoint_on_fresh_row(tmp_path: Path) -> None:
    fresh = _run_path(tmp_path, 0)
    _write_phase0_checkpoint(fresh)
    assignment = {
        "assignments": {
            "completed": [],
            "east5": [0],
            "europe": [],
            "resumable_east5": [],
        }
    }

    with pytest.raises(ValueError, match="Fresh run 0 has resumable or completed artifacts"):
        audit_resolved_training_paths(
            assignment=assignment,
            paths_by_order={0: str(fresh)},
            expected_root=str(tmp_path),
        )
