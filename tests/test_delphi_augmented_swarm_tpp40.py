# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import dataclass, replace

import pytest

from experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 import (
    _assignment_orders,
    _parse_run_orders,
    _regional_input_path,
    _reject_assignment_option_typos,
    _require_assignment_contract,
    _require_regional_input_path,
    _require_tpu_placement,
    _scientific_identity_hash,
    _tensor_parallel_size,
)


@dataclass(frozen=True)
class _IdentitySpec:
    scientific_coordinate: str
    target_flops: float
    train_steps: int
    realized_train_tokens: int
    expected_checkpoint_step: int
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    tensor_parallel_size: int


def test_scientific_identity_excludes_horizon_and_deployment() -> None:
    source = _IdentitySpec(
        scientific_coordinate="fit_007",
        target_flops=3e18,
        train_steps=3007,
        realized_train_tokens=1_000_000,
        expected_checkpoint_step=3006,
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        tensor_parallel_size=4,
    )
    deployed = replace(
        source,
        target_flops=3e19,
        train_steps=27_336,
        realized_train_tokens=10_000_000,
        expected_checkpoint_step=27_335,
        tpu_type="v6e-8",
        tpu_region="europe-west4",
        tpu_zone="europe-west4-a",
        tensor_parallel_size=2,
    )

    assert _scientific_identity_hash([source]) == _scientific_identity_hash([deployed])
    assert _scientific_identity_hash([source]) != _scientific_identity_hash(
        [replace(deployed, scientific_coordinate="fit_008")]
    )


def test_parse_run_orders_accepts_disjoint_ranges() -> None:
    assert _parse_run_orders("0,3-5,279") == (0, 3, 4, 5, 279)
    assert _parse_run_orders("all", expected_runs=3) == (0, 1, 2)


@pytest.mark.parametrize("value", ["1,1", "4-2", "-1", "280", "1,,2"])
def test_parse_run_orders_rejects_invalid_manifests(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_run_orders(value)


def test_regional_input_path_preserves_immutable_suffix() -> None:
    canonical = "gs://marin-us-east5/path/to/input.json"

    assert _regional_input_path(canonical, region="europe-west4") == "gs://marin-eu-west4/path/to/input.json"


def test_explicit_regional_input_rejects_cross_region_path() -> None:
    with pytest.raises(ValueError, match="must be under"):
        _require_regional_input_path(
            "gs://marin-us-east5/path/to/input.json",
            region="europe-west4",
            label="input",
        )


def test_assignment_file_mechanically_selects_region(tmp_path) -> None:
    payload = {
        "expected_runs": 6,
        "east5_root": (
            "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
            "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
        ),
        "europe_root": (
            "gs://marin-eu-west4/pinlin_calvin_xu/data_mixture/"
            "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
        ),
        "assignments": {
            "completed": [0],
            "east5": [1, 2, 3],
            "europe": [4, 5],
            "resumable_east5": [1],
        },
        "strata": {"east5": {"qsplit": 3}, "europe": {"qsplit": 2}},
    }
    payload["assignment_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps(payload))

    orders, audit = _assignment_orders(
        str(path),
        "europe",
        tpu_region="europe-west4",
        experiment_name="pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815",
        expected_assignment_sha256=payload["assignment_sha256"],
        expected_runs=6,
    )

    assert orders == (4, 5)
    assert audit["assignment_sha256"] == payload["assignment_sha256"]
    assert audit["assignment_completed_replayed_for_eval_count"] == 0

    east5_orders, east5_audit = _assignment_orders(
        str(path),
        "east5",
        tpu_region="us-east5",
        experiment_name="pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815",
        expected_assignment_sha256=payload["assignment_sha256"],
        expected_runs=6,
    )

    assert east5_orders == (0, 1, 2, 3)
    assert east5_audit["assignment_completed_replayed_for_eval_count"] == 1


def test_assignment_file_rejects_region_mismatch(tmp_path) -> None:
    payload = {
        "expected_runs": 2,
        "east5_root": "gs://marin-us-east5/experiment",
        "europe_root": "gs://marin-eu-west4/experiment",
        "assignments": {"completed": [], "east5": [0], "europe": [1], "resumable_east5": []},
    }
    payload["assignment_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="requires TPU region"):
        _assignment_orders(
            str(path),
            "europe",
            tpu_region="us-east5",
            experiment_name="experiment",
            expected_assignment_sha256=payload["assignment_sha256"],
            expected_runs=2,
        )


@pytest.mark.parametrize("group", ["completed", "east5", "europe", "resumable_east5"])
def test_assignment_file_rejects_duplicate_group_orders(tmp_path, group: str) -> None:
    assignments = {"completed": [], "east5": [0], "europe": [1], "resumable_east5": []}
    assignments[group] = [0, 0]
    payload = {
        "expected_runs": 2,
        "east5_root": "gs://marin-us-east5/experiment",
        "europe_root": "gs://marin-eu-west4/experiment",
        "assignments": assignments,
    }
    payload["assignment_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate"):
        _assignment_orders(
            str(path),
            "east5",
            tpu_region="us-east5",
            experiment_name="experiment",
            expected_assignment_sha256=payload["assignment_sha256"],
            expected_runs=2,
        )


def test_tpp40_tpu_topology_uses_physical_device_count() -> None:
    assert _tensor_parallel_size(896, "v5p-8") == 1
    assert _tensor_parallel_size(896, "v6e-8") == 1
    with pytest.raises(ValueError, match="Unsupported"):
        _tensor_parallel_size(896, "v4-8")


def test_tpp40_placement_rejects_cross_region_zone() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        _require_tpu_placement(tpu_type="v6e-8", region="europe-west4", zone="us-east5-b")


def test_tpp40_placement_rejects_accelerator_in_wrong_zone() -> None:
    with pytest.raises(ValueError, match="not configured"):
        _require_tpu_placement(tpu_type="v6e-8", region="europe-west4", zone="europe-west4-b")


def test_tpp40_placement_accepts_configured_accelerator_zone() -> None:
    _require_tpu_placement(tpu_type="v6e-8", region="europe-west4", zone="europe-west4-a")
    _require_tpu_placement(tpu_type="v5p-8", region="us-east5", zone="us-east5-a")


def test_production_europe_requires_frozen_assignment() -> None:
    with pytest.raises(ValueError, match="requires --assignment-file"):
        _require_assignment_contract(
            experiment_name=("pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"),
            tpu_region="europe-west4",
            assignment_file=None,
        )


def test_assignment_option_typos_are_not_forwarded() -> None:
    with pytest.raises(ValueError, match="Unknown assignment options"):
        _reject_assignment_option_typos(["--assignment_file", "manifest.json"])
