# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

from tile_lifetime.h100_contract_map_benchmark import (
    REVIEWED_NUMERICAL_FLOORS_SHA256,
    BackendVariant,
    UlpAcceptanceMode,
    default_h100_contract_map_benchmark_plan,
)

ROOT = Path(__file__).parents[3]
FIXTURE = Path(__file__).parent / "fixtures/h100_contract_map_numerical_calibration.json"
GENERATOR = ROOT / "lib/tile_lifetime/benchmarks/h100_contract_map_numerical_calibration.py"


def test_cpu_calibration_covers_every_reviewed_case_output_and_heldout_seed() -> None:
    calibration = json.loads(FIXTURE.read_text())
    plan = default_h100_contract_map_benchmark_plan()
    records = calibration["records"]

    assert REVIEWED_NUMERICAL_FLOORS_SHA256 == "1db5b96bfc8adca9b352db6983c05bb9bed9b73431952f0b5d1fa705824dbc38"
    assert calibration["reviewed_numerical_floors_sha256"] == REVIEWED_NUMERICAL_FLOORS_SHA256
    assert calibration["jax_version"] == "0.10.1"
    assert calibration["platform"] == "cpu"
    assert len(records) == 64
    assert {(record["case_id"], record["role"], record["seed_kind"]) for record in records} == {
        (case.case_id, role, seed_kind)
        for case in plan.cases
        for role in ("forward", "dx", "dw0", "dw1")
        for seed_kind in ("canonical", "heldout_0", "heldout_1", "heldout_2")
    }

    ordinary = next(floor for floor in plan.numerical_floors if floor.backend is BackendVariant.ORDINARY_XLA)
    fast = next(floor for floor in plan.numerical_floors if floor.backend is BackendVariant.SHUTTLE_FAST)
    assert ordinary.ulp_acceptance is fast.ulp_acceptance is UlpAcceptanceMode.DIAGNOSTIC_ONLY
    for output_floor in ordinary.output_floors:
        fixture_floor = calibration["output_floors"][output_floor.output]
        assert fixture_floor["predeclared_maximum_absolute_error"] == output_floor.maximum_absolute_error
        assert fixture_floor["predeclared_mean_absolute_error"] == output_floor.mean_absolute_error
        assert fixture_floor["observed_maximum_absolute_error"] <= output_floor.maximum_absolute_error
        assert fixture_floor["observed_mean_absolute_error"] <= output_floor.mean_absolute_error
        assert fast.output_floor(output_floor.output) == output_floor

    canonical = [record for record in records if record["seed_kind"] == "canonical"]
    assert all(record["maximum_ulp_distance"] > 4 for record in canonical)
    assert all(record["mean_ulp_distance"] > 0.25 for record in canonical)


def test_checked_in_cpu_calibration_is_exactly_regenerable() -> None:
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = f"{ROOT / 'lib/tile_lifetime/src'}:{ROOT}"

    completed = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
