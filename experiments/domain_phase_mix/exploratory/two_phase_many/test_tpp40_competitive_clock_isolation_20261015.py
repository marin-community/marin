# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""TP044 source preparation checks; never evaluate its 24 scientific cases."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_competitive_clock_isolation_20261015 as runner,
)


def invented_case() -> runner.reference.Case:
    return runner.reference.Case("invented_test", "toy", "mixed", "equal_group", 1.3, 0.37, "toy", 1.8, (0.29, 0.69))


def test_metadata_selection_preserves_exact_tp041_subset_without_evaluation() -> None:
    cases = runner.cases()
    assert len(cases) == len({case.case_id for case in cases}) == 24
    assert {case.clock for case in cases} == {"equal_group", "global_pool"}
    assert {case.phase_first for case in cases} == {(0.625, 0.0), (0.5, 0.5), (0.375, 1.0)}
    assert all(case.inventory == "mixed" and case.aggregate_first == 0.5 and case.mu == 3 for case in cases)
    assert runner.reference.state is not runner.competitive
    assert runner.BACKEND.state is runner.competitive


def test_paired_toy_changes_only_state_clock_and_retains_both_readouts() -> None:
    case = invented_case()
    control = runner.competitive.Quadrature(1e-11, 512)
    old = runner.reference.evaluate_case(case, control)
    paired = runner.evaluate_pair(case, old, control)
    assert paired["status"] == "complete" and paired["numerical_checks_passed"]
    assert paired["novelty"] is old
    new = paired["donor_hazard"]
    assert old["features"]["native_benefit"] == new["features"]["native_benefit"]
    assert old["phase_epochs"] == new["phase_epochs"]
    assert np.max(abs(np.asarray(old["features"]["state_contrast"]) - new["features"]["state_contrast"])) > 1e-5
    assert set(new["head_values"]) == {"first_heavier", "first_lighter"}
    for head, changes in paired["head_changes_donor_minus_novelty"].items():
        assert changes["native_tied_loss"] == 0
        assert (
            changes["additive_gain"]
            == new["head_values"][head]["additive_gain"] - old["head_values"][head]["additive_gain"]
        )


def test_numerical_failure_keeps_reference_and_case(monkeypatch: pytest.MonkeyPatch) -> None:
    case = invented_case()
    old = {"case": asdict(case), "status": "complete", "numerical_checks_passed": True}

    def failed(*args: object) -> None:
        raise ArithmeticError("Invented panel-budget failure")

    monkeypatch.setattr(runner.BACKEND, "evaluate_case", failed)
    result = runner.evaluate_pair(case, old, runner.competitive.Quadrature(1e-11, 512))
    assert result["status"] == "paired_numerical_failure"
    assert not result["numerical_checks_passed"]
    assert result["novelty"] is old and result["case"] == asdict(case)
    assert result["donor_hazard"]["error"] == "Invented panel-budget failure"


def test_reference_adoption_retains_payload_and_rejects_changed_geometry(tmp_path: Path) -> None:
    case = invented_case()
    result = {"case": asdict(case), "status": "numerical_failure", "error": "retained historical toy"}
    path = tmp_path / "cells" / f"{case.case_id}.json"
    runner.reference.write(path, {"result": result, "result_sha256": runner.reference.payload_digest(result)})
    saved, digest = runner.load_reference(case, tmp_path)
    assert saved["error"] == result["error"] and digest == runner.reference.digest(path)
    changed = runner.reference.Case(case.case_id, "toy", "mixed", "equal_group", 1.4, 0.37, "toy", 1.8, (0.29, 0.69))
    with pytest.raises(ValueError, match="covariates"):
        runner.load_reference(changed, tmp_path)
