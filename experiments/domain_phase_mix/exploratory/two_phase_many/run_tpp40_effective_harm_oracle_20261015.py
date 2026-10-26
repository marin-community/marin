# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Matched effective-harm oracle using the frozen TP40-028 procedure privately.

Only the readout and output/provenance bindings change. The fixed truth head,
shape, floor, two worlds, seven strengths, split geometry, NNLS procedure and
functional-recovery gates are unchanged. No cross-law or learned fit is included.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_oracle_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_effective_harm_readout_20261015 as law

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_effective_harm_readout_20261015"
PHYSICAL_SEAL = "eda1b26c39ed08a8e1615d09f7cab482380329521ee6378ac477d1fe640d726e"
PHYSICAL_AUDIT = "3417eef5648c98cf031bf0e8f565f7579533b979c4bfdbb945ad4de6f3f4ce2c"
benchmark = physical.benchmark
mariner = physical.mariner
SHAPE, FLOOR, WORLDS = physical.SHAPE, physical.FLOOR, physical.WORLDS
EXECUTION_CONTRACT = {**physical.EXECUTION_CONTRACT, "readout_family": "late_refresh_effective_harm"}


def design(early: np.ndarray, late: np.ndarray, alpha: float, shape: dict[str, float], strength: float) -> np.ndarray:
    return law.design_and_offset(early, late, alpha, shape, "late_refresh_effective_harm", {"lambda": strength})[0]


def source_paths() -> set[Path]:
    return physical.source_paths() | {
        Path(__file__),
        Path(law.__file__),
        ROOT / "test_tpp40_effective_harm_readout_20261015.py",
        ROOT / "test_tpp40_effective_harm_oracle_20261015.py",
        OUTPUT / "registration.json",
        OUTPUT / "registration.draft.v1_pure_equations.json",
        OUTPUT / "audit/root_pure_review_v1.json",
        physical.OUTPUT / "execution_identity.json",
        physical.OUTPUT / "result_manifest.json",
        physical.OUTPUT / "audit/oracle_replay_report_v1.json",
        physical.OUTPUT / "descriptive/effective_harm_assessment_v1.md",
        physical.OUTPUT / "descriptive/effective_harm_prior_rows_v1.json",
        physical.OUTPUT / "descriptive/effective_harm_receipt_v1.json",
    }


def verify_adoption() -> None:
    """Adopt the audited physical-harm procedure and reviewed pure readout exactly."""
    physical.registration()
    old = json.loads((physical.OUTPUT / "execution_identity.json").read_text())
    if physical.source_identity() != old:
        raise ValueError("Frozen physical-harm oracle dependencies changed")
    audit = physical.OUTPUT / "audit/oracle_replay_report_v1.json"
    if benchmark.sha256(physical.OUTPUT / "result_manifest.json") != PHYSICAL_SEAL:
        raise ValueError("Adopted physical-harm oracle seal changed")
    if benchmark.sha256(audit) != PHYSICAL_AUDIT or not json.loads(audit.read_text())["passed"]:
        raise ValueError("Adopted physical-harm oracle numerical audit must pass")
    seal = json.loads((physical.OUTPUT / "result_manifest.json").read_text())
    sealed = {item["path"]: item["sha256"] for item in seal["files"]}
    benchmark.verify_hashes(sealed, physical.OUTPUT)
    for path in (audit, physical.OUTPUT / "execution_identity.json"):
        if benchmark.sha256(path) != sealed[str(path.relative_to(physical.OUTPUT))]:
            raise ValueError("Adopted oracle evidence is absent from its seal")
    pure = json.loads((OUTPUT / "registration.draft.v1_pure_equations.json").read_text())
    review = json.loads((OUTPUT / "audit/root_pure_review_v1.json").read_text())
    if (
        not review["no_blocking_issue"]
        or benchmark.sha256(OUTPUT / "registration.draft.v1_pure_equations.json") != review["draft_sha256"]
    ):
        raise ValueError("Pure readout draft differs from its root-reviewed version")
    benchmark.verify_hashes(pure["source_hashes"])


def registration() -> dict[str, Any]:
    config = json.loads((OUTPUT / "registration.json").read_text())
    if config["candidate_id"] != "TP40-029" or config["status"] != "registered":
        raise ValueError("Root must register the reviewed TP40-029 oracle before execution")
    if config["execution_contract"] != EXECUTION_CONTRACT:
        raise ValueError("Registered oracle protocol differs from the unchanged two-world procedure")
    if (
        config["physical_oracle_seal_sha256"] != PHYSICAL_SEAL
        or config["physical_oracle_audit_sha256"] != PHYSICAL_AUDIT
    ):
        raise ValueError("Registered physical-oracle prerequisite differs")
    if config["environment"] != benchmark.environment_identity():
        raise ValueError("Registered numerical runtime differs")
    paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(config["frozen_reference_inputs"]) != paths:
        raise ValueError("Registration must pin the complete reviewed source/input set")
    benchmark.verify_hashes(config["frozen_reference_inputs"])
    verify_adoption()
    return config


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen effective-harm source/input/runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "effective_harm_known_shape_floor_oracle",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Separate exact effective-harm oracle approval is required")
    return current


def isolated_oracle() -> ModuleType:
    """Rebind readout, artifacts and approval without changing the frozen module."""
    name = f"{__name__}._physical_oracle"
    spec = importlib.util.spec_from_file_location(name, physical.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.OUTPUT = OUTPUT
    module.law = law
    module.design = design
    module.registration = registration
    module.source_identity = source_identity
    module.execution_identity = execution_identity
    return module


_ORACLE = isolated_oracle()
oracle_fit = _ORACLE.oracle_fit
oracle_diagnostic = _ORACLE.oracle_diagnostic
local_sensitivity = _ORACLE.local_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "preflight", "oracle"))
    args = parser.parse_args()
    {"freeze": _ORACLE.freeze, "preflight": _ORACLE.preflight, "oracle": _ORACLE.oracle}[args.stage]()


if __name__ == "__main__":
    main()
