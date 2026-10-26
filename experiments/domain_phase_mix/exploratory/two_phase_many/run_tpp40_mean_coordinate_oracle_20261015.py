# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Four matching-law known-shape/floor oracles using the sealed TP028 procedure.

Each private family instance retains the unchanged seven-candidate NNLS search,
inner/full heads, query records and finite-recovery gates. No cross-law fit or
learned estimator is included. Data stages require exact registration, source
identity and separate execution approval; importing this module reads no data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_late_refresh_oracle_20261015 as physical
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mean_coordinate_controls_20261015 as law

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_mean_coordinate_controls_20261015"
FAMILIES = ("hazard_coordinate_mean", "benefit_coordinate_mean")
CANDIDATES = {"TP40-032": FAMILIES[0], "TP40-033": FAMILIES[1]}
WORLDS = tuple({"family": family, **world} for family in FAMILIES for world in physical.WORLDS)
SHAPE, FLOOR = physical.SHAPE, physical.FLOOR
PHYSICAL_SEAL = "eda1b26c39ed08a8e1615d09f7cab482380329521ee6378ac477d1fe640d726e"
PHYSICAL_AUDIT = "3417eef5648c98cf031bf0e8f565f7579533b979c4bfdbb945ad4de6f3f4ce2c"
PURE_DRAFT = "505ba5ae3ca6357bd24b9ad7446fd571810e9c33fa77eb9971e1f1ee457a5083"
APPROVAL_SCOPE = "mean_coordinate_matching_known_shape_floor_oracles"
EXECUTION_CONTRACT = {
    **physical.EXECUTION_CONTRACT,
    "readout_families": list(FAMILIES),
    "worlds": list(WORLDS),
    "oracle_searches": 4,
    "conditional_nnls_solves": 112,
    "cross_law_fits": False,
}
benchmark, mariner = physical.benchmark, physical.mariner


def design(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    strength: float,
    *,
    model: str,
) -> np.ndarray:
    return law.design_and_offset(early, late, alpha, shape, model, {"lambda": strength})[0]


def source_paths() -> set[Path]:
    return physical.source_paths() | {
        Path(__file__),
        Path(law.__file__),
        ROOT / "test_tpp40_mean_coordinate_controls_20261015.py",
        ROOT / "test_tpp40_mean_coordinate_oracle_20261015.py",
        OUTPUT / "registration.json",
        OUTPUT / "registration.draft.v1_pure_equations.json",
        OUTPUT / "prior_route_algebra_v1.md",
        OUTPUT / "prior_sources_v1.json",
        OUTPUT / "audit/author_validation_v1.json",
        physical.OUTPUT / "execution_identity.json",
        physical.OUTPUT / "result_manifest.json",
        physical.OUTPUT / "audit/oracle_replay_report_v1.json",
    }


def verify_adoption() -> None:
    """Adopt the sealed oracle, exact fixed truth and reviewed pure-law draft."""
    physical.registration()
    if physical.source_identity() != json.loads((physical.OUTPUT / "execution_identity.json").read_text()):
        raise ValueError("Frozen physical-harm oracle source/input/runtime changed")
    seal_path = physical.OUTPUT / "result_manifest.json"
    audit_path = physical.OUTPUT / "audit/oracle_replay_report_v1.json"
    if benchmark.sha256(seal_path) != PHYSICAL_SEAL:
        raise ValueError("Adopted physical-harm oracle seal changed")
    if benchmark.sha256(audit_path) != PHYSICAL_AUDIT or not json.loads(audit_path.read_text())["passed"]:
        raise ValueError("Adopted physical-harm oracle audit must pass")
    sealed = {item["path"]: item["sha256"] for item in json.loads(seal_path.read_text())["files"]}
    benchmark.verify_hashes(sealed, physical.OUTPUT)
    for path in (audit_path, physical.OUTPUT / "execution_identity.json"):
        if benchmark.sha256(path) != sealed[str(path.relative_to(physical.OUTPUT))]:
            raise ValueError("Oracle prerequisite evidence is absent from its seal")
    pure_path = OUTPUT / "registration.draft.v1_pure_equations.json"
    if benchmark.sha256(pure_path) != PURE_DRAFT:
        raise ValueError("Root-reviewed pure-equation draft changed")
    benchmark.verify_hashes(json.loads(pure_path.read_text())["frozen_reference_inputs"])


def registration() -> dict[str, Any]:
    config = json.loads((OUTPUT / "registration.json").read_text())
    if config["candidate_ids"] != CANDIDATES or config["status"] != "registered":
        raise ValueError("Root must register the reviewed TP032/033 oracle before execution")
    proposal = config["oracle_proposal"]
    if config["execution_contract"] != EXECUTION_CONTRACT or not proposal["implemented"]:
        raise ValueError("Registered four-world oracle protocol differs")
    if tuple(config["lambda_grid"]) != law.STRENGTHS or tuple(proposal["grid"]) != law.STRENGTHS:
        raise ValueError("Registered mean-coordinate strength grid differs")
    if (
        tuple(proposal["worlds"]) != WORLDS
        or proposal["noise_sd"] != 0
        or proposal["tolerance_bpb"] != physical.TOLERANCE
    ):
        raise ValueError("Registered mean-coordinate worlds, noise or oracle tolerance differ")
    if (
        config["physical_oracle_seal_sha256"] != PHYSICAL_SEAL
        or config["physical_oracle_audit_sha256"] != PHYSICAL_AUDIT
    ):
        raise ValueError("Registered oracle prerequisite differs")
    if config["environment"] != benchmark.environment_identity():
        raise ValueError("Registered numerical runtime differs")
    expected_paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(config["frozen_reference_inputs"]) != expected_paths:
        raise ValueError("Registration must pin the complete reviewed source/input set")
    benchmark.verify_hashes(config["frozen_reference_inputs"])
    verify_adoption()
    return config


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    registration()
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve prior execution identity; register a separate amended execution")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen mean-coordinate source/input/runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Separate exact mean-coordinate oracle approval is required")
    return current


def isolated_oracle(model: str) -> ModuleType:
    """Bind one matching-law family without mutating the canonical oracle."""
    if model not in FAMILIES:
        raise ValueError(f"Unknown oracle family: {model}")
    name = f"{__name__}._{model}"
    spec = importlib.util.spec_from_file_location(name, physical.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.OUTPUT = OUTPUT / model
    module.WORLDS = tuple(world for world in WORLDS if world["family"] == model)
    module.law = law
    module.design = partial(design, model=model)
    module.registration = registration
    module.source_identity = source_identity
    module.execution_identity = execution_identity
    return module


ORACLES = {family: isolated_oracle(family) for family in FAMILIES}


def write_stage_summary(stage: str) -> None:
    """Join verified family artifacts without recomputing any fit or metric."""
    if stage not in ("preflight", "oracle"):
        raise ValueError(f"Unknown oracle stage: {stage}")
    frozen = execution_identity()
    output_name = "preflight.json" if stage == "preflight" else "oracle_summary.json"
    receipt_name = "preflight_receipt.json" if stage == "preflight" else "oracle_summary_receipt.json"
    records, receipts = [], {}
    for family, module in ORACLES.items():
        receipt_path = module.OUTPUT / receipt_name
        module.verified_record(receipt_path, frozen, (output_name,))
        record = json.loads((module.OUTPUT / output_name).read_text())
        expected = tuple(world for world in WORLDS if world["family"] == family)
        if record["identity"] != frozen or tuple(item["world"] for item in record["worlds"]) != expected:
            raise ValueError("Family summary identity or matching-law world order changed")
        records.append(record)
        receipts[str(receipt_path.relative_to(OUTPUT))] = benchmark.sha256(receipt_path)
    summary = {
        "identity": frozen,
        "worlds": [item for record in records for item in record["worlds"]],
        "learned_floor_domain_feasible": all(record["learned_floor_domain_feasible"] for record in records),
        "interpretation": (
            "Four matching-law known-shape/floor finite-grid checks. Learned-floor feasibility is separate "
            "from oracle recovery. No cross-law, learned-estimator, empirical or raw-optimum qualification."
        ),
    }
    if stage == "preflight":
        summary["oracle_execution_valid"] = all(record["oracle_execution_valid"] for record in records)
    else:
        summary["passed"] = all(record["passed"] for record in records)
        summary["eligible_for_separate_learned_procedure_review"] = (
            summary["passed"] and summary["learned_floor_domain_feasible"]
        )
    benchmark.write_json(OUTPUT / output_name, summary)
    benchmark.write_json(
        OUTPUT / receipt_name,
        {
            "identity": execution_identity(),
            "outputs": {output_name: benchmark.sha256(OUTPUT / output_name)},
            "family_receipts": receipts,
        },
    )


def preflight() -> None:
    execution_identity()
    for module in ORACLES.values():
        module.preflight()
    write_stage_summary("preflight")


def oracle() -> None:
    frozen = execution_identity()
    receipt = physical.verified_record(OUTPUT / "preflight_receipt.json", frozen, ("preflight.json",))
    expected_receipts = {f"{family}/preflight_receipt.json" for family in FAMILIES}
    if set(receipt["family_receipts"]) != expected_receipts:
        raise ValueError("Preflight must cover both registered families")
    benchmark.verify_hashes(receipt["family_receipts"], OUTPUT)
    report = json.loads((OUTPUT / "preflight.json").read_text())
    if report["identity"] != frozen or not report["oracle_execution_valid"]:
        raise ValueError("All four truth worlds must pass preflight before any fitting")
    for module in ORACLES.values():
        module.oracle()
    write_stage_summary("oracle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "preflight", "oracle"))
    args = parser.parse_args()
    {"freeze": freeze, "preflight": preflight, "oracle": oracle}[args.stage]()


if __name__ == "__main__":
    main()
