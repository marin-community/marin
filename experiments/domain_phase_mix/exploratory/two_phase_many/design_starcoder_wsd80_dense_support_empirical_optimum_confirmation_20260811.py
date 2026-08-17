# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

"""Freeze fresh-seed confirmations of dense-support empirical policy-class minima."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_dense_support_surfaces_20260808 as source_designer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SOURCE_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_surface_design_20260808.json"
OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs/starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
)
SOURCE_OBSERVATIONS_PATH = OUTPUT_DIR / "coverage_observations.csv"
DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.csv"
SELECTION_PATH = OUTPUT_DIR / "selected_policies.csv"
REPORT_PATH = OUTPUT_DIR / "report.md"

DESIGN_VERSION = "2026-08-11-v1"
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
DISCOVERY_SEED = 20_260_711
FRESH_SEEDS = (20_260_821, 20_260_822, 20_260_823, 20_260_824, 20_260_825)
POLICY_CLASSES = ("tied", "untied")
MINIMUM_UNTIED_CONTRAST = 0.04
EXPECTED_BLOCKS = 28
EXPECTED_COORDINATES_PER_BLOCK = 125
EXPECTED_COVERAGE_ROWS = EXPECTED_BLOCKS * EXPECTED_COORDINATES_PER_BLOCK
EXPECTED_SELECTED_POLICIES = EXPECTED_BLOCKS * len(POLICY_CLASSES)
EXPECTED_RUNS = EXPECTED_SELECTED_POLICIES * len(FRESH_SEEDS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_payload() -> dict[str, Any]:
    payload = json.loads(SOURCE_DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = payload.pop("design_sha256")
    observed = _canonical_sha256(payload)
    if claimed != observed:
        raise ValueError(f"Source design self-hash mismatch: {claimed} != {observed}")
    payload["design_sha256"] = claimed
    return payload


def _coverage_observations(source_payload: dict[str, Any]) -> pd.DataFrame:
    observations = pd.read_csv(SOURCE_OBSERVATIONS_PATH)
    required = {
        "cell_id",
        "support_id",
        "coordinate_id",
        "run_name",
        "data_seed",
        "replicate_kind",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "phase_contrast",
        "bpb",
        "final_step",
        "expected_step",
        "is_alias",
    }
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Coverage observations are missing columns: {sorted(missing)}")
    if len(observations) != EXPECTED_COVERAGE_ROWS:
        raise ValueError(f"Expected {EXPECTED_COVERAGE_ROWS} complete coverage rows, got {len(observations)}")
    if observations[list(required)].isna().any().any():
        raise ValueError("Coverage observations contain missing required values")
    if set(observations["data_seed"].astype(int)) != {DISCOVERY_SEED}:
        raise ValueError("Coverage observations contain a non-discovery seed")
    if set(observations["replicate_kind"]) != {"coverage"}:
        raise ValueError("Coverage observations contain a non-coverage row")
    if not observations["final_step"].astype(int).eq(observations["expected_step"].astype(int)).all():
        raise ValueError("Coverage observations contain an incorrect final metric step")
    if not pd.to_numeric(observations["bpb"], errors="coerce").notna().all():
        raise ValueError("Coverage observations contain a non-finite BPB")
    block_sizes = observations.groupby(["cell_id", "support_id"], sort=False).size()
    if len(block_sizes) != EXPECTED_BLOCKS or not block_sizes.eq(EXPECTED_COORDINATES_PER_BLOCK).all():
        raise ValueError(f"Coverage block cardinality drifted: {block_sizes.to_dict()}")
    if observations.duplicated(["cell_id", "support_id", "coordinate_id"]).any():
        raise ValueError("Coverage observations contain duplicate block-coordinate rows")
    expected_coverage_aliases = sum(
        row["replicate_kind"] == "coverage" for row in source_payload["deterministic_aliases"]
    )
    if int(observations["is_alias"].astype(bool).sum()) != expected_coverage_aliases:
        raise ValueError("Coverage alias count differs from the frozen source design")
    return observations


def _selected_policies(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cell_id, support_id), block in observations.groupby(["cell_id", "support_id"], sort=True):
        contrast = (block["phase_1_starcoder"] - block["phase_0_starcoder"]).abs()
        masks = {
            "tied": contrast.le(1e-12),
            "untied": contrast.ge(MINIMUM_UNTIED_CONTRAST - 1e-12),
        }
        for policy_class in POLICY_CLASSES:
            eligible = block.loc[masks[policy_class]].sort_values(["bpb", "coordinate_id"], kind="stable")
            if eligible.empty:
                raise ValueError(f"{cell_id}, {support_id}: no eligible {policy_class} policy")
            winner = eligible.iloc[0]
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "support_id": str(support_id),
                    "policy_class": policy_class,
                    "coordinate_id": str(winner["coordinate_id"]),
                    "phase_0_starcoder": float(winner["phase_0_starcoder"]),
                    "phase_1_starcoder": float(winner["phase_1_starcoder"]),
                    "discovery_bpb": float(winner["bpb"]),
                    "discovery_run_name": str(winner["run_name"]),
                    "discovery_is_alias": bool(winner["is_alias"]),
                    "eligible_policy_count": len(eligible),
                }
            )
    selected = pd.DataFrame(rows).sort_values(["cell_id", "support_id", "policy_class"]).reset_index(drop=True)
    if len(selected) != EXPECTED_SELECTED_POLICIES:
        raise ValueError(f"Expected {EXPECTED_SELECTED_POLICIES} selected policies, got {len(selected)}")
    if selected["discovery_is_alias"].any():
        raise ValueError("A selected empirical optimum is a deterministic support alias")
    class_counts = selected.groupby(["cell_id", "support_id"])["policy_class"].nunique()
    if len(class_counts) != EXPECTED_BLOCKS or not class_counts.eq(len(POLICY_CLASSES)).all():
        raise ValueError("Selected policies do not contain both policy classes in every block")
    return selected


def _run_rows(source_payload: dict[str, Any], selected: pd.DataFrame) -> list[dict[str, Any]]:
    cells = {row["cell_id"]: row for row in source_payload["cells"]}
    supports = {(row["cell_id"], row["support_id"]): row for row in source_payload["supports"]}
    coordinates = {row["coordinate_id"]: row for row in source_payload["coordinates"]}
    source_seeds = {int(row["data_seed"]) for row in source_payload["runs"]}
    if source_seeds & set(FRESH_SEEDS):
        raise ValueError("Fresh confirmation seeds overlap source design seeds")

    rows: list[dict[str, Any]] = []
    for selection in selected.to_dict("records"):
        cell = cells[selection["cell_id"]]
        support = supports[(selection["cell_id"], selection["support_id"])]
        coordinate = coordinates[selection["coordinate_id"]]
        for seed in FRESH_SEEDS:
            row = source_designer._run_row(
                cell=cell,
                support=support,
                coordinate=coordinate,
                seed=seed,
                replicate_kind="empirical_optimum_confirmation",
            )
            row.update(
                {
                    "run_name": (
                        f"dsopt_{cell['cell_slug']}_{support['support_id']}_{selection['policy_class']}_"
                        f"{coordinate['coordinate_id']}_s{str(seed)[-4:]}"
                    ),
                    "policy_role": f"selected_{selection['policy_class']}",
                    "policy_class": selection["policy_class"],
                    "pair_seed": seed,
                    "discovery_bpb": selection["discovery_bpb"],
                    "discovery_run_name": selection["discovery_run_name"],
                    "selection_rule": (
                        "minimum exact final-step coverage BPB within block and policy class; "
                        f"untied requires absolute contrast >= {MINIMUM_UNTIED_CONTRAST:.2f}"
                    ),
                }
            )
            rows.append(row)

    if len(rows) != EXPECTED_RUNS or len({row["run_name"] for row in rows}) != EXPECTED_RUNS:
        raise ValueError("Confirmation rows are incomplete or non-unique")
    for cell_id in cells:
        for support_id in {item["support_id"] for item in source_payload["supports"] if item["cell_id"] == cell_id}:
            block = [row for row in rows if row["cell_id"] == cell_id and row["support_id"] == support_id]
            if len(block) != len(FRESH_SEEDS) * len(POLICY_CLASSES):
                raise ValueError(f"{cell_id}, {support_id}: incomplete confirmation block")
            for seed in FRESH_SEEDS:
                pair = [row for row in block if row["pair_seed"] == seed]
                if {row["policy_class"] for row in pair} != set(POLICY_CLASSES):
                    raise ValueError(f"{cell_id}, {support_id}, {seed}: incomplete policy-class pair")
    finite = [row for row in rows if row["support_id"] != "full"]
    if not all(row["starcoder_support_wraps"] for row in finite):
        raise ValueError("A selected finite-support policy does not wrap; explicit alias handling is required")
    return rows


def build_payload() -> dict[str, Any]:
    source_payload = _source_payload()
    observations = _coverage_observations(source_payload)
    selected = _selected_policies(observations)
    rows = _run_rows(source_payload, selected)
    payload: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "description": (
            "Fresh-seed paired confirmation of the empirical tied and untied minima "
            "in all dense horizon-by-support blocks."
        ),
        "primary_metric": PRIMARY_METRIC,
        "source_design_sha256": source_payload["design_sha256"],
        "source_observations_sha256": _sha256(SOURCE_OBSERVATIONS_PATH),
        "discovery_seed": DISCOVERY_SEED,
        "fresh_seeds": list(FRESH_SEEDS),
        "minimum_untied_absolute_contrast": MINIMUM_UNTIED_CONTRAST,
        "block_count": EXPECTED_BLOCKS,
        "selected_policy_count": EXPECTED_SELECTED_POLICIES,
        "expected_run_count": EXPECTED_RUNS,
        "cells": source_payload["cells"],
        "supports": source_payload["supports"],
        "selected_policies": selected.to_dict("records"),
        "runs": rows,
        "analysis_contract": {
            "primary_estimand": "fresh-seed paired tied-minus-untied BPB within each cell-support block",
            "higher_gain_is_better": True,
            "fresh_outcomes_only": True,
            "discovery_observation_may_not_be_pooled": True,
            "selection_scope": "all 28 blocks regardless of discovery gain sign",
            "claim_scope": (
                "expected performance of the selected discrete grid policies, not continuous policy-class optima"
            ),
            "uncertainty": "paired-seed mean, t interval, sign count, and complete per-seed differences",
            "multiplicity": (
                "report all blocks; any blockwise significance claim must control family-wise error over 28 blocks"
            ),
        },
        "training_environment": source_payload["training_environment"],
        "runtime_cache_contract": source_payload["runtime_cache_contract"],
    }
    payload["design_sha256"] = _canonical_sha256(payload)
    return payload


def write_outputs() -> None:
    payload = build_payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(payload["selected_policies"]).to_csv(SELECTION_PATH, index=False)
    pd.DataFrame(payload["runs"]).to_csv(RUN_MANIFEST_PATH, index=False)

    selected = pd.DataFrame(payload["selected_policies"])
    minima = selected.pivot(
        index=["cell_id", "support_id"], columns="policy_class", values="discovery_bpb"
    ).reset_index()
    minima["discovery_gain_tied_minus_untied_bpb"] = minima["tied"] - minima["untied"]
    report = [
        "# Dense-support empirical-optimum confirmation design",
        "",
        f"Frozen adaptive design: `{DESIGN_VERSION}` (`{payload['design_sha256']}`).",
        "",
        f"- Blocks: `{EXPECTED_BLOCKS}` (4 token horizons x 7 StarCoder support regimes).",
        "- Policies per block: one empirical tied minimum and one empirical untied minimum.",
        f"- Untied eligibility: absolute phase contrast >= `{MINIMUM_UNTIED_CONTRAST:.2f}`.",
        f"- Fresh paired seeds: `{len(FRESH_SEEDS)}`; total runs: `{EXPECTED_RUNS}`.",
        "- Every block is confirmed regardless of the discovery gain sign.",
        "- Confirmation estimates use fresh outcomes only; the selecting coverage seed is never pooled.",
        "- This confirms selected discrete grid policies, not continuous policy-class optima.",
        "",
        "## Discovery selections",
        "",
        minima.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "design_path": str(DESIGN_PATH),
                "design_sha256": payload["design_sha256"],
                "run_count": len(payload["runs"]),
                "report_path": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    write_outputs()
