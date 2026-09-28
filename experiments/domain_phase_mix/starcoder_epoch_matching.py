# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen finite-pool experiment comparing unmatched and epoch-matched proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path(__file__).with_name("starcoder_epoch_matching_design_20260908.json")
SOURCE_DESIGN = Path(__file__).with_name("starcoder_wsd80_dense_support_surface_design_20260808.json")
SOURCE_OBSERVATIONS = Path(__file__).parent / (
    "exploratory/two_phase_many/reference_outputs/" "starcoder_all_tied_curves_canonical_dsp_20260902/predictions.csv"
)
DESIGN_VERSION = "2026-09-08-v2"
SEQ_LEN = 2048
BATCH_SIZE = 128
BLOCK_SIZE = 2048
PROXY_STEPS = 1060
TARGET_STEPS = 28260
PARENT_BATCHES = 1068
MATCHED_BATCHES = 40
REFERENCE_SEED = 20260711
TRAINER_SEEDS = (REFERENCE_SEED, 20260908, 20260909)
PILOT_WEIGHTS = (0.0, 0.1, 0.3, 0.7, 1.0)
REFINEMENT_WEIGHTS = (0.2, 0.4, 0.5, 0.6, 0.9)
STAGES = ("pilot", "refinement", "primary", "replicated")
TRAINING_FLOPS_PER_TOKEN = 878499840
METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
COMPONENT_SHUFFLE_KEYS = {
    "nemotron_cc/hq_actual-llama3": (4181282030, 3509741613),
    "nemotron_cc/hq_synth-llama3": (1692708717, 814235673),
    "nemotron_cc/medium_high-llama3": (838510724, 1386979079),
    "nemotron_cc/medium-llama3": (3430063960, 3470476155),
    "nemotron_cc/medium_low-llama3": (34974657, 2948256024),
    "nemotron_cc/low_actual-llama3": (1427242644, 385010521),
    "dolma/starcoder": (898005854, 446240491),
}


@dataclass(frozen=True)
class RunSpec:
    """A newly trained policy; first_stage controls cumulative batch release."""

    run_name: str
    arm: str
    starcoder_weight: float
    trainer_seed: int
    data_seed: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    starcoder_support_batches: int
    first_stage: str
    coordinate_id: str


@dataclass(frozen=True)
class TargetObservation:
    """Archived endpoint loss, with explicit eligibility for the fixed-parent target."""

    starcoder_weight: float
    observed_bpb: float
    source_run_name: str
    source_observation_id: str
    reusable: bool
    reason: str


@dataclass(frozen=True)
class ExperimentDesign:
    """Persisted experiment contract; its digest excludes only design_sha256 itself."""

    design_version: str
    primary_metric: str
    design_sha256: str
    runs: tuple[RunSpec, ...]
    component_shuffle_keys: dict[str, tuple[int, int]]
    component_order: tuple[str, ...]
    training_environment: dict[str, str | bool]
    source_sha256: dict[str, str]
    target_observations: tuple[TargetObservation, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_sha256(payload: dict) -> str:
    """Hash JSON independent of whitespace and mapping insertion order."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_runs(design: ExperimentDesign, stage: str) -> tuple[RunSpec, ...]:
    """Select a cumulative stage, including earlier runs for safe resubmission."""
    if stage not in STAGES:
        raise ValueError(f"Unknown experiment stage: {stage}")
    if stage == "refinement":
        return tuple(
            run
            for run in design.runs
            if run.first_stage == "pilot"
            or (
                run.arm in ("unmatched", "matched")
                and run.trainer_seed == REFERENCE_SEED
                and run.starcoder_weight in REFINEMENT_WEIGHTS
            )
        )
    return tuple(run for run in design.runs if STAGES.index(run.first_stage) <= STAGES.index(stage))


def build_design() -> ExperimentDesign:
    """Construct a manifest from archived C40 outcomes and the frozen source design."""
    source = json.loads(SOURCE_DESIGN.read_text())
    claimed_hash = source.pop("design_sha256")
    if canonical_sha256(source) != claimed_hash:
        raise ValueError("Historical design checksum mismatch")
    historical = {row["run_name"]: row for row in source["runs"]}
    observations = []
    coordinates = []
    with SOURCE_OBSERVATIONS.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["curve_ref"] == "C40"]
    rows.sort(key=lambda row: float(row["starcoder_weight"]))
    if len(rows) != 26:
        raise ValueError("Expected all 26 C40 target coordinates")
    for row in rows:
        p = float(row["starcoder_weight"])
        source_name = row["training_run_id"].removeprefix("run:")
        run = historical[source_name]
        if run["data_seed"] != REFERENCE_SEED or run["total_steps"] != TARGET_STEPS:
            raise ValueError(f"Target seed or horizon changed: {source_name}")
        if abs(run["phase_0_starcoder"] - p) > 1e-14 or abs(run["phase_1_starcoder"] - p) > 1e-14:
            raise ValueError(f"Target policy is not tied at the expected weight: {source_name}")
        if not row["observation_id"].endswith("@endpoint"):
            raise ValueError(f"Target measurement is not an endpoint: {source_name}")
        if run["support_id"] != "m100":
            if run["support_id"] != "full" or run["starcoder_total_sequences"] > PARENT_BATCHES * BATCH_SIZE:
                raise ValueError(f"Full-pool alias exceeds the benchmark parent: {source_name}")
        elif run["starcoder_support_batches"] != PARENT_BATCHES:
            raise ValueError(f"Target parent cap changed: {source_name}")
        reusable = p < 1.0
        observations.append(
            TargetObservation(
                p,
                float(row["observed_bpb"]),
                source_name,
                row["observation_id"],
                reusable,
                (
                    "same ordered parent, or a prefix-equivalent no-wrap alias"
                    if reusable
                    else "excluded: dropping zero-weight web components changed the StarCoder shuffle key"
                ),
            )
        )
        coordinates.append((run["coordinate_id"], p))

    runs = []
    for seed in TRAINER_SEEDS:
        for coordinate, p in coordinates:
            for arm, cap in (("unmatched", PARENT_BATCHES), ("matched", MATCHED_BATCHES)):
                if arm == "matched" and p == 0:
                    continue  # Exactly the same web-only stream; analysis aliases the unmatched run.
                stage = "replicated" if seed != REFERENCE_SEED else "pilot" if p in PILOT_WEIGHTS else "primary"
                runs.append(
                    RunSpec(
                        f"epmatch_{arm}_{coordinate}_s{seed}",
                        arm,
                        p,
                        seed,
                        REFERENCE_SEED,
                        PROXY_STEPS,
                        848,
                        PROXY_STEPS * BATCH_SIZE * SEQ_LEN,
                        cap,
                        stage,
                        coordinate,
                    )
                )
    runs.append(
        RunSpec(
            "epmatch_target_c124_s20260711",
            "target",
            1.0,
            REFERENCE_SEED,
            REFERENCE_SEED,
            TARGET_STEPS,
            22608,
            TARGET_STEPS * BATCH_SIZE * SEQ_LEN,
            PARENT_BATCHES,
            "pilot",
            "c124",
        )
    )
    design = ExperimentDesign(
        DESIGN_VERSION,
        METRIC,
        "",
        tuple(runs),
        COMPONENT_SHUFFLE_KEYS,
        tuple(COMPONENT_SHUFFLE_KEYS),
        {
            "jax_version": "0.11.1",
            "numpy_version": "2.3.5",
            "jax_default_prng_impl": "threefry2x32",
            "jax_enable_x64": False,
        },
        {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in (SOURCE_DESIGN, SOURCE_OBSERVATIONS)},
        tuple(observations),
    )
    payload = design.to_dict()
    payload.pop("design_sha256")
    payload["design_sha256"] = canonical_sha256(payload)
    return _decode_design(payload)


def _decode_design(payload: dict) -> ExperimentDesign:
    actual_hash = payload["design_sha256"]
    unsigned = {key: value for key, value in payload.items() if key != "design_sha256"}
    if canonical_sha256(unsigned) != actual_hash:
        raise ValueError("Experiment design checksum mismatch")
    design = ExperimentDesign(
        payload["design_version"],
        payload["primary_metric"],
        actual_hash,
        tuple(RunSpec(**row) for row in payload["runs"]),
        {name: tuple(key) for name, key in payload["component_shuffle_keys"].items()},
        tuple(payload["component_order"]),
        payload["training_environment"],
        payload["source_sha256"],
        tuple(TargetObservation(**row) for row in payload["target_observations"]),
    )
    if len({run.run_name for run in design.runs}) != len(design.runs):
        raise ValueError("Duplicate planned run identities")
    for run in design.runs:
        if run.materialized_tokens != run.total_steps * BATCH_SIZE * SEQ_LEN:
            raise ValueError(f"Incorrect token accounting: {run.run_name}")
        if run.boundary_step * 5 != run.total_steps * 4 or run.boundary_step * BATCH_SIZE % BLOCK_SIZE:
            raise ValueError(f"Unaligned WSD80 boundary: {run.run_name}")
        if run.arm == "unmatched" and run.total_steps > run.starcoder_support_batches:
            raise ValueError(f"Unmatched proxy can repeat the parent: {run.run_name}")
    return design


def load_design(path: Path = DESIGN_PATH) -> ExperimentDesign:
    """Load and checksum-validate the self-contained frozen manifest without remote I/O."""
    return _decode_design(json.loads(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESIGN_PATH)
    args = parser.parse_args()
    design = build_design()
    encoded = json.dumps(design.to_dict(), indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != encoded:
        raise ValueError("Existing frozen manifest differs; review and version the experiment instead of overwriting")
    args.output.write_text(encoded)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "design_sha256": design.design_sha256,
                "stage_runs": {stage: len(select_runs(design, stage)) for stage in STAGES},
            }
        )
    )


if __name__ == "__main__":
    main()
