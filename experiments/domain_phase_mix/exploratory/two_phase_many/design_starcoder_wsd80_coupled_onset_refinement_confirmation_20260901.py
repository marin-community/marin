# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///

"""Freeze discovery-only BO refinement and independent fresh confirmation."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    design_starcoder_wsd80_coupled_onset_dense_surfaces_20260830 as source,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    design_starcoder_wsd80_lr_onset_dense_surfaces_20260825 as lr_design,
)

DOMAIN_PHASE_MIX_DIR = SCRIPT_DIR.parents[1]
OBSERVATIONS_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "starcoder_wsd80_coupled_onset_dense_surface_results_20260901"
    / "observations.csv"
)
OUTPUT_PATH = DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_coupled_onset_refinement_confirmation_design_20260901.json.gz"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_coupled_onset_refinement_confirmation_20260901"
DESIGN_VERSION = "2026-09-01-coupled-refinement-confirmation-v1"
DISCOVERY_SEED = source.DISCOVERY_SEED
CONFIRMATION_SEEDS = source.CONFIRMATION_SEEDS
NOISE_ANCHOR_BPB = 0.001182
LOCAL_BASIN_MARGIN_BPB = 0.008
MINIMUM_LOCAL_POINTS = 12
GRID_STEP = 0.005
EXISTING_EXCLUSION_RADIUS = 0.012
BATCH_EXCLUSION_RADIUS = 0.025
ACQUISITIONS_PER_ARM = 8
KERNEL_LENGTH_SCALES = ((0.04, 0.04), (0.08, 0.08), (0.16, 0.08))
ARM_POLICIES = {
    "coupled_0p60": ("c096", "c042", "c109", "c016"),
    "coupled_0p80": ("c109", "c016"),
    "coupled_0p90": ("c109", "c067", "c016"),
}


def _expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    safe = np.maximum(sd, 1e-12)
    z = (best - mean) / safe
    return (best - mean) * norm.cdf(z) + safe * norm.pdf(z)


def _candidate_grid(points: np.ndarray) -> np.ndarray:
    p0 = np.arange(0.0, 1.0 + GRID_STEP / 2.0, GRID_STEP)
    p1 = np.arange(0.0, 1.0 + GRID_STEP / 2.0, GRID_STEP)
    x0, x1 = np.meshgrid(p0, p1, indexing="ij")
    grid = np.column_stack([x0.ravel(), x1.ravel()])
    inside = Delaunay(points).find_simplex(grid) >= 0
    eligible = np.abs(grid[:, 1] - grid[:, 0]) >= lr_design.MINIMUM_UNTIED_ABSOLUTE_CONTRAST
    return grid[inside & eligible]


def _minimum_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((points[:, None, :] - reference[None, :, :]) ** 2, axis=2)).min(axis=1)


def acquire(arm: pd.DataFrame) -> tuple[list[dict[str, float]], dict[str, float]]:
    eligible = arm[arm.selection_class.eq("eligible_untied")].copy()
    all_points = eligible[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(float)
    all_target = eligible.programming_languages_bpb.to_numpy(float)
    local = eligible[eligible.programming_languages_bpb <= all_target.min() + LOCAL_BASIN_MARGIN_BPB]
    if len(local) < MINIMUM_LOCAL_POINTS:
        local = eligible.nsmallest(MINIMUM_LOCAL_POINTS, "programming_languages_bpb")
    local_points = local[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(float)
    local_target = local.programming_languages_bpb.to_numpy(float)
    candidates = _candidate_grid(local_points)
    candidates = candidates[_minimum_distance(candidates, all_points) >= EXISTING_EXCLUSION_RADIUS]
    means = []
    variances = []
    target_scale = float(local_target.std(ddof=1))
    if target_scale <= 0.0:
        raise ValueError("Bayesian acquisition requires nonzero endpoint variation")
    for length_scale in KERNEL_LENGTH_SCALES:
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
            * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5),
            alpha=(NOISE_ANCHOR_BPB / target_scale) ** 2,
            normalize_y=True,
            optimizer=None,
        )
        model.fit(local_points, local_target)
        mean, sd = model.predict(candidates, return_std=True)
        means.append(mean)
        variances.append(sd**2)
    committee_means = np.stack(means)
    mean = committee_means.mean(axis=0)
    variance = np.stack(variances).mean(axis=0) + committee_means.var(axis=0)
    sd = np.sqrt(np.maximum(variance, 0.0))
    acquisition = _expected_improvement(mean, sd, float(all_target.min()))
    selected: list[dict[str, float]] = []
    available = np.ones(len(candidates), dtype=bool)
    for order in range(ACQUISITIONS_PER_ARM):
        if not available.any():
            raise ValueError("Bayesian acquisition exhausted its candidate grid")
        index = int(np.argmax(np.where(available, acquisition, -np.inf)))
        point = candidates[index]
        selected.append(
            {
                "acquisition_order": order,
                "phase_0_starcoder": float(point[0]),
                "phase_1_starcoder": float(point[1]),
                "predicted_mean_bpb": float(mean[index]),
                "predicted_sd_bpb": float(sd[index]),
                "expected_improvement": float(acquisition[index]),
            }
        )
        available &= np.linalg.norm(candidates - point[None, :], axis=1) >= BATCH_EXCLUSION_RADIUS
    return selected, {
        "eligible_observations": float(len(eligible)),
        "local_basin_observations": float(len(local)),
        "candidate_grid_size": float(len(candidates)),
        "incumbent_bpb": float(all_target.min()),
        "local_basin_margin_bpb": LOCAL_BASIN_MARGIN_BPB,
    }


def _row_from_policy(
    template: dict[str, Any],
    *,
    stage: str,
    coordinate_id: str,
    phase_0: float,
    phase_1: float,
    seed: int,
    run_order: int,
    acquisition: dict[str, float] | None = None,
) -> dict[str, Any]:
    fraction = float(template["realized_onset_fraction"])
    aggregate = fraction * phase_0 + (1.0 - fraction) * phase_1
    contrast = phase_1 - phase_0
    if abs(contrast) <= 1e-15:
        normalized = 0.0
    else:
        limit = source._maximum_contrast(aggregate, fraction, 1 if contrast > 0.0 else -1)
        normalized = contrast / limit
    cell = {
        **source.lr_only_design._cell(source.lr_only_design._load_source_design()),
        "boundary_step": int(template["boundary_step"]),
        "realized_phase_0_fraction": fraction,
    }
    support = source.lr_only_design._supports(source.lr_only_design._load_source_design())[source.PRIMARY_SUPPORT_ID]
    phase_0_sequences, phase_1_sequences = source.source_design._realized_starcoder_sequences(
        cell=cell,
        phase_0=phase_0,
        phase_1=phase_1,
        data_seed=seed,
    )
    support_tokens = int(support["starcoder_realized_support_tokens"])
    onset_slug = str(template["arm_id"]).removeprefix("coupled_")
    stage_slug = "bo" if stage == "bayesian_refinement_discovery" else "cf"
    identity = {
        "design_version": DESIGN_VERSION,
        "stage": stage,
        "arm_id": template["arm_id"],
        "coordinate_id": coordinate_id,
        "seed": seed,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
    }
    row = {
        **template,
        "row_id": f"coupled_onset_successor_{lr_design.canonical_sha256(identity)[:24]}",
        "run_order": run_order,
        "run_name": f"lrcd_{stage_slug}_m100_{onset_slug}_{coordinate_id}_s{seed % 10000:04d}",
        "stage": stage,
        "coordinate_id": coordinate_id,
        "coordinate_sources": [stage],
        "selection_class": (
            "tied"
            if abs(contrast) <= 1e-12
            else (
                "eligible_untied"
                if abs(contrast) >= lr_design.MINIMUM_UNTIED_ABSOLUTE_CONTRAST
                else "ineligible_near_tied"
            )
        ),
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder": aggregate,
        "phase_contrast": contrast,
        "normalized_fiber_position": normalized,
        "starcoder_phase_0_sequences": phase_0_sequences,
        "starcoder_phase_1_sequences": phase_1_sequences,
        "starcoder_total_sequences": phase_0_sequences + phase_1_sequences,
        "starcoder_phase_0_epochs": phase_0_sequences * source.base.SEQ_LEN / support_tokens,
        "starcoder_phase_1_epochs": phase_1_sequences * source.base.SEQ_LEN / support_tokens,
        "starcoder_support_wraps": (phase_0_sequences + phase_1_sequences) * source.base.SEQ_LEN > support_tokens,
        "data_seed": seed,
        "trainer_seed": seed,
    }
    if acquisition is not None:
        row["acquisition"] = acquisition
    return row


def build_payload() -> dict[str, Any]:
    source_payload = json.loads(gzip.decompress(source.OUTPUT_PATH.read_bytes()))
    observations = pd.read_csv(OBSERVATIONS_PATH)
    if len(observations) != 375 or observations.row_id.nunique() != 375:
        raise ValueError("Refinement requires the complete 375-row discovery surface")
    source_rows = {(row["arm_id"], row["coordinate_id"]): row for row in source_payload["runs"]}
    rows: list[dict[str, Any]] = []
    acquisitions: dict[str, Any] = {}
    order = 0
    for arm_id in sorted(ARM_POLICIES):
        selected, summary = acquire(observations[observations.arm_id.eq(arm_id)])
        acquisitions[arm_id] = {"summary": summary, "selected": selected}
        template = source_rows[(arm_id, "c016")]
        for item in selected:
            coordinate_id = f"bo_{arm_id.removeprefix('coupled_')}_{int(item['acquisition_order']):02d}"
            rows.append(
                _row_from_policy(
                    template,
                    stage="bayesian_refinement_discovery",
                    coordinate_id=coordinate_id,
                    phase_0=float(item["phase_0_starcoder"]),
                    phase_1=float(item["phase_1_starcoder"]),
                    seed=DISCOVERY_SEED,
                    run_order=order,
                    acquisition=item,
                )
            )
            order += 1
    for arm_id, policies in ARM_POLICIES.items():
        for seed in CONFIRMATION_SEEDS:
            for coordinate_id in policies:
                template = source_rows[(arm_id, coordinate_id)]
                rows.append(
                    _row_from_policy(
                        template,
                        stage="fresh_confirmation",
                        coordinate_id=coordinate_id,
                        phase_0=float(template["phase_0_starcoder"]),
                        phase_1=float(template["phase_1_starcoder"]),
                        seed=seed,
                        run_order=order,
                    )
                )
                order += 1
    if len(rows) != 96 or len({row["row_id"] for row in rows}) != 96 or len({row["run_name"] for row in rows}) != 96:
        raise ValueError("Successor inventory must contain 24 BO rows and 72 unique confirmation rows")
    stage_counts = pd.Series([row["stage"] for row in rows]).value_counts().to_dict()
    if stage_counts != {"fresh_confirmation": 72, "bayesian_refinement_discovery": 24}:
        raise ValueError(f"Unexpected successor stage counts: {stage_counts}")
    source_hash = lr_design.file_sha256(source.OUTPUT_PATH)
    observation_hash = lr_design.file_sha256(OBSERVATIONS_PATH)
    return {
        "design_version": DESIGN_VERSION,
        "description": "Discovery-only untied BO refinement plus an independent fixed 72-run fresh-seed confirmation.",
        "source_design_sha256": source_payload["design_sha256"],
        "source_design_file_sha256": source_hash,
        "source_observations_sha256": observation_hash,
        "training_environment": source_payload["training_environment"],
        "runtime_cache_contract": source_payload["runtime_cache_contract"],
        "source_placement": source_payload["source_placement"],
        "cell": source_payload["cell"],
        "support": source_payload["support"],
        "arms": source_payload["arms"],
        "discovery_seed": DISCOVERY_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "stage_counts": stage_counts,
        "expected_run_count": len(rows),
        "bayesian_refinement": {
            "role": "discovery-only under-sampling falsifier; excluded from every confirmation statistic and cell",
            "target": source.PRIMARY_METRIC,
            "eligible_surface": "eligible untied only",
            "candidate_region": "convex hull of eligible observations within 0.008 BPB of the arm incumbent",
            "acquisitions_per_arm": ACQUISITIONS_PER_ARM,
            "grid_step": GRID_STEP,
            "existing_exclusion_radius": EXISTING_EXCLUSION_RADIUS,
            "batch_exclusion_radius": BATCH_EXCLUSION_RADIUS,
            "kernel_length_scales": [list(value) for value in KERNEL_LENGTH_SCALES],
            "noise_anchor_bpb": NOISE_ANCHOR_BPB,
            "falsifier": (
                "If a 0.60T acquisition beats c042 by more than 0.001182 BPB, report material 0.60T under-sampling; "
                "the frozen confirmation remains unchanged."
            ),
            "arms": acquisitions,
        },
        "confirmation_contract": {
            "provenance": "Post-hoc hypothesis tested on eight previously reserved fresh seeds.",
            "E1_primary": {
                "estimand": "per-arm discovery argmin tied minus discovery argmin eligible untied",
                "pairs": {
                    "coupled_0p60": ["c096", "c042"],
                    "coupled_0p80": ["c109", "c016"],
                    "coupled_0p90": ["c109", "c067"],
                },
                "test": (
                    "For each reserved seed s and arm a, gain[a,s] = tied_bpb[a,s] - untied_bpb[a,s]. "
                    "Form paired cross-arm differences gain[0.80,s] - gain[0.60,s] and "
                    "gain[0.90,s] - gain[0.60,s]. Apply separate one-sample t tests to the eight paired "
                    "differences, each against mean <= 0 at one-sided alpha 0.05. The intersection-union "
                    "claim passes only if both one-sided lower confidence bounds exceed zero."
                ),
            },
            "E2_secondary": "fixed c109 tied versus c016 untied at all three onsets",
            "E3_secondary": "fixed c109 tied versus each arm's selected untied policy",
            "common_random_numbers": "Within each arm and seed, policies share data and trainer seeds.",
            "power_check": "Observed per-seed gain SD must be no more than 2 * 0.001182 BPB or power claims are void.",
            "C4": "secondary descriptive endpoint only; excluded from policy selection and confirmation family",
            "policies_by_arm": {key: list(value) for key, value in ARM_POLICIES.items()},
            "selection_caveat": (
                "E1 fixes discovery-selected arm-specific cells and removes evaluation noise at those cells, but it "
                "does not remove discovery selection bias. A passing E1 with a failing E3 is an optimum-location "
                "result, not evidence for a general onset mechanism. E2 transports a 0.80T-selected untied policy."
            ),
        },
        "metrics": source_payload["metrics"],
        "completeness_contract": {
            "required": "all 24 adaptive rows and all 72 fixed confirmation rows",
            "valid_endpoint": source_payload["completeness_contract"]["valid_endpoint"],
            "failure_rule": "retry the exact frozen identity; never drop, replace, or reselect after a failed row",
            "analysis_order": (
                "The BO under-sampling audit is descriptive. E1-E3 are computed only after all 72 fresh "
                "confirmation endpoints are complete and valid."
            ),
        },
        "checkpoint_contract": {
            "all_rows": "terminal permanent checkpoint only",
            "temporary_recovery": source_payload["checkpoint_contract"]["temporary_recovery"],
        },
        "rows": rows,
    }


def write_outputs() -> dict[str, Any]:
    payload = build_payload()
    payload["design_sha256"] = lr_design.canonical_sha256(payload)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    OUTPUT_PATH.write_bytes(gzip.compress(serialized, mtime=0))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload["rows"]).to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)
    acquisition_rows = []
    for arm_id, arm in payload["bayesian_refinement"]["arms"].items():
        for row in arm["selected"]:
            acquisition_rows.append({"arm_id": arm_id, **row})
    pd.DataFrame(acquisition_rows).to_csv(OUTPUT_DIR / "acquisitions.csv", index=False)
    report = [
        "# StarCoder coupled-onset refinement and confirmation design",
        "",
        "- 24 discovery-only BO rows: eight eligible untied acquisitions in each arm.",
        "- 72 fresh confirmation rows: nine arm-policy cells by eight reserved seeds.",
        "- BO outcomes cannot alter the confirmation inventory or statistic.",
        "- E1 tests arm-specific achievable optima; E2 transports one fixed pair; E3 holds the tied comparator fixed.",
        "- C4 is descriptive and cannot select a policy.",
        "",
        "## Acquisitions",
        "",
        pd.DataFrame(acquisition_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report))
    return payload


def main() -> None:
    payload = write_outputs()
    print(f"Wrote {OUTPUT_PATH} ({len(payload['rows'])} rows, {payload['design_sha256']})")


if __name__ == "__main__":
    main()
