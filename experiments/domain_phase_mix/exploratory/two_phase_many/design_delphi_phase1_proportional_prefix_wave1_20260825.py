# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scipy==1.17.0",
# ]
# ///
"""Freeze Wave 1 of the Delphi proportional-prefix continuation search."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_harsh_cap_branches_20260825 as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_CANDIDATE_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
DEFAULT_SELECTED_PREFIXES = REFERENCE_OUTPUTS / "delphi_phase0_proportional_prefix_20260825" / "selected_prefixes.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825"
DEFAULT_FRONTIER_CONTRACT = DEFAULT_OUTPUT_DIR / "validated_frontier_contract.json"
TARGET_PREFIX = "proportional_control"
FRONTIER_BLEND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
FRONTIER_REPEAT_DATA_SEEDS = (971_000, 971_001, 971_002, 971_003)
EXTRA_TIED_DATA_SEED = 971_003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--selected-prefixes", type=Path, default=DEFAULT_SELECTED_PREFIXES)
    parser.add_argument("--frontier-contract", type=Path, default=DEFAULT_FRONTIER_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_frontier(path: Path, buckets: tuple[str, ...]) -> tuple[np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text())
    if payload.get("contract_version") != "delphi_validated_frontier_comparator_20260825_v1":
        raise ValueError("Validated frontier contract changed")
    counts_by_bucket = payload.get("runtime_counts")
    if not isinstance(counts_by_bucket, dict) or set(counts_by_bucket) != set(buckets):
        raise ValueError("Validated frontier bucket set changed")
    counts = np.asarray([int(counts_by_bucket[bucket]) for bucket in buckets], dtype=int)
    if np.any(counts < 0) or counts.sum() != base.MIXTURE_BLOCK_SIZE:
        raise ValueError("Validated frontier is not a runtime-exact mixture")
    confirmation = payload.get("confirmation")
    if not isinstance(confirmation, dict) or int(confirmation.get("paired_rows", 0)) != 9:
        raise ValueError("Validated frontier confirmation contract changed")
    return counts / base.MIXTURE_BLOCK_SIZE, payload


def set_continuation(
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    continuation_id: str,
    mixture: np.ndarray,
    *,
    source: str,
    center: np.ndarray,
    buckets: tuple[str, ...],
    phase0_exposure: np.ndarray,
    phase1_scale: np.ndarray,
) -> None:
    summary_mask = summary.continuation_id.eq(continuation_id)
    if int(summary_mask.sum()) != 1:
        raise ValueError(f"Expected one summary row for {continuation_id}")
    runtime = base.runtime_weights(mixture)
    phase1_exposure = runtime * phase1_scale
    if np.any(phase0_exposure + phase1_exposure > base.TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12):
        raise ValueError(f"Deployment anchor violates the epoch cap: {continuation_id}")
    summary.loc[summary_mask, "source"] = source
    summary.loc[summary_mask, "hellinger_to_tied"] = float(base.hellinger(runtime[None, :], center[None, :])[0])
    summary.loc[summary_mask, "max_phase0_materialized_epoch"] = float(phase0_exposure.max())
    summary.loc[summary_mask, "max_phase1_materialized_epoch"] = float(phase1_exposure.max())
    summary.loc[summary_mask, "max_total_materialized_epoch"] = float((phase0_exposure + phase1_exposure).max())

    weight_mask = weights.continuation_id.eq(continuation_id)
    group = weights.loc[weight_mask]
    if tuple(group.bucket) != buckets:
        raise ValueError(f"Bucket order changed for {continuation_id}")
    counts = base.common_design.runtime_counts(runtime)
    weights.loc[weight_mask, "source"] = source
    weights.loc[weight_mask, "phase_1_count"] = counts
    weights.loc[weight_mask, "phase_1_weight"] = runtime
    weights.loc[weight_mask, "phase_1_materialized_epochs"] = phase1_exposure
    weights.loc[weight_mask, "total_materialized_epochs"] = phase0_exposure + phase1_exposure


def append_continuation(
    summary: pd.DataFrame,
    weights: pd.DataFrame,
    continuation_id: str,
    mixture: np.ndarray,
    *,
    role: str,
    prefix_repeat_seed: int,
    data_seed: int,
    source: str,
    center: np.ndarray,
    buckets: tuple[str, ...],
    phase0_exposure: np.ndarray,
    phase1_scale: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = base.runtime_weights(mixture)
    counts = base.common_design.runtime_counts(runtime)
    phase1_exposure = runtime * phase1_scale
    total_exposure = phase0_exposure + phase1_exposure
    if float(total_exposure.max()) > base.TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12:
        raise ValueError(f"Repeat violates the epoch cap: {continuation_id}")
    summary_row = {
        "prefix_candidate_id": TARGET_PREFIX,
        "continuation_id": continuation_id,
        "role": role,
        "fit_budget": False,
        "prefix_repeat_seed": prefix_repeat_seed,
        "data_seed": data_seed,
        "source": source,
        "hellinger_to_tied": float(base.hellinger(runtime[None, :], center[None, :])[0]),
        "max_phase0_materialized_epoch": float(phase0_exposure.max()),
        "max_phase1_materialized_epoch": float(phase1_exposure.max()),
        "max_total_materialized_epoch": float(total_exposure.max()),
    }
    weight_rows = [
        {
            "prefix_candidate_id": TARGET_PREFIX,
            "continuation_id": continuation_id,
            "role": role,
            "fit_budget": False,
            "prefix_repeat_seed": prefix_repeat_seed,
            "data_seed": data_seed,
            "source": source,
            "bucket": bucket,
            "phase_1_count": int(count),
            "phase_1_weight": float(weight),
            "phase_1_materialized_epochs": float(phase1_epoch),
            "total_materialized_epochs": float(total_epoch),
        }
        for bucket, count, weight, phase1_epoch, total_epoch in zip(
            buckets, counts, runtime, phase1_exposure, total_exposure, strict=True
        )
    ]
    return (
        pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True),
        pd.concat([weights, pd.DataFrame(weight_rows)], ignore_index=True),
    )


def build_design(
    candidate_weights_path: Path,
    selected_prefixes_path: Path,
    frontier_contract_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary, weights, manifest = base.build_design(candidate_weights_path, selected_prefixes_path)
    panel = base.common_design.load_canonical_panel_geometry()
    center = base.candidate_centers(candidate_weights_path, (TARGET_PREFIX,), panel.buckets)[TARGET_PREFIX]
    phase0_exposure = center * panel.c0
    frontier, frontier_payload = validated_frontier(frontier_contract_path, panel.buckets)

    replacement_ids = sorted(summary.loc[summary.fit_budget, "continuation_id"])[-len(FRONTIER_BLEND_FRACTIONS) :]
    for continuation_id, fraction in zip(replacement_ids, FRONTIER_BLEND_FRACTIONS, strict=True):
        mixture = base.runtime_weights((1.0 - fraction) * center + fraction * frontier)
        set_continuation(
            summary,
            weights,
            continuation_id,
            mixture,
            source=f"deployment_anchor:validated_cap4_frontier:{fraction:.2f}",
            center=center,
            buckets=panel.buckets,
            phase0_exposure=phase0_exposure,
            phase1_scale=panel.c1,
        )

    exact_frontier_id = replacement_ids[-1]
    for position, data_seed in enumerate(FRONTIER_REPEAT_DATA_SEEDS):
        summary, weights = append_continuation(
            summary,
            weights,
            f"validated_frontier_repeat_{position}",
            frontier,
            role="validated_frontier_transfer_repeat",
            prefix_repeat_seed=0,
            data_seed=data_seed,
            source=f"repeat_of:{exact_frontier_id}",
            center=center,
            buckets=panel.buckets,
            phase0_exposure=phase0_exposure,
            phase1_scale=panel.c1,
        )
    for continuation_id, prefix_repeat_seed in (("tied_fresh_3", 0), ("tied_prefix_seed1_4", 1)):
        summary, weights = append_continuation(
            summary,
            weights,
            continuation_id,
            center,
            role="fresh_tied_control" if prefix_repeat_seed == 0 else "prefix_state_tied_control",
            prefix_repeat_seed=prefix_repeat_seed,
            data_seed=EXTRA_TIED_DATA_SEED,
            source="tied",
            center=center,
            buckets=panel.buckets,
            phase0_exposure=phase0_exposure,
            phase1_scale=panel.c1,
        )

    fit_weights = np.stack(
        [
            weights.loc[weights.continuation_id.eq(continuation_id), "phase_1_weight"].to_numpy(dtype=float)
            for continuation_id in summary.loc[summary.fit_budget, "continuation_id"]
        ]
    )
    diagnostics_by_prefix = cast(dict[str, dict[str, object]], manifest["diagnostics"])
    diagnostics = diagnostics_by_prefix[TARGET_PREFIX]
    fit_summary = summary.loc[summary.fit_budget]
    diagnostics["sqrt_feature_rank"] = base.rank(base.feature_matrix(fit_weights, center, "sqrt"))
    diagnostics["direct_feature_rank"] = base.rank(base.feature_matrix(fit_weights, center, "direct"))
    diagnostics["selected_hellinger_min"] = float(fit_summary.hellinger_to_tied.min())
    diagnostics["selected_hellinger_median"] = float(fit_summary.hellinger_to_tied.median())
    diagnostics["selected_hellinger_max"] = float(fit_summary.hellinger_to_tied.max())
    diagnostics["selected_anchor_rows"] = int(fit_summary.source.str.startswith(("anchor:", "deployment_anchor:")).sum())
    diagnostics["selected_local_rows"] = int((fit_summary.hellinger_to_tied <= base.LOCAL_HELLINGER_MAX).sum())
    diagnostics["selected_mid_rows"] = int(
        (
            (fit_summary.hellinger_to_tied > base.LOCAL_HELLINGER_MAX)
            & (fit_summary.hellinger_to_tied <= base.MID_HELLINGER_MAX)
        ).sum()
    )
    diagnostics["selected_far_deployment_rows"] = int((fit_summary.hellinger_to_tied > base.MID_HELLINGER_MAX).sum())
    diagnostics["validated_frontier_exact_continuation_id"] = exact_frontier_id
    if (diagnostics["sqrt_feature_rank"], diagnostics["direct_feature_rank"]) != (39, 38):
        raise ValueError(f"Deployment-anchor replacement broke design rank: {diagnostics}")

    role_counts = summary.role.value_counts().to_dict()
    if len(summary) != 102 or int(summary.fit_budget.sum()) != base.FIT_ROWS_PER_PREFIX:
        raise ValueError("Proportional Wave-1 row allocation changed")
    if summary.continuation_id.nunique() != len(summary):
        raise ValueError("Proportional Wave-1 continuation identities are not unique")
    manifest["contract_version"] = "delphi_phase1_proportional_prefix_wave1_20260825_v2"
    manifest["rows"] = {
        "fit_per_prefix": base.FIT_ROWS_PER_PREFIX,
        "sealed_referees_per_prefix": base.REFEREE_ROWS_PER_PREFIX,
        "controls_per_prefix": len(summary) - base.FIT_ROWS_PER_PREFIX - base.REFEREE_ROWS_PER_PREFIX,
        "total": len(summary),
    }
    manifest["role_counts_per_prefix"] = role_counts
    manifest["research_question"] = (
        "Can phase-1 optimization from an exact proportional phase-0 state match or beat the validated "
        "two-phase frontier?"
    )
    confirmation = cast(dict[str, object], frontier_payload["confirmation"])
    manifest["comparators"] = {
        "primary": "exact proportional tied continuation under common random numbers",
        "validated_frontier_mean_uncheatable_bpb": confirmation["candidate_mean_uncheatable_bpb"],
        "validated_frontier_continuation_id": frontier_payload["continuation_id"],
        "claim_scope": (
            "Wave 1 is a v5p-prefix to v6e-continuation discovery panel; any promoted frontier requires "
            "same-hardware confirmation before a canonical performance claim"
        ),
    }
    design = cast(dict[str, object], manifest["design"])
    fit_anchors = cast(list[str], design["fit_anchors"])
    design["fit_anchors"] = [*fit_anchors, "validated_cap4_frontier"]
    design["validated_frontier_blend_fractions"] = list(FRONTIER_BLEND_FRACTIONS)
    design["validated_frontier_exact_continuation_id"] = exact_frontier_id
    design["validated_frontier_repeat_data_seeds"] = list(FRONTIER_REPEAT_DATA_SEEDS)
    design["outcome_selected_anchor_used_as_geometry_repeller"] = False
    design["noise_groups"] = {
        "proportional_tied_prefix_seed0": 5,
        "proportional_tied_prefix_seed1": 5,
        "validated_frontier_from_proportional_prefix_seed0": 5,
    }
    provenance = cast(dict[str, object], manifest["provenance"])
    provenance["validated_frontier_contract_sha256"] = file_sha256(frontier_contract_path)
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    summary, weights, manifest = build_design(
        args.candidate_weights,
        args.selected_prefixes,
        args.frontier_contract,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    summary.to_csv(summary_path, index=False)
    weights.loc[:, list(base.WEIGHT_ARTIFACT_COLUMNS)].to_csv(weights_path, index=False)
    payload = {
        **manifest,
        "artifacts": {
            summary_path.name: file_sha256(summary_path),
            weights_path.name: file_sha256(weights_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
