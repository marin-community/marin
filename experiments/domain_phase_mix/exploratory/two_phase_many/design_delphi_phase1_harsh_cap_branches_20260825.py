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
"""Freeze support-safe branch coverage for selected harsh-cap Delphi prefixes.

The design identifies each fixed-prefix continuation response separately. It
uses full-rank one-sided coverage when a selected prefix lies on the simplex
boundary, rather than requiring antithetic log-ratio rays that cannot add a
bucket whose tied weight is zero.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import fsspec
import numpy as np
import pandas as pd
from scipy import linalg

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as common_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_CANDIDATE_WEIGHTS = (
    REFERENCE_OUTPUTS / "delphi_phase0_harsh_cap_candidates_20260825" / "training_candidate_weights.csv"
)
DEFAULT_SELECTED_PREFIXES = REFERENCE_OUTPUTS / "delphi_phase0_harsh_cap_validation_20260825" / "selected_prefixes.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_harsh_cap_branches_20260825"
HISTORICAL_FRONTIER_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_validation_20260712/mixtures/dphase_unch05_eff_e0p005.csv"
)
HISTORICAL_FRONTIER_SHA256 = "57a2aa39a5b0e07d40fc6f55f14aaa86327c332e9ef86738b1cca547924c4a59"
DESIGN_SEED = 20_260_825
FIT_ROWS_PER_PREFIX = 80
REFEREE_ROWS_PER_PREFIX = 8
CONTROL_ROWS_PER_PREFIX = 8
ANCHOR_ROWS = 12
LOCAL_ROWS = 48
MID_ROWS = FIT_ROWS_PER_PREFIX - ANCHOR_ROWS - LOCAL_ROWS
LOCAL_HELLINGER_MAX = 0.12
MID_HELLINGER_MAX = 0.30
MINIMUM_HELLINGER = 0.01
TOTAL_MATERIALIZED_EPOCH_CAP = 10.0
DIRICHLET_DRAWS_PER_SCALE = 5_000
TRANSFER_DRAWS = 20_000
FIT_DATA_SEED = 970_000
FRESH_TIED_DATA_SEEDS = (971_000, 971_001, 971_002)
MIXTURE_BLOCK_SIZE = common_design.MIXTURE_BLOCK_SIZE
WEIGHT_ARTIFACT_COLUMNS = (
    "prefix_candidate_id",
    "continuation_id",
    "bucket",
    "phase_1_count",
    "phase_1_weight",
    "phase_1_materialized_epochs",
    "total_materialized_epochs",
)


@dataclass(frozen=True)
class CandidatePoint:
    counts: tuple[int, ...]
    source: str

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.counts, dtype=float) / MIXTURE_BLOCK_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--selected-prefixes", type=Path, default=DEFAULT_SELECTED_PREFIXES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def runtime_weights(weights: np.ndarray) -> np.ndarray:
    return common_design.runtime_counts(weights) / MIXTURE_BLOCK_SIZE


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.sqrt(np.clip(left, 0.0, None)) - np.sqrt(np.clip(right, 0.0, None)), axis=-1) / np.sqrt(2.0)


def selected_candidate_ids(payload: dict[str, object]) -> tuple[str, ...]:
    aliases = payload.get("selected_aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("Selected-prefix manifest has no selected aliases")
    candidate_ids = tuple(str(row["canonical_candidate_id"]) for row in aliases)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Selected-prefix manifest repeats a canonical candidate")
    return candidate_ids


def candidate_centers(path: Path, selected: tuple[str, ...], buckets: tuple[str, ...]) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    result: dict[str, np.ndarray] = {}
    for candidate_id in selected:
        rows = frame[frame.candidate_id.eq(candidate_id)]
        if tuple(rows.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {candidate_id}")
        counts = rows.phase_0_count.to_numpy(dtype=int)
        if counts.sum() != MIXTURE_BLOCK_SIZE or np.any(counts < 0):
            raise ValueError(f"Invalid runtime counts for {candidate_id}")
        result[candidate_id] = cast(np.ndarray, counts / MIXTURE_BLOCK_SIZE)
    return result


def anchor_mixtures(buckets: tuple[str, ...], proportional: np.ndarray) -> dict[str, np.ndarray]:
    payload = read_uri_bytes(HISTORICAL_FRONTIER_URI)
    if hashlib.sha256(payload).hexdigest() != HISTORICAL_FRONTIER_SHA256:
        raise ValueError("Historical frontier mixture changed")
    frontier = pd.read_csv(io.BytesIO(payload)).set_index("domain")
    if set(frontier.index) != set(buckets):
        raise ValueError("Historical frontier bucket set changed")
    anchors = {
        "historical_frontier": frontier.loc[list(buckets), "phase_1_weight"].to_numpy(dtype=float),
        "proportional": proportional,
        "uniform": np.full(len(buckets), 1.0 / len(buckets)),
    }
    return {name: runtime_weights(weights) for name, weights in anchors.items()}


def support_ok(weights: np.ndarray, phase0_exposure: np.ndarray, phase1_scale: np.ndarray) -> bool:
    return bool(np.all(phase0_exposure + weights * phase1_scale <= TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12))


def add_point(
    points: dict[tuple[int, ...], CandidatePoint],
    weights: np.ndarray,
    source: str,
    center: np.ndarray,
    phase0_exposure: np.ndarray,
    phase1_scale: np.ndarray,
) -> None:
    runtime = runtime_weights(weights)
    radius = float(hellinger(runtime[None, :], center[None, :])[0])
    if radius < MINIMUM_HELLINGER or radius > MID_HELLINGER_MAX + 1e-12:
        return
    if not support_ok(runtime, phase0_exposure, phase1_scale):
        return
    counts = tuple(common_design.runtime_counts(runtime).tolist())
    points.setdefault(counts, CandidatePoint(counts=counts, source=source))


def maximum_feasible_blend(
    center: np.ndarray,
    anchor: np.ndarray,
    phase0_exposure: np.ndarray,
    phase1_scale: np.ndarray,
) -> float:
    if support_ok(runtime_weights(anchor), phase0_exposure, phase1_scale):
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        mixed = runtime_weights((1.0 - midpoint) * center + midpoint * anchor)
        if support_ok(mixed, phase0_exposure, phase1_scale):
            lower = midpoint
        else:
            upper = midpoint
    return lower


def generate_pool(
    center: np.ndarray,
    anchors: dict[str, np.ndarray],
    phase0_exposure: np.ndarray,
    phase1_scale: np.ndarray,
) -> tuple[dict[tuple[int, ...], CandidatePoint], list[tuple[int, ...]]]:
    points: dict[tuple[int, ...], CandidatePoint] = {}
    anchor_keys: list[tuple[int, ...]] = []
    for name, anchor in anchors.items():
        maximum = maximum_feasible_blend(center, anchor, phase0_exposure, phase1_scale)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            strength = maximum * fraction
            mixed = runtime_weights((1.0 - strength) * center + strength * anchor)
            before = set(points)
            add_point(points, mixed, f"anchor:{name}:{fraction:.2f}", center, phase0_exposure, phase1_scale)
            anchor_keys.extend(sorted(set(points) - before))

    generator = np.random.default_rng(DESIGN_SEED)
    for concentration in (20.0, 50.0, 100.0, 250.0):
        alpha = 0.5 + concentration * center
        for draw in generator.dirichlet(alpha, size=DIRICHLET_DRAWS_PER_SCALE):
            for blend in (0.2, 0.5, 1.0):
                add_point(
                    points,
                    (1.0 - blend) * center + blend * draw,
                    f"dirichlet:{concentration:g}:{blend:g}",
                    center,
                    phase0_exposure,
                    phase1_scale,
                )

    center_counts = common_design.runtime_counts(center)
    donors = np.flatnonzero(center_counts > 0)
    for _ in range(TRANSFER_DRAWS):
        counts = center_counts.copy()
        moves = int(generator.integers(1, 9))
        for _ in range(moves):
            eligible_donors = donors[counts[donors] > 0]
            if not len(eligible_donors):
                break
            donor = int(generator.choice(eligible_donors))
            recipient = int(generator.integers(len(counts)))
            if donor == recipient:
                continue
            amount = int(generator.integers(1, min(65, counts[donor] + 1)))
            counts[donor] -= amount
            counts[recipient] += amount
        add_point(
            points,
            counts / MIXTURE_BLOCK_SIZE,
            "runtime_count_transfer",
            center,
            phase0_exposure,
            phase1_scale,
        )
    return points, list(dict.fromkeys(anchor_keys))


def feature_matrix(weights: np.ndarray, center: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sqrt":
        return np.sqrt(weights) - np.sqrt(center)
    if kind == "direct":
        return weights - center
    raise ValueError(f"Unknown feature kind: {kind}")


def rank(features: np.ndarray) -> int:
    singular = np.linalg.svd(features, compute_uv=False)
    return int(np.sum(singular > singular[0] * 1e-10)) if len(singular) else 0


def qr_rows(features: np.ndarray, count: int) -> list[int]:
    _, _, pivots = linalg.qr(features.T, pivoting=True, mode="economic")
    return [int(index) for index in pivots[:count]]


def maximin_rows(features: np.ndarray, available: np.ndarray, selected: list[int], count: int) -> list[int]:
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(features, norms[:, None], out=np.zeros_like(features), where=norms[:, None] > 0)
    chosen = list(selected)
    remaining = available.copy()
    remaining[chosen] = False
    result: list[int] = []
    for _ in range(count):
        candidates = np.flatnonzero(remaining)
        if not len(candidates):
            break
        if chosen:
            similarity = normalized[candidates] @ normalized[chosen].T
            scores = np.min(np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity)), axis=1)
        else:
            scores = norms[candidates]
        position = int(candidates[int(np.argmax(scores))])
        result.append(position)
        chosen.append(position)
        remaining[position] = False
    return result


def select_fit_points(
    points: dict[tuple[int, ...], CandidatePoint],
    anchor_keys: list[tuple[int, ...]],
    center: np.ndarray,
) -> tuple[list[CandidatePoint], dict[str, object]]:
    pool = list(points.values())
    weights = np.stack([point.weights for point in pool])
    radii = hellinger(weights, center[None, :])
    features = feature_matrix(weights, center, "sqrt")
    anchor_indices = [position for position, point in enumerate(pool) if point.counts in set(anchor_keys)]
    anchor_indices = sorted(anchor_indices, key=lambda index: (radii[index], pool[index].source))[:ANCHOR_ROWS]

    local_available = (radii <= LOCAL_HELLINGER_MAX) & (radii >= MINIMUM_HELLINGER)
    local_candidates = np.flatnonzero(local_available)
    if rank(features[local_candidates]) != len(center):
        raise ValueError("Local candidate pool does not span the square-root response coordinates")
    local_pivots = local_candidates[qr_rows(features[local_candidates], len(center))].tolist()
    selected = list(dict.fromkeys([*anchor_indices, *local_pivots]))
    local_needed = max(0, ANCHOR_ROWS + LOCAL_ROWS - len(selected))
    selected.extend(maximin_rows(features, local_available, selected, local_needed))

    mid_available = (radii > LOCAL_HELLINGER_MAX) & (radii <= MID_HELLINGER_MAX)
    selected.extend(maximin_rows(features, mid_available, selected, MID_ROWS))
    if len(selected) < FIT_ROWS_PER_PREFIX:
        selected.extend(
            maximin_rows(features, np.ones(len(pool), dtype=bool), selected, FIT_ROWS_PER_PREFIX - len(selected))
        )
    selected = list(dict.fromkeys(selected))[:FIT_ROWS_PER_PREFIX]
    if len(selected) != FIT_ROWS_PER_PREFIX:
        raise ValueError(f"Could select only {len(selected)} fit rows")

    selected_weights = weights[selected]
    sqrt_rank = rank(feature_matrix(selected_weights, center, "sqrt"))
    direct_rank = rank(feature_matrix(selected_weights, center, "direct"))
    if (sqrt_rank, direct_rank) != (len(center), len(center) - 1):
        raise ValueError(f"Selected design is rank deficient: sqrt={sqrt_rank}, direct={direct_rank}")
    selected_radii = radii[selected]
    diagnostics: dict[str, object] = {
        "pool_rows": len(pool),
        "selected_rows": len(selected),
        "sqrt_feature_rank": sqrt_rank,
        "direct_feature_rank": direct_rank,
        "selected_hellinger_min": float(selected_radii.min()),
        "selected_hellinger_median": float(np.median(selected_radii)),
        "selected_hellinger_max": float(selected_radii.max()),
        "selected_anchor_rows": sum(pool[index].source.startswith("anchor:") for index in selected),
        "selected_zero_center_buckets": int(np.sum(center == 0.0)),
    }
    return [pool[index] for index in selected], diagnostics


def select_referee_points(
    points: dict[tuple[int, ...], CandidatePoint],
    selected: list[CandidatePoint],
    center: np.ndarray,
) -> list[CandidatePoint]:
    pool = list(points.values())
    selected_counts = {point.counts for point in selected}
    available = np.asarray([point.counts not in selected_counts for point in pool], dtype=bool)
    features = feature_matrix(np.stack([point.weights for point in pool]), center, "sqrt")
    selected_indices = [position for position, point in enumerate(pool) if point.counts in selected_counts]
    referee_indices = maximin_rows(features, available, selected_indices, REFEREE_ROWS_PER_PREFIX)
    if len(referee_indices) != REFEREE_ROWS_PER_PREFIX:
        raise ValueError(f"Could select only {len(referee_indices)} sealed referee rows")
    return [pool[index] for index in referee_indices]


def design_rows(
    candidate_id: str,
    center: np.ndarray,
    selected: list[CandidatePoint],
    referees: list[CandidatePoint],
    buckets: tuple[str, ...],
    phase0_scale: np.ndarray,
    phase1_scale: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    phase0_exposure = center * phase0_scale

    def append(
        continuation_id: str,
        weights: np.ndarray,
        *,
        role: str,
        fit_budget: bool,
        prefix_repeat_seed: int,
        data_seed: int,
        source: str,
    ) -> None:
        phase1_exposure = weights * phase1_scale
        summary_rows.append(
            {
                "prefix_candidate_id": candidate_id,
                "continuation_id": continuation_id,
                "role": role,
                "fit_budget": fit_budget,
                "prefix_repeat_seed": prefix_repeat_seed,
                "data_seed": data_seed,
                "source": source,
                "hellinger_to_tied": float(hellinger(weights[None, :], center[None, :])[0]),
                "max_phase0_materialized_epoch": float(phase0_exposure.max()),
                "max_phase1_materialized_epoch": float(phase1_exposure.max()),
                "max_total_materialized_epoch": float((phase0_exposure + phase1_exposure).max()),
            }
        )
        counts = common_design.runtime_counts(weights)
        for position, (bucket, count, weight) in enumerate(zip(buckets, counts, weights, strict=True)):
            weight_rows.append(
                {
                    "prefix_candidate_id": candidate_id,
                    "continuation_id": continuation_id,
                    "role": role,
                    "fit_budget": fit_budget,
                    "prefix_repeat_seed": prefix_repeat_seed,
                    "data_seed": data_seed,
                    "source": source,
                    "bucket": bucket,
                    "phase_1_count": int(count),
                    "phase_1_weight": float(weight),
                    "phase_1_materialized_epochs": float(phase1_exposure[position]),
                    "total_materialized_epochs": float(phase0_exposure[position] + phase1_exposure[position]),
                }
            )

    ordered = sorted(selected, key=lambda point: (not point.source.startswith("anchor:"), point.source, point.counts))
    for position, point in enumerate(ordered):
        append(
            f"fit_{position:03d}",
            point.weights,
            role="fixed_prefix_response_fit",
            fit_budget=True,
            prefix_repeat_seed=0,
            data_seed=FIT_DATA_SEED,
            source=point.source,
        )
    for position, point in enumerate(referees):
        append(
            f"referee_{position:03d}",
            point.weights,
            role="sealed_geometry_referee",
            fit_budget=False,
            prefix_repeat_seed=0,
            data_seed=FIT_DATA_SEED,
            source=point.source,
        )
    append(
        "tied_common_random",
        center,
        role="common_random_tied_control",
        fit_budget=False,
        prefix_repeat_seed=0,
        data_seed=FIT_DATA_SEED,
        source="tied",
    )
    for position, data_seed in enumerate(FRESH_TIED_DATA_SEEDS):
        append(
            f"tied_fresh_{position}",
            center,
            role="fresh_tied_control",
            fit_budget=False,
            prefix_repeat_seed=0,
            data_seed=data_seed,
            source="tied",
        )
    for position, data_seed in enumerate((FIT_DATA_SEED, *FRESH_TIED_DATA_SEEDS)):
        append(
            f"tied_prefix_seed1_{position}",
            center,
            role="prefix_state_tied_control",
            fit_budget=False,
            prefix_repeat_seed=1,
            data_seed=data_seed,
            source="tied",
        )
    return summary_rows, weight_rows


def build_design(
    candidate_weights_path: Path,
    selected_prefixes_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    selected_payload = json.loads(selected_prefixes_path.read_text())
    selected_ids = selected_candidate_ids(selected_payload)
    panel = common_design.load_canonical_panel_geometry()
    centers = candidate_centers(candidate_weights_path, selected_ids, panel.buckets)
    anchors = anchor_mixtures(panel.buckets, runtime_weights(panel.proportional))
    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    diagnostics = {}
    for candidate_id in selected_ids:
        center = centers[candidate_id]
        phase0_exposure = center * panel.c0
        tied_total = phase0_exposure + center * panel.c1
        if tied_total.max() > TOTAL_MATERIALIZED_EPOCH_CAP + 1e-12:
            raise ValueError(f"Tied continuation violates the total epoch cap for {candidate_id}")
        pool, anchor_keys = generate_pool(center, anchors, phase0_exposure, panel.c1)
        selected, candidate_diagnostics = select_fit_points(pool, anchor_keys, center)
        referees = select_referee_points(pool, selected, center)
        candidate_summary, candidate_weights = design_rows(
            candidate_id,
            center,
            selected,
            referees,
            panel.buckets,
            panel.c0,
            panel.c1,
        )
        summary_rows.extend(candidate_summary)
        weight_rows.extend(candidate_weights)
        diagnostics[candidate_id] = candidate_diagnostics
    summary = pd.DataFrame(summary_rows)
    weights = pd.DataFrame(weight_rows)
    expected_rows = len(selected_ids) * (FIT_ROWS_PER_PREFIX + REFEREE_ROWS_PER_PREFIX + CONTROL_ROWS_PER_PREFIX)
    if len(summary) != expected_rows or int(summary.fit_budget.sum()) != len(selected_ids) * FIT_ROWS_PER_PREFIX:
        raise ValueError("Harsh-cap branch row allocation changed")
    if (
        summary.groupby("prefix_candidate_id").continuation_id.nunique().min()
        != FIT_ROWS_PER_PREFIX + REFEREE_ROWS_PER_PREFIX + CONTROL_ROWS_PER_PREFIX
    ):
        raise ValueError("Continuation identities are not unique within each prefix")
    manifest: dict[str, object] = {
        "contract_version": "delphi_phase1_harsh_cap_branches_20260825_v1",
        "selected_candidate_ids": list(selected_ids),
        "target_metric": "uncheatable_bpb",
        "rows": {
            "fit_per_prefix": FIT_ROWS_PER_PREFIX,
            "sealed_referees_per_prefix": REFEREE_ROWS_PER_PREFIX,
            "controls_per_prefix": CONTROL_ROWS_PER_PREFIX,
            "total": len(summary),
        },
        "design": {
            "geometry": (
                "fixed deployment anchors followed by outcome-blind D-optimal and maximin coverage "
                "in square-root simplex coordinates"
            ),
            "fit_anchors": ["historical_uncheatable_frontier", "proportional", "uniform"],
            "historical_frontier_is_prior_outcome_selected": True,
            "old_prefix_model_candidate_used": False,
            "common_random_number_data_seed": FIT_DATA_SEED,
            "fresh_tied_data_seeds": list(FRESH_TIED_DATA_SEEDS),
            "local_rows": LOCAL_ROWS,
            "mid_rows": MID_ROWS,
            "anchor_row_budget": ANCHOR_ROWS,
            "local_hellinger_max": LOCAL_HELLINGER_MAX,
            "mid_hellinger_max": MID_HELLINGER_MAX,
            "total_materialized_epoch_cap": TOTAL_MATERIALIZED_EPOCH_CAP,
            "boundary_behavior": "one-sided full-rank coverage is allowed when tied has zero-weight buckets",
        },
        "diagnostics": diagnostics,
        "provenance": {
            "design_seed": DESIGN_SEED,
            "candidate_weights_sha256": file_sha256(candidate_weights_path),
            "selected_prefixes_sha256": file_sha256(selected_prefixes_path),
            "historical_frontier_uri": HISTORICAL_FRONTIER_URI,
            "historical_frontier_sha256": HISTORICAL_FRONTIER_SHA256,
        },
    }
    return summary, weights, manifest


def main() -> None:
    args = parse_args()
    summary, weights, manifest = build_design(args.candidate_weights, args.selected_prefixes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "continuation_summary.csv"
    weights_path = args.output_dir / "continuation_weights.csv"
    summary.to_csv(summary_path, index=False)
    weights.loc[:, list(WEIGHT_ARTIFACT_COLUMNS)].to_csv(weights_path, index=False)
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
