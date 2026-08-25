# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Audit the StarCoder design matrices of `hierarchical_phase_bucket_replay`.

Checks the audit claim that both StarCoder swarms produce a seven-column design
with numerical rank five because, for singleton families, `family_overexposure`
and `family_member_replay` are the same column.

The design is built through the exact Observatory path used for these swarms:
`export_mixture_fit_observatory.family_dataset` ->
`benchmark_hierarchical_coverage_grp_20260715.build_design`.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

VARIANT = hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
CACHE_DIR = observatory.CACHE_DIR


def cached_config(swarm_id: str, policy_class: str) -> hierarchical_grp.Config | None:
    """Selected config persisted by the Observatory cache, if present."""
    path = CACHE_DIR / swarm_id / "starcoder_bpb" / policy_class / "hierarchical_phase_bucket_replay.json"
    if not path.exists():
        return None
    tuning = json.loads(path.read_text())["fitDetail"]["tuning"]
    shape_parameters = tuning["shapeParameters"]
    shape = family_grp.Shape(
        exponent=float(shape_parameters["exponent"]),
        late_multiplier=float(shape_parameters["lateMultiplier"]),
        forgetting_rate=float(shape_parameters["forgettingRate"]),
        penalty_threshold=float(shape_parameters["penaltyThreshold"]),
        quality_discount=1.0,
    )
    return hierarchical_grp.Config(
        variant=VARIANT,
        shape_index=-1,
        shape=shape,
        l2=float(tuning["l2"]),
        residual_shrink=float(tuning["residualShrink"]),
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )


def duplicate_pairs(values: np.ndarray, names: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """Column pairs that are identical after scaling each column to unit max-abs."""
    scale = np.max(np.abs(values), axis=0)
    scaled = values / np.where(scale > 0.0, scale, 1.0)
    pairs: list[tuple[str, str, float]] = []
    for left, right in combinations(range(values.shape[1]), 2):
        gap = float(np.max(np.abs(scaled[:, left] - scaled[:, right])))
        if gap == 0.0 or gap <= 1e-12:
            pairs.append((names[left], names[right], gap))
    return pairs


def near_duplicate_pairs(values: np.ndarray, names: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """Column pairs with |Pearson correlation| above 0.999 (collinearity, not identity)."""
    centered = values - values.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered, axis=0)
    safe = np.where(norm > 0.0, norm, 1.0)
    unit = centered / safe
    pairs: list[tuple[str, str, float]] = []
    for left, right in combinations(range(values.shape[1]), 2):
        if norm[left] == 0.0 or norm[right] == 0.0:
            continue
        correlation = float(unit[:, left] @ unit[:, right])
        if abs(correlation) > 0.999:
            pairs.append((names[left], names[right], correlation))
    return pairs


def condition_number(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])


def report_design(label: str, dataset: family_grp.Dataset, config: hierarchical_grp.Config) -> dict[str, object]:
    design = hierarchical_grp.build_design(dataset, config)
    values = design.values
    rank = int(np.linalg.matrix_rank(values))
    centered = values - values.mean(axis=0, keepdims=True)
    centered_rank = int(np.linalg.matrix_rank(centered))
    singular = np.linalg.svd(centered, compute_uv=False)
    exact = duplicate_pairs(values, design.names)
    near = near_duplicate_pairs(values, design.names)
    zero_columns = [design.names[i] for i in range(values.shape[1]) if np.all(values[:, i] == values[0, i])]

    print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
    print(f"  rows (observations)          : {values.shape[0]}")
    print(f"  columns                      : {values.shape[1]}")
    print(f"  matrix_rank(raw design)      : {rank}   deficiency = {values.shape[1] - rank}")
    print(f"  matrix_rank(centered design) : {centered_rank}   deficiency = {values.shape[1] - centered_rank}")
    print(f"  condition number (centered)  : {condition_number(values):.6g}")
    print(f"  centered singular values     : {np.array2string(singular, precision=4, max_line_width=200)}")
    print(
        f"  shape: exponent={config.shape.exponent:.6f} late_multiplier={config.shape.late_multiplier:.6f} "
        f"forgetting_rate={config.shape.forgetting_rate:.6g} penalty_threshold={config.shape.penalty_threshold:.6f}"
    )
    print(f"  l2={config.l2} residual_shrink={config.residual_shrink}")
    print("  column names:")
    for index, name in enumerate(design.names):
        print(f"    [{index}] {name}")
    print(f"  EXACT duplicate column pairs (unit max-abs scaled, max |diff| <= 1e-12): {len(exact)}")
    for left, right, gap in exact:
        print(f"    {left}  ==  {right}   (max abs scaled diff = {gap:.3e})")
    exact_keys = {(left, right) for left, right, _gap in exact}
    extra_near = [(left, right, corr) for left, right, corr in near if (left, right) not in exact_keys]
    print(f"  near-duplicate pairs (|corr| > 0.999, excluding the exact duplicates): {len(extra_near)}")
    for left, right, correlation in extra_near:
        print(f"    {left}  ~  {right}   (corr = {correlation:+.9f})")
    if zero_columns:
        print(f"  constant columns: {zero_columns}")

    return {
        "label": label,
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "rank": rank,
        "centered_rank": centered_rank,
        "deficiency": int(values.shape[1] - rank),
        "condition_number": condition_number(values),
        "exact_duplicates": [(left, right) for left, right, _gap in exact],
        "names": list(design.names),
    }


def verify_singleton_identity(dataset: family_grp.Dataset, config: hierarchical_grp.Config) -> None:
    """Confirm the mechanism: for a singleton family the two harm columns are the same formula."""
    exposure = hierarchical_grp.retained_exposure(dataset, config.shape)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    aggregate = hierarchical_grp.overexposure_harm(family_total, config.shape.penalty_threshold)
    member = hierarchical_grp.overexposure_harm(exposure, config.shape.penalty_threshold)
    replay = np.column_stack([member[:, members].mean(axis=1) for members in dataset.family_members])
    print("\n  mechanism check (family_overexposure vs family_member_replay):")
    for index, name in enumerate(dataset.family_names):
        gap = float(np.max(np.abs(aggregate[:, index] - replay[:, index])))
        print(f"    family {name!r} (|members|={len(dataset.family_members[index])}): max abs diff = {gap:.3e}")


def sweep_shapes(label: str, dataset: family_grp.Dataset, policy_class: str) -> None:
    """Rank is a property of the column set, not of the tuned shape; confirm over the whole shape grid."""
    shapes = observatory.hierarchical_phase_replay_shape_candidates(policy_class)
    ranks: dict[int, int] = {}
    conditions: list[float] = []
    duplicate_counts: set[int] = set()
    for shape in shapes:
        config = hierarchical_grp.Config(VARIANT, -1, shape, 0.0, 1.0, 0.0, 0.0)
        design = hierarchical_grp.build_design(dataset, config)
        rank = int(np.linalg.matrix_rank(design.values))
        ranks[rank] = ranks.get(rank, 0) + 1
        conditions.append(condition_number(design.values))
        duplicate_counts.add(len(duplicate_pairs(design.values, design.names)))
    finite = [c for c in conditions if np.isfinite(c)]
    condition_text = (
        f"[{min(finite):.4g}, {max(finite):.4g}]" if finite else "all infinite (exactly singular centered design)"
    )
    print(
        f"\n  shape sweep ({len(shapes)} candidate shapes, {policy_class}): rank histogram {ranks}, "
        f"exact-duplicate-pair counts {sorted(duplicate_counts)}, "
        f"condition number range {condition_text}"
        f"{f' ({len(conditions) - len(finite)}/{len(conditions)} infinite)' if len(finite) != len(conditions) else ''}"
    )


def main() -> None:
    print("code path under audit")
    print(
        f"  design builder      : {Path(hierarchical_grp.__file__).resolve()}::build_design "
        f"(line {hierarchical_grp.build_design.__code__.co_firstlineno})"
    )
    print(
        f"  observatory wrapper : {Path(observatory.__file__).resolve()}::hierarchical_phase_replay_fit -> "
        f"family_dataset -> hierarchical_grp.fit_model -> build_design"
    )
    print(f"  variant             : {VARIANT.value}")
    print(f"  cosine source CSV   : {observatory.COSINE_DATA}")
    print(f"  wsd80 source CSV    : {observatory.WSD80_DATA}")

    cosine_raw = observatory.load_cosine_starcoder()
    wsd80_raw = observatory.load_wsd80_starcoder(cosine_raw)

    summary: list[dict[str, object]] = []
    for swarm_id, raw in (("starcoder_cosine", cosine_raw), ("starcoder_wsd80", wsd80_raw)):
        structured = observatory.family_dataset(raw)
        sizes = [len(members) for members in structured.family_members]
        print(f"\n{'#' * 100}")
        print(f"# {swarm_id}: dataset '{raw.name}'")
        print(f"{'#' * 100}")
        print(f"  observations (n)  : {structured.n}")
        print(f"  buckets/domains(m): {structured.m}  -> {list(structured.domains)}")
        print(f"  families          : {len(structured.family_names)} -> {list(structured.family_names)}")
        print(f"  family sizes      : {sizes}  all singleton = {all(size == 1 for size in sizes)}")
        print(
            f"  c0 = {np.array2string(structured.c0, precision=6)}   c1 = {np.array2string(structured.c1, precision=6)}"
        )

        for policy_class in (observatory.TWO_PHASE, observatory.SINGLE_PHASE):
            if policy_class == observatory.SINGLE_PHASE:
                fit_raw, _indices = observatory.tied_policy_subset(raw)
            else:
                fit_raw = raw
            fit_dataset = observatory.family_dataset(fit_raw)
            config = cached_config(swarm_id, policy_class)
            source = "Observatory cache (actually selected)"
            if config is None:
                config = hierarchical_grp.Config(
                    VARIANT,
                    0,
                    observatory.hierarchical_phase_replay_shape_candidates(policy_class)[0],
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                )
                source = "fallback: first shape candidate (no cache entry)"
            record = report_design(f"{swarm_id} / {policy_class} / config from {source}", fit_dataset, config)
            verify_singleton_identity(fit_dataset, config)
            sweep_shapes(swarm_id, fit_dataset, policy_class)
            summary.append(record)

    # Robustness: the exporter reads the 20260711 WSD80 panel. A refined panel exists on disk;
    # rebinding the module constant shows whether the rank story depends on which panel is used.
    refined = (
        observatory.SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_observed_metrics.csv"
    )
    if refined.exists():
        original = observatory.WSD80_DATA
        observatory.WSD80_DATA = refined
        try:
            refined_dataset = observatory.family_dataset(observatory.load_wsd80_starcoder(cosine_raw))
        finally:
            observatory.WSD80_DATA = original
        config = cached_config("starcoder_wsd80", observatory.TWO_PHASE)
        assert config is not None
        summary.append(
            report_design(
                "starcoder_wsd80 (refined 20260714 panel, not the exporter default) / two_phase",
                refined_dataset,
                config,
            )
        )

    print(f"\n{'=' * 100}\nVERDICT TABLE\n{'=' * 100}")
    print(f"{'design':<70} {'cols':>5} {'rank':>5} {'def':>4} {'dupPairs':>9}")
    for record in summary:
        print(
            f"{record['label'][:68]:<70} {record['columns']:>5} {record['rank']:>5} "
            f"{record['deficiency']:>4} {len(record['exact_duplicates']):>9}"
        )

    claim_columns = 7
    claim_rank = 5
    verdicts = [
        (record["label"], record["columns"] == claim_columns, record["rank"] == claim_rank) for record in summary
    ]
    print("\nclaim: 'seven columns but rank five'")
    for label, column_ok, rank_ok in verdicts:
        print(f"  {label[:68]:<70} columns==7: {column_ok}   rank==5: {rank_ok}")


if __name__ == "__main__":
    main()
