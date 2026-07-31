# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///
"""Materialize a two-anchor fixed-aggregate phase-order DOE at 60M/1.2B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = SCRIPT_DIR / "reference_outputs/60m_39bucket_checkpoint_audit_20260724"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/60m_fixed_aggregate_phase_order_panel_20260725"

EXPERIMENT_BUDGET = 1_200_000_000
BATCH_SIZE = 128
SEQ_LEN = 2048
PHASE_0_NOMINAL_FRACTION = 0.8
MIXTURE_BLOCK_SIZE = 2048
TOTAL_STEPS = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
STEP_ALIGNMENT = MIXTURE_BLOCK_SIZE // math.gcd(BATCH_SIZE, MIXTURE_BLOCK_SIZE)
PHASE_1_START_STEP = (int(TOTAL_STEPS * PHASE_0_NOMINAL_FRACTION) // STEP_ALIGNMENT) * STEP_ALIGNMENT
ALPHA_0 = PHASE_1_START_STEP / TOTAL_STEPS
ALPHA_1 = 1.0 - ALPHA_0
RUN_ID_BASE = 7_250_000
DATA_SEED_BASE = 7_252_000
TRAINER_SEED = 0
SPANNING_DIRECTIONS_PER_ANCHOR = 19
PRIMARY_PHASE_TVS = (0.08, 0.10, 0.12, 0.15)
MECHANISTIC_PHASE_TV = 0.12
CURVATURE_PHASE_TV = 0.24
SEED_BLOCKS_PER_ANCHOR = 4
EXPECTED_ROWS = 140
EXPECTED_UNIQUE_POLICIES = 126
GEOMETRY_TOLERANCE = 2e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

assert TOTAL_STEPS == 4577
assert PHASE_1_START_STEP == 3648


@dataclass(frozen=True)
class MechanisticDirection:
    direction_id: str
    label: str
    left_domains: tuple[str, ...]
    right_domains: tuple[str, ...]
    hypothesis: str
    include_curvature: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _domains(frame: pd.DataFrame) -> tuple[str, ...]:
    domains = tuple(
        column.removeprefix("phase_0_")
        for column in frame.columns
        if column.startswith("phase_0_") and f"phase_1_{column.removeprefix('phase_0_')}" in frame.columns
    )
    if len(domains) != 39:
        raise ValueError(f"Expected 39 domains, found {len(domains)}")
    return domains


def _vector(row: pd.Series, domains: tuple[str, ...], phase: int) -> np.ndarray:
    values = row[[f"phase_{phase}_{domain}" for domain in domains]].to_numpy(dtype=float)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Anchor weights must be finite and non-negative")
    return values / values.sum()


def _policy_hash(phase_0: np.ndarray, phase_1: np.ndarray) -> str:
    payload = np.round(np.concatenate([phase_0, phase_1]), 12).astype("<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _anchors(domains: tuple[str, ...]) -> dict[str, tuple[np.ndarray, str]]:
    fit_one = pd.read_csv(AUDIT_DIR / "fit_single_phase.csv", low_memory=False)
    fit_two = pd.read_csv(AUDIT_DIR / "fit_two_phase.csv", low_memory=False)
    uncheatable = fit_one.loc[fit_one["run_name"].eq("singleavg_run_00125")]
    proportional = fit_two.loc[fit_two["run_name"].eq("baseline_proportional")]
    if len(uncheatable) != 1 or len(proportional) != 1:
        raise ValueError("Could not resolve the frozen Uncheatable and proportional anchors")
    result = {
        "uncheatable_frontier": (_vector(uncheatable.iloc[0], domains, 0), "singleavg_run_00125"),
        "proportional": (_vector(proportional.iloc[0], domains, 0), "baseline_proportional"),
    }
    for anchor_id, (weights, _) in result.items():
        if not np.isclose(weights.sum(), 1.0, atol=GEOMETRY_TOLERANCE):
            raise ValueError(f"{anchor_id} does not sum to one")
    return result


def _normalize_direction(anchor: np.ndarray, values: np.ndarray) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    direction -= direction.sum() * anchor
    direction -= direction.sum() / len(direction)
    if abs(direction.sum()) > GEOMETRY_TOLERANCE:
        raise ValueError("Direction is not in the simplex tangent space")
    phase_tv = 0.5 * np.abs(direction).sum()
    if phase_tv <= 0:
        raise ValueError("Direction has zero phase contrast")
    return direction / phase_tv


def _weighted_correlation(left: np.ndarray, right: np.ndarray, anchor: np.ndarray) -> float:
    denominator = np.sqrt(np.sum(left * left / anchor) * np.sum(right * right / anchor))
    return float(np.sum(left * right / anchor) / denominator)


def _spanning_directions(anchor: np.ndarray, *, seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = []
    for _ in range(2048):
        log_contrast = rng.choice((-1.0, 1.0), size=len(anchor))
        log_contrast -= float(anchor @ log_contrast)
        candidates.append(_normalize_direction(anchor, anchor * log_contrast))

    selected = [candidates.pop(0)]
    while len(selected) < count:
        best_index = min(
            range(len(candidates)),
            key=lambda index: max(
                abs(_weighted_correlation(candidates[index], existing, anchor)) for existing in selected
            ),
        )
        selected.append(candidates.pop(best_index))
    return selected


def _mechanistic_registry(domains: tuple[str, ...]) -> tuple[MechanisticDirection, ...]:
    def complement(left: set[str]) -> tuple[str, ...]:
        return tuple(domain for domain in domains if domain not in left)

    dolmino = tuple(domain for domain in domains if domain.startswith("dolmino_"))
    high_quality = tuple(domain for domain in domains if domain.startswith("dolma3_cc/") and domain.endswith("_high"))
    non_cc = {domain for domain in domains if not domain.startswith("dolma3_cc/")}
    technical = {
        domain
        for domain in domains
        if any(topic in domain for topic in ("science_math", "education_and_jobs", "electronics_and_hardware"))
    } | {
        "dolma3_stack_edu",
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolmino_stack_edu_fim",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_code",
        "dolmino_synth_math",
        "dolmino_synth_thinking",
    }
    knowledge = {
        domain
        for domain in domains
        if any(
            topic in domain for topic in ("science_math", "education_and_jobs", "history_and_geography", "literature")
        )
    } | {
        "dolma3_arxiv",
        "dolma3_wikipedia",
        "dolmino_olmocr_pdfs_hq",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_math",
        "dolmino_synth_thinking",
    }
    high_and_curated = set(high_quality) | non_cc
    reasoning = {domain for domain in domains if domain.startswith("dolmino_synth_")} | {
        "dolmino_stem_heavy_crawl",
        "dolmino_stack_edu_fim",
        "dolma3_finemath_3plus",
    }
    professional = {
        domain
        for domain in domains
        if any(
            topic in domain
            for topic in (
                "finance_and_business",
                "health",
                "education_and_jobs",
                "science_math",
                "electronics_and_hardware",
                "industrial",
            )
        )
    } | {"dolma3_arxiv", "dolma3_stack_edu", "dolmino_stem_heavy_crawl"}
    registry = (
        MechanisticDirection(
            "dolmino_vs_broad",
            "Dolmino late versus broad pretraining early",
            dolmino,
            complement(set(dolmino)),
            "late_specialization",
            True,
        ),
        MechanisticDirection(
            "cc_high_vs_remainder",
            "Common Crawl high-quality late versus the remaining pool early",
            high_quality,
            complement(set(high_quality)),
            "quality_late",
            True,
        ),
        MechanisticDirection(
            "curated_noncc_vs_cc",
            "Curated non-Common-Crawl sources late versus Common Crawl early",
            tuple(domain for domain in domains if domain in non_cc),
            complement(non_cc),
            "curated_sources_late",
            True,
        ),
        MechanisticDirection(
            "professional_knowledge",
            "Professional and technical knowledge late versus other topics early",
            tuple(domain for domain in domains if domain in professional),
            complement(professional),
            "professional_knowledge_late",
            True,
        ),
        MechanisticDirection(
            "technical_specialization",
            "Technical, code, and mathematical sources late versus other sources early",
            tuple(domain for domain in domains if domain in technical),
            complement(technical),
            "technical_late",
        ),
        MechanisticDirection(
            "knowledge_specialization",
            "Document and knowledge-intensive sources late versus other sources early",
            tuple(domain for domain in domains if domain in knowledge),
            complement(knowledge),
            "knowledge_late",
        ),
        MechanisticDirection(
            "high_and_curated_vs_low_cc",
            "High-quality Common Crawl and curated sources late versus low-quality Common Crawl early",
            tuple(domain for domain in domains if domain in high_and_curated),
            complement(high_and_curated),
            "quality_and_curation_late",
        ),
        MechanisticDirection(
            "reasoning_specialization",
            "Synthetic reasoning and advanced technical sources late versus other sources early",
            tuple(domain for domain in domains if domain in reasoning),
            complement(reasoning),
            "reasoning_late",
        ),
    )
    known = set(domains)
    for item in registry:
        missing = (set(item.left_domains) | set(item.right_domains)) - known
        if missing:
            raise ValueError(f"{item.direction_id} references unknown domains: {sorted(missing)}")
        if set(item.left_domains) & set(item.right_domains):
            raise ValueError(f"{item.direction_id} has overlapping sides")
    return registry


def _group_direction(
    anchor: np.ndarray,
    domains: tuple[str, ...],
    left_domains: tuple[str, ...],
    right_domains: tuple[str, ...],
) -> np.ndarray:
    domain_index = {domain: index for index, domain in enumerate(domains)}
    left = np.asarray([domain_index[domain] for domain in left_domains], dtype=int)
    right = np.asarray([domain_index[domain] for domain in right_domains], dtype=int)
    left_mass = float(anchor[left].sum())
    right_mass = float(anchor[right].sum())
    if left_mass <= 0 or right_mass <= 0:
        raise ValueError("Mechanistic contrast has a zero-mass side")
    direction = np.zeros_like(anchor)
    direction[left] = -anchor[left] / left_mass
    direction[right] = anchor[right] / right_mass
    return _normalize_direction(anchor, direction)


def _max_antithetic_phase_tv(anchor: np.ndarray, unit_direction: np.ndarray) -> float:
    active = np.abs(unit_direction) > 0
    return float(np.min(anchor[active] / (max(ALPHA_0, ALPHA_1) * np.abs(unit_direction[active]))))


def _phase_information(anchor: np.ndarray, phase_0: np.ndarray, phase_1: np.ndarray) -> float:
    def kl(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0
        return float(np.sum(left[positive] * np.log(left[positive] / right[positive])))

    return ALPHA_0 * kl(phase_0, anchor) + ALPHA_1 * kl(phase_1, anchor)


def _existing_policy_hashes() -> set[str]:
    hashes: set[str] = set()
    for name in ("fit_two_phase.csv", "fit_single_phase.csv", "heldout_observations.csv"):
        frame = pd.read_csv(AUDIT_DIR / name, usecols=["policy_hash"])
        hashes.update(frame["policy_hash"].dropna().astype(str))
    return hashes


def build_panel() -> tuple[pd.DataFrame, tuple[str, ...]]:
    fit_two = pd.read_csv(AUDIT_DIR / "fit_two_phase.csv", nrows=1)
    domains = _domains(fit_two)
    anchors = _anchors(domains)
    mechanistic = _mechanistic_registry(domains)
    existing_hashes = _existing_policy_hashes()
    rows: list[dict[str, object]] = []
    pair_index_by_anchor: dict[str, int] = {anchor_id: 0 for anchor_id in anchors}
    base_sentinel_blocks: dict[tuple[str, str], int] = {}

    def add_row(
        *,
        anchor_id: str,
        anchor: np.ndarray,
        source_anchor_run_name: str,
        direction_id: str,
        direction_label: str,
        direction_family: str,
        hypothesis: str,
        sign: str,
        unit_direction: np.ndarray,
        target_phase_tv: float,
        replicate_index: int,
        seed_block: int,
        pair_id: str,
    ) -> None:
        multiplier = 1.0 if sign == "plus" else -1.0
        contrast = multiplier * target_phase_tv * unit_direction
        phase_0 = anchor + ALPHA_1 * contrast
        phase_1 = anchor - ALPHA_0 * contrast
        if phase_0.min() < -GEOMETRY_TOLERANCE or phase_1.min() < -GEOMETRY_TOLERANCE:
            raise ValueError(f"{anchor_id}/{pair_id}/{sign} leaves the simplex")
        phase_0 = np.maximum(phase_0, 0)
        phase_1 = np.maximum(phase_1, 0)
        aggregate = ALPHA_0 * phase_0 + ALPHA_1 * phase_1
        aggregate_error = float(np.max(np.abs(aggregate - anchor)))
        phase_tv = float(0.5 * np.abs(phase_0 - phase_1).sum())
        policy_sha256 = _policy_hash(phase_0, phase_1)
        candidate_id = f"{anchor_id}_{pair_id}_{sign}_r{replicate_index}"
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "run_id": RUN_ID_BASE + len(rows),
            "anchor_id": anchor_id,
            "source_anchor_run_name": source_anchor_run_name,
            "pair_id": pair_id,
            "direction_id": direction_id,
            "direction_label": direction_label,
            "direction_family": direction_family,
            "hypothesis": hypothesis,
            "sign": sign,
            "orientation": "named_left_later" if sign == "plus" else "named_left_earlier",
            "replicate_index": replicate_index,
            "seed_block": seed_block,
            "data_seed": DATA_SEED_BASE + list(anchors).index(anchor_id) * 100 + seed_block,
            "trainer_seed": TRAINER_SEED,
            "realized_phase_0_fraction": ALPHA_0,
            "realized_phase_1_fraction": ALPHA_1,
            "target_phase_tv": target_phase_tv,
            "phase_tv": phase_tv,
            "phase_information_kl": _phase_information(anchor, phase_0, phase_1),
            "max_antithetic_phase_tv": _max_antithetic_phase_tv(anchor, unit_direction),
            "aggregate_max_abs_error": aggregate_error,
            "max_phase_weight": float(max(phase_0.max(), phase_1.max())),
            "policy_sha256": policy_sha256,
            "is_control": False,
        }
        for index, domain in enumerate(domains):
            row[f"phase_0_{domain}"] = float(phase_0[index])
            row[f"phase_1_{domain}"] = float(phase_1[index])
            row[f"aggregate_{domain}"] = float(anchor[index])
        rows.append(row)

    for anchor_index, (anchor_id, (anchor, source_anchor_run_name)) in enumerate(anchors.items()):
        for seed_block in range(SEED_BLOCKS_PER_ANCHOR):
            row: dict[str, object] = {
                "candidate_id": f"{anchor_id}_tied_control_s{seed_block}",
                "run_id": RUN_ID_BASE + len(rows),
                "anchor_id": anchor_id,
                "source_anchor_run_name": source_anchor_run_name,
                "pair_id": f"control_s{seed_block}",
                "direction_id": "tied_control",
                "direction_label": "Same-seed tied aggregate control",
                "direction_family": "tied_control",
                "hypothesis": "noise_control",
                "sign": "control",
                "orientation": "phase_tied",
                "replicate_index": seed_block,
                "seed_block": seed_block,
                "data_seed": DATA_SEED_BASE + anchor_index * 100 + seed_block,
                "trainer_seed": TRAINER_SEED,
                "realized_phase_0_fraction": ALPHA_0,
                "realized_phase_1_fraction": ALPHA_1,
                "target_phase_tv": 0.0,
                "phase_tv": 0.0,
                "phase_information_kl": 0.0,
                "max_antithetic_phase_tv": 0.0,
                "aggregate_max_abs_error": 0.0,
                "max_phase_weight": float(anchor.max()),
                "policy_sha256": _policy_hash(anchor, anchor),
                "is_control": True,
            }
            for index, domain in enumerate(domains):
                row[f"phase_0_{domain}"] = float(anchor[index])
                row[f"phase_1_{domain}"] = float(anchor[index])
                row[f"aggregate_{domain}"] = float(anchor[index])
            rows.append(row)

        spanning = _spanning_directions(
            anchor,
            seed=7_250 + anchor_index,
            count=SPANNING_DIRECTIONS_PER_ANCHOR,
        )
        for local_index, direction in enumerate(spanning):
            global_index = anchor_index * SPANNING_DIRECTIONS_PER_ANCHOR + local_index
            target_tv = PRIMARY_PHASE_TVS[global_index % len(PRIMARY_PHASE_TVS)]
            if _max_antithetic_phase_tv(anchor, direction) < target_tv:
                raise ValueError(f"span_{global_index:02d} cannot reach target TV {target_tv}")
            pair_index = pair_index_by_anchor[anchor_id]
            pair_index_by_anchor[anchor_id] += 1
            tv_index = global_index % len(PRIMARY_PHASE_TVS)
            seed_block = (local_index // len(PRIMARY_PHASE_TVS) + 2 * tv_index + anchor_index) % SEED_BLOCKS_PER_ANCHOR
            for sign in ("plus", "minus"):
                add_row(
                    anchor_id=anchor_id,
                    anchor=anchor,
                    source_anchor_run_name=source_anchor_run_name,
                    direction_id=f"span_{global_index:02d}",
                    direction_label=f"Aggregate-weighted tangent direction {global_index:02d}",
                    direction_family="spanning_tangent",
                    hypothesis="agnostic_phase_order",
                    sign=sign,
                    unit_direction=direction,
                    target_phase_tv=target_tv,
                    replicate_index=0,
                    seed_block=seed_block,
                    pair_id=f"span_{global_index:02d}_tv{target_tv:g}",
                )

        for item in mechanistic:
            direction = _group_direction(anchor, domains, item.left_domains, item.right_domains)
            if _max_antithetic_phase_tv(anchor, direction) < MECHANISTIC_PHASE_TV:
                raise ValueError(f"{anchor_id}/{item.direction_id} cannot reach primary mechanistic TV")
            pair_index = pair_index_by_anchor[anchor_id]
            pair_index_by_anchor[anchor_id] += 1
            seed_block = pair_index % SEED_BLOCKS_PER_ANCHOR
            base_sentinel_blocks[(anchor_id, item.direction_id)] = seed_block
            for sign in ("plus", "minus"):
                add_row(
                    anchor_id=anchor_id,
                    anchor=anchor,
                    source_anchor_run_name=source_anchor_run_name,
                    direction_id=item.direction_id,
                    direction_label=item.label,
                    direction_family="mechanistic_primary",
                    hypothesis=item.hypothesis,
                    sign=sign,
                    unit_direction=direction,
                    target_phase_tv=MECHANISTIC_PHASE_TV,
                    replicate_index=0,
                    seed_block=seed_block,
                    pair_id=f"{item.direction_id}_tv{MECHANISTIC_PHASE_TV:g}",
                )

            if item.include_curvature:
                if _max_antithetic_phase_tv(anchor, direction) < CURVATURE_PHASE_TV:
                    raise ValueError(f"{anchor_id}/{item.direction_id} cannot reach curvature TV")
                pair_index = pair_index_by_anchor[anchor_id]
                pair_index_by_anchor[anchor_id] += 1
                curvature_seed_block = pair_index % SEED_BLOCKS_PER_ANCHOR
                for sign in ("plus", "minus"):
                    add_row(
                        anchor_id=anchor_id,
                        anchor=anchor,
                        source_anchor_run_name=source_anchor_run_name,
                        direction_id=item.direction_id,
                        direction_label=item.label,
                        direction_family="mechanistic_curvature",
                        hypothesis=item.hypothesis,
                        sign=sign,
                        unit_direction=direction,
                        target_phase_tv=CURVATURE_PHASE_TV,
                        replicate_index=0,
                        seed_block=curvature_seed_block,
                        pair_id=f"{item.direction_id}_tv{CURVATURE_PHASE_TV:g}",
                    )

        for sentinel in mechanistic[:2]:
            direction = _group_direction(anchor, domains, sentinel.left_domains, sentinel.right_domains)
            base_block = base_sentinel_blocks[(anchor_id, sentinel.direction_id)]
            repeat_block = (base_block + 2) % SEED_BLOCKS_PER_ANCHOR
            for sign in ("plus", "minus"):
                add_row(
                    anchor_id=anchor_id,
                    anchor=anchor,
                    source_anchor_run_name=source_anchor_run_name,
                    direction_id=sentinel.direction_id,
                    direction_label=sentinel.label,
                    direction_family="sentinel_repeat",
                    hypothesis=sentinel.hypothesis,
                    sign=sign,
                    unit_direction=direction,
                    target_phase_tv=MECHANISTIC_PHASE_TV,
                    replicate_index=1,
                    seed_block=repeat_block,
                    pair_id=f"{sentinel.direction_id}_tv{MECHANISTIC_PHASE_TV:g}",
                )

    panel = pd.DataFrame(rows)
    if len(panel) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} panel rows, found {len(panel)}")
    expected_family_counts = {
        "mechanistic_curvature": 16,
        "mechanistic_primary": 32,
        "sentinel_repeat": 8,
        "spanning_tangent": 76,
        "tied_control": 8,
    }
    family_counts = panel["direction_family"].value_counts().sort_index().to_dict()
    if family_counts != expected_family_counts:
        raise ValueError(f"Unexpected direction-family counts: {family_counts}")
    if panel["candidate_id"].duplicated().any() or panel["run_id"].duplicated().any():
        raise ValueError("Panel candidate IDs and run IDs must be unique")
    if panel["policy_sha256"].nunique() != EXPECTED_UNIQUE_POLICIES:
        raise ValueError(
            f"Expected {EXPECTED_UNIQUE_POLICIES} unique policies, found {panel['policy_sha256'].nunique()}"
        )
    treatments = panel.loc[~panel["is_control"]]
    overlapping = sorted(set(treatments["policy_sha256"]) & existing_hashes)
    if overlapping:
        raise ValueError(f"Treatment policies overlap the audited 60M archive: {overlapping[:5]}")
    if panel["aggregate_max_abs_error"].max() > GEOMETRY_TOLERANCE:
        raise ValueError("Panel does not preserve aggregate weights exactly")
    if not np.allclose(panel["phase_tv"], panel["target_phase_tv"], atol=GEOMETRY_TOLERANCE):
        raise ValueError("Realized phase TV differs from the frozen target")

    treatment_groups = treatments.groupby(["anchor_id", "pair_id", "replicate_index"], sort=False)
    for key, group in treatment_groups:
        if len(group) != 2 or set(group["sign"]) != {"plus", "minus"}:
            raise ValueError(f"{key} is not an antithetic pair")
        if group["data_seed"].nunique() != 1:
            raise ValueError(f"{key} does not share a data seed")
    spanning = treatments.loc[treatments["direction_family"].eq("spanning_tangent")]
    for key, group in spanning.groupby(["anchor_id", "target_phase_tv"]):
        if group["seed_block"].nunique() < 3:
            raise ValueError(f"{key} is aliased to fewer than three seed blocks")
    return panel, domains


def write_report(panel: pd.DataFrame, output_dir: Path) -> None:
    treatment = panel.loc[~panel["is_control"]].copy()
    summary = {
        "anchor_counts": panel["anchor_id"].value_counts().sort_index().to_dict(),
        "direction_family_counts": panel["direction_family"].value_counts().sort_index().to_dict(),
        "max_aggregate_error": float(panel["aggregate_max_abs_error"].max()),
        "max_phase_weight": float(panel["max_phase_weight"].max()),
        "phase_tv_max": float(panel["phase_tv"].max()),
        "phase_tv_min_nonzero": float(treatment["phase_tv"].min()),
        "realized_phase_0_fraction": ALPHA_0,
        "realized_phase_1_fraction": ALPHA_1,
        "rows": len(panel),
        "table9_anchor_rows_deferred": 32,
        "unique_policies": int(panel["policy_sha256"].nunique()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = f"""# 60M fixed-aggregate phase-order panel

This frozen panel uses only the **Uncheatable frontier** and **proportional** anchors. It does not inspect
or infer a Table-9 frontier anchor. The 32 rows that require that anchor remain deferred until native
Table-9 fit-panel completion is available.

- Rows: **{len(panel)}**
- Unique policy coordinates: **{panel["policy_sha256"].nunique()}**
- Exact anchor allocation: **70 rows per anchor**
- Maximum aggregate reconstruction error: **{panel["aggregate_max_abs_error"].max():.3e}**
- Realized phase fractions after block alignment: **{ALPHA_0:.9f} / {ALPHA_1:.9f}**
- Phase TV range among treatments: **{treatment["phase_tv"].min():.3f}-{treatment["phase_tv"].max():.3f}**
- Maximum phase weight: **{panel["max_phase_weight"].max():.3f}**

For each aggregate anchor \\(a\\), treatment rows use
\\(w_0=a+{ALPHA_1:.9f}d,\\;w_1=a-{ALPHA_0:.9f}d\\), so
\\({ALPHA_0:.9f}w_0+{ALPHA_1:.9f}w_1=a\\). Every treatment has an
antithetic order reversal under the same data and trainer seeds. Four tied controls per anchor cover the
four seed blocks. Two mechanistic contrasts are repeated under a second data seed.

## Composition

{panel["direction_family"].value_counts().rename_axis("family").reset_index(name="rows").to_markdown(index=False)}
"""
    (output_dir / "report.md").write_text(report)

    figure = px.scatter(
        treatment,
        x="phase_tv",
        y="max_phase_weight",
        color="anchor_id",
        symbol="direction_family",
        hover_data=[
            "candidate_id",
            "direction_id",
            "sign",
            "replicate_index",
            "phase_information_kl",
            "max_antithetic_phase_tv",
        ],
        color_discrete_sequence=["#d73027", "#1a9850"],
        title="Frozen 60M phase-order panel geometry",
        labels={"phase_tv": "Phase total variation", "max_phase_weight": "Maximum phase weight"},
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(
        output_dir / "phase_panel_geometry.html",
        include_plotlyjs=True,
        config=PLOT_CONFIG,
    )


def main() -> None:
    args = parse_args()
    panel, _ = build_panel()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "candidate_manifest.csv"
    panel.to_csv(panel_path, index=False)
    write_report(panel, args.output_dir)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    summary["manifest"] = str(panel_path)
    summary["manifest_sha256"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
