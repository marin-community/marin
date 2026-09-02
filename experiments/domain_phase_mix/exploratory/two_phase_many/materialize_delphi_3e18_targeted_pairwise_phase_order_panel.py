# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize targeted fixed-aggregate phase-order interventions at Delphi 3e18.

The panel isolates phase placement by exchanging mass between either two named
buckets or two predeclared groups while preserving the aggregate mixture. It
uses antithetic phase orders, same-seed tied controls, and second-seed repeats
of selected order contrasts so the order-effect noise floor is estimable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_3e18_aggressive_phase_asymmetry_panel as aggressive,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_3e18_frontier_phase_fiber_panel as fiber,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_targeted_pairwise_phase_order_20260724"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_targeted_pairwise_phase_order_20260724"
)

RUN_ID_BASE = 7_240_000
DATA_SEED_BASE = 7_241_000
N_SEED_BLOCKS = 8
DYAD_LOW_RADIUS_FRACTION = 0.35
DYAD_SINGLE_RADIUS_FRACTION = 0.55
DYAD_HIGH_RADIUS_FRACTION = 0.75
COMPOSITE_PHASE_TV_LEVELS = (0.05, 0.08)
EXPECTED_DYADS = 24
EXPECTED_DYAD_LEVELS = 38
EXPECTED_REPLICATE_LEVELS = 4
EXPECTED_COMPOSITE_LEVELS = 4
EXPECTED_TREATMENT_INSTANCES = EXPECTED_DYAD_LEVELS + EXPECTED_REPLICATE_LEVELS + EXPECTED_COMPOSITE_LEVELS
EXPECTED_CONTROL_ROWS = 2 * N_SEED_BLOCKS
EXPECTED_TREATMENT_ROWS = 4 * EXPECTED_TREATMENT_INSTANCES
EXPECTED_TOTAL_ROWS = EXPECTED_CONTROL_ROWS + EXPECTED_TREATMENT_ROWS
AGGREGATE_TOLERANCE = 2e-12
SIMPLEX_TOLERANCE = 2e-12
NOVELTY_TOLERANCE = 1e-10
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class DirectionSpec:
    direction_id: str
    contrast_family: str
    direction_label: str
    left_domains: tuple[str, ...]
    right_domains: tuple[str, ...]
    level_kind: str
    primary_hypothesis: str
    analysis_role: str
    replicate_high_level: bool = False


@dataclass(frozen=True)
class DirectionLevel:
    direction: DirectionSpec
    level_id: str
    level_value: float
    level_kind: str
    replicate_index: int
    seed_block: int


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    anchor_id: str
    anchor_run_name: str
    anchor_source_run_name: str
    contrast_family: str
    direction_id: str
    direction_label: str
    primary_hypothesis: str
    analysis_role: str
    left_domains: tuple[str, ...]
    right_domains: tuple[str, ...]
    sign: str
    later_side: str
    level_id: str
    level_kind: str
    level_value: float
    replicate_index: int
    seed_block: int
    data_seed: int
    trainer_seed: int
    common_symmetric_max_phase_tv: float
    target_phase_tv: float
    phase_0_weights: np.ndarray
    phase_1_weights: np.ndarray


def _quality_pair(topic: str, level_kind: str, *, replicate: bool = False) -> DirectionSpec:
    return DirectionSpec(
        direction_id=f"quality_{topic}",
        contrast_family="quality_pair",
        direction_label=f"{topic.replace('_', ' ')} high-quality versus low-quality",
        left_domains=(f"dolma3_cc/{topic}_high",),
        right_domains=(f"dolma3_cc/{topic}_low",),
        level_kind=level_kind,
        primary_hypothesis="quality_high_late",
        analysis_role="exploratory_pair",
        replicate_high_level=replicate,
    )


def _dyad(
    direction_id: str,
    contrast_family: str,
    left_domain: str,
    right_domain: str,
    level_kind: str,
    primary_hypothesis: str,
    *,
    replicate: bool = False,
) -> DirectionSpec:
    return DirectionSpec(
        direction_id=direction_id,
        contrast_family=contrast_family,
        direction_label=f"{left_domain} versus {right_domain}",
        left_domains=(left_domain,),
        right_domains=(right_domain,),
        level_kind=level_kind,
        primary_hypothesis=primary_hypothesis,
        analysis_role="primary_pair" if replicate else "exploratory_pair",
        replicate_high_level=replicate,
    )


def _direction_registry(domains: tuple[str, ...]) -> tuple[DirectionSpec, ...]:
    quality_pairs = (
        _quality_pair("science_math_and_technology", "dual", replicate=True),
        _quality_pair("finance_and_business", "dual"),
        _quality_pair("health", "dual"),
        _quality_pair("entertainment", "dual"),
        _quality_pair("literature", "single"),
        _quality_pair("education_and_jobs", "single"),
        _quality_pair("food_and_dining", "single"),
        _quality_pair("crime_and_law", "single"),
        _quality_pair("games", "single"),
    )
    semantic_pairs = (
        _dyad(
            "semantic_common_hq_finance",
            "semantic_pair",
            "dolmino_common_crawl_hq",
            "dolma3_cc/finance_and_business_high",
            "dual",
            "dolmino_late",
        ),
        _dyad(
            "semantic_olmocr_arxiv",
            "semantic_pair",
            "dolmino_olmocr_pdfs_hq",
            "dolma3_arxiv",
            "dual",
            "dolmino_late",
        ),
        _dyad(
            "semantic_fim_stack",
            "semantic_pair",
            "dolmino_stack_edu_fim",
            "dolma3_stack_edu",
            "dual",
            "code_late",
            replicate=True,
        ),
        _dyad(
            "semantic_synth_code_stack",
            "semantic_pair",
            "dolmino_synth_code",
            "dolma3_stack_edu",
            "dual",
            "code_late",
        ),
        _dyad(
            "semantic_synth_math_finemath",
            "semantic_pair",
            "dolmino_synth_math",
            "dolma3_finemath_3plus",
            "dual",
            "math_reasoning_late",
        ),
        _dyad(
            "semantic_synth_qa_literature",
            "semantic_pair",
            "dolmino_synth_qa",
            "dolma3_cc/literature_high",
            "dual",
            "qa_objective_reversal",
            replicate=True,
        ),
        _dyad(
            "semantic_stem_science",
            "semantic_pair",
            "dolmino_stem_heavy_crawl",
            "dolma3_cc/science_math_and_technology_high",
            "single",
            "math_reasoning_late",
        ),
        _dyad(
            "semantic_instruction_education",
            "semantic_pair",
            "dolmino_synth_instruction",
            "dolma3_cc/education_and_jobs_high",
            "single",
            "instruction_late",
        ),
    )
    robustness_pairs = (
        _dyad(
            "robust_common_hq_literature",
            "robustness_pair",
            "dolmino_common_crawl_hq",
            "dolma3_cc/literature_high",
            "dual",
            "broad_pretraining_early",
        ),
        _dyad(
            "robust_fim_common_hq",
            "robustness_pair",
            "dolmino_stack_edu_fim",
            "dolmino_common_crawl_hq",
            "dual",
            "code_late",
        ),
        _dyad(
            "robust_synth_code_common_hq",
            "robustness_pair",
            "dolmino_synth_code",
            "dolmino_common_crawl_hq",
            "dual",
            "code_late",
        ),
        _dyad(
            "robust_olmocr_common_hq",
            "robustness_pair",
            "dolmino_olmocr_pdfs_hq",
            "dolmino_common_crawl_hq",
            "single",
            "document_quality_late",
        ),
        _dyad(
            "robust_olmocr_wikipedia",
            "robustness_pair",
            "dolmino_olmocr_pdfs_hq",
            "dolma3_wikipedia",
            "single",
            "document_quality_late",
        ),
        _dyad(
            "robust_stack_common_hq",
            "robustness_pair",
            "dolma3_stack_edu",
            "dolmino_common_crawl_hq",
            "dual",
            "code_late",
        ),
        _dyad(
            "robust_synth_qa_common_hq",
            "robustness_pair",
            "dolmino_synth_qa",
            "dolmino_common_crawl_hq",
            "single",
            "qa_objective_reversal",
        ),
    )
    quality_high = tuple(domain for domain in domains if domain.startswith("dolma3_cc/") and domain.endswith("_high"))
    quality_low = tuple(domain for domain in domains if domain.startswith("dolma3_cc/") and domain.endswith("_low"))
    dolmino = tuple(domain for domain in domains if domain.startswith("dolmino_"))
    broad = tuple(domain for domain in domains if not domain.startswith("dolmino_"))
    composites = (
        DirectionSpec(
            direction_id="composite_quality_high_low",
            contrast_family="structured_composite",
            direction_label="all Common Crawl high-quality buckets versus matched low-quality buckets",
            left_domains=quality_high,
            right_domains=quality_low,
            level_kind="composite",
            primary_hypothesis="quality_high_late",
            analysis_role="primary_composite",
            replicate_high_level=True,
        ),
        DirectionSpec(
            direction_id="composite_dolmino_broad",
            contrast_family="structured_composite",
            direction_label="all Dolmino buckets versus all non-Dolmino buckets",
            left_domains=dolmino,
            right_domains=broad,
            level_kind="composite",
            primary_hypothesis="dolmino_late",
            analysis_role="secondary_composite",
        ),
    )
    registry = (*quality_pairs, *semantic_pairs, *robustness_pairs, *composites)
    unknown = {
        domain
        for direction in registry
        for domain in (*direction.left_domains, *direction.right_domains)
        if domain not in domains
    }
    if unknown:
        raise ValueError(f"Direction registry contains unknown domains: {sorted(unknown)}")
    if len(registry) != EXPECTED_DYADS + len(composites):
        raise ValueError(f"Direction count changed: {len(registry)}")
    return registry


def _load_frozen_anchors(domains: tuple[str, ...]) -> tuple[list[fiber.Anchor], pd.DataFrame]:
    """Load the two preregistered historical anchors by immutable hashes."""
    scores = fiber._one_phase_scores()
    anchors = []
    audit_rows = []
    for anchor_id, run_name, expected_mixture_sha256, expected_weight_vector_sha256 in fiber.ANCHORS:
        matching_positions = np.flatnonzero(scores["wandb_run_base"].eq(run_name).to_numpy())
        if len(matching_positions) != 1:
            raise ValueError(f"Expected one source row for {run_name}, found {len(matching_positions)}")
        row_position = int(matching_positions[0])
        row = scores.iloc[row_position]
        if row["mixture_sha256"] != expected_mixture_sha256:
            raise ValueError(f"Mixture hash changed for {run_name}: {row['mixture_sha256']}")
        phase_0 = json.loads(row["phase_0_weights_json"])
        phase_1 = json.loads(row["phase_1_weights_json"])
        early = np.asarray([float(phase_0[domain]) for domain in domains])
        late = np.asarray([float(phase_1[domain]) for domain in domains])
        if np.max(np.abs(early - late)) > SIMPLEX_TOLERANCE:
            raise ValueError(f"Anchor {run_name} is not phase tied")
        weight_vector_sha256 = fiber._weight_vector_sha256(domains, early)
        if weight_vector_sha256 != expected_weight_vector_sha256:
            raise ValueError(f"Weight-vector hash changed for {run_name}: {weight_vector_sha256}")
        uncheatable_rank = float(scores["uncheatable_bpb"].rank(method="min").iloc[row_position])
        table9_rank = float(scores["table9_macro_bpb"].rank(method="min").iloc[row_position])
        anchor = fiber.Anchor(
            anchor_id=anchor_id,
            run_name=run_name,
            source_run_name=run_name,
            mixture_sha256=expected_mixture_sha256,
            weight_vector_sha256=weight_vector_sha256,
            weights=early,
            uncheatable_3e18=float(row["uncheatable_bpb"]),
            table9_3e18=float(row["table9_macro_bpb"]),
            uncheatable_one_phase_rank=uncheatable_rank,
            table9_one_phase_rank=table9_rank,
            one_phase_policy_count=int(scores["mixture_sha256"].nunique()),
        )
        anchors.append(anchor)
        audit_rows.append(
            {
                "anchor_id": anchor_id,
                "source_run_name": run_name,
                "mixture_sha256": expected_mixture_sha256,
                "weight_vector_sha256": weight_vector_sha256,
                "source_uncheatable_bpb": anchor.uncheatable_3e18,
                "source_table9_macro_bpb": anchor.table9_3e18,
                "current_uncheatable_row_rank": uncheatable_rank,
                "current_table9_row_rank": table9_rank,
                "one_phase_policy_count": anchor.one_phase_policy_count,
                "min_weight": float(early.min()),
                "max_weight": float(early.max()),
            }
        )
    return anchors, pd.DataFrame(audit_rows)


def _direction_vector(anchor: np.ndarray, direction: DirectionSpec, domains: tuple[str, ...]) -> np.ndarray:
    domain_index = {domain: index for index, domain in enumerate(domains)}
    left_indices = np.asarray([domain_index[domain] for domain in direction.left_domains])
    right_indices = np.asarray([domain_index[domain] for domain in direction.right_domains])
    if set(left_indices) & set(right_indices):
        raise ValueError(f"{direction.direction_id} has overlapping sides")
    vector = np.zeros_like(anchor)
    if len(left_indices) == 1 and len(right_indices) == 1:
        vector[left_indices[0]] = 1.0
        vector[right_indices[0]] = -1.0
    else:
        left_mass = float(anchor[left_indices].sum())
        right_mass = float(anchor[right_indices].sum())
        vector[left_indices] = anchor[left_indices] / left_mass
        vector[right_indices] = -anchor[right_indices] / right_mass
    if abs(float(vector.sum())) > SIMPLEX_TOLERANCE:
        raise ValueError(f"{direction.direction_id} does not conserve mass")
    if abs(float(0.5 * np.abs(vector).sum()) - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError(f"{direction.direction_id} does not have unit phase TV")
    return vector


def _maximum_signed_phase_tv(
    anchor: np.ndarray,
    direction: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    return min(
        aggressive._maximum_phase_tv(anchor, direction, alpha0, alpha1),
        aggressive._maximum_phase_tv(anchor, -direction, alpha0, alpha1),
    )


def _common_symmetric_limits(
    anchors: list[fiber.Anchor],
    registry: tuple[DirectionSpec, ...],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> dict[str, float]:
    limits = {}
    for direction in registry:
        limits[direction.direction_id] = min(
            _maximum_signed_phase_tv(
                anchor.weights,
                _direction_vector(anchor.weights, direction, domains),
                alpha0,
                alpha1,
            )
            for anchor in anchors
        )
    return limits


def _direction_levels(
    registry: tuple[DirectionSpec, ...],
    common_limits: dict[str, float],
) -> list[DirectionLevel]:
    levels = []
    for direction_index, direction in enumerate(registry):
        seed_block = direction_index % N_SEED_BLOCKS
        if direction.level_kind == "dual":
            base_levels = (
                ("r035", DYAD_LOW_RADIUS_FRACTION, "radius_fraction"),
                ("r075", DYAD_HIGH_RADIUS_FRACTION, "radius_fraction"),
            )
        elif direction.level_kind == "single":
            base_levels = (("r055", DYAD_SINGLE_RADIUS_FRACTION, "radius_fraction"),)
        elif direction.level_kind == "composite":
            base_levels = tuple(
                (f"tv{round(100 * phase_tv):02d}", phase_tv, "absolute_phase_tv")
                for phase_tv in COMPOSITE_PHASE_TV_LEVELS
            )
        else:
            raise ValueError(f"Unknown level kind {direction.level_kind}")
        for level_id, level_value, level_kind in base_levels:
            levels.append(
                DirectionLevel(
                    direction=direction,
                    level_id=level_id,
                    level_value=level_value,
                    level_kind=level_kind,
                    replicate_index=0,
                    seed_block=seed_block,
                )
            )
        if direction.replicate_high_level:
            level_id, level_value, level_kind = base_levels[-1]
            levels.append(
                DirectionLevel(
                    direction=direction,
                    level_id=level_id,
                    level_value=level_value,
                    level_kind=level_kind,
                    replicate_index=1,
                    seed_block=(seed_block + N_SEED_BLOCKS // 2) % N_SEED_BLOCKS,
                )
            )
    dyad_level_count = sum(
        level.direction.contrast_family != "structured_composite" and level.replicate_index == 0 for level in levels
    )
    if dyad_level_count != EXPECTED_DYAD_LEVELS:
        raise ValueError("Base dyad-level count changed")
    if sum(level.replicate_index == 1 for level in levels) != EXPECTED_REPLICATE_LEVELS:
        raise ValueError("Order-contrast replicate count changed")
    composite_level_count = sum(
        level.direction.contrast_family == "structured_composite" and level.replicate_index == 0 for level in levels
    )
    if composite_level_count != EXPECTED_COMPOSITE_LEVELS:
        raise ValueError("Structured-composite level count changed")
    for level in levels:
        target_tv = (
            level.level_value * common_limits[level.direction.direction_id]
            if level.level_kind == "radius_fraction"
            else level.level_value
        )
        if target_tv >= common_limits[level.direction.direction_id] - SIMPLEX_TOLERANCE:
            raise ValueError(f"{level.direction.direction_id}/{level.level_id} reaches its feasibility boundary")
    return levels


def build_candidates(
    anchors: list[fiber.Anchor],
    registry: tuple[DirectionSpec, ...],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> tuple[list[Candidate], list[DirectionLevel], dict[str, float]]:
    common_limits = _common_symmetric_limits(anchors, registry, domains, alpha0, alpha1)
    levels = _direction_levels(registry, common_limits)
    candidates = []
    for anchor_index, anchor in enumerate(anchors):
        for seed_block in range(N_SEED_BLOCKS):
            candidates.append(
                Candidate(
                    candidate_id=f"pairdoe_a{anchor_index}_center_s{seed_block:02d}",
                    anchor_id=anchor.anchor_id,
                    anchor_run_name=anchor.run_name,
                    anchor_source_run_name=anchor.source_run_name,
                    contrast_family="center_control",
                    direction_id="center",
                    direction_label="phase-tied frontier control",
                    primary_hypothesis="noise_control",
                    analysis_role="same_seed_control",
                    left_domains=(),
                    right_domains=(),
                    sign="center",
                    later_side="tied",
                    level_id="center",
                    level_kind="center",
                    level_value=0.0,
                    replicate_index=0,
                    seed_block=seed_block,
                    data_seed=DATA_SEED_BASE + seed_block,
                    trainer_seed=0,
                    common_symmetric_max_phase_tv=0.0,
                    target_phase_tv=0.0,
                    phase_0_weights=anchor.weights.copy(),
                    phase_1_weights=anchor.weights.copy(),
                )
            )
        for level in levels:
            direction = _direction_vector(anchor.weights, level.direction, domains)
            target_tv = (
                level.level_value * common_limits[level.direction.direction_id]
                if level.level_kind == "radius_fraction"
                else level.level_value
            )
            for sign, signed_direction, later_side in (
                ("plus", direction, "left"),
                ("minus", -direction, "right"),
            ):
                phase_0, phase_1 = aggressive._phase_weights(
                    anchor.weights,
                    target_tv * signed_direction,
                    alpha0,
                    alpha1,
                )
                candidates.append(
                    Candidate(
                        candidate_id=(
                            f"pairdoe_a{anchor_index}_{level.direction.direction_id}_{level.level_id}"
                            f"_rep{level.replicate_index}_{sign}"
                        ),
                        anchor_id=anchor.anchor_id,
                        anchor_run_name=anchor.run_name,
                        anchor_source_run_name=anchor.source_run_name,
                        contrast_family=level.direction.contrast_family,
                        direction_id=level.direction.direction_id,
                        direction_label=level.direction.direction_label,
                        primary_hypothesis=level.direction.primary_hypothesis,
                        analysis_role=(
                            "order_noise_replicate" if level.replicate_index else level.direction.analysis_role
                        ),
                        left_domains=level.direction.left_domains,
                        right_domains=level.direction.right_domains,
                        sign=sign,
                        later_side=later_side,
                        level_id=level.level_id,
                        level_kind=level.level_kind,
                        level_value=level.level_value,
                        replicate_index=level.replicate_index,
                        seed_block=level.seed_block,
                        data_seed=DATA_SEED_BASE + level.seed_block,
                        trainer_seed=0,
                        common_symmetric_max_phase_tv=common_limits[level.direction.direction_id],
                        target_phase_tv=target_tv,
                        phase_0_weights=phase_0,
                        phase_1_weights=phase_1,
                    )
                )
    if len(candidates) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} candidates, found {len(candidates)}")
    return candidates, levels, common_limits


def _policy_sha256(domains: tuple[str, ...], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def validate_candidates(
    candidates: list[Candidate],
    anchors: list[fiber.Anchor],
    registry: tuple[DirectionSpec, ...],
    levels: list[DirectionLevel],
    common_limits: dict[str, float],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    anchor_lookup = {anchor.anchor_id: anchor for anchor in anchors}
    weights = np.asarray([[candidate.phase_0_weights, candidate.phase_1_weights] for candidate in candidates])
    if float(weights.min()) < -SIMPLEX_TOLERANCE:
        index = np.unravel_index(np.argmin(weights), weights.shape)
        raise ValueError(f"Panel has negative phase weight {weights[index]} at {index}")
    if np.max(np.abs(weights.sum(axis=2) - 1.0)) > SIMPLEX_TOLERANCE:
        raise ValueError("A phase mixture does not sum to one")

    fit_weights = fiber._existing_fit_weights(domains)
    heldout_weights = aggressive._heldout_weights(domains)
    references = np.concatenate([fit_weights, heldout_weights], axis=0)
    min_fit_tv = fiber._weighted_policy_tv(weights, fit_weights, alpha0, alpha1).min(axis=1)
    min_prior_tv = fiber._weighted_policy_tv(weights, references, alpha0, alpha1).min(axis=1)
    dolmino_mask = np.asarray([domain.startswith("dolmino_") for domain in domains])
    rows = []
    for index, candidate in enumerate(candidates):
        anchor = anchor_lookup[candidate.anchor_id]
        aggregate = alpha0 * candidate.phase_0_weights + alpha1 * candidate.phase_1_weights
        aggregate_error = float(np.max(np.abs(aggregate - anchor.weights)))
        if aggregate_error > AGGREGATE_TOLERANCE:
            raise ValueError(f"{candidate.candidate_id} aggregate error is {aggregate_error}")
        phase_tv = float(0.5 * np.abs(candidate.phase_1_weights - candidate.phase_0_weights).sum())
        if abs(phase_tv - candidate.target_phase_tv) > 2e-11:
            raise ValueError(
                f"{candidate.candidate_id} phase TV {phase_tv} does not match target {candidate.target_phase_tv}"
            )
        rows.append(
            {
                "run_order": index,
                "run_id": RUN_ID_BASE + index,
                "candidate_id": candidate.candidate_id,
                "policy_sha256": _policy_sha256(domains, weights[index]),
                "anchor_id": candidate.anchor_id,
                "anchor_run_name": candidate.anchor_run_name,
                "anchor_source_run_name": candidate.anchor_source_run_name,
                "contrast_family": candidate.contrast_family,
                "direction_id": candidate.direction_id,
                "direction_label": candidate.direction_label,
                "primary_hypothesis": candidate.primary_hypothesis,
                "analysis_role": candidate.analysis_role,
                "left_domains_json": json.dumps(candidate.left_domains),
                "right_domains_json": json.dumps(candidate.right_domains),
                "sign": candidate.sign,
                "later_side": candidate.later_side,
                "level_id": candidate.level_id,
                "level_kind": candidate.level_kind,
                "level_value": candidate.level_value,
                "replicate_index": candidate.replicate_index,
                "seed_block": candidate.seed_block,
                "data_seed": candidate.data_seed,
                "trainer_seed": candidate.trainer_seed,
                "common_symmetric_max_phase_tv": candidate.common_symmetric_max_phase_tv,
                "target_phase_tv": candidate.target_phase_tv,
                "phase_tv": phase_tv,
                "phase_information_kl": max(
                    0.0,
                    fiber._phase_information_kl(
                        candidate.phase_0_weights,
                        candidate.phase_1_weights,
                        aggregate,
                        alpha0,
                        alpha1,
                    ),
                ),
                "aggregate_max_abs_error": aggregate_error,
                "phase_0_dolmino_share": float(candidate.phase_0_weights[dolmino_mask].sum()),
                "phase_1_dolmino_share": float(candidate.phase_1_weights[dolmino_mask].sum()),
                "max_weight": float(weights[index].max()),
                "min_weight": float(weights[index].min()),
                "min_fit_policy_tv": float(min_fit_tv[index]),
                "min_prior_policy_tv": float(min_prior_tv[index]),
            }
        )
    manifest = pd.DataFrame(rows)

    if manifest["candidate_id"].nunique() != EXPECTED_TOTAL_ROWS:
        raise ValueError("Candidate IDs are not unique")
    expected_anchor_counts = {anchor.anchor_id: EXPECTED_TOTAL_ROWS // 2 for anchor in anchors}
    if manifest["anchor_id"].value_counts().to_dict() != expected_anchor_counts:
        raise ValueError(f"Anchor counts changed: {manifest['anchor_id'].value_counts().to_dict()}")
    if manifest["sign"].value_counts().to_dict() != {
        "plus": EXPECTED_TREATMENT_ROWS // 2,
        "minus": EXPECTED_TREATMENT_ROWS // 2,
        "center": EXPECTED_CONTROL_ROWS,
    }:
        raise ValueError(f"Sign counts changed: {manifest['sign'].value_counts().to_dict()}")

    treatment_mask = ~manifest["contrast_family"].eq("center_control")
    treatment_rows = manifest.loc[treatment_mask]
    policy_counts = treatment_rows["policy_sha256"].value_counts()
    repeated = policy_counts.loc[policy_counts.gt(1)]
    if len(repeated) != 4 * EXPECTED_REPLICATE_LEVELS or set(repeated) != {2}:
        raise ValueError(f"Intentional treatment repeats changed: {repeated.to_dict()}")
    for policy_sha256 in repeated.index:
        repeated_rows = treatment_rows.loc[treatment_rows["policy_sha256"].eq(policy_sha256)]
        if set(repeated_rows["replicate_index"]) != {0, 1}:
            raise ValueError(f"Repeated policy {policy_sha256} lacks base and second-seed observations")
        if repeated_rows["data_seed"].nunique() != 2:
            raise ValueError(f"Repeated policy {policy_sha256} does not have independent seeds")
    if float(treatment_rows["min_prior_policy_tv"].min()) <= NOVELTY_TOLERANCE:
        duplicate = treatment_rows.sort_values("min_prior_policy_tv").iloc[0]
        raise ValueError(
            f"Candidate {duplicate['candidate_id']} aliases prior work at TV {duplicate['min_prior_policy_tv']}"
        )

    control_keys = set(
        zip(
            manifest.loc[~treatment_mask, "anchor_id"],
            manifest.loc[~treatment_mask, "seed_block"],
            strict=True,
        )
    )
    for key, group in treatment_rows.groupby(
        ["anchor_id", "direction_id", "level_id", "replicate_index"],
        sort=False,
    ):
        if len(group) != 2 or set(group["sign"]) != {"plus", "minus"}:
            raise ValueError(f"{key} is not an antithetic pair")
        if group["data_seed"].nunique() != 1:
            raise ValueError(f"{key} does not share one data seed")
        if (str(key[0]), int(group["seed_block"].iloc[0])) not in control_keys:
            raise ValueError(f"{key} lacks a same-seed tied control")
        pair = weights[group.index.to_numpy(dtype=int)]
        anchor = anchor_lookup[str(key[0])].weights
        if np.max(np.abs(pair.mean(axis=0) - np.stack([anchor, anchor]))) > AGGREGATE_TOLERANCE:
            raise ValueError(f"{key} is not centered on its tied anchor")
    for key, group in treatment_rows.loc[treatment_rows["replicate_index"].eq(0)].groupby(
        ["direction_id", "anchor_id"],
        sort=False,
    ):
        if group["level_id"].nunique() > 1 and group["seed_block"].nunique() != 1:
            raise ValueError(f"{key} levels do not share one seed block")

    expected_triads = (
        frozenset(("dolmino_synth_qa", "dolmino_common_crawl_hq", "dolma3_cc/literature_high")),
        frozenset(("dolmino_stack_edu_fim", "dolma3_stack_edu", "dolmino_common_crawl_hq")),
        frozenset(("dolmino_synth_code", "dolma3_stack_edu", "dolmino_common_crawl_hq")),
    )
    dyad_edges = {
        frozenset((*direction.left_domains, *direction.right_domains))
        for direction in registry
        if len(direction.left_domains) == len(direction.right_domains) == 1
    }
    for triad in expected_triads:
        if sum(edge.issubset(triad) for edge in dyad_edges) != 3:
            raise ValueError(f"Additivity triad is incomplete: {sorted(triad)}")

    registry_rows = []
    for direction in registry:
        direction_levels = [level for level in levels if level.direction.direction_id == direction.direction_id]
        registry_rows.append(
            {
                "direction_id": direction.direction_id,
                "contrast_family": direction.contrast_family,
                "direction_label": direction.direction_label,
                "left_domains_json": json.dumps(direction.left_domains),
                "right_domains_json": json.dumps(direction.right_domains),
                "level_kind": direction.level_kind,
                "primary_hypothesis": direction.primary_hypothesis,
                "analysis_role": direction.analysis_role,
                "replicate_high_level": direction.replicate_high_level,
                "common_symmetric_max_phase_tv": common_limits[direction.direction_id],
                "level_ids_json": json.dumps(
                    [level.level_id for level in direction_levels if level.replicate_index == 0]
                ),
                "base_seed_block": next(level.seed_block for level in direction_levels if level.replicate_index == 0),
                "replicate_seed_block": next(
                    (level.seed_block for level in direction_levels if level.replicate_index == 1),
                    None,
                ),
                "treatment_observations": len(
                    treatment_rows.loc[treatment_rows["direction_id"].eq(direction.direction_id)]
                ),
            }
        )
    registry_frame = pd.DataFrame(registry_rows)
    family_counts = manifest["contrast_family"].value_counts().to_dict()
    seed_counts = manifest.groupby(["anchor_id", "seed_block"]).size()
    summary: dict[str, object] = {
        "panel_rows": len(manifest),
        "control_observations": int((~treatment_mask).sum()),
        "treatment_observations": int(treatment_mask.sum()),
        "unique_treatment_policies": int(treatment_rows["policy_sha256"].nunique()),
        "repeated_treatment_policies": len(repeated),
        "direction_count": len(registry),
        "dyad_count": sum(direction.contrast_family != "structured_composite" for direction in registry),
        "dyad_level_count": EXPECTED_DYAD_LEVELS,
        "replicated_order_contrast_count": EXPECTED_REPLICATE_LEVELS,
        "structured_composite_level_count": EXPECTED_COMPOSITE_LEVELS,
        "rows_per_anchor": manifest.groupby("anchor_id").size().to_dict(),
        "rows_per_contrast_family": family_counts,
        "rows_per_seed_block_range": [int(seed_counts.min()), int(seed_counts.max())],
        "realized_phase_fractions": {"phase_0": alpha0, "phase_1": alpha1},
        "max_aggregate_error": float(manifest["aggregate_max_abs_error"].max()),
        "min_phase_weight": float(weights.min()),
        "max_phase_weight": float(weights.max()),
        "phase_tv_range": [float(manifest["phase_tv"].min()), float(manifest["phase_tv"].max())],
        "minimum_treatment_prior_policy_tv": float(treatment_rows["min_prior_policy_tv"].min()),
        "maximum_treatment_prior_policy_tv": float(treatment_rows["min_prior_policy_tv"].max()),
        "native_table9_scheduled": True,
        "analysis_uses_prior_panel_outcomes": True,
        "target_outcomes_used_to_construct_candidates": False,
    }
    return manifest, weights, registry_frame, summary


def render_diagnostics(manifest: pd.DataFrame, registry: pd.DataFrame, output_dir: Path) -> None:
    treatments = manifest.loc[~manifest["contrast_family"].eq("center_control")].copy()
    geometry = px.scatter(
        treatments,
        x="phase_tv",
        y="phase_information_kl",
        color="max_weight",
        symbol="contrast_family",
        facet_col="anchor_id",
        hover_name="candidate_id",
        hover_data=[
            "direction_label",
            "primary_hypothesis",
            "sign",
            "level_id",
            "replicate_index",
            "seed_block",
            "min_prior_policy_tv",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Targeted phase-order panel: geometry and information",
    )
    geometry.update_layout(width=1500, height=780, margin={"l": 70, "r": 150, "t": 130, "b": 70})
    geometry.write_html(output_dir / "panel_geometry.html", include_plotlyjs=True, config=PLOT_CONFIG)

    power = px.bar(
        registry.sort_values("common_symmetric_max_phase_tv"),
        x="common_symmetric_max_phase_tv",
        y="direction_id",
        color="contrast_family",
        orientation="h",
        hover_data=["direction_label", "primary_hypothesis", "level_ids_json", "treatment_observations"],
        title="Maximum cross-anchor antithetic phase TV by intervention",
    )
    power.update_layout(width=1200, height=1000, margin={"l": 280, "r": 80, "t": 100, "b": 70})
    power.write_html(output_dir / "direction_power_audit.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    registry: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Delphi 3e18 targeted pairwise phase-order panel",
        "",
        "## Scientific question",
        "",
        (
            "Which bucket-ordering heuristics survive controlled interventions that change only phase placement, "
            "not total token allocation? This panel replaces focal-versus-rest contrasts with minimal two-bucket "
            "exchanges and two predeclared family-level exchanges."
        ),
        "",
        "For tied aggregate $a$, phase fractions $\\alpha_0,\\alpha_1$, and zero-sum contrast $d$,",
        "",
        "$$p^{(0)}=a-\\alpha_1d,\\qquad p^{(1)}=a+\\alpha_0d,$$",
        "",
        "so $\\alpha_0p^{(0)}+\\alpha_1p^{(1)}=a$. The antithetic pair uses $d$ and $-d$ at one seed. Define",
        "",
        "$$O=\\frac{L(+d)-L(-d)}{2},\\qquad C=\\frac{L(+d)+L(-d)}{2}-L(0).$$",
        "",
        "$O$ is the odd phase-order contrast. $C$ is the even asymmetry cost or curvature check.",
        "",
        "## Frozen 200-run design",
        "",
        f"- {EXPECTED_CONTROL_ROWS} phase-tied controls: one for each of {N_SEED_BLOCKS} seeds at each anchor.",
        f"- {EXPECTED_DYADS} named dyads and {EXPECTED_DYAD_LEVELS} base dyad-level instances.",
        f"- {EXPECTED_REPLICATE_LEVELS} selected order contrasts repeated at a second seed to estimate $\\sigma(O)$.",
        (
            f"- Two structured composites at phase TV {COMPOSITE_PHASE_TV_LEVELS[0]:.2f} and "
            f"{COMPOSITE_PHASE_TV_LEVELS[1]:.2f}: quality-high versus quality-low, and Dolmino versus non-Dolmino."
        ),
        f"- {EXPECTED_TREATMENT_ROWS} treatment observations and {EXPECTED_TOTAL_ROWS} total checkpoints.",
        "- Both levels of a dual direction and both antithetic signs share one seed block.",
        "- Every treatment shares its seed with a tied control; each direct replicate uses a different seed block.",
        "- The direction registry records the frozen base and replicate seed-block assignments.",
        "- Every checkpoint receives Uncheatable and Marin-native Table-9 BPB evaluation.",
        "- Target outcomes from this panel are sealed until the design, source hash, and analysis are recorded.",
        "",
        "## Primary tests",
        "",
        "1. Quality-high-late: structured high-versus-low Common Crawl composite.",
        "2. Code-late: FIM-versus-Stack and synthetic-code-versus-Stack dyads.",
        "3. QA objective reversal: QA-versus-literature dyad and the QA/Common-Crawl/literature additivity loop.",
        "",
        "All-Dolmino-late is a preregistered secondary falsification. Remaining dyads are exploratory. Five small "
        "quality dyads are retained for topic coverage; an unresolved result is not evidence of no effect.",
        "",
        "## Frozen inference",
        "",
        (
            "- Estimate objective-specific $\\sigma(O)$ from second-seed repeats of quality science, FIM versus "
            "Stack, QA versus literature, and the TV=0.08 quality composite."
        ),
        (
            "- Cross-check that estimate with dual-level consistency "
            "$O(0.75)/0.75-O(0.35)/0.35$ across all dual dyads; do not silently assume homoscedasticity."
        ),
        "- Call an individual contrast resolved only when its uncertainty clears the preregistered threshold.",
        "- Apply Benjamini-Yekutieli FDR correction to dependent exploratory tests.",
        "- Report each anchor and objective separately before any pooled summary.",
        "- Test additivity on the three complete dyad triangles in per-TV gradient space.",
        "- Do not report selected minima or use this panel to optimize a policy.",
        "",
        "## Anchor audit",
        "",
        anchor_audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Geometry and novelty audit",
        "",
        f"- Maximum aggregate error: {summary['max_aggregate_error']:.3e}.",
        f"- Phase TV range: {summary['phase_tv_range']}.",
        f"- Minimum phase weight: {summary['min_phase_weight']:.6g}.",
        f"- Maximum phase weight: {summary['max_phase_weight']:.6g}.",
        f"- Unique treatment policies: {summary['unique_treatment_policies']}.",
        f"- Repeated treatment policies: {summary['repeated_treatment_policies']} (all intentional second seeds).",
        (
            "- Minimum treatment distance to prior fit or heldout policy: "
            f"{summary['minimum_treatment_prior_policy_tv']:.6g}."
        ),
        "",
        "## Direction registry",
        "",
        registry.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Scope",
        "",
        (
            "Claims are local to the two frozen Delphi 3e18 frontier aggregates and the 80/20 WSD schedule. "
            "The panel estimates phase placement at fixed aggregate exposure; it does not identify aggregate-mixture "
            "quality, arbitrary curricula, or repetition effects caused by changing total bucket exposure."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains = tuple(augmented.DOMAIN_NAMES)
    alpha0, alpha1 = fiber._realized_phase_fractions()
    anchors, anchor_audit = _load_frozen_anchors(domains)
    registry = _direction_registry(domains)
    candidates, levels, common_limits = build_candidates(anchors, registry, domains, alpha0, alpha1)
    manifest, weights, registry_frame, summary = validate_candidates(
        candidates,
        anchors,
        registry,
        levels,
        common_limits,
        domains,
        alpha0,
        alpha1,
    )
    manifest_path = args.output_dir / "candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    anchor_audit.to_csv(args.output_dir / "anchor_audit.csv", index=False)
    registry_frame.to_csv(args.output_dir / "direction_registry.csv", index=False)
    fiber.write_long_weights(args.output_dir, manifest, weights, domains)
    source_path, source_sha256 = fiber.write_launcher_source_panel(args.output_dir, manifest, weights, domains)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary.update(
        {
            "candidate_manifest_sha256": manifest_sha256,
            "launcher_source_panel_sha256": source_sha256,
            "gcs_launcher_source_panel": f"{args.gcs_output_dir}/source/launcher_source_panel-{source_sha256[:16]}.csv",
            "gcs_candidate_manifest": f"{args.gcs_output_dir}/source/candidate_manifest-{manifest_sha256[:16]}.csv",
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_diagnostics(manifest, registry_frame, args.output_dir)
    write_report(args.output_dir, manifest, registry_frame, anchor_audit, summary)
    if args.upload:
        fiber.upload_artifact(source_path, str(summary["gcs_launcher_source_panel"]))
        fiber.upload_artifact(manifest_path, str(summary["gcs_candidate_manifest"]))
        for name in (
            "summary.json",
            "report.md",
            "phase_weights.csv",
            "anchor_audit.csv",
            "direction_registry.csv",
        ):
            fiber.upload_artifact(args.output_dir / name, f"{args.gcs_output_dir}/source/{name}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
