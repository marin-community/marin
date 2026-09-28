# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize large fixed-aggregate WSD annealing contrasts around 3e18 frontier anchors.

Unlike the earlier boundary-normalized isotropic panel, this design controls the
total-variation distance between phase mixtures directly. It combines balanced
random domain partitions, antithetic phase orders, and conventional curricula
that move curated or Dolmino data into the boundary-aligned low-LR final phase
while retaining broad-data replay.
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
    materialize_delphi_3e18_frontier_phase_fiber_panel as fiber,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_3e18_frontier_random_phase_population_panel as random_population,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "delphi_3e18_aggressive_phase_asymmetry_20260722"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_aggressive_phase_asymmetry_20260722"
)

N_RANDOM_PARTITIONS = 16
N_SEED_BLOCKS = 16
TV_LEVELS = (0.10, 0.25, 0.50)
LATE_DOLMINO_SHARES = (0.75, 0.90, 1.00)
DOLMINO_CONTINUUM_REPLICATES = 3
DOLMINO_REPEAT_SEED_BLOCKS = (14, 15)
RUN_ID_BASE = 7_223_000
DATA_SEED_BASE = 7_224_000
PARTITION_SEED = 20_260_722
AGGREGATE_TOLERANCE = 2e-12
SIMPLEX_TOLERANCE = 2e-12
NOVELTY_TOLERANCE = 1e-10
EXPECTED_ROWS_PER_ANCHOR = (
    N_SEED_BLOCKS
    + N_RANDOM_PARTITIONS * 2 * len(TV_LEVELS)
    + 8 * len(TV_LEVELS)
    + DOLMINO_CONTINUUM_REPLICATES * len(LATE_DOLMINO_SHARES)
)
EXPECTED_TOTAL_ROWS = 2 * EXPECTED_ROWS_PER_ANCHOR
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

HANDCRAFTED_RECIPIENTS: dict[str, tuple[str, ...]] = {
    "dolmino_all": tuple(domain for domain in augmented.DOMAIN_NAMES if domain.startswith("dolmino_")),
    "synthetic_all": (
        "dolmino_synth_code",
        "dolmino_synth_instruction",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
    "code": (
        "dolma3_cc/electronics_and_hardware_high",
        "dolma3_cc/electronics_and_hardware_low",
        "dolma3_stack_edu",
        "dolmino_stack_edu_fim",
        "dolmino_synth_code",
    ),
    "math_reasoning": (
        "dolma3_cc/science_math_and_technology_high",
        "dolma3_cc/science_math_and_technology_low",
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
    "knowledge_documents": (
        "dolma3_cc/history_and_geography_high",
        "dolma3_cc/history_and_geography_low",
        "dolma3_cc/literature_high",
        "dolma3_cc/literature_low",
        "dolma3_arxiv",
        "dolma3_wikipedia",
        "dolmino_olmocr_pdfs_hq",
    ),
    "instruction_qa": (
        "dolmino_olmocr_pdfs_hq",
        "dolmino_synth_instruction",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
    "premium_nonweb": (
        "dolma3_stack_edu",
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolma3_wikipedia",
        "dolmino_olmocr_pdfs_hq",
        "dolmino_stack_edu_fim",
        "dolmino_synth_code",
        "dolmino_synth_instruction",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
    "education_science": (
        "dolma3_cc/education_and_jobs_high",
        "dolma3_cc/science_math_and_technology_high",
        "dolma3_stack_edu",
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolmino_stack_edu_fim",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    anchor_id: str
    anchor_run_name: str
    anchor_source_run_name: str
    contrast_family: str
    direction_id: str
    direction_label: str
    sign: str
    replicate_index: int
    seed_block: int
    data_seed: int
    trainer_seed: int
    target_phase_tv: float
    recipient_domains: tuple[str, ...]
    phase_0_weights: np.ndarray
    phase_1_weights: np.ndarray


def _phase_weights(
    anchor: np.ndarray,
    contrast: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(contrast.sum())) > SIMPLEX_TOLERANCE:
        raise ValueError("A phase contrast does not conserve mixture mass")
    phase_0 = anchor - alpha1 * contrast
    phase_1 = anchor + alpha0 * contrast
    return phase_0, phase_1


def _group_transfer_direction(anchor: np.ndarray, recipient_mask: np.ndarray) -> np.ndarray:
    recipient_mass = float(anchor[recipient_mask].sum())
    donor_mass = float(anchor[~recipient_mask].sum())
    if recipient_mass <= 0 or donor_mass <= 0:
        raise ValueError("A group-transfer direction needs nonempty recipient and donor mass")
    direction = np.empty_like(anchor)
    direction[recipient_mask] = anchor[recipient_mask] / recipient_mass
    direction[~recipient_mask] = -anchor[~recipient_mask] / donor_mass
    if abs(float(direction.sum())) > SIMPLEX_TOLERANCE:
        raise ValueError("A group-transfer direction does not conserve mixture mass")
    if abs(float(0.5 * np.abs(direction).sum()) - 1.0) > SIMPLEX_TOLERANCE:
        raise ValueError("A group-transfer direction does not have unit phase TV")
    return direction


def _maximum_phase_tv(
    anchor: np.ndarray,
    direction: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> float:
    limits = []
    positive = direction > 0
    negative = direction < 0
    limits.extend((anchor[positive] / (alpha1 * direction[positive])).tolist())
    limits.extend((anchor[negative] / (-alpha0 * direction[negative])).tolist())
    return float(min(limits))


def _balanced_partitions(
    anchors: list[fiber.Anchor],
    domains: tuple[str, ...],
) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(PARTITION_SEED)
    partitions = []
    seen: set[tuple[int, ...]] = set()
    while len(partitions) < N_RANDOM_PARTITIONS:
        mask = rng.random(len(domains)) < 0.5
        if not mask[0]:
            mask = ~mask
        key = tuple(np.flatnonzero(mask).tolist())
        if key in seen:
            continue
        masses = [float(anchor.weights[mask].sum()) for anchor in anchors]
        if not all(0.40 <= mass <= 0.60 for mass in masses):
            continue
        seen.add(key)
        partitions.append(tuple(domain for domain, selected in zip(domains, mask, strict=True) if selected))
    return partitions


def _candidate(
    *,
    anchor: fiber.Anchor,
    anchor_index: int,
    contrast_family: str,
    direction_id: str,
    direction_label: str,
    sign: str,
    replicate_index: int,
    seed_block: int,
    target_phase_tv: float,
    recipient_domains: tuple[str, ...],
    phase_0_weights: np.ndarray,
    phase_1_weights: np.ndarray,
) -> Candidate:
    tv_label = f"tv{round(100 * target_phase_tv):03d}"
    seed_label = f"_seed{seed_block:02d}" if contrast_family == "center_control" else ""
    replicate_label = f"_rep{replicate_index}" if replicate_index > 0 else ""
    return Candidate(
        candidate_id=(f"agphase_a{anchor_index}_{direction_id}_{sign}_{tv_label}{seed_label}{replicate_label}"),
        anchor_id=anchor.anchor_id,
        anchor_run_name=anchor.run_name,
        anchor_source_run_name=anchor.source_run_name,
        contrast_family=contrast_family,
        direction_id=direction_id,
        direction_label=direction_label,
        sign=sign,
        replicate_index=replicate_index,
        seed_block=seed_block,
        data_seed=DATA_SEED_BASE + seed_block,
        trainer_seed=0,
        target_phase_tv=target_phase_tv,
        recipient_domains=recipient_domains,
        phase_0_weights=phase_0_weights,
        phase_1_weights=phase_1_weights,
    )


def build_candidates(
    anchors: list[fiber.Anchor],
    domains: tuple[str, ...],
    alpha0: float,
    alpha1: float,
) -> tuple[list[Candidate], list[tuple[str, ...]]]:
    partitions = _balanced_partitions(anchors, domains)
    domain_index = {domain: index for index, domain in enumerate(domains)}
    candidates = []
    for anchor_index, anchor in enumerate(anchors):
        for seed_block in range(N_SEED_BLOCKS):
            candidates.append(
                _candidate(
                    anchor=anchor,
                    anchor_index=anchor_index,
                    contrast_family="center_control",
                    direction_id="center",
                    direction_label="phase-tied frontier control",
                    sign="center",
                    replicate_index=0,
                    seed_block=seed_block,
                    target_phase_tv=0.0,
                    recipient_domains=(),
                    phase_0_weights=anchor.weights.copy(),
                    phase_1_weights=anchor.weights.copy(),
                )
            )

        for partition_index, recipient_domains in enumerate(partitions):
            recipient_mask = np.asarray([domain in recipient_domains for domain in domains])
            direction = _group_transfer_direction(anchor.weights, recipient_mask)
            seed_block = partition_index
            for sign, signed_direction in (("plus", direction), ("minus", -direction)):
                if _maximum_phase_tv(anchor.weights, signed_direction, alpha0, alpha1) < max(TV_LEVELS):
                    raise ValueError(f"Partition {partition_index}/{anchor.anchor_id}/{sign} cannot reach TV=0.5")
                for phase_tv in TV_LEVELS:
                    phase_0, phase_1 = _phase_weights(
                        anchor.weights,
                        phase_tv * signed_direction,
                        alpha0,
                        alpha1,
                    )
                    candidates.append(
                        _candidate(
                            anchor=anchor,
                            anchor_index=anchor_index,
                            contrast_family="balanced_partition",
                            direction_id=f"partition_{partition_index:02d}",
                            direction_label=f"balanced random domain partition {partition_index:02d}",
                            sign=sign,
                            replicate_index=0,
                            seed_block=seed_block,
                            target_phase_tv=phase_tv,
                            recipient_domains=recipient_domains,
                            phase_0_weights=phase_0,
                            phase_1_weights=phase_1,
                        )
                    )

        for recipe_index, (recipe_name, recipient_domains) in enumerate(HANDCRAFTED_RECIPIENTS.items()):
            unknown = set(recipient_domains) - set(domains)
            if unknown:
                raise ValueError(f"Recipe {recipe_name} has unknown domains: {sorted(unknown)}")
            recipient_mask = np.zeros(len(domains), dtype=bool)
            recipient_mask[[domain_index[domain] for domain in recipient_domains]] = True
            direction = _group_transfer_direction(anchor.weights, recipient_mask)
            if _maximum_phase_tv(anchor.weights, direction, alpha0, alpha1) < max(TV_LEVELS):
                raise ValueError(f"Recipe {recipe_name}/{anchor.anchor_id} cannot reach TV=0.5")
            for phase_tv in TV_LEVELS:
                phase_0, phase_1 = _phase_weights(
                    anchor.weights,
                    phase_tv * direction,
                    alpha0,
                    alpha1,
                )
                candidates.append(
                    _candidate(
                        anchor=anchor,
                        anchor_index=anchor_index,
                        contrast_family="handcrafted_late_quality",
                        direction_id=f"recipe_{recipe_name}",
                        direction_label=f"{recipe_name.replace('_', ' ')} late with complement replay",
                        sign="plus",
                        replicate_index=0,
                        seed_block=recipe_index,
                        target_phase_tv=phase_tv,
                        recipient_domains=recipient_domains,
                        phase_0_weights=phase_0,
                        phase_1_weights=phase_1,
                    )
                )

        dolmino_mask = np.asarray([domain.startswith("dolmino_") for domain in domains])
        aggregate_dolmino_share = float(anchor.weights[dolmino_mask].sum())
        for endpoint_index, late_share in enumerate(LATE_DOLMINO_SHARES):
            phase_1 = np.empty_like(anchor.weights)
            phase_1[dolmino_mask] = anchor.weights[dolmino_mask] * late_share / aggregate_dolmino_share
            phase_1[~dolmino_mask] = anchor.weights[~dolmino_mask] * (1.0 - late_share) / (1.0 - aggregate_dolmino_share)
            phase_0 = (anchor.weights - alpha1 * phase_1) / alpha0
            phase_tv = float(0.5 * np.abs(phase_1 - phase_0).sum())
            seed_blocks = (8 + endpoint_index, *DOLMINO_REPEAT_SEED_BLOCKS)
            for replicate_index, seed_block in enumerate(seed_blocks):
                candidates.append(
                    _candidate(
                        anchor=anchor,
                        anchor_index=anchor_index,
                        contrast_family="dolmino_late_continuum",
                        direction_id=f"dolmino_late_{round(100 * late_share):03d}",
                        direction_label=f"{late_share:.0%} Dolmino in the late phase",
                        sign="plus",
                        replicate_index=replicate_index,
                        seed_block=seed_block,
                        target_phase_tv=phase_tv,
                        recipient_domains=tuple(domain for domain in domains if domain.startswith("dolmino_")),
                        phase_0_weights=phase_0,
                        phase_1_weights=phase_1,
                    )
                )

        anchor_count = sum(candidate.anchor_id == anchor.anchor_id for candidate in candidates)
        if anchor_count != EXPECTED_ROWS_PER_ANCHOR:
            raise ValueError(f"Expected {EXPECTED_ROWS_PER_ANCHOR} rows for {anchor.anchor_id}, found {anchor_count}")
    if len(candidates) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} candidates, found {len(candidates)}")
    return candidates, partitions


def _heldout_weights(domains: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(fiber.HELDOUT_PATH)
    rows = []
    for row in frame.to_dict(orient="records"):
        phase_0_json = row.get("phase_0_weights_json")
        phase_1_json = row.get("phase_1_weights_json")
        if not isinstance(phase_0_json, str) or not isinstance(phase_1_json, str):
            continue
        phases = [json.loads(phase_0_json), json.loads(phase_1_json)]
        rows.append([[float(phases[phase][domain]) for domain in domains] for phase in (0, 1)])
    return np.asarray(rows)


def _policy_sha256(domains: tuple[str, ...], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def validate_candidates(
    candidates: list[Candidate],
    anchors: list[fiber.Anchor],
    domains: tuple[str, ...],
    partitions: list[tuple[str, ...]],
    alpha0: float,
    alpha1: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    anchor_lookup = {anchor.anchor_id: anchor for anchor in anchors}
    weights = np.asarray([[candidate.phase_0_weights, candidate.phase_1_weights] for candidate in candidates])
    if float(weights.min()) < -SIMPLEX_TOLERANCE:
        index = np.unravel_index(np.argmin(weights), weights.shape)
        raise ValueError(f"Panel has negative phase weight {weights[index]} at {index}")
    if np.max(np.abs(weights.sum(axis=2) - 1.0)) > SIMPLEX_TOLERANCE:
        raise ValueError("A phase mixture does not sum to one")

    fit_weights = fiber._existing_fit_weights(domains)
    heldout_weights = _heldout_weights(domains)
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
        aggregate_epochs = np.asarray(
            [
                augmented.SIMULATED_EPOCH_TARGET_BUDGET
                * aggregate[domain_index]
                / augmented.TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain]
                for domain_index, domain in enumerate(domains)
            ]
        )
        recipient_mask = np.asarray([domain in candidate.recipient_domains for domain in domains])
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
                "sign": candidate.sign,
                "replicate_index": candidate.replicate_index,
                "seed_block": candidate.seed_block,
                "data_seed": candidate.data_seed,
                "trainer_seed": candidate.trainer_seed,
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
                "recipient_domain_count": len(candidate.recipient_domains),
                "recipient_aggregate_share": float(anchor.weights[recipient_mask].sum()),
                "phase_0_dolmino_share": float(candidate.phase_0_weights[dolmino_mask].sum()),
                "phase_1_dolmino_share": float(candidate.phase_1_weights[dolmino_mask].sum()),
                "phase_0_broad_share": float(candidate.phase_0_weights[~dolmino_mask].sum()),
                "phase_1_broad_share": float(candidate.phase_1_weights[~dolmino_mask].sum()),
                "max_weight": float(weights[index].max()),
                "min_weight": float(weights[index].min()),
                "max_simulated_epoch": float(aggregate_epochs.max()),
                "q95_simulated_epoch": float(np.quantile(aggregate_epochs, 0.95)),
                "min_fit_policy_tv": float(min_fit_tv[index]),
                "min_prior_policy_tv": float(min_prior_tv[index]),
                "recipient_domains_json": json.dumps(candidate.recipient_domains),
            }
        )
    manifest = pd.DataFrame(rows)

    non_control = ~manifest["contrast_family"].eq("center_control")
    non_control_rows = manifest.loc[non_control]
    policy_counts = non_control_rows.groupby("policy_sha256").size()
    repeated_policy_counts = policy_counts.loc[policy_counts.gt(1)]
    expected_repeated_policies = len(anchors) * len(LATE_DOLMINO_SHARES)
    if len(repeated_policy_counts) != expected_repeated_policies:
        raise ValueError(
            f"Expected {expected_repeated_policies} repeated treatment policies, found {len(repeated_policy_counts)}"
        )
    if set(repeated_policy_counts) != {DOLMINO_CONTINUUM_REPLICATES}:
        raise ValueError(f"Treatment replicate counts changed: {repeated_policy_counts.to_dict()}")
    if not policy_counts.loc[~policy_counts.index.isin(repeated_policy_counts.index)].eq(1).all():
        raise ValueError("A non-continuum treatment policy is duplicated")
    for policy_sha256 in repeated_policy_counts.index:
        replicate_rows = non_control_rows.loc[non_control_rows["policy_sha256"].eq(policy_sha256)]
        if set(replicate_rows["contrast_family"]) != {"dolmino_late_continuum"}:
            raise ValueError(f"Repeated policy {policy_sha256} is not a Dolmino continuum schedule")
        if set(replicate_rows["replicate_index"]) != set(range(DOLMINO_CONTINUUM_REPLICATES)):
            raise ValueError(f"Repeated policy {policy_sha256} has incorrect replicate indices")
        if replicate_rows["data_seed"].nunique() != DOLMINO_CONTINUUM_REPLICATES:
            raise ValueError(f"Repeated policy {policy_sha256} does not use independent data seeds")
    if float(manifest.loc[non_control, "min_prior_policy_tv"].min()) <= NOVELTY_TOLERANCE:
        duplicate = manifest.loc[non_control].sort_values("min_prior_policy_tv").iloc[0]
        raise ValueError(
            f"Candidate {duplicate['candidate_id']} aliases prior work at TV {duplicate['min_prior_policy_tv']}"
        )

    random_rows = manifest.loc[manifest["contrast_family"].eq("balanced_partition")]
    for (anchor_id, direction_id, phase_tv), group in random_rows.groupby(
        ["anchor_id", "direction_id", "target_phase_tv"]
    ):
        if set(group["sign"]) != {"plus", "minus"} or len(group) != 2:
            raise ValueError(f"{anchor_id}/{direction_id}/TV={phase_tv} is not an antithetic pair")
        pair = weights[group.index.to_numpy(dtype=int)]
        center = pair.mean(axis=0)
        anchor = anchor_lookup[str(anchor_id)].weights
        if np.max(np.abs(center - np.stack([anchor, anchor]))) > AGGREGATE_TOLERANCE:
            raise ValueError(f"{anchor_id}/{direction_id}/TV={phase_tv} pair is not centered on its anchor")

    expected_families = {
        "center_control": 2 * N_SEED_BLOCKS,
        "balanced_partition": 2 * N_RANDOM_PARTITIONS * 2 * len(TV_LEVELS),
        "handcrafted_late_quality": 2 * len(HANDCRAFTED_RECIPIENTS) * len(TV_LEVELS),
        "dolmino_late_continuum": 2 * len(LATE_DOLMINO_SHARES) * DOLMINO_CONTINUUM_REPLICATES,
    }
    family_counts = manifest["contrast_family"].value_counts().to_dict()
    if family_counts != expected_families:
        raise ValueError(f"Panel family counts changed: {family_counts} != {expected_families}")
    seed_counts = manifest.groupby(["anchor_id", "seed_block"]).size()
    if len(seed_counts) != 2 * N_SEED_BLOCKS or int(seed_counts.min()) < 7 or int(seed_counts.max()) > 10:
        raise ValueError(f"Seed-block balance changed: {seed_counts.to_dict()}")

    partition_matrix = np.asarray(
        [[domain in partition for domain in domains] for partition in partitions],
        dtype=float,
    )
    centered_partitions = partition_matrix - partition_matrix.mean(axis=1, keepdims=True)
    partition_rank = int(np.linalg.matrix_rank(centered_partitions))
    summary: dict[str, object] = {
        "panel_rows": len(manifest),
        "non_control_observations": int(non_control.sum()),
        "unique_non_control_policies": int(manifest.loc[non_control, "policy_sha256"].nunique()),
        "replicated_treatment_policies": len(repeated_policy_counts),
        "dolmino_continuum_replicates_per_policy": DOLMINO_CONTINUUM_REPLICATES,
        "repeat_control_rows": int((~non_control).sum()),
        "anchor_count": len(anchors),
        "rows_per_anchor": manifest.groupby("anchor_id").size().to_dict(),
        "rows_per_contrast_family": family_counts,
        "rows_per_seed_block_range": [int(seed_counts.min()), int(seed_counts.max())],
        "random_partition_count": len(partitions),
        "random_partition_rank": partition_rank,
        "phase_tv_levels": list(TV_LEVELS),
        "late_dolmino_shares": list(LATE_DOLMINO_SHARES),
        "realized_phase_fractions": {"phase_0": alpha0, "phase_1": alpha1},
        "max_aggregate_error": float(manifest["aggregate_max_abs_error"].max()),
        "min_phase_weight": float(weights.min()),
        "max_phase_weight": float(weights.max()),
        "phase_tv_range": [float(manifest["phase_tv"].min()), float(manifest["phase_tv"].max())],
        "phase_information_kl_range": [
            float(manifest["phase_information_kl"].min()),
            float(manifest["phase_information_kl"].max()),
        ],
        "minimum_non_control_prior_policy_tv": float(manifest.loc[non_control, "min_prior_policy_tv"].min()),
        "maximum_non_control_prior_policy_tv": float(manifest.loc[non_control, "min_prior_policy_tv"].max()),
        "selection_uses_prior_one_phase_3e18_outcomes": True,
        "selection_uses_two_phase_outcomes": False,
        "native_table9_scheduled": True,
    }
    return manifest, weights, summary


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    non_controls = manifest.loc[~manifest["contrast_family"].eq("center_control")].copy()
    geometry = px.scatter(
        non_controls,
        x="phase_tv",
        y="phase_information_kl",
        color="max_weight",
        symbol="contrast_family",
        facet_col="anchor_id",
        hover_name="candidate_id",
        hover_data=[
            "direction_label",
            "sign",
            "replicate_index",
            "seed_block",
            "phase_0_dolmino_share",
            "phase_1_dolmino_share",
            "min_prior_policy_tv",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Aggressive phase-asymmetry panel: information, concentration, and contrast",
    )
    geometry.update_layout(width=1500, height=780, margin={"l": 70, "r": 150, "t": 130, "b": 70})
    geometry.write_html(output_dir / "phase_asymmetry_geometry.html", include_plotlyjs=True, config=PLOT_CONFIG)

    conventional = non_controls.loc[
        non_controls["contrast_family"].isin(("handcrafted_late_quality", "dolmino_late_continuum"))
    ]
    shares = px.scatter(
        conventional,
        x="phase_0_dolmino_share",
        y="phase_1_dolmino_share",
        color="phase_tv",
        symbol="anchor_id",
        hover_name="candidate_id",
        hover_data=[
            "direction_label",
            "replicate_index",
            "phase_0_broad_share",
            "phase_1_broad_share",
            "max_weight",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Conventional curricula: Dolmino exposure early versus late",
    )
    shares.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"color": "#506274", "dash": "dash"})
    shares.update_layout(width=1100, height=800)
    shares.write_html(output_dir / "conventional_dolmino_schedules.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    anchor_audit: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    geometry = (
        manifest.groupby(["anchor_id", "contrast_family", "target_phase_tv"])[
            [
                "phase_information_kl",
                "max_weight",
                "phase_0_dolmino_share",
                "phase_1_dolmino_share",
                "min_prior_policy_tv",
            ]
        ]
        .agg(["count", "min", "median", "max"])
        .round(6)
    )
    lines = [
        "# Delphi 3e18 aggressive frontier WSD annealing-content panel",
        "",
        "## Scientific question",
        "",
        (
            "At fixed total token allocation, what data should be concentrated in the boundary-aligned low-learning-"
            "rate final 20% of WSD training near strong one-phase frontier mixtures? The intervention directly "
            "controls phase TV rather than stopping at the first small-bucket simplex boundary."
        ),
        "",
        "For anchor mixture $a$ and phase fractions $\\alpha_0,\\alpha_1$, every candidate is",
        "",
        "$$p^{(0)}=a-\\alpha_1\\Delta,\\qquad p^{(1)}=a+\\alpha_0\\Delta,$$",
        "",
        "so $\\alpha_0p^{(0)}+\\alpha_1p^{(1)}=a$. This fixes token aggregate, not learning-rate-weighted exposure: "
        "phase 1 is the final WSD decay window and therefore has a different optimization role. Non-control "
        "candidates directly target",
        "",
        "$$\\operatorname{TV}(p^{(0)},p^{(1)})=\\frac12\\lVert\\Delta\\rVert_1.$ $".replace("$ $", "$$"),
        "",
        "## Frozen design",
        "",
        f"- {N_RANDOM_PARTITIONS} balanced random domain partitions shared across anchors.",
        f"- Antithetic phase orders at TV {', '.join(str(value) for value in TV_LEVELS)}.",
        f"- {len(HANDCRAFTED_RECIPIENTS)} conventional good-data-late group transfers at the same TV levels.",
        (
            "- Explicit 75%, 90%, and 100% Dolmino late-phase schedules with broad-data replay when nonzero; "
            f"each policy is trained with {DOLMINO_CONTINUUM_REPLICATES} matched-control data seeds."
        ),
        f"- {N_SEED_BLOCKS} tied seed controls per anchor.",
        f"- {EXPECTED_TOTAL_ROWS} checkpoints total; every checkpoint receives Uncheatable and native Table-9 BPB.",
        "- Target outcomes from this panel were not used to construct candidates.",
        "",
        "## Anchor audit",
        "",
        anchor_audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Geometry and novelty audit",
        "",
        f"- Rows: {summary['panel_rows']}.",
        f"- Non-control observations: {summary['non_control_observations']}.",
        f"- Unique non-control policies: {summary['unique_non_control_policies']}.",
        f"- Replicated treatment policies: {summary['replicated_treatment_policies']}.",
        f"- Repeat control observations: {summary['repeat_control_rows']}.",
        f"- Balanced-partition rank: {summary['random_partition_rank']} of at most {N_RANDOM_PARTITIONS}.",
        f"- Maximum aggregate error: {summary['max_aggregate_error']:.3e}.",
        f"- Phase TV range: {summary['phase_tv_range']}.",
        f"- Phase-information range: {summary['phase_information_kl_range']}.",
        f"- Maximum phase weight: {summary['max_phase_weight']:.6f}.",
        f"- Minimum distance to any prior fit or heldout policy: {summary['minimum_non_control_prior_policy_tv']:.6g}.",
        "",
        geometry.to_markdown(floatfmt=".6f"),
        "",
        "## Analysis boundary",
        "",
        (
            "This is an experiment about annealing-window content under a fixed 80/20 WSD schedule, not a "
            "schedule-independent estimate of pure data order. Analyze each anchor, contrast family, direction, "
            "sign, and TV stratum separately before pooling. The balanced partitions estimate the distribution of "
            "large diffuse phase contrasts; the handcrafted schedules test named conventional curricula. Use tied "
            "controls by seed block and require consistency across TV levels before treating an apparent gain as "
            "phase-order signal."
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
    anchors, anchor_audit = random_population.load_policy_anchors(domains)
    candidates, partitions = build_candidates(anchors, domains, alpha0, alpha1)
    manifest, weights, summary = validate_candidates(
        candidates,
        anchors,
        domains,
        partitions,
        alpha0,
        alpha1,
    )
    manifest_path = args.output_dir / "candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    anchor_audit.to_csv(args.output_dir / "anchor_audit.csv", index=False)
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
    render_diagnostics(manifest, args.output_dir)
    write_report(args.output_dir, manifest, anchor_audit, summary)
    if args.upload:
        fiber.upload_artifact(source_path, str(summary["gcs_launcher_source_panel"]))
        fiber.upload_artifact(manifest_path, str(summary["gcs_candidate_manifest"]))
        for name in ("summary.json", "report.md", "phase_weights.csv", "anchor_audit.csv"):
            fiber.upload_artifact(args.output_dir / name, f"{args.gcs_output_dir}/source/{name}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
