# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch tied one-bucket epoch dose-response panels at 60M and Delphi 3e18.

For a focal bucket with token-proportional weight ``p_i``, a multiplier ``m``
sets its tied weight to ``m p_i``. The other 38 buckets retain their
token-proportional ratios:

    w_i = m p_i
    w_j = p_j (1 - m p_i) / (1 - p_i),  j != i

This estimates a conditional one-bucket dose response around the proportional
mixture. It does not identify a context-free optimum for each bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS, uncheatable_datasets
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as delphi
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    EXPERIMENT_BUDGET,
    NUM_TRAIN_STEPS,
    PHASE_NAMES,
    SEQ_LEN,
    TARGET_BUDGET,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = (
    SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "bucket_epoch_dose_response_20260729"
)

PANEL_TAG = "bucket-epoch-dose-response-20260729"
EXPERIMENT_NAMES = {
    ("60m", "pilot"): "pinlin_calvin_xu/data_mixture/be60p_20260729",
    ("60m", "full"): "pinlin_calvin_xu/data_mixture/be60_20260729",
    ("delphi_3e18", "pilot"): "pinlin_calvin_xu/data_mixture/bed3p_20260729",
    ("delphi_3e18", "full"): "pinlin_calvin_xu/data_mixture/bed3_20260729",
}
EPOCH_MULTIPLIERS = (0.0, 0.25, 0.5, 2.0, 4.0, 8.0, 16.0, 32.0)
PROPORTIONAL_MULTIPLIER = 1.0
MAX_FOCAL_WEIGHT = 0.5
PROPORTIONAL_EPOCHS = TARGET_BUDGET_DOLMA3_COMMON_CRAWL / TOP_LEVEL_TOTAL_AVAILABLE_TOKENS
REALIZED_60M_TRAIN_TOKENS = NUM_TRAIN_STEPS * BATCH_SIZE * SEQ_LEN
EXPECTED_INTERVENTIONS = 276
EXPECTED_POINTS = EXPECTED_INTERVENTIONS + 1

MAIN_DATA_SEED = 20_260_729
MAIN_TRAINER_SEED = 0
MAIN_SIMULATED_EPOCH_SUBSET_SEED = 20_260_729
PILOT_TRAINER_SEEDS = tuple(range(6))
PILOT_SUBSET_SEEDS = tuple(range(MAIN_SIMULATED_EPOCH_SUBSET_SEED, MAIN_SIMULATED_EPOCH_SUBSET_SEED + 6))
PILOT_PROBE_SUBSET_SEEDS = PILOT_SUBSET_SEEDS[:3]
PILOT_PROBE_DOMAINS = (
    "dolma3_cc/art_and_design_high",
    "dolma3_arxiv",
    "dolmino_synth_math",
)
PILOT_PROBE_MULTIPLIERS = (4.0, 16.0)
PILOT_LOW_DOSE_PROBE_DOMAIN = "dolma3_cc/art_and_design_high"
PILOT_LOW_DOSE_PROBE_MULTIPLIER = 0.25
PILOT_TRAINER_PROBE_DOMAIN = "dolma3_cc/art_and_design_high"
PILOT_TRAINER_PROBE_MULTIPLIER = 16.0
EXPECTED_PILOT_RUNS = (
    len(PILOT_TRAINER_SEEDS)
    + len(PILOT_SUBSET_SEEDS)
    - 1
    + len(PILOT_PROBE_DOMAINS) * len(PILOT_PROBE_MULTIPLIERS) * len(PILOT_PROBE_SUBSET_SEEDS)
    + len(PILOT_PROBE_SUBSET_SEEDS)
    + len(PILOT_TRAINER_SEEDS)
    - 1
)
RUN_ID_BASE = {
    ("60m", "full"): 7_290_000,
    ("delphi_3e18", "full"): 7_291_000,
    ("60m", "pilot"): 7_292_000,
    ("delphi_3e18", "pilot"): 7_293_000,
}

DEFAULT_TPU_TYPE = "v5p-8"
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 56
TABLE9_REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
TABLE9_RESOURCES = ResourceConfig.with_tpu("v6e-8", regions=["us-east5"], zone="us-east5-b", disk="80g")
SKIP_EVAL_HARNESS_ENV_VAR = "LEVANTER_SKIP_EVAL_HARNESS"
PALOMA_COMPONENT_PREFIX = "paloma/"
UNCHEATABLE_COMPONENT_PREFIX = "uncheatable_eval/"
UNCHEATABLE_CACHE_VERSION = "2026.06.28"
HF_HUB_DISABLE_XET_ENV_VAR = "HF_HUB_DISABLE_XET"
RUN_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


class Scale(StrEnum):
    """Supported training configurations."""

    SIXTY_M = "60m"
    DELPHI_3E18 = "delphi_3e18"


class Stage(StrEnum):
    """Submission stages for the variance-gated design."""

    PILOT = "pilot"
    FULL = "full"


@dataclass(frozen=True)
class EpochSweepPoint:
    """One unique tied policy in the shared cross-scale design."""

    point_index: int
    point_id: str
    point_kind: str
    focal_index: int | None
    focal_domain: str | None
    epoch_multiplier: float
    target_simulated_epochs: float
    focal_weight: float | None
    complement_scale: float | None
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class EpochSweepRunSpec:
    """Scale-resolved identity and provenance for one training run."""

    scale: str
    stage: str
    run_order: int
    run_id: int
    run_name: str
    point_id: str
    point_kind: str
    seed_block: str
    replicate_index: int
    focal_index: int | None
    focal_domain: str | None
    epoch_multiplier: float
    target_simulated_epochs: float
    focal_weight: float | None
    complement_scale: float | None
    trainer_seed: int
    data_seed: int
    simulated_epoch_subset_seed: int
    experiment_budget: int
    target_budget: int
    num_train_steps: int
    expected_checkpoint_step: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class SaveManifestConfig:
    """Configuration for persisting the exact submitted design."""

    output_path: str
    scale: str
    stage: str
    run_specs_json: str


@dataclass(frozen=True)
class LaunchArtifacts:
    """Manifest, training, and native Table-9 graph for one scale."""

    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep]

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _slug(value: str) -> str:
    slug = RUN_NAME_PATTERN.sub("_", value).strip("_").lower()
    if not slug:
        raise ValueError(f"Could not derive a slug from {value!r}")
    return slug


def _multiplier_code(multiplier: float) -> str:
    return f"{multiplier:g}".replace(".", "p")


def _proportional_weights() -> dict[str, float]:
    return {domain: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] / TOP_LEVEL_TOTAL_AVAILABLE_TOKENS for domain in DOMAIN_NAMES}


def _tied_phase_weights(weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase: dict(weights) for phase in PHASE_NAMES}


def _intervention_weights(
    proportional: dict[str, float],
    *,
    focal_domain: str,
    multiplier: float,
) -> tuple[dict[str, float], float]:
    focal_proportional = proportional[focal_domain]
    focal_weight = multiplier * focal_proportional
    if focal_weight > MAX_FOCAL_WEIGHT + 1e-12:
        raise ValueError(f"{focal_domain}/{multiplier:g} exceeds the focal-weight cap")
    complement_scale = (1.0 - focal_weight) / (1.0 - focal_proportional)
    weights = {
        domain: focal_weight if domain == focal_domain else weight * complement_scale
        for domain, weight in proportional.items()
    }
    return weights, complement_scale


def build_points() -> list[EpochSweepPoint]:
    """Build the 277 unique tied policies shared by both scales."""
    proportional = _proportional_weights()
    points = [
        EpochSweepPoint(
            point_index=0,
            point_id="proportional_anchor",
            point_kind="proportional_anchor",
            focal_index=None,
            focal_domain=None,
            epoch_multiplier=PROPORTIONAL_MULTIPLIER,
            target_simulated_epochs=PROPORTIONAL_EPOCHS,
            focal_weight=None,
            complement_scale=None,
            phase_weights=_tied_phase_weights(proportional),
        )
    ]
    for focal_index, focal_domain in enumerate(DOMAIN_NAMES):
        for multiplier in EPOCH_MULTIPLIERS:
            focal_weight = multiplier * proportional[focal_domain]
            if focal_weight > MAX_FOCAL_WEIGHT + 1e-12:
                continue
            weights, complement_scale = _intervention_weights(
                proportional,
                focal_domain=focal_domain,
                multiplier=multiplier,
            )
            points.append(
                EpochSweepPoint(
                    point_index=len(points),
                    point_id=(f"d{focal_index:02d}_{_slug(focal_domain)}_m{_multiplier_code(multiplier)}"),
                    point_kind="focal_bucket_dose",
                    focal_index=focal_index,
                    focal_domain=focal_domain,
                    epoch_multiplier=multiplier,
                    target_simulated_epochs=multiplier * PROPORTIONAL_EPOCHS,
                    focal_weight=focal_weight,
                    complement_scale=complement_scale,
                    phase_weights=_tied_phase_weights(weights),
                )
            )
    _validate_points(points)
    return points


def _policy_hash(point: EpochSweepPoint) -> str:
    payload = {phase: [point.phase_weights[phase][domain] for domain in DOMAIN_NAMES] for phase in PHASE_NAMES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _validate_points(points: list[EpochSweepPoint]) -> None:
    if len(points) != EXPECTED_POINTS:
        raise ValueError(f"Expected {EXPECTED_POINTS} points, found {len(points)}")
    if Counter(point.point_kind for point in points) != {
        "proportional_anchor": 1,
        "focal_bucket_dose": EXPECTED_INTERVENTIONS,
    }:
        raise ValueError("Unexpected point-kind counts")
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("Point IDs are not unique")
    if len({_policy_hash(point) for point in points}) != len(points):
        raise ValueError("The panel contains duplicate policy coordinates")

    proportional = _proportional_weights()
    for point in points:
        phase_0 = point.phase_weights["phase_0"]
        phase_1 = point.phase_weights["phase_1"]
        if phase_0 != phase_1:
            raise ValueError(f"{point.point_id} is not phase tied")
        if set(phase_0) != set(DOMAIN_NAMES):
            raise ValueError(f"{point.point_id} has the wrong domain set")
        if any(weight < 0 for weight in phase_0.values()):
            raise ValueError(f"{point.point_id} has a negative weight")
        if not math.isclose(sum(phase_0.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{point.point_id} weights do not sum to one")
        if point.focal_domain is None:
            if any(
                not math.isclose(phase_0[domain], proportional[domain], rel_tol=0.0, abs_tol=1e-15)
                for domain in DOMAIN_NAMES
            ):
                raise ValueError("The proportional anchor changed")
            continue

        focal_domain = point.focal_domain
        if point.focal_weight is None or point.complement_scale is None:
            raise ValueError(f"{point.point_id} is missing intervention diagnostics")
        if point.focal_weight > MAX_FOCAL_WEIGHT + 1e-12:
            raise ValueError(f"{point.point_id} exceeds the focal-weight cap")
        if not math.isclose(
            phase_0[focal_domain],
            point.epoch_multiplier * proportional[focal_domain],
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(f"{point.point_id} has the wrong focal weight")
        for domain in DOMAIN_NAMES:
            if domain == focal_domain:
                continue
            ratio = phase_0[domain] / proportional[domain]
            if not math.isclose(ratio, point.complement_scale, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{point.point_id} changed complement ratios")
        realized_epochs = (
            TARGET_BUDGET_DOLMA3_COMMON_CRAWL * phase_0[focal_domain] / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[focal_domain]
        )
        if not math.isclose(
            realized_epochs,
            point.target_simulated_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{point.point_id} realizes {realized_epochs} epochs")


def _run_name(point: EpochSweepPoint) -> str:
    if point.focal_index is None:
        return "p000_proportional_anchor"
    return f"p{point.point_index:03d}_d{point.focal_index:02d}_" f"m{_multiplier_code(point.epoch_multiplier)}"


def _experiment_name(scale: Scale, stage: Stage) -> str:
    return EXPERIMENT_NAMES[(scale.value, stage.value)]


def _run_specs_60m(points: list[EpochSweepPoint], *, stage: Stage) -> list[EpochSweepRunSpec]:
    return [
        EpochSweepRunSpec(
            scale=Scale.SIXTY_M,
            stage=stage,
            run_order=point.point_index,
            run_id=RUN_ID_BASE[(Scale.SIXTY_M.value, stage.value)] + point.point_index,
            run_name=_run_name(point),
            point_id=point.point_id,
            point_kind=point.point_kind,
            seed_block="main_grid",
            replicate_index=0,
            focal_index=point.focal_index,
            focal_domain=point.focal_domain,
            epoch_multiplier=point.epoch_multiplier,
            target_simulated_epochs=point.target_simulated_epochs,
            focal_weight=point.focal_weight,
            complement_scale=point.complement_scale,
            trainer_seed=MAIN_TRAINER_SEED,
            data_seed=MAIN_DATA_SEED,
            simulated_epoch_subset_seed=MAIN_SIMULATED_EPOCH_SUBSET_SEED,
            experiment_budget=REALIZED_60M_TRAIN_TOKENS,
            target_budget=TARGET_BUDGET,
            num_train_steps=NUM_TRAIN_STEPS,
            expected_checkpoint_step=NUM_TRAIN_STEPS - 1,
            phase_weights=point.phase_weights,
        )
        for point in points
    ]


def _pilot_run_specs(full_specs: list[EpochSweepRunSpec], *, scale: Scale) -> list[EpochSweepRunSpec]:
    by_point_id = {spec.point_id: spec for spec in full_specs}
    anchor = by_point_id["proportional_anchor"]
    pilot_specs: list[EpochSweepRunSpec] = []

    for replicate_index, trainer_seed in enumerate(PILOT_TRAINER_SEEDS):
        pilot_specs.append(
            replace(
                anchor,
                stage=Stage.PILOT,
                run_name=f"a_t{trainer_seed}",
                seed_block="anchor_trainer_seed",
                replicate_index=replicate_index,
                trainer_seed=trainer_seed,
            )
        )

    alternate_subset_seeds = [
        subset_seed for subset_seed in PILOT_SUBSET_SEEDS if subset_seed != MAIN_SIMULATED_EPOCH_SUBSET_SEED
    ]
    for replicate_index, subset_seed in enumerate(alternate_subset_seeds, start=1):
        pilot_specs.append(
            replace(
                anchor,
                stage=Stage.PILOT,
                run_name=f"a_u{subset_seed % 100:02d}",
                seed_block="anchor_subset_seed",
                replicate_index=replicate_index,
                simulated_epoch_subset_seed=subset_seed,
            )
        )

    for focal_domain in PILOT_PROBE_DOMAINS:
        focal_index = DOMAIN_NAMES.index(focal_domain)
        for multiplier in PILOT_PROBE_MULTIPLIERS:
            point_id = f"d{focal_index:02d}_{_slug(focal_domain)}_m{_multiplier_code(multiplier)}"
            base_spec = by_point_id[point_id]
            for replicate_index, subset_seed in enumerate(PILOT_PROBE_SUBSET_SEEDS):
                pilot_specs.append(
                    replace(
                        base_spec,
                        stage=Stage.PILOT,
                        run_name=f"q{focal_index:02d}_m{_multiplier_code(multiplier)}_u{subset_seed % 100:02d}",
                        seed_block="high_replay_subset_seed",
                        replicate_index=replicate_index,
                        simulated_epoch_subset_seed=subset_seed,
                    )
                )

    low_dose_probe_index = DOMAIN_NAMES.index(PILOT_LOW_DOSE_PROBE_DOMAIN)
    low_dose_probe_id = (
        f"d{low_dose_probe_index:02d}_{_slug(PILOT_LOW_DOSE_PROBE_DOMAIN)}_"
        f"m{_multiplier_code(PILOT_LOW_DOSE_PROBE_MULTIPLIER)}"
    )
    low_dose_probe = by_point_id[low_dose_probe_id]
    for replicate_index, subset_seed in enumerate(PILOT_PROBE_SUBSET_SEEDS):
        pilot_specs.append(
            replace(
                low_dose_probe,
                stage=Stage.PILOT,
                run_name=(
                    f"q{low_dose_probe_index:02d}_m{_multiplier_code(PILOT_LOW_DOSE_PROBE_MULTIPLIER)}_"
                    f"u{subset_seed % 100:02d}"
                ),
                seed_block="low_dose_subset_seed",
                replicate_index=replicate_index,
                simulated_epoch_subset_seed=subset_seed,
            )
        )

    trainer_probe_index = DOMAIN_NAMES.index(PILOT_TRAINER_PROBE_DOMAIN)
    trainer_probe_id = (
        f"d{trainer_probe_index:02d}_{_slug(PILOT_TRAINER_PROBE_DOMAIN)}_"
        f"m{_multiplier_code(PILOT_TRAINER_PROBE_MULTIPLIER)}"
    )
    trainer_probe = by_point_id[trainer_probe_id]
    for replicate_index, trainer_seed in enumerate(PILOT_TRAINER_SEEDS[1:], start=1):
        pilot_specs.append(
            replace(
                trainer_probe,
                stage=Stage.PILOT,
                run_name=(
                    f"q{trainer_probe_index:02d}_m{_multiplier_code(PILOT_TRAINER_PROBE_MULTIPLIER)}_" f"t{trainer_seed}"
                ),
                seed_block="high_replay_trainer_seed",
                replicate_index=replicate_index,
                trainer_seed=trainer_seed,
            )
        )

    run_id_base = RUN_ID_BASE[(scale.value, Stage.PILOT.value)]
    return [
        replace(spec, run_order=run_order, run_id=run_id_base + run_order) for run_order, spec in enumerate(pilot_specs)
    ]


def _delphi_candidate(analysis_output_path: str) -> Any:
    scaling_fits = delphi._read_scaling_fits(analysis_output_path)
    return delphi._candidate_for_budget(scaling_fits=scaling_fits)


def _run_specs_delphi(
    points: list[EpochSweepPoint],
    *,
    analysis_output_path: str,
    stage: Stage,
) -> list[EpochSweepRunSpec]:
    candidate = _delphi_candidate(analysis_output_path)
    realized_train_tokens = candidate.train_steps * delphi.TARGET_BATCH_SIZE * delphi.SEQ_LEN_DELPHI

    run_specs: list[EpochSweepRunSpec] = []
    for point in points:
        run_spec = EpochSweepRunSpec(
            scale=Scale.DELPHI_3E18,
            stage=stage,
            run_order=point.point_index,
            run_id=RUN_ID_BASE[(Scale.DELPHI_3E18.value, stage.value)] + point.point_index,
            run_name=_run_name(point),
            point_id=point.point_id,
            point_kind=point.point_kind,
            seed_block="main_grid",
            replicate_index=0,
            focal_index=point.focal_index,
            focal_domain=point.focal_domain,
            epoch_multiplier=point.epoch_multiplier,
            target_simulated_epochs=point.target_simulated_epochs,
            focal_weight=point.focal_weight,
            complement_scale=point.complement_scale,
            trainer_seed=MAIN_TRAINER_SEED,
            data_seed=MAIN_DATA_SEED,
            simulated_epoch_subset_seed=MAIN_SIMULATED_EPOCH_SUBSET_SEED,
            experiment_budget=realized_train_tokens,
            target_budget=TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
            num_train_steps=candidate.train_steps,
            expected_checkpoint_step=candidate.train_steps - 1,
            phase_weights=point.phase_weights,
        )
        run_specs.append(run_spec)
    return run_specs


def _delphi_specs_for_run_specs(
    run_specs: list[EpochSweepRunSpec],
    *,
    analysis_output_path: str,
    stage: Stage,
) -> list[delphi.DelphiSwarmRunSpec]:
    candidate = _delphi_candidate(analysis_output_path)
    params = int(candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    tensor_parallel_size = delphi._tensor_parallel_size(candidate.model_config.hidden_dim, DEFAULT_TPU_TYPE)
    delphi_specs: list[delphi.DelphiSwarmRunSpec] = []
    for run_spec in run_specs:
        max_epoch, q95_epoch, phase_tv = delphi._weight_diagnostics(run_spec.phase_weights)
        delphi_spec = delphi.DelphiSwarmRunSpec(
            run_order=run_spec.run_order,
            run_id=run_spec.run_id,
            run_name=run_spec.run_name,
            source_run_name=run_spec.point_id,
            source_experiment=_experiment_name(Scale.DELPHI_3E18, stage),
            panel_source=f"bucket_epoch_dose_response_{stage.value}",
            target_flops=delphi.TARGET_FLOPS,
            tpu_type=DEFAULT_TPU_TYPE,
            tpu_region=DEFAULT_TPU_REGION,
            tpu_zone=DEFAULT_TPU_ZONE,
            batch_size=delphi.TARGET_BATCH_SIZE,
            train_steps=run_spec.num_train_steps,
            realized_train_tokens=run_spec.experiment_budget,
            expected_checkpoint_step=run_spec.expected_checkpoint_step,
            model_hidden_dim=int(candidate.model_config.hidden_dim),
            model_layers=int(candidate.model_config.num_layers),
            non_embedding_params=non_embedding_params,
            total_trainable_params=params,
            tensor_parallel_size=tensor_parallel_size,
            data_seed=run_spec.data_seed,
            trainer_seed=run_spec.trainer_seed,
            phase_boundary=delphi.PHASE_BOUNDARIES[0],
            phase_0_fraction=delphi.PHASE_FRACTIONS["phase_0"],
            phase_1_fraction=delphi.PHASE_FRACTIONS["phase_1"],
            simulated_epoch_target_budget=TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
            available_top_level_tokens=TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
            max_simulated_epoch=max_epoch,
            q95_simulated_epoch=q95_epoch,
            mean_phase_tv_to_proportional=phase_tv,
            phase_weights=run_spec.phase_weights,
            simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        )
        delphi_specs.append(delphi_spec)
    return delphi_specs


def save_manifest(config: SaveManifestConfig) -> None:
    """Persist the exact design and scale-resolved run manifest."""
    run_specs = [EpochSweepRunSpec(**item) for item in json.loads(config.run_specs_json)]
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    manifest = {
        "panel_tag": PANEL_TAG,
        "scale": config.scale,
        "stage": config.stage,
        "estimand": "conditional one-bucket dose response around the token-proportional complement",
        "policy_formula": {
            "focal": "w_i = m p_i",
            "complement": "w_j = p_j (1 - m p_i) / (1 - p_i)",
        },
        "phase_mode": "tied",
        "epoch_multipliers": list(EPOCH_MULTIPLIERS),
        "proportional_multiplier": PROPORTIONAL_MULTIPLIER,
        "proportional_simulated_epochs": PROPORTIONAL_EPOCHS,
        "max_focal_weight": MAX_FOCAL_WEIGHT,
        "main_data_seed": MAIN_DATA_SEED,
        "main_trainer_seed": MAIN_TRAINER_SEED,
        "simulated_epoch_subset_seed": MAIN_SIMULATED_EPOCH_SUBSET_SEED,
        "pilot_design": {
            "trainer_seeds": list(PILOT_TRAINER_SEEDS),
            "subset_seeds": list(PILOT_SUBSET_SEEDS),
            "probe_subset_seeds": list(PILOT_PROBE_SUBSET_SEEDS),
            "probe_domains": list(PILOT_PROBE_DOMAINS),
            "probe_multipliers": list(PILOT_PROBE_MULTIPLIERS),
            "low_dose_probe_domain": PILOT_LOW_DOSE_PROBE_DOMAIN,
            "low_dose_probe_multiplier": PILOT_LOW_DOSE_PROBE_MULTIPLIER,
            "trainer_probe_domain": PILOT_TRAINER_PROBE_DOMAIN,
            "trainer_probe_multiplier": PILOT_TRAINER_PROBE_MULTIPLIER,
        },
        "smooth_targets": ["uncheatable_eval_bpb", "olmo_base_eval_table9_macro_bpb"],
        "right_censoring_rule": "A best value at the largest feasible tested dose is not called an optimum.",
        "run_count": len(run_specs),
        "run_specs": [asdict(spec) for spec in run_specs],
    }
    with fs.open(os.path.join(config.output_path, "design_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    run_fields = [field for field in asdict(run_specs[0]) if field != "phase_weights"]
    run_buffer = io.StringIO(newline="")
    run_writer = csv.DictWriter(run_buffer, fieldnames=run_fields)
    run_writer.writeheader()
    for spec in run_specs:
        row = asdict(spec)
        row.pop("phase_weights")
        run_writer.writerow(row)
    with fs.open(os.path.join(config.output_path, "run_manifest.csv"), "w") as handle:
        handle.write(run_buffer.getvalue())

    weight_buffer = io.StringIO(newline="")
    weight_writer = csv.DictWriter(
        weight_buffer,
        fieldnames=["scale", "run_name", "point_id", "phase", "domain", "weight"],
    )
    weight_writer.writeheader()
    for spec in run_specs:
        for phase in PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                weight_writer.writerow(
                    {
                        "scale": spec.scale,
                        "run_name": spec.run_name,
                        "point_id": spec.point_id,
                        "phase": phase,
                        "domain": domain,
                        "weight": spec.phase_weights[phase][domain],
                    }
                )
    with fs.open(os.path.join(config.output_path, "phase_weights.csv"), "w") as handle:
        handle.write(weight_buffer.getvalue())


def _validate_uncheatable_caches() -> None:
    prefix = marin_prefix_for_region(DEFAULT_TPU_REGION)
    missing = []
    for subset in UNCHEATABLE_SUBSETS:
        path = f"{prefix}/uncheatable_eval/{subset}-llama3/{UNCHEATABLE_CACHE_VERSION}/validation/.stats.json"
        if not fsspec.open(path).fs.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing Uncheatable validation caches: {missing}")


def _eval_provenance(run_spec: EpochSweepRunSpec, *, scale: Scale, stage: Stage) -> dict[str, Any]:
    return {
        "evaluator": "marin-native-table9-bpb",
        "panel": PANEL_TAG,
        "scale": scale,
        "stage": stage,
        "run_name": run_spec.run_name,
        "point_id": run_spec.point_id,
        "seed_block": run_spec.seed_block,
        "replicate_index": run_spec.replicate_index,
        "focal_domain": run_spec.focal_domain or "",
        "epoch_multiplier": run_spec.epoch_multiplier,
        "trainer_seed": run_spec.trainer_seed,
        "data_seed": run_spec.data_seed,
        "simulated_epoch_subset_seed": run_spec.simulated_epoch_subset_seed,
    }


def _configure_60m_training_step(training_step: ExecutorStep) -> ExecutorStep:
    config = training_step.config
    if not isinstance(config, TrainLmOnPodConfig):
        raise TypeError(f"Expected TrainLmOnPodConfig for {training_step.name!r}, got {type(config)!r}")
    train_config = config.train_config
    if not isinstance(train_config, TrainLmConfig):
        raise TypeError(f"Expected TrainLmConfig for {training_step.name!r}, got {type(train_config)!r}")
    data = train_config.data
    if not isinstance(data, LMMixtureDatasetConfig):
        raise TypeError(f"Expected LMMixtureDatasetConfig for {training_step.name!r}, got {type(data)!r}")

    paloma_components = {name for name in data.components if name.startswith(PALOMA_COMPONENT_PREFIX)}
    uncheatable_components = {name for name in data.components if name.startswith(UNCHEATABLE_COMPONENT_PREFIX)}
    if not paloma_components or not uncheatable_components:
        raise ValueError(f"{training_step.name} has unexpected validation components")
    components = {name: component for name, component in data.components.items() if name not in paloma_components}
    if isinstance(data.train_weights, dict):
        train_weights: dict[str, float] | list[tuple[int, dict[str, float]]] = {
            name: weight for name, weight in data.train_weights.items() if name not in paloma_components
        }
    elif isinstance(data.train_weights, list):
        train_weights = [
            (step, {name: weight for name, weight in weights.items() if name not in paloma_components})
            for step, weights in data.train_weights
        ]
    else:
        raise TypeError(f"Unexpected train_weights type: {type(data.train_weights)!r}")
    weight_stages = [train_weights] if isinstance(train_weights, dict) else train_weights
    for stage_config in weight_stages:
        weights = stage_config if isinstance(stage_config, dict) else stage_config[1]
        if any(weights.get(name, 0.0) != 0.0 for name in uncheatable_components):
            raise ValueError(f"{training_step.name} assigns nonzero weight to an Uncheatable validation component")
    data = replace(data, components=components, train_weights=train_weights)
    train_config = replace(train_config, data=data)
    env_vars = dict(config.env_vars or {})
    env_vars["MARIN_PREFIX"] = marin_prefix_for_region(DEFAULT_TPU_REGION)
    env_vars[SKIP_EVAL_HARNESS_ENV_VAR] = "1"
    return replace(
        training_step,
        config=replace(
            config,
            train_config=train_config,
            env_vars=env_vars,
        ),
    )


def _manifest_step(scale: Scale, stage: Stage, run_specs: list[EpochSweepRunSpec]) -> ExecutorStep:
    experiment_name = _experiment_name(scale, stage)
    return ExecutorStep(
        name=f"{experiment_name}/manifest",
        fn=save_manifest,
        config=SaveManifestConfig(
            output_path=this_output_path(),
            scale=scale,
            stage=stage,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        ),
    )


def _build_60m_artifacts(
    run_specs: list[EpochSweepRunSpec],
    *,
    stage: Stage,
) -> LaunchArtifacts:
    _validate_uncheatable_caches()
    experiment_name = _experiment_name(Scale.SIXTY_M, stage)
    resources = ResourceConfig.with_tpu(
        DEFAULT_TPU_TYPE,
        regions=[DEFAULT_TPU_REGION],
        zone=DEFAULT_TPU_ZONE,
    )
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=experiment_name,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        model_config=regmix_60m_proxy,
        resources=resources,
        eval_harness_tasks=(),
        runtime_cache_region=DEFAULT_TPU_REGION,
    )
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec in run_specs:
        training_step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run_spec.run_id, phase_weights=run_spec.phase_weights),
            name_prefix=experiment_name,
            run_name=run_spec.run_name,
            data_seed=run_spec.data_seed,
            trainer_seed=run_spec.trainer_seed,
            simulated_epoch_subset_seed=run_spec.simulated_epoch_subset_seed,
        )
        training_step = _configure_60m_training_step(training_step)
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=TABLE9_REQUEST_SET_DIR,
                resource_config=TABLE9_RESOURCES,
                wandb_group=f"olmo_base_eval_table9_bucket_epoch_dose_60m_{stage.value}_20260729",
                provenance=_eval_provenance(run_spec, scale=Scale.SIXTY_M, stage=stage),
            )
        )
    return LaunchArtifacts(
        manifest_step=_manifest_step(Scale.SIXTY_M, stage, run_specs),
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def _delphi_validation_configs() -> dict[str, Any]:
    steps = uncheatable_datasets(tokenizer=llama3_tokenizer)
    return {
        os.path.join("uncheatable_eval", name): step_to_lm_mixture_component(step, include_raw_paths=False)
        for name, step in steps.items()
    }


def _build_delphi_artifacts(
    run_specs: list[EpochSweepRunSpec],
    *,
    analysis_output_path: str,
    stage: Stage,
) -> LaunchArtifacts:
    experiment_name = _experiment_name(Scale.DELPHI_3E18, stage)
    delphi_specs = _delphi_specs_for_run_specs(
        run_specs,
        analysis_output_path=analysis_output_path,
        stage=stage,
    )
    validation_configs = _delphi_validation_configs()
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    for run_spec, delphi_spec in zip(run_specs, delphi_specs, strict=True):
        resources = ResourceConfig.with_tpu(
            DEFAULT_TPU_TYPE,
            regions=[DEFAULT_TPU_REGION],
            zone=DEFAULT_TPU_ZONE,
        )
        training_step = ExecutorStep(
            name=f"{experiment_name}/{run_spec.run_name}",
            fn=remote(
                delphi.run_delphi_swarm_training,
                resources=resources,
                env_vars={HF_HUB_DISABLE_XET_ENV_VAR: "1"},
            ),
            resources=resources,
            config=delphi.DelphiSwarmTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=delphi_spec,
                validation_configs=validation_configs,
                wandb_tags=(
                    "bucket-epoch-dose-response",
                    "phase-tied",
                    f"stage={stage.value}",
                    f"seed_block={run_spec.seed_block}",
                    f"point={run_spec.point_id}",
                    f"epoch_multiplier={run_spec.epoch_multiplier:g}",
                ),
            ),
        )
        training_steps.append(training_step)
        eval_steps.append(
            olmo_base_eval_step(
                name=f"t9_{run_spec.run_name}",
                checkpoint=training_step / f"hf/step-{run_spec.expected_checkpoint_step}",
                request_set_dir=TABLE9_REQUEST_SET_DIR,
                resource_config=TABLE9_RESOURCES,
                wandb_group=f"olmo_base_eval_table9_bucket_epoch_dose_delphi_{stage.value}_20260729",
                provenance=_eval_provenance(run_spec, scale=Scale.DELPHI_3E18, stage=stage),
            )
        )
    return LaunchArtifacts(
        manifest_step=_manifest_step(Scale.DELPHI_3E18, stage, run_specs),
        training_steps=training_steps,
        eval_steps=eval_steps,
    )


def _build_table9_recovery_step(
    run_spec: EpochSweepRunSpec,
    *,
    checkpoint: str,
    scale: Scale,
    stage: Stage,
) -> ExecutorStep:
    if checkpoint.startswith("gs://"):
        raise ValueError("--eval-only-checkpoint must be relative to MARIN_PREFIX")
    if not checkpoint.endswith(f"/hf/step-{run_spec.expected_checkpoint_step}"):
        raise ValueError(
            f"--eval-only-checkpoint must end with /hf/step-{run_spec.expected_checkpoint_step}: {checkpoint}"
        )
    return olmo_base_eval_step(
        name=f"t9_{run_spec.run_name}",
        checkpoint=InputName.hardcoded(checkpoint),
        request_set_dir=TABLE9_REQUEST_SET_DIR,
        resource_config=TABLE9_RESOURCES,
        wandb_group=f"olmo_base_eval_table9_bucket_epoch_dose_{scale.value}_{stage.value}_20260729",
        provenance=_eval_provenance(run_spec, scale=scale, stage=stage),
    )


def _validate_graph(
    artifacts: LaunchArtifacts,
    run_specs: list[EpochSweepRunSpec],
    *,
    scale: Scale,
    stage: Stage,
) -> None:
    expected_runs = EXPECTED_PILOT_RUNS if stage == Stage.PILOT else EXPECTED_POINTS
    if len(run_specs) != expected_runs:
        raise ValueError(f"{scale}/{stage} has {len(run_specs)} run specs; expected {expected_runs}")
    if len(artifacts.training_steps) != expected_runs or len(artifacts.eval_steps) != expected_runs:
        raise ValueError(f"{scale} does not have one train and Table-9 step per point")
    if len({step.name for step in artifacts.training_steps}) != expected_runs:
        raise ValueError(f"{scale} has duplicate training step names")
    for run_spec in run_specs:
        if run_spec.scale != scale:
            raise ValueError(f"{run_spec.run_name} has scale {run_spec.scale}")
        if run_spec.stage != stage:
            raise ValueError(f"{run_spec.run_name} has stage {run_spec.stage}")
        if run_spec.data_seed != MAIN_DATA_SEED:
            raise ValueError(f"{run_spec.run_name} changed the fixed data-order seed")
        if not isinstance(run_spec.simulated_epoch_subset_seed, int):
            raise ValueError(f"{run_spec.run_name} has no explicit subset seed")
        if run_spec.phase_weights["phase_0"] != run_spec.phase_weights["phase_1"]:
            raise ValueError(f"{run_spec.run_name} is not phase tied")
        if stage == Stage.FULL and (
            run_spec.trainer_seed != MAIN_TRAINER_SEED
            or run_spec.simulated_epoch_subset_seed != MAIN_SIMULATED_EPOCH_SUBSET_SEED
        ):
            raise ValueError(f"{run_spec.run_name} changed a common-random-number seed")

    if stage == Stage.PILOT:
        seed_block_counts = Counter(spec.seed_block for spec in run_specs)
        expected_seed_block_counts = {
            "anchor_trainer_seed": len(PILOT_TRAINER_SEEDS),
            "anchor_subset_seed": len(PILOT_SUBSET_SEEDS) - 1,
            "high_replay_subset_seed": (
                len(PILOT_PROBE_DOMAINS) * len(PILOT_PROBE_MULTIPLIERS) * len(PILOT_PROBE_SUBSET_SEEDS)
            ),
            "low_dose_subset_seed": len(PILOT_PROBE_SUBSET_SEEDS),
            "high_replay_trainer_seed": len(PILOT_TRAINER_SEEDS) - 1,
        }
        if seed_block_counts != expected_seed_block_counts:
            raise ValueError(f"Unexpected pilot seed blocks: {seed_block_counts}")


def _write_local_manifest(scale: Scale, stage: Stage, run_specs: list[EpochSweepRunSpec]) -> None:
    output_path = REFERENCE_OUTPUT_DIR / stage / scale
    output_path.mkdir(parents=True, exist_ok=True)
    save_manifest(
        SaveManifestConfig(
            output_path=str(output_path),
            scale=scale,
            stage=stage,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
        )
    )


def _validate_full_gate(pilot_evidence: str | None, *, scale: Scale) -> None:
    if pilot_evidence is None:
        raise ValueError("--stage full requires --pilot-evidence pointing to a passed pilot gate artifact")
    with fsspec.open(pilot_evidence, "r") as handle:
        evidence = json.load(handle)
    if evidence.get("gate_status") != "pass":
        raise ValueError(f"Pilot gate status is not pass in {pilot_evidence}")
    approved_scales = evidence.get("approved_scales")
    if not isinstance(approved_scales, list) or scale.value not in approved_scales:
        raise ValueError(f"Pilot gate does not approve scale {scale.value}")
    if not evidence.get("pilot_experiment_id"):
        raise ValueError("Pilot gate artifact is missing pilot_experiment_id provenance")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=[scale.value for scale in Scale], required=True)
    parser.add_argument("--stage", choices=[stage.value for stage in Stage], default=Stage.PILOT)
    parser.add_argument("--analysis-output-path", default=delphi.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--pilot-evidence")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--eval-only-run-name")
    parser.add_argument("--eval-only-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    scale = Scale(args.scale)
    stage = Stage(args.stage)

    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    if args.max_concurrent < 1 or args.max_concurrent > DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    expected_prefix = marin_prefix_for_region(DEFAULT_TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    if stage == Stage.FULL:
        _validate_full_gate(args.pilot_evidence, scale=scale)
    elif args.pilot_evidence is not None:
        raise ValueError("--pilot-evidence is only valid with --stage full")
    eval_only = args.eval_only_run_name is not None or args.eval_only_checkpoint is not None
    if eval_only and (args.eval_only_run_name is None or args.eval_only_checkpoint is None):
        raise ValueError("--eval-only-run-name and --eval-only-checkpoint must be specified together")

    points = build_points()
    with executor_context():
        if scale == Scale.SIXTY_M:
            full_run_specs = _run_specs_60m(points, stage=Stage.FULL)
            run_specs = _pilot_run_specs(full_run_specs, scale=scale) if stage == Stage.PILOT else full_run_specs
        else:
            full_run_specs = _run_specs_delphi(
                points,
                analysis_output_path=args.analysis_output_path,
                stage=Stage.FULL,
            )
            run_specs = _pilot_run_specs(full_run_specs, scale=scale) if stage == Stage.PILOT else full_run_specs
        if eval_only:
            matching_specs = [spec for spec in run_specs if spec.run_name == args.eval_only_run_name]
            if len(matching_specs) != 1:
                raise ValueError(
                    f"Expected exactly one run named {args.eval_only_run_name!r}; found {len(matching_specs)}"
                )
            recovery_step = _build_table9_recovery_step(
                matching_specs[0],
                checkpoint=args.eval_only_checkpoint,
                scale=scale,
                stage=stage,
            )
        elif scale == Scale.SIXTY_M:
            artifacts = _build_60m_artifacts(run_specs, stage=stage)
        else:
            artifacts = _build_delphi_artifacts(
                run_specs,
                analysis_output_path=args.analysis_output_path,
                stage=stage,
            )
    if eval_only:
        logger.info(
            "Validated native Table-9 recovery for %s from %s.",
            args.eval_only_run_name,
            args.eval_only_checkpoint,
        )
        if args.dry_run or os.getenv("CI") is not None:
            return
        recovery_description = (
            f"{PANEL_TAG}: recover native Table-9 evaluation for completed checkpoint {args.eval_only_run_name}."
        )
        executor_main(
            ExecutorMainConfig(max_concurrent=1),
            steps=[recovery_step],
            description=recovery_description,
        )
        return
    _validate_graph(artifacts, run_specs, scale=scale, stage=stage)
    _write_local_manifest(scale, stage, run_specs)
    logger.info(
        "Validated %d tied %s/%s training runs and %d native Table-9 evaluations.",
        len(artifacts.training_steps),
        scale,
        stage,
        len(artifacts.eval_steps),
    )
    if args.dry_run or os.getenv("CI") is not None:
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{PANEL_TAG}: {scale}/{stage} conditional one-bucket epoch dose responses around proportional; "
            "every tied training coordinate receives Uncheatable validation and native Table-9 evaluation."
        ),
    )


if __name__ == "__main__":
    main()
