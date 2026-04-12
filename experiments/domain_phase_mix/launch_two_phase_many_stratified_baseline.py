# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a standalone stratified baseline on one scaling-study model size."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum

from fray.cluster import ResourceConfig
from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig
from marin.execution.executor import ExecutorMainConfig, executor_main

from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import QSPLIT240_300M_EVAL_TASKS
from experiments.domain_phase_mix.proxy_sweep import (
    REGMIX_1_2B_CHINCHILLA_BUDGET,
    REGMIX_520M_CHINCHILLA_BUDGET,
    regmix_1_2b_muonh_base,
    regmix_1_2b_proxy,
    regmix_300m_muonh_base,
    regmix_300m_proxy,
    regmix_520m_muonh_base,
    regmix_520m_proxy,
    regmix_60m_proxy,
)
from experiments.domain_phase_mix.qsplit240_replay import (
    DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
    mirror_path,
    normalize_tpu_regions,
    resolve_latest_checkpoint_root,
    resolve_qsplit240_eval_cache_path_for_regions,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    EXPERIMENT_BUDGET as REGMIX_60M_1P2B_BUDGET,
    STRATIFIED_RUN_NAME,
    create_stratified_weight_config,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"
DEFAULT_RESUME_LATEST_CHECKPOINTS = True


class StratifiedScale(StrEnum):
    """Supported scaling-study ladders for the standalone stratified baseline."""

    REGMIX_60M_1P2B = "60m_1p2b"
    REGMIX_300M_6B = "300m_6b"
    REGMIX_520M_10P4B = "520m_10p4b"
    REGMIX_1_2B_24B = "1_2b_24b"


@dataclass(frozen=True)
class StratifiedScaleSpec:
    """Resolved training recipe for one stratified baseline launch."""

    scale: StratifiedScale
    name_prefix: str
    experiment_budget: int
    model_config: LmConfig
    optimizer_config: MuonHConfig | None
    tpu_type: str
    tpu_regions: tuple[str, ...]
    tpu_zone: str | None


SCALE_SPECS = {
    StratifiedScale.REGMIX_60M_1P2B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_60M_1P2B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
        experiment_budget=REGMIX_60M_1P2B_BUDGET,
        model_config=regmix_60m_proxy,
        optimizer_config=None,
        tpu_type="v5p-8",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    StratifiedScale.REGMIX_300M_6B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_300M_6B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
        experiment_budget=6_000_000_000,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type="v5p-8",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    StratifiedScale.REGMIX_520M_10P4B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_520M_10P4B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b",
        experiment_budget=REGMIX_520M_CHINCHILLA_BUDGET,
        model_config=regmix_520m_proxy,
        optimizer_config=regmix_520m_muonh_base,
        tpu_type="v5p-32",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
    StratifiedScale.REGMIX_1_2B_24B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_1_2B_24B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b",
        experiment_budget=REGMIX_1_2B_CHINCHILLA_BUDGET,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type="v5p-64",
        tpu_regions=DEFAULT_REGION_AGNOSTIC_TPU_REGIONS,
        tpu_zone=None,
    ),
}


def resolve_scale_spec(scale: StratifiedScale) -> StratifiedScaleSpec:
    """Return the canonical launch recipe for one supported scale."""
    return SCALE_SPECS[scale]


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=StratifiedScale, choices=list(StratifiedScale), required=True)
    parser.add_argument("--name-prefix")
    parser.add_argument("--tpu-type")
    parser.add_argument("--tpu-regions")
    parser.add_argument("--tpu-region")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    parser.add_argument(
        "--resume-latest-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME_LATEST_CHECKPOINTS,
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    spec = resolve_scale_spec(args.scale)
    name_prefix = args.name_prefix or spec.name_prefix
    tpu_type = args.tpu_type or spec.tpu_type
    tpu_regions = normalize_tpu_regions(args.tpu_region or args.tpu_regions or spec.tpu_regions)
    tpu_zone = args.tpu_zone or spec.tpu_zone

    if os.getenv("CI") is not None:
        logger.info("Skipping stratified baseline launch in CI environment for scale %s", spec.scale)
        return

    train_kwargs: dict[str, object] = {}
    if args.resume_latest_checkpoints:
        latest_checkpoint_root = resolve_latest_checkpoint_root(
            experiment_name_prefix=name_prefix,
            run_name=STRATIFIED_RUN_NAME,
            checkpoint_regions=tpu_regions,
        )
        if latest_checkpoint_root is not None:
            train_kwargs["initialize_from_checkpoint_path"] = mirror_path(latest_checkpoint_root)
            train_kwargs["reset_data_loader_on_init"] = False

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=spec.experiment_budget,
        model_config=spec.model_config,
        optimizer_config=spec.optimizer_config,
        resources=ResourceConfig.with_tpu(tpu_type, regions=list(tpu_regions), zone=tpu_zone),
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=resolve_qsplit240_eval_cache_path_for_regions(
            tpu_regions,
            args.eval_datasets_cache_path,
        ),
        runtime_cache_region=tpu_regions if len(tpu_regions) > 1 else tpu_regions[0],
    )
    training_step = experiment.create_training_step(
        weight_config=create_stratified_weight_config(),
        name_prefix=name_prefix,
        run_name=STRATIFIED_RUN_NAME,
        **train_kwargs,
    )
    logger.info(
        "Launching stratified baseline on %s with budget=%d, tpu=%s, regions=%s, zone=%s",
        spec.scale,
        spec.experiment_budget,
        tpu_type,
        ",".join(tpu_regions),
        tpu_zone,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=1),
        steps=[training_step],
        description=f"{name_prefix}: {STRATIFIED_RUN_NAME} ({spec.scale})",
    )


if __name__ == "__main__":
    main()
