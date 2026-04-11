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
from experiments.domain_phase_mix.qsplit240_replay import resolve_qsplit240_eval_cache_path
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    EXPERIMENT_BUDGET as REGMIX_60M_1P2B_BUDGET,
    STRATIFIED_RUN_NAME,
    create_stratified_weight_config,
    create_two_phase_dolma3_dolmino_top_level_experiment,
)

logger = logging.getLogger(__name__)

EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/qsplit240-300m-6b-expanded-tasks"


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
    tpu_region: str
    tpu_zone: str


SCALE_SPECS = {
    StratifiedScale.REGMIX_60M_1P2B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_60M_1P2B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
        experiment_budget=REGMIX_60M_1P2B_BUDGET,
        model_config=regmix_60m_proxy,
        optimizer_config=None,
        tpu_type="v5p-8",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
    ),
    StratifiedScale.REGMIX_300M_6B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_300M_6B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b",
        experiment_budget=6_000_000_000,
        model_config=regmix_300m_proxy,
        optimizer_config=regmix_300m_muonh_base,
        tpu_type="v5p-8",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
    ),
    StratifiedScale.REGMIX_520M_10P4B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_520M_10P4B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_520m_10p4b",
        experiment_budget=REGMIX_520M_CHINCHILLA_BUDGET,
        model_config=regmix_520m_proxy,
        optimizer_config=regmix_520m_muonh_base,
        tpu_type="v5p-32",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
    ),
    StratifiedScale.REGMIX_1_2B_24B: StratifiedScaleSpec(
        scale=StratifiedScale.REGMIX_1_2B_24B,
        name_prefix="pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_1_2b_24b",
        experiment_budget=REGMIX_1_2B_CHINCHILLA_BUDGET,
        model_config=regmix_1_2b_proxy,
        optimizer_config=regmix_1_2b_muonh_base,
        tpu_type="v5p-64",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
    ),
}


def resolve_scale_spec(scale: StratifiedScale) -> StratifiedScaleSpec:
    """Return the canonical launch recipe for one supported scale."""
    return SCALE_SPECS[scale]


def _resolve_eval_cache_path_for_region(eval_datasets_cache_path: str, tpu_region: str) -> str:
    resolved_path = resolve_qsplit240_eval_cache_path(eval_datasets_cache_path)
    source_prefix = "gs://marin-us-central1/"
    if resolved_path.startswith(source_prefix):
        return resolved_path.replace(source_prefix, f"gs://marin-{tpu_region}/", 1)
    return resolved_path


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=StratifiedScale, choices=list(StratifiedScale), required=True)
    parser.add_argument("--name-prefix")
    parser.add_argument("--tpu-type")
    parser.add_argument("--tpu-region")
    parser.add_argument("--tpu-zone")
    parser.add_argument("--eval-datasets-cache-path", default=EVAL_DATASETS_CACHE_PATH)
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    spec = resolve_scale_spec(args.scale)
    name_prefix = args.name_prefix or spec.name_prefix
    tpu_type = args.tpu_type or spec.tpu_type
    tpu_region = args.tpu_region or spec.tpu_region
    tpu_zone = args.tpu_zone or spec.tpu_zone

    if os.getenv("CI") is not None:
        logger.info("Skipping stratified baseline launch in CI environment for scale %s", spec.scale)
        return

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
        name=name_prefix,
        experiment_budget=spec.experiment_budget,
        model_config=spec.model_config,
        optimizer_config=spec.optimizer_config,
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        eval_harness_tasks=QSPLIT240_300M_EVAL_TASKS,
        eval_datasets_cache_path=_resolve_eval_cache_path_for_region(args.eval_datasets_cache_path, tpu_region),
        runtime_cache_region=tpu_region,
    )
    training_step = experiment.create_training_step(
        weight_config=create_stratified_weight_config(),
        name_prefix=name_prefix,
        run_name=STRATIFIED_RUN_NAME,
    )
    logger.info(
        "Launching stratified baseline on %s with budget=%d, tpu=%s, region=%s, zone=%s",
        spec.scale,
        spec.experiment_budget,
        tpu_type,
        tpu_region,
        tpu_zone,
    )
    executor_main(
        ExecutorMainConfig(max_concurrent=1),
        steps=[training_step],
        description=f"{name_prefix}: {STRATIFIED_RUN_NAME} ({spec.scale})",
    )


if __name__ == "__main__":
    main()
