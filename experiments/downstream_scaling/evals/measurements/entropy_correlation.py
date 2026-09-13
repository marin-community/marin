# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-completion correlation between decoder and advisor token-path entropy."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue, output_path_of
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.downstream_scaling.evals.measurements.entropy import (
    TokenPathEntropy,
    TokenPathExecutionConfig,
    TokenPathModelConfig,
    TokenPathSide,
)
from experiments.downstream_scaling.evals.measurements.schema import STATISTICS_FILENAME, StatisticRow, statistics_file
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenPathEntropyCorrelation:
    decoder_model: TokenPathModelConfig
    advisor_model: TokenPathModelConfig
    execution: TokenPathExecutionConfig
    entropy_name: str
    k: int = 16
    aggregate_workers: int = 32

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1 (got {self.k})")
        if self.aggregate_workers < 1:
            raise ValueError(f"aggregate_workers must be >= 1 (got {self.aggregate_workers})")

    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        decoder = TokenPathEntropy(
            model=self.decoder_model,
            execution=self.execution,
            side=TokenPathSide.A,
            k=self.k,
        ).make_statistic_step(
            name=f"{self.entropy_name}/decoder",
            prompts_path=prompts_path,
            alg_output_path=alg_output_path,
        )
        advisor = TokenPathEntropy(
            model=self.advisor_model,
            execution=self.execution,
            side=TokenPathSide.B,
            k=self.k,
        ).make_statistic_step(
            name=f"{self.entropy_name}/advisor",
            prompts_path=prompts_path,
            alg_output_path=alg_output_path,
        )
        return make_entropy_correlation_step(
            name=name,
            decoder_statistics_path=output_path_of(decoder) / STATISTICS_FILENAME,
            advisor_statistics_path=output_path_of(advisor) / STATISTICS_FILENAME,
            aggregate_workers=self.aggregate_workers,
        )


@dataclass(frozen=True)
class EntropyCorrelationStepConfig:
    output_path: str
    decoder_statistics_path: str
    advisor_statistics_path: str
    aggregate_workers: int


def make_entropy_correlation_step(
    *,
    name: str,
    decoder_statistics_path: str | InputName | MirroredValue,
    advisor_statistics_path: str | InputName | MirroredValue,
    aggregate_workers: int,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            run_entropy_correlation,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g", preemptible=True),
            pip_dependency_groups=["datakit"],
        ),
        config=EntropyCorrelationStepConfig(
            output_path=this_output_path(),
            decoder_statistics_path=version_path(decoder_statistics_path),  # type: ignore[arg-type]
            advisor_statistics_path=version_path(advisor_statistics_path),  # type: ignore[arg-type]
            aggregate_workers=aggregate_workers,
        ),
    )


def _correlate(prompt_id: str, items: Iterator[StatisticRow]) -> dict[str, Any]:
    rows_by_side: dict[str, StatisticRow] = {}
    for row in items:
        side = row.get("metadata", {}).get("side")
        if side not in (TokenPathSide.A.value, TokenPathSide.B.value):
            raise ValueError(f"Entropy row for {prompt_id!r} has invalid side {side!r}")
        if side in rows_by_side:
            raise ValueError(f"Duplicate side {side!r} for entropy correlation row {prompt_id!r}")
        rows_by_side[side] = row

    missing_sides = {TokenPathSide.A.value, TokenPathSide.B.value} - rows_by_side.keys()
    if missing_sides:
        raise ValueError(f"Missing sides {sorted(missing_sides)!r} for entropy correlation row {prompt_id!r}")

    decoder_values = rows_by_side[TokenPathSide.A.value]["values"]
    advisor_values = rows_by_side[TokenPathSide.B.value]["values"]
    if len(decoder_values) != len(advisor_values):
        raise ValueError(
            f"Entropy value count mismatch for {prompt_id!r}: {len(decoder_values)} decoder, "
            f"{len(advisor_values)} advisor"
        )

    from scipy.stats import spearmanr  # noqa: PLC0415  # optional dep: datakit extra

    values = []
    for completion_index, (decoder_value, advisor_value) in enumerate(zip(decoder_values, advisor_values, strict=True)):
        decoder_entropy = decoder_value["entropy"]
        advisor_entropy = advisor_value["entropy"]
        if len(decoder_entropy) != len(advisor_entropy):
            raise ValueError(
                f"Entropy series length mismatch for {(prompt_id, completion_index)!r}: "
                f"{len(decoder_entropy)} decoder, {len(advisor_entropy)} advisor"
            )
        values.append(
            {
                "n_steps": len(decoder_entropy),
                "rho": float(spearmanr(decoder_entropy, advisor_entropy).statistic),
            }
        )

    return {
        "id": prompt_id,
        "values": values,
        "metadata": {"statistic": "token_path_entropy_correlation", "estimator": "spearman"},
    }


def run_entropy_correlation(config: EntropyCorrelationStepConfig) -> None:
    path = statistics_file(config.output_path)
    pipeline = (
        Dataset.from_list([config.decoder_statistics_path, config.advisor_statistics_path])
        .load_jsonl()
        .group_by(key=lambda row: row["id"], reducer=_correlate, num_output_shards=1)
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="token-path-entropy-correlation",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote token-path entropy correlations to %s", path)
