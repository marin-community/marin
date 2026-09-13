# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate-hit counts along recorded cross-tokenizer token paths.

Each completion value is ``{"n_steps": int, "n_below": int}``: how many
decision steps the completion recorded, and how many of them carried a signal
strictly below a threshold. Counting only — the signals were computed at
generation time and are read back from the completion algorithm's
``token_paths.jsonl.gz`` sidecar.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.downstream_scaling.evals.measurements.schema import statistics_file
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"


@dataclass(frozen=True)
class GateHits:
    """Count recorded token-path steps on the decoder side of a threshold.

    Each completion value is ``{"n_steps": int, "n_below": int}``: the number
    of recorded decision steps, and how many had ``step[field]`` strictly below
    ``threshold`` (the steps a ``GateDirection.ADVISOR_AT_OR_ABOVE`` sweep
    routes to the decoder). ``kl_bytes_union`` can emit ``inf``; such a record
    fails the read loudly — ``load_jsonl`` decodes with strict msgspec, which
    rejects ``Infinity`` — rather than being counted.

    Every completion in the sidecar is counted against ``threshold``, including
    completions generated at other thresholds; the off-diagonal is
    counterfactual, and selecting the diagonal is the consumer's job.
    """

    # The per-step signal key in the sidecar: "kl" or "entropy".
    field: str
    threshold: float
    aggregate_workers: int = 32

    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,  # Statistic protocol; counting needs no prompts
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_gate_hits_step(name=name, alg_output_path=alg_output_path, statistic=self)


@dataclass(frozen=True)
class GateHitsStepConfig:
    output_path: str
    alg_output_path: str
    field: str
    threshold: float
    aggregate_workers: int


def make_gate_hits_step(
    *,
    name: str,
    alg_output_path: str | InputName | MirroredValue,
    statistic: GateHits,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(run_gate_hits, resources=ResourceConfig.with_cpu(cpu=1, ram="4g")),
        config=GateHitsStepConfig(
            output_path=this_output_path(),
            alg_output_path=version_path(alg_output_path),  # type: ignore[arg-type]
            field=versioned(statistic.field),  # type: ignore[arg-type]
            threshold=versioned(statistic.threshold),  # type: ignore[arg-type]
            aggregate_workers=statistic.aggregate_workers,
        ),
    )


def _gate_hit_record(record: Any, *, field: str, threshold: float) -> dict[str, Any]:
    """Validate one token-path record and reduce it to its gate-hit counts."""
    completion_index = record.get("completion_index") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("id"), str)
        or not isinstance(completion_index, int)
        or isinstance(completion_index, bool)
        or completion_index < 0
        or not isinstance(record.get("steps"), list)
    ):
        raise TypeError(f"Invalid token-path record: {record!r}")

    signals = [step[field] for step in record["steps"]]
    if any(not isinstance(signal, float | int) or isinstance(signal, bool) for signal in signals):
        raise TypeError(f"Non-numeric {field!r} in token-path record {(record['id'], completion_index)!r}")

    return {
        "id": record["id"],
        "completion_index": completion_index,
        "value": {"n_steps": len(signals), "n_below": sum(1 for signal in signals if signal < threshold)},
    }


def _aggregate_statistic_row(
    prompt_id: str,
    items: Iterator[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    records = list(items)
    indices = [record["completion_index"] for record in records]
    if indices != list(range(len(records))):
        raise ValueError(f"Statistic completion indices for {prompt_id!r} are {indices}, expected 0..{len(records) - 1}")
    return {
        "id": prompt_id,
        "values": [record["value"] for record in records],
        "metadata": metadata,
    }


def run_gate_hits(config: GateHitsStepConfig) -> None:
    token_paths_path = os.path.join(config.alg_output_path, TOKEN_PATHS_FILENAME)
    metadata = {"field": config.field, "threshold": config.threshold}
    path = statistics_file(config.output_path)
    pipeline = (
        Dataset.from_files(token_paths_path)
        .load_jsonl()
        .map(functools.partial(_gate_hit_record, field=config.field, threshold=config.threshold))
        .group_by(
            key=lambda record: record["id"],
            reducer=functools.partial(_aggregate_statistic_row, metadata=metadata),
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="gate-hits-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)
    logger.info("Wrote gate-hit statistics to %s", path)
