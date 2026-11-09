# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-owned protocol builders for the canonical evaluation catalog."""

from collections import defaultdict
from functools import cache, partial

from experiments.post_training.task_curation.datasets.evalchemy import (
    code,
    commonsense,
    financebench,
    harness,
    instructions,
    jee,
    long_context,
    math,
    mmlu,
    nupa,
    reasoning,
    simpleqa,
)
from experiments.post_training.task_curation.evaluation import (
    EvalDataset,
    EvaluationBenchmark,
    ProtocolVariant,
)
from experiments.post_training.task_curation.source import RlDataSource


def _build_protocol(
    builders: dict[ProtocolVariant, EvaluationBenchmark],
    variant: ProtocolVariant,
) -> tuple[RlDataSource[EvalDataset], ...]:
    return builders[variant].build(variant)


@cache
def evaluation_benchmarks() -> dict[str, EvaluationBenchmark]:
    declarations = [
        benchmark
        for family in (
            harness,
            mmlu,
            code,
            commonsense,
            instructions,
            jee,
            math,
            reasoning,
            financebench,
            simpleqa,
            long_context,
            nupa,
        )
        for benchmark in family.benchmarks()
    ]
    grouped = defaultdict(dict)
    for declaration in declarations:
        for variant in declaration.variants:
            if variant in grouped[declaration.name]:
                raise ValueError(f"Duplicate protocol builder: {declaration.name}/{variant.protocol}")
            grouped[declaration.name][variant] = declaration
    return {
        name: EvaluationBenchmark(name, tuple(builders), partial(_build_protocol, builders))
        for name, builders in grouped.items()
    }
