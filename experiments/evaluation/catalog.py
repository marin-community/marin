# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-owned protocol builders for the canonical evaluation catalog."""

from collections import defaultdict
from functools import cache, partial

from experiments.evaluation.benchmarks import (
    code,
    commonsense,
    financebench,
    gsm8k,
    instruction_following,
    lambada,
    multiple_choice,
    question_answering,
    jee,
    long_context,
    math,
    mmlu,
    nupa,
    reasoning,
    simpleqa,
)
from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.source import RlDataSource


def _build_protocol(
    builders: dict[ProtocolVariant, EvaluationBenchmark],
    variant: ProtocolVariant,
) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    return builders[variant].build(variant)


@cache
def evaluation_benchmarks() -> dict[str, EvaluationBenchmark]:
    declarations = [
        benchmark
        for family in (
            multiple_choice,
            question_answering,
            gsm8k,
            lambada,
            mmlu,
            code,
            commonsense,
            instruction_following,
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
