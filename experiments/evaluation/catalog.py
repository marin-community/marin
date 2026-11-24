# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-owned protocol builders for the canonical evaluation catalog."""

from collections import defaultdict
from functools import cache, partial

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.benchmarks import (
    agentic,
    biomedical,
    code,
    commonsense,
    econevals,
    facts_grounding,
    financebench,
    genomeqa,
    gsm8k,
    instruction_following,
    jee,
    lambada,
    lexam,
    livecodebench,
    long_context,
    longbench,
    math,
    mmlu,
    mmlu_redux,
    multichallenge,
    multiple_choice,
    nupa,
    physics_verified,
    question_answering,
    reasoning,
    scienceagentbench,
    simpleqa,
    supergpqa,
)
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
            mmlu_redux,
            lexam,
            code,
            livecodebench,
            commonsense,
            biomedical,
            instruction_following,
            jee,
            math,
            reasoning,
            financebench,
            facts_grounding,
            genomeqa,
            scienceagentbench,
            simpleqa,
            multichallenge,
            econevals,
            long_context,
            longbench,
            nupa,
            agentic,
            supergpqa,
            physics_verified,
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
