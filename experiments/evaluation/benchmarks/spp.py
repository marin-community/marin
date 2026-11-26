# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Persona Pretraining value benchmarks (Minder et al., arXiv:2608.13482).

Both benchmarks reproduce the authors' harness, epfl-dlab/spp-evals at cf545a0, and score the served model's
log-probabilities after its own chat template; they are meant for chat models. Neither uses a judge.

``spp-constitution`` is ConstitutionEval: 678 four-option scenarios whose single gold option follows the SPP
constitution, from `dlab-spp/constitution-eval` (CC-BY-4.0) at the harness's v1.4 pin. Each option sums the
log of its first-token letter probability, pooled over the bare and leading-space spellings that the served
tokenizer encodes as one token, across all four cyclic option orders. Accuracy is reported overall and by the
dataset's difficulty band.

``spp-airisk`` is the risky-choice rate on AIRiskDilemmas (Chiu et al., 2025), from the audited re-release
`dlab-spp/airisk_dilemmas` (CC-BY-4.0) with audit-confirmed risk tags. The population is the 1,281 dilemmas of the
harness's seeded 3,000-dilemma sample in which at least one action is risky; dilemmas with two risky actions count
as risky either way. Each action sums its " Action N" log-likelihood across both label orders. Lower is better.

Both protocols compare near-equal log-probabilities, so bfloat16 serving rounds many items to exact ties, which go
to the first choice as in the reference harness. Serving in float32 removes them.
"""

from functools import partial

from finestore.rl.curation import SourceOrigin
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metric,
    MetricKind,
    OperationKind,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.spp import (
    SPP_TAGS,
    airisk_item,
    constitution_item,
    convert_spp_document,
    read_airisk_file,
    read_constitution_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

VERSION = "2026.11.23"
CONSTITUTION_REVISION = "6957c3b89e21fd72bc88b5b841948684e6dbacc3"
CONSTITUTION_POPULATION = 678
AIRISK_REVISION = "b93c856a61e9a59680c6c947c7d040aa93a75ee4"
AIRISK_POPULATION = 1281


def _source(
    *,
    name: str,
    upstream_name: str,
    source: HfSource,
    item_builder,
    metric: Metric,
    population: int,
) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            origin=SourceOrigin.DLAB_SPP,
            upstream_dataset=source.repo.split("/", 1)[1],
            upstream_name=upstream_name,
            build_path="SPP",
            family="likelihood-choice",
            tags=(*SPP_TAGS, "single-turn", "benchmark"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="spp-" + name,
                source=source,
                convert=partial(convert_spp_document, benchmark=name),
                version=VERSION,
                intended_use=IntendedUse.EVAL,
            ),
            protocol=EvaluationProtocol(
                benchmark="spp-" + name,
                version=VERSION,
                operation=OperationKind.CHOICE_LIKELIHOOD,
                sampling=None,
                trials=TrialPolicy(count=1, seed=None),
                metrics=(metric,),
                primary_metric=metric.name,
                num_fewshot=0,
                summary=SummaryKind.MEAN,
                expected_population=population,
                likelihood_metrics=(LikelihoodMetric(name=metric.name, normalization=LikelihoodNormalization.RAW),),
            ),
            item_builder=item_builder,
            protocol_name="spp",
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        _source(
            name="constitution",
            upstream_name="ConstitutionEval",
            source=HfSource(
                "dlab-spp/constitution-eval",
                CONSTITUTION_REVISION,
                ("data/train-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
                read=read_constitution_file,
            ),
            item_builder=constitution_item,
            metric=Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),
            population=CONSTITUTION_POPULATION,
        ),
        _source(
            name="airisk",
            upstream_name="AIRiskDilemmas",
            source=HfSource(
                "dlab-spp/airisk_dilemmas",
                AIRISK_REVISION,
                ("full_audited.jsonl",),
                SourceFormat.JSONL,
                read=read_airisk_file,
            ),
            item_builder=airisk_item,
            # The operation's gold choices are the risky actions, so selecting one is the misaligned outcome.
            metric=Metric(name="risky_choice", kind=MetricKind.BINARY, higher_is_better=False),
            population=AIRISK_POPULATION,
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())
