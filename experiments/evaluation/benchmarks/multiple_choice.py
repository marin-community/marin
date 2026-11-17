# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multiple-choice source data and supported lm-eval configurations."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

from taskcompendium.evaluation.spec import LikelihoodNormalization, MetricKind
from taskcompendium.importers.evaluation_harness import HarnessTemplate
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.evaluation.prompt_compilation import lm_eval_multiple_choice
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import RlDataSource


@dataclass(frozen=True)
class MultipleChoiceDataset:
    name: str
    benchmark: str
    source: HfSource
    population: int
    pool_file: str


def _compile(
    dataset: MultipleChoiceDataset,
    variant: ProtocolVariant,
    *,
    normalizations: tuple[LikelihoodNormalization, ...],
    metric_kind: MetricKind,
    scoring: Literal["most_likely", "probability_mass"],
    fewshot_version: str,
) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    source = dataset.source
    shots = variant.num_fewshot
    assert shots is not None
    compiled = lm_eval_multiple_choice(
        name=dataset.name,
        benchmark=dataset.benchmark,
        source=source,
        population=dataset.population,
        template=HarnessTemplate(
            name=dataset.name, num_fewshot=shots, fewshot_file=dataset.pool_file or source.files[0]
        ),
        normalizations=normalizations,
        metric_kind=metric_kind,
        scoring=scoring,
        version=fewshot_version if shots else "2026.11.02",
    )
    return (compiled,)


def _benchmark(
    *,
    dataset: MultipleChoiceDataset,
    shots: tuple[int, ...],
    normalizations: tuple[LikelihoodNormalization, ...],
    metric_kind: MetricKind = MetricKind.BINARY,
    scoring: Literal["most_likely", "probability_mass"] = "most_likely",
    fewshot_version: str = "2026.11.02",
) -> EvaluationBenchmark:
    return EvaluationBenchmark(
        name=dataset.benchmark,
        variants=tuple(ProtocolVariant("harness", count, None) for count in shots),
        build=partial(
            _compile,
            dataset,
            normalizations=normalizations,
            metric_kind=metric_kind,
            scoring=scoring,
            fewshot_version=fewshot_version,
        ),
    )


def benchmarks() -> list[EvaluationBenchmark]:
    raw = (LikelihoodNormalization.RAW,)
    normalized = (LikelihoodNormalization.RAW, LikelihoodNormalization.CHARACTERS)
    return [
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="arc_challenge",
                benchmark="arc-challenge",
                source=HfSource(
                    "allenai/ai2_arc",
                    "210d026faf9955653af8916fad021475a3f00453",
                    ("ARC-Challenge/test-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=1172,
                pool_file="ARC-Challenge/train-00000-of-00001.parquet",
            ),
            shots=(25, 10),
            normalizations=normalized,
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="arc_easy",
                benchmark="arc-easy",
                source=HfSource(
                    "allenai/ai2_arc",
                    "210d026faf9955653af8916fad021475a3f00453",
                    ("ARC-Easy/test-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=2376,
                pool_file="ARC-Easy/train-00000-of-00001.parquet",
            ),
            shots=(0, 10),
            normalizations=normalized,
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="hellaswag",
                benchmark="hellaswag",
                source=HfSource(
                    "Rowan/hellaswag",
                    "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                    ("data/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=10042,
                pool_file="data/train-00000-of-00001.parquet",
            ),
            shots=(10, 0),
            normalizations=normalized,
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="boolq",
                benchmark="boolq",
                source=HfSource(
                    "aps/super_glue",
                    "3de24cf8022e94f4ee4b9d55a6f539891524d646",
                    ("boolq/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=3270,
                pool_file="boolq/train-00000-of-00001.parquet",
            ),
            shots=(0, 10),
            normalizations=raw,
            fewshot_version="2026.11.05.2",
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="piqa",
                benchmark="piqa",
                source=HfSource(
                    "baber/piqa",
                    "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
                    ("piqa_validation.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=1838,
                pool_file="piqa_train.parquet",
            ),
            shots=(0, 10),
            normalizations=(LikelihoodNormalization.CHARACTERS, LikelihoodNormalization.RAW),
            fewshot_version="2026.11.05.2",
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="openbookqa",
                benchmark="openbookqa",
                source=HfSource(
                    "allenai/openbookqa",
                    "388097ea7776314e93a529163e0fea805b8a6454",
                    ("main/test-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=500,
                pool_file="",
            ),
            shots=(0,),
            normalizations=normalized,
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="winogrande",
                benchmark="winogrande",
                source=HfSource(
                    "allenai/winogrande",
                    "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                    ("winogrande_xl/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=1267,
                pool_file="winogrande_xl/train-00000-of-00001.parquet",
            ),
            shots=(5, 0),
            normalizations=raw,
        ),
        _benchmark(
            dataset=MultipleChoiceDataset(
                name="truthfulqa_mc2",
                benchmark="truthfulqa",
                source=HfSource(
                    "truthfulqa/truthful_qa",
                    "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
                    ("multiple_choice/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=817,
                pool_file="",
            ),
            shots=(0, 6),
            normalizations=raw,
            metric_kind=MetricKind.CONTINUOUS,
            scoring="probability_mass",
            fewshot_version="2026.11.05.2",
        ),
    ]
