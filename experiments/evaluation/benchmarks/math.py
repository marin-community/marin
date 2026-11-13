# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Evalchemy math populations and source sampling policies."""

from dataclasses import replace
from functools import partial

from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationProtocol, Metrics, OperationKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.math import (
    boxed_math_task,
    convert_amc,
    convert_olympiad,
    minerva_template,
)
from taskcompendium.importers.evalchemy.scorers.math import CHAT_REQUEST_STOPS
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import MATH_GRADER, MATH_JUDGE_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.evaluation.prompt_compilation import text_generation_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MARIN_SOURCE_REVISION = "623134de90723e7f189036da005067f36817b6e7"
AIME24_SHA256 = "2dfe4f134aee7989cd9978218ca914b3c3db4685ea06d841560ec1391ba0a409"
OLYMPIAD_SHA256 = "bcafaabc0128adfeef69c830d1682254428887d4e1b96b5b1b03fb5d3818f958"
AIME25_SHA256 = "9affac318c4dd399482a67226936e229f954628567c7d7b7ee8f1360c15d6f7c"
AMC23_SHA256 = "02ae54425741562778916b5fa8f6ff07749689cfcd7a53e65e1d88a89e5065f9"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"


def fewshot_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    result = []
    for subject, population in (
        ("algebra", 1187),
        ("counting_and_probability", 474),
        ("geometry", 479),
        ("intermediate_algebra", 903),
        ("number_theory", 540),
        ("prealgebra", 871),
        ("precalculus", 546),
    ):
        result.append(
            text_generation_evaluation(
                name="minerva_math-" + subject,
                benchmark="minerva_math",
                source=HfSource(
                    "EleutherAI/hendrycks_math",
                    MATH_REVISION,
                    (subject + "/test-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=population,
                template=minerva_template(subject),
                metrics=("exact_match,none", "math_verify,none"),
                sampling=Sampling(
                    temperature=0,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=("Problem:",),
                    chat_template_kwargs={},
                ),
                scorer="minerva_math",
                grader=MATH_GRADER,
            )
        )
    return result


def bundled_source(directory: str, filename: str, sha256: str) -> UrlSource:
    return UrlSource(
        f"https://raw.githubusercontent.com/marin-community/marin/{MARIN_SOURCE_REVISION}/"
        f"vendor/evalchemy/eval/chat_benchmarks/{directory}/data/{filename}",
        sha256,
        filename + ".jsonl",
        SourceFormat.JSONL,
    )


def math_dataset(
    *,
    name: str,
    benchmark: str,
    source: HfSource | UrlSource,
    answer_field: str,
    population: int,
    trials: int,
    temperature: float,
) -> RlDataSource[EvaluationCorpusRecipe]:
    recipe = CurationRecipe(
        name="evalchemy-" + name,
        source=source,
        convert=partial(boxed_math_task, answer_field=answer_field, benchmark=benchmark),
        version=scorer_revision("boxed_math"),
        intended_use=IntendedUse.EVAL,
        grader=MATH_GRADER,
    )
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="math-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "math"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            recipe,
            EvaluationProtocol(
                benchmark=benchmark,
                version="2026.11.13",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=temperature,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0 if temperature else None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            partial(chat_item, response_selection="final_or_complete_boxed_reasoning"),
        ),
        pipeline=prepare_evaluation,
    )


def olympiad_source(*, judged: bool) -> RlDataSource[EvaluationCorpusRecipe]:
    benchmark = "OlympiadBench" if judged else "OlympiadBenchDeterministic"
    count = 10 if judged else 1
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + benchmark.lower(),
            title=benchmark,
            origin="Evalchemy",
            family="math-answer",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "math", "legacy-subset"),
            count=30,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="evalchemy-" + benchmark.lower(),
                source=bundled_source("OlympiadBench", "olympiadbench.jsonl", OLYMPIAD_SHA256),
                convert=partial(convert_olympiad, judged=judged),
                version=scorer_revision("olympiad"),
                intended_use=IntendedUse.EVAL,
                grader=MATH_JUDGE_GRADER if judged else MATH_GRADER,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                requires_judge=judged,
                version="2026.11.13",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=0.7 if judged else 0,
                    top_p=1,
                    top_k=None,
                    max_tokens=32768,
                    stop=CHAT_REQUEST_STOPS,
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=count, seed=0 if judged else None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=30,
            ),
            chat_item,
        ),
        pipeline=prepare_evaluation,
    )


def base_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        math_dataset(
            name="math500",
            benchmark="MATH500",
            source=HfSource("HuggingFaceH4/MATH-500", MATH500_REVISION, ("test.jsonl",), SourceFormat.JSONL),
            answer_field="answer",
            population=500,
            trials=1,
            temperature=0,
        ),
        math_dataset(
            name="aime24",
            benchmark="AIME24",
            source=UrlSource(
                f"https://raw.githubusercontent.com/marin-community/marin/{MARIN_SOURCE_REVISION}/"
                "vendor/evalchemy/eval/chat_benchmarks/AIME24/data/aime24.json",
                AIME24_SHA256,
                "aime24.jsonl",
                SourceFormat.JSONL,
            ),
            answer_field="expected_answer",
            population=30,
            trials=10,
            temperature=0.7,
        ),
    ]


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    aime = math_dataset(
        name="aime25",
        benchmark="AIME25",
        source=bundled_source("AIME25", "aime25.json", AIME25_SHA256),
        answer_field="answer",
        population=30,
        trials=10,
        temperature=0,
    )
    amc = math_dataset(
        name="amc23",
        benchmark="AMC23",
        source=bundled_source("AMC23", "amc23.json", AMC23_SHA256),
        answer_field="answer",
        population=40,
        trials=10,
        temperature=0,
    )
    assert aime.config is not None and amc.config is not None
    assert aime.config.protocol.sampling is not None and amc.config.protocol.sampling is not None
    aime = replace(
        aime,
        config=replace(
            aime.config,
            protocol=aime.config.protocol.model_copy(
                update={
                    "sampling": aime.config.protocol.sampling.model_copy(update={"max_tokens": 32768, "stop": ()}),
                }
            ),
        ),
    )
    amc = replace(
        amc,
        config=replace(
            amc.config,
            item_builder=chat_item,
            recipe=replace(amc.config.recipe, convert=convert_amc, version=scorer_revision("amc")),
            protocol=amc.config.protocol.model_copy(
                update={
                    "sampling": amc.config.protocol.sampling.model_copy(update={"max_tokens": 32768, "stop": ()}),
                }
            ),
        ),
    )
    return [*base_sources(), aime, amc, olympiad_source(judged=False), olympiad_source(judged=True)]


def benchmarks() -> list[EvaluationBenchmark]:
    populations = sources()
    return [
        *fixed_benchmarks(
            [
                source
                for source in populations
                if source.config is not None and not source.config.protocol.benchmark.startswith("OlympiadBench")
            ]
        ),
        *fixed_benchmarks([olympiad_source(judged=True)], "evalchemy", {"OlympiadBench": "olympiadbench"}),
        *fixed_benchmarks(
            [olympiad_source(judged=False)], "deterministic", {"OlympiadBenchDeterministic": "olympiadbench"}
        ),
        *fixed_benchmarks(fewshot_sources(), "cot", {"minerva_math": "hendrycks-math"}),
    ]
