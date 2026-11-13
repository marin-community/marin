# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned reasoning benchmark populations and evaluation protocols."""

from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.reasoning import (
    MMLU_VALIDATION_FILE,
    agieval_template,
    bbh_cot_template,
    bbh_likelihood_template,
    convert_gpqa,
    convert_mmlu,
    gpqa_likelihood_template,
    mmlu_pro_likelihood_template,
    reasoning_item,
)
from taskcompendium.importers.fewshot import pinned_prompts
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Converter, IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.evaluation.prompt_compilation import (
    multiple_choice_evaluation,
    text_generation_evaluation,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_TEST_FILE = "data/test-00000-of-00001.parquet"
BBH_REVISION = "b5306be6f827cfafbb545ff5a51f96916029b0fd"


def completion_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    raw = LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW)
    normalized = LikelihoodMetric(name="acc_norm,none", normalization=LikelihoodNormalization.CHARACTERS)
    result = [
        multiple_choice_evaluation(
            name="agieval_lsat_ar",
            benchmark="agieval_lsat_ar",
            source=HfSource(
                "hails/agieval-lsat-ar",
                "052cc636b612f5563329dd182fb6c2cad56681c8",
                ("data/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            population=230,
            template=agieval_template(),
            metrics=(raw, normalized),
        ),
        multiple_choice_evaluation(
            name="leaderboard_mmlu_pro",
            benchmark="leaderboard_mmlu_pro",
            source=HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            population=12032,
            template=mmlu_pro_likelihood_template(MMLU_VALIDATION_FILE),
            metrics=(raw,),
        ),
    ]
    for subject, population in (("diamond", 198), ("main", 448), ("extended", 546)):
        result.append(
            multiple_choice_evaluation(
                name="leaderboard_gpqa-" + subject,
                benchmark="leaderboard_gpqa",
                source=HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_" + subject + ".csv",), SourceFormat.CSV),
                population=population,
                template=gpqa_likelihood_template(),
                metrics=(normalized,),
                subject=subject,
            )
        )
    for key in pinned_prompts():
        if not key.startswith(("bbh/", "bbh_cot/")):
            continue
        variant, subject = key.split("/", 1)
        population = {"causal_judgement": 187, "penguins_in_a_table": 146, "snarks": 178}.get(subject, 250)
        source = HfSource(
            "SaylorTwift/bbh",
            BBH_REVISION,
            (subject + "/test-00000-of-00001.parquet",),
            SourceFormat.PARQUET,
        )
        if variant == "bbh":
            result.append(
                multiple_choice_evaluation(
                    name="leaderboard_bbh-" + subject,
                    benchmark="leaderboard_bbh",
                    source=source,
                    population=population,
                    template=bbh_likelihood_template(subject),
                    metrics=(normalized,),
                    subject=subject,
                )
            )
            continue
        result.append(
            text_generation_evaluation(
                name="bbh_cot_fewshot-" + subject,
                benchmark="bbh_cot_fewshot",
                source=source,
                population=population,
                template=bbh_cot_template(subject),
                metrics=("exact_match,get-answer",),
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=1024,
                    stop=("</s>", "Q", "\n\n"),
                    chat_template_kwargs={},
                ),
                scorer="bbh_cot",
                grader=BASIC_GRADER,
            )
        )
    return result


def choice_dataset(
    *,
    name: str,
    benchmark: str,
    source: HfSource,
    converter: Converter,
    population: int,
    trials: int,
    inputs: dict[str, HfSource],
    temperature: float | None = None,
) -> RlDataSource[EvaluationCorpusRecipe]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="generated-choice",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "reasoning", "benchmark:" + benchmark),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=converter,
                version=scorer_revision("mmlu_pro" if benchmark == "MMLUPro" else "gpqa"),
                intended_use=IntendedUse.EVAL,
                grader=BASIC_GRADER,
                inputs=inputs,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="2026.11.13",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=temperature,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=population,
                context_window=65536 if benchmark == "MMLUPro" else None,
            ),
            partial(reasoning_item, benchmark=benchmark),
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        choice_dataset(
            name="gpqa-diamond",
            benchmark="GPQADiamond",
            source=HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_diamond.csv",), SourceFormat.CSV),
            converter=convert_gpqa,
            population=198,
            trials=3,
            inputs={},
            temperature=0.7,
        ),
        choice_dataset(
            name="mmlu-pro",
            benchmark="MMLUPro",
            source=HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            converter=convert_mmlu,
            population=12032,
            trials=1,
            inputs={
                "fewshot": HfSource(
                    "TIGER-Lab/MMLU-Pro",
                    MMLU_PRO_REVISION,
                    (MMLU_VALIDATION_FILE,),
                    SourceFormat.PARQUET,
                )
            },
        ),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    completion = completion_sources()
    return [
        *fixed_benchmarks(sources(), names={"MMLUPro": "mmlu-pro", "GPQADiamond": "gpqa-diamond"}),
        *fixed_benchmarks(
            [
                source
                for source in completion
                if source.config is not None and source.config.protocol.benchmark != "bbh_cot_fewshot"
            ],
            "likelihood",
            {"leaderboard_mmlu_pro": "mmlu-pro", "leaderboard_gpqa": "gpqa", "leaderboard_bbh": "bbh"},
        ),
        *fixed_benchmarks(
            [
                source
                for source in completion
                if source.config is not None and source.config.protocol.benchmark == "bbh_cot_fewshot"
            ],
            "cot",
            {"bbh_cot_fewshot": "bbh"},
        ),
    ]
