# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GPQA-Diamond and MMLU-Pro generated-choice populations."""

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

from experiments.post_training.task_curation.datasets.evalchemy.environments import BASIC_GRADER
from experiments.post_training.task_curation.datasets.evalchemy.protocols import (
    fewshot_choice_source,
    fewshot_generation_source,
)
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_TEST_FILE = "data/test-00000-of-00001.parquet"
BBH_REVISION = "b5306be6f827cfafbb545ff5a51f96916029b0fd"


def likelihood_sources() -> list[RlDataSource[EvalDataset]]:
    raw = LikelihoodMetric(name="acc,none", normalization=LikelihoodNormalization.RAW)
    normalized = LikelihoodMetric(name="acc_norm,none", normalization=LikelihoodNormalization.CHARACTERS)
    result = [
        fewshot_choice_source(
            "agieval_lsat_ar",
            "agieval_lsat_ar",
            HfSource(
                "hails/agieval-lsat-ar",
                "052cc636b612f5563329dd182fb6c2cad56681c8",
                ("data/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            230,
            agieval_template(),
            (raw, normalized),
        ),
        fewshot_choice_source(
            "leaderboard_mmlu_pro",
            "leaderboard_mmlu_pro",
            HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            12032,
            mmlu_pro_likelihood_template(MMLU_VALIDATION_FILE),
            (raw,),
        ),
    ]
    for subject, population in (("diamond", 198), ("main", 448), ("extended", 546)):
        result.append(
            fewshot_choice_source(
                "leaderboard_gpqa-" + subject,
                "leaderboard_gpqa",
                HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_" + subject + ".csv",), SourceFormat.CSV),
                population,
                gpqa_likelihood_template(),
                (normalized,),
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
                fewshot_choice_source(
                    "leaderboard_bbh-" + subject,
                    "leaderboard_bbh",
                    source,
                    population,
                    bbh_likelihood_template(subject),
                    (normalized,),
                    subject=subject,
                )
            )
            continue
        result.append(
            fewshot_generation_source(
                "bbh_cot_fewshot-" + subject,
                "bbh_cot_fewshot",
                source,
                population,
                bbh_cot_template(subject),
                ("exact_match,get-answer",),
                Sampling(
                    temperature=0,
                    top_p=None,
                    top_k=None,
                    max_tokens=1024,
                    stop=("</s>", "Q", "\n\n"),
                    chat_template_kwargs={},
                ),
                "bbh_cot",
                BASIC_GRADER,
            )
        )
    return result


def choice_dataset(
    name: str,
    benchmark: str,
    source: HfSource,
    converter: Converter,
    rows: int,
    trials: int,
    temperature: float,
    inputs: dict[str, HfSource],
) -> RlDataSource[EvalDataset]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            title=benchmark,
            origin="Evalchemy",
            family="generated-choice",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "reasoning", "benchmark:" + benchmark),
            count=rows,
        ),
        config=EvalDataset(
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
                version="2026.11.02",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=temperature,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0 if temperature else None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=rows,
                context_window=65536 if benchmark == "MMLUPro" else None,
            ),
            partial(reasoning_item, benchmark=benchmark),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        choice_dataset(
            "gpqa-diamond",
            "GPQADiamond",
            HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_diamond.csv",), SourceFormat.CSV),
            convert_gpqa,
            198,
            3,
            0.7,
            {},
        ),
        choice_dataset(
            "mmlu-pro",
            "MMLUPro",
            HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            convert_mmlu,
            12032,
            1,
            0,
            {
                "fewshot": HfSource(
                    "TIGER-Lab/MMLU-Pro",
                    MMLU_PRO_REVISION,
                    (MMLU_VALIDATION_FILE,),
                    SourceFormat.PARQUET,
                )
            },
        ),
    ]
