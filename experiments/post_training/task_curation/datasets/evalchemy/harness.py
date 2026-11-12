# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned native likelihood corpora with lm-eval v0.4.12 prompt semantics."""

from functools import partial

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.scorers.harness_generation import COMPLETION_REQUEST_STOPS, GSM8K_REQUEST_STOPS
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    continuation_item,
    convert_continuation_document,
    convert_generation_document,
    generation_item,
    read_continuation_file,
    read_generation_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import (
    ALIAS_GRADER,
    BASIC_GRADER,
    MATH_GRADER,
)
from experiments.post_training.task_curation.datasets.evalchemy.likelihood import choice_source
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin


def sources() -> list[RlDataSource[EvalDataset]]:
    raw = (LikelihoodNormalization.RAW,)
    normalized = (LikelihoodNormalization.RAW, LikelihoodNormalization.CHARACTERS)
    result = []
    for name, repo, revision, file, population, shots, pool, normalizations in (
        (
            "arc_challenge",
            "allenai/ai2_arc",
            "210d026faf9955653af8916fad021475a3f00453",
            "ARC-Challenge/test-00000-of-00001.parquet",
            1172,
            25,
            "ARC-Challenge/train-00000-of-00001.parquet",
            normalized,
        ),
        (
            "arc_easy",
            "allenai/ai2_arc",
            "210d026faf9955653af8916fad021475a3f00453",
            "ARC-Easy/test-00000-of-00001.parquet",
            2376,
            0,
            "",
            normalized,
        ),
        (
            "hellaswag",
            "Rowan/hellaswag",
            "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "data/validation-00000-of-00001.parquet",
            10042,
            10,
            "data/train-00000-of-00001.parquet",
            normalized,
        ),
        (
            "boolq",
            "aps/super_glue",
            "3de24cf8022e94f4ee4b9d55a6f539891524d646",
            "boolq/validation-00000-of-00001.parquet",
            3270,
            0,
            "",
            raw,
        ),
        (
            "piqa",
            "baber/piqa",
            "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
            "piqa_validation.parquet",
            1838,
            0,
            "",
            normalized,
        ),
        (
            "openbookqa",
            "allenai/openbookqa",
            "388097ea7776314e93a529163e0fea805b8a6454",
            "main/test-00000-of-00001.parquet",
            500,
            0,
            "",
            normalized,
        ),
        (
            "winogrande",
            "allenai/winogrande",
            "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
            "winogrande_xl/validation-00000-of-00001.parquet",
            1267,
            5,
            "winogrande_xl/train-00000-of-00001.parquet",
            raw,
        ),
        (
            "truthfulqa_mc2",
            "truthfulqa/truthful_qa",
            "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
            "multiple_choice/validation-00000-of-00001.parquet",
            817,
            0,
            "",
            raw,
        ),
    ):
        result.append(
            choice_source(
                name, repo, revision, file, population, HarnessTemplate(name, shots, fewshot_file=pool), normalizations
            )
        )
    result.append(
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:lambada_openai",
                upstream_dataset="lambada_openai",
                upstream_name="lambada_openai",
                subset=None,
                build_path="Evalchemy",
                origin=SourceOrigin.EVALCHEMY,
                family="likelihood-continuation",
                tags=("eval", "source:evalchemy", "benchmark"),
                count=5153,
            ),
            config=EvalDataset(
                recipe=CurationRecipe(
                    name="evalchemy-lambada_openai",
                    source=HfSource(
                        "EleutherAI/lambada_openai",
                        "900124bf3b8235c6daf21033af9948b3f07346c4",
                        ("default/test/default.parquet",),
                        SourceFormat.PARQUET,
                        read=read_continuation_file,
                    ),
                    convert=convert_continuation_document,
                    version="2026.11.02",
                    intended_use=IntendedUse.EVAL,
                ),
                protocol=EvaluationProtocol(
                    benchmark="lambada_openai",
                    version="2026.11.02",
                    operation=OperationKind.CONTINUATION_LIKELIHOOD,
                    sampling=None,
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=(
                        Metric(name="acc,none", kind=MetricKind.BINARY, higher_is_better=True),
                        Metric(name="perplexity,none", kind=MetricKind.CONTINUOUS, higher_is_better=False),
                    ),
                    primary_metric="acc,none",
                    summary=SummaryKind.PERPLEXITY,
                    expected_population=5153,
                    likelihood_metrics=(
                        LikelihoodMetric(
                            name="acc,none", normalization=LikelihoodNormalization.RAW, scoring="greedy_accuracy"
                        ),
                        LikelihoodMetric(
                            name="perplexity,none", normalization=LikelihoodNormalization.RAW, scoring="log_likelihood"
                        ),
                    ),
                ),
                item_builder=continuation_item,
            ),
            pipeline=curate,
        )
    )
    result.extend(generation_sources())
    return result


def generation_sources() -> list[RlDataSource[EvalDataset]]:
    result = []
    for name, profile, repo, revision, file, pool, shots, population in (
        (
            "gsm8k",
            "gsm8k",
            "openai/gsm8k",
            "740312add88f781978c0658806c59bc2815b9866",
            "main/test-00000-of-00001.parquet",
            "main/train-00000-of-00001.parquet",
            5,
            1319,
        ),
        (
            "gsm8k-0shot",
            "gsm8k",
            "openai/gsm8k",
            "740312add88f781978c0658806c59bc2815b9866",
            "main/test-00000-of-00001.parquet",
            "",
            0,
            1319,
        ),
        (
            "triviaqa",
            "triviaqa",
            "mandarjoshi/trivia_qa",
            "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
            "rc.nocontext/validation-00000-of-00001.parquet",
            "rc.nocontext/train-00000-of-00001.parquet",
            5,
            17944,
        ),
        (
            "nq_open",
            "nq_open",
            "google-research-datasets/nq_open",
            "5dd9790a83002ad084ddeb7c420dc716852c6f28",
            "nq_open/validation-00000-of-00001.parquet",
            "nq_open/train-00000-of-00001.parquet",
            5,
            3610,
        ),
        (
            "drop",
            "drop",
            "EleutherAI/drop",
            "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3",
            "drop_validation.parquet",
            "drop_train.parquet",
            3,
            9536,
        ),
    ):
        template = HarnessTemplate(profile, shots, fewshot_file=pool)
        if profile == "gsm8k":
            metrics = ("exact_match,strict-match", "exact_match,flexible-extract")
        elif profile == "drop":
            metrics = ("em,extract_answer", "f1,extract_answer")
        else:
            metrics = ("exact_match,strict_answer", "exact_match,extract_answer")
        result.append(
            RlDataSource(
                info=SourceInfo(
                    id="Evalchemy:" + name,
                    upstream_dataset=name,
                    upstream_name=name,
                    subset=None,
                    build_path="Evalchemy",
                    origin=SourceOrigin.EVALCHEMY,
                    family="short-answer",
                    tags=("eval", "source:evalchemy", "benchmark", "family:short-answer", "task:" + profile),
                    count=population,
                ),
                config=EvalDataset(
                    recipe=CurationRecipe(
                        name="evalchemy-" + name,
                        source=HfSource(
                            repo,
                            revision,
                            (file,),
                            SourceFormat.PARQUET,
                            read=partial(read_generation_file, template=template),
                        ),
                        inputs={"fewshot": HfSource(repo, revision, (pool,), SourceFormat.PARQUET)} if shots else {},
                        convert=partial(convert_generation_document, benchmark=name),
                        grader={
                            "gsm8k": MATH_GRADER,
                            "triviaqa": ALIAS_GRADER,
                            "nq_open": ALIAS_GRADER,
                            "drop": BASIC_GRADER,
                        }[profile],
                        version=scorer_revision("harness_generation"),
                        intended_use=IntendedUse.EVAL,
                    ),
                    protocol=EvaluationProtocol(
                        benchmark=name,
                        version="2026.11.02",
                        operation=OperationKind.TEXT_GENERATION,
                        sampling=Sampling(
                            temperature=0,
                            top_p=None,
                            top_k=None,
                            max_tokens=None,
                            stop=GSM8K_REQUEST_STOPS if profile == "gsm8k" else COMPLETION_REQUEST_STOPS,
                            chat_template_kwargs={},
                        ),
                        trials=TrialPolicy(count=1, seed=None),
                        metrics=tuple(
                            Metric(
                                name=metric,
                                kind=MetricKind.CONTINUOUS if metric.startswith("f1,") else MetricKind.BINARY,
                                higher_is_better=True,
                            )
                            for metric in metrics
                        ),
                        primary_metric="f1,extract_answer" if profile == "drop" else metrics[0],
                        summary=SummaryKind.MEAN,
                        expected_population=population,
                    ),
                    item_builder=generation_item,
                ),
                pipeline=curate,
            )
        )
    return result
